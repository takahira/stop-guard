#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stop-guard — Claude Code Stop hook: catches leaked tool calls and empty end_turns, and requests a retry.

Background
----------
Opus 4.8 intermittently serialises a tool call into the *text* channel instead
of as a structured ``tool_use`` block: a stray boundary token (``count`` /
``call`` / ``court``) followed by a bare ``<invoke name="...">...</invoke>`` that
is missing the ``antml:`` namespace and ``function_calls`` wrapper. The harness
cannot parse it, so nothing runs.

The harness *already* recovers from this when the turn's ``stop_reason`` is
``tool_use`` (it injects "Your tool call was malformed... Please retry."). But
when the mis-serialised call coincides with ``end_turn`` the harness sees a
clean text-only completion and injects no retry -- so autonomous ScheduleWakeup
loops die silently (upstream issue #2763 / #67945; see docs/issue-triage.md).

This Stop hook closes that gap on the client side: when Claude is about to stop
and its last assistant text contains a leaked ``<invoke>``, we BLOCK the stop and
feed back a retry instruction -- giving the ``end_turn`` case the same recovery
the ``tool_use`` case already gets.

Contract
--------
Register as a ``Stop`` hook. Claude Code feeds JSON on stdin::

    {"hook_event_name":"Stop","stop_hook_active":false,
     "transcript_path":"/abs/path/session.jsonl", "session_id":"...", ...}

To block the stop and ask the model to retry we print::

    {"decision":"block","reason":"<retry instruction>"}

and exit 0. To allow the stop we exit 0 with no output. The hook is
**fail-open**: any error or ambiguous state allows the stop (we never strand the
agent because of our own bug).

Config (env vars)
-----------------
- ``STOP_GUARD_DISABLE=1``     -- no-op (always allow).
- ``STOP_GUARD_OBSERVE=1``     -- observe/log-only mode: detect and log, but
                                    NEVER block. Use this to dogfood the hook for a few
                                    days and confirm zero false positives before
                                    switching to blocking.
- ``STOP_GUARD_NO_EMPTY_TURN=1`` -- disable the empty/zero-content end_turn
                                    guard (Guard 2). It is ON by default: a
                                    turn that ends on end_turn with no actionable
                                    content (no non-whitespace text, no tool_use,
                                    no thinking block) is blocked with a "finish
                                    or explain why you stopped" instruction. The
                                    leak detector is independent of this flag.
- ``STOP_GUARD_TOKENS=a,b,c``  -- override stray-token list
                                    (default: count,call,court,course,invoke).
- ``STOP_GUARD_LOG=PATH``      -- detection log file (default
                                    ~/.claude/stop-guard.log). Detections are
                                    always appended here (best-effort).
- ``STOP_GUARD_NOLOG=1``       -- disable the detection log.
- ``STOP_GUARD_LOG_PATHS=1``   -- record the RAW session id and transcript path
                                    in the log (default: a stable correlation id,
                                    so no username or project path is stored).
- ``STOP_GUARD_DEBUG=1``       -- log decisions to stderr.

CLI (no stdin) for testing / corpus mining
------------------------------------------
- ``stop_guard.py --self-test``       -- run built-in detection samples.
- ``stop_guard.py --scan FILE.jsonl`` -- report leaked invoke in a transcript.
- ``stop_guard.py --check < text``    -- detect on raw text from stdin.
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import os
import re
import sys
import time
from typing import TypedDict

# Single source of truth for the version (pyproject reads this via hatchling).
__version__ = "0.1.1"


class LeakDetail(TypedDict):
    """Result shape of :func:`leak_details`."""
    leak: bool
    signature: str | None
    token: str | None

# Stray boundary tokens observed leaking before a bare <invoke> in real
# corpora: count / call / court / course (all seen in local transcripts), plus
# the literal "invoke" word. The set is open -- override via STOP_GUARD_TOKENS.
DEFAULT_TOKENS = ("count", "call", "court", "course", "invoke")

# A genuine leak is the *last* thing the model emitted before stopping, so the
# signature always sits at the tail of the turn. We only scan the trailing slice;
# this also bounds the cost of the regexes below on pathological multi-MB turns.
# We strip code fences on the full turn *before* applying this cap (see
# leak_details), so a fence opener far above the cap is still honored and an
# <invoke> inside it is not mistaken for an unfenced leak.
MAX_SCAN_CHARS = 64 * 1024

# Strip fenced / inline code first so prose that *discusses* the bug (this very
# file, a bug report, a code sample) is never mistaken for a real leak.
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# A code fence the model opened but never closed (partial output, or prose that
# trails into an example) runs to the end of the turn; strip it too so an
# unfenced-looking <invoke> inside it is not mistaken for a leak.
_DANGLING_FENCE_RE = re.compile(r"```.*$", re.DOTALL)
_INLINE_RE = re.compile(r"`[^`\n]*`")
# CommonMark indented code blocks (>=4 leading spaces or a tab) are a third code
# form the fence/inline strippers miss. Blank them too so an *indented* example of
# the corruption at the tail of a real answer is not mistaken for a leak. A genuine
# leak is emitted flush-left (never indented), so this only narrows false positives.
_INDENT_CODE_RE = re.compile(r"(?m)^(?:[ ]{4}|\t).*$")
# Case-insensitive closing-tag matcher. _stray_re matches the opener case-
# insensitively, so terminality/completeness checks must recognise a capitalised
# closing tag too (otherwise a capitalised close defeats the terminal guard).
_CLOSE_RE = re.compile(r"</invoke>", re.IGNORECASE)

# Signature A: a stray boundary token alone on a line, immediately followed by a
# bare <invoke ...> opening tag. This is the canonical corruption fingerprint.
# The token is captured (group 1) so callers can build a per-token histogram.
# Cached: the compiled pattern only depends on the (hashable) token tuple, so we
# never recompile per call (scan_corpus hits this once per assistant turn).
@functools.lru_cache(maxsize=None)
def _stray_re(tokens: tuple[str, ...]) -> "re.Pattern[str]":
    alt = "|".join(re.escape(t) for t in tokens)
    # IGNORECASE: stray tokens may appear capitalized (e.g. "Count", "Court")
    # in future model variants or localized outputs.
    return re.compile(r"(?mi)^[ \t]*(" + alt + r")[ \t]*\r?\n[ \t]*<invoke\b")

# Signature B: a structurally complete *bare* invoke element in the text channel.
# A real tool call never reaches the text channel (the harness extracts it), so a
# complete <invoke name=...>...</invoke> sitting in assistant text is a leak.
# `name=[^>]*>` anchors the opening tag so the lazy body cannot rescan from every
# `<invoke name=` occurrence to end-of-string (was O(N*M) on many unclosed tags).
_BARE_CALL_RE = re.compile(r"<invoke\s+name=[^>]*>.*?</invoke>", re.DOTALL | re.IGNORECASE)

RETRY_REASON = (
    "A tool call from your previous turn was mis-serialised into plain text "
    "(a bare <invoke> block leaked into the message body instead of executing). "
    "Re-issue that tool call now as a proper structured tool call. "
    "Do NOT re-emit the same bytes as text -- re-derive the correct invocation "
    "format from scratch. If you did not intend a tool call, reply with a short "
    "confirmation instead so this guard releases."
)

# Guard 2: emitted when the last assistant turn ended on end_turn but produced
# no actionable content (no non-whitespace text, no tool_use, no thinking block).
EMPTY_TURN_REASON = (
    "Your last turn produced no output (empty end_turn). "
    "Complete the task, or state explicitly why you are stopping."
)


def _strip_code(text: str) -> str:
    text = _FENCE_RE.sub(" ", text)            # closed ```...``` blocks
    text = _DANGLING_FENCE_RE.sub(" ", text)   # an unclosed ``` running to the end
    text = _INDENT_CODE_RE.sub(" ", text)      # CommonMark indented code blocks
    return _INLINE_RE.sub(" ", text)


def _tail_is_blank(stripped: str, pos: int) -> bool:
    """True if everything after ``pos`` is whitespace (the construct ends here)."""
    return stripped[pos:].strip() == ""


def _stray_is_terminal(stripped: str, m: "re.Match") -> bool:
    """A signature-A hit is terminal iff its ``<invoke>`` ends the turn.

    Closed element: terminal when only whitespace follows ``</invoke>``.
    Unclosed opener (no ``</invoke>``): a *truncated* leak, terminal only when
    the opener tag itself is the tail -- i.e. nothing but whitespace follows its
    ``>`` (or the tag is cut mid-attribute, so there is no ``>`` yet). Prose that
    *teaches* an unclosed ``<invoke ...>`` keeps explaining afterwards, so it is
    not terminal and must not trip.
    """
    m_close = _CLOSE_RE.search(stripped, m.end())
    if m_close is not None:
        return _tail_is_blank(stripped, m_close.end())
    gt = stripped.find(">", m.end())
    if gt == -1:
        return True  # opener truncated mid-attribute = the tail
    return _tail_is_blank(stripped, gt + 1)


def _terminal_bare_match(stripped: str):
    """Return the LAST complete bare ``<invoke>…</invoke>`` iff it ends the turn.

    Anchoring to the matched element (not the whole text) is essential: prose can
    show a *complete* example and then trail off into an unclosed ``<invoke>``
    mention; only when the complete element is itself the final content is it a
    real leak.
    """
    # Fast-path: if there is no closing tag at all, there is no complete element.
    # MAX_SCAN_CHARS bounds worst-case cost, but this avoids the O(N^2) regex
    # entirely for the common (clean) case.
    if _CLOSE_RE.search(stripped) is None:
        return None
    tail = collections.deque(_BARE_CALL_RE.finditer(stripped), maxlen=1)
    last = tail[0] if tail else None
    if last is not None and _tail_is_blank(stripped, last.end()):
        return last
    return None


def leak_details(text: str, tokens: tuple[str, ...] = DEFAULT_TOKENS) -> LeakDetail:
    """Return {leak, signature, token} for a piece of assistant text.

    ``token`` is the captured stray boundary token for signature A, else None.
    A genuine leak is *terminal* (the model emitted the mis-serialised call and
    stopped), so each signature is accepted only when its own ``<invoke>``
    construct is the last meaningful content -- prose that teaches or quotes an
    ``<invoke>`` and keeps explaining afterwards never trips, even unfenced.
    """
    if not text:
        return {"leak": False, "signature": None, "token": None}
    # Strip code FIRST, then cap to the trailing slice. Stripping before the cap is
    # essential: a code fence whose opener sits >MAX_SCAN_CHARS before the end would
    # otherwise be sliced away, leaving an <invoke> that was *inside* a fence looking
    # unfenced (a false positive). The strip regexes are linear, so running them on
    # the full turn is bounded by turn size; only the trailing slice can carry a
    # terminal leak anyway.
    stripped = _strip_code(text)
    if len(stripped) > MAX_SCAN_CHARS:
        stripped = stripped[-MAX_SCAN_CHARS:]
    m_iter = collections.deque(_stray_re(tuple(tokens)).finditer(stripped), maxlen=1)
    m = m_iter[0] if m_iter else None
    if m and _stray_is_terminal(stripped, m):
        return {"leak": True, "signature": "stray-token+invoke",
                "token": m.group(1).lower()}
    if _terminal_bare_match(stripped) is not None:
        return {"leak": True, "signature": "bare-invoke-element", "token": None}
    return {"leak": False, "signature": None, "token": None}


def detect_leak(text: str,
                tokens: tuple[str, ...] = DEFAULT_TOKENS) -> tuple[bool, str | None]:
    """Return (is_leak, signature_label) for a piece of assistant text."""
    d = leak_details(text, tokens)
    return d["leak"], d["signature"]


def _content_to_text(content: "str | list | None") -> str:
    """Flatten an assistant message's content into its text-channel string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


# Public alias: scan_corpus (and other callers) flatten content without reaching
# into a private name.
content_to_text = _content_to_text


def last_assistant_turn(transcript_path: str) -> "tuple[str, str | list | None]":
    """Return ``(text, content)`` of the last assistant turn in ONE transcript pass.

    ``text`` is the flattened text channel (leak guard); ``content`` is the raw
    ``content`` value -- blocks, not text -- which the empty-turn guard needs to
    see structured blocks (tool_use / thinking). Both reflect the FINAL assistant
    turn, so a stale leak from an earlier turn never causes a false-positive
    block when the final turn is actually clean.

    Reads fail-open: any error yields ``("", None)`` -- the same values as an
    absent assistant turn -- so the guards allow the stop (never block).
    """
    content: "str | list | None" = None
    current_id: "str | None" = None
    try:
        with open(transcript_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    # A row we cannot parse makes the tail of the transcript
                    # AMBIGUOUS. Silently skipping it left `content` holding an
                    # EARLIER turn, so a truncated row after a clean completion
                    # made the guard block on a leak that belonged to a turn the
                    # user had already moved past. Ambiguity must allow the stop,
                    # so drop what we have; a later well-formed assistant row
                    # supersedes this and restores the verdict.
                    content = None
                    current_id = None
                    continue
                # Defensive: json.loads accepts any JSON value, so a line holding
                # `null`, a number, or a list parses fine but has no .get(). The
                # AttributeError used to escape to run_hook's fail-open, silently
                # disabling the guard for EVERY Stop in that session. Skip the
                # line instead (cf. scan_corpus's per-event isolation).
                if not isinstance(evt, dict) or evt.get("type") != "assistant":
                    continue
                msg = evt.get("message", evt)
                if not isinstance(msg, dict):
                    continue
                row_content = msg.get("content")
                row_id = msg.get("id")
                # Claude Code can split ONE assistant message across several JSONL
                # rows -- one per content block -- that all carry the same
                # `message.id`. Judging only the final row makes a `[thinking]` row
                # followed by an empty `[text]` row look like an empty turn, so the
                # guard blocks a normal completion and contradicts its own contract
                # ("a thinking block means the turn is not empty"). Merge consecutive
                # rows that share an id; rows without an id keep the previous
                # one-row-per-turn behaviour.
                #
                # Non-assistant rows deliberately do NOT reset `current_id`: one
                # assistant message that issues PARALLEL tool calls is written as
                # several rows with the tool_result `user` rows interleaved between
                # them, so those rows are the same logical message and must merge.
                # Verified across 5,771 real transcripts / 133,922 assistant rows:
                # every id that reappeared after a non-assistant row (8,956 cases)
                # was this parallel-tool-call pattern, never two distinct turns.
                if (
                    row_id is not None
                    and row_id == current_id
                    and isinstance(content, list)
                    and isinstance(row_content, list)
                ):
                    content = content + row_content
                else:
                    content = row_content
                    current_id = row_id
    except (OSError, UnicodeDecodeError) as exc:
        # Read failure is fail-open (allow the stop), but make it observable --
        # otherwise it is indistinguishable from a genuinely clean turn.
        _debug(f"could not read transcript {transcript_path!r}: {exc}; failing open")
        return "", None
    return _content_to_text(content), content


def last_assistant_text(transcript_path: str) -> str:
    """Extract the text channel of the last assistant turn in a JSONL transcript."""
    return last_assistant_turn(transcript_path)[0]


def last_assistant_content(transcript_path: str) -> "str | list | None":
    """Return the raw ``content`` of the last assistant turn (blocks, not text)."""
    return last_assistant_turn(transcript_path)[1]


def is_empty_turn(content: "str | list | None") -> bool:
    """True if an assistant turn's content has NO actionable result.

    Strict (this is the false-positive-safety story): empty iff it contains none
    of a non-whitespace text block, a tool_use block, or a thinking /
    redacted_thinking block. Detection logic is consistent with the reliability
    report in scan_corpus.py.
    A real text-only final answer has non-whitespace text and is NOT empty.

    An UNRECOGNISED block type counts as actionable, i.e. NOT empty. This used to
    go the other way, described as covering "a hypothetical image block" -- but
    the set of block types is open and already contains real server-side kinds
    (``server_tool_use``, ``web_search_tool_result``). A turn whose only content
    was a web search was judged empty and blocked, which is exactly the spurious
    retry this guard exists to avoid. Ignoring an unknown block can only ever
    manufacture a false positive; treating it as content can only ever let a
    genuinely empty turn through, and that is the safe direction here.
    """
    if content is None:
        return False  # cannot judge -> not empty (fail-open: never block)
    if isinstance(content, str):
        return content.strip() == ""
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, str):
            if block.strip():
                return False
            continue
        if not isinstance(block, dict):
            return False  # unparseable block: assume content, never block
        btype = block.get("type")
        if btype == "text":
            if (block.get("text") or "").strip():
                return False
            continue  # a whitespace-only text block really is empty
        # Anything that is not a text block -- tool_use, thinking, or a type this
        # version has never heard of -- is actionable content.
        return False
    return True


def _tokens_from_env():
    raw = os.environ.get("STOP_GUARD_TOKENS", "")
    toks = tuple(t.strip() for t in raw.split(",") if t.strip())
    return toks or DEFAULT_TOKENS


def _debug(msg: str):
    if os.environ.get("STOP_GUARD_DEBUG"):
        print(f"[stop-guard] {msg}", file=sys.stderr)


def _log_path():
    return os.environ.get(
        "STOP_GUARD_LOG",
        os.path.join(os.path.expanduser("~"), ".claude", "stop-guard.log"),
    )


def _corr_id(value):
    """A stable correlation id for a value that must not be stored verbatim.

    The detection log used to record the absolute transcript path and the raw
    session id. Neither is needed to make -- or to audit -- a blocking decision,
    but a path like ``/Users/<name>/work/<client>/...`` puts a username and
    private project names into a file that travels in support bundles, backups
    and shared machines. A truncated digest keeps the only property the log
    actually uses (records from one session group together) and drops the rest.

    This is a correlation id, NOT a secret: the input space of paths is small
    enough to brute-force if someone already knows what to look for. Use
    STOP_GUARD_LOG_PATHS=1 when you genuinely need the raw values for diagnosis.
    """
    if not value:
        return None
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()[:12]


def _detection_record(record: dict) -> dict:
    """Apply the log's privacy policy to one record before it is written."""
    if os.environ.get("STOP_GUARD_LOG_PATHS"):
        return record
    out = dict(record)
    if "session_id" in out:
        out["session"] = _corr_id(out.pop("session_id"))
    if "transcript" in out:
        out["transcript_id"] = _corr_id(out.pop("transcript"))
    return out


def _log_detection(record: dict):
    """Append one JSON detection record to the log (best-effort, fail-open)."""
    if os.environ.get("STOP_GUARD_NOLOG"):
        return
    path = _log_path()
    record = _detection_record(record)
    try:
        parent = os.path.dirname(path)
        if parent:  # empty when STOP_GUARD_LOG is a bare filename (cwd-relative)
            os.makedirs(parent, exist_ok=True)
        # 0600 on creation: the log records when an agent leaked a tool-call tag,
        # which is not something other local users need to read. An existing
        # file's mode is left alone -- the operator may have chosen it.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        # A detection we failed to record is dangerous in observe mode: the
        # operator reads "0 detections" as "no false positives" and promotes to
        # blocking on a false premise. Warn on stderr unconditionally (not gated
        # behind DEBUG) so a broken log is never silent.
        print(f"[stop-guard] WARNING: detection NOT logged to {path!r}: {exc}",
              file=sys.stderr)


def run_hook(stdin_text: str) -> int:
    """Core hook logic. Returns process exit code; prints decision JSON if blocking."""
    try:
        if os.environ.get("STOP_GUARD_DISABLE"):
            _debug("disabled via env; allowing")
            return 0
        try:
            payload = json.loads(stdin_text) if stdin_text.strip() else {}
        except json.JSONDecodeError:
            _debug("unparseable stdin; failing open")
            return 0

        # Re-entrancy guard: if we already blocked once this stop cycle, don't loop.
        if payload.get("stop_hook_active"):
            _debug("stop_hook_active=true; allowing to avoid loop")
            return 0

        transcript = payload.get("transcript_path")
        if not transcript or not os.path.exists(transcript):
            _debug(f"no transcript ({transcript!r}); failing open")
            return 0

        observe = bool(os.environ.get("STOP_GUARD_OBSERVE"))

        # Both guards judge the same final assistant turn, so read the transcript
        # ONCE and share the result (issue #2: avoid two full scans per Stop).
        text, content = last_assistant_turn(transcript)

        # Guard 1: invoke-leak. A leaked <invoke> is terminal text, so the turn is
        # never "empty" by the strict definition below -- but evaluating the leak
        # first keeps the two branches coherent and gives the leak its specific
        # retry text.
        d = leak_details(text, _tokens_from_env())
        if d["leak"]:
            _log_detection({
                "ts": int(time.time()),
                "session_id": payload.get("session_id"),
                "transcript": transcript,
                "signature": d["signature"],
                "token": d["token"],
                "action": "observe" if observe else "block",
            })
            if observe:
                _debug(f"leak detected ({d['signature']}/{d['token']}); observe mode, allowing")
                return 0
            _debug(f"leak detected ({d['signature']}/{d['token']}); blocking and requesting retry")
            json.dump({"decision": "block", "reason": RETRY_REASON}, sys.stdout)
            sys.stdout.write("\n")
            return 0

        # Guard 2: empty / zero-content end_turn. Orthogonal to the leak path.
        # Default ON; disable with STOP_GUARD_NO_EMPTY_TURN=1. The Stop hook
        # only fires on an end_turn-equivalent stop, so a missing/null stop_reason
        # is treated the same as end_turn (matching the leak path, which never
        # gates on stop_reason).
        if not os.environ.get("STOP_GUARD_NO_EMPTY_TURN"):
            if is_empty_turn(content):
                _log_detection({
                    "ts": int(time.time()),
                    "session_id": payload.get("session_id"),
                    "transcript": transcript,
                    "signature": "empty-end_turn",
                    "token": None,
                    "action": "observe" if observe else "block",
                })
                if observe:
                    _debug("empty end_turn detected; observe mode, allowing")
                    return 0
                _debug("empty end_turn detected; blocking and asking model to finish")
                json.dump({"decision": "block", "reason": EMPTY_TURN_REASON}, sys.stdout)
                sys.stdout.write("\n")
                return 0

        _debug("no leak, not empty; allowing stop")
        return 0
    except Exception as exc:  # fail-open: any unexpected error must never strand the agent
        print(f"[stop-guard] unexpected error; failing open: {exc}", file=sys.stderr)
        return 0


# ---------------------------------------------------------------------------
# CLI helpers (testing / corpus mining; not used in the hook path)
# ---------------------------------------------------------------------------
# Note on language: one sample below keeps its original Japanese prose line. These
# fixtures are verbatim excerpts from the transcript corpus that motivated the
# detector (see docs/issue-triage.md), and the surrounding prose is deliberately
# unmodified so the sample still represents what the guard actually sees. Only the
# stray token and the `<invoke>` element carry the leak signature; the prose
# language is incidental to detection.
_SELFTEST_SAMPLES = [
    ("clean prose", "All done. I ran the build and tests; everything passes.", False),
    (
        "stray count+invoke (leak)",
        "Let me check the file.\n\ncount\n<invoke name=\"Bash\">\n"
        "<parameter name=\"command\">ls</parameter>\n</invoke>",
        True,
    ),
    (
        "court+invoke (leak)",
        "court\n<invoke name=\"mcp__workspace__bash\">\n"
        "<parameter name=\"command\">grep -c x f</parameter>\n</invoke>",
        True,
    ),
    (
        "course+invoke (leak, seen in corpus)",
        "Phase 1 から着手します。\n\ncourse\n<invoke name=\"Bash\">\n"
        "<parameter name=\"command\">git status</parameter>\n</invoke>",
        True,
    ),
    (
        "bare complete invoke (leak)",
        "Here goes.\n<invoke name=\"Read\">\n"
        "<parameter name=\"file_path\">/a</parameter>\n</invoke>",
        True,
    ),
    (
        "prose discussing the bug in a fence (must NOT trip)",
        "The bug looks like:\n```\ncount\n<invoke name=\"Bash\">...</invoke>\n```\n"
        "We handle it with a Stop hook.",
        False,
    ),
    (
        "inline-code mention (must NOT trip)",
        "The harness sees a stray `count` then a bare `<invoke name=\"Bash\">` tag.",
        False,
    ),
]


def _cmd_self_test() -> int:
    failures = 0
    for label, text, expected in _SELFTEST_SAMPLES:
        got, sig = detect_leak(text)
        ok = got == expected
        flag = "ok " if ok else "FAIL"
        print(f"[{flag}] {label}: leak={got} sig={sig}")
        if not ok:
            failures += 1
    print(f"\n{len(_SELFTEST_SAMPLES) - failures}/{len(_SELFTEST_SAMPLES)} passed")
    return 1 if failures else 0


def _cmd_scan(path: str) -> int:
    if not os.path.exists(path):
        print(f"[stop-guard] WARNING: file not found: {path!r}", file=sys.stderr)
        print(json.dumps({"path": path, "leak": False, "signature": None,
                          "error": "file not found"}, ensure_ascii=False))
        return 1
    text = last_assistant_text(path)
    is_leak, sig = detect_leak(text, _tokens_from_env())
    print(json.dumps({"path": path, "leak": is_leak, "signature": sig}, ensure_ascii=False))
    return 0


def _cmd_check() -> int:
    is_leak, sig = detect_leak(sys.stdin.read(), _tokens_from_env())
    print(json.dumps({"leak": is_leak, "signature": sig}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="stop-guard Stop hook / detector")
    parser.add_argument("--version", action="version", version=f"stop-guard {__version__}")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--self-test", action="store_true", help="run built-in detection samples")
    g.add_argument("--scan", metavar="JSONL", help="scan a transcript's last assistant turn")
    g.add_argument("--check", action="store_true", help="detect leak on raw text from stdin")
    args = parser.parse_args(argv)

    if args.self_test:
        return _cmd_self_test()
    if args.scan:
        return _cmd_scan(args.scan)
    if args.check:
        return _cmd_check()
    # Default: hook mode -- read Claude Code's JSON event from stdin.
    try:
        stdin_text = sys.stdin.read()
    except Exception:
        stdin_text = ""
    return run_hook(stdin_text)


if __name__ == "__main__":
    sys.exit(main())

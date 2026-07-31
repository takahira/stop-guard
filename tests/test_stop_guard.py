# -*- coding: utf-8 -*-
"""Tests for stop-guard. Run: python -m pytest  (or python -m unittest)."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stop_guard as ig  # noqa: E402
import scan_corpus  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _run_hook(payload, env=None):
    """Invoke the hook with a JSON payload; return (exit_code, stdout)."""
    old = dict(os.environ)
    if env:
        os.environ.update(env)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = ig.run_hook(json.dumps(payload))
        return code, buf.getvalue()
    finally:
        os.environ.clear()
        os.environ.update(old)


class DetectLeak(unittest.TestCase):
    def test_stray_count(self):
        ok, sig = ig.detect_leak('x\n\ncount\n<invoke name="Bash">\n</invoke>')
        self.assertTrue(ok)
        self.assertEqual(sig, "stray-token+invoke")

    def test_stray_court_and_call(self):
        for tok in ("court", "call", "course", "count", "invoke"):
            ok, sig = ig.detect_leak(f'{tok}\n<invoke name="Read">\n</invoke>')
            self.assertTrue(ok, tok)
            self.assertEqual(sig, "stray-token+invoke", tok)

    def test_bare_complete_invoke(self):
        ok, sig = ig.detect_leak('go\n<invoke name="Read">\n'
                                 '<parameter name="file_path">/a</parameter>\n</invoke>')
        self.assertTrue(ok)
        self.assertEqual(sig, "bare-invoke-element")

    def test_clean_prose(self):
        ok, _ = ig.detect_leak("All done; tests pass and nothing else to do.")
        self.assertFalse(ok)

    def test_fenced_discussion_not_flagged(self):
        ok, _ = ig.detect_leak('Bug:\n```\ncount\n<invoke name="Bash">x</invoke>\n```\ndone')
        self.assertFalse(ok)

    def test_inline_code_not_flagged(self):
        ok, _ = ig.detect_leak('stray `count` then `<invoke name="Bash">` shown to user')
        self.assertFalse(ok)

    def test_empty(self):
        self.assertEqual(ig.detect_leak(""), (False, None))

    def test_leak_details_captures_token(self):
        d = ig.leak_details('course\n<invoke name="Bash">\n</invoke>')
        self.assertTrue(d["leak"])
        self.assertEqual(d["signature"], "stray-token+invoke")
        self.assertEqual(d["token"], "course")

    def test_leak_details_bare_has_no_token(self):
        d = ig.leak_details('go\n<invoke name="Read">\n'
                            '<parameter name="file_path">/a</parameter>\n</invoke>')
        self.assertEqual(d["token"], None)

    def test_custom_tokens(self):
        # "boom" is not a default token -> not a stray match, and no complete
        # element -> overall no leak.
        ok, _ = ig.detect_leak('boom\n<invoke name="Bash">', tokens=("count",))
        self.assertFalse(ok)
        ok, _ = ig.detect_leak('boom\n<invoke name="Bash">', tokens=("boom",))
        self.assertTrue(ok)


class FalsePositiveGuards(unittest.TestCase):
    """A real leak is terminal (the model emitted the call and stopped). Prose
    that teaches or quotes an <invoke> keeps explaining, so it must not trip --
    even when it is NOT fenced (the fence stripping only covers fenced mentions)."""

    def test_unfenced_teaching_not_flagged(self):
        # A complete <invoke> shown as an example, with prose after it.
        ok, _ = ig.detect_leak(
            'ツールはこう書きます:\n<invoke name="Bash">\n'
            '<parameter name="command">ls</parameter>\n</invoke>\n'
            'というXML構造です。')
        self.assertFalse(ok)

    def test_unfenced_quote_of_leak_not_flagged(self):
        # Quoting the corruption (stray token + bare invoke) mid-prose.
        ok, _ = ig.detect_leak(
            'ログにこう出ていました。\ncourt\n<invoke name="Bash">\n</invoke>\n'
            'これが例のバグです。')
        self.assertFalse(ok)

    def test_unclosed_fence_not_flagged(self):
        # A code fence opened but never closed must still be stripped.
        ok, _ = ig.detect_leak(
            'Here is the shape:\n```\ncount\n<invoke name="Bash"></invoke>\n')
        self.assertFalse(ok)

    def test_complete_example_then_trailing_mention_not_flagged(self):
        # A complete <invoke> example mid-prose, with the text trailing off into
        # an unclosed <invoke> mention. The complete element is NOT terminal
        # (more text follows), so the bare-element signature must not fire on it.
        ok, _ = ig.detect_leak(
            'For example <invoke name="A">\n<parameter name="x">1</parameter>\n</invoke> '
            'is complete. You can also start one like <invoke name="B">')
        self.assertFalse(ok)

    def test_unclosed_invoke_in_prose_not_flagged(self):
        # A stray-token + UNCLOSED <invoke ...> shown in explanatory prose, with
        # text continuing after the opener tag. The opener is not the tail, so it
        # must not trip (signature A's truncated-leak path requires the opener to
        # end the turn).
        ok, _ = ig.detect_leak(
            'バグはこう見えます。\ncount\n<invoke name="Bash"> ... と続きます。これが問題。')
        self.assertFalse(ok)

    def test_terminal_leak_still_flagged(self):
        # The genuine form (invoke is the last thing in the turn) still trips.
        ok, sig = ig.detect_leak(
            'CI を確認します。\ncourt\n<invoke name="Bash">\n'
            '<parameter name="command">gh run list</parameter>\n</invoke>')
        self.assertTrue(ok)
        self.assertEqual(sig, "stray-token+invoke")

    def test_indented_code_example_not_flagged(self):
        # An indented (4-space) code example of the corruption at the tail of an
        # answer must not be flagged — the indented-code strip must remove it.
        text = (
            "Here is what the bug looks like:\n\n"
            "    count\n"
            '    <invoke name="Bash">\n'
            '    <parameter name="command">ls</parameter>\n'
            "    </invoke>\n"
        )
        ok, _ = ig.detect_leak(text)
        self.assertFalse(ok)
        # Control: the same content flush-left IS flagged.
        flush_text = (
            "Here is what the bug looks like:\n\n"
            "count\n"
            '<invoke name="Bash">\n'
            '<parameter name="command">ls</parameter>\n'
            "</invoke>\n"
        )
        ok_flush, sig_flush = ig.detect_leak(flush_text)
        self.assertTrue(ok_flush)
        self.assertEqual(sig_flush, "stray-token+invoke")


class TranscriptExtraction(unittest.TestCase):
    def test_last_assistant_text_leaked(self):
        txt = ig.last_assistant_text(os.path.join(FIX, "leaked_endturn.jsonl"))
        self.assertIn("<invoke", txt)

    def test_last_assistant_text_clean(self):
        # The last assistant turn is plain text; earlier tool_use is ignored.
        txt = ig.last_assistant_text(os.path.join(FIX, "clean.jsonl"))
        self.assertIn("Done.", txt)
        self.assertNotIn("<invoke", txt)

    def test_real_sanitized_fixture_detected(self):
        # Real corruption captured from a local transcript (sanitized). The last
        # assistant turn carries a "court" stray token.
        txt = ig.last_assistant_text(os.path.join(FIX, "real_leak_sanitized.jsonl"))
        d = ig.leak_details(txt)
        self.assertTrue(d["leak"])
        self.assertEqual(d["token"], "court")


class MalformedTranscriptLines(unittest.TestCase):
    """Lines that are valid JSON but not dict events (e.g. ``null``), or events
    whose ``message`` is not a dict, must be skipped -- not raise AttributeError.
    The AttributeError used to escape to run_hook's outer fail-open, silently
    disabling the guard for every Stop of the session."""

    LEAK_TURN = {"type": "assistant", "message": {
        "role": "assistant", "content": [{"type": "text", "text":
            "Resuming.\n\ncount\n<invoke name=\"Bash\">\n"
            "<parameter name=\"command\">ls</parameter>\n</invoke>"}]},
        "stop_reason": "end_turn"}

    def _write_transcript(self, tmp: str) -> str:
        path = os.path.join(tmp, "malformed.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("null\n")                # valid JSON, not a dict
            fh.write("42\n")                  # valid JSON, not a dict
            fh.write('["a", "b"]\n')          # valid JSON, not a dict
            fh.write(json.dumps({"type": "assistant",
                                 "message": "not a dict"}) + "\n")
            fh.write(json.dumps(self.LEAK_TURN) + "\n")
        return path

    def test_hook_still_blocks_leak_despite_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_transcript(tmp)
            code, out = _run_hook(
                {"hook_event_name": "Stop", "stop_hook_active": False,
                 "transcript_path": path}, env={"STOP_GUARD_NOLOG": "1"})
            self.assertEqual(code, 0)
            decision = json.loads(out)
            self.assertEqual(decision["decision"], "block")

    def test_extractors_skip_non_dict_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_transcript(tmp)
            self.assertIn("<invoke", ig.last_assistant_text(path))
            content = ig.last_assistant_content(path)
            self.assertIsInstance(content, list)

    def test_scan_cli_survives_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_transcript(tmp)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = ig.main(["--scan", path])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(buf.getvalue())["leak"])


class EmptyEndTurnDetection(unittest.TestCase):
    """Guard 2: a turn that ended on end_turn but produced no actionable
    content (no non-whitespace text, no tool_use, no thinking) is empty."""

    def test_empty_content_list_is_empty(self):
        self.assertTrue(ig.is_empty_turn([]))

    def test_whitespace_only_text_is_empty(self):
        self.assertTrue(ig.is_empty_turn([{"type": "text", "text": "   \n\t "}]))

    def test_nonwhitespace_text_is_not_empty(self):
        self.assertFalse(ig.is_empty_turn([{"type": "text", "text": "Done."}]))

    def test_tool_use_block_is_not_empty(self):
        self.assertFalse(ig.is_empty_turn([{"type": "tool_use", "id": "t1",
                                            "name": "Bash", "input": {}}]))

    def test_thinking_block_is_not_empty(self):
        self.assertFalse(ig.is_empty_turn([{"type": "thinking",
                                            "thinking": "hmm"}]))
        self.assertFalse(ig.is_empty_turn([{"type": "redacted_thinking",
                                            "data": "x"}]))

    def test_string_content_with_text_is_not_empty(self):
        self.assertFalse(ig.is_empty_turn("Final answer."))

    def test_string_content_whitespace_is_empty(self):
        self.assertTrue(ig.is_empty_turn("   "))

    def test_last_assistant_content_returns_blocks(self):
        content = ig.last_assistant_content(
            os.path.join(FIX, "empty_endturn.jsonl"))
        self.assertEqual(content, [])


class EmptyEndTurnHook(unittest.TestCase):
    def test_blocks_on_empty_end_turn(self):
        code, out = _run_hook({"hook_event_name": "Stop", "stop_hook_active": False,
                               "transcript_path": os.path.join(FIX, "empty_endturn.jsonl")},
                              env={"STOP_GUARD_NOLOG": "1"})
        self.assertEqual(code, 0)
        decision = json.loads(out)
        self.assertEqual(decision["decision"], "block")
        self.assertIn("empty end_turn", decision["reason"].lower())

    def test_blocks_on_whitespace_only_end_turn(self):
        code, out = _run_hook({"hook_event_name": "Stop", "stop_hook_active": False,
                               "transcript_path": os.path.join(FIX, "whitespace_endturn.jsonl")},
                              env={"STOP_GUARD_NOLOG": "1"})
        decision = json.loads(out)
        self.assertEqual(decision["decision"], "block")

    def test_normal_text_end_turn_not_blocked(self):
        # A real final answer (non-whitespace text) must never be blocked.
        code, out = _run_hook({"hook_event_name": "Stop", "stop_hook_active": False,
                               "transcript_path": os.path.join(FIX, "clean.jsonl")})
        self.assertEqual(out.strip(), "")

    def test_tool_use_turn_not_blocked(self):
        # The last assistant turn is a tool_use block (e.g. a fixture whose final
        # assistant content is structured tool use). It must not be treated empty.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tooluse.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "assistant", "message": {
                    "role": "assistant", "content": [
                        {"type": "tool_use", "id": "t1", "name": "Bash",
                         "input": {"command": "ls"}}]},
                    "stop_reason": "tool_use"}) + "\n")
            code, out = _run_hook(
                {"hook_event_name": "Stop", "stop_hook_active": False,
                 "transcript_path": path}, env={"STOP_GUARD_NOLOG": "1"})
            self.assertEqual(out.strip(), "")

    def test_reentrancy_guard_blocks_nothing(self):
        # stop_hook_active must prevent the empty-turn block from looping.
        code, out = _run_hook({"hook_event_name": "Stop", "stop_hook_active": True,
                               "transcript_path": os.path.join(FIX, "empty_endturn.jsonl")})
        self.assertEqual(out.strip(), "")

    def test_disable_env_allows_empty_turn(self):
        code, out = _run_hook({"hook_event_name": "Stop", "stop_hook_active": False,
                               "transcript_path": os.path.join(FIX, "empty_endturn.jsonl")},
                              env={"STOP_GUARD_DISABLE": "1"})
        self.assertEqual(out.strip(), "")

    def test_empty_turn_guard_can_be_disabled(self):
        # The empty-turn branch is independently disable-able while leak stays on.
        code, out = _run_hook(
            {"hook_event_name": "Stop", "stop_hook_active": False,
             "transcript_path": os.path.join(FIX, "empty_endturn.jsonl")},
            env={"STOP_GUARD_NO_EMPTY_TURN": "1", "STOP_GUARD_NOLOG": "1"})
        self.assertEqual(out.strip(), "")
        # ...but a leak is still blocked with the guard disabled.
        code, out = _run_hook(
            {"hook_event_name": "Stop", "stop_hook_active": False,
             "transcript_path": os.path.join(FIX, "leaked_endturn.jsonl")},
            env={"STOP_GUARD_NO_EMPTY_TURN": "1", "STOP_GUARD_NOLOG": "1"})
        self.assertEqual(json.loads(out)["decision"], "block")

    def test_empty_turn_logs_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "guard.log")
            _run_hook({"hook_event_name": "Stop", "stop_hook_active": False,
                       "session_id": "sess-empty",
                       "transcript_path": os.path.join(FIX, "empty_endturn.jsonl")},
                      env={"STOP_GUARD_LOG": log})
            with open(log, encoding="utf-8") as fh:
                rec = json.loads(fh.read().splitlines()[0])
            self.assertEqual(rec["signature"], "empty-end_turn")
            self.assertEqual(rec["action"], "block")


class HookDecision(unittest.TestCase):
    def test_blocks_on_leak(self):
        code, out = _run_hook({"hook_event_name": "Stop", "stop_hook_active": False,
                               "transcript_path": os.path.join(FIX, "leaked_endturn.jsonl")},
                              env={"STOP_GUARD_NOLOG": "1"})
        self.assertEqual(code, 0)
        decision = json.loads(out)
        self.assertEqual(decision["decision"], "block")
        self.assertIn("tool call", decision["reason"].lower())

    def test_allows_clean(self):
        code, out = _run_hook({"hook_event_name": "Stop", "stop_hook_active": False,
                               "transcript_path": os.path.join(FIX, "clean.jsonl")})
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "")

    def test_allows_fenced_discussion(self):
        code, out = _run_hook({"hook_event_name": "Stop", "stop_hook_active": False,
                               "transcript_path": os.path.join(FIX, "discussing_bug.jsonl")})
        self.assertEqual(out.strip(), "")

    def test_reentrancy_guard(self):
        # Even with a leaked transcript, stop_hook_active must prevent re-block.
        code, out = _run_hook({"hook_event_name": "Stop", "stop_hook_active": True,
                               "transcript_path": os.path.join(FIX, "leaked_endturn.jsonl")})
        self.assertEqual(out.strip(), "")

    def test_disable_env(self):
        code, out = _run_hook({"hook_event_name": "Stop", "stop_hook_active": False,
                               "transcript_path": os.path.join(FIX, "leaked_endturn.jsonl")},
                              env={"STOP_GUARD_DISABLE": "1"})
        self.assertEqual(out.strip(), "")

    def test_missing_transcript_fails_open(self):
        code, out = _run_hook({"hook_event_name": "Stop", "stop_hook_active": False,
                               "transcript_path": "/no/such/file.jsonl"})
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "")

    def test_garbage_stdin_fails_open(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = ig.run_hook("not json{{{")
        self.assertEqual(code, 0)
        self.assertEqual(buf.getvalue().strip(), "")

    def test_run_hook_non_utf8_transcript_fails_open(self):
        # The transcript file EXISTS but contains invalid UTF-8 bytes; reading it
        # raises UnicodeDecodeError inside last_assistant_text / last_assistant_content.
        # The hook must fail-open: exit 0, no block output.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_utf8.jsonl")
            with open(path, "wb") as fh:
                fh.write(b"\xff\xfe not utf8\n")
            code, out = _run_hook(
                {"hook_event_name": "Stop", "stop_hook_active": False,
                 "transcript_path": path},
                env={"STOP_GUARD_NOLOG": "1"})
            self.assertEqual(code, 0)
            # No block decision in output
            self.assertNotIn("block", out)


class ObserveAndLogging(unittest.TestCase):
    def test_observe_mode_does_not_block_but_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "guard.log")
            code, out = _run_hook(
                {"hook_event_name": "Stop", "stop_hook_active": False,
                 "session_id": "sess-1",
                 "transcript_path": os.path.join(FIX, "leaked_endturn.jsonl")},
                env={"STOP_GUARD_OBSERVE": "1", "STOP_GUARD_LOG": log})
            self.assertEqual(out.strip(), "")          # observe = never block
            self.assertTrue(os.path.exists(log))
            with open(log, encoding="utf-8") as fh:
                rec = json.loads(fh.read().splitlines()[0])
            self.assertEqual(rec["action"], "observe")
            self.assertEqual(rec["session_id"], "sess-1")
            self.assertEqual(rec["token"], "court")

    def test_block_mode_also_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "guard.log")
            code, out = _run_hook(
                {"hook_event_name": "Stop", "stop_hook_active": False,
                 "transcript_path": os.path.join(FIX, "leaked_endturn.jsonl")},
                env={"STOP_GUARD_LOG": log})
            self.assertEqual(json.loads(out)["decision"], "block")
            with open(log, encoding="utf-8") as fh:
                rec = json.loads(fh.read().splitlines()[0])
            self.assertEqual(rec["action"], "block")

    def test_bare_filename_log_does_not_crash(self):
        # STOP_GUARD_LOG with no directory component must still write (was a
        # silent FileNotFoundError swallowed by fail-open).
        with tempfile.TemporaryDirectory() as tmp:
            old = os.getcwd()
            os.chdir(tmp)
            try:
                _run_hook(
                    {"hook_event_name": "Stop", "stop_hook_active": False,
                     "transcript_path": os.path.join(FIX, "leaked_endturn.jsonl")},
                    env={"STOP_GUARD_LOG": "guard.log"})
                self.assertTrue(os.path.exists(os.path.join(tmp, "guard.log")))
            finally:
                os.chdir(old)

    def test_custom_tokens_via_env(self):
        # STOP_GUARD_TOKENS must flow through the hook path, not just the API.
        # The text is a stray token + an UNCLOSED <invoke> opener at the tail, so
        # there is no complete element for signature B -- the only thing that can
        # trip it is "boom" being treated as a stray token, which only happens
        # when STOP_GUARD_TOKENS makes it one.
        with tempfile.TemporaryDirectory() as tmp:
            leak = os.path.join(tmp, "boom.jsonl")
            with open(leak, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "assistant", "message": {
                    "role": "assistant",
                    "content": [{"type": "text",
                                 "text": 'go\nboom\n<invoke name="Bash">'}]},
                    "stop_reason": "end_turn"}) + "\n")
            # default tokens: "boom" is unknown and there is no complete element.
            code, out = _run_hook(
                {"hook_event_name": "Stop", "stop_hook_active": False,
                 "transcript_path": leak}, env={"STOP_GUARD_NOLOG": "1"})
            self.assertEqual(out.strip(), "")               # not a leak by default
            # with the env token, the same turn is now a stray-token leak.
            code, out = _run_hook(
                {"hook_event_name": "Stop", "stop_hook_active": False,
                 "transcript_path": leak},
                env={"STOP_GUARD_TOKENS": "boom", "STOP_GUARD_NOLOG": "1"})
            self.assertEqual(json.loads(out)["decision"], "block")

    def test_nolog_disables_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "guard.log")
            _run_hook(
                {"hook_event_name": "Stop", "stop_hook_active": False,
                 "transcript_path": os.path.join(FIX, "leaked_endturn.jsonl")},
                env={"STOP_GUARD_LOG": log, "STOP_GUARD_NOLOG": "1"})
            self.assertFalse(os.path.exists(log))

    def test_clean_turn_writes_no_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "guard.log")
            _run_hook(
                {"hook_event_name": "Stop", "stop_hook_active": False,
                 "transcript_path": os.path.join(FIX, "clean.jsonl")},
                env={"STOP_GUARD_LOG": log})
            self.assertFalse(os.path.exists(log))


class ScanCorpus(unittest.TestCase):
    def test_scan_fixtures_dir(self):
        files, turns, sessions, skipped, unparseable = scan_corpus.scan(FIX)
        summary = scan_corpus.summarize(files, turns, sessions, skipped, unparseable)
        # Ground-truth counts verified by running scan_corpus.scan(FIX) directly.
        # corrupted_turns=6: 2 from real_leak_sanitized, 1 from stale_leak_then_empty,
        # 1 from leaked_endturn, 1 from report_corpus/session_a, 1 from session_b.
        self.assertEqual(summary["corrupted_turns"], 6)
        self.assertEqual(summary["by_token"], {"court": 4, "course": 2})
        # the fenced discussion fixture must NOT contribute a leak
        self.assertTrue(all("discussing_bug" not in os.path.basename(t["file"])
                            for t in turns))

    def test_addressable_equals_end_turn_leaks(self):
        # The hook fires once per turn completion, so EVERY end_turn leak is a Stop
        # it can act on -- addressable = the end_turn count, regardless of whether
        # the leaked turn is the final one in the saved transcript.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "midleak.jsonl")
            a_leak = {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": 'court\n<invoke name="Bash">\n</invoke>'}]},
                "stop_reason": "end_turn"}
            a_clean = {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "All done, nothing further."}]},
                "stop_reason": "end_turn"}
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(a_leak) + "\n")
                fh.write(json.dumps({"type": "user", "message": {"role": "user",
                                                                 "content": "go on"}}) + "\n")
                fh.write(json.dumps(a_clean) + "\n")
            res = scan_corpus.scan(tmp)
            summary = scan_corpus.summarize(*res)
            # the mid-session end_turn leak still counts as addressable (a Stop
            # fired on it when it happened, even though more turns followed).
            self.assertEqual(summary["corrupted_turns"], 1)
            self.assertEqual(summary["hook_addressable_turns"], 1)

    def test_unreadable_and_unparseable_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "bad.jsonl"), "w", encoding="utf-8") as fh:
                fh.write("not json{{{\n")
                fh.write(json.dumps({"type": "assistant", "message": {"role": "assistant",
                    "content": [{"type": "text", "text": "clean"}]},
                    "stop_reason": "end_turn"}) + "\n")
            res = scan_corpus.scan(tmp)
            summary = scan_corpus.summarize(*res)
            self.assertEqual(summary["unparseable_lines"], 1)
            self.assertEqual(summary["corrupted_turns"], 0)


class ReliabilityReport(unittest.TestCase):
    """The reliability report categorises incidents over a corpus tree (read-only).
    Fixture corpus has a deterministic, known mix."""

    CORPUS = os.path.join(FIX, "report_corpus")

    def test_category_counts_match_fixture(self):
        report = scan_corpus.report(self.CORPUS)
        cat = report["categories"]
        self.assertEqual(cat["invoke_leak"], 2)
        self.assertEqual(cat["empty_end_turn"], 1)
        self.assertEqual(cat["malformed_retry_injected"], 1)
        self.assertEqual(cat["api_error"], 1)

    def test_invoke_leak_split_by_stop_reason(self):
        report = scan_corpus.report(self.CORPUS)
        self.assertEqual(report["invoke_leak_by_stop_reason"],
                         {"end_turn": 1, "tool_use": 1})

    def test_prose_mention_not_counted_as_retry_injection(self):
        # session_b's user msg mentions the malformed-retry text mid-sentence
        # but does NOT start with the exact prefix -> must not be counted.
        report = scan_corpus.report(self.CORPUS)
        self.assertEqual(report["categories"]["malformed_retry_injected"], 1)

    def test_sessions_affected_counted(self):
        report = scan_corpus.report(self.CORPUS)
        sess = report["sessions_affected"]
        self.assertEqual(sess["invoke_leak"], 2)   # one per file
        self.assertEqual(sess["empty_end_turn"], 1)
        self.assertEqual(sess["api_error"], 1)

    def test_per_day_and_recent_days_present(self):
        report = scan_corpus.report(self.CORPUS)
        self.assertIn("2026-06-20", report["by_day"])
        self.assertIn("2026-06-21", report["by_day"])
        self.assertEqual(report["by_day"]["2026-06-20"]["invoke_leak"], 1)
        self.assertEqual(report["by_day"]["2026-06-21"]["api_error"], 1)
        # recent_days is an ordered list of (day, counts) tuples/lists.
        recent_days = [d[0] for d in report["recent_days"]]
        self.assertEqual(recent_days, ["2026-06-20", "2026-06-21"])

    def test_per_day_rates_present(self):
        report = scan_corpus.report(self.CORPUS)
        # 2 distinct days, 2 invoke_leak -> rate 1.0 per active day.
        self.assertEqual(report["distinct_days"], 2)
        self.assertAlmostEqual(report["per_day_rate"]["invoke_leak"], 1.0)

    def test_report_cli_runs_readonly(self):
        # --report over the fixture corpus prints without error and mutates nothing.
        before = sorted(os.listdir(self.CORPUS))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = scan_corpus.main(["--root", self.CORPUS, "--report"])
        self.assertEqual(rc, 0)
        self.assertIn("invoke_leak", buf.getvalue())
        self.assertEqual(sorted(os.listdir(self.CORPUS)), before)


class StaleLeak(unittest.TestCase):
    """D1 / D2 / D3 / D4: regression tests for B1 (stale-leak) and B2 (null text)."""

    STALE_FIX = os.path.join(FIX, "stale_leak_then_empty.jsonl")

    def test_stale_leak_does_not_block_as_leak(self):
        """D1: three-turn transcript — leaked invoke (A), tool-use-only (B),
        empty end_turn (C). Guard 1 must NOT fire on the stale turn A text.
        Guard 2 must fire on the final empty turn C.
        The block reason must be EMPTY_TURN_REASON, not RETRY_REASON.
        """
        code, out = _run_hook(
            {"hook_event_name": "Stop", "stop_hook_active": False,
             "transcript_path": self.STALE_FIX},
            env={"STOP_GUARD_NOLOG": "1"})
        self.assertEqual(code, 0)
        decision = json.loads(out)
        self.assertEqual(decision["decision"], "block")
        # Must be the empty-turn message, not the leak/retry message.
        self.assertIn("empty end_turn", decision["reason"].lower())
        self.assertNotIn("mis-serialised", decision["reason"])
        self.assertEqual(decision["reason"], ig.EMPTY_TURN_REASON)

    def test_observe_mode_empty_turn_logs_correct_signature(self):
        """D2: observe mode on the stale-leak fixture — no block, log records
        action='observe' with signature='empty-end_turn'."""
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "guard.log")
            code, out = _run_hook(
                {"hook_event_name": "Stop", "stop_hook_active": False,
                 "session_id": "sess-observe-empty",
                 "transcript_path": self.STALE_FIX},
                env={"STOP_GUARD_OBSERVE": "1", "STOP_GUARD_LOG": log})
            self.assertEqual(out.strip(), "")           # observe = never block
            self.assertTrue(os.path.exists(log))
            with open(log, encoding="utf-8") as fh:
                rec = json.loads(fh.read().splitlines()[0])
            self.assertEqual(rec["action"], "observe")
            self.assertEqual(rec["signature"], "empty-end_turn")

    def test_none_content_is_not_empty(self):
        """D3: is_empty_turn(None) must return False (fail-open contract)."""
        self.assertFalse(ig.is_empty_turn(None))

    def test_null_text_block_does_not_raise(self):
        """D4: a content list with {"type":"text","text":null} must not raise.
        is_empty_turn treats it as empty (True), _content_to_text returns "".
        The hook must return exit 0 for such a transcript (no crash).
        """
        # _content_to_text must not raise on null text.
        result = ig._content_to_text([{"type": "text", "text": None}])
        self.assertEqual(result, "")

        # is_empty_turn must treat null-text block as empty (no non-whitespace).
        self.assertTrue(ig.is_empty_turn([{"type": "text", "text": None}]))

        # The hook must not crash on a transcript containing a null-text block.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "null_text.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "assistant", "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": None}]},
                    "stop_reason": "end_turn"}) + "\n")
            code, out = _run_hook(
                {"hook_event_name": "Stop", "stop_hook_active": False,
                 "transcript_path": path},
                env={"STOP_GUARD_NOLOG": "1"})
            self.assertEqual(code, 0)
            # A null-text block is empty (no actionable content) -> Guard 2 blocks.
            decision = json.loads(out)
            self.assertEqual(decision["decision"], "block")


class KnownFalsePositiveBoundary(unittest.TestCase):
    """T1: document the known false-positive boundary for an unfenced, complete,
    terminal <invoke> teaching example."""

    def test_unfenced_terminal_example_is_flagged(self):
        # This IS a known false positive: an unfenced, complete, terminal <invoke>
        # teaching example. Tolerated because OBSERVE mode + model self-correction
        # on retry; documented here so the boundary is visible.
        # If you suppress this FP, flip the assertion to assertFalse — do NOT revert your fix.
        ok, sig = ig.detect_leak(
            'Use <invoke name="X"><parameter name="p">v</parameter></invoke>')
        self.assertTrue(ok)
        self.assertEqual(sig, "bare-invoke-element")


class BK2Regression(unittest.TestCase):
    """T5: regression for BK2 — a .jsonl file with valid first line then invalid
    UTF-8 bytes mid-content must not crash scan(); skipped_files must be incremented."""

    def test_invalid_utf8_mid_file_does_not_crash_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad_utf8.jsonl")
            # Write one valid JSONL line followed by raw invalid UTF-8 bytes.
            # No real paths used; <redacted> placeholder keeps this fixture clean.
            with open(bad, "wb") as fh:
                valid_line = json.dumps({"type": "assistant", "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "All done."}]},
                    "stop_reason": "end_turn"}).encode("utf-8")
                fh.write(valid_line + b"\n")
                fh.write(b"\xff\xfe")   # invalid UTF-8 sequence
            # scan() must not raise; the bad file increments skipped_files.
            files, turns, sessions, skipped, unparseable = scan_corpus.scan(tmp)
            summary = scan_corpus.summarize(files, turns, sessions, skipped, unparseable)
            # The file was opened successfully but failed mid-read -> skipped_files.
            self.assertEqual(summary["skipped_files"], 1)
            # No corrupted turns extracted from the bad file.
            self.assertEqual(summary["corrupted_turns"], 0)


class BK2ReportRegression(unittest.TestCase):
    """T6: regression for F2 — report() mid-file UnicodeDecodeError must not
    crash; the file is counted as skipped, and report() returns normally."""

    def test_invalid_utf8_mid_file_does_not_crash_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad_utf8_report.jsonl")
            # Write one valid JSONL line followed by raw invalid UTF-8 bytes,
            # mirroring the BK2 scan() regression but exercising report().
            with open(bad, "wb") as fh:
                valid_line = json.dumps({
                    "type": "assistant",
                    "timestamp": "2026-06-20T12:00:00Z",
                    "message": {"role": "assistant",
                                "content": [{"type": "text", "text": "All done."}],
                                "stop_reason": "end_turn"},
                }).encode("utf-8")
                fh.write(valid_line + b"\n")
                fh.write(b"\xff\xfe")   # invalid UTF-8 sequence
            # report() must not raise; the bad file must be counted as skipped.
            result = scan_corpus.report(tmp)
            self.assertEqual(result["skipped_files"], 1)
            # No leak or empty-turn counted from the unreadable portion.
            self.assertEqual(result["categories"]["invoke_leak"], 0)
            self.assertEqual(result["categories"]["empty_end_turn"], 0)


class F3ScanNonDictMessage(unittest.TestCase):
    """T7: regression for F3 — scan() must continue and count an unparseable line
    when a JSONL event's message field is a list (non-dict) rather than a dict."""

    def test_non_dict_message_counted_as_unparseable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_msg.jsonl")
            # An assistant event whose message is a list instead of a dict causes
            # msg.get("content") to raise AttributeError; this must be caught and
            # counted, and scan must continue to the next line.
            bad_line = json.dumps({"type": "assistant", "message": ["not", "a", "dict"],
                                   "stop_reason": "end_turn"})
            good_line = json.dumps({"type": "assistant", "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "All done."}]},
                "stop_reason": "end_turn"})
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(bad_line + "\n")
                fh.write(good_line + "\n")
            files, turns, sessions, skipped, unparseable = scan_corpus.scan(tmp)
            summary = scan_corpus.summarize(files, turns, sessions, skipped, unparseable)
            # The bad line increments unparseable_lines; the good line is processed.
            self.assertEqual(summary["unparseable_lines"], 1)
            # The file itself is not skipped (only the one bad line).
            self.assertEqual(summary["skipped_files"], 0)
            # No leak in the good line.
            self.assertEqual(summary["corrupted_turns"], 0)


class R1MaskedTruncatedLeakRegression(unittest.TestCase):
    """R1: sig-A must use the LAST match, not the first.

    An earlier non-terminal complete <invoke>...</invoke> teaching example must
    not mask a later truncated terminal leak.  Regression for the .search()
    (first-match) asymmetry between sig-A and sig-B.
    """

    def test_earlier_complete_does_not_mask_later_truncated_leak(self):
        # First <invoke> is a complete, non-terminal teaching example (A closes
        # and prose continues).  Second <invoke name="B"> is an unclosed opener
        # at the tail — a genuine truncated leak.  sig-A must detect the second.
        text = (
            'Example:\ncourt\n<invoke name="A">\n</invoke>\n'
            'Now:\ncourt\n<invoke name="B">'
        )
        ok, sig = ig.detect_leak(text)
        self.assertTrue(ok, "expected leak on masked-truncated text")
        self.assertEqual(sig, "stray-token+invoke")


class T2MaxScanCharsTruncationTest(unittest.TestCase):
    """T2: MAX_SCAN_CHARS truncation — a trailing leak within the last 64 KB
    must still be caught even when the text is much larger."""

    def test_trailing_leak_past_max_scan_chars_is_caught(self):
        # Build text longer than 64 KB; the leak is in the trailing portion.
        text = "x" * (65 * 1024) + '\ncourt\n<invoke name="Bash">\n</invoke>'
        ok, sig = ig.detect_leak(text)
        self.assertTrue(ok, "expected leak to be caught after MAX_SCAN_CHARS truncation")
        self.assertEqual(sig, "stray-token+invoke")


class T4DanglingFenceOver64KB(unittest.TestCase):
    """T4: a dangling (unclosed) fence whose opener sits more than MAX_SCAN_CHARS
    before the end must still suppress an invoke inside it — because code is now
    stripped before the cap is applied (EDIT G fix)."""

    def test_dangling_fence_over_64kb_not_flagged(self):
        # Build a text: open fence, >64 KB filler, then a stray-token + bare invoke
        # still inside the open fence (fence never closed).
        filler = "filler line\n" * (ig.MAX_SCAN_CHARS // len("filler line\n") + 10)
        text = (
            "```\n"
            + filler
            + "count\n"
            + '<invoke name="Bash">\n'
            + '<parameter name="command">ls</parameter>\n'
            + "</invoke>\n"
        )
        # The total text is well above MAX_SCAN_CHARS so the old code (cap first,
        # strip second) would miss the fence opener and flag a false positive.
        self.assertGreater(len(text), ig.MAX_SCAN_CHARS)
        ok, _ = ig.detect_leak(text)
        self.assertFalse(ok, "invoke inside a dangling fence must not be flagged")


class T5CapitalizedCloseTag(unittest.TestCase):
    """T5: a capitalised closing tag </INVOKE> must be recognized as terminal."""

    def test_capitalized_close_tag_terminal_is_detected(self):
        # A terminal stray-token + bare invoke with a capitalised closing tag.
        text = (
            "Let me check.\n\n"
            "count\n"
            '<invoke name="Bash">\n'
            '<parameter name="command">ls</parameter>\n'
            "</INVOKE>"
        )
        ok, sig = ig.detect_leak(text)
        self.assertTrue(ok, "capitalised </INVOKE> must still be recognised as terminal")
        self.assertEqual(sig, "stray-token+invoke")


class T3ReportNonDictMessageRegression(unittest.TestCase):
    """T3: report() must continue and count an unparseable line when a JSONL
    event's message field is a list (non-dict) rather than a dict."""

    def test_non_dict_message_counted_as_unparseable_in_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_msg_report.jsonl")
            bad_line = json.dumps({
                "type": "assistant",
                "timestamp": "2026-06-20T10:00:00Z",
                "message": ["not", "a", "dict"],
                "stop_reason": "end_turn",
            })
            good_line = json.dumps({
                "type": "assistant",
                "timestamp": "2026-06-20T11:00:00Z",
                "message": {"role": "assistant",
                             "content": [{"type": "text", "text": "All done."}]},
                "stop_reason": "end_turn",
            })
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(bad_line + "\n")
                fh.write(good_line + "\n")
            result = scan_corpus.report(tmp)
            # The bad line increments unparseable_lines; report continues.
            self.assertEqual(result["unparseable_lines"], 1)
            # No leak or empty-turn counted from the bad or good lines.
            self.assertEqual(result["categories"]["invoke_leak"], 0)
            self.assertEqual(result["categories"]["empty_end_turn"], 0)
            # skipped_files stays 0 (only one line was unparseable, not the file).
            self.assertEqual(result["skipped_files"], 0)


if __name__ == "__main__":
    unittest.main()

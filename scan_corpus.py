#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scan_corpus -- quantify the INVOKE-leak bug across a transcript corpus.

Walks every ``*.jsonl`` under a projects root (default ~/.claude/projects),
runs the stop-guard detector over **every assistant turn**, and reports:

- total corrupted turns and how many sessions are affected,
- the stop_reason breakdown (``end_turn`` = halted / needed a resend,
  ``tool_use`` = harness auto-retry recovered it) -- the central claim of
  upstream issue #2763,
- a per-token histogram (count / call / court / course / ...),
- the date range (from file mtime).

This is the reproducible form of the ad-hoc scan in ``docs/issue-triage.md`` and
doubles as the **before/after effectiveness baseline** for the Stop hook:
re-run it after the hook has been live for a while and compare the ``end_turn``
count (those are the turns the hook would have converted into a retry).

Usage::

    python3 scan_corpus.py                 # scan ~/.claude/projects
    python3 scan_corpus.py --root DIR      # scan a specific tree
    python3 scan_corpus.py --json          # machine-readable summary
    python3 scan_corpus.py --list          # also print each corrupted turn
    python3 scan_corpus.py --report        # broad reliability report (read-only)
    python3 scan_corpus.py --report --json # report as JSON

The ``--report`` mode is a broader incident categoriser (no third-party
dependencies; requires stop_guard.py in the same directory plus the Python
standard library). It counts invoke_leak (split by stop_reason),
empty_end_turn,
malformed_retry_injected (exact-prefix harness injections), and api_error, with
per-day rates and a recent-days breakdown. It reuses stop_guard's helpers
(leak_details / content_to_text / is_empty_turn) instead of duplicating logic.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stop_guard as ig  # noqa: E402


def scan(root: str) -> tuple[list, list, set, int, int]:
    files = glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)
    turns = []          # list of dicts: file, line, signature, token, stop_reason
    sessions = set()
    skipped_files = 0       # could not be read at all
    unparseable_lines = 0   # individual JSONL lines that failed to decode
    for f in files:
        try:
            fh = open(f, encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            skipped_files += 1
            print(f"[scan_corpus] WARNING: {f!r} partially read or unreadable; counted as skipped",
                  file=sys.stderr)
            continue
        try:
            file_mtime = os.path.getmtime(f)
        except OSError:
            file_mtime = 0.0
        try:
            with fh:
                for i, line in enumerate(fh):   # stream lines (transcripts can be large)
                    try:
                        if not line.strip():
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            unparseable_lines += 1
                            continue
                        if evt.get("type") != "assistant":
                            continue
                        msg = evt.get("message", evt)
                        try:
                            d = ig.leak_details(ig.content_to_text(msg.get("content")))
                        except (AttributeError, TypeError):
                            unparseable_lines += 1
                            continue
                        if not d["leak"]:
                            continue
                        sr = msg.get("stop_reason") or evt.get("stop_reason") or "null"
                        turns.append({"file": f, "line": i, "signature": d["signature"],
                                      "token": d["token"], "stop_reason": sr,
                                      "mtime": file_mtime})
                        sessions.add(f)
                    except (UnicodeDecodeError, AttributeError, TypeError):
                        unparseable_lines += 1
        except (UnicodeDecodeError, OSError, AttributeError, TypeError):
            # A mid-file read error (e.g. invalid UTF-8 bytes encountered by the
            # iterator's __next__, a deleted file mid-walk, or a malformed message
            # field) must not abort the whole walk. Mirror report()'s per-file
            # isolation: count the file as skipped and move on.
            skipped_files += 1
            print(f"[scan_corpus] WARNING: {f!r} partially read or unreadable; counted as skipped",
                  file=sys.stderr)
    return files, turns, sessions, skipped_files, unparseable_lines


def summarize(files: list, turns: list, sessions: set, skipped_files: int = 0, unparseable_lines: int = 0) -> dict:
    by_stop = Counter(t["stop_reason"] for t in turns)
    by_token = Counter(t["token"] or f"({t['signature']})" for t in turns)
    by_session = defaultdict(Counter)
    for t in turns:
        by_session[t["file"]][t["stop_reason"]] += 1
    # The Stop hook fires once per turn completion (verified against Claude Code
    # hook docs), reading the assistant turn that just triggered the stop. So the
    # hook is evaluated on EVERY end_turn stop -- the addressable population is the
    # end_turn leaks (tool_use leaks are mid-response and the harness auto-recovers
    # them; the Stop hook never fires there). It is NOT limited to the final turn
    # of the saved transcript.
    mtimes = [t["mtime"] for t in turns]
    span = None
    if mtimes:
        fmt = "%Y-%m-%d"
        try:
            span = (datetime.datetime.fromtimestamp(min(mtimes)).strftime(fmt),
                    datetime.datetime.fromtimestamp(max(mtimes)).strftime(fmt))
        except (ValueError, OverflowError, OSError):
            span = None
    return {
        "scanned_transcripts": len(files),
        "skipped_files": skipped_files,
        "unparseable_lines": unparseable_lines,
        "corrupted_turns": len(turns),
        "affected_sessions": len(sessions),
        "hook_addressable_turns": by_stop.get("end_turn", 0),
        "by_stop_reason": dict(by_stop),
        "by_token": dict(by_token),
        "date_span": span,
        "per_session": {f: dict(c) for f, c in by_session.items()},
    }


# ---------------------------------------------------------------------------
# Reliability report: a broad incident categorisation across the corpus.
# No third-party dependencies; requires stop_guard.py in the same directory
# plus the Python standard library.
# READ-ONLY. Reuses stop_guard's helpers (leak_details / content_to_text /
# is_empty_turn) rather than re-deriving detection logic.
# ---------------------------------------------------------------------------

# Harness-injected retry marker. Matched as an EXACT PREFIX (not a prose mention)
# so a turn that merely *discusses* the text is never counted as an injection.
MALFORMED_RETRY_PREFIX = "Your tool call was malformed and could not be parsed. Please retry."

# Categories reported, in display order.
REPORT_CATEGORIES = (
    "malformed_retry_injected",
    "invoke_leak",
    "empty_end_turn",
    "api_error",
)


def _day(ts: "str | None") -> "str | None":
    """ISO date (YYYY-MM-DD) from an event timestamp, or None."""
    return ts[:10] if isinstance(ts, str) and len(ts) >= 10 else None


def report(root: str) -> dict:
    """Categorise reliability incidents across ``root`` (read-only).

    Counts invoke_leak (split by stop_reason), empty_end_turn,
    malformed_retry_injected (exact-prefix harness injections), and api_error.
    Returns totals, sessions affected, per-day counts, per-day rates, and a
    recent-days breakdown.
    """
    files = glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)
    categories: Counter = Counter()
    by_day: defaultdict = defaultdict(Counter)
    leak_by_stop: Counter = Counter()
    sessions_hit: defaultdict = defaultdict(set)
    days_seen: set = set()
    skipped_files = 0
    unparseable_lines = 0

    for f in files:
        try:
            fh = open(f, encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            skipped_files += 1
            continue
        try:
            with fh:
                for line in fh:
                    try:
                        if not line.strip():
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            unparseable_lines += 1
                            continue
                        try:
                            _classify_event(evt, f, categories, by_day, leak_by_stop,
                                            sessions_hit, days_seen)
                        except (TypeError, AttributeError):
                            unparseable_lines += 1
                    except UnicodeDecodeError:
                        unparseable_lines += 1
        except (UnicodeDecodeError, OSError, AttributeError, TypeError):
            # A mid-file read error (e.g. invalid UTF-8 bytes encountered by the
            # iterator's __next__, a deleted file mid-walk, or a malformed message
            # field) must not abort the whole walk. Mirror scan()'s per-file
            # isolation: count the file as skipped and move on.
            skipped_files += 1
            print(f"[scan_corpus] WARNING: {f!r} partially read or unreadable; counted as skipped",
                  file=sys.stderr)

    ndays = len(days_seen) or 1
    span = (min(days_seen), max(days_seen)) if days_seen else None
    recent_days = [(d, dict(by_day[d])) for d in sorted(by_day)[-14:]]
    return {
        "scanned_files": len(files),
        "skipped_files": skipped_files,
        "unparseable_lines": unparseable_lines,
        "distinct_days": len(days_seen),
        "date_span": span,
        "categories": {c: categories.get(c, 0) for c in REPORT_CATEGORIES},
        "invoke_leak_by_stop_reason": dict(leak_by_stop),
        "sessions_affected": {c: len(sessions_hit.get(c, ())) for c in REPORT_CATEGORIES},
        "per_day_rate": {c: categories.get(c, 0) / ndays for c in REPORT_CATEGORIES},
        "by_day": {d: dict(c) for d, c in by_day.items()},
        "recent_days": recent_days,
    }


def _classify_event(evt, f, categories, by_day, leak_by_stop, sessions_hit, days_seen):
    """Classify one transcript event into the report buckets (in place)."""
    msg = evt.get("message", evt)
    d = _day(evt.get("timestamp"))
    if d:
        days_seen.add(d)

    def hit(cat: str):
        categories[cat] += 1
        if d:
            by_day[d][cat] += 1
        sessions_hit[cat].add(f)

    if evt.get("isApiErrorMessage") is True:
        hit("api_error")
        return

    t = evt.get("type")
    if t == "user":
        txt = ig.content_to_text(msg.get("content")).strip()
        if txt.startswith(MALFORMED_RETRY_PREFIX):
            hit("malformed_retry_injected")
        return

    if t == "assistant":
        content = msg.get("content")
        det = ig.leak_details(ig.content_to_text(content))
        if det["leak"]:
            sr = msg.get("stop_reason") or evt.get("stop_reason") or "null"
            hit("invoke_leak")
            leak_by_stop[sr] += 1
        elif ig.is_empty_turn(content):
            sr = msg.get("stop_reason") or evt.get("stop_reason") or "null"
            # The Stop hook only fires on end_turn-equivalent stops; treat
            # null/missing stop_reason as end_turn to match that scope.
            if sr in ("end_turn", "null"):
                hit("empty_end_turn")


def _print_report(root: str) -> int:
    r = report(root)
    span = r["date_span"]
    print(f"scanned files : {r['scanned_files']}")
    if r.get("skipped_files") or r.get("unparseable_lines"):
        print(f"  WARNING: skipped {r.get('skipped_files', 0)} unreadable files, "
              f"{r.get('unparseable_lines', 0)} unparseable lines")
    if span:
        print(f"date span     : {span[0]} .. {span[1]}  ({r['distinct_days']} distinct days)")
    print()
    print(f"{'category':28} {'count':>7} {'sessions':>9} {'per-day':>8}")
    no_timestamps = r["distinct_days"] == 0
    if no_timestamps:
        print("  (no timestamps found; per-day rate not meaningful)")
    for c in REPORT_CATEGORIES:
        per_day_str = "     N/A" if no_timestamps else f"{r['per_day_rate'][c]:8.2f}"
        print(f"{c:28} {r['categories'][c]:7} {r['sessions_affected'][c]:9} "
              f"{per_day_str}")
    print()
    print("invoke_leak by stop_reason:", r["invoke_leak_by_stop_reason"])
    print()
    print("=== last 14 active days ===")
    for day_str, counts in r["recent_days"]:
        print(f"  {day_str}  "
              f"retry_inj={counts.get('malformed_retry_injected', 0):3}  "
              f"leak={counts.get('invoke_leak', 0):3}  "
              f"empty={counts.get('empty_end_turn', 0):3}  "
              f"api={counts.get('api_error', 0):3}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scan a transcript corpus for INVOKE leaks")
    ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects"),
                    help="projects root to scan (default ~/.claude/projects)")
    ap.add_argument("--json", action="store_true", help="emit JSON summary")
    ap.add_argument("--list", action="store_true", help="print each corrupted turn")
    ap.add_argument("--report", action="store_true",
                    help="broad reliability report (invoke_leak / empty_end_turn / "
                         "malformed_retry_injected / api_error) -- read-only")
    args = ap.parse_args(argv)

    if args.report:
        if args.json:
            print(json.dumps(report(args.root), ensure_ascii=False, indent=2))
            return 0
        return _print_report(args.root)

    files, turns, sessions, skipped, unparseable = scan(args.root)
    summary = summarize(files, turns, sessions, skipped, unparseable)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    s = summary
    print(f"scanned transcripts : {s['scanned_transcripts']}")
    if s["skipped_files"] or s["unparseable_lines"]:
        print(f"  (skipped {s['skipped_files']} unreadable files, "
              f"{s['unparseable_lines']} unparseable lines)")
    print(f"corrupted turns     : {s['corrupted_turns']}")
    print(f"affected sessions   : {s['affected_sessions']}")
    print(f"hook-addressable    : {s['hook_addressable_turns']}  "
          f"(end_turn leaks = a Stop fires on each, so the hook can convert it to a retry)")
    if s["date_span"]:
        print(f"date span           : {s['date_span'][0]} .. {s['date_span'][1]}")
    print("\nby stop_reason (end_turn = halted / needed resend; tool_use = harness auto-recovers):")
    for k, v in sorted(s["by_stop_reason"].items(), key=lambda x: -x[1]):
        print(f"  {k:10} {v}")
    print("\nby stray token:")
    for k, v in sorted(s["by_token"].items(), key=lambda x: -x[1]):
        print(f"  {k:14} {v}")
    if args.list:
        print("\ncorrupted turns:")
        for t in turns:
            print(f"  {os.path.basename(t['file'])} line{t['line']} "
                  f"[{t['signature']}/{t['token']}] stop={t['stop_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

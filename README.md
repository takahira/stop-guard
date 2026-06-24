# stop-guard

**A zero-dependency Claude Code Stop hook that rescues silent, empty, or leaked end-of-turn failures.**

It catches turns that end with **no usable result** and injects a retry, so your
autonomous loops don't die without finishing.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/takahira/stop-guard)

---

It guards two distinct end-of-turn failure modes:

1. **invoke-leak** — Opus 4.8 serializes a tool call into the text channel
   instead of executing it (the "court bug").
2. **empty / zero-content end_turn** — a turn ends with no text, no tool call,
   and no thinking. This one is **model-independent** (also hits Sonnet 4.6 and
   headless `-p` runs), so the guard isn't tied to a single model's bug.

The common seam: *an `end_turn` that produced nothing actionable.* The harness
treats it as a normal completion and moves on; the hook instead asks the model
to finish.

---

## Guard 1 — the invoke-leak bug

On `claude-opus-4-8`, a tool call is sometimes serialized into the **text
channel** as a bare tool-call element that is missing the `antml:` namespace
and the `function_calls` wrapper — often preceded by a stray token such as
`count` / `call` / `court` / `course` / `invoke`. The harness can't parse it, so **nothing
runs**.

The same corruption splits by `stop_reason`:

| `stop_reason` | Harness behavior | Result |
| --- | --- | --- |
| `tool_use` | Injects "Your tool call was malformed… Please retry." | **Self-heals** |
| `end_turn` | Treated as a normal text-only completion, no retry | **Silent death** |

The harness already knows how to recover — it just doesn't fire when the
mis-serialization lands on `end_turn`. **stop-guard fills exactly that gap**:
a Stop hook that detects the leaked tool call and emits a `block` decision so
the model retries, completing the same recovery the `tool_use` path gets, but
client-side only.

---

## Guard 2 — empty / zero-content end_turn

Independently of any leak, a turn sometimes ends on `end_turn` with **no
actionable content at all** — no non-whitespace text, no tool call, no thinking
block. The harness counts it as a successful completion, so a headless `-p` run
exits 0 with nothing produced and an autonomous loop quietly stops. This is
**model-independent** (reported on Sonnet 4.6 and headless runs, not just Opus).

The guard's definition of "empty" is deliberately strict so it **never blocks a
real text-only answer**: a turn with any non-whitespace text is not empty. When
it does fire, it blocks with a request to finish or to state why it stopped.

Disable this guard alone with `STOP_GUARD_NO_EMPTY_TURN=1` (leak detection
stays on). Disable everything with `STOP_GUARD_DISABLE=1`.

---

## What it does *not* do

This is a **pinpoint** guard, not a cure-all:

- Only the `end_turn` invoke-leak (the `tool_use` variant is already handled by
  the harness) and the empty/zero-content `end_turn`.
- Not parse-failure ("retry also failed"), not fabricated tool results. Those
  are separate failure modes a Stop hook can't safely catch.
- The invoke-leak guard is effectively Opus-4.8-specific (Sonnet 4.6 / Haiku
  4.5 / Fable 5 don't leak); the empty-turn guard is model-independent.
- The invoke-leak guard becomes obsolete once the upstream harness fixes the
  `end_turn` path; the empty-turn guard addresses a separate gap that is also
  still open upstream.

---

## Install

Requires Python 3.9+. No third-party dependencies.

1. Drop `stop_guard.py` somewhere stable.
2. Merge `settings-snippet.json` into `~/.claude/settings.json` (use an
   absolute path to the script).
3. **Run in observe mode first** (see below), confirm zero false positives on
   your own history, then enable blocking.

The hook is **fail-open**: any error → it stays silent and never blocks a turn.

---

## Confirm it's working

Smoke-test the detector directly, without touching your `settings.json`:

```bash
python3 stop_guard.py --self-test            # run the built-in detection samples
python3 stop_guard.py --scan SESSION.jsonl   # check one transcript's last assistant turn
echo "some text" | python3 stop_guard.py --check
```

Once the hook is registered, run Claude Code with `STOP_GUARD_DEBUG=1` to echo
every decision to stderr, and tail the detection log to see what fired:

```bash
tail -f ~/.claude/stop-guard.log
```

Each detection is one JSON line, for example:

```json
{"ts": 1781740800, "session_id": "…", "transcript": "/…/session.jsonl", "signature": "stray-token+invoke", "token": "court", "action": "block"}
```

If the log can't be written, stop-guard prints a `WARNING: detection NOT logged`
line to stderr (unconditionally, so a broken log is never silent — important in
observe mode, where "0 detections" must mean "nothing fired", not "logging failed").

**Uninstall:** remove the `Stop` entry you added from `~/.claude/settings.json`.

---

## Observe → block gating

The detector is tuned to favor false-negatives over false-positives (a wrong
block in normal conversation is the worst outcome). Still, verify on your own
corpus before trusting it.

### Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `STOP_GUARD_OBSERVE=1` | off | Observe mode: detect and log, but **never block**. Use this to dogfood the hook for a few days and confirm zero false positives before switching to blocking mode. |
| `STOP_GUARD_LOG=PATH` | `~/.claude/stop-guard.log` | Path for the detection log. Detections are always appended here (best-effort). |
| `STOP_GUARD_NOLOG=1` | off | Disable the detection log entirely. |
| `STOP_GUARD_NO_EMPTY_TURN=1` | off | Disable Guard 2 (empty/zero-content end_turn). Guard 1 (invoke-leak) stays active. |
| `STOP_GUARD_DISABLE=1` | off | Disable both guards entirely (the hook becomes a no-op). |
| `STOP_GUARD_TOKENS` | `count,call,court,course,invoke` | Override the stray-token list. Comma-separated; replaces the built-in set entirely. |
| `STOP_GUARD_DEBUG` | unset | Verbose detection logging to stderr. |

```bash
# How often the invoke-leak actually hits you (counts split by stop_reason;
# the end_turn count is the hook-addressable population)
python3 scan_corpus.py

# Broader reliability report: invoke_leak (by stop_reason), empty_end_turn,
# harness retry injections, and api_error — with per-day rates
python3 scan_corpus.py --report
```

`--report` is read-only and self-contained; point it at a different corpus with
`--root DIR`, or add `--json` for machine-readable output.

---

## How detection works

- Flags a leaked tool call only when the bare invoke construct is the **last
  meaningful content** of the turn (the "terminal guard"), so prose that quotes or
  teaches the syntax and keeps explaining afterwards is never blocked.
- Strips fenced, inline, **and** indented code before scanning, so a code *example*
  of the corruption is ignored — fence or indent such examples.
- A complete invoke element (or a stray token immediately followed by a bare invoke)
  left **unfenced and unindented** as the literal final content of a turn is treated
  as a real leak and blocked; this is the deliberate detection contract, the retry
  instruction offers a one-line release ("reply with a short confirmation"), and
  observe mode never blocks while you dogfood.
- Strips code on the full turn *before* capping to the last 64 KB, so a fence
  opener far above the cap is still honored and an invoke inside it is not mistaken
  for an unfenced leak. Memoizes the regex.
- For the empty-turn guard, treats a turn as empty only when it has no
  non-whitespace text **and** no tool-use **and** no thinking block — so a real
  final answer is never blocked. Both guards share the `stop_hook_active`
  re-entrancy guard, so a block is requested at most once per stop cycle.

See [docs/issue-triage.md](docs/issue-triage.md) for the upstream issue mapping
and the corruption signatures.

---

## Test

```bash
python3 -m unittest discover -s tests -p "test_*.py" -t . -v
```

73 tests, stdlib `unittest` only. Fixtures under `tests/fixtures/` are
sanitized transcript snippets.

---

## Files

| File | Purpose |
| --- | --- |
| `stop_guard.py` | The Stop hook + detector (also runnable for ad-hoc scans) |
| `scan_corpus.py` | Measure real incident rate across your transcript corpus (`--report` for the broad reliability breakdown) |
| `settings-snippet.json` | Hook registration snippet for `settings.json` |
| `tests/` | unittest suite + sanitized fixtures |
| `docs/issue-triage.md` | Upstream issue mapping + corruption signatures |

---

## License

MIT — see [LICENSE](LICENSE).

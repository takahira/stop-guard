# Contributing

## Run the test suite

```bash
python3 -m unittest discover -s tests -p "test_*.py" -t .
```

All tests must pass before opening a PR. The suite uses stdlib `unittest` only — no test dependencies to install.

## Adding detection logic

- Add a regression test for any new detection case (or false-positive boundary).
- Confirm that no existing test breaks (no change to detection outcome for existing samples).
- Detection is token/structure-based; changes to the stray-token list or regex must come with a new test that exercises the new path.

## Zero third-party dependencies

`stop_guard.py` and `scan_corpus.py` use the Python standard library only. Do not add `import` statements that require a `pip install`.

## Observe → block gating

When changing detection logic, validate in observe mode first:

```bash
STOP_GUARD_OBSERVE=1 python3 stop_guard.py
```

Run against your own corpus with `scan_corpus.py --report` before enabling blocking for a new detection rule.

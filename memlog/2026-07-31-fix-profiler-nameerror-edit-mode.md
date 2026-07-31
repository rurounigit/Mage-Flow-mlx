# Fix: NameError in single edit mode — `Profiler` not defined

## Date
2026-07-31

## Symptom
Running single edit mode (`--image` without `--worker`) crashed with:

```
NameError: name 'Profiler' is not defined
```

at `generate.py` line 797, inside `_run_edit()`.

## Root Cause
`Profiler` was imported as a **local variable** inside `main()` (line 191):

```python
from mage_mlx.profiler import Profiler, LiveReport, _C
```

This makes `Profiler` available only within `main()`'s local scope. The `_run_edit()`
function (a separate top-level function) does not have access to it.

The txt2img single mode (line 517) worked because that code is directly inside
`main()`, where `Profiler` is in scope. Worker modes worked because `worker.py`
imports `Profiler` at the module level (line 39).

## Fix
Changed line 797 in `_run_edit()`:

```diff
- thermal_state = Profiler.get_thermal_state()
+ thermal_state = prof.get_thermal_state()
```

`get_thermal_state()` is a `@classmethod` on `Profiler`, so it can be called on
an instance. `prof` is already passed as a parameter to `_run_edit()`, making
this the most minimal and correct fix — no new imports needed.

## Verification
- `grep -n 'Profiler' generate.py` confirms `Profiler` is now only referenced
  at lines 191, 193, and 517 — all inside `main()` where it is imported.
- No bare `Profiler` references remain in `_run_edit()` or any other function
  outside `main()`.

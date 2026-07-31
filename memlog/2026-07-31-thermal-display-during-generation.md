# Thermal Display During Generation (Non-Verbose Mode)

## Problem

In both worker modes (txt2img and edit) without the `--metadata` flag, the
thermal state text (`Thermal: CPU=2 (SERIOUS)`) was only displayed after the
progress bar finished — showing only the last thermal state instead of
informing the user during generation.

## Root Cause

`print_thermal_state()` in `mage_mlx/profiler.py` deferred printing in
non-verbose mode (`if not self.verbose: return`), storing the state in
`self._thermal_state` and only printing it at the end via
`_print_thermal_state_line()` in `print_summary()`.

## Solution

Modified `print_thermal_state()` to print the thermal line immediately in
non-verbose mode, using ANSI cursor control to keep it on a dedicated line
below the progress bar:

- **First call**: `print()` (newline to move below the bar) + print thermal
  text with `end=""` (no trailing newline)
- **Subsequent calls**: `\033[1B` (cursor down to thermal line) + `\033[K`
  (clear line) + reprint thermal text

Coordinated with `progress_bar()` and `finish()`:
- `progress_bar()`: when cursor is on the thermal line, moves UP (`\033[1A`)
  to the bar line before redrawing the bar
- `finish()`: unchanged — cursor is already on the bar line after the last
  `progress_bar()` call
- `print_summary()`: skips reprinting the thermal line (already on screen),
  with a fallback for the edge case where it was never printed

### Cursor tracking

Added `_cursor_on_thermal_line` flag to track whether the cursor is on the
thermal line (N+1) or the bar line (N). This prevents `progress_bar()` from
moving UP when the cursor is already on the bar line (e.g., after a previous
`progress_bar()` call).

## Files Modified

- `mage_mlx/profiler.py`:
  - `__init__`: Added `_thermal_line_printed` and `_cursor_on_thermal_line` flags
  - `progress_bar()`: Move cursor UP when on thermal line, set flag to False
  - `print_thermal_state()`: Print thermal line immediately in non-verbose mode
    using cursor DOWN movement for updates
- `finish()`: Added `_cursor_on_thermal_line` check — moves UP to bar line if cursor is on thermal line (edge case)
- `print_summary()`: Skip thermal line print if already printed, with fallback

## Verification

- All 94 existing tests pass (30 thermal + 61 edit worker + 3 quantization)
- Terminal simulation confirms:
  - Thermal text appears below the bar from the beginning
  - Thermal text updates as each prompt starts (NOMINAL → FAIR → SERIOUS)
  - Only one bar is ever displayed (updates in-place)
  - Final output shows `100%` on bar line and thermal text on line below

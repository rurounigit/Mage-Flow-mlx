# Terminal Output: Unified Single Bar + Overview Table

## Date: 2026-07-28

## Summary

Unified the terminal output across all three modes (single generation, edit, worker) to show **ONE SINGLE progress bar** and renamed the per-prompt results table to "Overview" with an overhead row.

## Changes

### 1. Overview Table (renamed from "Per-Prompt Results")
- Renamed table header from "Per-Prompt Results" to "Overview"
- Added an overhead row showing the time difference between total wall clock and sum of generation times
  - Single/edit mode: `overhead (load + encode + decode)`
  - Worker mode: `text encode / decode` + `overhead (load + encode + decode)`
- Fixed column alignment: RAM values now vertically aligned even when "lazy loading" label is shown

### 2. Single Progress Bar (ONE BAR, NEVER MULTIPLE)
- **Non-verbose mode** (without `--metadata`):
  - `progress_bar()` uses `\r\033[K` on ONE SINGLE LINE
  - ZERO `print()` statements with newlines between header and summary
  - No prompt headers, no phase headers, no cache status messages
- **Verbose mode** (with `--metadata`):
  - Full prompt headers, metadata blocks, and phase information shown
  - Progress bar still updates in-place on one line

### 3. Worker Mode Fixes
- Fixed indentation bug that caused only the first prompt to be processed
- Suppressed Phase 1/Phase 2 headers, Cache HIT/MISS messages in non-verbose mode
- Gated `print_run_metadata` and "Metadata saved" behind `if args.metadata`
- `pipeline_reload` renamed to `pipeline_load` for consistency

### 4. Lazy Loading Column Alignment
- Widened the time column and reduced phase-name column width
- Reduced space between phase name and "lazy loading" label so RAM values are vertically aligned

## Files Modified
- `mage_mlx/profiler.py` — LiveReport class: progress_bar, stop_phase, print_summary
- `mage_mlx/worker.py` — run_worker: verbose gating, indentation fix
- `generate.py` — main(): always create LiveReport + Profiler, gate metadata output

## Verification
- Single mode: 1 bar (16 events: python_startup → total_wall_clock), Overview table with overhead row ✓
- Edit mode: 1 bar, Overview table with overhead row ✓
- Worker mode: 1 bar, Overview table with text encode/decode + overhead rows ✓

## Additional Fix: Progress Bar Callback in Single Mode
- The `on_phase_complete` callback was inside `if verbose:` block, so in non-verbose mode
  DiT generation steps (dit_step_1-4, vae_decode) never reached `progress_bar()`
- Moved callback setup outside `if verbose:` so it runs in ALL modes
- Single mode now shows all 16 events updating on ONE SINGLE BAR

## Additional Fix: Worker Mode "Metadata saved" Line
- The "Metadata saved to ..." line was appearing in worker mode without `--metadata`
- Gated only the "Metadata saved" print behind `if args.metadata`
- The Run Metadata block remains visible in all modes (only the file-save confirmation is suppressed)

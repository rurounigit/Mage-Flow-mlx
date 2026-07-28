# Metadata Output Rewrite — 2026-07-28

## Objective

Rewrite the md/json file generation for all three modes (single generation,
worker, edit) so that the output files correspond exactly to the terminal
output with the `--metadata` parameter.

## Changes

### `mage_mlx/profiler.py`

- **`PhaseRecord`**: Added `saved_file` field to track the green-arrow file
  path in the phase table.
- **`Profiler`**: Added incremental save attributes:
  - `metadata_path` — base path for saving (when set, files are written
    after every phase completes)
  - `metadata` — run-level metadata dict
  - `overview` — list of per-prompt overview dicts
  - `summary` — summary dict (total_time, peak_ram, prompts_count, overhead,
    text_encode_time)
- **`Profiler.stop()`**: Now triggers `save_metadata()` when `metadata_path`
  is set, so files are written after every phase completes.
- **`Profiler.set_saved_file()`**: New method to set the saved_file for a
  named phase record.
- **`Profiler.to_markdown()`**: Rewritten to match the terminal output
  structure:
  1. Phase table (with saved_file column and per-phase metadata)
  2. Summary section (total time, peak RAM, prompts count)
  3. Overview table (per-prompt results)
  4. Overhead row (and text encode/decode row for worker mode)
  5. Run Metadata block
- **`Profiler.to_dict()`**: Rewritten to include `overview` and `summary`
  in the JSON output.
- **`Profiler.save_metadata()`**: Rewritten to accept `overview` and
  `summary` parameters and generate both JSON and markdown files.
- **`LiveReport`**: Added `profiler` parameter to `__init__()`. The
  `add_prompt()` method now updates `profiler.overview` and triggers
  incremental saves. The `stop_phase()` method syncs `saved_file` to the
  profiler record.

### `generate.py`

- **`LiveReport` creation**: Now passes `profiler=prof` to enable
  incremental saves.
- **Single generation mode**: Sets `prof.metadata_path` and
  `prof.metadata` before generation starts. Updated final save call to
  use the new `save_metadata()` signature with `overview` and `summary`.
- **Edit mode**: Sets `prof.metadata_path` and `prof.metadata` before
  `_run_edit()`. Updated final save call to use the new signature.
- **Worker mode**: The `run_worker()` function handles metadata setup
  internally.

### `mage_mlx/worker.py`

- **`run_worker()`**: Sets `profiler.metadata_path` and
  `profiler.metadata` before generation starts. Updated final
  `save_metadata()` call to use the new signature with `overview` and
  `summary`. The `LiveReport.add_prompt()` method triggers incremental
  saves after each prompt completes.

## Incremental Saving

When `--metadata` is enabled, the profiler writes md/json files to disk
after every phase completes. This means:

- If the process crashes mid-run, the last completed phase/prompt is
  already saved.
- The files are always up-to-date with the latest data.
- The `LiveReport.add_prompt()` method triggers a save after each prompt
  completes, so the overview table is always current.

## Verification

Tested with simulated data for all three modes:

- **Single generation**: Phase table, summary, overview, overhead row,
  run metadata — all correct.
- **Worker mode**: Phase table with numbered phases, summary, overview
  with multiple prompts, text encode/decode row, overhead row, run
  metadata — all correct.
- **Edit mode**: Same structure as single generation but with edit phases.

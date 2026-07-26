# Metadata Output + Quantize Reload Fix

**Date:** 2026-07-25
**Files changed:** `generate.py`, `mage_mlx/profiler.py`, `mage_mlx/worker.py`

## Summary

Added `--metadata` CLI flag (consolidated from `--profile`) that enables
phase-level profiling, prints the terminal report, AND saves profile data
with run-level metadata as both JSON and markdown files. Also fixed a bug
where per-prompt `quantize` changes in worker mode were silently ignored.

## Changes

### `mage_mlx/profiler.py`
- Added `import json` and `from typing import Any`
- Added `to_dict()` — returns all phase records as a JSON-serializable dict
- Added `to_markdown(title, metadata)` — returns markdown string with optional
  metadata table at top, profile data as markdown table
- Added `save_metadata(path, metadata, title, prompts)` — saves both:
  - `path + ".json"` — JSON with metadata, phases, prompts, and total
  - `path + ".md"` — markdown with metadata table + profile table
- Added `prompts` parameter to `save_metadata` for per-prompt metadata (worker mode)

### `generate.py`
- Added `import json`, `from datetime import datetime`, `from typing import Optional`
- Removed `--profile` flag, consolidated into `--metadata`
- Added `--metadata` CLI flag (boolean) — enables profiling, prints terminal
  report, and saves JSON + markdown files
- Added `_get_model_name(model_dir)` — converts folder name to HF path
  (e.g. `models/microsoft_Mage-Flow-Turbo` → `microsoft/Mage-Flow-Turbo`)
- Added `_get_base_model(model_dir)` — reads `_class_name` from
  `transformer_config.json`, falls back to model_dir
- Added `_collect_metadata(...)` — builds the metadata dict with all required
  fields: model, base_model, quantize, generation_time_seconds, created_at,
  image_path, image_paths, image_strength, peak_memory_gib
- Wired up `--metadata` in all three modes:
  - Single prompt: saves `output.json` + `output.md`
  - Worker: passes `metadata_enabled` to `run_worker`
  - Edit: saves `output.json` + `output.md`
- Profiler is now enabled when `--metadata` is set
- Added per-prompt metadata to profile table: prompt, resolution, steps, quantize
- For txt2img: `image_path` = null, `image_paths` = null (only applies to edit)
- For edit: `image_path` = target image, `image_paths` = reference images
- `generation_time_seconds` uses `total_wall_clock` (not generation phase time)

### `mage_mlx/worker.py`
- Added `metadata_enabled` parameter to `run_worker()`
- Added `_get_model_name()` and `_get_base_model()` helpers
- Added `from datetime import datetime`
- **Quantize reload fix**: when `needs_pipeline` is True in the generation
  loop, the pipeline is now reloaded via `MageFlowPipeline.from_pretrained()`
  with the new model/quantize. The text encoder is unloaded from the new
  pipeline (we use cached embeddings). Profiler phases `pipeline_reload_N`
  and `text_encoder_unload_N` are recorded.
- Added `quantize` and `prompt` to per-generation metadata (alongside
  `resolution` and `steps`)
- Collects per-prompt metadata (prompt, negative_prompt, quantize, resolution,
  steps, generation_time_seconds, image_path) for JSON output
- Stops `total_wall_clock` before saving metadata (so the MD/JSON shows
  `total_wall_clock` instead of "Sum of all phases")
- At the end of the run, saves `test_prompts.json` + `test_prompts.md` with
  all metadata and per-prompt info
- `image_path` and `image_paths` are null in worker mode (only applies to edit)

## Metadata fields

| Field | Source |
|-------|--------|
| `model` | `_get_model_name(model_dir)` — HF path |
| `base_model` | `transformer_config.json` `_class_name` or model_dir |
| `quantize` | `params.get("quantize")` — per-prompt in profile table |
| `generation_time_seconds` | `profiler.get_elapsed("total_wall_clock")` |
| `created_at` | `datetime.now().isoformat()` |
| `image_path` | Target image (edit) or `null` (txt2img/worker) |
| `image_paths` | Reference images (edit) or `null` (txt2img/worker) |
| `image_strength` | `null` (txt2img only) |
| `peak_memory_gib` | `profiler._get_rss_gib()` at end of run |

## Per-prompt profile table metadata

Each `generation_N` phase in the profile table includes:
- `resolution` (e.g. `1024x1024`)
- `steps` (e.g. `4`)
- `quantize` (e.g. `None` or `4`)
- `prompt` (the full prompt text)

## Usage

```bash
# Single prompt with metadata
python generate.py --prompt "A cat" --metadata

# Worker mode with metadata
python generate.py --worker prompts.jsonl --metadata
```

## Quantize reload behavior

When a prompt in the JSONL file has a different `quantize` value than the
previous prompt, the pipeline is reloaded silently. The reload is recorded
as a profiler phase (`pipeline_reload_N`) so it appears in the profile table.
Cached text embeddings remain valid (quantize only affects the DiT, not the
text encoder).

## Log

======================================================================
  Mage-Flow MLX
======================================================================
  Phase                                          Time     Peak RAM
  ──────────────────────────────────────────────────────────────
  Importing Mage-Flow MLX pipeline from microsoft/Mage-Flow-Edit-Turbo...
  Current working directory: /Users/tilman/projects/Mage-Flow-mlx
  Python executable: /Users/tilman/projects/Mage-Flow-mlx/.venv/bin/python
  Imported MageFlowPipeline successfully

## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0000 | 0.22 |  | |
| pipeline_load | 2.3328 | 8.27 |  | |
| dit_load | 1.4250 | 7.93 |  | |
| vae_load | 0.0000 | 7.93 |  | |
| edit_step_1 | 6.2750 | 7.87 |  | |
| edit_step_2 | 6.0500 | 7.87 |  | |
| edit_step_3 | 6.1263 | 7.87 |  | |
| edit_step_4 | 6.1966 | 7.87 |  | |
| vae_decode | 0.3561 | 7.87 |  | |
| edit | 34.8707 | 7.86 |  | |
prompt=change the shoe to deep burgundy polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0650 | 7.86 | test_10_shoe_edited.png | |
| total_wall_clock | 37.7208 |  |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*

## Summary

| Metric | Value |
|--------|-------|
| Total time | 37.7208 |
| Peak RAM | 8.27 |
| Prompts | 1 |

## Overview

| # | Time (s) | Peak RSS (GiB) | Resolution | Steps | File |
|---|----------|----------------|------------|-------|------|
| 1 | 34.8707 | 7.87 | 1024x1024 | 4 | test_10_shoe_edited.png |
| — | 2.8501 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | Mage-Flow-Edit-Turbo |
| base_model | microsoft/Mage-Flow-Edit-Turbo |
| generation_time_seconds | 37.720838624984026 |
| created_at | 2026-07-28T20:31:25.845546 |
| image_path | test_10_shoe.png |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.26607609540224 |

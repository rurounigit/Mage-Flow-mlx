## Log

======================================================================
  Mage-Flow MLX
======================================================================
  Importing Mage-Flow MLX pipeline from microsoft/Mage-Flow-Edit-Turbo...
  Current working directory: /Users/tilman/projects/Mage-Flow-mlx
  Python executable: /Users/tilman/projects/Mage-Flow-mlx/.venv/bin/python
  Imported MageFlowPipeline successfully

## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0 | 0.22 |  |  |
| pipeline_load | 3.0 | 8.27 |  |  |
| dit_load | 1.6 | 7.93 |  |  |
| vae_load | 0.0 | 7.93 |  |  |
| edit_step_1 | 7.2 | 7.87 |  |  |
| edit_step_2 | 7.2 | 7.87 |  |  |
| edit_step_3 | 7.4 | 7.87 |  |  |
| edit_step_4 | 7.6 | 7.87 |  |  |
| vae_decode | 1.9 | 7.87 |  |  |
| edit | 42.6 | 7.86 |  | prompt=change the shoe to deep burgundy polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=42 |
| save_png | 0.2 | 7.86 | test_10_shoe_edited.png |  |
| total_wall_clock | 46.6 |  |  |  |

## Summary

| Metric | Value |
|--------|-------|
| Total time | 46.6 |
| Peak RAM | 8.27 |
| Prompts | 1 |

## Overview

|_ | Time (s) | Peak RSS (GiB) | Resolution | Steps | Seed | File |
|---|----------|----------------|------------|-------|------|------|
| 1 | 42.6 | 7.87 | 1024x1024 | 4 | 42 | test_10_shoe_edited.png |
| — | 4.0 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | Mage-Flow-Edit-Turbo |
| base_model | microsoft/Mage-Flow-Edit-Turbo |
| generation_time_seconds | 46.6 |
| created_at | 2026-07-28T22:58:14.495011 |
| image_path | test_10_shoe.png |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.27 |

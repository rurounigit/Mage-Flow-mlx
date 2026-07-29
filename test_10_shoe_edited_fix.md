## Log

======================================================================
  Mage-Flow MLX
======================================================================
  Importing Mage-Flow MLX pipeline from models/microsoft_Mage-Flow-Edit-Turbo...
  Current working directory: /Users/tilman/projects/Mage-Flow-mlx
  Python executable: /Users/tilman/projects/Mage-Flow-mlx/.venv/bin/python
  Imported MageFlowPipeline successfully

## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0 | 0.22 |  |  |
| pipeline_load | 4.0 | 8.27 |  |  |
| dit_load | 2.5 | 13.88 |  |  |
| vae_load | 0.0 | 13.88 |  |  |
| edit_step_1 | 20.3 | 13.90 |  |  |
| edit_step_2 | 39.0 | 13.90 |  |  |
| edit_step_3 | 75.9 | 13.90 |  |  |
| edit_step_4 | 80.9 | 13.90 |  |  |
| vae_decode | 1.3 | 13.89 |  |  |
| edit | 222.4 | 13.88 |  | prompt=change the shoe to deep burgundy polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=42 |
| save_png | 0.2 | 13.88 | test_10_shoe_edited_fix.png |  |
| total_wall_clock | 227.2 |  |  |  |

## Summary

| Metric | Value |
|--------|-------|
| Total time | 227.2 |
| Peak RAM | 13.90 |
| Prompts | 1 |

## Overview

|_ | Time (s) | Peak RSS (GiB) | Resolution | Steps | Seed | File |
|---|----------|----------------|------------|-------|------|------|
| 1 | 222.4 | 13.90 | 1024x1024 | 4 | 42 | test_10_shoe_edited_fix.png |
| — | 4.8 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Edit-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 227.2 |
| created_at | 2026-07-29T13:22:08.592329 |
| image_path | test_10_shoe.png |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 13.9 |

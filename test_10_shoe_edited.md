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
| pipeline_load | 2.5 | 8.27 |  |  |
| dit_load | 1.6 | 7.93 |  |  |
| vae_load | 0.0 | 7.93 |  |  |
| edit_step_1 | 7.0 | 7.87 |  |  |
| edit_step_2 | 6.9 | 7.87 |  |  |
| edit_step_3 | 7.0 | 7.87 |  |  |
| edit_step_4 | 7.2 | 7.87 |  |  |
| vae_decode | 2.5 | 7.87 |  |  |
| edit | 41.8 | 7.86 |  | prompt=change the shoe to deep burgundy polished leather with a translucent smoke-gr..., resolution=1024x1024, steps=4, quantize=None |
| save_png | 0.2 | 7.86 | test_10_shoe_edited.png |  |
| total_wall_clock | 45.1 |  |  |  |

## Summary

| Metric | Value |
|--------|-------|
| Total time | 45.1 |
| Peak RAM | 8.27 |
| Prompts | 1 |

## Overview

| # | Time (s) | Peak RSS (GiB) | Resolution | Steps | File |
|---|----------|----------------|------------|-------|------|
| 1 | 41.8 | 7.87 | 1024x1024 | 4 | test_10_shoe_edited.png |
| — | 3.4 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | Mage-Flow-Edit-Turbo |
| base_model | microsoft/Mage-Flow-Edit-Turbo |
| generation_time_seconds | 45.141750584123656 |
| created_at | 2026-07-28T21:22:17.845730 |
| image_path | test_10_shoe.png |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.26607609540224 |

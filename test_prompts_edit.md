## Log

======================================================================
  Mage-Flow MLX
======================================================================
  Importing Mage-Flow MLX pipeline from models/microsoft_Mage-Flow-Edit-Turbo...
  Current working directory: /Users/tilman/projects/Mage-Flow-mlx
  Python executable: /Users/tilman/projects/Mage-Flow-mlx/.venv/bin/python
  Imported MageFlowPipeline successfully
  Loaded 2 edit prompts from test_prompts_edit.jsonl

## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0 | 0.22 |  |  |
| pipeline_load | 3.4 | 8.27 |  |  |
| text_encoder_unload | 0.2 | 0.44 |  |  |
| dit_load | 1.5 | 7.92 |  |  |
| vae_load | 0.0 | 7.92 |  |  |
| edit_step_1 | 3.7 | 7.87 |  |  |
| edit_step_2 | 3.6 | 7.87 |  |  |
| edit_step_3 | 3.6 | 7.87 |  |  |
| edit_step_4 | 3.6 | 7.87 |  |  |
| vae_decode | 0.4 | 7.86 |  |  |
| generation_1 | 15.0 | 7.86 |  | prompt=change the shoe to metallic green mesh with a translucent  sole and subtle fur details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=42 |
| save_1 | 0.2 | 7.86 | test_10_shoe_edited_worker_01.png |  |
| edit_step_1 | 3.7 | 7.87 |  |  |
| edit_step_2 | 3.9 | 7.87 |  |  |
| edit_step_3 | 3.9 | 7.87 |  |  |
| edit_step_4 | 3.9 | 7.87 |  |  |
| vae_decode | 0.4 | 7.86 |  |  |
| generation_2 | 15.9 | 7.86 |  | prompt=change the shoe to deep blue polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=43 |
| save_2 | 0.3 | 7.86 | test_10_shoe_edited_worker_02.png |  |
| total_wall_clock | 36.9 |  |  |  |

## Summary

| Metric | Value |
|--------|-------|
| Total time | 36.9 |
| Peak RAM | 8.27 |
| Prompts | 2 |

## Overview

|_ | Time (s) | Peak RSS (GiB) | Resolution | Steps | Seed | File |
|---|----------|----------------|------------|-------|------|------|
| 1 | 15.0 | 7.86 | 1024x1024 | 4 | 42 | test_10_shoe_edited_worker_01.png |
| 2 | 15.9 | 7.86 | 1024x1024 | 4 | 43 | test_10_shoe_edited_worker_02.png |
| — | 1.6 | — | — | — | text encode / decode |
| — | 4.4 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Edit-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 36.9 |
| created_at | 2026-07-29T15:04:06.426047 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.27 |

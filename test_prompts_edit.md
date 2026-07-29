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
| pipeline_load | 3.5 | 8.27 |  |  |
| text_encoder_unload | 0.1 | 0.44 |  |  |
| dit_load | 1.4 | 13.88 |  |  |
| vae_load | 0.0 | 13.88 |  |  |
| edit_step_1 | 22.0 | 13.89 |  |  |
| edit_step_2 | 44.9 | 13.89 |  |  |
| edit_step_3 | 28.3 | 13.89 |  |  |
| edit_step_4 | 67.9 | 13.89 |  |  |
| vae_decode | 1.5 | 13.90 |  |  |
| generation_1 | 164.7 | 13.90 |  | prompt=change the shoe to metallic green mesh with a translucent  sole and subtle fur details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=42 |
| save_1 | 0.2 | 13.90 | test_10_shoe_edited_worker_01.png |  |
| edit_step_1 | 28.7 | 13.90 |  |  |
| edit_step_2 | 26.7 | 13.90 |  |  |
| edit_step_3 | 37.6 | 13.90 |  |  |
| edit_step_4 | 52.6 | 13.90 |  |  |
| vae_decode | 1.8 | 13.90 |  |  |
| generation_2 | 147.6 | 13.90 |  | prompt=change the shoe to deep blue polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=43 |
| save_2 | 0.2 | 13.90 | test_10_shoe_edited_worker_02.png |  |
| total_wall_clock | 318.3 |  |  |  |

## Summary

| Metric | Value |
|--------|-------|
| Total time | 318.3 |
| Peak RAM | 13.90 |
| Prompts | 2 |

## Overview

|_ | Time (s) | Peak RSS (GiB) | Resolution | Steps | Seed | File |
|---|----------|----------------|------------|-------|------|------|
| 1 | 164.7 | 13.90 | 1024x1024 | 4 | 42 | test_10_shoe_edited_worker_01.png |
| 2 | 147.6 | 13.90 | 1024x1024 | 4 | 43 | test_10_shoe_edited_worker_02.png |
| — | 1.5 | — | — | — | text encode / decode |
| — | 4.4 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Edit-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 318.3 |
| created_at | 2026-07-29T13:02:01.309564 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 13.9 |

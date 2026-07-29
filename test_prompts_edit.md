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
| pipeline_load | 3.8 | 8.27 |  |  |
| text_encoder_unload | 0.2 | 0.41 |  |  |
| dit_load | 1.5 | 7.92 |  |  |
| vae_load | 0.0 | 7.92 |  |  |
| edit_step_1 | 4.0 | 7.87 |  |  |
| edit_step_2 | 3.7 | 7.87 |  |  |
| edit_step_3 | 3.8 | 7.87 |  |  |
| edit_step_4 | 3.8 | 7.87 |  |  |
| vae_decode | 1.3 | 7.86 |  |  |
| generation_1 | 16.8 | 7.86 |  | prompt=change the shoe to metallic green mesh with a translucent  sole and subtle fur details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=42 |
| save_1 | 0.2 | 7.86 | test_10_shoe_edited_worker_01.png |  |
| edit_step_1 | 4.0 | 7.87 |  |  |
| edit_step_2 | 3.9 | 7.87 |  |  |
| edit_step_3 | 4.0 | 7.87 |  |  |
|-------|----------|----------------|------------|----------|
| **Sum of all phases** | **51.0** | | | |

## Overview

|_ | Time (s) | Peak RSS (GiB) | Resolution | Steps | Seed | File |
|---|----------|----------------|------------|-------|------|------|
| 1 | 16.8 | 7.86 | 1024x1024 | 4 | 42 | test_10_shoe_edited_worker_01.png |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Edit-Turbo |
| base_model | MageFlow |
| generation_time_seconds | None |
| created_at | 2026-07-29T15:15:02.711831 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | None |

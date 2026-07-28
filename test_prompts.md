## Log

======================================================================
  Mage-Flow MLX
======================================================================
  Importing Mage-Flow MLX pipeline from models/microsoft_Mage-Flow-Turbo...
  Current working directory: /Users/tilman/projects/Mage-Flow-mlx
  Python executable: /Users/tilman/projects/Mage-Flow-mlx/.venv/bin/python
  Imported MageFlowPipeline successfully
  Loaded 2 prompts from test_prompts.jsonl

## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0 | 0.22 |  |  |
| text_encoder_load | 0.0 | 0.23 |  | tensors=713 |
| tokenizer_load | 2.2 | 0.44 |  |  |
| pipeline_load | 2.3 | 0.44 |  |  |
| text_encoder_unload | 0.0 | 0.44 |  |  |
| dit_load | 2.0 | 7.67 |  |  |
| vae_load | 0.1 | 7.93 |  | tensors=728 |
| dit_step_1 | 3.3 | 7.93 |  |  |
| dit_step_2 | 3.2 | 7.93 |  |  |
| dit_step_3 | 3.2 | 7.93 |  |  |
| dit_step_4 | 3.2 | 7.93 |  |  |
| vae_decode | 2.5 | 7.94 |  |  |
| generation_1 | 15.4 | 7.93 |  | prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting., resolution=1024x1024, steps=4, quantize=None |
| save_1 | 0.3 | 7.93 | test_01_airship_dragon_new.png |  |
| dit_step_1 | 3.4 | 7.93 |  |  |
| dit_step_2 | 3.3 | 7.93 |  |  |
| dit_step_3 | 3.1 | 7.93 |  |  |
|-------|----------|----------------|------------|----------|
| **Sum of all phases** | **47.4** | | | |

## Overview

| # | Time (s) | Peak RSS (GiB) | Resolution | Steps | File |
|---|----------|----------------|------------|-------|------|
| 1 | 15.4 | 7.93 | 1024x1024 | 4 | test_01_airship_dragon_new.png |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | None |
| created_at | 2026-07-28T22:37:34.101782 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | None |

## Log

======================================================================
  Mage-Flow MLX
======================================================================
  Importing Mage-Flow MLX pipeline from models/microsoft_Mage-Flow-Turbo...
  Current working directory: /Users/tilman/projects/Mage-Flow-mlx
  Python executable: /Users/tilman/projects/Mage-Flow-mlx/.venv/bin/python
  Imported MageFlowPipeline successfully

## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0 | 0.22 |  |  |
| text_encoder_load | 0.0 | 0.23 |  | tensors=713 |
| tokenizer_load | 2.3 | 0.44 |  |  |
| pipeline_load | 2.3 | 0.44 |  |  |
| text_encode | 1.6 | 7.49 |  | cache=MISS |
| text_encoder_unload | 0.2 | 0.25 |  |  |
| dit_load | 1.4 | 7.67 |  |  |
| vae_load | 0.1 | 7.93 |  | tensors=728 |
| dit_step_1 | 3.3 | 7.93 |  |  |
| dit_step_2 | 3.3 | 7.93 |  |  |
| dit_step_3 | 3.3 | 7.93 |  |  |
| dit_step_4 | 3.3 | 7.93 |  |  |
| vae_decode | 2.4 | 7.94 |  |  |
| generation | 19.0 | 7.93 |  | prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic., resolution=1024x1024, steps=4, quantize=None, seed=42 |
| save_png | 0.1 | 7.93 | test_10_shoe.png |  |
| total_wall_clock | 21.5 |  |  |  |

## Summary

| Metric | Value |
|--------|-------|
| Total time | 21.5 |
| Peak RAM | 7.94 |
| Prompts | 1 |

## Overview

|_ | Time (s) | Peak RSS (GiB) | Resolution | Steps | Seed | File |
|---|----------|----------------|------------|-------|------|------|
| 1 | 19.0 | 7.94 | 1024x1024 | 4 | 42 | test_10_shoe.png |
| — | 2.5 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 21.5 |
| created_at | 2026-07-28T22:56:36.363874 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.94 |

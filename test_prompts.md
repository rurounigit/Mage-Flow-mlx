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
| tokenizer_load | 2.3 | 0.44 |  |  |
| pipeline_load | 2.4 | 0.44 |  |  |
| text_encode_1 | 1.5 | 7.90 |  | cache=MISS |
| text_encode_2 | 0.1 | 7.79 |  | cache=MISS |
| text_encoder_unload | 0.1 | 0.43 |  |  |
| dit_load | 1.4 | 7.67 |  |  |
| vae_load | 0.1 | 7.93 |  | tensors=728 |
| dit_step_1 | 3.2 | 7.93 |  |  |
| dit_step_2 | 3.2 | 7.93 |  |  |
| dit_step_3 | 3.3 | 7.93 |  |  |
| dit_step_4 | 3.2 | 7.93 |  |  |
| vae_decode | 2.1 | 7.94 |  |  |
| generation_1 | 15.1 | 7.93 |  | prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting., resolution=1024x1024, steps=4, quantize=None, seed=42 |
| save_1 | 0.1 | 7.93 | test_01_airship_dragon_new.png |  |
| dit_step_1 | 4.3 | 7.93 |  |  |
| dit_step_2 | 3.1 | 7.93 |  |  |
| dit_step_3 | 3.2 | 7.93 |  |  |
| dit_step_4 | 3.9 | 7.93 |  |  |
| vae_decode | 0.9 | 7.94 |  |  |
| generation_2 | 15.5 | 7.93 |  | prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting., resolution=1024x1024, steps=4, quantize=None, seed=43 |
| save_2 | 0.1 | 7.93 | test_02_cosmic_wolf_new.png |  |
| total_wall_clock | 36.4 |  |  |  |

## Summary

| Metric | Value |
|--------|-------|
| Total time | 36.4 |
| Peak RAM | 7.94 |
| Prompts | 2 |

## Overview

|_ | Time (s) | Peak RSS (GiB) | Resolution | Steps | Seed | File |
|---|----------|----------------|------------|-------|------|------|
| 1 | 15.1 | 7.93 | 1024x1024 | 4 | 42 | test_01_airship_dragon_new.png |
| 2 | 15.5 | 7.93 | 1024x1024 | 4 | 43 | test_02_cosmic_wolf_new.png |
| — | 3.2 | — | — | — | text encode / decode |
| — | 2.7 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 36.4 |
| created_at | 2026-07-28T22:59:36.986371 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.94 |

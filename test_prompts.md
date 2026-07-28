## Log

======================================================================
  Mage-Flow MLX
======================================================================
  Phase                                          Time     Peak RAM
  ──────────────────────────────────────────────────────────────
  Importing Mage-Flow MLX pipeline from models/microsoft_Mage-Flow-Turbo...
  Current working directory: /Users/tilman/projects/Mage-Flow-mlx
  Python executable: /Users/tilman/projects/Mage-Flow-mlx/.venv/bin/python
  Imported MageFlowPipeline successfully
  Loaded 2 prompts from test_prompts.jsonl
Phase 1: Pre-encoding prompts (Qwen batch mode)
  Prompt 1/2: Cache HIT — skipping Qwen encode
  Prompt 2/2: Cache HIT — skipping Qwen encode
  Qwen unloaded (batch encoding complete)
Phase 2: Generating images (DiT + VAE)

## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0000 | 0.22 |  | |
| text_encoder_load | 0.0093 | 0.23 |  | |
tensors=713
| tokenizer_load | 2.1508 | 0.44 |  | |
| pipeline_load | 2.1814 | 0.44 |  | |
| text_encoder_unload | 0.0456 | 0.44 |  | |
| dit_load | 1.3840 | 7.73 |  | |
| vae_load | 0.0592 | 7.93 |  | |
tensors=728
| dit_step_1 | 3.1480 | 7.93 |  | |
| dit_step_2 | 3.0249 | 7.93 |  | |
| dit_step_3 | 3.1472 | 7.93 |  | |
| dit_step_4 | 3.0708 | 7.93 |  | |
| vae_decode | 0.4205 | 7.94 |  | |
| generation_1 | 12.8567 | 7.93 |  | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.0520 | 7.93 | test_01_airship_dragon_new.png | |
| dit_step_1 | 3.0128 | 7.93 |  | |
| dit_step_2 | 3.1973 | 7.93 |  | |
| dit_step_3 | 3.0566 | 7.93 |  | |
| dit_step_4 | 3.1273 | 7.93 |  | |
| vae_decode | 0.4627 | 7.94 |  | |
| generation_2 | 12.9043 | 7.93 |  | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0563 | 7.93 | test_02_cosmic_wolf_new.png | |
| total_wall_clock | 29.6219 |  |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*

## Summary

| Metric | Value |
|--------|-------|
| Total time | 29.6219 |
| Peak RAM | 7.94 |
| Prompts | 2 |

## Overview

| # | Time (s) | Peak RSS (GiB) | Resolution | Steps | File |
|---|----------|----------------|------------|-------|------|
| 1 | 12.8567 | 7.93 | 1024x1024 | 4 | test_01_airship_dragon_new.png |
| 2 | 12.9043 | 7.93 | 1024x1024 | 4 | test_02_cosmic_wolf_new.png |
| — | 1.4981 | — | — | — | text encode / decode |
| — | 2.3627 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 29.62187795783393 |
| created_at | 2026-07-28T20:34:11.262052 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.944723533466458 |

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
| python_startup | 0.0000 | 0.22 |  |  |
| text_encoder_load | 0.0092 | 0.23 |  | tensors=713 |
| tokenizer_load | 2.1917 | 0.44 |  |  |
| pipeline_load | 2.2256 | 0.44 |  |  |
| text_encoder_unload | 0.0470 | 0.44 |  |  |
| dit_load | 1.3632 | 8.11 |  |  |
| vae_load | 0.0597 | 8.29 |  | tensors=728 |
| dit_step_1 | 3.2055 | 7.93 |  |  |
| dit_step_2 | 3.1484 | 7.93 |  |  |
| dit_step_3 | 3.1159 | 7.93 |  |  |
| dit_step_4 | 3.2204 | 7.93 |  |  |
| vae_decode | 2.1112 | 7.94 |  |  |
| generation_1 | 14.8466 | 7.93 |  | prompt=black and white pencil drawing on rough paper in the style of monet, a steamp..., resolution=1024x1024, steps=4, quantize=None |
| save_1 | 0.0495 | 7.93 | test_01_airship_dragon_new.png |  |
| dit_step_1 | 3.0684 | 7.93 |  |  |
| dit_step_2 | 3.1683 | 7.93 |  |  |
| dit_step_3 | 3.2112 | 7.93 |  |  |
| dit_step_4 | 3.1155 | 7.93 |  |  |
| vae_decode | 0.4922 | 7.94 |  |  |
| generation_2 | 13.0971 | 7.93 |  | prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, ..., resolution=1024x1024, steps=4, quantize=None |
| save_2 | 0.0547 | 7.93 | test_02_cosmic_wolf_new.png |  |
| total_wall_clock | 31.8131 |  |  |  |

*Note: phase times are nested; child phases are subsets of parent phases.*

## Summary

| Metric | Value |
|--------|-------|
| Total time | 31.8131 |
| Peak RAM | 8.29 |
| Prompts | 2 |

## Overview

| # | Time (s) | Peak RSS (GiB) | Resolution | Steps | File |
|---|----------|----------------|------------|-------|------|
| 1 | 14.8466 | 7.93 | 1024x1024 | 4 | test_01_airship_dragon_new.png |
| 2 | 13.0971 | 7.93 | 1024x1024 | 4 | test_02_cosmic_wolf_new.png |
| — | 1.4790 | — | — | — | text encode / decode |
| — | 2.3904 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 31.813066124916077 |
| created_at | 2026-07-28T20:59:12.932176 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.289459228515625 |

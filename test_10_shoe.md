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

## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0000 | 0.22 |  |  |
| text_encoder_load | 0.0086 | 0.23 |  | tensors=713 |
| tokenizer_load | 2.1608 | 0.44 |  |  |
| pipeline_load | 2.1896 | 0.44 |  |  |
| text_encoder_unload | 0.0491 | 0.44 |  |  |
| dit_load | 1.8664 | 7.67 |  |  |
| vae_load | 0.0670 | 7.93 |  | tensors=728 |
| dit_step_1 | 3.2650 | 7.93 |  |  |
| dit_step_2 | 3.1626 | 7.93 |  |  |
| dit_step_3 | 3.2098 | 7.93 |  |  |
| dit_step_4 | 3.1515 | 7.93 |  |  |
| vae_decode | 2.3948 | 7.94 |  |  |
| generation | 17.2619 | 7.93 |  | prompt=An unbranded futuristic running shoe made from white technical mesh with a vi..., resolution=1024x1024, steps=4, quantize=None |
| save_png | 0.0767 | 7.93 | test_10_shoe.png |  |
| total_wall_clock | 19.5564 |  |  |  |

*Note: phase times are nested; child phases are subsets of parent phases.*

## Summary

| Metric | Value |
|--------|-------|
| Total time | 19.5564 |
| Peak RAM | 7.94 |
| Prompts | 1 |

## Overview

| # | Time (s) | Peak RSS (GiB) | Resolution | Steps | File |
|---|----------|----------------|------------|-------|------|
| 1 | 17.2619 | 7.94 | 1024x1024 | 4 | test_10_shoe.png |
| — | 2.2945 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 19.55641495785676 |
| created_at | 2026-07-28T20:57:49.282528 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.944555686786771 |

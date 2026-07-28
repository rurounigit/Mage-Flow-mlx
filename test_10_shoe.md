## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0000 | 0.22 |  | |
| text_encoder_load | 0.0092 | 0.23 |  | |
tensors=713
| tokenizer_load | 2.3030 | 0.44 |  | |
| pipeline_load | 2.3300 | 0.44 |  | |
| text_encoder_unload | 0.0433 | 0.44 |  | |
| dit_load | 2.1789 | 7.67 |  | |
| vae_load | 0.0674 | 7.93 |  | |
tensors=728
| dit_step_1 | 3.1708 | 7.93 |  | |
| dit_step_2 | 3.1886 | 7.93 |  | |
| dit_step_3 | 3.1403 | 7.93 |  | |
| dit_step_4 | 3.1887 | 7.93 |  | |
| vae_decode | 2.1585 | 7.94 |  | |
| generation | 17.2445 | 7.93 |  | |
prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.2150 | 7.93 | test_10_shoe.png | |
| total_wall_clock | 19.8159 |  |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*

## Summary

| Metric | Value |
|--------|-------|
| Total time | 19.8159 |
| Peak RAM | 7.94 |
| Prompts | 1 |

## Overview

| # | Time (s) | Peak RSS (GiB) | Resolution | Steps | File |
|---|----------|----------------|------------|-------|------|
| 1 | 17.2445 | 7.94 | 1024x1024 | 4 | test_10_shoe.png |
| — | 2.5715 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 19.81594058405608 |
| created_at | 2026-07-28T18:15:51.040489 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.944555686786771 |

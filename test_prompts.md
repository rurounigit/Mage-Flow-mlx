## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0000 | 0.22 |  | |
| text_encoder_load | 0.0095 | 0.23 |  | |
tensors=713
| tokenizer_load | 2.3371 | 0.44 |  | |
| pipeline_load | 2.3727 | 0.44 |  | |
| text_encoder_unload | 0.0450 | 0.44 |  | |
| dit_load | 1.3671 | 8.02 |  | |
| vae_load | 0.0600 | 7.95 |  | |
tensors=728
| dit_step_1 | 3.0593 | 7.93 |  | |
| dit_step_2 | 2.9961 | 7.93 |  | |
| dit_step_3 | 3.0770 | 7.93 |  | |
| dit_step_4 | 3.1644 | 7.93 |  | |
| vae_decode | 0.4243 | 7.94 |  | |
| generation_1 | 12.7628 | 7.93 |  | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.0493 | 7.93 | test_01_airship_dragon_new.png | |
| dit_step_1 | 3.1374 | 7.93 |  | |
| dit_step_2 | 3.1615 | 7.93 |  | |
| dit_step_3 | 3.1627 | 7.93 |  | |
| dit_step_4 | 3.1510 | 7.93 |  | |
| vae_decode | 0.4043 | 7.94 |  | |
| generation_2 | 13.0562 | 7.93 |  | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0659 | 7.93 | test_02_cosmic_wolf_new.png | |
| total_wall_clock | 29.8554 |  |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*

## Summary

| Metric | Value |
|--------|-------|
| Total time | 29.8554 |
| Peak RAM | 8.02 |
| Prompts | 2 |

## Overview

| # | Time (s) | Peak RSS (GiB) | Resolution | Steps | File |
|---|----------|----------------|------------|-------|------|
| 1 | 12.7628 | 7.93 | 1024x1024 | 4 | test_01_airship_dragon_new.png |
| 2 | 13.0562 | 7.93 | 1024x1024 | 4 | test_02_cosmic_wolf_new.png |
| — | 1.4816 | — | — | — | text encode / decode |
| — | 2.5549 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 29.85543595906347 |
| created_at | 2026-07-28T19:21:35.434646 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.020980834960938 |

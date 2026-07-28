## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 19.995710958959535 |
| created_at | 2026-07-28T16:22:38.315042 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.944372588768601 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0089 | 0.23 | |
tensors=713
| tokenizer_load | 2.2442 | 0.44 | |
| pipeline_load | 2.2728 | 0.44 | |
| text_encode | 1.4252 | 7.66 | |
cache=MISS
| text_encoder_unload | 0.0879 | 1.36 | |
| dit_load | 1.3343 | 7.67 | |
| vae_load | 0.0614 | 7.93 | |
tensors=728
| dit_step_1 | 3.1312 | 7.93 | |
| dit_step_2 | 3.1531 | 7.93 | |
| dit_step_3 | 3.0938 | 7.93 | |
| dit_step_4 | 3.1338 | 7.93 | |
| vae_decode | 2.1067 | 7.94 | |
| generation | 17.6196 | 7.93 | |
prompt=test
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0765 | 7.93 | |
| total_wall_clock | 19.9957 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
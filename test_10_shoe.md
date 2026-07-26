## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 20.068566250032745 |
| created_at | 2026-07-26T23:17:14.575347 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 15.41837185062468 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| dit_load | 1.4562 | 7.67 | |
| vae_load | 0.0870 | 7.93 | |
tensors=728
| text_encoder_load | 0.0091 | 7.93 | |
tensors=713
| tokenizer_load | 2.2475 | 7.93 | |
| pipeline_load | 3.8433 | 7.93 | |
| text_encode | 1.8224 | 15.42 | |
| text_encoder_unload | 0.2455 | 7.93 | |
| dit_step_1 | 3.8994 | 7.93 | |
| dit_step_2 | 3.1317 | 7.93 | |
| dit_step_3 | 3.1765 | 7.93 | |
| dit_step_4 | 3.1699 | 7.93 | |
| vae_decode | 0.4168 | 7.94 | |
| generation | 16.1339 | 7.93 | |
prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0622 | 7.93 | |
| total_wall_clock | 20.0686 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
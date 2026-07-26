## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 18.146768083097413 |
| created_at | 2026-07-26T22:43:02.106496 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 15.41837185062468 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| dit_load | 1.2966 | 7.89 | |
| vae_load | 0.0625 | 8.14 | |
tensors=728
| text_encoder_load | 0.0096 | 8.15 | |
tensors=713
| tokenizer_load | 2.0545 | 7.99 | |
| pipeline_load | 3.4618 | 7.99 | |
| text_encode | 1.4153 | 15.42 | |
| text_encoder_unload | 0.0856 | 7.93 | |
| dit_step_1 | 3.0631 | 7.93 | |
| dit_step_2 | 3.2031 | 7.93 | |
| dit_step_3 | 3.1949 | 7.93 | |
| dit_step_4 | 3.1552 | 7.93 | |
| vae_decode | 0.0014 | 7.93 | |
| generation | 14.5876 | 7.93 | |
prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0652 | 7.93 | |
| total_wall_clock | 18.1468 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
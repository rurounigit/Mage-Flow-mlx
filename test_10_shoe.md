## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 16.867392542073503 |
| created_at | 2026-07-27T15:16:48.654489 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.054428100585938 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| dit_load | 1.3122 | 7.89 | |
| vae_load | 0.0588 | 8.05 | |
tensors=728
| text_encoder_load | 0.0078 | 8.05 | |
tensors=713
| tokenizer_load | 2.1618 | 7.93 | |
| pipeline_load | 3.5824 | 7.93 | |
| text_encoder_unload | 0.0414 | 7.93 | |
| dit_step_1 | 3.1995 | 7.93 | |
| dit_step_2 | 3.1139 | 7.93 | |
| dit_step_3 | 3.1578 | 7.93 | |
| dit_step_4 | 3.1923 | 7.93 | |
| vae_decode | 0.4246 | 7.94 | |
| generation | 13.1870 | 7.93 | |
prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0662 | 7.93 | |
| total_wall_clock | 16.8674 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
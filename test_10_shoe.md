## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 18.04541991604492 |
| created_at | 2026-07-28T16:24:29.746133 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.312576293945312 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0086 | 0.23 | |
tensors=713
| tokenizer_load | 2.2163 | 0.44 | |
| pipeline_load | 2.2452 | 0.44 | |
| text_encoder_unload | 0.0462 | 0.44 | |
| dit_load | 1.3604 | 8.11 | |
| vae_load | 0.0584 | 8.31 | |
tensors=728
| dit_step_1 | 3.0288 | 7.93 | |
| dit_step_2 | 2.9924 | 7.93 | |
| dit_step_3 | 2.9813 | 7.93 | |
| dit_step_4 | 2.9612 | 7.93 | |
| vae_decode | 2.2068 | 7.94 | |
| generation | 15.7080 | 7.93 | |
prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0649 | 7.93 | |
| total_wall_clock | 18.0454 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
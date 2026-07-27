## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 16.231196416076273 |
| created_at | 2026-07-27T17:43:36.690365 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.187973022460938 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0086 | 0.23 | |
tensors=713
| tokenizer_load | 2.1134 | 0.44 | |
| pipeline_load | 2.1445 | 0.44 | |
| text_encoder_unload | 0.0423 | 0.44 | |
| dit_load | 1.3465 | 8.11 | |
| vae_load | 0.0606 | 8.19 | |
tensors=728
| dit_step_1 | 3.0411 | 7.93 | |
| dit_step_2 | 3.0001 | 7.93 | |
| dit_step_3 | 2.9948 | 7.93 | |
| dit_step_4 | 2.9807 | 7.93 | |
| vae_decode | 0.4296 | 7.94 | |
| generation | 13.9862 | 7.93 | |
prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0663 | 7.93 | |
| total_wall_clock | 16.2312 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 19.3895260830177 |
| created_at | 2026-07-28T13:58:38.334537 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.944555686786771 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0086 | 0.23 | |
tensors=713
| tokenizer_load | 2.2035 | 0.44 | |
| pipeline_load | 2.2329 | 0.44 | |
| text_encoder_unload | 0.0447 | 0.44 | |
| dit_load | 1.8658 | 7.67 | |
| vae_load | 0.0642 | 7.93 | |
tensors=728
| dit_step_1 | 3.1218 | 7.93 | |
| dit_step_2 | 3.1487 | 7.93 | |
| dit_step_3 | 3.2087 | 7.93 | |
| dit_step_4 | 3.2139 | 7.93 | |
| vae_decode | 2.1251 | 7.94 | |
| generation | 16.9375 | 7.93 | |
prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.1907 | 7.93 | |
| total_wall_clock | 19.3895 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
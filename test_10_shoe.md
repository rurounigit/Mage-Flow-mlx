## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 17.980781082995236 |
| created_at | 2026-07-26T19:49:22.697260 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.9434814453125 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| dit_load | 0.0021 | 0.22 | |
| vae_load | 0.0063 | 0.23 | |
tensors=728
| text_encoder_load | 0.0064 | 0.23 | |
tensors=713
| pipeline_load | 2.1434 | 0.44 | |
| text_encode | 1.3659 | 7.94 | |
| text_encoder_unload | 0.0538 | 7.94 | |
| dit_step_1 | 3.1246 | 7.94 | |
| dit_step_2 | 3.0960 | 7.94 | |
| dit_step_3 | 3.0362 | 7.94 | |
| dit_step_4 | 3.1951 | 7.94 | |
| vae_decode | 0.0009 | 7.94 | |
| generation | 15.7733 | 7.94 | |
prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0636 | 7.94 | |
| total_wall_clock | 17.9808 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
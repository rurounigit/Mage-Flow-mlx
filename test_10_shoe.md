## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 17.88152245793026 |
| created_at | 2026-07-27T16:56:35.725096 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.041351318359375 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0087 | 0.23 | |
tensors=713
| tokenizer_load | 2.0679 | 0.44 | |
| pipeline_load | 2.0923 | 0.44 | |
| text_encoder_unload | 0.0402 | 0.44 | |
| dit_load | 1.3187 | 8.04 | |
| vae_load | 0.0829 | 7.93 | |
tensors=728
| dit_step_1 | 3.2740 | 7.93 | |
| dit_step_2 | 3.0693 | 7.93 | |
| dit_step_3 | 3.2044 | 7.93 | |
| dit_step_4 | 3.1621 | 7.93 | |
| vae_decode | 1.4647 | 7.94 | |
| generation | 15.6956 | 7.93 | |
prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0670 | 7.93 | |
| total_wall_clock | 17.8815 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
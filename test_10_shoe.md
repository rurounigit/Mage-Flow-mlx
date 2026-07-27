## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 20.02885470900219 |
| created_at | 2026-07-27T16:13:35.434910 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.944555694237351 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0087 | 0.23 | |
tensors=713
| tokenizer_load | 1.9394 | 0.44 | |
| pipeline_load | 1.9636 | 0.44 | |
| text_encode | 1.3851 | 7.69 | |
cache=MISS
| text_encoder_unload | 0.0923 | 1.23 | |
| dit_load | 1.3525 | 7.67 | |
| vae_load | 0.1077 | 7.93 | |
tensors=728
| vae_load | 1.4922 | 7.93 | |
| dit_load | 1.4984 | 7.93 | |
| dit_step_1 | 3.2332 | 7.93 | |
| dit_step_2 | 3.1820 | 7.93 | |
| dit_step_3 | 3.2447 | 7.93 | |
| dit_step_4 | 3.2205 | 7.93 | |
| vae_decode | 2.0405 | 7.94 | |
| generation | 17.9711 | 7.93 | |
prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0696 | 7.93 | |
| total_wall_clock | 20.0289 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
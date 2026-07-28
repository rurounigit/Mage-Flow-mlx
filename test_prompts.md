## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 30.102915541967377 |
| created_at | 2026-07-28T14:00:24.939279 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.944723533466458 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0086 | 0.23 | |
tensors=713
| tokenizer_load | 2.2261 | 0.44 | |
| pipeline_load | 2.2589 | 0.44 | |
| text_encoder_unload | 0.0462 | 0.44 | |
| dit_load | 1.3628 | 7.67 | |
| vae_load | 0.0636 | 7.93 | |
tensors=728
| dit_step_1 | 3.1129 | 7.93 | |
| dit_step_2 | 3.1815 | 7.93 | |
| dit_step_3 | 3.0248 | 7.93 | |
| dit_step_4 | 3.1498 | 7.93 | |
| vae_decode | 0.4224 | 7.94 | |
| generation_1 | 12.9422 | 7.93 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.0535 | 7.93 | |
| dit_step_1 | 3.0787 | 7.93 | |
| dit_step_2 | 3.1394 | 7.93 | |
| dit_step_3 | 3.2109 | 7.93 | |
| dit_step_4 | 3.1647 | 7.93 | |
| vae_decode | 0.5811 | 7.94 | |
| generation_2 | 13.2269 | 7.93 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0599 | 7.93 | |
| total_wall_clock | 30.1029 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
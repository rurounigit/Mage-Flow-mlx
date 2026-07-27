## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 28.756967875058763 |
| created_at | 2026-07-27T17:44:47.142046 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.360031127929688 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0084 | 0.23 | |
tensors=713
| tokenizer_load | 2.1416 | 0.44 | |
| pipeline_reload | 2.1740 | 0.44 | |
| text_encoder_unload | 0.0441 | 0.44 | |
| dit_load | 1.3180 | 8.11 | |
| vae_load | 0.0616 | 8.36 | |
tensors=728
| dit_step_1 | 3.0313 | 7.93 | |
| dit_step_2 | 3.0476 | 7.93 | |
| dit_step_3 | 3.0295 | 7.93 | |
| dit_step_4 | 2.9460 | 7.93 | |
| vae_decode | 0.4299 | 7.94 | |
| generation_1 | 12.5374 | 7.93 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.0533 | 7.93 | |
| dit_step_1 | 3.0076 | 7.93 | |
| dit_step_2 | 3.0215 | 7.93 | |
| dit_step_3 | 3.0136 | 7.93 | |
| dit_step_4 | 2.9632 | 7.93 | |
| vae_decode | 0.4019 | 7.94 | |
| generation_2 | 12.4500 | 7.93 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0478 | 7.93 | |
| total_wall_clock | 28.7570 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
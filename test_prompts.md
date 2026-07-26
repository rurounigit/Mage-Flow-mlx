## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 36.843967083026655 |
| created_at | 2026-07-26T17:57:05.895373 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 6.7169189453125 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| dit_load | 0.0016 | 0.22 | |
| vae_load | 0.0068 | 0.23 | |
tensors=728
| text_encoder_load | 0.0064 | 0.23 | |
tensors=713
| pipeline_reload | 2.2755 | 0.44 | |
| text_encoder_unload | 0.0512 | 0.44 | |
| dit_step_1 | 3.0949 | 6.72 | |
| dit_step_2 | 3.1512 | 6.72 | |
| dit_step_3 | 3.1406 | 6.72 | |
| dit_step_4 | 3.1429 | 6.72 | |
| vae_decode | 0.0010 | 6.72 | |
| generation_1 | 14.5084 | 6.72 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk airship battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.0966 | 6.72 | |
| dit_step_1 | 8.6472 | 6.72 | |
| dit_step_2 | 3.2850 | 6.72 | |
| dit_step_3 | 3.2231 | 6.72 | |
| dit_step_4 | 3.2124 | 6.72 | |
| vae_decode | 0.0008 | 6.72 | |
| generation_2 | 19.8528 | 6.72 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, bioluminescent fur, star trails, ethereal purple and blue lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0550 | 6.72 | |
| total_wall_clock | 36.8440 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
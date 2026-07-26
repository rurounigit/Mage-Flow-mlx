## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 31.526261124992743 |
| created_at | 2026-07-26T20:39:49.487925 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 6.2744140625 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| dit_load | 0.0019 | 0.23 | |
| vae_load | 0.0068 | 0.23 | |
tensors=728
| text_encoder_load | 0.0086 | 0.23 | |
tensors=713
| pipeline_reload | 2.2030 | 0.44 | |
| text_encoder_unload | 0.0620 | 0.44 | |
| dit_step_1 | 3.1865 | 6.27 | |
| dit_step_2 | 3.0965 | 6.27 | |
| dit_step_3 | 3.1794 | 6.27 | |
| dit_step_4 | 3.1910 | 6.27 | |
| vae_decode | 0.0012 | 6.27 | |
| generation_1 | 14.6122 | 6.27 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.1752 | 6.27 | |
| dit_step_1 | 4.4254 | 6.27 | |
| dit_step_2 | 3.1777 | 6.27 | |
| dit_step_3 | 3.2111 | 6.27 | |
| dit_step_4 | 3.1816 | 6.27 | |
| vae_decode | 0.0007 | 6.27 | |
| generation_2 | 14.4145 | 6.27 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0524 | 6.27 | |
| total_wall_clock | 31.5263 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
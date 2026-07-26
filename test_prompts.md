## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 38.11718891595956 |
| created_at | 2026-07-26T17:37:58.314506 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 6.3945770263671875 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| dit_load | 0.0016 | 0.22 | |
| vae_load | 0.0064 | 0.23 | |
tensors=728
| text_encoder_load | 0.0063 | 0.23 | |
tensors=713
| pipeline_reload | 2.2625 | 0.44 | |
| text_encoder_unload | 0.0490 | 0.44 | |
| dit_step_1 | 3.1588 | 6.39 | |
| dit_step_2 | 2.9962 | 6.39 | |
| dit_step_3 | 3.0031 | 6.39 | |
| dit_step_4 | 2.9801 | 6.39 | |
| vae_decode | 0.0015 | 6.39 | |
| generation_1 | 13.8127 | 6.39 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk airship battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.1386 | 6.39 | |
| dit_step_1 | 10.1290 | 6.39 | |
| dit_step_2 | 3.1969 | 6.39 | |
| dit_step_3 | 3.2520 | 6.39 | |
| dit_step_4 | 3.2125 | 6.39 | |
| vae_decode | 0.0008 | 6.39 | |
| generation_2 | 21.7984 | 6.39 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, bioluminescent fur, star trails, ethereal purple and blue lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0510 | 6.39 | |
| total_wall_clock | 38.1172 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
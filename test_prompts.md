## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 39.38593124994077 |
| created_at | 2026-07-26T17:26:23.499835 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 5.9844207763671875 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| dit_load | 0.0016 | 0.22 | |
| vae_load | 0.0064 | 0.23 | |
tensors=728
| text_encoder_load | 0.0066 | 0.23 | |
tensors=713
| pipeline_reload | 2.0993 | 0.44 | |
| text_encoder_unload | 0.0479 | 0.44 | |
| dit_step_1 | 3.1667 | 5.98 | |
| dit_step_2 | 3.1540 | 5.98 | |
| dit_step_3 | 3.1716 | 5.98 | |
| dit_step_4 | 3.1747 | 5.98 | |
| vae_decode | 0.0011 | 5.98 | |
| generation_1 | 14.5487 | 5.98 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk airship battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.1508 | 5.98 | |
| dit_step_1 | 9.3176 | 5.98 | |
| dit_step_2 | 3.6279 | 5.98 | |
| dit_step_3 | 3.6253 | 5.98 | |
| dit_step_4 | 3.2990 | 5.98 | |
| vae_decode | 0.0008 | 5.98 | |
| generation_2 | 22.4799 | 5.98 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, bioluminescent fur, star trails, ethereal purple and blue lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0563 | 5.98 | |
| total_wall_clock | 39.3859 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 30.86596783308778 |
| created_at | 2026-07-26T13:23:10.912821 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 5.832366943359375 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| dit_load | 0.0017 | 0.22 | |
| vae_load | 0.0061 | 0.23 | |
| text_encoder_load | 0.0062 | 0.23 | |
| pipeline_reload | 2.3447 | 0.44 | |
| text_encoder_unload | 0.0543 | 0.44 | |
| dit_step_1 | 3.3579 | 5.83 | |
| dit_step_2 | 3.1558 | 5.83 | |
| dit_step_3 | 3.1925 | 5.83 | |
| dit_step_4 | 3.2157 | 5.83 | |
| vae_decode | 0.0013 | 5.83 | |
| generation_1 | 14.8165 | 5.83 | |
| | | | prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk airship battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting |
| | | | resolution=1024x1024 |
| | | | steps=4 |
| | | | quantize=None |
| save_1 | 0.2368 | 5.83 | |
| dit_step_1 | 3.1986 | 5.83 | |
| dit_step_2 | 3.2640 | 5.83 | |
| dit_step_3 | 3.2287 | 5.83 | |
| dit_step_4 | 3.2443 | 5.83 | |
| vae_decode | 0.0008 | 5.83 | |
| generation_2 | 13.3586 | 5.83 | |
| | | | prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, bioluminescent fur, star trails, ethereal purple and blue lighting |
| | | | resolution=1024x1024 |
| | | | steps=4 |
| | | | quantize=None |
| save_2 | 0.0507 | 5.83 | |
| total_wall_clock | 30.8660 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
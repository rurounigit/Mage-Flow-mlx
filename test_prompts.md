## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 31.73891074990388 |
| created_at | 2026-07-26T14:03:26.512610 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 5.9209747314453125 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| dit_load | 0.0019 | 0.22 | |
| vae_load | 0.0065 | 0.23 | |
| text_encoder_load | 0.0062 | 0.23 | |
| pipeline_reload | 2.3536 | 0.44 | |
| text_encoder_unload | 0.0625 | 0.44 | |
| dit_step_1 | 3.3144 | 5.92 | |
| dit_step_2 | 3.0519 | 5.92 | |
| dit_step_3 | 3.1714 | 5.92 | |
| dit_step_4 | 3.1238 | 5.92 | |
| vae_decode | 0.0048 | 5.92 | |
| generation_1 | 14.6603 | 5.92 | |
| prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk airship battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.2681 | 5.92 | |
| dit_step_1 | 4.2412 | 5.92 | |
| dit_step_2 | 3.2260 | 5.92 | |
| dit_step_3 | 3.2160 | 5.92 | |
| dit_step_4 | 3.2181 | 5.92 | |
| vae_decode | 0.0007 | 5.92 | |
| generation_2 | 14.3094 | 5.92 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, bioluminescent fur, star trails, ethereal purple and blue lighting
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0778 | 5.92 | |
| total_wall_clock | 31.7389 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
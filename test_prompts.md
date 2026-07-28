## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 30.86447141703684 |
| created_at | 2026-07-28T13:26:28.372131 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.370223999023438 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0087 | 0.23 | |
tensors=713
| tokenizer_load | 2.2978 | 0.44 | |
| pipeline_reload | 2.3310 | 0.44 | |
| text_encoder_unload | 0.0459 | 0.44 | |
| dit_load | 1.3203 | 8.11 | |
| vae_load | 0.0601 | 8.37 | |
tensors=728
| dit_step_1 | 3.2144 | 7.93 | |
| dit_step_2 | 3.1686 | 7.93 | |
| dit_step_3 | 3.1198 | 7.93 | |
| dit_step_4 | 3.1120 | 7.93 | |
| vae_decode | 0.4254 | 7.94 | |
| generation_1 | 13.0891 | 7.93 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.0511 | 7.93 | |
| dit_step_1 | 3.8979 | 7.93 | |
| dit_step_2 | 3.1056 | 7.93 | |
| dit_step_3 | 3.2169 | 7.93 | |
| dit_step_4 | 3.1161 | 7.93 | |
| vae_decode | 0.4503 | 7.94 | |
| generation_2 | 13.8284 | 7.93 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0598 | 7.93 | |
| total_wall_clock | 30.8645 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
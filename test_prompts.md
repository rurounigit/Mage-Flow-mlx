## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 32.731791708036326 |
| created_at | 2026-07-26T20:09:57.570662 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.70550537109375 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| dit_load | 0.0015 | 0.23 | |
| vae_load | 0.0063 | 0.23 | |
tensors=728
| text_encoder_load | 0.0066 | 0.23 | |
tensors=713
| pipeline_reload | 2.1853 | 0.44 | |
| text_encode_1 | 1.3655 | 7.70 | |
cache=MISS
| text_encode_2 | 0.1056 | 7.70 | |
cache=MISS
| text_encoder_unload | 0.0729 | 7.71 | |
| dit_step_1 | 3.1759 | 7.71 | |
| dit_step_2 | 3.0417 | 7.71 | |
| dit_step_3 | 3.2079 | 7.71 | |
| dit_step_4 | 3.1401 | 7.71 | |
| vae_decode | 0.0009 | 7.71 | |
| generation_1 | 14.4646 | 7.71 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.0625 | 7.71 | |
| dit_step_1 | 4.2481 | 7.71 | |
| dit_step_2 | 3.2153 | 7.71 | |
| dit_step_3 | 3.1937 | 7.71 | |
| dit_step_4 | 3.1879 | 7.71 | |
| vae_decode | 0.0008 | 7.71 | |
| generation_2 | 14.4139 | 7.71 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0583 | 7.71 | |
| total_wall_clock | 32.7318 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
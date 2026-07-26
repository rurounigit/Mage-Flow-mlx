## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 32.91365245799534 |
| created_at | 2026-07-26T22:42:22.012801 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 15.418539697304368 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| dit_load | 1.3180 | 7.89 | |
| vae_load | 0.0587 | 7.93 | |
tensors=728
| text_encoder_load | 0.0074 | 7.93 | |
tensors=713
| tokenizer_load | 2.1008 | 7.93 | |
| pipeline_reload | 3.5237 | 7.93 | |
| text_encode_1 | 1.8842 | 15.42 | |
cache=MISS
| text_encode_2 | 0.3696 | 15.42 | |
cache=MISS
| text_encoder_unload | 0.0840 | 7.93 | |
| dit_step_1 | 3.4323 | 7.93 | |
| dit_step_2 | 3.1641 | 7.93 | |
| dit_step_3 | 3.2225 | 7.93 | |
| dit_step_4 | 3.2099 | 7.93 | |
| vae_decode | 0.0036 | 7.93 | |
| generation_1 | 13.4861 | 7.93 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.0691 | 7.93 | |
| dit_step_1 | 3.2395 | 7.93 | |
| dit_step_2 | 3.1957 | 7.93 | |
| dit_step_3 | 3.1799 | 7.93 | |
| dit_step_4 | 3.2140 | 7.93 | |
| vae_decode | 0.0011 | 7.93 | |
| generation_2 | 13.2588 | 7.93 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0490 | 7.93 | |
| total_wall_clock | 32.9137 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
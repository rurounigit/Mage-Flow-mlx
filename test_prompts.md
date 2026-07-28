## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 31.0804558750242 |
| created_at | 2026-07-28T16:52:41.751099 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.9545135498046875 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0122 | 0.23 | |
tensors=713
| tokenizer_load | 2.2229 | 0.44 | |
| pipeline_load | 2.2562 | 0.44 | |
| text_encoder_unload | 0.0412 | 0.44 | |
| dit_load | 1.3612 | 7.95 | |
| vae_load | 0.0767 | 7.93 | |
tensors=728
| dit_step_1 | 3.2647 | 7.93 | |
| dit_step_2 | 3.0321 | 7.93 | |
| dit_step_3 | 3.1877 | 7.93 | |
| dit_step_4 | 3.1584 | 7.93 | |
| vae_decode | 0.9563 | 7.94 | |
| generation_1 | 13.9181 | 7.93 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.0547 | 7.93 | |
| dit_step_1 | 3.1661 | 7.93 | |
| dit_step_2 | 3.1096 | 7.93 | |
| dit_step_3 | 3.1034 | 7.93 | |
| dit_step_4 | 3.2995 | 7.93 | |
| vae_decode | 0.5221 | 7.94 | |
| generation_2 | 13.2395 | 7.93 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0523 | 7.93 | |
| total_wall_clock | 31.0805 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
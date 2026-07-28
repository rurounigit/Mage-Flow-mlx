## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 30.130097582936287 |
| created_at | 2026-07-28T13:52:20.082617 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.370712280273438 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0098 | 0.23 | |
tensors=713
| tokenizer_load | 2.2353 | 0.44 | |
| pipeline_load | 2.2620 | 0.44 | |
| text_encoder_unload | 0.0409 | 0.44 | |
| dit_load | 1.3309 | 8.11 | |
| vae_load | 0.0624 | 8.37 | |
tensors=728
| dit_step_1 | 3.0647 | 7.93 | |
| dit_step_2 | 3.1078 | 7.93 | |
| dit_step_3 | 3.2297 | 7.93 | |
| dit_step_4 | 3.1510 | 7.93 | |
| vae_decode | 0.4216 | 7.94 | |
| generation_1 | 13.0183 | 7.93 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.0492 | 7.93 | |
| dit_step_1 | 3.1854 | 7.93 | |
| dit_step_2 | 3.1555 | 7.93 | |
| dit_step_3 | 3.2389 | 7.93 | |
| dit_step_4 | 3.1635 | 7.93 | |
| vae_decode | 0.4520 | 7.94 | |
| generation_2 | 13.2420 | 7.93 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0571 | 7.93 | |
| total_wall_clock | 30.1301 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
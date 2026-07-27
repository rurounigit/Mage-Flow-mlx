## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 30.175262999953702 |
| created_at | 2026-07-27T17:05:18.178081 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.187469482421875 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0120 | 0.23 | |
tensors=713
| tokenizer_load | 2.2721 | 0.44 | |
| pipeline_reload | 2.3084 | 0.44 | |
| text_encoder_unload | 0.0560 | 0.44 | |
| dit_load | 1.3287 | 8.11 | |
| vae_load | 0.0582 | 8.19 | |
tensors=728
| dit_step_1 | 3.1482 | 7.93 | |
| dit_step_2 | 2.9752 | 7.93 | |
| dit_step_3 | 2.9845 | 7.93 | |
| dit_step_4 | 3.0109 | 7.93 | |
| vae_decode | 0.4381 | 7.94 | |
| generation_1 | 12.7484 | 7.93 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.0487 | 7.93 | |
| dit_step_1 | 3.6204 | 7.93 | |
| dit_step_2 | 3.0275 | 7.93 | |
| dit_step_3 | 3.1479 | 7.93 | |
| dit_step_4 | 3.0026 | 7.93 | |
| vae_decode | 0.6416 | 7.94 | |
| generation_2 | 13.4880 | 7.93 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0578 | 7.93 | |
| total_wall_clock | 30.1753 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
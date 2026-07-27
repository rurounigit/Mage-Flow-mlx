## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 29.39754858298693 |
| created_at | 2026-07-27T16:52:37.294671 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.160751342773438 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0093 | 0.23 | |
tensors=713
| tokenizer_load | 2.3561 | 0.44 | |
| pipeline_reload | 2.3893 | 0.44 | |
| text_encoder_unload | 0.0535 | 0.44 | |
| dit_load | 1.3377 | 8.10 | |
| vae_load | 0.0591 | 8.16 | |
tensors=728
| dit_step_1 | 3.0045 | 7.93 | |
| dit_step_2 | 2.9553 | 7.93 | |
| dit_step_3 | 2.9677 | 7.93 | |
| dit_step_4 | 2.9464 | 7.93 | |
| vae_decode | 0.4367 | 7.94 | |
| generation_1 | 12.3574 | 7.93 | |
prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk hairdryer battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.
resolution=1024x1024
steps=4
quantize=None
| save_1 | 0.0501 | 7.93 | |
| dit_step_1 | 3.3366 | 7.93 | |
| dit_step_2 | 3.0719 | 7.93 | |
| dit_step_3 | 3.0169 | 7.93 | |
| dit_step_4 | 2.9912 | 7.93 | |
| vae_decode | 0.5522 | 7.94 | |
| generation_2 | 13.0164 | 7.93 | |
prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, fur, star trails, ethereal purple and rainbow lighting.
resolution=1024x1024
steps=4
quantize=None
| save_2 | 0.0565 | 7.93 | |
| total_wall_clock | 29.3975 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
## Metadata

| Field | Value |
|-------|-------|
| model | Mage-Flow-Edit-Turbo |
| base_model | microsoft/Mage-Flow-Edit-Turbo |
| generation_time_seconds | 54.03205737506505 |
| created_at | 2026-07-26T22:45:11.389739 |
| image_path | test_10_shoe.png |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 16.137024026364088 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| pipeline_load | 4.4439 | 16.12 | |
| edit_step_1 | 9.6761 | 16.14 | |
| edit_step_2 | 6.5106 | 16.14 | |
| edit_step_3 | 6.3527 | 16.14 | |
| edit_step_4 | 6.5244 | 16.14 | |
| vae_decode | 0.5043 | 16.13 | |
| edit | 48.8738 | 16.12 | |
prompt=change the shoe to deep burgundy polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0588 | 16.12 | |
| total_wall_clock | 54.0321 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
## Metadata

| Field | Value |
|-------|-------|
| model | Mage-Flow-Edit-Turbo |
| base_model | microsoft/Mage-Flow-Edit-Turbo |
| generation_time_seconds | 52.69775204209145 |
| created_at | 2026-07-26T23:20:57.408630 |
| image_path | test_10_shoe.png |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 16.137024026364088 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| pipeline_load | 4.6079 | 16.12 | |
| edit_step_1 | 9.2270 | 16.14 | |
| edit_step_2 | 6.7637 | 16.14 | |
| edit_step_3 | 6.5878 | 16.14 | |
| edit_step_4 | 6.6121 | 16.14 | |
| vae_decode | 0.3705 | 16.13 | |
| edit | 47.3652 | 16.12 | |
prompt=change the shoe to deep burgundy polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0619 | 16.12 | |
| total_wall_clock | 52.6978 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
## Metadata

| Field | Value |
|-------|-------|
| model | Mage-Flow-Edit-Turbo |
| base_model | microsoft/Mage-Flow-Edit-Turbo |
| generation_time_seconds | 45.719846999971196 |
| created_at | 2026-07-27T17:42:37.652903 |
| image_path | test_10_shoe.png |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.26607609540224 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| pipeline_load | 2.6096 | 8.27 | |
| vae_load | 1.7832 | 7.93 | |
| dit_load | 1.8252 | 7.93 | |
| edit_step_1 | 7.5722 | 7.87 | |
| edit_step_2 | 7.4836 | 7.87 | |
| edit_step_3 | 7.5083 | 7.87 | |
| edit_step_4 | 7.0548 | 7.87 | |
| vae_decode | 1.7088 | 7.87 | |
| edit | 42.5662 | 7.86 | |
prompt=change the shoe to deep burgundy polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0709 | 7.86 | |
| total_wall_clock | 45.7198 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
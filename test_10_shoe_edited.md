## Metadata

| Field | Value |
|-------|-------|
| model | Mage-Flow-Edit-Turbo |
| base_model | microsoft/Mage-Flow-Edit-Turbo |
| generation_time_seconds | 43.61237750004511 |
| created_at | 2026-07-28T13:51:31.134294 |
| image_path | test_10_shoe.png |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.26607609540224 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| pipeline_load | 2.9559 | 8.27 | |
| dit_load | 1.4746 | 7.93 | |
| vae_load | 0.0000 | 7.93 | |
| edit_step_1 | 6.8492 | 7.87 | |
| edit_step_2 | 6.5109 | 7.87 | |
| edit_step_3 | 6.6004 | 7.87 | |
| edit_step_4 | 6.6734 | 7.87 | |
| vae_decode | 2.1873 | 7.87 | |
| edit | 39.8630 | 7.86 | |
prompt=change the shoe to deep burgundy polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0713 | 7.86 | |
| total_wall_clock | 43.6124 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
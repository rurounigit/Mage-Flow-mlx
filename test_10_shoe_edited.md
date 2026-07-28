## Metadata

| Field | Value |
|-------|-------|
| model | Mage-Flow-Edit-Turbo |
| base_model | microsoft/Mage-Flow-Edit-Turbo |
| generation_time_seconds | 38.3735010829987 |
| created_at | 2026-07-28T13:59:34.686430 |
| image_path | test_10_shoe.png |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.26607609540224 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| pipeline_load | 2.3280 | 8.27 | |
| dit_load | 1.4340 | 7.93 | |
| vae_load | 0.0000 | 7.93 | |
| edit_step_1 | 6.3353 | 7.87 | |
| edit_step_2 | 6.1406 | 7.87 | |
| edit_step_3 | 6.3973 | 7.87 | |
| edit_step_4 | 6.4724 | 7.87 | |
| vae_decode | 0.3518 | 7.87 | |
| edit | 35.5493 | 7.86 | |
prompt=change the shoe to deep burgundy polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0647 | 7.86 | |
| total_wall_clock | 38.3735 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
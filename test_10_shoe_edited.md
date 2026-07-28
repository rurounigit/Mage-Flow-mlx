## Metadata

| Field | Value |
|-------|-------|
| model | Mage-Flow-Edit-Turbo |
| base_model | microsoft/Mage-Flow-Edit-Turbo |
| generation_time_seconds | 38.5569124170579 |
| created_at | 2026-07-28T13:24:43.708926 |
| image_path | test_10_shoe.png |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.26607609540224 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| pipeline_load | 2.3424 | 8.27 | |
| dit_load | 1.4226 | 7.93 | |
| vae_load | 0.0000 | 7.93 | |
| edit_step_1 | 6.3784 | 7.87 | |
| edit_step_2 | 6.4520 | 7.87 | |
| edit_step_3 | 6.2582 | 7.87 | |
| edit_step_4 | 6.3588 | 7.87 | |
| vae_decode | 0.3603 | 7.87 | |
| edit | 35.6857 | 7.86 | |
prompt=change the shoe to deep burgundy polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0626 | 7.86 | |
| total_wall_clock | 38.5569 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
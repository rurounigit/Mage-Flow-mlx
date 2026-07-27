## Metadata

| Field | Value |
|-------|-------|
| model | Mage-Flow-Edit-Turbo |
| base_model | microsoft/Mage-Flow-Edit-Turbo |
| generation_time_seconds | 49.16183404205367 |
| created_at | 2026-07-27T16:16:11.850124 |
| image_path | test_10_shoe.png |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 16.120695484802127 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| pipeline_load | 4.6605 | 16.12 | |
| edit_step_1 | 7.0683 | 7.87 | |
| edit_step_2 | 6.1394 | 7.87 | |
| edit_step_3 | 6.2241 | 7.87 | |
| edit_step_4 | 6.4007 | 7.87 | |
| vae_decode | 0.3473 | 7.87 | |
| edit | 43.9506 | 7.86 | |
prompt=change the shoe to deep burgundy polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0691 | 7.86 | |
| total_wall_clock | 49.1618 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
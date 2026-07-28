## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0000 | 0.22 |  | |
| pipeline_load | 2.7341 | 8.27 |  | |
| dit_load | 1.4420 | 7.93 |  | |
| vae_load | 0.0000 | 7.93 |  | |
| edit_step_1 | 6.5111 | 7.87 |  | |
| edit_step_2 | 6.2447 | 7.87 |  | |
| edit_step_3 | 6.2254 | 7.87 |  | |
| edit_step_4 | 6.3351 | 7.87 |  | |
| vae_decode | 1.8007 | 7.87 |  | |
| edit | 37.3175 | 7.86 |  | |
prompt=change the shoe to deep burgundy polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.0614 | 7.86 | test_10_shoe_edited.png | |
| total_wall_clock | 40.5948 |  |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*

## Summary

| Metric | Value |
|--------|-------|
| Total time | 40.5948 |
| Peak RAM | 8.27 |
| Prompts | 1 |

## Overview

| # | Time (s) | Peak RSS (GiB) | Resolution | Steps | File |
|---|----------|----------------|------------|-------|------|
| 1 | 37.3175 | 7.87 | 1024x1024 | 4 | test_10_shoe_edited.png |
| — | 3.2773 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | Mage-Flow-Edit-Turbo |
| base_model | microsoft/Mage-Flow-Edit-Turbo |
| generation_time_seconds | 40.59479712508619 |
| created_at | 2026-07-28T19:20:34.993333 |
| image_path | test_10_shoe.png |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.26607609540224 |

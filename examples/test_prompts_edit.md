## Log

======================================================================
  Mage-Flow MLX
======================================================================
  Importing Mage-Flow MLX pipeline from models/microsoft_Mage-Flow-Edit-Turbo...
  Current working directory: /Users/tilman/projects/Mage-Flow-mlx
  Python executable: /Users/tilman/projects/Mage-Flow-mlx/.venv/bin/python
  Imported MageFlowPipeline successfully
  Loaded 10 edit prompts from test_prompts_edit.jsonl

## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0 | 0.22 |  |  |
| pipeline_load | 3.3 | 8.27 |  |  |
| text_encode_1 | 0.5 | 8.27 |  | cache=MISS |
| text_encode_2 | 0.3 | 8.27 |  | cache=MISS |
| text_encode_3 | 0.3 | 8.27 |  | cache=MISS |
| text_encode_4 | 0.3 | 8.27 |  | cache=MISS |
| text_encode_5 | 0.3 | 8.27 |  | cache=MISS |
| text_encode_6 | 0.3 | 8.27 |  | cache=MISS |
| text_encode_7 | 0.3 | 8.27 |  | cache=MISS |
| text_encode_8 | 0.3 | 8.27 |  | cache=MISS |
| text_encode_9 | 0.3 | 8.27 |  | cache=MISS |
| text_encode_10 | 0.3 | 8.28 |  | cache=MISS |
| text_encoder_unload | 0.1 | 1.61 |  |  |
| dit_load | 1.5 | 7.93 |  |  |
| vae_load | 0.0 | 7.93 |  |  |
| vae_encode_ref_1 | 0.2 | 7.87 |  | cache=MISS |
| edit_step_1 | 3.7 | 7.87 |  |  |
| edit_step_2 | 3.5 | 7.87 |  |  |
| edit_step_3 | 3.6 | 7.87 |  |  |
| edit_step_4 | 3.6 | 7.87 |  |  |
| vae_decode | 0.3 | 7.87 |  |  |
| generation_1 | 15.0 | 7.87 |  | prompt=change the shoe to metallic green mesh with a translucent  sole and subtle fur details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=42 |
| save_1 | 0.2 | 7.87 | output/test_10_shoe_edited_worker_01.png |  |
| vae_encode_ref_2 | 0.1 | 7.87 |  | cache=MISS |
| edit_step_1 | 3.6 | 7.88 |  |  |
| edit_step_2 | 3.6 | 7.88 |  |  |
| edit_step_3 | 3.6 | 7.88 |  |  |
| edit_step_4 | 3.6 | 7.88 |  |  |
| vae_decode | 0.3 | 7.87 |  |  |
| generation_2 | 15.1 | 7.87 |  | prompt=change the shoe to deep blue polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=43 |
| save_2 | 0.2 | 7.87 | output/test_10_shoe_edited_worker_02.png |  |
| vae_encode_ref_3 | 0.1 | 7.87 |  | cache=MISS |
| edit_step_1 | 3.6 | 7.88 |  |  |
| edit_step_2 | 3.7 | 7.88 |  |  |
| edit_step_3 | 3.6 | 7.88 |  |  |
| edit_step_4 | 3.7 | 7.88 |  |  |
| vae_decode | 0.3 | 7.87 |  |  |
| generation_3 | 15.2 | 7.87 |  | prompt=change the shoe to a matte black finish with a red-orange translucent sole and small green accents. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=44 |
| save_3 | 0.2 | 7.87 | output/test_10_shoe_edited_worker_03.png |  |
| vae_encode_ref_4 | 0.1 | 7.87 |  | cache=MISS |
| edit_step_1 | 3.7 | 7.88 |  |  |
| edit_step_2 | 3.8 | 7.88 |  |  |
| edit_step_3 | 3.8 | 7.88 |  |  |
| edit_step_4 | 3.9 | 7.88 |  |  |
| vae_decode | 0.3 | 7.87 |  |  |
| generation_4 | 15.7 | 7.87 |  | prompt=change the shoe to beige canvas with a solid white sole and navy blue stitching. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=45 |
| save_4 | 0.2 | 7.87 | output/test_10_shoe_edited_worker_04.png |  |
| vae_encode_ref_5 | 0.1 | 7.87 |  | cache=MISS |
| edit_step_1 | 3.7 | 7.88 |  |  |
| edit_step_2 | 3.9 | 7.88 |  |  |
| edit_step_3 | 3.9 | 7.88 |  |  |
| edit_step_4 | 3.9 | 7.88 |  |  |
| vae_decode | 0.4 | 7.87 |  |  |
| generation_5 | 16.0 | 7.87 |  | prompt=change the shoe to full gray suede with an off-white sole and silver eyelets. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=46 |
| save_5 | 0.2 | 7.87 | output/test_10_shoe_edited_worker_05.png |  |
| vae_encode_ref_6 | 0.1 | 7.87 |  | cache=MISS |
| edit_step_1 | 3.9 | 7.88 |  |  |
| edit_step_2 | 4.0 | 7.88 |  |  |
| edit_step_3 | 4.0 | 7.88 |  |  |
| edit_step_4 | 4.1 | 7.88 |  |  |
| vae_decode | 0.4 | 7.87 |  |  |
| generation_6 | 16.6 | 7.87 |  | prompt=change the shoe to iridescent fabric with a glowing blue translucent sole and purple laces. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=47 |
| save_6 | 0.2 | 7.87 | output/test_10_shoe_edited_worker_06.png |  |
| vae_encode_ref_7 | 0.1 | 7.87 |  | cache=MISS |
| edit_step_1 | 4.0 | 7.88 |  |  |
| edit_step_2 | 4.1 | 7.88 |  |  |
| edit_step_3 | 4.1 | 7.88 |  |  |
| edit_step_4 | 4.2 | 7.88 |  |  |
| vae_decode | 0.4 | 7.87 |  |  |
| generation_7 | 17.0 | 7.87 |  | prompt=change the shoe to reflective silver material with a clear sole and bright yellow pull tabs. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=48 |
| save_7 | 0.2 | 7.87 | output/test_10_shoe_edited_worker_07.png |  |
| vae_encode_ref_8 | 0.1 | 7.87 |  | cache=MISS |
| edit_step_1 | 4.3 | 7.88 |  |  |
| edit_step_2 | 4.5 | 7.88 |  |  |
| edit_step_3 | 4.6 | 7.88 |  |  |
| edit_step_4 | 4.7 | 7.88 |  |  |
| vae_decode | 0.4 | 7.87 |  |  |
| generation_8 | 18.7 | 7.87 |  | prompt=change the shoe to light pink woven textile with a translucent pink sole and white embroidered details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=49 |
| save_8 | 0.2 | 7.87 | output/test_10_shoe_edited_worker_08.png |  |
| vae_encode_ref_9 | 0.1 | 7.87 |  | cache=MISS |
| edit_step_1 | 4.8 | 7.88 |  |  |
| edit_step_2 | 5.0 | 7.88 |  |  |
| edit_step_3 | 5.2 | 7.88 |  |  |
| edit_step_4 | 5.3 | 7.88 |  |  |
| vae_decode | 0.5 | 7.87 |  |  |
| generation_9 | 21.0 | 7.87 |  | prompt=change the shoe to black knitted mesh with a black sole and a reflective silver stripe. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=50 |
| save_9 | 0.2 | 7.87 | output/test_10_shoe_edited_worker_09.png |  |
| vae_encode_ref_10 | 0.2 | 7.87 |  | cache=MISS |
| edit_step_1 | 5.4 | 7.88 |  |  |
| edit_step_2 | 5.6 | 7.88 |  |  |
| edit_step_3 | 5.7 | 7.88 |  |  |
| edit_step_4 | 5.6 | 7.88 |  |  |
| vae_decode | 0.5 | 7.87 |  |  |
| generation_10 | 23.1 | 7.87 |  | prompt=change the shoe to olive green nylon with a speckled gum sole and orange lace accents. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background., resolution=1024x1024, steps=4, quantize=None, seed=51 |
| save_10 | 0.2 | 7.87 | output/test_10_shoe_edited_worker_10.png |  |
| total_wall_clock | 184.3 |  |  |  |

## Summary

| Metric | Value |
|--------|-------|
| Total time | 184.3 |
| Peak RAM | 8.28 |
| Prompts | 10 |

## Overview

|_ | Time (s) | Peak RSS (GiB) | Resolution | Steps | Seed | File |
|---|----------|----------------|------------|-------|------|------|
| 1 | 15.0 | 7.87 | 1024x1024 | 4 | 42 | output/test_10_shoe_edited_worker_01.png |
| 2 | 15.1 | 7.87 | 1024x1024 | 4 | 43 | output/test_10_shoe_edited_worker_02.png |
| 3 | 15.2 | 7.87 | 1024x1024 | 4 | 44 | output/test_10_shoe_edited_worker_03.png |
| 4 | 15.7 | 7.87 | 1024x1024 | 4 | 45 | output/test_10_shoe_edited_worker_04.png |
| 5 | 16.0 | 7.87 | 1024x1024 | 4 | 46 | output/test_10_shoe_edited_worker_05.png |
| 6 | 16.6 | 7.87 | 1024x1024 | 4 | 47 | output/test_10_shoe_edited_worker_06.png |
| 7 | 17.0 | 7.87 | 1024x1024 | 4 | 48 | output/test_10_shoe_edited_worker_07.png |
| 8 | 18.7 | 7.87 | 1024x1024 | 4 | 49 | output/test_10_shoe_edited_worker_08.png |
| 9 | 21.0 | 7.87 | 1024x1024 | 4 | 50 | output/test_10_shoe_edited_worker_09.png |
| 10 | 23.1 | 7.87 | 1024x1024 | 4 | 51 | output/test_10_shoe_edited_worker_10.png |
| — | 6.4 | — | — | — | text encode / decode |
| — | 4.6 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Edit-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 184.3 |
| created_at | 2026-07-30T12:33:17.665806 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 8.28 |

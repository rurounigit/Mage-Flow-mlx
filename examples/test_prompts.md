## Log

======================================================================
  Mage-Flow MLX
======================================================================
  Importing Mage-Flow MLX pipeline from models/microsoft_Mage-Flow-Turbo...
  Current working directory: /Users/tilman/projects/Mage-Flow-mlx
  Python executable: /Users/tilman/projects/Mage-Flow-mlx/.venv/bin/python
  Imported MageFlowPipeline successfully
  Loaded 10 prompts from test_prompts.jsonl

## Phases

| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |
|-------|----------|----------------|------------|----------|
| python_startup | 0.0 | 0.22 |  |  |
| text_encoder_load | 0.0 | 0.24 |  | tensors=713 |
| tokenizer_load | 1.9 | 0.45 |  |  |
| pipeline_load | 1.9 | 0.45 |  |  |
| text_encode_1 | 1.4 | 7.95 |  | cache=MISS |
| text_encode_2 | 0.1 | 7.95 |  | cache=MISS |
| text_encode_3 | 0.1 | 7.95 |  | cache=MISS |
| text_encode_4 | 0.1 | 7.95 |  | cache=MISS |
| text_encode_5 | 0.1 | 7.95 |  | cache=MISS |
| text_encode_6 | 0.1 | 7.95 |  | cache=MISS |
| text_encode_7 | 0.1 | 7.95 |  | cache=MISS |
| text_encode_8 | 0.1 | 7.95 |  | cache=MISS |
| text_encode_9 | 0.1 | 7.95 |  | cache=MISS |
| text_encode_10 | 0.1 | 7.95 |  | cache=MISS |
| text_encoder_unload | 0.1 | 2.20 |  |  |
| dit_load | 1.3 | 7.67 |  |  |
| vae_load | 0.1 | 7.93 |  | tensors=728 |
| dit_step_1 | 3.1 | 7.93 |  |  |
| dit_step_2 | 3.2 | 7.93 |  |  |
| dit_step_3 | 3.2 | 7.93 |  |  |
| dit_step_4 | 3.2 | 7.93 |  |  |
| vae_decode | 0.4 | 7.95 |  |  |
| generation_1 | 13.2 | 7.93 |  | prompt=black and white pencil drawing on rough paper in the style of monet, a steampunk airship battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting, resolution=1024x1024, steps=4, quantize=None, seed=42 |
| save_1 | 0.1 | 7.93 | output/test_01_airship_dragon_new.png |  |
| dit_step_1 | 3.2 | 7.93 |  |  |
| dit_step_2 | 3.2 | 7.93 |  |  |
| dit_step_3 | 3.2 | 7.93 |  |  |
| dit_step_4 | 3.2 | 7.93 |  |  |
| vae_decode | 0.4 | 7.95 |  |  |
| generation_2 | 13.2 | 7.93 |  | prompt=Anime shonen style, an astronaut riding a cosmic wolf through a nebula, bioluminescent fur, star trails, ethereal purple and blue lighting, resolution=1024x1024, steps=4, quantize=None, seed=43 |
| save_2 | 0.0 | 7.93 | output/test_02_cosmic_wolf_new.png |  |
| dit_step_1 | 3.2 | 7.93 |  |  |
| dit_step_2 | 3.2 | 7.93 |  |  |
| dit_step_3 | 3.2 | 7.93 |  |  |
| dit_step_4 | 3.2 | 7.93 |  |  |
| vae_decode | 0.4 | 7.95 |  |  |
| generation_3 | 13.3 | 7.93 |  | prompt=90s selfie grunge vice a portrait of a punk girl sitting on an old sofa, resolution=1024x1024, steps=4, quantize=None, seed=100 |
| save_3 | 0.0 | 7.93 | output/test_03_girl_new.png |  |
| dit_step_1 | 3.2 | 7.93 |  |  |
| dit_step_2 | 3.2 | 7.93 |  |  |
| dit_step_3 | 3.2 | 7.93 |  |  |
| dit_step_4 | 3.2 | 7.93 |  |  |
| vae_decode | 0.4 | 7.95 |  |  |
| generation_4 | 13.3 | 7.93 |  | prompt=A school of cybernetic koi fish swimming through glowing circuit boards, neon green and electric blue, glitch art aesthetic, ultra sharp focus, resolution=1024x1024, steps=4, quantize=None, seed=45 |
| save_4 | 0.1 | 7.93 | output/test_04_circuit_koi_new.png |  |
| dit_step_1 | 3.2 | 7.93 |  |  |
| dit_step_2 | 3.2 | 7.93 |  |  |
| dit_step_3 | 3.2 | 7.93 |  |  |
| dit_step_4 | 3.2 | 7.93 |  |  |
| vae_decode | 0.4 | 7.95 |  |  |
| generation_5 | 13.2 | 7.93 |  | prompt=A library that exists inside a giant tree, floating books, magical glow, fantasy realism, resolution=1024x1024, steps=4, quantize=None, seed=2893 |
| save_5 | 0.1 | 7.93 | output/test_05_tree_library_new.png |  |
| dit_step_1 | 3.2 | 7.93 |  |  |
| dit_step_2 | 3.1 | 7.93 |  |  |
| dit_step_3 | 3.2 | 7.93 |  |  |
| dit_step_4 | 3.2 | 7.93 |  |  |
| vae_decode | 0.6 | 7.95 |  |  |
| generation_6 | 13.4 | 7.93 |  | prompt=A retro-futuristic diner on Mars, neon signs, red planet landscape visible through windows, 1950s sci-fi movie poster style, the name of the diner is 'mage', resolution=1024x1024, steps=4, quantize=None, seed=47 |
| save_6 | 0.1 | 7.93 | output/test_06_mars_diner_new.png |  |
| dit_step_1 | 4.8 | 7.93 |  |  |
| dit_step_2 | 3.2 | 7.93 |  |  |
| dit_step_3 | 3.3 | 7.93 |  |  |
| dit_step_4 | 3.2 | 7.93 |  |  |
| vae_decode | 0.6 | 7.95 |  |  |
| generation_7 | 15.2 | 7.93 |  | prompt=A close-up portrait of an elderly African man with deep wrinkles, wearing a traditional hat, soft natural lighting, ultra realistic., resolution=1024x1024, steps=4, quantize=None, seed=42 |
| save_7 | 0.1 | 7.93 | output/test_07_old_man_new.png |  |
| dit_step_1 | 3.4 | 7.93 |  |  |
| dit_step_2 | 3.4 | 7.93 |  |  |
| dit_step_3 | 3.5 | 7.93 |  |  |
| dit_step_4 | 3.6 | 7.93 |  |  |
| vae_decode | 0.6 | 7.95 |  |  |
| generation_8 | 14.7 | 7.93 |  | prompt=Create a lifelike portrait of a man in his 30s with a strong jawline and dark, short hair with dramatic side lighting and high contrast in an editorial photo style., resolution=1024x1024, steps=4, quantize=None, seed=1 |
| save_8 | 0.1 | 7.93 | output/test_08_man_new.png |  |
| dit_step_1 | 4.1 | 7.93 |  |  |
| dit_step_2 | 4.2 | 7.93 |  |  |
| dit_step_3 | 4.3 | 7.93 |  |  |
| dit_step_4 | 4.6 | 7.93 |  |  |
| vae_decode | 1.1 | 7.95 |  |  |
| generation_9 | 18.3 | 7.93 |  | prompt=text in a bold simple font without serifs with a red background: 'Democracy will not come Today, this year Nor ever Through compromise and fear. I have as much right As the other fellow has To stand On my two feet.', resolution=1024x1024, steps=4, quantize=None, seed=343637 |
| save_9 | 0.1 | 7.93 | output/test_09_text_new.png |  |
| dit_step_1 | 4.4 | 7.93 |  |  |
| dit_step_2 | 4.4 | 7.93 |  |  |
| dit_step_3 | 4.8 | 7.93 |  |  |
| dit_step_4 | 4.6 | 7.93 |  |  |
| vae_decode | 0.8 | 7.95 |  |  |
| generation_10 | 19.1 | 7.93 |  | prompt=extreme close up of a hand, visible pores and hair, shallow depth of field, resolution=1024x1024, steps=4, quantize=None, seed=42 |
| save_10 | 0.1 | 7.93 | output/test_blurp_text_new.png |  |
| total_wall_clock | 153.7 |  |  |  |

## Summary

| Metric | Value |
|--------|-------|
| Total time | 153.7 |
| Peak RAM | 7.95 |
| Prompts | 10 |

## Overview

|_ | Time (s) | Peak RSS (GiB) | Resolution | Steps | Seed | File |
|---|----------|----------------|------------|-------|------|------|
| 1 | 13.2 | 7.93 | 1024x1024 | 4 | 42 | output/test_01_airship_dragon_new.png |
| 2 | 13.2 | 7.93 | 1024x1024 | 4 | 43 | output/test_02_cosmic_wolf_new.png |
| 3 | 13.3 | 7.93 | 1024x1024 | 4 | 100 | output/test_03_girl_new.png |
| 4 | 13.3 | 7.93 | 1024x1024 | 4 | 45 | output/test_04_circuit_koi_new.png |
| 5 | 13.2 | 7.93 | 1024x1024 | 4 | 2893 | output/test_05_tree_library_new.png |
| 6 | 13.4 | 7.93 | 1024x1024 | 4 | 47 | output/test_06_mars_diner_new.png |
| 7 | 15.2 | 7.93 | 1024x1024 | 4 | 42 | output/test_07_old_man_new.png |
| 8 | 14.7 | 7.93 | 1024x1024 | 4 | 1 | output/test_08_man_new.png |
| 9 | 18.3 | 7.93 | 1024x1024 | 4 | 343637 | output/test_09_text_new.png |
| 10 | 19.1 | 7.93 | 1024x1024 | 4 | 42 | output/test_blurp_text_new.png |
| — | 3.7 | — | — | — | text encode / decode |
| — | 3.2 | — | — | — | overhead (load + encode + decode) |

## Run Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 153.7 |
| created_at | 2026-07-30T12:10:02.248789 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.95 |

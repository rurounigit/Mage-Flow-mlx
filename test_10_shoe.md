## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 22.043026541010477 |
| created_at | 2026-07-28T13:23:27.217900 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.944555686786771 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0122 | 0.23 | |
tensors=713
| tokenizer_load | 2.1373 | 0.44 | |
| pipeline_load | 2.1737 | 0.44 | |
| text_encoder_unload | 0.0444 | 0.44 | |
| dit_load | 2.3034 | 7.67 | |
| vae_load | 0.0743 | 7.93 | |
tensors=728
| dit_step_1 | 5.5304 | 7.93 | |
| dit_step_2 | 3.0604 | 7.93 | |
| dit_step_3 | 3.2256 | 7.93 | |
| dit_step_4 | 3.2354 | 7.93 | |
| vae_decode | 2.0131 | 7.94 | |
| generation | 19.6146 | 7.93 | |
prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.2245 | 7.93 | |
| total_wall_clock | 22.0430 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
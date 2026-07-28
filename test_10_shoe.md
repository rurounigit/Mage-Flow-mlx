## Metadata

| Field | Value |
|-------|-------|
| model | microsoft/Mage-Flow-Turbo |
| base_model | MageFlow |
| generation_time_seconds | 21.65900716604665 |
| created_at | 2026-07-28T13:50:20.189840 |
| image_path | None |
| image_paths | None |
| image_strength | None |
| peak_memory_gib | 7.944555686786771 |

## Mage-Flow MLX Profiler

| Phase | Time (s) | Peak RSS (GiB) | Metadata |
|-------|----------|----------------|----------|
| python_startup | 0.0000 | 0.22 | |
| text_encoder_load | 0.0090 | 0.23 | |
tensors=713
| tokenizer_load | 2.0654 | 0.44 | |
| pipeline_load | 2.0914 | 0.44 | |
| text_encoder_unload | 0.0489 | 0.44 | |
| dit_load | 2.4143 | 7.67 | |
| vae_load | 0.1132 | 7.93 | |
tensors=728
| dit_step_1 | 3.6862 | 7.93 | |
| dit_step_2 | 3.1507 | 7.93 | |
| dit_step_3 | 3.2341 | 7.93 | |
| dit_step_4 | 3.2001 | 7.93 | |
| vae_decode | 2.5459 | 7.94 | |
| generation | 18.6296 | 7.93 | |
prompt=An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.
resolution=1024x1024
steps=4
quantize=None
| save_png | 0.9071 | 7.93 | |
| total_wall_clock | 21.6590 |  | |

*Note: phase times are nested; child phases are subsets of parent phases.*
# Edit Worker Implementation

## Overview

Implemented a new `--edit` flag for the worker mode that enables image editing using the
Mage-Flow-Edit-Turbo model. The edit worker mirrors the regular worker's architecture but
with a different model loading strategy and reference image handling.

## Key Design Decisions

### 1. Model Loading Strategy (Lazy Load)

The edit worker uses a two-phase loading strategy to minimize peak RAM:

- **Phase 1 (Text Encoding):** Load text encoder (Qwen3-VL) + tokenizer, encode all prompts,
  then unload Qwen. This takes ~0.4 GiB.
- **Phase 1.5 (DiT+VAE Load):** After Qwen is unloaded, load DiT + VAE (~7.9 GiB).
- **Phase 2 (Generation):** Run the edit denoising loop, decode, and save images.

This reduces peak RAM from Qwen + DiT + VAE (~15.4 GiB) to max(Qwen, DiT+VAE) (~13.9 GiB).

### 2. Shared Text Encoder

The edit model (`microsoft_Mage-Flow-Edit-Turbo`) does not include text encoder weights.
Instead, it shares the text encoder from `models/shared/mage_flow_qwen3vl/text_encoder.safetensors`.
The worker loads the text encoder from this shared path and passes it to `MageFlowEdit`.

### 3. Tokenizer Loading

The edit model does not include tokenizer files locally. The worker loads the tokenizer
from `Qwen/Qwen3-VL-8B-Instruct` (cached from the regular worker) using
`AutoTokenizer.from_pretrained()` and wraps it in `MageFlowTokenizer`.

### 4. VAE Weight Key Mapping

The converted VAE weights use already-MLX-native key prefixes (`decoder_model.`, `encoder.`,
`dconv_encoder.`) and layout. Two fixes were applied:

- `transform_vae_key`: Added passthrough for already-converted keys starting with
  `decoder_model.`, `encoder.`, or `dconv_encoder.`.
- `transform_vae_weight`: Made a no-op since VAE weights are already in MLX-native layout
  `(out_ch, kH, in_ch, kW)`. The previous transpose `(0, 2, 3, 1)` corrupted already-converted
  weights.

### 5. Text Encoder Skip in `load_dit_vae`

`MageFlowEdit.load_dit_vae()` was modified to skip the `text_encoder` component during
weight loading when `self.text_encoder` is already loaded (shared from base model).
This prevents `FileNotFoundError` when the edit model directory lacks text_encoder weights.

### 6. Missing/Malformed Image Path Handling

The edit worker validates all image paths before processing:

- **Missing path:** If the image file does not exist, the prompt is skipped with a warning.
- **Malformed image:** If the image cannot be opened/verified by PIL, the prompt is skipped
  with a warning.
- **Missing field:** If neither `image` nor `ref_images` field is present, the prompt is
  skipped with a warning.

This ensures the worker continues processing remaining prompts even if some have invalid
image references.

## CLI Usage

```bash
# Edit worker with metadata output (JSON + MD)
python generate.py --worker prompts.jsonl --edit --metadata

# Edit worker without metadata (terminal output only)
python generate.py --worker prompts.jsonl --edit
```

## JSONL Format

```jsonl
{"prompt": "change the shoe to metallic green", "image": "test_10_shoe.png", "seed": 42, "output": "edited.png"}
{"prompt": "change the shoe to blue leather", "image": "test_10_shoe.png", "ref_images": ["test_10_shoe.png", "extra_ref.png"], "seed": 43, "output": "edited2.png"}
```

### Fields

- `prompt` (required): The edit prompt describing the desired changes.
- `image` (required): Path to the target image to edit.
- `ref_images` (optional): Additional reference images.
- `seed` (optional): Random seed. Defaults to a random value.
- `output` (optional): Output filename. Defaults to `edit_output_{line_num}.png`.
- `num_inference_steps` (optional): Number of denoising steps.
- `width`, `height` (optional): Output resolution.
- `guidance` (optional): Guidance scale.
- `negative_prompt` (optional): Negative prompt.

## Files Modified

- `mage_mlx/worker.py`: Added `run_edit_worker()` function with lazy loading, embedding
  cache, vision cache, terminal output, and metadata generation.
- `mage_mlx/mflux_src/mflux/models/mage_flow/variants/edit/mage_flow_edit.py`: Modified
  `load_dit_vae()` to accept `model_dir` and `quantize` parameters, skip text_encoder
  when already loaded.
- `mage_mlx/mflux_src/mflux/models/mage_flow/mage_flow_initializer.py`: Modified
  `_init_tokenizers()` to gracefully handle missing tokenizer files (sets empty dict),
  and `_init_tokenizers()` to accept `tokenizer_path` parameter.
- `mage_mlx/mflux_src/mflux/models/mage_flow/weights/mage_flow_weight_mapping.py`:
  Extended `transform_vae_key()` for already-converted keys, made `transform_vae_weight()`
  a no-op for MLX-native VAE weights.
- `generate.py`: Added `--edit` flag and edit worker routing.

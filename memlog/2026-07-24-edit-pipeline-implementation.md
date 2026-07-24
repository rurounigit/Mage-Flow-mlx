# Edit Pipeline Implementation

## Date: 2026-07-24

## Summary

Ported mflux's `MageFlowEdit` image editing pipeline to Mage-Flow-mlx,
enabling text-guided image editing with reference images.

## Changes

### New Files

1. **`mage_mlx/edit.py`** — Core edit pipeline
   - `MageFlowEditUtil`: Reference image VAE encoding, latent packing, cache key computation
   - `MageFlowEdit`: Full edit pipeline with:
     - Reference image encoding via VAE → packed latents
     - Multi-modal text encoding (edit prompt + reference images)
     - Target + reference latent concatenation along sequence dim
     - Multi-image `img_shapes` for the DiT
     - `target_length` slicing to extract target prediction
     - CFG (classifier-free guidance)
     - Optional velocity renormalization
     - 4-step flow matching loop
     - VAE decode

2. **`mage_mlx/vision_model.py`** — Qwen3-VL vision tower (ported from mflux)
   - `MageFlowPatchEmbed`: Patch embedding with convolutional projection
   - `MageFlowAttention`: Multi-head attention with per-image frame indexing
   - `MageFlowMLP`: GELU-activated MLP
   - `MageFlowQwen3VLMerger`: Projection from hidden to vision output dim
   - `MageFlowQwen3VLVisionModel`: Full vision model with stacked blocks

3. **`mage_mlx/prompt_processor.py`** — Shared prompt templates
   - `MageFlowPromptProcessor`: Static methods for txt2img and edit templates
   - `format_text_to_image()`: Text-to-image prompt template
   - `format_edit()`: Edit prompt template with image placeholders
   - `trim_or_pad_tokens()`: Token length management

4. **`mage_mlx/processor.py`** — Qwen3-VL image processor + tokenizer wrapper
   - `MageFlowQwen3VLProcessor`: Handles image preprocessing, tokenization, and `<|image_pad|>` expansion

### Modified Files

5. **`mage_mlx/text_encoder.py`** — Replaced with native MLX Qwen3-VL
   - `MageFlowTextEncoder`: Native MLX implementation (no PyTorch dependency)
   - `encode_text_to_image()`: Text-only encoding for txt2img
   - `encode_edit()`: Multi-modal encoding (text + reference images) for edit
   - Integrated vision tower for multi-modal processing

6. **`mage_mlx/vae.py`** — Added edit support
   - `pack_latents()`: NHWC → sequence [1, H*W, 128]
   - `unpack_latents()`: Sequence → NHWC
   - `encode()`: VAE encoding with `encode_moments()` → `_moments()`
   - `decode()`: VAE decoding with Gaussian shading

7. **`mage_mlx/pipeline.py`** — Updated for new text encoder
   - `MageFlowTokenizer`: Tokenizer wrapper exposing `tokenizer` and `processor`
   - `from_pretrained()`: Loads tokenizer, creates `MageFlowTokenizer` wrapper
   - `generate()`: Uses `encode_text_to_image()` instead of `__call__(prompt)`

8. **`mage_mlx/loader.py`** — Vision weight conversion
   - `map_text_encoder_key()`: Now keeps vision tower weights (maps to `visual.*`)
   - `ensure_mlx_model()`: No longer strips vision tower keys during conversion

9. **`mage_mlx/embedding_cache.py`** — Updated imports
   - Uses `MageFlowPromptProcessor` instead of `MAGE_FLOW_TEMPLATE`

10. **`mage_mlx/__init__.py`** — Updated exports
    - Exports `MageFlowEdit`, `MageFlowEditUtil`, `MageFlowPromptProcessor`,
      `MageFlowQwen3VLProcessor`, `MageFlowTextEncoder`, `MageFlowQwen3VLVisionModel`

11. **`generate.py`** — Added edit subcommand
    - `--image`: Target image path
    - `--ref-images`: Comma-separated reference image paths
    - `--renormalization`: Velocity renormalization flag
    - `_run_edit()`: Edit pipeline entry point

## Architecture

The edit pipeline works as follows:

1. **Reference encoding**: Reference images are resized, normalized, and encoded
   via VAE → packed latents [1, N_refs*H*W, 128]

2. **Prompt encoding**: The edit prompt (text + reference images) is encoded
   via the multi-modal text encoder (Qwen3-VL with vision tower)

3. **Latent concatenation**: Target latents [1, H*W, 128] are concatenated
   with reference latents along the sequence dimension

4. **DiT inference**: The DiT runs with multi-image `img_shapes`
   [(1, H, W), (1, H, W), ...] and the concatenated latents

5. **Slicing**: Output is sliced with `target_length` to extract only the
   target prediction

6. **CFG**: Unconditional pass with negative prompt, then
   `v_uncond + scale * (v_cond - v_uncond)`

7. **Flow matching**: 4-step Euler integration

8. **VAE decode**: Final latent → image

## CLI Usage

```bash
# Text-to-image (unchanged)
python generate.py --prompt "A futuristic cityscape" --output output.png

# Image editing (new)
python generate.py --prompt "make the sky purple" \
    --image target.png \
    --ref-images ref1.png,ref2.png \
    --output edited.png
```

## Notes

- The `MageFlowTextEncoder` is now a native MLX implementation (no PyTorch
  dependency for inference), replacing the previous `Qwen3VLTextEncoder`
  which used `transformers` for tokenization and a custom MLX model.
- Vision tower weights are now preserved during model conversion (previously
  stripped), enabling multi-modal encoding for image editing.
- The edit pipeline reuses the existing DiT, VAE, and scheduler components,
  adding only the reference encoding, latent concatenation, and multi-image
  shape handling on top.

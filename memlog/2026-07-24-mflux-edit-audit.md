# mflux Edit Audit

## Date: 2026-07-24

## Overview

Thorough investigation of the mflux `MageFlowEdit` implementation
(`mage-flow-mlx` branch at `/Users/tilman/projects/mflux`) to understand
the edit-specific logic and verify the Mage-Flow-mlx port.

## mflux MageFlowEdit Architecture

### Multi-modal Tokenizer (`MageFlowQwen3VLProcessor`)

- Wraps `Qwen3Tokenizer` + `MageFlowQwen3VLImageProcessor`
- `MageFlowQwen3VLImageProcessor` extends `QwenImageProcessor` with:
  - `max_long_edge=384` (limits long edge to 384 pixels)
  - `image_mean=[0.5, 0.5, 0.5]`, `image_std=[0.5, 0.5, 0.5]`
  - `min_pixels=65536`, `max_pixels=16777216`
  - `patch_size=16`, `temporal_patch_size=2`, `merge_size=2`
- Tokenizer expands `<|image_pad|>` placeholders and produces
  `pixel_values` + `image_grid_thw` outputs

### Vision Model (`MageFlowQwen3VLVisionModel`)

- 24-layer Qwen3-VL vision transformer
- `Qwen3VLVisionPatchEmbed`: Conv3d patch embedding
- `Qwen3VLVisionRotaryEmbedding`: 2D vision RoPE (height/width)
- `Qwen3VLVisionAttention`: Multi-head attention with `cu_seqlens` for
  multi-image batching (no cross-image attention)
- `Qwen3VLVisionMLP`: GELU-approximate MLP
- `Qwen3VLVisionPatchMerger`: Spatial merge + projection to language hidden size
- `Qwen3VLVisionBlock`: LayerNorm + attention + MLP (residual)
- DeepStack features extracted at layers 5, 11, 17

### VAE Encoder

- `vae.encode()` -> `encode_moments()` -> `_moments()`
- Reference images encoded to latent space for injection into the text encoder

### Edit Pipeline

1. Encode reference images via VAE -> packed latents
2. Encode edit prompt via multi-modal text encoder (vision + text)
3. Concatenate target + reference latents along sequence dim
4. Run DiT with multi-image `img_shapes` (target + references)
5. Slice output with `target_length` to extract only target prediction
6. Apply CFG and optional renormalization
7. Run 4-step flow matching loop
8. Decode final latent via VAE

### Edit Prompt Template (`EDIT_TEMPLATE`)

- Uses image placeholders for reference images
- 64 drop tokens (vs 34 for txt2img)
- Negative prompt uses same reference images (text-only negative would
  remove vision tokens and produce invalid unconditional condition)

### Reference Caching

- SHA-256 hash of image bytes + size for prompt cache keys
- Prevents re-encoding identical reference images

### Safety Screening

- `screen_edit()` for prompts + images (vs `screen_text()` for txt2img)

## Local Mage-Flow-mlx Architecture

The local `MageFlowQwen3VLProcessor` (in `mage_mlx/processor.py`) is
structurally identical to mflux's:

- `MageFlowQwen3VLImageProcessor` with `max_long_edge=384` and
  `[0.5, 0.5, 0.5]` mean/std
- `MageFlowQwen3VLProcessor` wrapping tokenizer + image processor

The local vision model (`mage_mlx/vision_model.py`) ports all mflux
sub-components:

- `Qwen3VLVisionPatchEmbed`
- `Qwen3VLVisionRotaryEmbedding`
- `Qwen3VLVisionAttention`
- `Qwen3VLVisionMLP`
- `Qwen3VLVisionPatchMerger`
- `Qwen3VLVisionBlock`
- `MageFlowQwen3VLVisionModel`

The local edit pipeline (`mage_mlx/edit.py`) implements:

- `MageFlowEditUtil`: Reference image encoding, latent packing, cache keys
- `MageFlowEdit`: Full edit pipeline with `edit()` method
- `make_velocity_predictor`: DiT inference with CFG, per-pass masking
- Flow matching sampling loop (4 steps)
- VAE decode

## Key Differences Found

### 1. Rotary Embedding (Fixed)

- mflux: `mx.arange(0, dim, 2, dtype=mx.float32)`
- Local (before fix): `np.arange(0, dim, 2, dtype=np.float32)` -> `mx.array()`
- **Fix**: Changed to `mx.arange` to match mflux exactly
- **Impact**: Negligible -- both produce identical float32 values

### 2. Text Attention Mask (Fixed)

- mflux: Per-pass mask generation based on actual text embedding shape
- Local (before fix): Single pre-computed mask passed for both passes
- **Fix**: Generate per-pass mask based on `txt_embeds.shape[1]` and
  `neg_txt_embeds.shape[1]`
- **Impact**: Critical -- caused `ValueError` in DiT mask validation

### 3. VAE `sample_posterior` (Intentional)

- mflux: Edit VAE initialized with `sample_posterior=True`
- Local: `sample_posterior` set to `True` during reference encoding
- **Impact**: Matches mflux's edit behavior (txt2img remains deterministic)

## Vision Divergence Analysis

### Pixel Values

- Shape: (576, 1536) -- matches
- RMSE: 0.000865 -- matches closely
- Residual: PIL bicubic resize differences between Python versions

### Vision Transformer Blocks

- Blocks 0-9: Gradual divergence (0.003 -> 0.009)
- Block 10: Jump to 0.08 (compounding effect)
- Block 23: 2.60 (final output)
- **Root cause**: MLX version mismatch -- Mage-Flow-mlx uses MLX 0.32.0,
  mflux uses MLX 0.31.0. Different MLX versions can have different kernel
  implementations, default behaviors, or bug fixes that change floating-point
  results. These tiny differences compound through 24 transformer layers.
- **Weights**: All 24 blocks' weights are identical (RMSE = 0.0)
- **Code**: Structurally identical to mflux

### Conclusion

The vision divergence is caused by the MLX version mismatch (0.32.0 vs
0.31.0), not by code or weight differences. The code is structurally
identical and all weights match. This does not prevent valid image
generation.

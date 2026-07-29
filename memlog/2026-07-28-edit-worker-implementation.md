# Edit Worker Implementation

## Date: 2026-07-28

## Summary

Implemented a new `--edit` flag for the worker mode that enables batch image editing
using the Mage-Flow Edit pipeline. The edit worker mirrors the txt2img worker's
architecture (load Qwen → encode all prompts → unload Qwen → load DiT+VAE → generate)
but with multimodal edit encoding and VAE reference latent caching.

## New Files

### `mage_mlx/vision_cache.py`
Standalone VAE reference latent cache. Caches the VAE-encoded reference image latents
keyed by raw image bytes + image size + VAE checkpoint signature. This avoids
re-encoding the same reference images through the VAE on every generation, saving
~0.5-1.0s per prompt.

### `test_edit_worker.py`
Comprehensive test suite for the edit worker's JSONL loading logic, image path
validation, cache key construction, and parameter merging. All tests pass without
requiring GPU or model weights.

## Modified Files

### `mage_mlx/embedding_cache.py`
- Added `ref_image_hashes` parameter to `make_key()` for edit-specific cache keys.
  Edit embeddings depend on both the prompt text AND the reference images, so the
  cache key must include image hashes to avoid serving wrong embeddings.
- Added `ref_image_hashes` to the cache key hash computation.

### `mage_mlx/worker.py`
- Added `EDIT_VALID_PARAMS` (extends `VALID_PARAMS` with `image` and `ref_images`).
- Added `load_edit_prompts()`: JSONL loader with image path validation.
  - Validates image existence and openability (PIL `verify()`).
  - Skips prompts with missing/malformed image paths with a warning.
  - Normalizes `image` (target) and `ref_images` (additional references).
  - Supports `ref_images` as a single string or list.
- Added `_hash_image_bytes()`: SHA-256 hash of raw image file bytes.
- Added `run_edit_worker()`: The main edit worker function.
  - Phase 0: Load `MageFlowEdit` (text encoder + tokenizer only, DiT+VAE deferred).
  - Phase 1: For each prompt — load reference images, compute image hashes,
    check embedding cache (keyed by prompt + ref image hashes), encode edit
    prompt via Qwen (multimodal `encode_edit`), save to cache. Unload Qwen
    after all prompts.
  - Phase 1.5: Load DiT + VAE (after Qwen is unloaded — reduces peak RAM).
  - Phase 2: For each prompt — check vision cache for VAE-encoded reference
    latents, encode via VAE if miss, run edit denoising loop, decode, save.
  - Handles pipeline reload (model/quantize change) and scheduler reset
    (steps change) between prompts.
  - Full LiveReport terminal output and JSON+MD metadata generation.

### `generate.py`
- Added `--edit` flag: `action="store_true"`, requires `--worker`.
- Added edit worker mode block: calls `run_edit_worker()` with CLI defaults
  (including `renormalization`).
- Added validation: `--edit` requires `--worker` to be set.

## Architecture

```
generate.py --worker prompts.jsonl --edit --metadata
    ↓
run_edit_worker(jsonl_path, defaults, profiler, metadata_enabled, report)
    ↓
Phase 0: MageFlowEdit(load_dit_vae=False)  ← text encoder + tokenizer only
    ↓
Phase 1: For each prompt:
    load reference images → compute hashes → check EmbeddingCache
    → encode_edit (Qwen multimodal) → cache.put()
    ↓
    text_encoder.unload() + gc + clear_cache  ← Qwen freed
    ↓
Phase 1.5: edit.load_dit_vae()  ← DiT + VAE loaded
    ↓
Phase 2: For each prompt:
    check VisionCache → encode_references (VAE) if miss
    → denoising loop (edit_step_N) → VAE decode → save
    ↓
    profiler.save_metadata() → JSON + MD files
```

## Memory Optimization

The key optimization mirrors the txt2img worker: Qwen (~7.5 GiB) is loaded and
unloaded BEFORE DiT + VAE (~7.9 GiB) are loaded, so peak RAM is
max(Qwen, DiT+VAE) instead of Qwen + DiT + VAE simultaneously (~15.4 GiB).

## Cache Keys

### Embedding Cache (text encoder)
```
key = hash(prompt, negative_prompt, te_path, ref_image_hashes)
```
Edit embeddings depend on both prompt text AND reference images, so the cache
key includes SHA-256 hashes of all reference image files.

### Vision Cache (VAE reference latents)
```
key = hash(image_bytes, image_size, vae_path)
```
VAE-encoded reference latents depend on the raw image pixels and VAE weights,
so the cache key includes raw image bytes and VAE checkpoint signature.

## JSONL Format

```jsonl
{"prompt": "make the sky blue", "image": "input.png", "output": "output1.png"}
{"prompt": "add a hat", "image": "input.png", "ref_images": ["style.png"], "seed": 42}
{"prompt": "change color", "ref_images": ["ref1.png", "ref2.png"], "guidance": 5.0}
```

### Fields
- `prompt` (required): Edit instruction text.
- `image` (required if no `ref_images`): Target image to edit.
- `ref_images` (required if no `image`): List of reference image paths.
  If `image` is also present, it's prepended to the reference list.
  If only `ref_images` is present, the first entry becomes the target image.
- `output` (optional): Output path. Defaults to `edit_output_N.png`.
- `seed`, `steps`, `guidance`, `width`, `height`, `quantize`, `model`,
  `negative_prompt`, `renormalization`: Same as txt2img worker.

## Error Handling

- Missing `prompt` field → skip with warning.
- Missing `image` and `ref_images` → skip with warning.
- Non-existent image path → skip with warning.
- Malformed/unopenable image → skip with warning.
- Invalid JSON line → skip with warning.
- Unknown parameter names → skip with warning.
- Comments (`#`) and blank lines → silently skipped.

## Terminal Output

The edit worker uses the same `LiveReport` system as the txt2img worker:
- Phase headers (magenta bold) with prompt index.
- Per-phase metadata (prompt, resolution, steps, quantize, seed).
- Real-time phase timing bars.
- Cache HIT/MISS labels for text encoding.
- Overview table with per-prompt timing and peak RSS.
- Run-level metadata summary.
- JSON + MD file output when `--metadata` is set.

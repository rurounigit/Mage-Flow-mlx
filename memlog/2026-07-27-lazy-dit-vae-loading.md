# Lazy DiT + VAE Loading Optimization

**Date**: 2026-07-27
**Branch**: feat/optimization-set-1
**Author**: Claude

## Problem

The original loading order loaded DiT + VAE + Qwen text encoder all at once
during `from_pretrained()`. This meant that during prompt encoding, all three
models were resident in Metal memory simultaneously, resulting in a peak RAM
of ~15.4 GiB (7.9 GiB DiT+VAE + 7.5 GiB Qwen).

## Solution

Reordered the loading sequence so that:

1. **Text encoder (Qwen)** is loaded first → prompts are encoded → Qwen is
   unloaded
2. **DiT + VAE** are loaded only after Qwen is unloaded

This reduces peak RAM from ~15.4 GiB to ~7.9 GiB on cache miss, because Qwen
(~7.5 GiB) is never resident alongside DiT + VAE (~7.9 GiB) simultaneously.

## Changes

### `mage_mlx/pipeline.py`

- **`__init__`**: Changed `transformer` and `vae` type hints from required to
  `Optional`, allowing `None` when only the text encoder is loaded.
- **`from_pretrained_text_encoder()`** (new): Class method that loads only the
  text encoder + tokenizer. Returns a pipeline with `transformer=None, vae=None`.
- **`load_dit_vae()`** (new): Instance method that loads DiT + VAE weights into
  an existing pipeline. Call this after text encoding is complete.
- **`_generate_from_embeds()`**: Added validation that `transformer` and `vae`
  are not `None` before generating.
- **`from_pretrained()`**: Unchanged — still loads all three models for
  backward compatibility (used by benchmark cleanup mode and tests).

### `mage_mlx/worker.py`

- **Phase 0**: Replaced `from_pretrained()` with `from_pretrained_text_encoder()`
  — loads only text encoder + tokenizer.
- **Phase 1.5** (new): Added `load_dit_vae()` call after Qwen is unloaded and
  before image generation begins.
- **Phase 2 reload**: Changed pipeline reload from `from_pretrained()` to
  `load_dit_vae()` — only DiT + VAE are reloaded, not the text encoder.
- Updated module docstring and `run_worker()` docstring.

### `generate.py`

- **Single-prompt mode**: Replaced `from_pretrained()` with
  `from_pretrained_text_encoder()`. Added inline text encoding logic (cache
  check → encode → unload Qwen → `load_dit_vae()` → `_generate_from_embeds()`).
- Added `import gc` and `import mlx.core as mx` to top-level imports.
- Updated profiler phase tracking to include `dit_load` and `vae_load` phases.
- **Edit mode**: Unchanged — still uses `MageFlowEdit` which loads all models
  at once (vision tower needed throughout).
- **Benchmark cleanup mode**: Unchanged — still uses `from_pretrained()`.

## Verified Results

| Mode | Before (cache miss) | After (cache miss) | Cache hit |
|------|---------------------|--------------------|-----------|
| Worker | 15.42 GiB | **7.94 GiB** | **8.16 GiB** |
| Single-prompt | 15.42 GiB | **7.94 GiB** | **7.94 GiB** |
| Edit | 16.12 GiB | 16.12 GiB (unchanged) | N/A |

### Key observations from test runs:

1. **Cache miss (single-prompt)**: Peak RAM is 7.94 GiB. The `text_encode`
   phase peaks at 7.94 GiB (Qwen loaded), then Qwen is immediately unloaded
   to 1.84 GiB before DiT+VAE are loaded. Peak = max(text_encode, dit+vae).

2. **Cache hit (single-prompt)**: Peak RAM is 7.94 GiB. Qwen is never loaded;
   embeddings are loaded from disk cache. Same as cache miss because the
   DiT+VAE loading dominates.

3. **Cache hit (worker)**: Peak RAM is 8.16 GiB. Slightly higher than single-prompt
   due to both embeddings being resident in memory simultaneously.

4. **No duplicate profiler phases**: The `on_phase_complete` callback filter
   now includes `dit_load` and `vae_load` in the explicit set, preventing
   double-reporting when `load_dit_vae()` handles profiler start/stop internally.

5. **"lazy loading" label**: Added `loading_mode` parameter to
   `LiveReport.stop_phase()`. The `on_phase_complete` callback in both
   `generate.py` and `worker.py` passes `loading_mode="lazy loading"` for
   the `text_encoder_load` phase, since Qwen weights are lazy-loaded during
   `text_encode` (not during the constructor). This replaces the misleading
   `0.0s` timing with a descriptive label.

## Backward Compatibility

- `from_pretrained()` is unchanged and still works for all existing callers.
- `generate()` method is unchanged and still works when all models are pre-loaded.
- Edit mode is unchanged.
- Benchmark cleanup mode is unchanged.

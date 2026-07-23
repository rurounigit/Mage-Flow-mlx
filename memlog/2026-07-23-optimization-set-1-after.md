# Optimization Set 1 — After Results (Prompt Queue Mode + Cache Fix + Bypass)

## Date: 2026-07-23

## What Changed Since Baseline

1. **Prompt queue mode**: Qwen loaded once, all prompts encoded in batch, Qwen unloaded once
2. **Embedding cache fix**: `mx.save()` requires `.npy` extension — fixed temp file naming
3. **`_generate_with_cached_embeds` fix**: Now returns correct positive/negative embeddings separately
4. **`cleanup_strategy="unload_only"`**: Worker uses this to avoid unnecessary gc/cache clearing
5. **`_generate_from_embeds()` bypass**: Worker calls this instead of `pipeline.generate()` for cached prompts,
   skipping text encoding and Qwen unloading entirely

## Test Configuration

- **Model**: Mage-Flow MLX (4-bit quantized DiT)
- **Resolution**: 1024×1024 (default), 512×512 (prompt 4)
- **Steps**: 4 (default), 8 (prompt 5)
- **Quantization**: 4-bit cache
- **Worker mode**: JSONL persistent worker (DiT + VAE resident)
- **Prompts**: 5 (3 unique, 2 repeated for cache testing)
- **Cache state**: All 5 prompts had cache hits from previous run

## Phase-Level Timings

### Pipeline Loading (one-time, cached weights)

| Phase | Time (s) | Notes |
|-------|----------|-------|
| python_startup | 0.00 | Python + MLX import |
| dit_load | 0.0078 | 4-bit DiT from cached MLX weights |
| vae_load | 0.0066 | VAE from cached MLX weights |
| text_encoder_load | 0.00 | Text encoder wrapper (lazy) |
| pipeline_reload | 0.0227 | Pipeline assembly |
| text_encoder_unload | 0.0185 | Qwen unloaded (all cache hits, no encoding) |

**Total pipeline load + unload: ~0.05s** — cached MLX weights load almost instantly.

### Phase 1: Pre-Encoding (Qwen Batch Mode)

All 5 prompts got cache hits — Qwen was not loaded at all.

| Prompt | Status | Time (s) | Notes |
|--------|--------|----------|-------|
| 1 | Cache HIT | 0.00 | Skipped — Qwen not loaded |
| 2 | Cache HIT | 0.00 | Skipped — Qwen not loaded |
| 3 | Cache HIT | 0.00 | Skipped — Qwen not loaded |
| 4 | Cache HIT | 0.00 | Skipped — Qwen not loaded |
| 5 | Cache HIT | 0.00 | Skipped — Qwen not loaded |

**Total Phase 1: ~0.02s** — vs ~3.94s in previous run (Qwen loaded once, 2 encodes).

### Phase 2: Generation (DiT + VAE, cached embeddings, no text encoding)

| Phase | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 |
|-------|----------|----------|----------|----------|----------|
| dit_step_1 | 3.03 | 2.98 | 2.93 | 0.91 | 3.14 |
| dit_step_2 | 2.89 | 2.90 | 3.07 | 0.58 | 3.07 |
| dit_step_3 | 2.88 | 2.89 | 3.02 | 0.59 | 3.01 |
| dit_step_4 | 2.89 | 2.89 | 3.00 | 0.58 | 3.03 |
| dit_step_5 | — | — | — | — | 3.02 |
| dit_step_6 | — | — | — | — | 3.06 |
| dit_step_7 | — | — | — | — | 3.21 |
| dit_step_8 | — | — | — | — | 3.07 |
| vae_decode | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| generation_total | 12.79 | 12.04 | 12.65 | 3.36 | 25.30 |
| save | 0.06 | 0.05 | 0.06 | 0.01 | 0.06 |

### Comparison: Three Runs

| Metric | Baseline (pipeline.generate per prompt) | Previous (patch text encoder) | Current (_generate_from_embeds bypass) |
|--------|-------------------|----------------------|------------------------|
| Cache hits | 0/5 | 3/5 | 5/5 |
| Qwen loads | 5 (per prompt) | 1 (batch) | 0 (all cached) |
| Phase 1 total | ~16.33s | ~3.94s | ~0.02s |
| Phase 2 total | ~78s | ~92s | ~66s |
| Total wall clock | 94.42s | 95.83s | 66.44s |
| Per-image avg | ~19s | ~19s | ~13s |
| First DiT step (prompt 2) | 7.49s | 12.27s | 2.98s |

### Key Observations

1. **31% faster total wall clock**: 66.44s vs 95.83s. The bypass of `pipeline.generate()` eliminated
   `text_encoder.unload()` + `mx.clear_cache()` per prompt, which was discarding reusable allocations.

2. **All 5 prompts got cache hits**: Qwen was not loaded at all. The embedding cache from the previous
   run was used entirely. Cache files are ~118 KB each — tiny compared to 8 GiB of Qwen weights.

3. **DiT steps are now consistent**: First step of each prompt is ~3s (1024×1024) or ~0.9s (512×512).
   No more 12s spikes from `mx.clear_cache()` discarding allocations.

4. **Phase 1 is negligible**: 0.02s (just the Qwen unload, since all cache hits).

5. **RoPE cos/sin cache is working**: DiT steps are consistent at ~3s, indicating no redundant
   cos/sin computations across the 12 attention blocks.

6. **Scheduler reset works**: Prompt 5 (8 steps) correctly runs 8 DiT steps.

7. **Resolution change works**: Prompt 4 (512×512) generates correctly with faster DiT steps
   (~0.58s vs ~3s for 1024×1024).

### What Was Fixed

The root cause of the "slowness" was that `pipeline.generate()` called `text_encoder.unload()` per
prompt, which calls `mx.clear_cache()`. This discarded reusable MLX allocations, forcing re-allocation
on the next DiT step. The first DiT step of each subsequent prompt was 2-4× slower (e.g., 12.27s vs 3.03s).

The fix: `_generate_from_embeds()` bypasses `pipeline.generate()` entirely for cached prompts. It
directly runs DiT steps + VAE decode using pre-encoded embeddings, skipping text encoding and unloading.

### Next Steps

1. **HTTP server mode** for remote generation requests
2. **Evaluate `mx.compile`** for fixed-shape DiT/VAE paths
3. **Fix memory unit conversion** in profiler (values show ~4677 GiB, likely bytes→GiB issue on macOS)
4. **Fix profiler phase name collisions** (duplicate names overwrite in dict)

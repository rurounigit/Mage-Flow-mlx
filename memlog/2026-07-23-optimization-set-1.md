# Optimization Set 1 — Profiling Results

## Date: 2026-07-23

## Test Configuration

- **Model**: Mage-Flow MLX (4-bit quantized DiT)
- **Resolution**: 1024×1024 (default), 512×512 (prompt 4)
- **Steps**: 4 (default), 8 (prompt 5)
- **Quantization**: 4-bit cache
- **Worker mode**: JSONL persistent worker (DiT + VAE resident)
- **Prompts**: 5 (3 unique, 2 repeated for cache testing)

## Phase-Level Timings

### Pipeline Loading (one-time, cached weights)

| Phase | Time (s) | Notes |
|-------|----------|-------|
| python_startup | 0.00 | Python + MLX import |
| dit_load | 0.0076 | 4-bit DiT from cached MLX weights |
| vae_load | 0.0065 | VAE from cached MLX weights |
| text_encoder_load | 0.00 | Text encoder wrapper (lazy) |
| pipeline_reload | 0.0203 | Pipeline assembly |

**Total pipeline load: ~0.03s** — cached MLX weights load almost instantly.

### Per-Prompt Breakdown (5 prompts)

| Phase | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 |
|-------|----------|----------|----------|----------|----------|
| text_encode | 3.76 | 4.54 | 3.58 | 3.10 | 1.35 |
| text_encoder_unload | 0.10 | 0.69 | 0.71 | 0.75 | 0.18 |
| dit_step_1 | 3.12 | 7.49 | 5.65 | 4.23 | 2.97 |
| dit_step_2 | 3.00 | 3.00 | 3.00 | 0.56 | 2.94 |
| dit_step_3 | 3.01 | 3.02 | 3.00 | 0.56 | 2.93 |
| dit_step_4 | 3.01 | 3.01 | 2.99 | 0.56 | 2.93 |
| dit_step_5 | — | — | — | — | 2.96 |
| dit_step_6 | — | — | — | — | 3.00 |
| dit_step_7 | — | — | — | — | 3.04 |
| dit_step_8 | — | — | — | — | 3.06 |
| vae_decode | 0.004 | 0.002 | 0.004 | 0.001 | 0.002 |
| generation_total | 16.78 | 22.19 | 19.37 | 9.88 | 25.78 |
| save | 0.07 | 0.07 | 0.06 | 0.03 | 0.06 |

### Key Observations

1. **Pipeline loading is negligible** (~0.03s) — cached MLX weights are fast.

2. **Text encoding (Qwen load + encode + unload)** is the most expensive per-prompt phase: 1.35–4.54s. This includes loading ~8GB of Qwen weights, tokenizing, and the forward pass.

3. **DiT steps** are ~3s each. The first step is sometimes slower (3.1–7.5s) due to MLX compilation overhead, then stabilizes at ~3s. Prompt 4 (512×512) steps are faster (~0.56s) due to smaller latent size.

4. **VAE decode** is extremely fast: ~0.001–0.004s.

5. **RoPE cos/sin cache is working**: DiT steps are consistent at ~3s, indicating no redundant cos/sin computations across the 12 attention blocks.

6. **Scheduler reset works**: Prompt 5 (8 steps) shows "Resetting scheduler" and correctly runs 8 DiT steps.

7. **Resolution change works**: Prompt 4 (512×512) generates correctly with different latent dimensions.

### Embedding Cache Status

All 5 prompts showed "Embedding cache MISS". The cache is being checked but not populated. This needs investigation — the cache directory may not exist or the save logic may have a bug.

### Total Wall Clock

- **Total**: 94.42s for 5 images
- **Per-image average**: ~19s (first image includes pipeline load)
- **Subsequent images**: ~15–22s each

### Comparison: Without Persistent Worker

Without the persistent worker, each image would require:
- Python startup: ~2–3s
- DiT loading: ~5–10s (4-bit quantized)
- VAE loading: ~1–2s
- Qwen loading: ~3–4s
- Text encoding: ~1s
- DiT steps: ~12s (4 × 3s)
- VAE decode: ~0.004s
- PNG save: ~0.07s

**Estimated total without worker**: ~125–150s for 5 images
**With worker**: ~94s for 5 images
**Improvement**: ~35% from persistent worker alone

### Next Steps

1. **Fix embedding cache** — investigate why cache is not being populated
2. **Benchmark cleanup strategies** — run `--benchmark-cleanup` to compare unload/gc/cache combinations
3. **Fix memory unit conversion** in profiler (values show ~7862 GiB, likely bytes→GiB conversion issue)
4. **Add HTTP server mode** for remote generation requests
5. **Benchmark Qwen resident mode** — keep Qwen loaded for faster repeated encoding

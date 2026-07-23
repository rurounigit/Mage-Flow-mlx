# Optimization Set 1 — Implementation Log

## Date
2026-07-23

## Summary
Implemented the first benchmark-driven optimization set for Mage-Flow MLX:
phase-level profiler, persistent JSONL worker, prompt-embedding cache,
RoPE cos/sin cache, and cleanup benchmark.

## Files Changed

### New Files
- `mage_mlx/profiler.py` — Phase-level timing and memory profiler
- `mage_mlx/embedding_cache.py` — Persistent prompt embedding cache
- `mage_mlx/worker.py` — Persistent JSONL worker mode
- `memlog/2026-07-23-optimization-set-1.md` — This file

### Modified Files
- `mage_mlx/rope.py` — RoPE cos/sin caching (avoid 96 redundant cos/sin evals per 4-step generation)
- `mage_mlx/pipeline.py` — Added `cleanup_strategy` parameter to `generate()`; added `CLEANUP_STRATEGIES` constant
- `generate.py` — Added `--worker`, `--benchmark-cleanup` flags; `--prompt` now optional when `--worker` is used
- `mage_mlx/__init__.py` — Export `Profiler` and `EmbeddingCache`

## Detailed Changes

### 1. Phase-Level Profiler (`mage_mlx/profiler.py`)
- `Profiler` dataclass with `start()`/`stop()` phase timing via `time.perf_counter`
- Peak RSS memory tracking (auto-detects macOS bytes vs Linux KB)
- Formatted report output with phase name, time, and peak RSS
- Zero overhead when disabled (all calls guarded by `if profiler:`)
- Integrated into `generate.py` and `pipeline.py` with phases:
  - `python_startup`, `pipeline_load`, `dit_load`, `vae_load`, `text_encoder_load`
  - `text_encode`, `text_encoder_unload`, `dit_step_{1..N}`, `vae_decode`
  - `save_png`, `total_wall_clock`

### 2. RoPE Cos/Sin Cache (`mage_mlx/rope.py`)
- `MageFlowEmbedRope.__call__` now returns `(cos, sin)` tuple instead of raw angle values
- Cos/sin cached by `(frame, height, width, idx)` in `self._cache`
- `apply_rotary_emb_mageflow` accepts either pre-computed `(cos, sin)` tuple or raw angles (backward compatible)
- Avoids 24 redundant `mx.cos`/`mx.sin` evaluations per DiT step (2 per block × 12 blocks)
- Saves ~96 cos/sin evaluations across a 4-step generation
- No quality tradeoff — same mathematical result, just cached

### 3. Prompt Embedding Cache (`mage_mlx/embedding_cache.py`)
- `EmbeddingCache` class with `make_key()`, `get()`, `put()`, `clear()` methods
- Cache key incorporates: formatted prompt text, negative prompt, text-encoder checkpoint signature (size + mtime), tokenizer/template version
- Embeddings stored as `.npy` files (~240 KB for [1, 30, 2560] BF16)
- Atomic writes (temp file + rename) for crash safety
- Metadata validation (version check) on load
- Cache directory: `models/mage_flow_mlx/embedding_cache/`

### 4. Persistent JSONL Worker (`mage_mlx/worker.py`)
- `run_worker()` function that loads pipeline once, processes prompts from JSONL
- Per-prompt parameter overrides via JSON fields (CLI defaults as fallback)
- Parameter categories:
  - No reload needed: `prompt`, `negative_prompt`, `seed`, `guidance`, `width`, `height`, `output`
  - Scheduler reset: `steps`
  - Full pipeline reload: `model`, `quantize`
- `needs_reload()` function determines if reload/scheduler reset is needed
- `merge_params()` merges CLI defaults with per-prompt overrides
- JSONL format: one JSON object per line with `prompt` (required) and optional parameters

### 5. Cleanup Benchmark (`generate.py` + `pipeline.py`)
- `--benchmark-cleanup` CLI flag tests all 4 cleanup strategies:
  - `unload_only`: `text_encoder.unload()` only
  - `unload+gc`: unload + `gc.collect()`
  - `unload+cache`: unload + `mx.clear_cache()`
  - `all_three`: all three (current default)
- `cleanup_strategy` parameter added to `MageFlowPipeline.generate()`
- `CLEANUP_STRATEGIES` constant in pipeline.py
- Benchmark creates a fresh pipeline for each strategy and runs generation with profiling

## Usage

### Single image (unchanged)
```bash
.venv/bin/python generate.py --prompt "A cat" --profile
```

### JSONL worker mode
```bash
cat > prompts.jsonl << 'EOF'
{"prompt": "A cat", "seed": 42, "output": "cat.png"}
{"prompt": "A dog", "seed": 43, "output": "dog.png", "steps": 8}
{"prompt": "A bird", "seed": 44, "output": "bird.png", "width": 512, "height": 512}
EOF
.venv/bin/python generate.py --worker prompts.jsonl --profile
```

### Cleanup benchmark
```bash
.venv/bin/python generate.py --benchmark-cleanup --prompt "test" --profile
```

### Embedding cache (programmatic)
```python
from mage_mlx import EmbeddingCache
cache = EmbeddingCache("models/mage_flow_mlx")
key = cache.make_key(prompt="A cat", negative_prompt=" ")
embeds = cache.get(key)
if embeds is None:
    embeds = text_encoder("A cat")
    cache.put(key, embeds)
```

## Verification
- All 7 files pass Python syntax validation (`ast.parse`)
- All imports work correctly via `.venv/bin/python`
- `--worker` and `--benchmark-cleanup` flags appear in `--help` output
- `EmbeddingCache.make_key()` produces deterministic SHA-256 keys
- `CLEANUP_STRATEGIES` contains all 4 strategies
- Profiler prints formatted report with phase timings and peak RSS

## Next Steps
- Run actual generation with `--profile` to collect baseline phase timings
- Use profiler data to identify the slowest phases
- Implement prompt-embedding caching in the worker's cache-miss path
- Consider HTTP server mode as a wrapper around the JSONL worker

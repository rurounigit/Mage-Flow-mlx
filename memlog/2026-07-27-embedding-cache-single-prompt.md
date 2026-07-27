# Embedding Cache for Single-Prompt Mode

**Date:** 2026-07-27
**Branch:** feat/optimization-set-1
**Author:** Cline

## Objective

Extend the existing `EmbeddingCache` (previously worker-mode only) to single-prompt
generation mode (`--prompt`), so that repeated prompts skip Qwen text encoding entirely,
saving ~7.5 GiB peak RAM and ~24% generation time.

## Changes

### `mage_mlx/pipeline.py`

- Added `embedding_cache: Optional[EmbeddingCache] = None` and `te_path: Optional[str] = None`
  parameters to `MageFlowPipeline.generate()`.
- Replaced the unconditional text encoding section with cache-aware logic:
  - Checks the embedding cache for the positive prompt before encoding.
  - On cache hit: uses cached embeddings, skips `encode_text_to_image()` entirely,
    and does not start the `text_encode` profiler phase.
  - On cache miss: encodes with Qwen, caches the result for future runs.
  - Negative prompt (if `guidance > 1.0`) is also cached/retrieved independently.
  - `text_encode` profiler phase is only started when encoding is actually needed.
  - Cache HIT/MISS status is recorded as metadata on the `text_encode` phase.
- The `text_encoder_unload` phase is always called (to clean up the text encoder object),
  matching worker-mode behavior.

### `generate.py`

- In single-prompt mode, creates an `EmbeddingCache` instance and resolves the text encoder
  path via `resolve_text_encoder_path()`.
- Passes `embedding_cache` and `te_path` to `pipeline.generate()`.

## Backward Compatibility

- `embedding_cache` defaults to `None` — existing callers of `generate()` are unaffected.
- When `embedding_cache is None`, the code path is identical to the original behavior
  (always encodes with Qwen, no cache check).
- The only difference is that the `text_encode` phase now has a `cache=MISS` metadata tag,
  which is harmless and informative.
- Worker mode (`--worker`) is completely unaffected — it uses its own cache logic via
  `_generate_from_embeds()`.
- Edit mode (`--image`) is unaffected — edit prompts include vision tokens and cannot be
  cached as plain text embeddings.

## Test Results

### Cache MISS (first run with new prompt)

```
prompt: "A red rubber ball on a wooden table, simple product photography, soft lighting"

text_encode           2.0s   15.42GiB
text_encoder_unload   0.3s   7.93GiB
generation            16.8s  15.42GiB
total_wall_clock      21.1s
Peak RAM: 15.42GiB
```

JSON metadata: `text_encode.cache = MISS`

### Cache HIT (second run, same prompt)

```
prompt: "A red rubber ball on a wooden table, simple product photography, soft lighting"

Cache HIT — skipping Qwen encode
text_encoder_unload   0.0s   7.93GiB
generation            12.4s   7.94GiB
total_wall_clock      16.1s
Peak RAM: 7.94GiB
```

JSON metadata: `text_encode` phase absent (skipped entirely)

### Image Integrity

Both runs produce **byte-identical** PNG output (`cmp` confirms identical).

### Worker Mode (unaffected)

```
Prompt 1/2: Cache HIT — skipping Qwen encode
Prompt 2/2: Cache HIT — skipping Qwen encode
Peak RAM: 7.94GiB
Total time: 28.7s
```

### Backward Compatibility (no --metadata flag)

Runs successfully without `--metadata` (profiler disabled, `embedding_cache` still created
but `profiler=None`).

## Summary

| Run | Cache | Peak RAM | Total Time | text_encode Phase |
|-----|-------|----------|------------|-------------------|
| Single-prompt 1 | MISS | 15.42 GiB | 21.1s | Present (2.0s, 15.42 GiB) |
| Single-prompt 2 | HIT | 7.94 GiB | 16.1s | Absent (skipped) |
| Worker (cache hit) | HIT | 7.94 GiB | 28.7s | Absent (skipped) |

**RAM savings on cache hit: 7.48 GiB (48% reduction)**
**Time savings on cache hit: 5.0s (24% reduction)**

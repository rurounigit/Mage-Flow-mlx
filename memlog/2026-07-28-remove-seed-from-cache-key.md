# Remove Seed from Embedding Cache Key

**Date:** 2026-07-28
**Author:** Cline

## Objective

Remove the generation `seed` from the embedding cache key, since text embeddings
are seed-independent. The seed only affects DiT latent initialization, not text
encoding, so including it in the cache key created duplicate cache entries for
the same prompt with different seeds.

## Rationale

The `EmbeddingCache.make_key()` method previously included `seed` in the
SHA-256 key data. This meant that the same prompt with seeds 42, 43, 44, etc.
would each produce a separate cache entry containing identical embeddings —
wasting disk space (~240 KB per duplicate entry) and preventing cache hits
across different seeds.

Since `mx.random.seed(seed)` is called inside `_generate_from_embeds()` (not
during text encoding), the text encoder output is deterministic for a given
prompt + text encoder weights, regardless of seed.

## Changes

### `mage_mlx/embedding_cache.py`
- Removed `seed: Optional[int] = None` parameter from `make_key()`
- Removed `"seed": seed` from the `key_data` dictionary
- Bumped `EMBEDDING_CACHE_VERSION` from `2` → `3` (invalidates all old cache
  entries that were keyed with seed, ensuring a clean break)
- Updated docstring to explain why seed is intentionally excluded

### `mage_mlx/worker.py`
- Removed `seed=params["seed"]` from the `cache.make_key()` call

### `generate.py`
- Removed `seed=args.seed` from both `embedding_cache.make_key()` calls
  (positive prompt and negative prompt)

### `mage_mlx/pipeline.py`
- Removed `seed` from both `embedding_cache.make_key()` calls in `generate()`
  (positive prompt and negative prompt)

## Cache Invalidation

Bumping `EMBEDDING_CACHE_VERSION` from 2 to 3 ensures that old cache entries
(version 2 in their `.json` metadata) are explicitly rejected by `get()`:

```python
if meta.get("version") != EMBEDDING_CACHE_VERSION:
    return None
```

Old `.npy`/`.json` files remain on disk but are never usable. They can be
cleaned with `cache.clear()` if desired.

## Impact

| Aspect | Before | After |
|---|---|---|
| Same prompt + different seed | Separate cache entries (duplicates) | Shared cache entry |
| Same prompt + same seed | Cache hit | Cache hit (unchanged) |
| Old cache entries | Orphaned (never looked up) | Explicitly rejected (version mismatch) |
| Generated images | Identical | Identical (no functional change) |
| Disk usage | ~240 KB per seed | ~240 KB per unique prompt |

## Backward Compatibility

- `make_key()` still accepts the same core parameters (prompt, negative_prompt,
  te_path, template_version) — only `seed` was removed
- All call sites have been updated
- The `seed` parameter is still used for `mx.random.seed()` in the DiT pipeline
  and for metadata tracking — only the cache key generation is affected

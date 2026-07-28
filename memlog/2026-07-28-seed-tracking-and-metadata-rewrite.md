# Seed Tracking and Metadata Rewrite

## Date
2026-07-28

## Summary
Added per-prompt seed tracking across all three generation modes (single generation, worker, edit mode) and ensured the md/json metadata files correspond to the terminal output with the `--metadata` parameter.

## Changes

### 1. Embedding Cache (`mage_mlx/embedding_cache.py`)
- Added `seed` parameter to `make_key()` method
- Seed is included in the cache key hash so different seeds produce different cache entries
- This prevents cache collisions when the same prompt is used with different seeds

### 2. Single Generation Mode (`generate.py`)
- Added `seed=args.seed` to both `make_key()` calls (positive and negative prompt)
- Added `seed` to `report.add_metadata()` calls (terminal output)
- Added `seed` to `prof.set_metadata()` calls (md/json output)
- Added `seed` to `report.add_prompt()` call (overview table)
- Added `seed` to overview dict in `save_metadata()` block

### 3. Worker Mode (`mage_mlx/worker.py`)
- Added `seed=params["seed"]` to `cache.make_key()` call
- Added `seed` to `report.add_metadata()` calls (terminal output)
- Added `seed` to `profiler.set_metadata()` calls (md/json output)
- Added `seed` to `report.add_prompt()` call (overview table)
- Added `seed` to `prompt_metadata` dict
- Added `seed` to overview dict in `save_metadata()` block

### 4. Edit Mode (`generate.py` - `_run_edit`)
- Added `seed` to `report.add_metadata()` calls (terminal output)
- Added `seed` to `prof.set_metadata()` calls (md/json output)
- Added `seed` to `report.add_prompt()` call (overview table)
- Added `seed` to overview dict in `save_metadata()` block

### 5. Profiler (`mage_mlx/profiler.py`)
- Added `seed: Optional[int]` field to `_PromptRow` dataclass
- Added `seed` parameter to `LiveReport.add_prompt()` method
- Added `seed` to overview dict in `add_prompt()` (for incremental saves)
- Added `Seed` column to md overview table in `to_markdown()`
- Added `Seed` column to terminal overview table in `print_summary()`

## Files Modified
- `mage_mlx/embedding_cache.py`
- `generate.py`
- `mage_mlx/worker.py`
- `mage_mlx/profiler.py`

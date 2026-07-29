# Edit Worker Bug Fixes (2026-07-29)

## Context
The `--edit` flag for the worker mode was implemented but had several bugs that
prevented it from running. The CLI command
`python generate.py --worker test_prompts_edit.jsonl --edit --metadata` failed
with `TypeError: cannot unpack non-iterable NoneType`.

## Bugs Found and Fixed

### 1. Edit worker block ordering (generate.py)
The edit worker block (`if args.worker and args.edit:`) was placed AFTER the
regular worker block (`if args.worker:`), so `--edit` was caught by the regular
worker first. The regular worker's `load_prompts()` uses `VALID_PARAMS` (no
`image` field), so it rejected the JSONL lines with "unknown parameters:
{'image'}".

**Fix**: Changed `if args.worker:` to `if args.worker and not args.edit:`.

### 2. None return instead of tuple (worker.py)
Both `run_worker()` (line 199) and `run_edit_worker()` (line 734) returned bare
`None` when no valid prompts were found, but the caller does
`metadata, prompt_metadata = run_worker(...)`, causing
`TypeError: cannot unpack non-iterable NoneType`.

**Fix**: Changed both to `return None, None`.

### 3. Model selection doesn't check --edit (generate.py)
When `--edit` was set without `--image`, the model defaulted to
`models/microsoft_Mage-Flow-Turbo` (txt2img) instead of the edit model.

**Fix**: Added `args.edit` to the condition:
`if args.image is not None or args.edit`.

### 4. Stray directory creation (worker.py)
`EmbeddingCache` and `VisionCache` constructors call `os.makedirs()`, creating
`microsoft/Mage-Flow-Edit-Turbo/embedding_cache/` when initialized with the HF
ID. This stray directory intercepted `PathResolution` (which checks
`exists_locally` before `is_hf_cached`), preventing the HF ID from being
resolved to the HF cache.

**Fix**: Moved cache initialization after `MageFlowEdit` constructor and changed
cache to use local path `models/microsoft_Mage-Flow-Edit-Turbo`.

### 5. encode_edit AttributeError (worker.py)
`run_edit_worker` called `edit.text_encoder.encode_edit()` but the mflux
`MageFlowTextEncoder` doesn't have this method. `encode_edit` is a static
method on `MageFlowConditioning`.

**Fix**: Replaced `edit.text_encoder.encode_edit(...)` with
`MageFlowConditioning.encode_edit(..., text_encoder=edit.text_encoder, ...)`
and added the import.

## Verification
- All 35 tests pass (22 original + 13 new integration tests)
- CLI command `python generate.py --worker test_prompts_edit.jsonl --edit
  --metadata` generates 2 images and metadata files successfully
- No stray `microsoft/` directory created
- Peak RAM: 7.93GiB (lazy loading working)
- Caching: Cache MISS for both prompts, Qwen loaded once and unloaded

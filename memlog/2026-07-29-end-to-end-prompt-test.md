# End-to-End Prompt Test — 2026-07-29

## Summary

Created and ran `tests/test_end_to_end_prompts.py`, an end-to-end test that
actually invokes the full Mage-Flow MLX pipeline (loading real model weights,
generating images on the GPU) via subprocess — exactly as a user would from the
terminal. All 8 prompts ran sequentially, each waiting for the previous to
finish, and all output checks passed with zero problems.

## Test File

`tests/test_end_to_end_prompts.py`

The test runs 8 prompts in order via `subprocess.run()`, capturing stdout/stderr
for each. After all prompts complete, it checks:

1. **Terminal output** — exit code 0, no "Traceback", "Error:", "Exception",
   "FatalError" in stdout or stderr
2. **PNG files** — exist, non-zero size, valid PNG magic bytes
3. **JSON metadata** — valid JSON, has `metadata` and `phases` keys, `metadata`
   sub-dict has `model`, `generation_time_seconds`, `created_at`, phases list
   is non-empty
4. **MD metadata** — non-empty, contains `## Phases` and `## Run Metadata`
   sections

## Prompts Run (in order)

| # | Name | Command | Metadata? |
|---|------|---------|-----------|
| 1 | single_txt2img_with_metadata | `--prompt "shoe" --width 1024 --height 1024 --output test_10_shoe.png --metadata` | Yes |
| 2 | single_txt2img_no_metadata | `--prompt "shoe" --width 1024 --height 1024` | No |
| 3 | single_edit_with_metadata | `--prompt "edit" --image test_10_shoe.png --output test_10_shoe_edited.png --metadata` | Yes |
| 4 | single_edit_no_metadata | `--prompt "edit" --image test_10_shoe.png` | No |
| 5 | edit_worker_with_metadata | `--worker test_prompts_edit.jsonl --metadata --edit` | Yes |
| 6 | txt2img_worker_with_metadata | `--worker test_prompts.jsonl --metadata` | Yes |
| 7 | edit_worker_no_metadata | `--worker test_prompts_edit.jsonl --edit` | No |
| 8 | txt2img_worker_no_metadata | `--worker test_prompts.jsonl` | No |

## Results

```
=================== 1 passed, 1 warning in 300.21s (0:05:00) ===================
```

### Terminal Output Checks

All 8 prompts exited with code 0. No tracebacks, errors, or exceptions found
in any stdout or stderr output.

### PNG Output Checks

| Prompt | File | Status |
|--------|------|--------|
| 1 | output/test_10_shoe.png | ✓ OK |
| 3 | output/test_10_shoe_edited.png | ✓ OK |

(Prompts 2, 4, 7, 8 don't specify `--output`, so they use auto-generated
filenames in `output/`. Prompts 5, 6 use worker mode with outputs specified
in the JSONL files.)

### JSON Metadata Checks

| Prompt | File | Status |
|--------|------|--------|
| 1 | output/test_10_shoe.json | ✓ OK |
| 3 | output/test_10_shoe_edited.json | ✓ OK |
| 5 | output/test_prompts_edit.json | ✓ OK |
| 6 | output/test_prompts.json | ✓ OK |

All JSON files are valid, contain `metadata` and `phases` keys, and the
`metadata` sub-dict has `model`, `generation_time_seconds`, `created_at`.

### MD Metadata Checks

| Prompt | File | Status |
|--------|------|--------|
| 1 | output/test_10_shoe.md | ✓ OK |
| 3 | output/test_10_shoe_edited.md | ✓ OK |
| 5 | output/test_prompts_edit.md | ✓ OK |
| 6 | output/test_prompts.md | ✓ OK |

All MD files are non-empty and contain `## Phases` and `## Run Metadata`
sections.

## Timing Summary

| Prompt | Total Time | Peak RAM |
|--------|-----------|----------|
| 1 (txt2img + metadata) | 18.4s | 7.94 GiB |
| 2 (txt2img, no metadata) | 16.7s | 8.24 GiB |
| 3 (edit + metadata) | 22.1s | 8.27 GiB |
| 4 (edit, no metadata) | 22.4s | 8.27 GiB |
| 5 (edit worker + metadata) | 49.0s | 8.27 GiB |
| 6 (txt2img worker + metadata) | 51.0s | 7.95 GiB |
| 7 (edit worker, no metadata) | 65.3s | 8.27 GiB |
| 8 (txt2img worker, no metadata) | 46.3s | 8.37 GiB |

## Notes

- The `pytestmark = pytest.mark.timeout(6000)` line causes a
  `PytestUnknownMarkWarning` since `pytest-timeout` is not installed. This is
  a harmless warning — the subprocess timeout of 600s per prompt provides the
  actual timeout protection.
- Cache HIT was observed for prompts 1, 3, 5 (shoe prompt and edit prompts
  were previously cached from earlier runs).
- Cache MISS was observed for prompt 6 (txt2img worker with different prompts
  from test_prompts.jsonl).
- Vision cache HIT was observed for edit worker prompts (reference images
  were previously VAE-encoded).

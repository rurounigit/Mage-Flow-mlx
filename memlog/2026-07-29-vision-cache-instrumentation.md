# Vision Cache Instrumentation

## Problem

The vision cache (VAE-encoded reference latents for Mage-Flow Edit) was working
correctly — it created `.npy` + `.json` files on first run and loaded them on
second run — but there was **zero visibility** into whether it was hitting or
missing. Unlike the embedding cache (which prints "Cache HIT/MISS — skipping
Qwen encode"), the vision cache had no logging, no profiler phase, no
LiveReport row, and no metadata in the JSON output.

## Root Cause

In `run_edit_worker()` (worker.py), the vision cache code was a bare:

```python
reference_latents = vision_cache.get(vision_key)
if reference_latents is None:
    reference_latents = MageFlowEditUtil.encode_references(...)
    vision_cache.put(vision_key, reference_latents)
```

No profiler.start/stop, no print statements, no report.stop_phase, no metadata.
The phase was invisible to the LiveReport and the summary table.

## Changes

### 1. `mage_mlx/worker.py` — Vision cache instrumentation (run_edit_worker)

Added profiler phase `vae_encode_ref_{i+1}` around the cache get/put, with
HIT/MISS logging that mirrors the embedding cache pattern:

```python
if profiler:
    profiler.start(f"vae_encode_ref_{i + 1}")
reference_latents = vision_cache.get(vision_key)
if reference_latents is not None:
    cache_status = "HIT"
    if report and report.verbose:
        print(f"  Vision cache HIT — skipping VAE encode for ref image(s)")
else:
    cache_status = "MISS"
    reference_latents = MageFlowEditUtil.encode_references(...)
    vision_cache.put(vision_key, reference_latents)
    if report and report.verbose:
        print(f"  Vision cache MISS — encoding ref image(s) with VAE")
if profiler:
    profiler.stop(f"vae_encode_ref_{i + 1}")
    profiler.set_metadata(f"vae_encode_ref_{i + 1}", "cache", cache_status)
    if report:
        report.stop_phase(f"vae_encode_ref_{i + 1}", ...)
        report.add_metadata(f"vae_encode_ref_{i + 1}", "cache", cache_status)
```

### 2. `mage_mlx/worker.py` — `_EXPLICIT_PREFIXES` (run_edit_worker)

Added `"vae_encode_ref_"` to the `_EXPLICIT_PREFIXES` tuple so the
`_on_phase_complete` callback doesn't double-report the phase (it's already
reported explicitly via `report.stop_phase()`).

### 3. `mage_mlx/worker.py` — Summary `text_encode_time` (run_edit_worker)

Added `or rec.name.startswith("vae_encode_ref_")` to the `text_encode_time`
calculation so the vision cache time is included in the "text encode / decode"
row of the summary table (alongside text_encode, text_encoder_unload,
dit_load, vae_load).

### 4. `mage_mlx/profiler.py` — `LiveReport.print_summary`

Added `or ph.name.startswith("vae_encode_ref_")` to the same condition in
`print_summary()` so the terminal summary table also includes the vision cache
time in the "text encode / decode" row.

## Verification

### HIT run (cache populated from previous run)

```
  Vision cache HIT — skipping VAE encode for ref image(s)

  vae_encode_ref_1                               0.0s   7.85GiB
  cache:HIT
```

### MISS run (cache cleared)

```
  Vision cache MISS — encoding ref image(s) with VAE

  vae_encode_ref_1                               0.2s   7.86GiB
  cache:MISS
```

### Summary table (both runs)

```
    —      2.0s      7.92GiB              —       —   text encode / decode
    —      5.4s            —              —       —   overhead (load + encode + decode)
```

### Metadata JSON

The `vae_encode_ref_1` and `vae_encode_ref_2` phases appear in the `phases`
array with `metadata: {"cache": "MISS"}` (or "HIT"), and the
`summary.text_encode_time` includes the vision cache time.

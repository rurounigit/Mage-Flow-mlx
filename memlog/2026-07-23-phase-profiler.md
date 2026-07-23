# 2026-07-23 — Phase-level profiler instrumentation

## Objective

Add phase-level timing and memory tracking across the full Mage-Flow MLX
generation lifecycle, so that subsequent optimizations are data-driven rather
than speculative.

## Changes

### New file: `mage_mlx/profiler.py`

A lightweight `Profiler` dataclass with:

- `start(name)` / `stop(name)` — timed phases using `time.perf_counter()`
- Peak RSS tracking via `resource.getrusage(RUSAGE_SELF).ru_maxrss`
  (auto-detects macOS bytes vs. Linux KB)
- `print_report()` — formatted table of phase name, elapsed seconds, and
  peak RSS in GiB
- `save_report(path)` — writes the report to a file
- Zero overhead when `enabled=False` (all methods are no-ops)

### Modified: `generate.py`

- Added `--profile` CLI flag
- When enabled, wraps the full lifecycle:
  - `python_startup` — import + argparse
  - `pipeline_load` — `from_pretrained()` (which internally times sub-phases)
  - `generation` — `pipeline.generate()` (which internally times sub-phases)
  - `save_png` — PIL image save
  - `total_wall_clock` — end-to-end

### Modified: `mage_mlx/pipeline.py`

- `from_pretrained()` accepts optional `profiler` parameter and instruments:
  - `dit_load` — DiT weight loading (BF16 or quantized cache)
  - `vae_load` — VAE weight loading
  - `text_encoder_load` — Qwen3-VL text encoder instantiation
- `generate()` accepts optional `profiler` parameter and instruments:
  - `text_encode` — Qwen prompt + negative prompt encoding
  - `text_encoder_unload` — Qwen unload + gc.collect() + mx.clear_cache()
  - `dit_step_{1..N}` — each DiT denoising step (including CFG forward)
  - `vae_decode` — VAE latent-to-image decode

## Usage

```bash
# Normal generation (no profiling overhead)
python3 generate.py --prompt "A red apple on a wooden table"

# With phase-level profiling
python3 generate.py --prompt "A red apple on a wooden table" --profile
```

## Expected output (with --profile)

```
============================================================
  Mage-Flow MLX Profiler
============================================================
  Phase                                  Time (s)   Peak RSS (GiB)
  ----------------------------------------------------------------
  python_startup                            1.2345             2.34
  dit_load                                  5.6789            12.45
  vae_load                                  0.5678             2.78
  text_encoder_load                         0.0001             2.78
  text_encode                               2.3456            10.89
  text_encoder_unload                       0.1234             2.78
  dit_step_1                                3.4567             8.90
  dit_step_2                                3.2345             8.90
  dit_step_3                                3.1234             8.90
  dit_step_4                                3.0123             8.90
  vae_decode                                1.5678             4.56
  save_png                                  0.2345             4.56
  total_wall_clock                         20.0456
============================================================
```

## Design notes

- The profiler is **opt-in** via `--profile` flag. When disabled, all
  `profiler.start()` / `profiler.stop()` calls are guarded by `if profiler:`
  checks, so there is zero overhead on normal runs.
- Sub-phase timing is nested within the parent phase. The parent phase
  (`pipeline_load`, `generation`) is timed in `generate.py`, while the
  sub-phases are timed inside `pipeline.py`. The sub-phase times are
  included in the parent phase's elapsed time.
- Memory tracking uses `ru_maxrss` which reports the peak RSS since process
  start, not per-phase. This means the RSS column shows the cumulative
  peak, which is still useful for understanding memory pressure.
- The profiler is designed to be reusable for future optimizations
  (persistent worker, prompt embedding cache, etc.).

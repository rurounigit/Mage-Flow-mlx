# Thermal Throttling Analysis & Monitoring

## Date
2026-07-30

## Problem
The text embedding cache and visual cache appeared to have no or even negative
effect on generation time. In worker modes, expected time savings were not
realized. Timings kept rising across consecutive prompts.

## Root Cause: Thermal Throttling on MacBook Air M5

Analysis of the logs in `output/logs.txt` reveals a clear pattern of
**thermal throttling** on the MacBook Air M5 (passive cooling, no fan):

### Evidence from logs

**Edit worker run (cache HIT, 10 prompts):**
- Prompt 1: 14.9s (edit_step ~3.7s each)
- Prompt 5: 16.4s (edit_step ~4.0s each)
- Prompt 7: 19.5s (edit_step ~4.7s each)
- Prompt 8: 23.5s (edit_step ~5.5s each)
- Prompt 9: 26.8s (edit_step ~6.5s each)
- Prompt 10: 27.0s (edit_step ~6.5s each)

**Edit worker run (cache MISS, 10 prompts):**
- Prompt 1: 15.4s (edit_step ~3.7s each)
- Prompt 5: 27.4s (edit_step ~7.0s each)
- Prompt 6: 32.1s (edit_step ~7.8s each)
- Prompt 10: 33.1s (edit_step ~8.0s each)

The step times increase monotonically across prompts, even when caches are
HIT. This is the signature of thermal throttling: the SoC heats up during
sustained compute, and the CPU/GPU frequency is reduced to stay within
thermal limits.

### Why caches appear ineffective

1. **Text embedding cache**: On cache HIT, Qwen encoding is skipped (~0.3s
   saved per prompt). But the DiT denoising loop (4 steps × ~3.2s = ~12.8s)
   dominates total time. The cache savings are real but small relative to
   the total.

2. **Vision cache**: On cache HIT, VAE encoding of reference images is
   skipped (~0.2s saved per prompt). Again, small relative to the ~13s
   DiT+decode loop.

3. **Thermal throttling masks cache benefits**: As the chip heats up,
   each DiT step slows from ~3.2s to ~7-8s. The cache savings (~0.3-0.5s)
   are dwarfed by the thermal slowdown (~4-5s per step).

### Why worker mode shows rising times

In worker mode, all 10 prompts are generated in a single process with
models resident. The DiT + VAE stay loaded in Metal memory, and the
compute-intensive denoising loop runs back-to-back. This causes the SoC
to heat up progressively, triggering thermal throttling.

In single-prompt mode, each run is a fresh process — the chip starts
cool, so the first prompt is fast (~14-15s). But running multiple
single-prompt invocations back-to-back would show the same rising pattern.

## Solution: Thermal State Monitoring

Added a `thermal.py` module that reads macOS thermal state via the
**notify framework** (`notify_register_check` via ctypes) as the primary
detection method, with **sysctl** as a fallback. The notify framework
provides a 0-3 scale (0=NOMINAL, 1=FAIR, 2=SERIOUS, 3=CRITICAL) and
works on macOS 26 (Tahoe) without sudo. The sysctl fallback reads
`machdep.xcpm.cpu_thermal_level` and `machdep.xcpm.gpu_thermal_level`
(0-100 scale, higher = more throttled).

### Files changed

- **`mage_mlx/thermal.py`** (new): `get_thermal_state()` and
  `format_thermal_state()` functions. Uses `notify_register_check` via
  ctypes to read `com.apple.system.thermalpressurelevel` (0-3 scale).
  Falls back to `sysctl -n machdep.xcpm.cpu_thermal_level` and
  `machdep.xcpm.gpu_thermal_level` (0-100 scale) when notify is
  unavailable. Labels: NOMINAL, FAIR, SERIOUS, CRITICAL.

- **`mage_mlx/profiler.py`**: Added `thermal_state` field to `PhaseRecord`
  dataclass. Added `Profiler.get_thermal_state()` static method and
  `Profiler.set_thermal_state()` instance method. Added `thermal_state`
  property to `Profiler` that returns the most recent thermal state from
  phase records. `to_dict()`, `to_markdown()`, and `save_metadata()` all
  include thermal state in their output. `LiveReport.print_thermal_state()`
  prints thermal state with color-coding (green/yellow/magenta/red) and
  is guarded by `if not self.verbose: return` so it's a no-op in
  non-verbose mode. `LiveReport.print_summary()` color-codes the thermal
  column in the overview table using the same NOMINAL/FAIR/SERIOUS/CRITICAL
  labels.

- **`mage_mlx/__init__.py`**: Exported `get_thermal_state`.

- **`generate.py`**: Thermal state is captured at the start of each prompt
  (before generation begins) and printed to the terminal in verbose mode
  only (`if report.verbose:`). It's always captured for metadata output
  (per-prompt summary table and JSON/markdown files).

- **`mage_mlx/worker.py`**: Same integration for both `run_worker()`
  (txt2img) and `run_edit_worker()` (edit) modes.

- **`tests/test_thermal.py`** (new): Comprehensive tests for thermal state
  detection, label thresholds, formatting, and Profiler integration.

### How to use

```bash
# Run with thermal monitoring (verbose mode shows thermal state per prompt)
python generate.py --worker prompts.jsonl --metadata

# Non-verbose mode (no --metadata) still captures thermal state for
# metadata output but doesn't print it to the terminal
python generate.py --worker prompts.jsonl
```

### Expected output

```
  Thermal: CPU=0 (NOMINAL)
  ──────────────────────────────────────────────────────────────
  Prompt 1/10
  ...
  dit_step_1                                     3.2s      7.93GiB
  ...
  generation_1                                  14.9s      7.86GiB

  Thermal: CPU=2 (SERIOUS)
  ──────────────────────────────────────────────────────────────
  Prompt 2/10
  ...
  dit_step_1                                     3.6s      7.87GiB
  ...
  generation_2                                  15.4s      7.87GiB
```

The thermal state column in the summary table will show the throttling
level at the start of each prompt, making it easy to correlate rising
timings with thermal state.

## Recommendations

1. **For benchmarking**: Run single-prompt invocations with cooling breaks
   between them, or use a fan-cooled Mac (Mac Studio, Mac Pro, or MacBook
   Pro with M-series Pro/Max).

2. **For batch generation**: The worker mode is still the most efficient
   approach (models stay resident, no reload overhead). The thermal
   throttling affects all approaches equally.

3. **For development**: Use `--metadata` to see thermal state in the
   summary table. This makes it immediately visible when throttling
   is affecting results.

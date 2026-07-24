# Worker Profiler Output Deduplication

**Date:** 2026-07-24
**File changed:** `mage_mlx/worker.py`

## Problem

When running `generate.py --worker prompts.jsonl --profile`, the profiler report
was printed **twice**:

1. **"Mage-Flow MLX Worker — Full Run Profile"** — printed by `worker.py` at the
   end of `run_worker()` (line 322). At this point `total_wall_clock` had **not**
   been stopped yet (it's stopped in `generate.py` after `run_worker()` returns),
   so the report fell back to showing "Sum of all phases" = 252.28s — which is
   misleading because it double-counts nested phases (e.g., `generation_N`
   includes `dit_step_N` children).

2. **"Mage-Flow MLX Profiler"** — printed by `generate.py` (line 165) after
   `prof.stop("total_wall_clock")` was called. This report correctly showed
   `total_wall_clock` = 132.89s.

Both reports contained identical phase data; only the total line differed.

## Fix

Removed the `profiler.print_report("Mage-Flow MLX Worker — Full Run Profile")`
call from `worker.py` (lines 321–322). The single report from `generate.py`
is now the only output, and it includes the correct `total_wall_clock` value.

## Verification

- Confirmed no remaining references to "Full Run Profile" in the codebase.
- Confirmed `generate.py` line 165 still calls `prof.print_report()` after
  `prof.stop("total_wall_clock")`, ensuring the single report includes the
  wall-clock total.
- PEP 8 spacing (two blank lines between top-level functions) preserved.

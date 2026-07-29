# Edit Worker Bug Fixes

## Issues Found After Initial Implementation

### 1. Single Edit Mode Regression (Crash)

**Symptom:** `python generate.py --prompt "..." --image test.png --metadata` crashes with:
```
FileNotFoundError: No safetensors files found in models/microsoft_Mage-Flow-Edit-Turbo/text_encoder
```

**Root cause:** `_run_edit()` in `generate.py` created `MageFlowEdit` with `load_dit_vae=False`
but **without passing `text_encoder`**. The `MageFlowInitializer.init()` sets
`skip_components = {"text_encoder"} if text_encoder is not None else None` — since
`text_encoder` was None, ALL components (including text_encoder) were loaded. But the
edit model directory doesn't contain text_encoder weights.

**Fix:** In `_run_edit()`, load the text encoder from the shared path
(`models/shared/mage_flow_qwen3vl/text_encoder.safetensors`) and pass it to
`MageFlowEdit(text_encoder=te)`. Also set the tokenizer from
`Qwen/Qwen3-VL-8B-Instruct` (cached from regular worker), wrapped in
`MageFlowTokenizer`. This mirrors what `run_edit_worker` already does.

**Files changed:** `generate.py` (`_run_edit` function)

### 2. Worker Edit Mode High RAM (13.9 GiB instead of ~8 GiB)

**Symptom:** Worker edit mode uses 13.9 GiB peak RAM, while single edit mode uses 8.27 GiB.
The user's 24GB MacBook Air experiences OOM pressure.

**Root cause:** In `MageFlowInitializer.init()`, when `load_dit_vae=False` and
`text_encoder` is pre-loaded, `skip_components` only included `{"text_encoder"}`.
But ALL DiT + VAE weights were still loaded into memory during `pipeline_load`
(8.27 GiB peak), then loaded again during `load_dit_vae()` (13.88 GiB peak).
The weights were loaded but never applied to model objects (since `load_dit_vae=False`).

**Fix:** Extended `skip_components` in `MageFlowInitializer.init()` to also include
`vae` and `transformer` when `load_dit_vae=False`:
```python
skip_components = set()
if text_encoder is not None:
    skip_components.add("text_encoder")
if not load_dit_vae:
    skip_components.add("vae")
    skip_components.add("transformer")
if not skip_components:
    skip_components = None
```

This eliminates the wasteful weight loading during `pipeline_load`, reducing peak RAM
from 13.9 GiB to ~8 GiB (matching single edit mode).

**Files changed:** `mage_mlx/mflux_src/mflux/models/mage_flow/mage_flow_initializer.py`

### 3. Worker Edit Mode Slow Generation (164s/147s vs 42s)

**Symptom:** Worker edit mode generation takes 164s/147s per prompt (4 steps), while
single edit mode takes 42s. This is ~4x slower.

**Root cause:** The worker's Phase 2 denoising loop (in `run_edit_worker()`) didn't
call the callback lifecycle methods (`ctx.before_loop`, `ctx.in_loop`, `ctx.after_loop`)
and didn't clean up the computation graph (`del predict, velocity, model_input`) after
the loop. The `generate_image()` method in `mage_flow_edit.py` does both. Without proper
graph cleanup, MLX may retain computation graphs between steps, causing slowdowns.

**Fix:** Added the missing callback calls and graph cleanup to the worker's denoising
loop, matching `generate_image()` exactly:
- `ctx = edit.callbacks.start(seed=seed, prompt=params["prompt"], config=config)`
- `ctx.before_loop(target_latents)` before the loop
- `ctx.in_loop(step, target_latents)` inside the loop
- `del predict, velocity, model_input` after the loop
- `ctx.after_loop(target_latents)` after the loop
- Added `KeyboardInterrupt` handling with `ctx.interruption()`

**Files changed:** `mage_mlx/worker.py` (`run_edit_worker` Phase 2)

## Verification

- All 35 tests in `test_edit_worker.py` pass.
- Single edit mode (`--image` without `--worker`) now works correctly (image generated).
- Worker edit mode (`--worker ... --edit`) generates images with metadata.

## Final regression investigation and correction

The first repair attempt fixed crashes but did not restore correctness. Both
single-edit and edit-worker output became garbled and DiT/VAE memory increased
to 13.90 GiB. Commit `bb898d2` was used as the regression boundary.

### Root cause: incompatible checkpoint module trees

The project converter (`loader.ensure_mlx_model`) produces a flat, MLX-native
checkpoint in `models/` with compact names such as:

- `transformer_blocks.N.img_mlp.fc1`
- `transformer_blocks.N.img_mod`
- `dconv_encoder.*`
- VAE `*.layers.*` wrapper names

The edit path instantiated vendored mflux model classes, which expect names
such as `img_mlp.net.0.proj`, `img_mod.1`, and `encoder.*`. Loading used
`strict=False`, while validation was bypassed when a shared text encoder was
injected. Only 253/397 DiT keys and 232/728 VAE keys matched. The unmatched
parameters stayed randomly initialized FP32:

```
transformer: 1.758 GiB BF16 + 11.816 GiB FP32
VAE:         0.074 GiB BF16 + 0.229 GiB FP32
active:      13.876 GiB
```

This explains both broken images and the RAM regression.

### Final fix

- Restored `models/microsoft_Mage-Flow-Edit-Turbo` as the default.
- Kept `ensure_mlx_model()` as the source of truth: local converted model,
  otherwise HF cache/download followed by conversion into `models/`.
- Added the inverse converter key mapping when loading converted files into
  the vendored edit model.
- Enforced checkpoint coverage validation even when Qwen is injected.
- Recognized flat converted checkpoints in `PathResolution` without false
  warnings.
- Fixed local Edit paths so an incomplete cache resolves to
  `microsoft/Mage-Flow-Edit-Turbo`, not the text-to-image Turbo repository.
- Removed duplicate primary-image conditioning in single edit.
- Bumped edit cache formats and stored the exact attention mask.
- Preserved reference order in embedding keys.
- Included all references, target resolution, VAE signature, and posterior
  seed in vision-cache identity.

Final coverage and memory:

```
DiT: 397/397 keys, 0 missing, 0 unexpected, 0 shape mismatches
VAE: 728/728 keys, 0 missing, 0 unexpected
DiT: 7.666 GiB BF16
VAE: 0.188 GiB BF16
active DiT+VAE: 7.855 GiB
```

### Measured end-to-end results

Single edit, exact reported prompt, 1024x1024, four steps:

```
pipeline_load: 4.5s / 8.27 GiB
edit steps:    4.0s, 3.8s, 3.8s, 4.0s / 7.87 GiB
edit total:    19.8s
wall clock:    24.7s
peak RAM:      8.27 GiB
```

Edit worker, 512x512 cache miss:

```
Qwen unload:   0.46 GiB
DiT load:      7.92 GiB
generation:    3.5s / 7.86 GiB
wall clock:    10.9s
peak RAM:      8.27 GiB
```

Edit worker cache hit remained at 8.27 GiB and completed in 10.0s. Terminal
output was also verified without `--metadata`. JSON and Markdown files were
created in metadata mode. The repaired 1024 image was copied to
`test_10_shoe_edited_repaired.png` for visual review.

Regression suite: 50 tests pass. `py_compile` and `git diff --check` pass.

### Final exact worker verification

The original command was rerun unchanged:

```bash
python generate.py --worker test_prompts_edit.jsonl --metadata --edit
```

It encoded both prompts before unloading Qwen, then loaded DiT/VAE once and
produced both 1024x1024 files:

```
Qwen unload: 0.55 GiB
dit_load: 1.6s / 7.93 GiB
generation_1: 18.5s / 7.86 GiB
generation_2: 19.2s / 7.86 GiB
total: 45.7s
peak: 8.27 GiB
```

`test_prompts_edit.json` and `test_prompts_edit.md` were regenerated. Missing
and malformed image CLI cases were also run: each emitted a line-specific
warning, processed zero prompts, loaded no model, and created no image output.

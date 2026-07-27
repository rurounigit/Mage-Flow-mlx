# Lazy DiT+VAE Loading for Edit Mode

## Date
2026-07-27

## Problem
Edit mode (MageFlowEdit) loaded Qwen text encoder (~8 GiB), DiT (~4 GiB), and VAE (~1 GiB)
simultaneously during pipeline initialization, resulting in a peak RAM of **16.12 GiB**.
This was unnecessarily high because Qwen is only needed for prompt encoding, and DiT+VAE
are only needed for the denoising loop — they never overlap in functionality.

## Solution
Applied the same lazy-loading pattern already used in txt2img mode (`from_pretrained_text_encoder`
+ `load_dit_vae`) to the edit pipeline:

1. **MageFlowInitializer.init()** — Added `load_dit_vae: bool = True` parameter. When `False`,
   only the text encoder is created and weights are applied. VAE and transformer are set to `None`.

2. **MageFlowEdit.__init__()** — Added `load_dit_vae: bool = True` parameter, passed through to
   `MageFlowInitializer.init()`. Stores `_model_path`, `_quantize`, and `_model_config` for later use.

3. **MageFlowEdit.load_dit_vae()** — New method that re-resolves the model path, re-loads weights
   from disk, creates VAE and transformer instances, and applies weights. Reports `dit_load` and
   `vae_load` profiler phases.

4. **MageFlowEdit.generate_image()** — Restructured to encode text FIRST (Qwen only), then unload
   Qwen, THEN load DiT+VAE. Reference encoding happens after DiT+VAE are loaded (VAE is already
   resident). This ensures Qwen and DiT+VAE are never in Metal memory simultaneously.

5. **generate.py _run_edit()** — Passes `load_dit_vae=False` to the constructor. Added explicit
   `stop_phase` calls for `dit_load` and `vae_load` (these are in `_EXPLICIT_EXACT` so the callback
   skips them). Added `dit_load`/`vae_load` to `get_max_phase_rss` calls.

## Results

### Edit Mode (before)
```
pipeline_load    4.8s   16.12GiB
edit_step_1      6.7s    7.87GiB
edit_step_2-4    ~6.2s   7.87GiB
Peak RAM: 16.12GiB
```

### Edit Mode (after)
```
pipeline_load    3.0s    8.27GiB   ← text encoder only
dit_load         1.5s    7.93GiB   ← DiT loaded after Qwen unloaded
vae_load         1.4s    7.93GiB   ← VAE loaded after Qwen unloaded
edit_step_1      7.1s    7.87GiB
edit_step_2-4    ~6.7s   7.87GiB
Peak RAM: 8.27GiB                   ← 48.7% reduction
```

## Files Modified
- `mage_mlx/mflux_src/mflux/models/mage_flow/mage_flow_initializer.py` — Added `load_dit_vae` param
- `mage_mlx/mflux_src/mflux/models/mage_flow/variants/edit/mage_flow_edit.py` — Lazy loading + reordering
- `generate.py` — Pass `load_dit_vae=False`, add profiler phase reporting

## Notes
- Edit steps are slightly slower (~7s vs ~6.3s) due to lazy DiT+VAE loading overhead, but the
  48.7% peak RAM reduction is a significant improvement for memory-constrained environments.
- The txt2img mode already had this optimization (from_pretrained_text_encoder + load_dit_vae
  in pipeline.py), so no changes were needed there.

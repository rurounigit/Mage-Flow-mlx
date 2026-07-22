# 2026-07-22 — Mage-Flow MLX generation repair

## Outcome

- Restored coherent, prompt-faithful text-to-image generation.
- Validated with: `A red apple on a wooden table, studio photograph` at 256×256, four steps, CFG 5.
- VAE round-trip improved from 7.67 dB to 34.75 dB PSNR.

## Root causes corrected

1. Non-contiguous NumPy convolution transposes corrupted saved OHWI kernels.
2. Text encoder keys mapped to a nonexistent nested module path; 904 tensors were silently skipped.
3. Timestep frequency code omitted `exp()` and the pipeline passed an already scaled timestep.
4. Negative RoPE positions were reversed.
5. Static flow shift used the dynamic exponential formula instead of diffusers' rational formula.
6. Attention Q/K used LayerNorm instead of RMSNorm.
7. VAE GroupNorm did not use PyTorch-compatible channel grouping.
8. Decoder patch-conditioning feature order differed from PyTorch.
9. Four-bit DiT quantization accumulated destructive numerical error; 8-bit is now the supported default.
10. CLI default selection now prefers an existing `mage_flow_mlx_8bit` directory so legacy 4-bit weights are not selected accidentally.

## Verification

- Converted VAE kernel max difference from expected transpose: `0.0`.
- Text encoder load coverage: `904/904`, no missing or unexpected tensors.
- DiT load coverage: all learned tensors; only generated RoPE buffers are absent from the checkpoint.
- Scheduler sigmas: `[1.0, 0.9473683, 0.85714287, 0.6666667, 0.0]`.
- Timestep embedding maximum reference difference: `5.5e-05`.
- RoPE angle maximum reference difference: `2.4e-07`.
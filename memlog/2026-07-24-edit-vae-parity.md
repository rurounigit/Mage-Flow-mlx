# Mage-Flow Edit VAE parity investigation

## Scope
Compared Mage-Flow-mlx Edit against the original mflux implementation using the shoe reference image, seed 42, Turbo guidance 1.0, four steps, and 512x512 output.

## Findings
- Original mflux preserves the source scene better than the local implementation.
- Reference latent shapes match: `[1, 1024, 128]`.
- Reference latent comparison: RMSE `0.4305`, correlation `0.8942`.
- Prompt formatting, vision placeholders, 64-token edit trimming, posterior sampling, target/reference ordering, image shapes, scheduler, and packing were checked against mflux.
- mflux explicitly computes GroupNorm statistics in FP32. `mage_mlx/vae.py::Normalize` now matches this behavior and casts back to the activation dtype.
- FP32 LayerNorm was tested for the encoder but slightly worsened parity and was reverted.
- mflux image preprocessing is NCHW followed by the VAE transpose to NHWC; the local port provides equivalent NHWC input directly.

## Validation
- `python3 -m py_compile mage_mlx/*.py`: passed.
- `git diff --check`: passed.
- pytest could not run because pytest is not installed in the project virtual environment.

## Remaining work
Per-layer VAE activation or source-weight comparison is still required to locate the remaining encoder discrepancy. No speculative conversion change was applied.

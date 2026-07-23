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
9. Four-bit DiT quantization accumulated destructive numerical error; 8-bit restored semantics but still lost substantial fine detail.
10. Quantized inference was removed. Conversion now preserves source BF16 tensors bit-for-bit.
11. Qwen is unloaded after prompt encoding so the BF16 DiT has sufficient unified memory at 1024×1024.

## Verification

- Converted VAE kernel max difference from expected transpose: `0.0`.
- Text encoder load coverage: `904/904`, no missing or unexpected tensors.
- DiT load coverage: all learned tensors; only generated RoPE buffers are absent from the checkpoint.
- Scheduler sigmas: `[1.0, 0.9473683, 0.85714287, 0.6666667, 0.0]`.
- Timestep embedding maximum reference difference: `5.5e-05`.
- RoPE angle maximum reference difference: `2.4e-07`.
- BF16 conversion: DiT 397 tensors/8.23 GB, text encoder 398 tensors/8.04 GB, VAE 728 tensors/0.28 GB; all tensors are BF16.
- BF16 1024×1024 generation: 36.34 seconds, 16.96 GB peak footprint, zero swap.
- Visual BF16 result: coherent cityscape with materially sharper structures and fine detail than the 8-bit output.

## 2026-07-23 — Experimental runtime quantization follow-up

### Failure chain

1. An earlier quantization attempt overwrote `transformer.safetensors` with a
   packed checkpoint (745 tensors, `uint32` `img_in.weight` shaped `[3072, 16]`).
2. The pipeline instantiated regular `nn.Linear` layers and loaded the packed
   arrays into them, causing either dtype errors or `mx.addmm` shape failures.
3. Manual calls to `Linear.to_quantized()` packed weights but did not reliably
   replace every module in the model tree.
4. Quantizing every valid DiT linear layer generated structurally scrambled blue
   noise. Initial attention/MLP and MLP-only policies also showed structured
   artifacts because both still included the pathological final image MLP.

### Corrections

- Restored the canonical 397-tensor, 7.7 GiB BF16 transformer from the local
  Hugging Face source checkpoint.
- Added safetensors-header validation so a packed checkpoint cannot be accepted
  as the BF16 base cache.
- Made conversion atomic with a temporary `.safetensors` file and `os.replace`.
- Runtime quantization now uses `nn.quantize(..., class_predicate=...)`, which
  replaces matching `nn.Linear` modules with `nn.QuantizedLinear` correctly.
- Layer-by-layer tracing identified `transformer_blocks.11.img_mlp.fc1` as the
  pathological layer. Quantizing that layer alone caused 49.7% relative error
  in the final velocity prediction (cosine similarity 0.903).
- Four-bit mode quantizes 143 block attention/MLP matrices while preserving
  modulation, boundary/timestep/norm/output projections and the pathological
  final image MLP expansion in BF16.

### Verification

- BF16 cache: 397 tensors; `img_in.weight` is BF16 `[3072, 128]`; no
  `.scales`/`.biases` quantization state.
- BF16 1024×1024 portrait: coherent and prompt-faithful, 18.26 seconds.
- Full quantization: catastrophic scrambled output.
- Single-layer quantization used the correct packing/layout path; group size 32
  produced lower weight/output error than 64 or 128.
- Quantizing blocks 0–10 fc1 layers accumulated gradually to 10.9% final-output
  error. Adding block 11 image fc1 jumped to 48.8%; block 11 text fc1 alone had
  no effect on the image output.
- Corrected broad 4-bit policy (143 layers, sensitive layer excluded): coherent,
  prompt-faithful portrait without structured artifacts, 16.19 seconds.

### Persistent quantized caches

- Added lazy, separate `transformer_quant4.safetensors` and
  `transformer_quant8.safetensors` caches; canonical BF16 remains immutable.
- Added sidecar JSON metadata containing format, policy version, bit depth,
  group size, mode, sensitive exclusions, and BF16 source size/mtime signature.
- Cache files are validated by metadata and representative packed/BF16 tensor
  headers. Stale or incompatible caches are rebuilt from BF16.
- Writes are atomic: packed weights are completed first, then metadata is
  published. Missing metadata prevents an incomplete cache from being used.
- Loading a valid cache quantizes the empty model structure first and then loads
  packed `weight`, `scales`, and `biases`, matching MFlux's stored-quantized order.
- 4-bit cache: 4.548 GB; 8-bit cache: 5.889 GB (decimal sizes).
- Freshly quantized versus cache-reloaded DiT outputs had max absolute difference
  `0.0` for both 4-bit and 8-bit variants.
- Normal pipeline reuse was verified for both variants. Cached 4-bit CLI
  generation remained visually coherent and completed denoising/decoding in
  15.97 seconds.
# v0.0.1 — Initial Release

**Mage-Flow MLX: Native Apple Silicon port of Microsoft's Mage-Flow (4B MMDiT) using MLX.**

A translation of Microsoft's [Mage-Flow](https://github.com/microsoft/Mage) from PyTorch/CUDA to native [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. No PyTorch or CUDA needed at inference time — only MLX.

## Highlights

- **Text-to-image** — 4-step flow-matching Turbo inference at 1024×1024 (and arbitrary multiples of 16)
- **Image editing** — single and batch edit mode via the dedicated `Mage-Flow-Edit-Turbo` checkpoint
- **Auto weight download & conversion** — PyTorch BF16 weights from HuggingFace are auto-downloaded, converted to MLX BF16, and cached on first run
- **Persistent quantization** — 4-bit and 8-bit DiT variants cached separately; canonical BF16 checkpoint never modified
- **Batch worker mode** — JSONL prompt files with resident models, embedding cache, and vision cache
- **Phase profiler** — `--metadata` instruments every phase of generation with peak RSS tracking and crash-safe incremental writes (JSON + markdown)
- **Gaussian-Shading watermark** — 256-bit steganographic "MageFlow" message embedded in initial latents

## Requirements

- Mac: M1/M2/M3/M4/M5 series (Apple Silicon)
- RAM: 24 GB recommended
- Disk: ~17 GB for BF16 weights (+ HF cache); optional 4-bit (4.2 GB) / 8-bit (5.5 GB) quantized caches
- Python: 3.11+

## Quick Start

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install mlx mlx-lm safetensors torch huggingface_hub pillow numpy regex

python generate.py --prompt "A futuristic cityscape at sunset, photorealistic"
```

## Acknowledgements

This project is a port of Microsoft's [Mage](https://github.com/microsoft/Mage). See the original repository for the PyTorch/CUDA implementation and license.

The image editing mode is based on the Mage-Flow edit PR for [mflux](https://github.com/mfluxml/mflux) by [Ivan Fioravanti](https://github.com/ivanfioravanti).

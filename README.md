# Mage-Flow MLX

Native Apple Silicon port of Microsoft's **Mage-Flow** (4B MMDiT) using [MLX](https://github.com/ml-explore/mlx).

## Overview

Mage-Flow is a compact 4B-scale generative stack for text-to-image generation, built from:

- **Mage-VAE** — lightweight high-fidelity latent tokenizer (16× downsample, 128 latent channels)
- **NR-MMDiT** — 4B Native-Resolution Multimodal Diffusion Transformer (12 double-stream blocks, 2D multi-scale RoPE)
- **Qwen3-VL** — text encoder (2560 hidden, 36 layers, 32Q/8KV heads)

This port translates the PyTorch/CUDA implementation to native MLX, running entirely on Apple Silicon.

## Requirements

- **Mac**: M1/M2/M3/M4/M5 series (Apple Silicon)
- **RAM**: 24 GB recommended for the default 8-bit conversion
- **Python**: 3.11+

## Quick Start

```bash
# 1. Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# 2. Install dependencies
uv pip install mlx mlx-lm safetensors torch huggingface_hub pillow numpy regex

# 3. Convert weights (8-bit by default; recommended for image quality)
python convert_weights.py

# 4. Generate an image
python generate.py --prompt "A futuristic cityscape at sunset, photorealistic"
```

Generation defaults to four denoising steps and CFG 5. Use `--guidance 1` to disable CFG or `--negative-prompt "..."` to customize its unconditional branch.
If both legacy `models/mage_flow_mlx` (4-bit) and `models/mage_flow_mlx_8bit` directories exist, the CLI automatically selects the validated 8-bit model. Use `--model` to override this selection.

## Architecture

| Component | PyTorch (Mage-Flow) | MLX Port |
|---|---|---|
| **DiT** | 4B MMDiT, 128 latent ch, 3072 hidden, 24 heads, 12 blocks | MLX `nn.Module`, 8-bit quantized |
| **Text Encoder** | Qwen3-VL (2560 hidden, 36 layers) | mlx-lm Qwen3-VL, 8-bit quantized |
| **VAE** | MageVAE (DConvEncoder + DConvDenoiser + CoD) | MLX `nn.Conv2d` (NHWC) |
| **Scheduler** | FlowMatchEulerDiscrete (shift=6.0, 4 steps) | MLX port (mflux pattern) |

### Key MLX Translations

- **Conv2d weights**: `[Out, In, H, W]` → `[Out, H, W, In]` (NHWC layout)
- **2D RoPE**: Complex-number rotary embeddings with 3 axes (frame=16, height=56, width=56)
- **Joint attention**: Text+image tokens packed, single SDPA forward
- **Quantization**: 8-bit group quantization (group_size=64) for DiT + text encoder. The converter supports `--bits 4`, but naive 4-bit DiT quantization does not preserve acceptable generation quality.

## Memory Budget (24 GB MacBook Air M5)

| Component | 8-bit Quantized |
|---|---|
| DiT transformer | ~4.6 GB |
| Qwen3-VL text encoder | ~4.5 GB |
| MageVAE | ~0.5 GB |
| Activations + KV | ~2-3 GB |
| **Total** | **~12-15 GB** |

## Project Structure

```
mage-flow-mlx/
├── pyproject.toml          # Project configuration
├── convert_weights.py      # PyTorch → MLX conversion + quantization
├── generate.py             # CLI for text-to-image generation
├── mage_mlx/
│   ├── __init__.py         # Package exports
│   ├── rope.py             # 2D multi-scale RoPE (MageFlowEmbedRope)
│   ├── timestep.py         # Timestep embedding (qwen_proj style)
│   ├── dit.py              # MageFlow DiT (12 double-stream MMDiT blocks)
│   ├── text_encoder.py     # Qwen3-VL text encoder (mlx-lm)
│   ├── vae.py              # MageVAE (DConvEncoder + DConvDenoiser + CoD)
│   ├── scheduler.py        # FlowMatchEulerDiscreteScheduler
│   └── pipeline.py         # MageFlowPipeline (end-to-end)
└── README.md
```

## License

This project is a port of Microsoft's Mage-Flow. See the original [Mage repository](https://github.com/microsoft/Mage) for the original implementation and license.

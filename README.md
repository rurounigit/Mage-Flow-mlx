# Mage-Flow MLX

Native Apple Silicon port of Microsoft's **Mage-Flow** (4B MMDiT) using [MLX](https://github.com/ml-explore/mlx).

## Overview

Mage-Flow is a compact 4B-scale generative stack for text-to-image generation, built from:

- **Mage-VAE** — lightweight high-fidelity latent tokenizer (16× downsample, 128 latent channels)
- **NR-MMDiT** — 4B Native-Resolution Multimodal Diffusion Transformer (12 double_stream blocks, 2D multi-scale RoPE)
- **Qwen3-VL** — text encoder (2560 hidden, 36 layers, 32Q/8KV heads)

This port translates the PyTorch/CUDA implementation to native MLX, running entirely on Apple Silicon.

## Requirements

- **Mac**: M1/M2/M3/M4/M5 series (Apple Silicon)
- **RAM**: 24 GB recommended
- **Disk**: about 17 GB for converted BF16 weights, plus the Hugging Face cache;
  optional persistent DiT caches add about 4.2 GB (4-bit) and 5.5 GB (8-bit)
- **Python**: 3.11+

## Quick Start

```bash
# 1. Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# 2. Install dependencies
uv pip install mlx mlx-lm safetensors torch huggingface_hub pillow numpy regex

# 3. Generate an image (auto-downloads and converts weights on first run)
python generate.py --prompt "A futuristic cityscape at sunset, photorealistic"
```

The converted model preserves the original BF16 precision. Optional quantized
variants keep sensitive conditioning and final-block layers in BF16 and are
cached separately without modifying the canonical checkpoint.

### Automatic Model Download & Conversion

You no longer need to run `convert_weights.py` manually. When you run `generate.py`, the pipeline will:

1. Check if converted MLX weights exist locally (default: `models/mage_flow_mlx/`)
2. If missing, automatically download the raw PyTorch weights from HuggingFace (`microsoft/Mage-Flow-Turbo`)
3. Convert them to native MLX BF16 format on-the-fly
4. Cache the converted weights for instant startup on subsequent runs

When `--quantize 4` or `--quantize 8` is requested for the first time, the
pipeline derives and atomically saves a persistent packed DiT variant:

```text
models/mage_flow_mlx/transformer_quant4.safetensors
models/mage_flow_mlx/transformer_quant4.json
models/mage_flow_mlx/transformer_quant8.safetensors
models/mage_flow_mlx/transformer_quant8.json
```

Later runs construct the matching `nn.QuantizedLinear` module layout and load
the packed cache directly, skipping BF16 DiT loading and repeated quantization.
Metadata records the bit depth, group size, policy version, excluded sensitive
layers, and BF16 source signature. A stale or incompatible cache is rebuilt.

You can also explicitly pass a HuggingFace repo ID:

```bash
# Uses microsoft/Mage-Flow-Turbo (auto-downloads and converts on first run)
python generate.py --model microsoft/Mage-Flow-Turbo --prompt "..."

# Or use a local converted directory
python generate.py --model models/mage_flow_mlx --prompt "..."
```

If you prefer to pre-convert weights manually (e.g., for offline use or custom repos), you can still use:

```bash
python convert_weights.py --repo microsoft/Mage-Flow-Turbo --output models/mage_flow_mlx
```

## Generation parameters

```text
python generate.py [OPTIONS]
```

| Parameter | Default | Description |
|---|---:|---|
| `--prompt TEXT` | required | Description of the image to generate. Concrete subjects, composition, lighting, and style generally produce the most predictable result. |
| `--model PATH` | `models/mage_flow_mlx` | Directory containing converted MLX weights, or a HuggingFace repo ID (e.g. `microsoft/Mage-Flow-Turbo`). On first run, weights are auto-downloaded and converted. |
| `--steps INTEGER` | `4` | Number of flow-matching denoising steps. Mage-Flow-Turbo is trained for four steps; increasing this is not guaranteed to improve quality and changes the scheduler trajectory. |
| `--height INTEGER` | `1024` | Output height in pixels. Must be a positive multiple of 16. Higher resolutions require more unified memory and take longer. |
| `--width INTEGER` | `1024` | Output width in pixels. Must be a positive multiple of 16. Non-square aspect ratios are supported. |
| `--seed INTEGER` | `42` | Random seed used for initial latent noise. Reusing the same seed and parameters reproduces the same MLX result. |
| `--guidance FLOAT` | `1.0` | Classifier-free guidance (CFG) scale. `1.0` disables CFG and performs one DiT pass per step. Values above 1 strengthen prompt adherence but very high values can oversaturate or reduce natural detail. |
| `--negative-prompt TEXT` | one space (`" "`) | Text for the unconditional/negative CFG branch. It is only used when `--guidance` is greater than 1. |
| `--output PATH` | `output.png` | Destination image path. The format is inferred from the extension by Pillow. |
| `--quantize INT` | none | Use a persistent 4- or 8-bit DiT cache. The variant is created atomically on first use and reused afterward. The canonical checkpoint stays BF16; boundary, modulation, and the sensitive final image MLP expansion remain BF16 inside each mixed-precision variant. |


### Examples

```bash
# Default 1024×1024, four-step Turbo generation
python generate.py \
  --prompt "A futuristic cityscape at sunset, photorealistic"

# Portrait aspect ratio with a custom seed
python generate.py \
  --prompt "Editorial portrait of an astronaut, soft window light" \
  --width 768 --height 1024 --seed 123 \
  --output astronaut.png

# Use a negative prompt
python generate.py \
  --prompt "Product photograph of a wristwatch on black velvet" \
  --negative-prompt "blurry, distorted, text, watermark" \
```

## Architecture

| Component | PyTorch (Mage-Flow) | MLX Port |
|---|---|---|
| **DiT** | 4B MMDiT, 128 latent ch, 3072 hidden, 24 heads, 12 blocks | MLX `nn.Module`, BF16 |
| **Text Encoder** | Qwen3-VL (2560 hidden, 36 layers) | mlx-lm Qwen3-VL, BF16 with staged unloading |
| **VAE** | MageVAE (DConvEncoder + DConvDenoiser + CoD) | MLX `nn.Conv2d` (NHWC) |
| **Scheduler** | FlowMatchEulerDiscrete (shift=6.0, 4 steps) | MLX port (mflux pattern) |

### Key MLX Translations

- **Conv2d weights**: `[Out, In, H, W]` → `[Out, H, W, In]` (NHWC layout)
- **2D RoPE**: Complex-number rotary embeddings with 3 axes (frame=16, height=56, width=56)
- **Joint attention**: Text+image tokens packed, single SDPA forward
- **Precision**: Canonical weights remain BF16. Quantized variants are stored under distinct filenames and never overwrite the BF16 cache.

## Memory Budget (24 GB MacBook Air M5)

| Component | BF16 weights |
|---|---|
| DiT transformer | ~8.23 GB |
| Qwen3-VL text encoder | ~8.04 GB (unloaded before denoising) |
| MageVAE | ~0.28 GB |
| Activations and working memory | Resolution-dependent |

The text encoder and DiT briefly coexist while prompts are encoded. Afterward, the pipeline releases Qwen and clears the MLX cache before creating denoising activations. This staged policy allows BF16 inference on a 24 GB unified-memory Mac.

## Project Structure

```
mage-flow-mlx/
├── pyproject.toml          # Project configuration
├── convert_weights.py      # PyTorch BF16 → MLX BF16 conversion (manual, optional)
├── generate.py             # CLI for text-to-image generation
├── mage_mlx/
│   ├── __init__.py         # Package exports
│   ├── loader.py           # Auto-download & conversion with caching
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

# Mage-Flow MLX

Native Apple Silicon port of Microsoft's **Mage-Flow** (4B MMDiT) using [MLX](https://github.com/ml-explore/mlx), partly based on the Mage-Flow edit PR for [mflux](https://github.com/mfluxml/mflux) by [Ivan Fioravanti](https://github.com/ivanfioravanti). Rewired for low memory setups and batch processing.

Speed:

| Macbook Air M5 24GB |
|---|---|---|
| 17-18s | generation |
| 18-22s | editing |
| Peak RAM | <8GB |

| Parameter | Default | Description |
|---|---|---|
| `--prompt TEXT` | required | Description of the image to generate. Concrete subjects, composition, lighting, and style generally produce the most predictable result. |
| `--model PATH` | `models/microsoft_Mage-Flow-Turbo` | Directory containing converted MLX weights, or a HuggingFace repo ID (e.g. `microsoft/Mage-Flow-Turbo`). On first run, weights are auto-downloaded and converted. |
| `--steps INTEGER` | `4` | Number of flow-matching denoising steps. Mage-Flow-Turbo is trained for four steps; increasing this is not guaranteed to improve quality and changes the scheduler trajectory. |
| `--height INTEGER` | `1024` | Output height in pixels. Must be a positive multiple of 16. Higher resolutions require more unified memory and take longer. |
| `--width INTEGER` | `1024` | Output width in pixels. Must be a positive multiple of 16. Non-square aspect ratios are supported. |
| `--seed INTEGER` | `42` | Random seed used for initial latent noise. Reusing the same seed and parameters reproduces the same MLX result. |
| `--guidance FLOAT` | `1.0` | Classifier-free guidance (CFG) scale. `1.0` disables CFG and performs one DiT pass per step. Values above 1 strengthen prompt adherence but very high values can oversaturate or reduce natural detail. |
| `--negative-prompt TEXT` | one space (`" "`) | Text for the unconditional/negative CFG branch. It is only used when `--guidance` is greater than 1. |
| `--output PATH` | `none` | Output image path. Bare filename → `output/` subfolder; absolute path with filename → saved there; absolute path without filename → default name from metadata (resolution, steps, seed, quantize) + unique ID; omitted → default name in `output/` subfolder. Existing files are silently overwritten. |
| `--quantize INT` | none | Use a persistent 4- or 8-bit DiT cache. The variant is created atomically on first use and reused afterward. The canonical checkpoint stays BF16; boundary, modulation, and the sensitive final image MLP expansion remain BF16 inside each mixed-precision variant. |
| `--worker PATH` | none | Run in persistent JSONL worker mode. Models (DiT, VAE, tokenizer) stay resident across all prompts in the file. Uses prompt queue mode: Qwen is loaded once, all prompts are text-encoded, then Qwen is unloaded. Repeated prompts with the same text skip Qwen entirely via the embedding cache. |
| `--metadata` | none | Enable phase-level profiling, print terminal report, and save JSON + markdown files |

the json worker is even faster but in case of the Macbook Air especially, thermal throtteling will kick in after 4-6 edits (which brings down the speed).
You will see your current thermal status in the terminal output.


<table align="center" style="border-collapse: collapse; width: 100%; max-width: 1536px; table-layout: fixed;">
  <!-- Row 1: Images (Zero padding / margins) -->
  <tr>
     <td valign="top" style="padding: 0 0px; width: 33.33%; line-height: 0;">
        <a href="/examples/test_06_mars_diner_new.png" style="display: block; margin: 0; padding: 0;">
          <img style="width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; border: none; margin: 0;" src="/examples/test_06_mars_diner_new.png"/>
        </a>
     </td>
     <td valign="top" style="padding: 0 0px; width: 33.33%; line-height: 0;">
        <a href="/examples/test_02_cosmic_wolf_new.png" style="display: block; margin: 0; padding: 0;">
          <img style="width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; border: none; margin: 0;" src="/examples/test_02_cosmic_wolf_new.png"/>
        </a>
     </td>
     <td valign="top" style="padding: 0 0px; width: 33.33%; line-height: 0;">
        <a href="/examples/test_03_girl_new.png" style="display: block; margin: 0; padding: 0;">
          <img style="width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; border: none; margin: 0;" src="/examples/test_03_girl_new.png"/>
        </a>
     </td>
     <td valign="top" style="padding: 0 0px; width: 33.33%; line-height: 0;">
        <a href="/examples/test_07_old_man_new.png" style="display: block; margin: 0; padding: 0;">
          <img style="width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; border: none; margin: 0;" src="/examples/test_07_old_man_new.png"/>
        </a>
     </td>
  </tr>
  <!-- Row 2: Prompts (Tightly attached underneath) -->
  <tr>
     <td valign="top" style="padding: 4px 5px 0 5px; line-height: 1.3; font-size: 14px;">"A retro-futuristic diner on Mars, neon signs, red planet landscape visible through windows, 1950s sci-fi movie poster style, the name of the diner is 'mage'", seed: 47
     </td>
     <td valign="top" style="padding: 4px 5px 0 5px; line-height: 1.3; font-size: 14px;">"Anime shonen style, an astronaut riding a cosmic wolf through a nebula, bioluminescent fur, star trails, ethereal purple and blue lighting", seed: 43
     </td>
     <td valign="top" style="padding: 4px 5px 0 5px; line-height: 1.3; font-size: 14px;">"90s selfie grunge vice a portrait of a punk girl sitting on an old sofa", seed: 100
     </td>
     <td valign="top" style="padding: 4px 5px 0 5px; line-height: 1.3; font-size: 14px;">"A close-up portrait of an elderly African man with deep wrinkles, wearing a traditional hat, soft natural lighting, ultra realistic.", seed: 42
     </td>
  </tr>
</table>


## Requirements

- **Mac:** M1/M2/M3/M4/M5 series (Apple Silicon)
- **RAM:** 24 GB recommended
- **Disk:** about 17 GB for converted BF16 weights, plus the Hugging Face cache;
  optional persistent DiT caches add about 4.2 GB (4-bit) and 5.5 GB (8-bit)
- **Python:** 3.11+

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/rurounigit/Mage-Flow-mlx.git
cd Mage-Flow-mlx

# 2. Install dependencies (creates .venv automatically)
uv sync

# 3. Generate an image (auto-downloads and converts weights on first run)
uv run python generate.py --prompt "A futuristic cityscape at sunset, photorealistic"
```

Discover all options:

```bash
uv run python generate.py --help
```

Or use the installed console script:

```bash
uv run mage-flow-generate --prompt "A futuristic cityscape at sunset, photorealistic"
```

The converted model preserves the original BF16 precision. Optional quantized
variants keep sensitive conditioning and final-block layers in BF16 and are
cached separately without modifying the canonical checkpoint.

### Automatic Model Download & Conversion

You do not need to run `convert_weights.py` manually. When you run `generate.py`, the pipeline will:

1. Check if converted MLX weights exist locally (default: `models/microsoft_Mage-Flow-Turbo/`)
2. If missing, automatically download the raw PyTorch weights from HuggingFace (`microsoft/Mage-Flow-Turbo`)
3. Convert them to native MLX BF16 format on-the-fly
4. Cache the converted weights for instant startup on subsequent runs

When `--quantize 4` or `--quantize 8` is requested for the first time, the
pipeline derives and atomically saves a persistent packed DiT variant:

```text
models/microsoft_Mage-Flow-Turbo/transformer_quant4.safetensors
models/microsoft_Mage-Flow-Turbo/transformer_quant4.json
models/microsoft_Mage-Flow-Turbo/transformer_quant8.safetensors
models/microsoft_Mage-Flow-Turbo/transformer_quant8.json
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
python generate.py --model models/microsoft_Mage-Flow-Turbo --prompt "..."
```

If you prefer to pre-convert weights manually (e.g., for offline use or custom repos), you can still use:

```bash
python convert_weights.py --repo microsoft/Mage-Flow-Turbo --output models/microsoft_Mage-Flow-Turbo
```

## Generation Parameters

```text
python generate.py [OPTIONS]
```

| Parameter | Default | Description |
|---|---|---|
| `--prompt TEXT` | required | Description of the image to generate. Concrete subjects, composition, lighting, and style generally produce the most predictable result. |
| `--model PATH` | `models/microsoft_Mage-Flow-Turbo` | Directory containing converted MLX weights, or a HuggingFace repo ID (e.g. `microsoft/Mage-Flow-Turbo`). On first run, weights are auto-downloaded and converted. |
| `--steps INTEGER` | `4` | Number of flow-matching denoising steps. Mage-Flow-Turbo is trained for four steps; increasing this is not guaranteed to improve quality and changes the scheduler trajectory. |
| `--height INTEGER` | `1024` | Output height in pixels. Must be a positive multiple of 16. Higher resolutions require more unified memory and take longer. |
| `--width INTEGER` | `1024` | Output width in pixels. Must be a positive multiple of 16. Non-square aspect ratios are supported. |
| `--seed INTEGER` | `42` | Random seed used for initial latent noise. Reusing the same seed and parameters reproduces the same MLX result. |
| `--guidance FLOAT` | `1.0` | Classifier-free guidance (CFG) scale. `1.0` disables CFG and performs one DiT pass per step. Values above 1 strengthen prompt adherence but very high values can oversaturate or reduce natural detail. |
| `--negative-prompt TEXT` | one space (`" "`) | Text for the unconditional/negative CFG branch. It is only used when `--guidance` is greater than 1. |
| `--output PATH` | `none` | Output image path. Bare filename → `output/` subfolder; absolute path with filename → saved there; absolute path without filename → default name from metadata (resolution, steps, seed, quantize) + unique ID; omitted → default name in `output/` subfolder. Existing files are silently overwritten. |
| `--quantize INT` | none | Use a persistent 4- or 8-bit DiT cache. The variant is created atomically on first use and reused afterward. The canonical checkpoint stays BF16; boundary, modulation, and the sensitive final image MLP expansion remain BF16 inside each mixed-precision variant. |
| `--worker PATH` | none | Run in persistent JSONL worker mode. Models (DiT, VAE, tokenizer) stay resident across all prompts in the file. Uses prompt queue mode: Qwen is loaded once, all prompts are text-encoded, then Qwen is unloaded. Repeated prompts with the same text skip Qwen entirely via the embedding cache. |
| `--metadata` | none | Enable phase-level profiling, print terminal report, and save JSON + markdown files |

### Examples

```bash
# Default 1024×1024, seed 42, four-step Turbo generation
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
  --negative-prompt "blurry, distorted, text, watermark"
```

### Persistent Worker Mode (Batch Generation)

For generating multiple images, use the JSONL worker mode. Models (DiT, VAE, tokenizer) stay resident across all prompts, and Qwen is loaded once to encode all prompts before being unloaded. Repeated prompts with the same text skip Qwen entirely via the embedding cache.

```bash
# Create a JSONL prompts file
cat > prompts.jsonl << 'EOF'
{"prompt": "A serene mountain landscape at sunset", "seed": 42, "output": "mountain.png"}
{"prompt": "A futuristic cityscape at night", "seed": 43, "output": "city.png"}
{"prompt": "A serene mountain landscape at sunset", "seed": 44, "output": "mountain_v2.png"}
EOF

# Run the worker with profiling
python generate.py --worker prompts.jsonl --metadata
```

JSONL format (one JSON object per line):

| Field | Required | Description |
|---|---|---|
| `prompt` | yes | Text prompt |
| `output` | no | Output path (resolved at save time; bare filename → output/ subfolder, omitted → default name in output/) |
| `seed` | no | Random seed (default: 42) |
| `guidance` | no | CFG scale (default: 1.0) |
| `width` | no | Output width (default: 1024) |
| `height` | no | Output height (default: 1024) |
| `steps` | no | Denoising steps (default: 4) |
| `negative_prompt` | no | Negative prompt (default: " ") |

## Image Editing

### Single Edit

```bash
python generate.py \
  --prompt "change the shoe to burgundy leather" \
  --image test_10_shoe.png \
  --output test_10_shoe_edited.png
```

Providing `--image` automatically selects the dedicated `microsoft/Mage-Flow-Edit-Turbo` checkpoint. On first use, the loader downloads and converts it to a cached MLX directory; subsequent runs reuse that cache.

### Edit Worker Mode

<table align="center" style="border-collapse: collapse; width: 100%; max-width: 1536px; table-layout: fixed;">
  <!-- Row 1: Images (Zero padding / margins) -->
  <tr>
     <td valign="top" style="padding: 0 0px; width: 33.33%; line-height: 0;">
        <a href="/examples/test_10_shoe.png" style="display: block; margin: 0; padding: 0;">
          <img style="width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; border: none; margin: 0;" src="/examples/test_10_shoe.png"/>
        </a>
     </td>
     <td valign="top" style="padding: 0 0px; width: 33.33%; line-height: 0;">
        <a href="/examples/test_10_shoe_edited.png" style="display: block; margin: 0; padding: 0;">
          <img style="width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; border: none; margin: 0;" src="/examples/test_10_shoe_edited.png"/>
        </a>
     </td>
     <td valign="top" style="padding: 0 0px; width: 33.33%; line-height: 0;">
        <a href="/examples/test_10_shoe_edited_worker_01.png" style="display: block; margin: 0; padding: 0;">
          <img style="width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; border: none; margin: 0;" src="/examples/test_10_shoe_edited_worker_01.png"/>
        </a>
     </td>
     <td valign="top" style="padding: 0 0px; width: 33.33%; line-height: 0;">
        <a href="/examples/test_10_shoe_edited_worker_02.png" style="display: block; margin: 0; padding: 0;">
          <img style="width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; border: none; margin: 0;" src="/examples/test_10_shoe_edited_worker_02.png"/>
        </a>
     </td>
  </tr>
  <!-- Row 2: Prompts (Tightly attached underneath) -->
  <tr>
     <td valign="top" style="padding: 4px 5px 0 5px; line-height: 1.3; font-size: 14px;">"An unbranded futuristic running shoe made from white technical mesh with a vivid orange sole, floating above a pale gray studio surface, dramatic softbox lighting, premium product photography, photorealistic.", seed: 42
     </td>
     <td valign="top" style="padding: 4px 5px 0 5px; line-height: 1.3; font-size: 14px;">"change the shoe to deep burgundy polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.", seed: 43
     </td>
     <td valign="top" style="padding: 4px 5px 0 5px; line-height: 1.3; font-size: 14px;">"change the shoe to metallic green mesh with a translucent  sole and subtle fur details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.", seed: 42
     </td>
     <td valign="top" style="padding: 4px 5px 0 5px; line-height: 1.3; font-size: 14px;">"change the shoe to deep blue polished leather with a translucent smoke-gray sole and subtle chrome details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.", seed: 43
     </td>
  </tr>
</table>

<table align="center" style="border-collapse: collapse; width: 100%; max-width: 1536px; table-layout: fixed;">
  <!-- Row 1: Images (Zero padding / margins) -->
  <tr>
     <td valign="top" style="padding: 0 0px; width: 33.33%; line-height: 0;">
        <a href="/examples/test_10_shoe_edited_worker_03.png" style="display: block; margin: 0; padding: 0;">
          <img style="width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; border: none; margin: 0;" src="/examples/test_10_shoe_edited_worker_03.png"/>
        </a>
     </td>
     <td valign="top" style="padding: 0 0px; width: 33.33%; line-height: 0;">
        <a href="/examples/test_10_shoe_edited_worker_04.png" style="display: block; margin: 0; padding: 0;">
          <img style="width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; border: none; margin: 0;" src="/examples/test_10_shoe_edited_worker_04.png"/>
        </a>
     </td>
     <td valign="top" style="padding: 0 0px; width: 33.33%; line-height: 0;">
        <a href="/examples/test_10_shoe_edited_worker_05.png" style="display: block; margin: 0; padding: 0;">
          <img style="width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; border: none; margin: 0;" src="/examples/test_10_shoe_edited_worker_05.png"/>
        </a>
     </td>
     <td valign="top" style="padding: 0 0px; width: 33.33%; line-height: 0;">
        <a href="/examples/test_10_shoe_edited_worker_08.png" style="display: block; margin: 0; padding: 0;">
          <img style="width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; border: none; margin: 0;" src="/examples/test_10_shoe_edited_worker_08.png"/>
        </a>
     </td>
  </tr>
  <!-- Row 2: Prompts (Tightly attached underneath) -->
  <tr>
     <td valign="top" style="padding: 4px 5px 0 5px; line-height: 1.3; font-size: 14px;">"change the shoe to a matte black finish with a red-orange translucent sole and small green accents. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.", seed: 44
     </td>
     <td valign="top" style="padding: 4px 5px 0 5px; line-height: 1.3; font-size: 14px;">"change the shoe to beige canvas with a solid white sole and navy blue stitching. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.", seed: 45
     </td>
     <td valign="top" style="padding: 4px 5px 0 5px; line-height: 1.3; font-size: 14px;">"change the shoe to full gray suede with an off-white sole and silver eyelets. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.", seed: 46
     </td>
     <td valign="top" style="padding: 4px 5px 0 5px; line-height: 1.3; font-size: 14px;">"change the shoe to light pink woven textile with a translucent pink sole and white embroidered details. Preserve the exact silhouette, panel seams, laces, camera angle, floating pose, lighting, shadows, and background.", seed: 49
     </td>
  </tr>
</table>

```bash
python generate.py --worker prompts_edit.jsonl --edit
```

The edit worker mirrors the txt2img worker but with image validation, reference image hashing, and vision cache integration. Each JSONL line requires a `prompt` field and either an `image` field (target image to edit) or a `ref_images` field (list of reference image paths).

## Architecture

Mage-Flow is a compact 4B-parameter generative stack for text-to-image generation and image editing, built from:

- **Mage-VAE** — lightweight high-fidelity latent tokenizer (16× downsample, 128 latent channels)
- **NR-MMDiT** — 4B Native-Resolution Multimodal Diffusion Transformer (12 double_stream blocks, 2D multi-scale RoPE)
- **Qwen3-VL** — text encoder (2560 hidden, 36 layers, 32Q/8KV heads)

### Component Mapping

| Component | PyTorch (Mage-Flow) | MLX Port |
|---|---|---|
| **DiT** | 4B MMDiT, 128 latent ch, 3072 hidden, 24 heads, 12 blocks | MLX `nn.Module`, BF16 |
| **Text Encoder** | Qwen3-VL (2560 hidden, 36 layers) | Native MLX Qwen3-VL, BF16 with staged unloading |
| **VAE** | MageVAE (DConvEncoder + DConvDenoiser + CoD) | MLX `nn.Conv2d` (NHWC) |
| **Scheduler** | FlowMatchEulerDiscrete (shift=6.0, 4 steps) | MLX port (mflux pattern) |

### Key MLX Translations

- **Conv2d weights**: `[Out, In, H, W]` → `[Out, H, W, In]` (NHWC layout)
- **2D RoPE**: Complex-number rotary embeddings with 3 axes (frame=16, height=56, width=56)
- **Joint attention**: Text+image tokens packed, single SDPA forward
- **Precision**: Canonical weights remain BF16. Quantized variants are stored under distinct filenames and never overwrite the BF16 cache.

### DiT (4B MMDiT)

The Diffusion Transformer (`mage_mlx/dit.py`) is a 12-block double-stream MMDiT:

- **Input projection**: 128-dim latent → 3072-dim hidden via `img_in` (Linear)
- **Text projection**: 2560-dim → 3072-dim via `txt_norm` (RMSNorm) + `txt_in` (Linear)
- **Timestep embedding**: 256-dim sinusoidal → 3072-dim via SiLU + Linear (qwen_proj style)
- **Each double-stream block**:
  - Adaptive modulation: `img_mod` / `txt_mod` produce 6×3072 gate/scale/shift vectors
  - Joint attention: text and image tokens packed into `[text, image]` order, single SDPA forward
    - Per-head RMSNorm on Q/K (LayerNorm would destroy the model)
    - 2D RoPE applied to image tokens only
  - Feed-forward: Linear → GELU(tanh) → Linear
- **Output**: `AdaLayerNormContinuous` + `proj_out` → 128-dim velocity prediction

### VAE (MageVAE)

The VAE (`mage_mlx/vae.py`) is a lightweight high-fidelity latent tokenizer with three sub-components:

1. **DConvEncoder** (21 DiCoBlocks): image → (mean, logvar) latent at 1/16 resolution
2. **DConvDenoiser** (24 blocks): latent + zero noise → reconstructed image
3. **CoD Decoder**: latent → conditioning features for the denoiser

All Conv2d weights are transposed to NHWC layout. GroupNorm statistics are computed in FP32 even when activations are BF16.

### Text Encoder (Native MLX Qwen3-VL)

The text encoder (`mage_mlx/text_encoder.py`) is a native MLX implementation of Qwen3-VL, replacing the previous mlx-lm dependency:

- 36-layer transformer, 2560 hidden, 32 query heads / 8 KV heads (GQA)
- **mRoPE** (multi-dimensional RoPE) with 3D position IDs (temporal, height, width)
- DeepStack: visual features injected into the language model at layers 0, 1, 2
- Supports text-only encoding (txt2img) and multi-modal encoding (edit)

The vision tower (`mage_mlx/vision_model.py`) is a 24-layer vision transformer with Conv3d patch embedding, 2D vision RoPE, and spatial patch merging. It is lazily constructed only when edit mode needs it.

### Scheduler

The scheduler (`mage_mlx/scheduler.py`) is a FlowMatchEulerDiscrete with static shift=6.0:

- Linear sigmas: `linspace(1.0, 1/num_steps, num_steps)`
- Static rational time-shift: `sigma = shift * t / (1 + (shift - 1) * t)`
- Euler step: `x_{t+1} = x_t + (sigma_{t+1} - sigma_t) * v_t`
- The Euler step is compiled with `mx.compile` for performance

### 2D Multi-Scale RoPE

The RoPE module (`mage_mlx/rope.py`) uses complex-number rotary embeddings with three axes: frame (16), height (56), width (56), totaling head_dim=128.

**Optimization**: cos/sin tensors are cached by (frame, height, width, idx) so they are computed once per resolution and reused across all 12 blocks and all 4 denoising steps. This avoids 96 redundant cos/sin evaluations per generation.

### Timestep Embedding

The timestep embedding (`mage_mlx/timestep.py`) uses a 256-dim sinusoidal embedding with **BF16 rounding**. This is critical: the model was trained with this exact BF16 rounding, so using FP32 would produce slightly different embeddings and degrade output quality.

### Latent Creator (Gaussian-Shading Watermark)

The latent creator (`mage_mlx/latent_creator.py`) generates Mage's 128-channel initial latents with a steganographic watermark — a 256-bit message ("MageFlow") encoded into the noise pattern using a keyed PRNG. The watermark can be verified via `decode_gaussian_shading()` which returns a z-score and p-value.

## Pipeline & Memory Management

### Staged Loading

The pipeline (`mage_mlx/pipeline.py`) uses a **staged loading policy** to keep peak RAM under 24 GB:

```
Phase 1: Load Qwen (~8 GB) → Encode prompts → Unload Qwen
Phase 2: Load DiT + VAE (~7.9 GB) → Denoise + Decode
```

Peak RAM = max(Qwen, DiT+VAE) ≈ 8.0 GB instead of Qwen + DiT + VAE ≈ 15.4 GB.

Three loading modes are available:
- `from_pretrained()` — Full load (DiT + VAE + Text Encoder + Tokenizer)
- `from_pretrained_text_encoder()` — Text encoder only (DiT/VAE deferred)
- `load_dit_vae()` — Lazy DiT + VAE load after Qwen is unloaded

### Generation Methods

- `generate()` — Full pipeline: encode → unload Qwen → load DiT/VAE → denoise → decode
- `_generate_from_embeds()` — Bypass text encoding entirely; used by the worker when embeddings are cached or pre-encoded in batch. Eliminates allocation churn that made the first DiT step of subsequent prompts 2-4× slower.

## Quantization

Runtime quantization with quality-safe layer selection:

- **Policy** (`should_quantize_dit_layer`): selects DiT Linear layers for quantization
  - Must be in `transformer_blocks.*`
  - Must have `in_features >= 32` and `in_features % 32 == 0`
  - **Excludes**: conditioning projections (`.img_mod`, `.txt_mod`) and the final image MLP expansion (`transformer_blocks.11.img_mlp.fc1`) — quantizing this alone causes ~50% relative error
- **Persistent cache**: 4-bit and 8-bit variants stored as packed safetensors + metadata JSON
- **Cache validation**: checks metadata (policy version, bits, group size, base checkpoint signature) + representative tensor layouts
- **Atomic writes**: temp file + `os.replace()` for crash safety
- **Canonical BF16 checkpoint is never modified**

Usage:

```bash
python generate.py --prompt "..." --quantize 4   # 4-bit
python generate.py --prompt "..." --quantize 8   # 8-bit
```

The first run with `--quantize N` will quantize the DiT, save the packed cache, and print how many layers were quantized. Subsequent runs load the cache directly.

## Caching Systems

### Embedding Cache

Text embeddings are cached on disk (~240 KB each vs. 8 GB Qwen weights).

**Cache key** (SHA-256):
- Formatted prompt text (with chat template applied)
- Negative prompt text
- Text-encoder checkpoint signature (size + mtime)
- Tokenizer/template version
- Reference image hashes (for edit mode — multimodal embeddings depend on both prompt and images)

**Design note**: The generation seed is intentionally excluded from the cache key — text embeddings are seed-independent (the seed only affects DiT latent initialization, not text encoding), so including it would create duplicate cache entries for the same prompt with different seeds.

For a cache hit, Qwen loading and text encoding are skipped entirely. This is especially useful when testing seeds, resolutions, quantization levels, or scheduler changes with the same prompt.

Cache files are stored in `models/microsoft_Mage-Flow-Turbo/embedding_cache/`.

### Vision Cache

VAE-encoded reference image latents are cached on disk (~10 MB each vs. ~1 GB VAE weights).

**Cache key** (SHA-256):
- Raw image bytes hash (SHA-256)
- Image pixel dimensions
- VAE checkpoint signature (size + mtime)
- Generation seed (Mage-Flow samples the VAE posterior with the generation seed)

Cache files are stored in `models/microsoft_Mage-Flow-Edit-Turbo/vision_cache/`.

## Profiling & Metadata

### Phase-Level Profiler

The `--metadata` flag instruments every phase of generation:

- Python/import startup
- DiT load, VAE load, text encoder load
- Text encoding, Qwen unload
- Each DiT step, VAE decode, PNG save
- Total wall-clock time

The profiler tracks peak RSS (max of process RSS and MLX device memory) and supports incremental saves — metadata files are written after every phase so a crash mid-run still leaves a partial report on disk.

### Metadata Output

With `--metadata`, the profiler saves two files alongside the output image:

- `{base}.json` — structured data (metadata, phases, overview, summary, log)
- `{base}.md` — markdown matching the terminal output structure

The JSON structure:

```json
{
  "metadata": {
    "model": "microsoft/Mage-Flow-Turbo",
    "base_model": "MageFlowTransformer",
    "generation_time_seconds": 12.3,
    "created_at": "2026-07-29T20:55:00",
    "image_path": "output/image.png",
    "peak_memory_gib": 7.92
  },
  "phases": [
    {"name": "python_startup", "elapsed": 0.5, "peak_rss_gib": 0.3, ...},
    {"name": "pipeline_load", "elapsed": 1.2, "peak_rss_gib": 8.0, ...},
    ...
  ],
  "summary": {
    "total_time": 12.3,
    "peak_ram": 7.92,
    "prompts_count": 1
  }
}
```

### Terminal Output

The profiler renders a real-time terminal report with:
- Cyan bold separators and headers
- Phase table (Phase / Time / Peak RAM / Saved File / Metadata)
- Per-prompt summary table with relative time coloring (green = fast, red = slow)
- Run Metadata block at the end

In non-verbose mode (without `--metadata`), a progress bar is shown instead.

## Output Path Resolution

All generation modes (single, edit, worker, edit worker) use a unified output path resolver (`mage_mlx/output_resolver.py`):

1. **Bare filename** (e.g. `"image.png"`) — save into the `output/` subfolder
2. **Absolute path with filename** (e.g. `"/tmp/img.png"`) — save there
3. **Absolute path without filename** (e.g. `"/tmp/"`) — construct a default filename from metadata (resolution, steps, seed, quantization) plus a short unique identifier
4. **No output** (`None`) — same as case 3 but in the `output/` subfolder

In every case, if a file with the resolved name already exists it is silently overwritten. The target directory is created if it does not exist.

Metadata files are saved alongside the output image: `{base}.json` and `{base}.md`.

## Project Structure

```
mage-flow-mlx/
├── pyproject.toml          # Project configuration (uv)
├── generate.py             # CLI for text-to-image generation and editing
├── convert_weights.py      # PyTorch BF16 → MLX BF16 conversion (manual, optional)
├── conftest.py             # Project configuration
├── mage_mlx/
│   ├── __init__.py         # Package exports
│   ├── loader.py           # Auto-download & conversion with caching
│   ├── pipeline.py         # MageFlowPipeline (end-to-end orchestration)
│   ├── dit.py              # MageFlow DiT (12 double-stream MMDiT blocks)
│   ├── vae.py              # MageVAE (DConvEncoder + DConvDenoiser + CoD)
│   ├── text_encoder.py     # Native MLX Qwen3-VL text encoder
│   ├── vision_model.py     # Qwen3-VL vision tower (for edit)
│   ├── scheduler.py        # FlowMatchEulerDiscreteScheduler
│   ├── rope.py             # 2D multi-scale RoPE (MageFlowEmbedRope)
│   ├── timestep.py         # Sinusoidal timestep embedding (qwen_proj style)
│   ├── latent_creator.py   # Gaussian-Shading watermarked noise
│   ├── prompt_processor.py # Shared prompt templates & hidden-state processing
│   ├── processor.py        # Qwen3-VL multi-modal tokenizer/image processor
│   ├── embedding_cache.py  # Prompt embedding cache
│   ├── vision_cache.py     # VAE reference latent cache
│   ├── output_resolver.py  # Unified output path resolution
│   ├── profiler.py         # Phase-level timing and memory profiler
│   ├── worker.py           # Persistent JSONL worker (txt2img + edit)
│   └── mflux_src/          # Ported mflux library (edit pipeline internals)
├── memlog/                 # Development log
└── output/                 # Generated images and metadata
```

## Memory Budget (24 GB MacBook Air M5)

| Component | BF16 weights |
|---|---|
| DiT transformer | ~8.23 GB |
| Qwen3-VL text encoder | ~8.04 GB (unloaded before denoising) |
| MageVAE | ~0.28 GB |
| Activations and working memory | Resolution-dependent |

The text encoder and DiT briefly coexist while prompts are encoded. Afterward, the pipeline releases Qwen and clears the MLX cache before creating denoising activations. This staged policy allows BF16 inference on a 24 GB unified-memory Mac.

## License

This project is a port of Microsoft's Mage-Flow. See the original [Mage repository](https://github.com/microsoft/Mage) for the original implementation and license.

The image editing mode is based on the Mage-Flow edit PR for [mflux](https://github.com/mfluxml/mflux) by [Ivan Fioravanti](https://github.com/ivanfioravanti).

## Development Log

The `memlog/` directory contains a detailed development log documenting the iterative optimization process, including:

- Generation debugging and optimization sets
- Phase profiler development
- Edit pipeline implementation and audit
- Worker profiler deduplication
- Metadata output rewrites
- Seed tracking and cache key management
- Terminal output unification
- Vision cache instrumentation
- Edit worker implementation and bugfixes

"""MageFlowPipeline: End-to-end text-to-image generation on MLX.

Orchestrates the text encoder (Qwen3-VL), DiT (4B MMDiT), VAE (MageVAE),
and scheduler (FlowMatchEulerDiscrete) to generate images from text prompts.

Usage:
    from mage_mlx import MageFlowPipeline
    pipeline = MageFlowPipeline.from_pretrained("models/mage_flow_mlx")
    image = pipeline.generate("A futuristic cityscape at sunset")
"""

from __future__ import annotations

import gc
import json
import os
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .dit import MageFlow, MageFlowParams
from .loader import _read_safetensors_header, ensure_mlx_model
from .processor import MageFlowQwen3VLProcessor
from .scheduler import FlowMatchEulerDiscreteScheduler
from .text_encoder import MageFlowTextEncoder
from .vae import MageVAE


class MageFlowTokenizer:
    """Tokenizer wrapper exposing ``tokenizer`` and ``processor`` attributes.

    The ``tokenizer`` attribute is the raw HuggingFace tokenizer used for
    text-only encoding (txt2img). The ``processor`` attribute is the
    :class:`MageFlowQwen3VLProcessor` used for multi-modal encoding (edit).
    """

    def __init__(self, tokenizer, processor: MageFlowQwen3VLProcessor | None = None):
        self.tokenizer = tokenizer
        self.processor = processor if processor is not None else MageFlowQwen3VLProcessor(tokenizer)



QUANTIZATION_POLICY_VERSION = 1
QUANTIZATION_GROUP_SIZE = 32
SUPPORTED_QUANTIZATION_BITS = (4, 8)

# Valid cleanup strategies for text encoder unload
CLEANUP_STRATEGIES = ("unload_only", "unload+gc", "unload+cache", "all_three")


def should_quantize_dit_layer(path: str, module: nn.Module) -> bool:
    """Select quality-safe DiT layers for runtime weight quantization."""
    if not isinstance(module, nn.Linear):
        return False

    in_features = module.weight.shape[1]
    shape_is_supported = in_features >= 32 and in_features % 32 == 0
    is_transformer_block = path.startswith("transformer_blocks.")
    is_conditioning_projection = ".img_mod" in path or ".txt_mod" in path
    is_sensitive_final_image_mlp = (
        path == "transformer_blocks.11.img_mlp.fc1"
    )

    # The final image-stream fc1 is uniquely sensitive: quantizing it alone
    # causes approximately 50% relative error in the final prediction.
    return (
        shape_is_supported
        and is_transformer_block
        and not is_conditioning_projection
        and not is_sensitive_final_image_mlp
    )


def _quantized_cache_paths(model_dir: str, bits: int) -> tuple[str, str]:
    """Return packed-weight and metadata paths for a quantization level."""
    stem = os.path.join(model_dir, f"transformer_quant{bits}")
    return f"{stem}.safetensors", f"{stem}.json"


def _base_checkpoint_signature(path: str) -> dict[str, int]:
    """Return a cheap signature used to invalidate derived quantized caches."""
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _expected_quantization_metadata(base_path: str, bits: int) -> dict:
    """Build metadata that uniquely identifies the current packed layout."""
    return {
        "format": "mage-flow-mlx-quantized-transformer",
        "policy_version": QUANTIZATION_POLICY_VERSION,
        "bits": bits,
        "group_size": QUANTIZATION_GROUP_SIZE,
        "mode": "affine",
        "base_checkpoint": _base_checkpoint_signature(base_path),
        "excluded_layers": ["transformer_blocks.11.img_mlp.fc1"],
    }


def _is_valid_quantized_cache(
    weights_path: str,
    metadata_path: str,
    expected_metadata: dict,
) -> bool:
    """Validate metadata and representative packed/BF16 tensor layouts."""
    if not os.path.exists(weights_path) or not os.path.exists(metadata_path):
        return False
    try:
        with open(metadata_path) as f:
            if json.load(f) != expected_metadata:
                return False
        header = _read_safetensors_header(weights_path)
        packed = header.get("transformer_blocks.0.attn.to_q.weight", {})
        packed_scales = header.get("transformer_blocks.0.attn.to_q.scales", {})
        excluded = header.get("transformer_blocks.11.img_mlp.fc1.weight", {})
        packed_width = 3072 * expected_metadata["bits"] // 32
        return (
            packed.get("dtype") == "U32"
            and packed.get("shape") == [3072, packed_width]
            and packed_scales.get("dtype") in {"BF16", "F16", "F32"}
            and excluded.get("dtype") in {"BF16", "F16", "F32"}
            and excluded.get("shape") == [12288, 3072]
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _quantize_transformer(transformer: MageFlow, bits: int) -> int:
    """Replace quality-safe DiT Linear layers with QuantizedLinear layers."""
    nn.quantize(
        transformer,
        bits=bits,
        group_size=QUANTIZATION_GROUP_SIZE,
        class_predicate=should_quantize_dit_layer,
    )
    return sum(
        isinstance(module, nn.QuantizedLinear)
        for _, module in transformer.named_modules()
    )


def _save_quantized_cache(
    transformer: MageFlow,
    weights_path: str,
    metadata_path: str,
    metadata: dict,
) -> None:
    """Atomically save packed model weights followed by compatibility metadata."""
    weights_root, weights_extension = os.path.splitext(weights_path)
    weights_temp = f"{weights_root}.tmp{weights_extension}"
    metadata_temp = f"{metadata_path}.tmp"
    transformer.save_weights(weights_temp)
    os.replace(weights_temp, weights_path)
    with open(metadata_temp, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    os.replace(metadata_temp, metadata_path)


class MageFlowPipeline:
    """End-to-end Mage-Flow text-to-image pipeline for MLX.

    Args:
        transformer: MageFlow DiT model
        vae: MageVAE model
        text_encoder: Qwen3-VL text encoder
        num_steps: Number of denoising steps (4 for turbo)
    """

    def __init__(
        self,
        transformer: MageFlow,
        vae: MageVAE,
        text_encoder: MageFlowTextEncoder,
        tokenizer: "MageFlowTokenizer | None" = None,
        num_steps: int = 4,
    ):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=6.0,
            num_inference_steps=num_steps,
        )
        self.num_steps = num_steps


    @classmethod
    def from_pretrained(
        cls,
        model_dir: str = "models/mage_flow_mlx",
        num_steps: int = 4,
        quantize: int | None = None,
        profiler: Optional["object"] = None,
    ) -> "MageFlowPipeline":
        """Load a Mage-Flow MLX pipeline from a directory or HF repo ID.

        Args:
            model_dir: Directory containing converted MLX weights or HF repo ID
            num_steps: Number of denoising steps
            quantize: If set (4 or 8), quantize transformer weights to N bits
            profiler: Optional Profiler instance for phase-level timing

        Returns:
            MageFlowPipeline instance
        """
        if quantize is not None and quantize not in SUPPORTED_QUANTIZATION_BITS:
            raise ValueError(
                "quantize must be None, 4, or 8; "
                f"received {quantize}"
            )
        model_dir, actual_quantize = ensure_mlx_model(model_dir, quantize=quantize)

        # Load DiT config
        config_path = os.path.join(model_dir, "transformer_config.json")
        with open(config_path) as f:
            dit_config = json.load(f)

        params = MageFlowParams(
            in_channels=dit_config.get("in_channels", 128),
            out_channels=dit_config.get("out_channels", 128),
            context_in_dim=dit_config.get("context_in_dim", 2560),
            hidden_size=dit_config.get("hidden_size", 3072),
            num_heads=dit_config.get("num_heads", 24),
            depth=dit_config.get("depth", 12),
            axes_dim=dit_config.get("axes_dim", [16, 56, 56]),
            patch_size=dit_config.get("patch_size", 1),
        )
        transformer = MageFlow(params)

        # Load canonical BF16 or a compatible persistent packed cache.
        dit_weights_path = os.path.join(model_dir, "transformer.safetensors")
        if actual_quantize in SUPPORTED_QUANTIZATION_BITS:
            quantized_path, quantized_metadata_path = _quantized_cache_paths(
                model_dir, actual_quantize
            )
            expected_metadata = _expected_quantization_metadata(
                dit_weights_path, actual_quantize
            )
            cache_is_valid = _is_valid_quantized_cache(
                quantized_path,
                quantized_metadata_path,
                expected_metadata,
            )
            if cache_is_valid:
                if profiler:
                    profiler.start("dit_load")
                quantized_layers = _quantize_transformer(
                    transformer, actual_quantize
                )
                transformer.load_weights(quantized_path, strict=True)
                if profiler:
                    profiler.stop("dit_load")
                print(
                    f"  Loaded cached {actual_quantize}-bit DiT: "
                    f"{quantized_layers} quantized layers"
                )
            else:
                if profiler:
                    profiler.start("dit_load")
                weights = mx.load(dit_weights_path)
                transformer.load_weights(list(weights.items()), strict=False)
                print(f"  Loaded DiT: {len(weights)} tensors (BF16)")
                del weights
                print(f"  Quantizing DiT to {actual_quantize}-bit...")
                quantized_layers = _quantize_transformer(
                    transformer, actual_quantize
                )
                _save_quantized_cache(
                    transformer,
                    quantized_path,
                    quantized_metadata_path,
                    expected_metadata,
                )
                if profiler:
                    profiler.stop("dit_load")
                print(
                    f"  Cached {actual_quantize}-bit DiT at {quantized_path} "
                    f"({quantized_layers} quantized layers)"
                )
        else:
            if profiler:
                profiler.start("dit_load")
            weights = mx.load(dit_weights_path)
            transformer.load_weights(list(weights.items()), strict=False)
            if profiler:
                profiler.stop("dit_load")
            print(f"  Loaded DiT: {len(weights)} tensors (BF16)")

        # Load VAE
        if profiler:
            profiler.start("vae_load")
        vae_weights_path = os.path.join(model_dir, "vae.safetensors")
        vae = MageVAE(vae_weights_path, sample_posterior=True)
        if profiler:
            profiler.stop("vae_load")
        print(f"  Loaded VAE")

        # Load text encoder (lazy — model weights are loaded on first use)
        if profiler:
            profiler.start("text_encoder_load")
        te_weights_path = os.path.join(model_dir, "text_encoder.safetensors")
        text_encoder = MageFlowTextEncoder(
            model_path=te_weights_path if os.path.exists(te_weights_path) else None,
        )
        if profiler:
            profiler.stop("text_encoder_load")
        print(f"  Loaded text encoder")

        # Load tokenizer for text encoding
        from transformers import AutoTokenizer
        raw_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        tokenizer = MageFlowTokenizer(raw_tokenizer)

        return cls(transformer, vae, text_encoder, tokenizer=tokenizer, num_steps=num_steps)

    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        seed: int = 42,
        guidance_scale: float = 5.0,
        negative_prompt: str = " ",
        profiler: Optional["object"] = None,
        cleanup_strategy: str = "all_three",
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt
            height: Output image height (must be multiple of 16)
            width: Output image width (must be multiple of 16)
            seed: Random seed for reproducibility
            guidance_scale: Classifier-free guidance scale; 1 disables CFG
            negative_prompt: Prompt for the unconditional CFG branch
            profiler: Optional Profiler instance for phase-level timing
            cleanup_strategy: Qwen cleanup strategy after text encoding.
                One of: "unload_only", "unload+gc", "unload+cache", "all_three".
                "all_three" is the default (current behavior).

        Returns:
            PIL Image
        """
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("height and width must be positive multiples of 16")
        if guidance_scale < 1.0:
            raise ValueError("guidance_scale must be at least 1.0")
        if cleanup_strategy not in CLEANUP_STRATEGIES:
            raise ValueError(
                f"cleanup_strategy must be one of {CLEANUP_STRATEGIES}, "
                f"got {cleanup_strategy}"
            )

        mx.random.seed(seed)

        # Latent grid size (16x downsample)
        lat_h, lat_w = height // 16, width // 16

        # 1. Text encoding via Qwen3-VL
        if profiler:
            profiler.start("text_encode")
        print(f"  Encoding text: '{prompt[:80]}...'")
        txt_embeds, _ = self.text_encoder.encode_text_to_image(
            prompts=[prompt],
            tokenizer=self.tokenizer,
            max_sequence_length=2048,
        )
        mx.eval(txt_embeds)
        print(f"  Text embeddings: {txt_embeds.shape}")

        neg_txt_embeds = None
        if guidance_scale > 1.0:
            neg_txt_embeds, _ = self.text_encoder.encode_text_to_image(
                prompts=[negative_prompt],
                tokenizer=self.tokenizer,
                max_sequence_length=2048,
            )
            mx.eval(neg_txt_embeds)
            print(f"  Negative text embeddings: {neg_txt_embeds.shape}")
        if profiler:
            profiler.stop("text_encode")

        # Qwen is only needed for prompt encoding. Releasing its ~8.9 GB BF16
        # weights leaves ample unified memory for DiT activations at 1024².
        if profiler:
            profiler.start("text_encoder_unload")
        self.text_encoder.unload()
        if cleanup_strategy in ("unload+gc", "all_three"):
            gc.collect()
        if cleanup_strategy in ("unload+cache", "all_three"):
            mx.clear_cache()
        if profiler:
            profiler.stop("text_encoder_unload")
        print("  Unloaded text encoder")

        # 2. Initialize Gaussian noise in latent space (NHWC)
        latents = mx.random.normal((1, lat_h, lat_w, 128)).astype(mx.bfloat16)

        # 3. Flow matching sampling loop
        for i in range(self.num_steps):
            if profiler:
                profiler.start(f"dit_step_{i + 1}")
            sigma = self.scheduler.sigmas[i]

            # Reshape latent to sequence: [1, H*W, 128]
            latents_seq = latents.reshape(1, -1, 128)

            # MageFlowTimestepProjEmbeddings applies its own scale=1000, so the
            # transformer receives the normalized flow sigma, not sigma*1000.
            t_batch = mx.array([float(sigma)])

            # Run through DiT
            v_pred_seq = self.transformer(
                img=latents_seq,
                txt=txt_embeds,
                timesteps=t_batch,
                img_shapes=(1, lat_h, lat_w),
            )

            if neg_txt_embeds is not None:
                v_uncond_seq = self.transformer(
                    img=latents_seq,
                    txt=neg_txt_embeds,
                    timesteps=t_batch,
                    img_shapes=(1, lat_h, lat_w),
                )
                v_pred_seq = v_uncond_seq + guidance_scale * (
                    v_pred_seq - v_uncond_seq
                )

            # Reshape velocity prediction back to NHWC
            v_pred = v_pred_seq.reshape(1, lat_h, lat_w, 128)

            # Euler step
            latents = self.scheduler.step(v_pred, i, latents)

            # Free graph memory
            mx.eval(latents)
            if profiler:
                profiler.stop(f"dit_step_{i + 1}")
            print(f"  Step {i + 1}/{self.num_steps} complete (sigma={float(sigma):.4f})")

        # 4. Decode latent via VAE
        if profiler:
            profiler.start("vae_decode")
        print("  Decoding latent...")
        images = self.vae.decode(latents)  # [1, H, W, 3] in [-1, 1]
        if profiler:
            profiler.stop("vae_decode")

        # Convert to PIL
        img_array = (images[0] + 1.0) * 127.5
        img_array = mx.clip(img_array, 0, 255).astype(mx.uint8)
        img_np = np.array(img_array)
        return Image.fromarray(img_np)

    def _generate_from_embeds(
        self,
        txt_embeds: mx.array,
        neg_txt_embeds: Optional[mx.array],
        height: int = 1024,
        width: int = 1024,
        seed: int = 42,
        guidance_scale: float = 5.0,
        profiler: Optional["object"] = None,
    ) -> Image.Image:
        """Generate an image from pre-encoded text embeddings.

        Bypasses text encoding and Qwen unloading entirely. Used by the
        persistent worker when embeddings are cached or pre-encoded in batch.

        Args:
            txt_embeds: [1, seq_len, 2560] positive prompt embeddings
            neg_txt_embeds: [1, seq_len, 2560] negative prompt embeddings, or None
            height: Output image height (must be multiple of 16)
            width: Output image width (must be multiple of 16)
            seed: Random seed for reproducibility
            guidance_scale: Classifier-free guidance scale; 1 disables CFG
            profiler: Optional Profiler instance for phase-level timing

        Returns:
            PIL Image
        """
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("height and width must be positive multiples of 16")
        if guidance_scale < 1.0:
            raise ValueError("guidance_scale must be at least 1.0")

        mx.random.seed(seed)

        # Latent grid size (16x downsample)
        lat_h, lat_w = height // 16, width // 16

        # Initialize Gaussian noise in latent space (NHWC)
        latents = mx.random.normal((1, lat_h, lat_w, 128)).astype(mx.bfloat16)

        # Flow matching sampling loop
        for i in range(self.num_steps):
            if profiler:
                profiler.start(f"dit_step_{i + 1}")
            sigma = self.scheduler.sigmas[i]

            # Reshape latent to sequence: [1, H*W, 128]
            latents_seq = latents.reshape(1, -1, 128)

            # MageFlowTimestepProjEmbeddings applies its own scale=1000, so the
            # transformer receives the normalized flow sigma, not sigma*1000.
            t_batch = mx.array([float(sigma)])

            # Run through DiT
            v_pred_seq = self.transformer(
                img=latents_seq,
                txt=txt_embeds,
                timesteps=t_batch,
                img_shapes=(1, lat_h, lat_w),
            )

            if neg_txt_embeds is not None:
                v_uncond_seq = self.transformer(
                    img=latents_seq,
                    txt=neg_txt_embeds,
                    timesteps=t_batch,
                    img_shapes=(1, lat_h, lat_w),
                )
                v_pred_seq = v_uncond_seq + guidance_scale * (
                    v_pred_seq - v_uncond_seq
                )

            # Reshape velocity prediction back to NHWC
            v_pred = v_pred_seq.reshape(1, lat_h, lat_w, 128)

            # Euler step
            latents = self.scheduler.step(v_pred, i, latents)

            # Free graph memory
            mx.eval(latents)
            if profiler:
                profiler.stop(f"dit_step_{i + 1}")
            print(f"  Step {i + 1}/{self.num_steps} complete (sigma={float(sigma):.4f})")

        # Decode latent via VAE
        if profiler:
            profiler.start("vae_decode")
        print("  Decoding latent...")
        images = self.vae.decode(latents)  # [1, H, W, 3] in [-1, 1]
        if profiler:
            profiler.stop("vae_decode")

        # Convert to PIL
        img_array = (images[0] + 1.0) * 127.5
        img_array = mx.clip(img_array, 0, 255).astype(mx.uint8)
        img_np = np.array(img_array)
        return Image.fromarray(img_np)

    def _apply_memory_policy(self, width: int, height: int) -> None:
        """Apply memory-saving policies for constrained Macs.

        Based on alis-studio's VAE tiling pattern:
        - Tile VAE decode for >=1024^2 on <=24GB Macs
        - Otherwise use exact untiled decode
        """
        ram_gib = self._total_ram_gib()
        if ram_gib > 0 and ram_gib <= 24 and width * height >= 1024 * 1024:
            # Enable tiling on the VAE
            if hasattr(self.vae, "tiling_config"):
                from mflux.models.common.vae.tiling_config import TilingConfig
                self.vae.tiling_config = TilingConfig()
                print(f"  VAE tiling enabled for {width}x{height} decode")

    @staticmethod
    def _total_ram_gib() -> float:
        """Get total system RAM in GiB."""
        try:
            import subprocess
            out = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2
            )
            return int(out.stdout.strip()) / (1024 ** 3)
        except Exception:
            return 0.0

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

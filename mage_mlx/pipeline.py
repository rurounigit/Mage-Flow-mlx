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
import os
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from .dit import MageFlow, MageFlowParams
from .loader import ensure_mlx_model
from .scheduler import FlowMatchEulerDiscreteScheduler
from .text_encoder import Qwen3VLTextEncoder
from .vae import MageVAE


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
        text_encoder: Qwen3VLTextEncoder,
        num_steps: int = 4,
    ):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
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
    ) -> "MageFlowPipeline":
        """Load a Mage-Flow MLX pipeline from a directory or HF repo ID.

        Args:
            model_dir: Directory containing converted MLX weights or HF repo ID
            num_steps: Number of denoising steps

        Returns:
            MageFlowPipeline instance
        """
        model_dir = ensure_mlx_model(model_dir)

        # Load DiT config
        import json

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

        # Load DiT weights
        dit_weights_path = os.path.join(model_dir, "transformer.safetensors")
        if os.path.exists(dit_weights_path):
            weights = mx.load(dit_weights_path)
            if any(key.endswith((".scales", ".biases")) for key in weights):
                raise ValueError("Quantized DiT weights are unsupported; reconvert in BF16")
            transformer.load_weights(list(weights.items()), strict=False)
            print(f"  Loaded DiT: {len(weights)} tensors (BF16)")

        # Load VAE
        vae_weights_path = os.path.join(model_dir, "vae.safetensors")
        vae = MageVAE(vae_weights_path, sample_posterior=False)
        print(f"  Loaded VAE")

        # Load text encoder
        te_weights_path = os.path.join(model_dir, "text_encoder.safetensors")
        text_encoder = Qwen3VLTextEncoder(
            model_path=te_weights_path if os.path.exists(te_weights_path) else None,
        )
        print(f"  Loaded text encoder")

        return cls(transformer, vae, text_encoder, num_steps=num_steps)

    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        seed: int = 42,
        guidance_scale: float = 5.0,
        negative_prompt: str = " ",
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt
            height: Output image height (must be multiple of 16)
            width: Output image width (must be multiple of 16)
            seed: Random seed for reproducibility
            guidance_scale: Classifier-free guidance scale; 1 disables CFG
            negative_prompt: Prompt for the unconditional CFG branch

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

        # 1. Text encoding via Qwen3-VL
        print(f"  Encoding text: '{prompt[:80]}...'")
        txt_embeds = self.text_encoder(prompt)
        mx.eval(txt_embeds)
        print(f"  Text embeddings: {txt_embeds.shape}")

        neg_txt_embeds = None
        if guidance_scale > 1.0:
            neg_txt_embeds = self.text_encoder(negative_prompt)
            mx.eval(neg_txt_embeds)
            print(f"  Negative text embeddings: {neg_txt_embeds.shape}")

        # Qwen is only needed for prompt encoding. Releasing its ~8.9 GB BF16
        # weights leaves ample unified memory for DiT activations at 1024².
        self.text_encoder.unload()
        gc.collect()
        mx.clear_cache()
        print("  Unloaded text encoder")

        # 2. Initialize Gaussian noise in latent space (NHWC)
        latents = mx.random.normal((1, lat_h, lat_w, 128)).astype(mx.bfloat16)

        # 3. Flow matching sampling loop
        for i in range(self.num_steps):
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
            print(f"  Step {i + 1}/{self.num_steps} complete (sigma={float(sigma):.4f})")

        # 4. Decode latent via VAE
        print("  Decoding latent...")
        images = self.vae.decode(latents)  # [1, H, W, 3] in [-1, 1]

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

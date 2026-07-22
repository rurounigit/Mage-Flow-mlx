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
import mlx.nn as nn
import numpy as np
from PIL import Image

from .dit import MageFlow, MageFlowParams
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
        model_dir: str,
        num_steps: int = 4,
    ) -> "MageFlowPipeline":
        """Load a Mage-Flow MLX pipeline from a directory.

        Args:
            model_dir: Directory containing converted MLX weights
            num_steps: Number of denoising steps

        Returns:
            MageFlowPipeline instance
        """
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
            from safetensors import safe_open

            weights = {}
            is_quantized = False
            with safe_open(dit_weights_path, framework="numpy") as f:
                for key in f.keys():
                    weights[key] = mx.array(f.get_tensor(key))
                    if ".scales" in key or ".biases" in key:
                        is_quantized = True

            if is_quantized:
                nn.quantize(transformer, group_size=64, bits=4)

            transformer.load_weights(list(weights.items()))
            print(f"  Loaded DiT: {len(weights)} tensors (quantized={is_quantized})")

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
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt
            height: Output image height (must be multiple of 16)
            width: Output image width (must be multiple of 16)
            seed: Random seed for reproducibility

        Returns:
            PIL Image
        """
        mx.random.seed(seed)

        # Latent grid size (16x downsample)
        lat_h, lat_w = height // 16, width // 16

        # 1. Text encoding via Qwen3-VL
        print(f"  Encoding text: '{prompt[:80]}...'")
        txt_embeds = self.text_encoder(prompt)
        mx.eval(txt_embeds)
        print(f"  Text embeddings: {txt_embeds.shape}")

        # 2. Initialize Gaussian noise in latent space (NHWC)
        latents = mx.random.normal((1, lat_h, lat_w, 128))

        # 3. Flow matching sampling loop
        for i in range(self.num_steps):
            sigma = self.scheduler.sigmas[i]

            # Reshape latent to sequence: [1, H*W, 128]
            latents_seq = latents.reshape(1, -1, 128)

            # Timestep embedding
            t_batch = mx.array([float(sigma)])

            # Run through DiT
            v_pred_seq = self.transformer(
                img=latents_seq,
                txt=txt_embeds,
                timesteps=t_batch,
                img_shapes=(1, lat_h, lat_w),
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
        - Tile VAE decode for ≥1024² on ≤24GB Macs
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

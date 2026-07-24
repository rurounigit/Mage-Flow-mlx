"""MageFlowEdit: Image editing pipeline for Mage-Flow MLX.

Ports mflux's MageFlowEdit and MageFlowEditUtil for use in Mage-Flow-mlx.

The edit pipeline takes a target image, reference images, and a text instruction,
then generates an edited version of the target image. It works by:

1. Encoding reference images via VAE → packed latents [1, N_refs*H*W, 128]
2. Encoding the edit prompt via the multi-modal text encoder (vision + text)
3. Concatenating target + reference latents along the sequence dimension
4. Running the DiT with multi-image ``img_shapes`` (target + references)
5. Slicing the output with ``target_length`` to extract only the target prediction
6. Applying classifier-free guidance and optional renormalization
7. Running the 4-step flow matching loop
8. Decoding the final latent via VAE

Usage:
    from mage_mlx.edit import MageFlowEdit
    from mage_mlx.pipeline import MageFlowPipeline

    pipeline = MageFlowPipeline.from_pretrained("models/mage_flow_mlx")
    edit = MageFlowEdit(
        transformer=pipeline.transformer,
        vae=pipeline.vae,
        text_encoder=pipeline.text_encoder,
    )
    image = edit.edit(
        target_image=Image.open("target.png"),
        ref_images=[Image.open("ref1.png"), Image.open("ref2.png")],
        prompt="make the sky purple",
        seed=42,
    )
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from .dit import MageFlow
from .scheduler import FlowMatchEulerDiscreteScheduler
from .text_encoder import MageFlowTextEncoder
from .vae import MageVAE
from .latent_creator import MageFlowLatentCreator


def reference_cache_key(image_bytes: bytes, size: tuple[int, int]) -> str:
    """Compute a SHA-256 cache key for reference image encoding.

    The key combines the image content hash with its pixel dimensions,
    ensuring that the same image at different sizes produces different keys.

    Args:
        image_bytes: Raw image bytes
        size: (width, height) of the image

    Returns:
        64-character hex SHA-256 digest
    """
    hasher = hashlib.sha256()
    hasher.update(image_bytes)
    hasher.update(f"{size[0]}x{size[1]}".encode("utf-8"))
    return hasher.hexdigest()


class MageFlowEditUtil:
    """Utility class for Mage-Flow edit operations.

    Handles reference image loading, VAE encoding, latent packing, and
    cache key computation for reference caching.

    Args:
        vae: MageVAE instance for encoding reference images
    """

    def __init__(self, vae: MageVAE):
        self.vae = vae

    def encode_references(
        self,
        ref_images: list[Image.Image],
        target_height: int,
        target_width: int,
        seed: int = 42,
    ) -> tuple[mx.array, list[tuple[int, int, int]]]:
        """Encode reference images to packed latents.

        Args:
            ref_images: List of reference PIL images
            target_height: Target image height (for computing latent grid size)
            target_width: Target image width

        Returns:
            (packed_latents, ref_img_shapes) where packed_latents has shape
            [1, N_refs * lat_h * lat_w, 128] and ref_img_shapes is a list of
            (frames, height, width) tuples for each reference image.
        """
        lat_h = target_height // 16
        lat_w = target_width // 16
        packed_refs = []
        ref_img_shapes = []

        for ref_image in ref_images:
            # Resize to target dimensions
            ref_resized = ref_image.convert("RGB").resize(
                (target_width, target_height), Image.BICUBIC
            )
            # Normalize to [-1, 1]
            ref_array = np.array(ref_resized, dtype=np.float32) / 127.5 - 1.0
            ref_mx = mx.array(ref_array, dtype=self.vae.dconv_encoder.patch_cond_embed.weight.dtype)[None, ...]  # [1, H, W, 3]

            # Encode via VAE
            ref_latents = self.vae.encode(
                ref_mx,
                key=mx.random.key(seed),
            )  # [1, lat_h, lat_w, 128]
            packed = self.vae.pack_latents(ref_latents)  # [1, lat_h*lat_w, 128]
            packed_refs.append(packed)
            ref_img_shapes.append((1, lat_h, lat_w))

        if packed_refs:
            concatenated = mx.concatenate(packed_refs, axis=1)  # [1, N_refs*lat_h*lat_w, 128]
        else:
            concatenated = mx.zeros((1, 0, 128), dtype=mx.bfloat16)

        return concatenated, ref_img_shapes

    @staticmethod
    def cache_key(image: Image.Image) -> str:
        """Compute a cache key for a reference image."""
        img_bytes = image.tobytes()
        return reference_cache_key(img_bytes, image.size)


class MageFlowEdit:
    """Mage-Flow image editing pipeline for MLX.

    Combines a MageFlow DiT, MageVAE, and MageFlowTextEncoder to perform
    text-guided image editing. The edit process:
    1. Encodes reference images via VAE
    2. Encodes the edit prompt (text + reference images) via the text encoder
    3. Concatenates target + reference latents
    4. Runs the DiT with multi-image img_shapes
    5. Slices output with target_length
    6. Applies CFG and renormalization
    7. Runs the flow matching loop
    8. Decodes via VAE

    Args:
        transformer: MageFlow DiT model
        vae: MageVAE model
        text_encoder: MageFlowTextEncoder (native MLX Qwen3-VL)
        num_steps: Number of denoising steps (4 for turbo)
    """

    def __init__(
        self,
        transformer: MageFlow,
        vae: MageVAE,
        text_encoder: MageFlowTextEncoder,
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
        self.edit_util = MageFlowEditUtil(vae)

    def make_velocity_predictor(
        self,
        *,
        latents_seq: mx.array,
        target_length: int,
        txt_embeds: mx.array,
        neg_txt_embeds: mx.array | None,
        img_shapes: list[tuple[int, int, int]],
        text_attention_mask: mx.array | None,
        timesteps: mx.array,
        guidance_scale: float,
        renormalization: bool = False,
    ) -> mx.array:
        """Predict velocity for the target latent region.

        Concatenates target + reference latents, runs the DiT with multi-image
        img_shapes, slices the output with target_length, and applies CFG.

        Args:
            latents_seq: [1, total_length, 128] concatenated target+reference latents
            target_length: Number of target latent tokens (for slicing)
            txt_embeds: [1, seq, 2560] positive text embeddings
            neg_txt_embeds: [1, seq, 2560] negative text embeddings or None
            img_shapes: List of (frames, height, width) for each image
            timesteps: [1] flow matching timestep
            guidance_scale: CFG scale (1.0 disables CFG)
            renormalization: If True, renormalize the velocity prediction

        Returns:
            [1, target_length, 128] predicted velocity for the target region
        """
        # Run conditional pass
        v_pred_seq = self.transformer(
            img=latents_seq,
            txt=txt_embeds,
            timesteps=timesteps,
            img_shapes=img_shapes,
            text_attention_mask=text_attention_mask,
        )

        # Run unconditional pass for CFG
        if neg_txt_embeds is not None and guidance_scale != 1.0:
            v_uncond_seq = self.transformer(
                img=latents_seq,
                txt=neg_txt_embeds,
                timesteps=timesteps,
                img_shapes=img_shapes,
                text_attention_mask=text_attention_mask,
            )
            v_pred_seq = v_uncond_seq + guidance_scale * (v_pred_seq - v_uncond_seq)

        # Slice to extract only the target prediction
        v_pred_target = v_pred_seq[:, :target_length, :]

        # Optional renormalization
        if renormalization:
            v_pred_target = self._renormalize_velocity(v_pred_target, latents_seq[:, :target_length, :])

        return v_pred_target

    @staticmethod
    def _renormalize_velocity(v_pred: mx.array, latents: mx.array) -> mx.array:
        """Renormalize velocity prediction to match the magnitude of the input latents.

        This helps stabilize the flow matching process by ensuring the velocity
        prediction has a similar scale to the input latents.
        """
        pred_norm = mx.sqrt(mx.mean(v_pred**2, axis=-1, keepdims=True) + 1e-8)
        latent_norm = mx.sqrt(mx.mean(latents**2, axis=-1, keepdims=True) + 1e-8)
        scale = latent_norm / pred_norm
        return v_pred * scale

    def edit(
        self,
        target_image: Image.Image,
        ref_images: list[Image.Image],
        prompt: str,
        seed: int = 42,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 5.0,
        negative_prompt: str = " ",
        renormalization: bool = False,
        profiler: Optional["object"] = None,
        tokenizer: Any = None,
    ) -> Image.Image:
        """Edit a target image based on a text prompt and reference images.

        Args:
            target_image: The image to edit
            ref_images: List of reference images for style/content guidance
            prompt: Text instruction for the edit
            seed: Random seed
            height: Output height (multiple of 16)
            width: Output width (multiple of 16)
            guidance_scale: CFG scale (1.0 disables CFG)
            negative_prompt: Negative prompt for CFG
            renormalization: If True, renormalize velocity predictions
            profiler: Optional Profiler instance
            tokenizer: Tokenizer wrapper (with ``processor`` and ``tokenizer`` attributes)

        Returns:
            Edited PIL Image
        """
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("height and width must be positive multiples of 16")
        if guidance_scale < 1.0:
            raise ValueError("guidance_scale must be at least 1.0")
        if tokenizer is None:
            raise TypeError("a tokenizer wrapper is required for edit")

        mx.random.seed(seed)
        lat_h, lat_w = height // 16, width // 16
        target_length = lat_h * lat_w

        # 1. Encode reference images via VAE
        if profiler:
            profiler.start("ref_encode")
        ref_latents, ref_img_shapes = self.edit_util.encode_references(
            ref_images, height, width, seed=seed
        )
        mx.eval(ref_latents)
        print(f"  Reference latents: {ref_latents.shape}")
        if profiler:
            profiler.stop("ref_encode")

        # 2. Encode edit prompt (text + reference images)
        if profiler:
            profiler.start("text_encode")
        print(f"  Encoding edit prompt: '{prompt[:80]}...'")
        txt_embeds, txt_mask = self.text_encoder.encode_edit(
            prompts=[prompt],
            images_per_prompt=[ref_images],
            tokenizer=tokenizer,
            max_sequence_length=2048,
        )
        mx.eval(txt_embeds)
        print(f"  Edit text embeddings: {txt_embeds.shape}")

        neg_txt_embeds = None
        if guidance_scale > 1.0:
            # CFG branches for edit must have identical multimodal structure.
            # mflux encodes the negative instruction with the same reference
            # images; a text-only negative branch removes the vision tokens and
            # produces an invalid unconditional edit condition.
            neg_txt_embeds, _ = self.text_encoder.encode_edit(
                prompts=[negative_prompt],
                images_per_prompt=[ref_images],
                tokenizer=tokenizer,
                max_sequence_length=2048,
            )
            mx.eval(neg_txt_embeds)
            print(f"  Negative edit embeddings: {neg_txt_embeds.shape}")
        if profiler:
            profiler.stop("text_encode")

        # 3. Initialize the canonical MageFlow Gaussian-Shading noise in
        # NCHW, then convert to the pipeline's NHWC latent layout.
        packed_noise = MageFlowLatentCreator.create_noise(
            seed=seed,
            height=height,
            width=width,
            dtype=mx.bfloat16,
        )
        latents = self.vae.unpack_latents(packed_noise, lat_h, lat_w)

        # 4. Flow matching sampling loop
        for i in range(self.num_steps):
            if profiler:
                profiler.start(f"dit_step_{i + 1}")
            sigma = self.scheduler.sigmas[i]

            # Pack target latents: [1, lat_h, lat_w, 128] → [1, lat_h*lat_w, 128]
            target_latents_seq = self.vae.pack_latents(latents)  # [1, target_length, 128]

            # Concatenate target + reference latents along sequence dim
            latents_seq = mx.concatenate(
                [target_latents_seq, ref_latents], axis=1
            )  # [1, total_length, 128]

            # Build multi-image img_shapes
            img_shapes = [(1, lat_h, lat_w)] + ref_img_shapes

            # Timestep
            t_batch = mx.array([float(sigma)])

            # Predict velocity
            v_pred_seq = self.make_velocity_predictor(
                latents_seq=latents_seq,
                target_length=target_length,
                txt_embeds=txt_embeds,
                neg_txt_embeds=neg_txt_embeds,
                img_shapes=img_shapes,
                text_attention_mask=txt_mask,
                timesteps=t_batch,
                guidance_scale=guidance_scale,
                renormalization=renormalization,
            )

            # Reshape velocity back to NHWC
            v_pred = self.vae.unpack_latents(v_pred_seq, lat_h, lat_w)

            # Euler step
            latents = self.scheduler.step(v_pred, i, latents)

            # Free graph memory
            mx.eval(latents)
            if profiler:
                profiler.stop(f"dit_step_{i + 1}")
            print(f"  Step {i + 1}/{self.num_steps} complete (sigma={float(sigma):.4f})")

        # 5. Decode latent via VAE
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

    def __call__(self, *args, **kwargs):
        return self.edit(*args, **kwargs)

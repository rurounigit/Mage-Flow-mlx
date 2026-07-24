"""MageFlowQwen3VLProcessor: Multi-modal tokenizer/image processor for Mage-Flow edit.

Ports mflux's MageFlowQwen3VLProcessor and QwenImageProcessor for use in the
native MLX Qwen3-VL text encoder. Handles:
  - Image preprocessing (resize, normalize, patchify into Conv3d-compatible format)
  - ``<|image_pad|>`` placeholder expansion to ``prod(image_grid_thw) / merge_size^2`` tokens
  - Joint text + image tokenization with padding and truncation

Usage:
    from mage_mlx.processor import MageFlowQwen3VLProcessor

    processor = MageFlowQwen3VLProcessor(tokenizer)
    inputs = processor(text=formatted_prompt, images=reference_images, padding=True)
    # inputs: {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
"""

from __future__ import annotations

import math
from typing import Any, Union

import mlx.core as mx
import numpy as np
from PIL import Image


OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """Resize maintaining aspect ratio with factor-aligned dimensions."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class QwenImageProcessor:
    """Image preprocessing for Qwen3-VL vision.

    Resizes images to factor-aligned dimensions, normalizes to CLIP statistics,
    and patches into Conv3d-compatible format.

    Args:
        min_pixels: Minimum image pixels (default 65536)
        max_pixels: Maximum image pixels (default 16777216)
        patch_size: Spatial patch size (16)
        temporal_patch_size: Temporal patch size (2)
        merge_size: Spatial merge size (2)
        image_mean: Normalization mean
        image_std: Normalization std
    """

    def __init__(
        self,
        min_pixels: int = 65_536,
        max_pixels: int = 16_777_216,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
    ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

    def _preprocess(
        self,
        image: Image.Image,
        resized_height: int | None = None,
        resized_width: int | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Preprocess a single image into flattened patches.

        Returns:
            (flatten_patches, (grid_t, grid_h, grid_w))
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        height, width = image.size[1], image.size[0]

        if resized_height is None or resized_width is None:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        if (height, width) != (resized_height, resized_width):
            image = image.resize((resized_width, resized_height), Image.BICUBIC)

        image_np = np.array(image).astype(np.float32)
        image_np = image_np / 255.0

        mean_np = np.array(self.image_mean, dtype=np.float32)
        std_np = np.array(self.image_std, dtype=np.float32)
        image_np = (image_np - mean_np) / std_np

        image_np = image_np.transpose(2, 0, 1)
        patches = image_np[np.newaxis]  # Shape: (1, channel, height, width)

        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis],
                self.temporal_patch_size - (patches.shape[0] % self.temporal_patch_size),
                axis=0,
            )
            patches = np.concatenate([patches, repeats], axis=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)

        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: Union[Image.Image, list[Image.Image]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess a batch of images.

        Returns:
            (pixel_values, vision_grid_thws) where pixel_values has shape
            [total_patches, patch_dim] and vision_grid_thws has shape [num_images, 3].
        """
        if not isinstance(images, list):
            images = [images]

        pixel_values_list = []
        vision_grid_thws = []

        for image in images:
            patches, image_grid_thw = self._preprocess(image)
            pixel_values_list.append(patches)
            vision_grid_thws.append([image_grid_thw[0], image_grid_thw[1], image_grid_thw[2]])

        pixel_values = np.concatenate(pixel_values_list, axis=0) if pixel_values_list else np.array([])
        vision_grid_thws = np.array(vision_grid_thws)

        return pixel_values, vision_grid_thws

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> int:
        """Return the number of vision patches for a given image size."""
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        factor = self.patch_size * self.merge_size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size
        return grid_h * grid_w


class MageFlowQwen3VLImageProcessor(QwenImageProcessor):
    """Image preprocessing parameters embedded in the Mage-Flow checkpoint.

    Uses CLIP mean/std (0.5, 0.5, 0.5) and limits the long edge to 384 pixels
    for efficiency.
    """

    def __init__(self, max_long_edge: int | None = 384):
        super().__init__(
            min_pixels=65_536,
            max_pixels=16_777_216,
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
        self.max_long_edge = max_long_edge

    def _preprocess(
        self,
        image: Image.Image,
        resized_height: int | None = None,
        resized_width: int | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        if resized_height is None and resized_width is None:
            image = self._resize_long_edge(image)
        return super()._preprocess(
            image,
            resized_height=resized_height,
            resized_width=resized_width,
        )

    def _resize_long_edge(self, image: Image.Image) -> Image.Image:
        """Resize image so the long edge is at most max_long_edge pixels."""
        if self.max_long_edge is None or self.max_long_edge <= 0:
            return image
        width, height = image.size
        long_edge = max(width, height)
        if long_edge <= self.max_long_edge:
            return image
        scale = self.max_long_edge / long_edge
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        return image.resize((resized_width, resized_height), Image.BICUBIC)


class MageFlowQwen3VLProcessor:
    """Qwen3-VL tokenizer/image processor for Mage-Flow edit.

    The base processor expands each ``<|image_pad|>`` placeholder to
    ``prod(image_grid_thw) / merge_size**2`` tokens before tokenization.

    Args:
        tokenizer: HuggingFace tokenizer (Qwen3-VL)
        max_long_edge: Maximum long edge for image resizing (default 384)
    """

    def __init__(self, tokenizer: Any, max_long_edge: int | None = 384):
        self.tokenizer = tokenizer
        self.image_processor = MageFlowQwen3VLImageProcessor(max_long_edge=max_long_edge)
        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.image_token_id = (
            tokenizer.image_token_id
            if hasattr(tokenizer, "image_token_id")
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if hasattr(tokenizer, "video_token_id")
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )

    def __call__(
        self,
        images: Image.Image | list[Image.Image] | None = None,
        text: str | list[str] | None = None,
        padding: bool = True,
        return_tensors: str | None = None,
        max_length: int | None = 2112,
        truncation: bool = True,
    ) -> dict[str, Any]:
        """Process text and/or images into model inputs.

        Args:
            images: Image(s) to preprocess
            text: Text prompt(s) with ``<|image_pad|>`` placeholders
            padding: Whether to pad sequences
            return_tensors: "np" or None
            max_length: Maximum sequence length
            truncation: Whether to truncate

        Returns:
            Dict with "input_ids", "attention_mask", "pixel_values", "image_grid_thw"
        """
        image_inputs: dict[str, Any] = {}
        image_grid_thw = None
        if images is not None:
            image_list = images if isinstance(images, list) else [images]
            pixel_values, image_grid_thw = self.image_processor.preprocess(image_list)
            image_inputs = {
                "pixel_values": mx.array(pixel_values),
                "image_grid_thw": mx.array(image_grid_thw),
            }
        else:
            image_list = []

        if text is None:
            return image_inputs

        texts = [text] if isinstance(text, str) else text.copy()
        placeholder_count = sum(sample.count(self.image_token) for sample in texts)
        if placeholder_count != len(image_list):
            raise ValueError(
                f"found {placeholder_count} image placeholders for {len(image_list)} images"
            )

        if image_grid_thw is not None:
            placeholder = "<|mage_flow_image_placeholder|>"
            image_index = 0
            for text_index, sample in enumerate(texts):
                while self.image_token in sample:
                    token_count = int(np.prod(image_grid_thw[image_index])) // self.image_processor.merge_size**2
                    sample = sample.replace(self.image_token, placeholder * token_count, 1)
                    image_index += 1
                texts[text_index] = sample.replace(placeholder, self.image_token)

        tokenizer_kwargs: dict[str, Any] = {
            "padding": padding,
            "return_tensors": "pt" if return_tensors == "pt" else "np",
        }
        if max_length is not None:
            tokenizer_kwargs["max_length"] = max_length
        if truncation:
            tokenizer_kwargs["truncation"] = True

        text_inputs = self.tokenizer(texts, **tokenizer_kwargs)
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs.get("attention_mask")
        if return_tensors == "pt" and hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()
            attention_mask = attention_mask.numpy() if attention_mask is not None else None

        result = {
            **image_inputs,
            "input_ids": mx.array(np.asarray(input_ids)),
        }
        if attention_mask is not None:
            result["attention_mask"] = mx.array(np.asarray(attention_mask))
        return result

"""MageFlowPromptProcessor: Shared prompt templates and hidden-state processing.

Ports mflux's MageFlowPromptProcessor for use in both text-to-image and edit
pipelines. Handles:
  - Text-to-image template (34 drop tokens)
  - Edit template with image placeholders (64 drop tokens)
  - Trim-and-pad of hidden states with attention masks

Usage:
    from mage_mlx.prompt_processor import MageFlowPromptProcessor

    formatted = MageFlowPromptProcessor.format_edit("make the sky blue", num_images=1)
    embeds, mask = MageFlowPromptProcessor.process_edit_hidden_states(hidden_states, attention_mask)
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np


class MageFlowPromptProcessor:
    """Shared prompt templates and hidden-state processing for Mage-Flow.

    Constants mirror the released Mage-Flow checkpoint:
      - TEXT_TO_IMAGE_DROP_TOKENS = 34 (system prompt tokens to drop)
      - EDIT_DROP_TOKENS = 64 (system prompt + image placeholder tokens to drop)
      - MAX_CONDITION_TOKENS = 2048 (max text tokens passed to the DiT)
    """

    MAX_CONDITION_TOKENS = 2048
    TEXT_TO_IMAGE_DROP_TOKENS = 34
    EDIT_DROP_TOKENS = 64
    IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

    TEXT_TO_IMAGE_TEMPLATE = (
        "<|im_start|>system\n"
        "Describe the image by detailing the color, shape, size, texture, quantity, "
        "text, spatial relationships of the objects and background:"
        "<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    EDIT_TEMPLATE = (
        "<|im_start|>system\n"
        "Describe the key features of the input image (color, shape, size, texture,"
        " objects, background), then explain how the user's text instruction should alter or modify the image. "
        "Generate a new image that meets the user's requirements while maintaining consistency with the original "
        "input where appropriate.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    @classmethod
    def format_text_to_image(cls, prompt: str) -> str:
        """Format a text-to-image prompt with the system template."""
        return cls.TEXT_TO_IMAGE_TEMPLATE.format(prompt)

    @classmethod
    def format_edit(cls, instruction: str, num_images: int = 1) -> str:
        """Format an edit prompt with image placeholders.

        Args:
            instruction: The edit instruction text
            num_images: Number of reference images (each gets an image placeholder)

        Returns:
            Formatted prompt string with image placeholders
        """
        if num_images < 1:
            raise ValueError("an edit prompt requires at least one reference image")
        image_prefix = "".join(
            f"Image {image_index}: {cls.IMAGE_PLACEHOLDER}" for image_index in range(1, num_images + 1)
        )
        return cls.EDIT_TEMPLATE.format(image_prefix + instruction)

    @classmethod
    def process_text_to_image_hidden_states(
        cls,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Drop template tokens and right-pad for text-to-image conditioning.

        Args:
            hidden_states: [batch, sequence, channels] from the text encoder
            attention_mask: [batch, sequence] attention mask

        Returns:
            (padded_hidden_states, padded_attention_mask)
        """
        return cls.trim_and_pad_hidden_states(
            hidden_states,
            attention_mask,
            drop_tokens=cls.TEXT_TO_IMAGE_DROP_TOKENS,
        )

    @classmethod
    def process_edit_hidden_states(
        cls,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Drop template tokens and right-pad for edit conditioning.

        Args:
            hidden_states: [batch, sequence, channels] from the text encoder
            attention_mask: [batch, sequence] attention mask

        Returns:
            (padded_hidden_states, padded_attention_mask)
        """
        return cls.trim_and_pad_hidden_states(
            hidden_states,
            attention_mask,
            drop_tokens=cls.EDIT_DROP_TOKENS,
        )

    @staticmethod
    def trim_and_pad_hidden_states(
        hidden_states: mx.array,
        attention_mask: mx.array,
        *,
        drop_tokens: int,
        max_length: int = MAX_CONDITION_TOKENS,
    ) -> tuple[mx.array, mx.array]:
        """Drop template tokens per sample and return a right-padded MLX batch.

        For each sample in the batch:
        1. Select only active (non-padded) tokens using the attention mask
        2. Drop the first ``drop_tokens`` tokens (system prompt / template)
        3. Truncate to ``max_length`` tokens
        4. Right-pad all samples to the same length

        Args:
            hidden_states: [batch, sequence, channels]
            attention_mask: [batch, sequence]
            drop_tokens: Number of leading tokens to drop per sample
            max_length: Maximum number of tokens to keep per sample

        Returns:
            (padded_hidden_states, padded_attention_mask) where padded_hidden_states
            has shape [batch, max_sample_length, channels] and padded_attention_mask
            has shape [batch, max_sample_length].
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, channels]")
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must match the hidden-state batch and sequence dimensions")
        if drop_tokens < 0:
            raise ValueError("drop_tokens must be non-negative")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        # Token masks are small host-side metadata; selecting active indices once
        # avoids synchronizing any model activations.
        mask = np.asarray(attention_mask).astype(bool, copy=False)
        trimmed: list[mx.array] = []
        for batch_index, sample_mask in enumerate(mask):
            active_indices = np.flatnonzero(sample_mask)
            active = hidden_states[batch_index, mx.array(active_indices, dtype=mx.int32)]
            trimmed.append(active[drop_tokens : drop_tokens + max_length])

        max_len = max((sample.shape[0] for sample in trimmed), default=0)
        hidden_size = hidden_states.shape[-1]
        padded_hidden_states = []
        padded_masks = []
        for sample in trimmed:
            sample_length = sample.shape[0]
            padding_length = max_len - sample_length
            padded_hidden_states.append(
                mx.concatenate(
                    [sample, mx.zeros((padding_length, hidden_size), dtype=hidden_states.dtype)],
                    axis=0,
                )
            )
            padded_masks.append(
                mx.concatenate(
                    [
                        mx.ones((sample_length,), dtype=mx.int32),
                        mx.zeros((padding_length,), dtype=mx.int32),
                    ],
                    axis=0,
                )
            )

        if not padded_hidden_states:
            return hidden_states[:, :0], attention_mask[:, :0].astype(mx.int32)
        return mx.stack(padded_hidden_states), mx.stack(padded_masks)

    @classmethod
    def format_edits(cls, instructions: Sequence[str], image_counts: Sequence[int]) -> list[str]:
        """Format multiple edit prompts.

        Args:
            instructions: List of edit instruction strings
            image_counts: List of image counts (one per instruction)

        Returns:
            List of formatted prompt strings
        """
        if len(instructions) != len(image_counts):
            raise ValueError("instructions and image_counts must have the same length")
        return [
            cls.format_edit(instruction, num_images=image_count)
            for instruction, image_count in zip(instructions, image_counts, strict=True)
        ]

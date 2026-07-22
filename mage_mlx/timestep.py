"""Timestep embedding for Mage-Flow (qwen_proj style).

Port of MageFlowTimestepProjEmbeddings from PyTorch to MLX.

The timestep embedding uses:
  1. A sinusoidal DDPM-style embedding (256-dim) with bf16 rounding
  2. A SiLU + Linear projection to the model's hidden dimension

The bf16 rounding in step 1 is critical — the model was trained with this
exact rounding, so using fp32 would produce slightly different embeddings
and degrade output quality.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    """Sinusoidal timestep embeddings (DDPM convention).

    NOTE: The frequency table is downcast to ``timesteps.dtype`` (bf16) here —
    the model was trained with this exact bf16 rounding, so using fp32 would
    produce slightly different embeddings and degrade outputs.

    Args:
        timesteps: 1D array of timestep values
        embedding_dim: Output dimension (must be even)
        flip_sin_to_cos: If True, swap sine and cosine halves
        downscale_freq_shift: Frequency downscale shift
        scale: Scaling factor
        max_period: Maximum period for the sinusoidal embedding

    Returns:
        [N, embedding_dim] embedding tensor
    """
    assert timesteps.ndim == 1, "Timesteps should be a 1D array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(
        0, half_dim, dtype=mx.float32
    ) / (half_dim - downscale_freq_shift)

    # Downcast to timesteps dtype (bf16) — matches training
    emb = (exponent.astype(timesteps.dtype) if timesteps.dtype != mx.float32 else exponent)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]

    emb = scale * emb

    # Concat sine and cosine
    emb = mx.concat([mx.sin(emb), mx.cos(emb)], axis=-1)

    if flip_sin_to_cos:
        emb = mx.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # Zero-pad if odd
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])

    return emb


class Timesteps(nn.Module):
    """Sinusoidal timestep embedding module."""

    def __init__(
        self,
        num_channels: int = 256,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: mx.array) -> mx.array:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """MLP-based timestep embedding projection (diffusers TimestepEmbedding).

    Linear → SiLU → Linear
    """

    def __init__(self, in_channels: int = 256, time_embed_dim: int = 3072):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, time_embed_dim)
        self.lin2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.lin1(x)
        x = nn.silu(x)
        x = self.lin2(x)
        return x


class MageFlowTimestepProjEmbeddings(nn.Module):
    """Timestep projection embeddings for Mage-Flow (qwen_proj style).

    Combines sinusoidal embedding with an MLP projection:
      timestep → sinusoidal(256) → SiLU + Linear(256→D) → Linear(D→D)
    """

    def __init__(self, embedding_dim: int = 3072):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            scale=1000.0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def __call__(self, timestep: mx.array, hidden_states: mx.array) -> mx.array:
        """Compute timestep conditioning embedding.

        Args:
            timestep: [N] timestep values
            hidden_states: [N, ...] reference tensor for dtype

        Returns:
            [N, embedding_dim] conditioning embedding
        """
        timesteps_proj = self.time_proj(timestep)
        # Match dtype of hidden_states (bf16)
        timesteps_proj = timesteps_proj.astype(hidden_states.dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb

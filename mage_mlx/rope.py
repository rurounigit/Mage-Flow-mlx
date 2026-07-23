"""2D Multi-Scale RoPE (MageFlowEmbedRope) for Mage-Flow.

Port of Microsoft's MageFlowEmbedRope from PyTorch to MLX. Uses complex-number
rotary embeddings with three axes (frame, height, width) at different
frequencies (axes_dim = [16, 56, 56]).

Key translation notes:
  - PyTorch ``torch.polar(ones, angles)`` → MLX: precompute cos/sin from angles
  - PyTorch ``view_as_complex`` / ``view_as_real`` → MLX: manual real/imag split
  - Complex multiplication (a+bi)(c+di) = (ac-bd) + (ad+bc)i

Optimization: cos/sin caching
  ``MageFlowEmbedRope`` caches cos/sin tensors by (frame, height, width, idx)
  so that ``apply_rotary_emb_mageflow`` never recomputes ``mx.cos`` / ``mx.sin``
  for the same resolution. This avoids 24 redundant cos/sin evaluations per
  DiT step (2 per block × 12 blocks), or 96 across a 4-step generation.
"""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn


def rope_params(index: mx.array, dim: int, theta: float = 10000.0) -> mx.array:
    """Compute RoPE angle frequencies for a 1D position index.

    Args:
        index: 1D array of position indices [N]
        dim: frequency dimension (must be even)
        theta: base frequency

    Returns:
        Angle tensor [N, dim//2] (pre-polar, i.e. the raw angles)
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    # freqs = outer(index, 1/theta^(arange(0, dim, 2)/dim))
    exponents = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    inv_freq = 1.0 / (theta ** exponents)
    freqs = mx.outer(index.astype(mx.float32), inv_freq)
    return freqs


def apply_rotary_emb_mageflow(x: mx.array, freqs_cis: mx.array) -> mx.array:
    """Apply complex rotary embeddings to ``x`` using ``freqs_cis``.

    ``freqs_cis`` may be either:
    - Raw angle values [N, D//2] — cos/sin are computed here (backward compat)
    - Pre-computed cos/sin as a tuple (cos, sin) — no computation needed

    Args:
        x: [N, H, D] or [N, D] — query/key tensor (D must be even)
        freqs_cis: [N, D//2] angle values, or (cos, sin) tuple

    Returns:
        Rotated tensor, same shape as x
    """
    orig_shape = x.shape
    # Reshape to [..., D//2, 2] to split into real/imag pairs
    x = x.reshape(*x.shape[:-1], -1, 2)
    x_real = x[..., 0]
    x_imag = x[..., 1]

    # Accept either pre-computed (cos, sin) tuple or raw angle values
    if isinstance(freqs_cis, tuple):
        cos, sin = freqs_cis
        # Ensure cos/sin are broadcastable with x_real [batch, seq, heads, dim//2]
        # Cached cos/sin have shape [seq, dim//2]; unsqueeze to [seq, 1, dim//2]
        if cos.ndim == 2:
            cos = cos[:, None, :]
        if sin.ndim == 2:
            sin = sin[:, None, :]
    else:
        # freqs_cis has shape [N, D//2]; unsqueeze to broadcast with [N, H, D//2]
        angles = freqs_cis
        if angles.ndim == 2:
            angles = angles[:, None, :]
        cos = mx.cos(angles)
        sin = mx.sin(angles)

    # Complex multiplication: (a+bi) * (cos+i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos

    out = mx.stack([out_real, out_imag], axis=-1)
    return out.reshape(*orig_shape)


class MageFlowEmbedRope(nn.Module):
    """2D multi-scale RoPE for Mage-Flow's native-resolution DiT.

    Computes vision RoPE frequencies for packed image tokens. Text tokens
    are NOT rotated (no text RoPE is computed).

    Caches both raw angle values and pre-computed cos/sin tensors by
    (frame, height, width, idx) so that cos/sin are computed once per
    resolution and reused across all attention blocks and DiT steps.

    Args:
        theta: Base frequency (default 10000)
        axes_dim: Three frequency dimensions [frame_dim, height_dim, width_dim]
            Total must equal attention_head_dim (128 for Mage-Flow)
        scale_rope: Whether to use symmetric (scale) rope for height/width
    """

    def __init__(
        self,
        theta: float = 10000.0,
        axes_dim: Sequence[int] = (16, 56, 56),
        scale_rope: bool = True,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)
        assert sum(self.axes_dim) == 128, (
            f"sum(axes_dim) must equal head_dim (128), got {sum(self.axes_dim)}"
        )

        # Precompute pos/neg frequency tables (as angle values, not complex)
        pos_index = mx.arange(4096, dtype=mx.float32)
        # Match ``torch.arange(4096).flip(0) * -1 - 1`` exactly. The tail of
        # this table must contain [..., -2, -1] because symmetric RoPE slices
        # from the end for spatial positions near the origin.
        neg_index = mx.arange(4095, -1, -1, dtype=mx.float32) * -1.0 - 1.0

        self.pos_freqs = mx.concat(
            [
                rope_params(pos_index, self.axes_dim[0], theta),
                rope_params(pos_index, self.axes_dim[1], theta),
                rope_params(pos_index, self.axes_dim[2], theta),
            ],
            axis=1,
        )
        self.neg_freqs = mx.concat(
            [
                rope_params(neg_index, self.axes_dim[0], theta),
                rope_params(neg_index, self.axes_dim[1], theta),
                rope_params(neg_index, self.axes_dim[2], theta),
            ],
            axis=1,
        )

        self.scale_rope = scale_rope
        # Cache: (frame, height, width, idx) → (freqs_cis, cos, sin)
        self._cache: dict[tuple, tuple[mx.array, mx.array, mx.array]] = {}

    def _compute_video_freqs(
        self, frame: int, height: int, width: int, idx: int = 0
    ) -> mx.array:
        """Compute vision RoPE frequencies for a (frame, height, width) grid.

        Mirrors the PyTorch implementation: splits pos_freqs/neg_freqs by
        axes_dim//2, then tiles frame/height/width frequencies.
        """
        seq_len = frame * height * width
        half_dims = [d // 2 for d in self.axes_dim]
        # MLX split uses indices (numpy-style), not chunk sizes (PyTorch-style)
        # Split pos_freqs [4096, 64] into [4096, 8], [4096, 28], [4096, 28]
        cum = 0
        freqs_pos = []
        freqs_neg = []
        for hd in half_dims:
            freqs_pos.append(self.pos_freqs[:, cum:cum + hd])
            freqs_neg.append(self.neg_freqs[:, cum:cum + hd])
            cum += hd

        # Frame frequencies: [frame, 1, 1, dim//2] → [frame, height, width, dim//2]
        freqs_frame = freqs_pos[0][idx : idx + frame]
        freqs_frame = freqs_frame.reshape(frame, 1, 1, half_dims[0])
        freqs_frame = mx.broadcast_to(freqs_frame, (frame, height, width, half_dims[0]))

        if self.scale_rope:
            # Symmetric rope: negative indices for the second half of the spatial dim
            freqs_height = mx.concat(
                [freqs_neg[1][-(height - height // 2):], freqs_pos[1][: height // 2]],
                axis=0,
            )
            freqs_height = freqs_height.reshape(1, height, 1, half_dims[1])
            freqs_height = mx.broadcast_to(freqs_height, (frame, height, width, half_dims[1]))

            freqs_width = mx.concat(
                [freqs_neg[2][-(width - width // 2):], freqs_pos[2][: width // 2]],
                axis=0,
            )
            freqs_width = freqs_width.reshape(1, 1, width, half_dims[2])
            freqs_width = mx.broadcast_to(freqs_width, (frame, height, width, half_dims[2]))
        else:
            freqs_height = freqs_pos[1][:height]
            freqs_height = freqs_height.reshape(1, height, 1, half_dims[1])
            freqs_height = mx.broadcast_to(freqs_height, (frame, height, width, half_dims[1]))

            freqs_width = freqs_pos[2][:width]
            freqs_width = freqs_width.reshape(1, 1, width, half_dims[2])
            freqs_width = mx.broadcast_to(freqs_width, (frame, height, width, half_dims[2]))

        freqs = mx.concat([freqs_frame, freqs_height, freqs_width], axis=-1)
        return freqs.reshape(seq_len, half_dims[0] + half_dims[1] + half_dims[2])

    def __call__(
        self,
        video_fhw: tuple[int, int, int] | list[tuple[int, int, int]],
        max_img_len: int | None = None,
    ) -> mx.array:
        """Compute vision RoPE frequencies for packed image tokens.

        Returns pre-computed cos/sin as a tuple (cos, sin) instead of raw
        angle values, so that ``apply_rotary_emb_mageflow`` skips the
        expensive ``mx.cos`` / ``mx.sin`` calls.

        Args:
            video_fhw: (frame, height, width) or list of such tuples
            max_img_len: Pad to this length if provided

        Returns:
            (cos, sin) tuple of [seq_len, dim//2] tensors
        """
        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            key = (frame, height, width, idx)
            if key not in self._cache:
                freqs_cis = self._compute_video_freqs(frame, height, width, idx)
                # Pre-compute cos/sin once per resolution and cache them.
                # This avoids 24 redundant cos/sin evaluations per DiT step
                # (2 per block × 12 blocks), or 96 across 4 steps.
                cos = mx.cos(freqs_cis)
                sin = mx.sin(freqs_cis)
                self._cache[key] = (freqs_cis, cos, sin)
            vid_freqs.append(self._cache[key])

        # Concatenate cos/sin across all video segments
        cos_result = mx.concat([v[1] for v in vid_freqs], axis=0)
        sin_result = mx.concat([v[2] for v in vid_freqs], axis=0)

        if max_img_len is not None and cos_result.shape[0] < max_img_len:
            pad_len = max_img_len - cos_result.shape[0]
            cos_result = mx.pad(cos_result, [(0, pad_len), (0, 0)])
            sin_result = mx.pad(sin_result, [(0, pad_len), (0, 0)])

        return (cos_result, sin_result)
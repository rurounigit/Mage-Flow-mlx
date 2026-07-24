"""MageFlowQwen3VLVisionModel: Qwen3-VL vision tower for Mage-Flow edit.

Ports mflux's MageFlowQwen3VLVisionModel and its sub-components from
``mflux/models/mage_flow/model/mage_flow_text_encoder/vision_model.py`` and
``mflux/models/common_models/qwen3_vl/``.

Architecture:
  - Qwen3VLVisionPatchEmbed: Conv3d patch embedding (temporal + spatial)
  - Qwen3VLVisionRotaryEmbedding: 2D vision RoPE (height/width)
  - Qwen3VLVisionAttention: multi-head attention with cu_seqlens for multi-image
  - Qwen3VLVisionMLP: GELU-approximate MLP
  - Qwen3VLVisionPatchMerger: spatial merge + projection to language hidden size
  - Qwen3VLVisionBlock: LayerNorm + attention + MLP (residual)
  - MageFlowQwen3VLVisionModel: 24 vision blocks + patch embed + merger + DeepStack

The vision model processes reference images and produces visual embeddings that
are injected into the Qwen3-VL language model's token stream at image-token
positions. DeepStack features are extracted at layers 5, 11, 17 for early
injection into the language model.

Usage:
    from mage_mlx.vision_model import MageFlowQwen3VLVisionModel

    vision = MageFlowQwen3VLVisionModel()
    image_embeds, deepstack_embeds = vision(pixel_values, grid_thw, return_deepstack=True)
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np
from mlx import nn


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------
class Qwen3VLVisionPatchEmbed(nn.Module):
    """Conv3d patch embedding for Qwen3-VL vision.

    Input is flattened patches of shape [seq_len, in_channels * temporal_patch_size * patch_size^2].
    Output is [seq_len, embed_dim] after Conv3d with kernel=stride=[temporal_patch_size, patch_size, patch_size].
    """

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        stride = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        # Transpose to NDHWC for MLX Conv3d (channels-last on Apple Silicon)
        hidden_states = hidden_states.transpose(0, 2, 3, 4, 1)
        output = self.proj(hidden_states)
        output = output.reshape(seq_len, self.embed_dim)
        return output


# ---------------------------------------------------------------------------
# Vision Rotary Embedding
# ---------------------------------------------------------------------------
class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """Simple 1D RoPE for vision attention (height/width positions)."""

    def __init__(self, dim: int, theta: float = 10_000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        self.inv_freq = mx.array(inv_freq)

    def __call__(self, seqlen: int) -> mx.array:
        seq = mx.arange(seqlen, dtype=mx.float32)
        freqs = mx.outer(seq, self.inv_freq)
        return freqs


# ---------------------------------------------------------------------------
# Vision Attention
# ---------------------------------------------------------------------------
class Qwen3VLVisionAttention(nn.Module):
    """Multi-head attention with cu_seqlens for multi-image batching.

    Uses fused QKV projection and grouped attention per image (via cu_seqlens)
    to avoid cross-image attention. Applies 2D vision RoPE to Q/K.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("vision hidden size must be divisible by the number of heads")
        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: Sequence[int],
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        sequence_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.transpose(1, 0, 2, 3)
        query_states, key_states, value_states = mx.split(qkv, 3, axis=0)
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)

        # Hugging Face intentionally performs the vision rotary multiply in
        # FP32, then returns q/k to their original checkpoint dtype.
        query_dtype = query_states.dtype
        if position_embeddings is not None:
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states.astype(mx.float32),
                key_states.astype(mx.float32),
                position_embeddings[0].astype(mx.float32),
                position_embeddings[1].astype(mx.float32),
            )
            query_states = query_states.astype(query_dtype)
            key_states = key_states.astype(query_dtype)

        outputs = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
            query = query_states[start:end].transpose(1, 0, 2)
            key = key_states[start:end].transpose(1, 0, 2)
            value = value_states[start:end].transpose(1, 0, 2)
            query = mx.expand_dims(query, axis=0)
            key = mx.expand_dims(key, axis=0)
            value = mx.expand_dims(value, axis=0)

            output = mx.fast.scaled_dot_product_attention(
                query, key, value, scale=self.scaling
            )
            outputs.append(output.squeeze(0).transpose(1, 0, 2))

        attention_output = mx.concatenate(outputs, axis=0).reshape(sequence_length, self.dim)
        return self.proj(attention_output)

    @staticmethod
    def _apply_rotary_pos_emb(
        query_states: mx.array,
        key_states: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> tuple[mx.array, mx.array]:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
        query_embed = query_states * cos + Qwen3VLVisionAttention._rotate_half(query_states) * sin
        key_embed = key_states * cos + Qwen3VLVisionAttention._rotate_half(key_states) * sin
        return query_embed, key_embed

    @staticmethod
    def _rotate_half(hidden_states: mx.array) -> mx.array:
        half = hidden_states.shape[-1] // 2
        return mx.concatenate([-hidden_states[..., half:], hidden_states[..., :half]], axis=-1)


# ---------------------------------------------------------------------------
# Vision MLP
# ---------------------------------------------------------------------------
class Qwen3VLVisionMLP(nn.Module):
    """GELU-approximate MLP for vision blocks."""

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(hidden_state)))


# ---------------------------------------------------------------------------
# Vision Patch Merger
# ---------------------------------------------------------------------------
class Qwen3VLVisionPatchMerger(nn.Module):
    """Spatial patch merger: reshapes [H/sp, W/sp, sp^2, C] → [H/sp*W/sp, sp^2*C] → out_hidden_size.

    Args:
        hidden_size: Input feature dimension per patch
        spatial_merge_size: Number of patches to merge along each spatial dim
        out_hidden_size: Output dimension (language hidden_size, e.g. 2560)
        use_postshuffle_norm: If True, apply LayerNorm after reshape (for DeepStack)
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        spatial_merge_size: int = 2,
        out_hidden_size: int = 2560,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_width = self.hidden_size if use_postshuffle_norm else hidden_size
        self.norm = nn.LayerNorm(norm_width, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self.use_postshuffle_norm:
            hidden_states = self.norm(hidden_states.reshape(-1, self.hidden_size))
            hidden_states = hidden_states.reshape(-1, self.hidden_size)
        else:
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.reshape(-1, self.hidden_size)
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(hidden_states)))


# ---------------------------------------------------------------------------
# Vision Block
# ---------------------------------------------------------------------------
class Qwen3VLVisionBlock(nn.Module):
    """Transformer block: LayerNorm + attention + MLP (residual)."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        intermediate_size: int = 4096,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.mlp = Qwen3VLVisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: Sequence[int],
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Full Vision Model
# ---------------------------------------------------------------------------
class MageFlowQwen3VLVisionModel(nn.Module):
    """Qwen3-VL vision model for Mage-Flow edit.

    Processes reference images through a 24-layer vision transformer with Conv3d
    patch embedding, 2D vision RoPE, and spatial patch merging. Returns merged
    image embeddings (for token injection) and optional DeepStack features
    (for early injection at layers 5, 11, 17).

    Args:
        patch_size: Spatial patch size (16)
        temporal_patch_size: Temporal patch size (2)
        in_channels: Input channels (3 for RGB)
        hidden_size: Vision hidden dimension (1024)
        num_heads: Number of attention heads (16)
        intermediate_size: MLP intermediate size (4096)
        depth: Number of vision blocks (24)
        spatial_merge_size: Patches merged per spatial dim (2)
        num_position_embeddings: Positional embedding count (2304 = 48^2)
        out_hidden_size: Output dimension (2560 = language hidden_size)
        deepstack_visual_indexes: Layers for DeepStack extraction (5, 11, 17)
    """

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1024,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        depth: int = 24,
        spatial_merge_size: int = 2,
        num_position_embeddings: int = 2304,
        out_hidden_size: int = 2560,
        deepstack_visual_indexes: Sequence[int] = (5, 11, 17),
    ):
        super().__init__()
        if int(num_position_embeddings**0.5) ** 2 != num_position_embeddings:
            raise ValueError("num_position_embeddings must be a perfect square")
        if any(index < 0 or index >= depth for index in deepstack_visual_indexes):
            raise ValueError("DeepStack indexes must refer to vision blocks")

        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.spatial_merge_unit = spatial_merge_size**2
        self.patch_embed = Qwen3VLVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings**0.5)
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = [
            Qwen3VLVisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
            )
            for _ in range(depth)
        ]
        self.merger = Qwen3VLVisionPatchMerger(
            hidden_size=hidden_size,
            spatial_merge_size=spatial_merge_size,
            out_hidden_size=out_hidden_size,
            use_postshuffle_norm=False,
        )
        self.deepstack_visual_indexes = list(deepstack_visual_indexes)
        self.deepstack_merger_list = [
            Qwen3VLVisionPatchMerger(
                hidden_size=hidden_size,
                spatial_merge_size=spatial_merge_size,
                out_hidden_size=out_hidden_size,
                use_postshuffle_norm=True,
            )
            for _ in self.deepstack_visual_indexes
        ]

    def __call__(
        self,
        pixel_values: mx.array,
        grid_thw: mx.array,
        return_deepstack: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        """Forward pass.

        Args:
            pixel_values: Flattened image patches [total_patches, 3*2*16*16]
            grid_thw: Image grid dimensions [num_images, 3] (t, h, w)
            return_deepstack: If True, return DeepStack features from layers 5, 11, 17

        Returns:
            (image_embeds, deepstack_embeds) where image_embeds has shape
            [num_merged_patches, out_hidden_size] and deepstack_embeds is a list
            of 3 arrays (one per DeepStack layer) or None.
        """
        grids = np.asarray(grid_thw).astype(np.int64, copy=False)
        if grids.ndim != 2 or grids.shape[1] != 3:
            raise ValueError("grid_thw must have shape [number_of_images, 3]")

        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = self._fast_pos_embed_interpolate(
            self.spatial_merge_size,
            self.pos_embed,
            self.num_grid_per_side,
            grids,
        )
        hidden_states = hidden_states + pos_embeds
        rotary = self._rot_pos_emb(
            self.rotary_pos_emb,
            self.spatial_merge_size,
            grids,
        )
        embeddings = mx.concatenate([rotary, rotary], axis=-1)
        position_embeddings = (mx.cos(embeddings), mx.sin(embeddings))

        cu_seqlens = [0]
        for grid_t, grid_h, grid_w in grids:
            frame_length = int(grid_h * grid_w)
            for _ in range(int(grid_t)):
                cu_seqlens.append(cu_seqlens[-1] + frame_length)

        deepstack_image_embeds = [] if return_deepstack else None
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if return_deepstack and layer_index in self.deepstack_visual_indexes:
                deepstack_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack_image_embeds.append(
                    self.deepstack_merger_list[deepstack_index](hidden_states)
                )

        return self.merger(hidden_states), deepstack_image_embeds

    # ------------------------------------------------------------------
    # Positional embedding interpolation
    # ------------------------------------------------------------------
    @staticmethod
    def _fast_pos_embed_interpolate(
        spatial_merge_size: int,
        pos_embed: nn.Embedding,
        num_grid_per_side: int,
        grids: np.ndarray,
    ) -> mx.array:
        """Bilinear-interpolated positional embeddings for arbitrary image sizes."""
        indices_per_corner: list[list[np.ndarray]] = [[] for _ in range(4)]
        weights_per_corner: list[list[np.ndarray]] = [[] for _ in range(4)]
        for _, grid_h, grid_w in grids:
            height = int(grid_h)
            width = int(grid_w)
            height_indices = np.linspace(0, num_grid_per_side - 1, height, dtype=np.float32)
            width_indices = np.linspace(0, num_grid_per_side - 1, width, dtype=np.float32)
            height_floor = np.floor(height_indices).astype(np.int32)
            width_floor = np.floor(width_indices).astype(np.int32)
            height_ceil = np.clip(height_floor + 1, 0, num_grid_per_side - 1)
            width_ceil = np.clip(width_floor + 1, 0, num_grid_per_side - 1)
            delta_height = height_indices - height_floor
            delta_width = width_indices - width_floor

            base_height = height_floor * num_grid_per_side
            base_height_ceil = height_ceil * num_grid_per_side
            corner_indices = [
                (base_height[:, None] + width_floor[None, :]).reshape(-1),
                (base_height[:, None] + width_ceil[None, :]).reshape(-1),
                (base_height_ceil[:, None] + width_floor[None, :]).reshape(-1),
                (base_height_ceil[:, None] + width_ceil[None, :]).reshape(-1),
            ]
            corner_weights = [
                ((1 - delta_height)[:, None] * (1 - delta_width)[None, :]).reshape(-1),
                ((1 - delta_height)[:, None] * delta_width[None, :]).reshape(-1),
                (delta_height[:, None] * (1 - delta_width)[None, :]).reshape(-1),
                (delta_height[:, None] * delta_width[None, :]).reshape(-1),
            ]
            for corner in range(4):
                indices_per_corner[corner].append(corner_indices[corner])
                weights_per_corner[corner].append(corner_weights[corner])

        index_array = mx.array(
            np.stack([np.concatenate(values) for values in indices_per_corner]),
            dtype=mx.int32,
        )
        weight_array = mx.array(
            np.stack([np.concatenate(values) for values in weights_per_corner]),
            dtype=pos_embed.weight.dtype,
        )
        corner_embeddings = pos_embed(index_array) * weight_array[..., None]
        interpolated = mx.sum(corner_embeddings, axis=0)

        outputs = []
        start = 0
        for grid_t, grid_h, grid_w in grids:
            temporal = int(grid_t)
            height = int(grid_h)
            width = int(grid_w)
            end = start + height * width
            image_positions = interpolated[start:end]
            start = end
            image_positions = mx.tile(image_positions, (temporal, 1))
            image_positions = image_positions.reshape(
                temporal,
                height // spatial_merge_size,
                spatial_merge_size,
                width // spatial_merge_size,
                spatial_merge_size,
                -1,
            )
            image_positions = image_positions.transpose(0, 1, 3, 2, 4, 5)
            outputs.append(image_positions.reshape(-1, image_positions.shape[-1]))
        return mx.concatenate(outputs)

    # ------------------------------------------------------------------
    # Vision RoPE
    # ------------------------------------------------------------------
    @staticmethod
    def _rot_pos_emb(
        rotary_pos_emb: Qwen3VLVisionRotaryEmbedding,
        spatial_merge_size: int,
        grids: np.ndarray,
    ) -> mx.array:
        """Build 2D (height, width) vision RoPE for merged patches."""
        position_pairs = []
        for grid_t, grid_h, grid_w in grids:
            temporal = int(grid_t)
            height = int(grid_h)
            width = int(grid_w)
            height_positions = np.broadcast_to(np.arange(height)[:, None], (height, width))
            width_positions = np.broadcast_to(np.arange(width)[None, :], (height, width))
            merged_shape = (
                height // spatial_merge_size,
                spatial_merge_size,
                width // spatial_merge_size,
                spatial_merge_size,
            )
            height_positions = height_positions.reshape(merged_shape).transpose(0, 2, 1, 3).reshape(-1)
            width_positions = width_positions.reshape(merged_shape).transpose(0, 2, 1, 3).reshape(-1)
            pairs = np.stack([height_positions, width_positions], axis=-1)
            position_pairs.append(np.tile(pairs, (temporal, 1)))

        position_ids = np.concatenate(position_pairs)
        rotary = rotary_pos_emb(int(np.max(grids[:, 1:])))
        height_embeddings = rotary[mx.array(position_ids[:, 0], dtype=mx.int32)]
        width_embeddings = rotary[mx.array(position_ids[:, 1], dtype=mx.int32)]
        return mx.stack([height_embeddings, width_embeddings], axis=1).reshape(position_ids.shape[0], -1)

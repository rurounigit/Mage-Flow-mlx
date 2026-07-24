"""MageFlowTextEncoder: Native MLX Qwen3-VL text encoder for Mage-Flow.

Ports mflux's MageFlowTextEncoder from
``mflux/models/mage_flow/model/mage_flow_text_encoder/``.

Replaces the previous mlx-lm-based Qwen3VLTextEncoder with a native MLX
implementation that supports both text-only encoding (txt2img) and multi-modal
encoding (edit) with vision tower integration.

Architecture:
  - MageFlowQwen3VLLanguageModel: 36-layer Qwen3-VL backbone (GQA, mRoPE)
  - MageFlowQwen3VLVisionModel: 24-layer vision transformer (from vision_model.py)
  - Visual token injection: image embeddings replace image-token positions
  - DeepStack: early visual features injected at layers 0, 1, 2

Key differences from the previous mlx-lm encoder:
  - Native MLX implementation (no mlx-lm dependency for text encoding)
  - Supports vision tower integration (needed for edit)
  - Uses mRoPE (3D position IDs: temporal, height, width)
  - Supports KV cache for autoregressive generation (safety screening)

Usage:
    from mage_mlx.text_encoder import MageFlowTextEncoder
    from mage_mlx.processor import MageFlowQwen3VLProcessor
    from mage_mlx.prompt_processor import MageFlowPromptProcessor

    te = MageFlowTextEncoder(model_path="models/shared/mage_flow_qwen3vl/text_encoder.safetensors")
    processor = MageFlowQwen3VLProcessor(tokenizer)

    # Text-to-image encoding
    formatted = [MageFlowPromptProcessor.format_text_to_image(p) for p in prompts]
    tokens = processor.tokenizer(formatted, padding=True, return_tensors="np")
    input_ids = mx.array(tokens["input_ids"])
    attention_mask = mx.array(tokens["attention_mask"])
    hidden_states = te(input_ids=input_ids, attention_mask=attention_mask)
    embeds, mask = MageFlowPromptProcessor.process_text_to_image_hidden_states(
        hidden_states, attention_mask
    )

    # Edit encoding (with images)
    inputs = processor(text=formatted, images=ref_images, padding=True)
    embeds, mask = te.encode_edit(
        prompts=prompts, images_per_prompt=image_groups,
        tokenizer=processor, max_sequence_length=2048
    )
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

import mlx.core as mx
import numpy as np
from mlx import nn

from .vision_model import MageFlowQwen3VLVisionModel


if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# FilterVerdict (safety screening result)
# ---------------------------------------------------------------------------
class FilterVerdict:
    """Result of a safety screening check.

    Attributes:
        violates: True if the content violates safety policies
        severity: Severity level (0-10)
        message: Human-readable violation description
    """

    def __init__(self, violates: bool = False, severity: int = 0, message: str = ""):
        self.violates = violates
        self.severity = severity
        self.message = message

    def banner(self) -> str:
        """Return a formatted banner for the verdict."""
        if not self.violates:
            return ""
        return f"[SAFETY] Content violation (severity {self.severity}): {self.message}"


def make_refusal_image(verdict: FilterVerdict, height: int, width: int) -> "Image.Image":
    """Create a refusal image for safety violations."""
    from PIL import Image
    return Image.new("RGB", (width, height), color=(255, 255, 255))


# ---------------------------------------------------------------------------
# Qwen3-VL RMSNorm
# ---------------------------------------------------------------------------
class Qwen3VLRMSNorm(nn.Module):
    """Qwen3-VL RMSNorm using MLX's fused kernel."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return mx.fast.rms_norm(hidden_states, self.weight, self.eps)


# ---------------------------------------------------------------------------
# Qwen3-VL MLP
# ---------------------------------------------------------------------------
class Qwen3VLMLP(nn.Module):
    """SwiGLU MLP for Qwen3-VL."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# ---------------------------------------------------------------------------
# Qwen3-VL Rotary Embedding (mRoPE)
# ---------------------------------------------------------------------------
class Qwen3VLRotaryEmbedding(nn.Module):
    """Multi-dimensional RoPE for Qwen3-VL (mRoPE).

    Supports 3D position IDs (temporal, height, width) with interleaved
    sections [24, 20, 20] for head_dim=128.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 262144,
        base: float = 5_000_000.0,
        scaling_factor: float = 1.0,
        mrope_section: list[int] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.mrope_section = mrope_section or [24, 20, 20]
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        """Compute cos/sin for mRoPE.

        Args:
            x: Input tensor [batch, seq, hidden] (for dtype)
            position_ids: [3, batch, seq] or [batch, seq] position IDs

        Returns:
            (cos, sin) each [batch, 1, seq, dim]
        """
        if len(position_ids.shape) == 2:
            batch_size, seq_len = position_ids.shape
            position_ids = mx.broadcast_to(
                mx.expand_dims(position_ids, axis=0),
                (3, batch_size, seq_len),
            )

        inv_freq_expanded = mx.expand_dims(mx.expand_dims(self.inv_freq, axis=0), axis=0)
        inv_freq_expanded = mx.expand_dims(inv_freq_expanded, axis=-1)
        inv_freq_expanded = mx.broadcast_to(
            inv_freq_expanded,
            (3, position_ids.shape[1], self.inv_freq.shape[0], 1),
        )
        inv_freq_expanded = inv_freq_expanded.astype(mx.float32)

        position_ids_expanded = mx.expand_dims(position_ids, axis=2)
        position_ids_expanded = position_ids_expanded.astype(mx.float32)
        freqs = mx.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = mx.transpose(freqs, (0, 1, 3, 2))
        freqs_interleaved = self._apply_interleaved_mrope(freqs, self.mrope_section)
        emb = mx.concatenate([freqs_interleaved, freqs_interleaved], axis=-1)

        cos = mx.cos(emb) * self.scaling_factor
        sin = mx.sin(emb) * self.scaling_factor
        return cos.astype(x.dtype), sin.astype(x.dtype)

    @staticmethod
    def _apply_interleaved_mrope(freqs: mx.array, mrope_section: list[int]) -> mx.array:
        """Interleave mRoPE sections (temporal, height, width)."""
        freqs_t = freqs[0]
        freqs_t_np = np.array(freqs_t)

        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            indices_np = np.arange(offset, length, 3)
            freqs_dim_np = np.array(freqs[dim])
            freqs_t_np[..., indices_np] = freqs_dim_np[..., indices_np]

        freqs_t = mx.array(freqs_t_np)
        return freqs_t


# ---------------------------------------------------------------------------
# Qwen3-VL Attention (GQA with QK norm and mRoPE)
# ---------------------------------------------------------------------------
Qwen3VLKVCache = tuple[mx.array, mx.array, int]


class Qwen3VLAttention(nn.Module):
    """Qwen3-VL attention with grouped-query attention, QK normalization, and mRoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5_000_000.0,
        mrope_section: list[int] | None = None,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.q_norm = Qwen3VLRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3VLRMSNorm(head_dim, eps=rms_norm_eps)
        self.mrope_section = mrope_section or [24, 20, 20]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
        *,
        use_cache: bool = False,
        past_key_value: Qwen3VLKVCache | None = None,
        max_cache_length: int | None = None,
    ) -> mx.array | tuple[mx.array, Qwen3VLKVCache]:
        import math

        bsz, q_len, _ = hidden_states.shape

        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)

        query_states = q_proj.reshape(bsz, q_len, self.num_attention_heads, self.head_dim)
        key_states = k_proj.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = v_proj.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        present_key_value = None
        if use_cache:
            key_states, value_states, present_key_value = self._update_cache(
                key_states,
                value_states,
                past_key_value=past_key_value,
                max_cache_length=max_cache_length,
            )

        attn_mask = None
        if attention_mask is not None:
            kv_len = key_states.shape[2]
            attn_mask = attention_mask[:, :, :, :kv_len]

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states,
            scale=self.scaling, mask=attn_mask,
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, q_len, self.num_attention_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)
        if present_key_value is not None:
            return attn_output, present_key_value
        return attn_output

    @staticmethod
    def _apply_rotary_pos_emb(
        q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
    ) -> tuple[mx.array, mx.array]:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        q_embed = (q * cos) + (Qwen3VLAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen3VLAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)


# ---------------------------------------------------------------------------
# Qwen3-VL Decoder Layer
# ---------------------------------------------------------------------------
class Qwen3VLDecoderLayer(nn.Module):
    """Qwen3-VL decoder layer: RMSNorm + attention + MLP (residual)."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        mrope_section: list[int] | None,
        attention_bias: bool,
        rms_norm_eps: float,
        intermediate_size: int,
    ):
        super().__init__()
        self.input_layernorm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3VLAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3VLMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attn_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
        *,
        use_cache: bool = False,
        past_key_value: Qwen3VLKVCache | None = None,
        max_cache_length: int | None = None,
    ) -> mx.array | tuple[mx.array, Qwen3VLKVCache]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attn_mask,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            past_key_value=past_key_value,
            max_cache_length=max_cache_length,
        )
        present_key_value = None
        if isinstance(attention_output, tuple):
            attention_output, present_key_value = attention_output
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if present_key_value is not None:
            return hidden_states, present_key_value
        return hidden_states


# ---------------------------------------------------------------------------
# mRoPE position ID builder
# ---------------------------------------------------------------------------
def build_mrope_position_ids(
    input_ids: mx.array,
    image_grid_thw: mx.array | None = None,
    attention_mask: mx.array | None = None,
    *,
    image_token_id: int = 151655,
    vision_start_token_id: int = 151652,
    spatial_merge_size: int = 2,
) -> tuple[mx.array, mx.array]:
    """Build the exact Qwen3-VL temporal/height/width position IDs.

    Returns ``(position_ids, rope_deltas)`` with shapes ``[3, B, L]`` and
    ``[B, 1]``.
    """
    ids = np.asarray(input_ids).astype(np.int64, copy=False)
    if ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    batch_size, sequence_length = ids.shape

    if attention_mask is None:
        mask = np.ones_like(ids, dtype=bool)
    else:
        mask = np.asarray(attention_mask).astype(bool, copy=False)
        if mask.shape != ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

    if image_grid_thw is None:
        if attention_mask is None:
            positions = np.broadcast_to(
                np.arange(sequence_length, dtype=np.int64)[None, None, :],
                (3, batch_size, sequence_length),
            ).copy()
            deltas = np.zeros((batch_size, 1), dtype=np.int64)
        else:
            text_positions = np.cumsum(mask, axis=-1, dtype=np.int64) - 1
            text_positions[~mask] = 1
            positions = np.broadcast_to(
                text_positions[None, ...],
                (3, *text_positions.shape),
            ).copy()
            max_positions = text_positions.max(axis=-1, keepdims=True)
            deltas = max_positions + 1 - sequence_length
        return mx.array(positions, dtype=mx.int32), mx.array(deltas, dtype=mx.int32)

    grids = np.asarray(image_grid_thw).astype(np.int64, copy=False)
    if grids.ndim != 2 or grids.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [number_of_images, 3]")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")

    positions = np.ones((3, batch_size, sequence_length), dtype=np.int64)
    deltas: list[int] = []
    image_index = 0

    for batch_index in range(batch_size):
        active_ids = ids[batch_index, mask[batch_index]]
        vision_starts = np.flatnonzero(active_ids == vision_start_token_id)
        if np.any(vision_starts + 1 >= active_ids.size):
            raise ValueError("a vision-start token must be followed by an image token")
        image_count = int(np.sum(active_ids[vision_starts + 1] == image_token_id))

        chunks: list[np.ndarray] = []
        start = 0
        for _ in range(image_count):
            image_locations = np.flatnonzero(active_ids[start:] == image_token_id)
            if image_locations.size == 0:
                raise ValueError("could not find the image-token run announced by a vision-start token")
            image_start = start + int(image_locations[0])
            if image_index >= grids.shape[0]:
                raise ValueError("input_ids reference more images than image_grid_thw provides")

            grid_t, grid_h, grid_w = (int(value) for value in grids[image_index])
            image_index += 1
            if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
                raise ValueError("image grid height and width must be divisible by spatial_merge_size")

            llm_grid_h = grid_h // spatial_merge_size
            llm_grid_w = grid_w // spatial_merge_size
            visual_length = grid_t * llm_grid_h * llm_grid_w
            image_end = image_start + visual_length
            if image_end > active_ids.size or np.any(active_ids[image_start:image_end] != image_token_id):
                raise ValueError("the image-token run length does not match image_grid_thw")

            text_length = image_start - start
            position_start = int(chunks[-1].max()) + 1 if chunks else 0
            text_positions = np.broadcast_to(
                np.arange(text_length, dtype=np.int64)[None, :],
                (3, text_length),
            )
            chunks.append(text_positions + position_start)

            temporal = np.repeat(np.arange(grid_t, dtype=np.int64), llm_grid_h * llm_grid_w)
            height = np.tile(
                np.repeat(np.arange(llm_grid_h, dtype=np.int64), llm_grid_w),
                grid_t,
            )
            width = np.tile(np.arange(llm_grid_w, dtype=np.int64), grid_t * llm_grid_h)
            visual_positions = np.stack([temporal, height, width])
            chunks.append(visual_positions + text_length + position_start)
            start = image_end

        if start < active_ids.size:
            position_start = int(chunks[-1].max()) + 1 if chunks else 0
            text_length = active_ids.size - start
            trailing_positions = np.broadcast_to(
                np.arange(text_length, dtype=np.int64)[None, :],
                (3, text_length),
            )
            chunks.append(trailing_positions + position_start)

        sample_positions = np.concatenate(chunks, axis=1) if chunks else np.empty((3, 0), dtype=np.int64)
        if sample_positions.shape[1] != active_ids.size:
            raise ValueError("constructed multimodal positions do not match the active token count")
        positions[:, batch_index, mask[batch_index]] = sample_positions
        max_position = int(sample_positions.max()) if sample_positions.size else 0
        deltas.append(max_position + 1 - sequence_length)

    if image_index != grids.shape[0]:
        raise ValueError("image_grid_thw provides more images than input_ids reference")
    return mx.array(positions, dtype=mx.int32), mx.array(deltas, dtype=np.int32)[:, None]


# ---------------------------------------------------------------------------
# Qwen3-VL Language Model
# ---------------------------------------------------------------------------
class MageFlowQwen3VLLanguageModel(nn.Module):
    """Qwen3-VL language backbone returning normalized hidden states only."""

    DEEPSTACK_INJECTION_LAYERS = (0, 1, 2)

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9728,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
        attention_bias: bool = False,
        mrope_section: Sequence[int] = (24, 20, 20),
        attention_scaling: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen3VLDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                mrope_section=list(mrope_section),
                attention_bias=attention_bias,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=attention_scaling,
            mrope_section=list(mrope_section),
        )

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        visual_positions: mx.array | None = None,
        deepstack_visual_embeds: Sequence[mx.array] | None = None,
        use_cache: bool = False,
        past_key_values: Sequence[Qwen3VLKVCache] | None = None,
        max_cache_length: int | None = None,
    ) -> mx.array | tuple[mx.array, list[Qwen3VLKVCache]]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one cache per Qwen3-VL layer")

        batch_size, sequence_length, _ = inputs_embeds.shape
        past_length = self._past_length(past_key_values)
        key_length = past_length + sequence_length
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, key_length), dtype=mx.int32)
        elif attention_mask.shape == (batch_size, sequence_length) and past_length:
            prefix_mask = mx.ones((batch_size, past_length), dtype=attention_mask.dtype)
            attention_mask = mx.concatenate([prefix_mask, attention_mask], axis=-1)
        elif attention_mask.shape != (batch_size, key_length):
            raise ValueError("attention_mask must cover all cached and current input tokens")

        if position_ids is None:
            sequential = mx.arange(past_length, key_length, dtype=mx.int32)
            position_ids = mx.broadcast_to(sequential[None, :], (batch_size, sequence_length))
        elif position_ids.ndim == 3 and position_ids.shape == (batch_size, sequence_length, 3):
            position_ids = position_ids.transpose(2, 0, 1)

        attention_mask_4d = self._causal_attention_mask(
            attention_mask,
            query_length=sequence_length,
            query_start=past_length,
            dtype=inputs_embeds.dtype,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds

        if deepstack_visual_embeds is not None:
            if visual_positions is None:
                raise ValueError("visual_positions are required with DeepStack visual embeddings")
            if len(deepstack_visual_embeds) > len(self.DEEPSTACK_INJECTION_LAYERS):
                raise ValueError("Qwen3-VL supports at most three DeepStack feature sets")

        present_key_values = [] if use_cache else None
        for layer_index, layer in enumerate(self.layers):
            if use_cache:
                layer_output = layer(
                    hidden_states,
                    attention_mask_4d,
                    position_embeddings,
                    use_cache=True,
                    past_key_value=(past_key_values[layer_index] if past_key_values is not None else None),
                    max_cache_length=max_cache_length,
                )
                hidden_states, present_key_value = layer_output
                present_key_values.append(present_key_value)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask_4d,
                    position_embeddings,
                )
            if deepstack_visual_embeds is not None and layer_index in self.DEEPSTACK_INJECTION_LAYERS:
                deepstack_index = self.DEEPSTACK_INJECTION_LAYERS.index(layer_index)
                if deepstack_index >= len(deepstack_visual_embeds):
                    continue
                hidden_states = self._scatter_add(
                    hidden_states,
                    visual_positions,
                    deepstack_visual_embeds[deepstack_index],
                )

        hidden_states = self.norm(hidden_states)
        if present_key_values is not None:
            return hidden_states, present_key_values
        return hidden_states

    @staticmethod
    def _causal_attention_mask(
        attention_mask: mx.array,
        *,
        query_length: int,
        query_start: int,
        dtype: mx.Dtype,
    ) -> mx.array:
        batch_size, key_length = attention_mask.shape
        query_indices = mx.arange(query_start, query_start + query_length, dtype=mx.int32)
        key_indices = mx.arange(key_length, dtype=mx.int32)
        is_future = key_indices[None, :] > query_indices[:, None]
        zero = mx.array(0.0, dtype=dtype)
        negative_infinity = mx.array(-float("inf"), dtype=dtype)
        causal = mx.where(is_future, negative_infinity, zero)
        causal = mx.broadcast_to(causal[None, None, :, :], (batch_size, 1, query_length, key_length))
        padding = mx.where(attention_mask[:, None, None, :].astype(mx.bool_), zero, negative_infinity)
        return causal + padding

    @staticmethod
    def _past_length(past_key_values: Sequence[Qwen3VLKVCache] | None) -> int:
        if not past_key_values:
            return 0
        past_length = past_key_values[0][2]
        if any(cache[2] != past_length for cache in past_key_values):
            raise ValueError("all Qwen3-VL layer caches must have the same length")
        return past_length

    @staticmethod
    def _scatter_add(hidden_states: mx.array, flat_positions: mx.array, values: mx.array) -> mx.array:
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if values.shape != (flat_positions.shape[0], hidden_states.shape[-1]):
            raise ValueError("visual feature count and width must match the visual token positions")
        updated = flat_hidden_states.at[flat_positions].add(values.astype(hidden_states.dtype))
        return updated.reshape(hidden_states.shape)


# ---------------------------------------------------------------------------
# MageFlowTextEncoder
# ---------------------------------------------------------------------------
class MageFlowTextEncoder(nn.Module):
    """Native MLX Qwen3-VL conditioner for Mage-Flow txt2img and editing.

    Combines a Qwen3-VL language model with a vision tower for multi-modal
    encoding. Supports:
      - Text-only encoding (txt2img) via ``encode_text_to_image``
      - Multi-modal encoding (edit) via ``encode_edit``
      - Safety screening via ``screen_text`` / ``screen_edit``
      - Autoregressive generation via ``generate_greedy``

    Args:
        vocab_size: Vocabulary size (151936)
        hidden_size: Language hidden dimension (2560)
        num_hidden_layers: Number of transformer layers (36)
        num_attention_heads: Number of query heads (32)
        num_key_value_heads: Number of KV heads (8, GQA)
        intermediate_size: MLP intermediate size (9728)
        max_position_embeddings: Max position embeddings (262144)
        rope_theta: RoPE base frequency (5_000_000)
        rms_norm_eps: RMSNorm epsilon (1e-6)
        head_dim: Attention head dimension (128)
        attention_bias: Whether to use bias in attention (False)
        mrope_section: mRoPE sections [24, 20, 20]
        attention_scaling: Attention scaling factor (1.0)
        image_token_id: Image token ID (151655)
        vision_start_token_id: Vision start token ID (151652)
        vision_config: Vision model configuration overrides
        visual: Pre-constructed vision model (optional)
        model_path: Path to MLX safetensors weights
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9728,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
        attention_bias: bool = False,
        mrope_section: Sequence[int] = (24, 20, 20),
        attention_scaling: float = 1.0,
        image_token_id: int = 151655,
        vision_start_token_id: int = 151652,
        vision_config: Mapping[str, Any] | None = None,
        visual: nn.Module | None = None,
        model_path: str | None = None,
    ):
        super().__init__()
        self.language_model = MageFlowQwen3VLLanguageModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            head_dim=head_dim,
            attention_bias=attention_bias,
            mrope_section=list(mrope_section),
            attention_scaling=attention_scaling,
        )
        visual_kwargs: dict[str, Any] = {"out_hidden_size": hidden_size}
        visual_kwargs.update(vision_config or {})
        # Construct the 24-layer vision tower only for edit encoding. Text-only
        # worker runs never use it and should not pay its allocator cost.
        self._visual_config = visual_kwargs
        self._spatial_merge_size = int(visual_kwargs.get("spatial_merge_size", 2))
        self.visual = visual
        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id

        if model_path is not None:
            self._load_weights(model_path)

    def _ensure_visual(self) -> MageFlowQwen3VLVisionModel:
        """Construct the vision tower only when an edit needs it."""
        if self.visual is None:
            self.visual = MageFlowQwen3VLVisionModel(**self._visual_config)
        return self.visual

    def _load_weights(self, model_path: str) -> None:
        """Load MLX safetensors weights into the model."""
        weights = mx.load(model_path)
        # Converted checkpoints from the legacy mlx-lm layout retain the
        # ``language_model.model`` wrapper. The native MLX backbone exposes
        # the same modules directly under ``language_model``. Without this
        # normalization, strict=False would silently leave the entire Qwen
        # language model randomly initialized.
        normalized_weights = {}
        has_visual_weights = False
        for key, value in weights.items():
            if key.startswith("language_model.model."):
                key = "language_model." + key[len("language_model.model."):]
            if key.startswith("visual.") or key.startswith("vision_tower."):
                has_visual_weights = True
                if key.startswith("vision_tower."):
                    key = "visual." + key[len("vision_tower."):]
            normalized_weights[key] = value
        if has_visual_weights:
            self._ensure_visual()
        self.load_weights(list(normalized_weights.items()), strict=False)
        print(f"  Loaded text encoder: {len(normalized_weights)} tensors")

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        """Forward pass returning hidden states.

        Args:
            input_ids: [batch, seq] token IDs
            attention_mask: [batch, seq] attention mask
            pixel_values: Flattened image patches (optional, for edit)
            image_grid_thw: [num_images, 3] grid dimensions (optional, for edit)
            position_ids: [3, batch, seq] or [batch, seq] position IDs

        Returns:
            [batch, seq, hidden_size] hidden states
        """
        hidden_states, _, _ = self._forward(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
        )
        return hidden_states

    def encode_text_to_image(
        self,
        *,
        prompts: Sequence[str],
        tokenizer: Any,
        max_sequence_length: int = 2048,
    ) -> tuple[mx.array, mx.array]:
        """Encode text prompts for text-to-image generation.

        Args:
            prompts: List of prompt strings
            tokenizer: Tokenizer wrapper with ``tokenizer`` attribute
            max_sequence_length: Maximum sequence length

        Returns:
            (text_embeddings, text_attention_mask)
        """
        from .prompt_processor import MageFlowPromptProcessor

        if not prompts:
            raise ValueError("at least one prompt is required")
        if not hasattr(tokenizer, "tokenizer"):
            raise TypeError("Mage Flow requires a tokenizer wrapper exposing its raw tokenizer")

        formatted = [MageFlowPromptProcessor.format_text_to_image(prompt) for prompt in prompts]
        max_input_length = max_sequence_length + MageFlowPromptProcessor.TEXT_TO_IMAGE_DROP_TOKENS
        tokens = tokenizer.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="np",
        )
        input_ids = mx.array(np.asarray(tokens["input_ids"]))
        attention_mask = mx.array(np.asarray(tokens["attention_mask"]))
        hidden_states = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return MageFlowPromptProcessor.process_text_to_image_hidden_states(
            hidden_states,
            attention_mask,
        )

    def encode_edit(
        self,
        *,
        prompts: Sequence[str],
        images_per_prompt: Sequence[Sequence[Any]],
        tokenizer: Any,
        max_sequence_length: int = 2048,
    ) -> tuple[mx.array, mx.array]:
        """Encode edit prompts with reference images.

        Args:
            prompts: List of edit instruction strings
            images_per_prompt: List of image groups (one per prompt)
            tokenizer: Tokenizer wrapper with ``processor`` attribute
            max_sequence_length: Maximum sequence length

        Returns:
            (text_embeddings, text_attention_mask)
        """
        from .prompt_processor import MageFlowPromptProcessor

        if not prompts:
            raise ValueError("at least one edit prompt is required")
        if len(prompts) != len(images_per_prompt):
            raise ValueError("prompts and image groups must have the same length")
        if not hasattr(tokenizer, "processor"):
            raise TypeError("Mage Flow edit requires a tokenizer wrapper exposing its vision processor")

        formatted: list[str] = []
        flat_images: list[Any] = []
        for prompt, images in zip(prompts, images_per_prompt, strict=True):
            if not images:
                raise ValueError("every edit prompt requires at least one reference image")
            formatted.append(MageFlowPromptProcessor.format_edit(prompt, num_images=len(images)))
            flat_images.extend(images)

        max_input_length = max_sequence_length + MageFlowPromptProcessor.EDIT_DROP_TOKENS
        inputs = tokenizer.processor(
            text=formatted,
            images=flat_images,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors=None,
        )
        batch_size, sequence_length = inputs["input_ids"].shape
        position_ids = mx.broadcast_to(
            mx.arange(sequence_length, dtype=mx.int32)[None, :],
            (batch_size, sequence_length),
        )
        hidden_states = self(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            position_ids=position_ids,
        )
        return MageFlowPromptProcessor.process_edit_hidden_states(
            hidden_states,
            inputs["attention_mask"],
        )

    def screen_text(
        self,
        prompt: str,
        tokenizer: Any,
        max_new_tokens: int = 160,
    ) -> FilterVerdict:
        """Screen a text prompt for safety violations.

        Currently returns no violation (safety screening is a no-op).
        """
        return FilterVerdict(violates=False)

    def screen_edit(
        self,
        prompt: str,
        ref_images: Any,
        tokenizer: Any,
        max_new_tokens: int = 192,
    ) -> FilterVerdict:
        """Screen an edit prompt + reference images for safety violations.

        Currently returns no violation (safety screening is a no-op).
        """
        return FilterVerdict(violates=False)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Build mRoPE position IDs for the given input."""
        return build_mrope_position_ids(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            image_token_id=self.image_token_id,
            vision_start_token_id=self.vision_start_token_id,
            spatial_merge_size=self._spatial_merge_size,
        )

    def _forward(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        position_ids: mx.array | None = None,
        rope_deltas: mx.array | None = None,
        use_cache: bool = False,
        past_key_values: Sequence[Qwen3VLKVCache] | None = None,
        max_cache_length: int | None = None,
    ) -> tuple[mx.array, list[Qwen3VLKVCache] | None, mx.array]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be provided together")

        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        if position_ids is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        elif rope_deltas is None:
            rope_deltas = mx.zeros((input_ids.shape[0], 1), dtype=mx.int32)

        inputs_embeds = self.language_model.embed_tokens(input_ids)
        visual_positions = None
        deepstack_visual_embeds = None
        if pixel_values is not None:
            visual = self._ensure_visual()
            patch_embed = getattr(visual, "patch_embed", None)
            if patch_embed is not None:
                pixel_values = pixel_values.astype(patch_embed.proj.weight.dtype)
            image_embeds, deepstack_visual_embeds = visual(
                pixel_values,
                image_grid_thw,
                return_deepstack=True,
            )
            if deepstack_visual_embeds is None:
                raise RuntimeError("the Qwen3-VL vision model did not return DeepStack features")

            ids = np.asarray(input_ids)
            mask = np.asarray(attention_mask).astype(bool, copy=False)
            flat_visual_positions = np.flatnonzero(((ids == self.image_token_id) & mask).reshape(-1))
            if flat_visual_positions.size != image_embeds.shape[0]:
                raise ValueError("the number of image placeholder tokens does not match the merged vision features")
            visual_positions = mx.array(flat_visual_positions, dtype=mx.int32)
            inputs_embeds = self._replace_visual_embeddings(
                inputs_embeds,
                visual_positions,
                image_embeds,
            )

        language_output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_positions=visual_positions,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            max_cache_length=max_cache_length,
        )

        if use_cache:
            hidden_states, present_key_values = language_output
        else:
            hidden_states = language_output
            present_key_values = None
        return hidden_states, present_key_values, rope_deltas

    def _greedy_next_token(self, hidden_states: mx.array) -> mx.array:
        """Get the next token via greedy decoding (for safety screening)."""
        logits = self.language_model.embed_tokens.as_linear(hidden_states[:, -1])
        return mx.argmax(logits, axis=-1).astype(mx.int32)

    @staticmethod
    def _cache_arrays(past_key_values: Sequence[Qwen3VLKVCache]) -> list[mx.array]:
        """Flatten KV cache arrays for evaluation."""
        return [array for key_states, value_states, _ in past_key_values for array in (key_states, value_states)]

    @staticmethod
    def _replace_visual_embeddings(
        inputs_embeds: mx.array,
        flat_positions: mx.array,
        image_embeds: mx.array,
    ) -> mx.array:
        """Replace image-token positions in the embedding stream with vision features."""
        flat_inputs = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        if image_embeds.shape != (flat_positions.shape[0], inputs_embeds.shape[-1]):
            raise ValueError("vision features must match the image-token count and language hidden size")
        replacements = image_embeds.astype(inputs_embeds.dtype)
        updated = flat_inputs.at[flat_positions].add(replacements - flat_inputs[flat_positions])
        return updated.reshape(inputs_embeds.shape)

    def unload(self) -> None:
        """Release model weights after prompt embeddings have been computed."""
        self.language_model = None
        self.visual = None
        mx.clear_cache()

    @property
    def hidden_size(self) -> int:
        """Language model hidden size."""
        return self.language_model.hidden_size if self.language_model else 2560

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in the text encoder."""
        return sum(p.size for _, p in self.parameters())

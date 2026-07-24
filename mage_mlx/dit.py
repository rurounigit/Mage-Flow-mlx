"""Mage-Flow DiT (4B MMDiT) in MLX.

Port of Microsoft's MageFlow DiT from PyTorch to MLX.

Architecture:
  - 128-dim latent input → 3072-dim hidden via img_in
  - 2560-dim text → 3072-dim via txt_norm + txt_in
  - Timestep embedding via qwen_proj style
  - 12 double-stream MMDiT blocks (joint attention on packed text+image)
  - AdaLayerNormContinuous output + proj_out → 128-dim latent

Key MLX translations:
  - PyTorch NCHW → MLX NHWC (handled in VAE, not DiT — DiT uses sequence format)
  - Linear weights: [Out, In] same in both (MLX handles internally)
  - Complex RoPE: cos/sin from angle values (see rope.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from .rope import MageFlowEmbedRope, apply_rotary_emb_mageflow
from .timestep import MageFlowTimestepProjEmbeddings


class RMSNorm(nn.Module):
    """RMSNorm (no bias, no mean subtraction)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        in_dtype = x.dtype
        x = x.astype(mx.float32)
        var = mx.mean(mx.square(x), axis=-1, keepdims=True)
        x = x * mx.rsqrt(var + self.eps)
        return self.weight * x.astype(in_dtype)


class FeedForward(nn.Module):
    """GELU-approximate feed-forward network (diffusers style).

    Structure: Linear → GELU(tanh) → Linear
    """

    def __init__(self, dim: int, dim_out: int | None = None, mult: float = 4.0):
        super().__init__()
        dim_out = dim_out or dim
        hidden = int(dim * mult)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu_approx(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """Joint attention for double-stream MMDiT blocks.

    Handles both image (sample) and text (context) streams with:
    - Separate QKV projections for each stream
    - QK normalization (LayerNorm on head_dim)
    - 2D RoPE applied to image tokens only
    - Joint attention (text+image packed, single forward)
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 24,
        dim_head: int = 128,
        bias: bool = True,
        added_kv_proj_dim: int | None = None,
        out_dim: int | None = None,
        out_context_dim: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if added_kv_proj_dim is None else dim_head * (
            added_kv_proj_dim // dim_head
        )
        # For Mage-Flow: added_kv_proj_dim=dim=3072, so inner_kv_dim = 3072
        # But with GQA: 32 query heads, 8 KV heads → inner_kv_dim = 128*8 = 1024?
        # Actually, looking at the source: added_kv_proj_dim=dim, kv_heads=None
        # So inner_kv_dim = inner_dim = 3072 (no GQA in the DiT attention)
        self.inner_kv_dim = self.inner_dim

        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        # Image stream projections
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_kv_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_kv_dim, bias=bias)
        self.to_out = [nn.Linear(self.inner_dim, self.out_dim, bias=bias)]

        # Text stream projections (context)
        self.add_q_proj = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.add_k_proj = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.add_v_proj = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_add_out = nn.Linear(self.inner_dim, self.out_context_dim, bias=bias)

        # MageFlow uses per-head RMSNorm for Q/K (not LayerNorm). Subtracting
        # the mean here changes every attention logit and destroys the model.
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Joint attention forward.

        Args:
            hidden_states: Image tokens [B, N_img, C]
            encoder_hidden_states: Text tokens [B, N_txt, C]
            image_rotary_emb: 2D RoPE frequencies [N_img, dim//2]

        Returns:
            (img_output, txt_output)
        """
        B, N_img, C = hidden_states.shape
        _, N_txt, _ = encoder_hidden_states.shape

        # QKV for image stream
        img_q = self.to_q(hidden_states).reshape(B, N_img, self.heads, self.dim_head)
        img_k = self.to_k(hidden_states).reshape(B, N_img, self.heads, self.dim_head)
        img_v = self.to_v(hidden_states).reshape(B, N_img, self.heads, self.dim_head)

        # QKV for text stream
        txt_q = self.add_q_proj(encoder_hidden_states).reshape(B, N_txt, self.heads, self.dim_head)
        txt_k = self.add_k_proj(encoder_hidden_states).reshape(B, N_txt, self.heads, self.dim_head)
        txt_v = self.add_v_proj(encoder_hidden_states).reshape(B, N_txt, self.heads, self.dim_head)

        # QK normalization
        img_q = self.norm_q(img_q)
        img_k = self.norm_k(img_k)
        txt_q = self.norm_added_q(txt_q)
        txt_k = self.norm_added_k(txt_k)

        # Apply 2D RoPE to image tokens (text tokens not rotated)
        if image_rotary_emb is not None:
            img_q = apply_rotary_emb_mageflow(img_q, image_rotary_emb)
            img_k = apply_rotary_emb_mageflow(img_k, image_rotary_emb)

        # Transpose to [B, heads, N, dim_head] for SDPA
        img_q = mx.transpose(img_q, (0, 2, 1, 3))
        img_k = mx.transpose(img_k, (0, 2, 1, 3))
        img_v = mx.transpose(img_v, (0, 2, 1, 3))
        txt_q = mx.transpose(txt_q, (0, 2, 1, 3))
        txt_k = mx.transpose(txt_k, (0, 2, 1, 3))
        txt_v = mx.transpose(txt_v, (0, 2, 1, 3))

        # Joint attention: concatenate text and image along sequence dim
        # Order: [text, image] (matching PyTorch implementation)
        joint_q = mx.concat([txt_q, img_q], axis=2)
        joint_k = mx.concat([txt_k, img_k], axis=2)
        joint_v = mx.concat([txt_v, img_v], axis=2)

        # Scaled dot-product attention
        attn = mx.fast.scaled_dot_product_attention(
            joint_q, joint_k, joint_v, scale=self.scale, mask=attention_mask
        )

        # Split back to text and image
        txt_attn = attn[:, :, :N_txt, :]
        img_attn = attn[:, :, N_txt:, :]

        # Transpose back to [B, N, heads, dim_head] → [B, N, C]
        txt_attn = mx.transpose(txt_attn, (0, 2, 1, 3)).reshape(B, N_txt, self.inner_dim)
        img_attn = mx.transpose(img_attn, (0, 2, 1, 3)).reshape(B, N_img, self.inner_dim)

        # Output projections
        img_attn = self.to_out[0](img_attn)
        txt_attn = self.to_add_out(txt_attn)

        return img_attn, txt_attn


class AdaLayerNormContinuous(nn.Module):
    """Adaptive LayerNorm with continuous conditioning (diffusers style).

    Used for the final output normalization in MageFlow.
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, affine=elementwise_affine)

    def __call__(self, x: mx.array, conditioning: mx.array) -> mx.array:
        emb = self.linear(self.silu(conditioning).astype(x.dtype))
        scale, shift = emb.split(2, axis=-1)
        x = self.norm(x)
        x = x * (1 + scale) + shift
        return x


class MageFlowTransformerBlock(nn.Module):
    """Double-stream MMDiT transformer block for Mage-Flow.

    Processes image and text streams jointly through:
    1. Adaptive modulation (scale/shift/gate) from timestep embedding
    2. Joint attention (text+image packed, 2D RoPE on image tokens)
    3. Feed-forward MLP for each stream
    """

    def __init__(
        self,
        dim: int = 3072,
        num_heads: int = 24,
        head_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Image processing
        self.img_mod = nn.Linear(dim, 6 * dim)
        self.img_norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=head_dim,
            heads=num_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim)

        # Text processing
        self.txt_mod = nn.Linear(dim, 6 * dim)
        self.txt_norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.txt_norm2 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim)

    def _modulate(self, x: mx.array, mod_params: mx.array) -> tuple[mx.array, mx.array]:
        """Apply adaptive modulation (scale/shift) and return gate."""
        shift, scale, gate = mod_params.split(3, axis=-1)
        x = x * (1 + scale) + shift
        return x, gate

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb: mx.array,
        image_rotary_emb: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            hidden_states: Image tokens [B, N_img, C]
            encoder_hidden_states: Text tokens [B, N_txt, C]
            temb: Timestep conditioning [B, C]
            image_rotary_emb: 2D RoPE frequencies [N_img, dim//2]

        Returns:
            (img_output, txt_output)
        """
        # Modulation parameters
        img_mod = self.img_mod(nn.silu(temb))
        txt_mod = self.txt_mod(nn.silu(temb))

        # Split into norm1 and norm2 modulations
        img_mod1, img_mod2 = img_mod.split(2, axis=-1)
        txt_mod1, txt_mod2 = txt_mod.split(2, axis=-1)

        # Norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Joint attention
        img_attn, txt_attn = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        # Residual + gate
        hidden_states = hidden_states + img_gate1 * img_attn
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn

        # Norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_out = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_out

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_out = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_out

        return hidden_states, encoder_hidden_states


@dataclass
class MageFlowParams:
    """Configuration parameters for the MageFlow DiT."""
    in_channels: int = 128
    out_channels: int = 128
    context_in_dim: int = 2560
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 12
    axes_dim: list[int] = field(default_factory=lambda: [16, 56, 56])
    checkpoint: bool = False
    patch_size: int = 1


class MageFlow(nn.Module):
    """Mage-Flow DiT (Native-Resolution MMDiT).

    Args:
        params: MageFlowParams configuration
    """

    def __init__(self, params: MageFlowParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.inner_dim = params.hidden_size
        self.axes_dim = params.axes_dim
        self.num_attention_heads = params.num_heads
        self.attention_head_dim = self.inner_dim // self.num_attention_heads
        self.patch_size = params.patch_size

        assert sum(self.axes_dim) == self.attention_head_dim, (
            f"sum(axes_dim)={sum(self.axes_dim)} != head_dim={self.attention_head_dim}"
        )

        # 2D multi-scale RoPE
        self.pos_embed = MageFlowEmbedRope(
            theta=10000, axes_dim=self.axes_dim, scale_rope=True
        )

        # Input projections
        self.img_in = nn.Linear(self.in_channels, self.inner_dim)
        self.txt_norm = RMSNorm(params.context_in_dim, eps=1e-6)
        self.txt_in = nn.Linear(params.context_in_dim, self.inner_dim)

        # Timestep embedding
        self.time_text_embed = MageFlowTimestepProjEmbeddings(
            embedding_dim=self.inner_dim
        )

        # Transformer blocks
        self.transformer_blocks = [
            MageFlowTransformerBlock(
                dim=self.inner_dim,
                num_heads=self.num_attention_heads,
                head_dim=self.attention_head_dim,
            )
            for _ in range(params.depth)
        ]

        # Output
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim, self.patch_size * self.patch_size * self.out_channels
        )

    def __call__(
        self,
        img: mx.array,
        txt: mx.array,
        timesteps: mx.array,
        img_shapes: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
        text_attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            img: Image tokens [B, N_img, in_channels]
            txt: Text tokens [B, N_txt, context_in_dim]
            timesteps: [B] timestep values
            img_shapes: (frame, height, width) for RoPE

        Returns:
            Image velocity prediction [B, N_img, out_channels]
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        B, N_img, _ = img.shape
        N_txt = txt.shape[1]

        # Compute 2D RoPE frequencies
        if img_shapes is None:
            # Infer from image token count (single frame)
            lat_h = lat_w = int(N_img ** 0.5)
            img_shapes = (1, lat_h, lat_w)
        ms_pe = self.pos_embed(img_shapes)

        # Input projections
        img = self.img_in(img)
        txt = self.txt_norm(txt)

        # Timestep embedding
        timesteps = timesteps.astype(img.dtype)
        temb = self.time_text_embed(timesteps, img)

        # Text projection
        txt = self.txt_in(txt)

        # Build the mflux-compatible key mask: text mask followed by valid
        # image keys. Padded multimodal text tokens must not affect attention.
        attention_mask = None
        if text_attention_mask is not None:
            if text_attention_mask.shape != (B, N_txt):
                raise ValueError("text_attention_mask must match text conditioning shape")
            key_valid = mx.concatenate(
                [text_attention_mask.astype(mx.bool_), mx.ones((B, N_img), dtype=mx.bool_)], axis=-1
            )
            attention_mask = mx.where(
                key_valid[:, None, None, :],
                mx.array(0.0, dtype=img.dtype),
                mx.array(-1e9, dtype=img.dtype),
            )

        # Transformer blocks
        for block in self.transformer_blocks:
            img, txt = block(
                hidden_states=img,
                encoder_hidden_states=txt,
                temb=temb,
                image_rotary_emb=ms_pe,
            )

        # Output
        img = self.norm_out(img, temb)
        img = self.proj_out(img)
        return img

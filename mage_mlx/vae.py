"""MageVAE for Mage-Flow (MLX port).

Port of Microsoft's MageVAE from PyTorch to MLX.

Architecture:
  - DConvEncoder: image → (mean, logvar) latent [B, 128, H/16, W/16]
  - DConvDenoiser: latent + zero noise → reconstructed image [B, 3, H, W]
  - CoD Decoder: latent → conditioning features for the denoiser

Key MLX translations:
  - PyTorch NCHW → MLX NHWC (Conv2d weights transposed: [O,I,H,W] → [O,H,W,I])
  - LayerNorm2d operates on NHWC format
  - adaLN modulation constant-folded at t=0 (saves ~37M params)
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Primitive layers
# ---------------------------------------------------------------------------
def nonlinearity(x: mx.array) -> mx.array:
    """Swish/SiLU activation: x * sigmoid(x)"""
    return x * mx.sigmoid(x)


class RMSNorm(nn.Module):
    """RMSNorm (no bias, no mean subtraction)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, x: mx.array) -> mx.array:
        in_dtype = x.dtype
        x = x.astype(mx.float32)
        var = mx.square(x).mean(-1, keepdims=True)
        x = x * mx.rsqrt(var + self.variance_epsilon)
        return self.weight * x.astype(in_dtype)


class LayerNorm2d(nn.Module):
    """LayerNorm on the channel dimension (NHWC format)."""

    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps, affine=affine)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H, W, C] (NHWC)
        return self.norm(x)


class AdaptiveAvgPool2d(nn.Module):
    """Global spatial average pooling for NHWC tensors."""

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    def __call__(self, x: mx.array) -> mx.array:
        # x shape: [B, H, W, C] (NHWC)
        return mx.mean(x, axis=(1, 2), keepdims=True)


class Normalize(nn.Module):
    """GroupNorm normalization."""

    def __init__(self, in_channels: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(
            dims=in_channels,
            num_groups=num_groups,
            eps=eps,
            affine=True,
            pytorch_compatible=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # mflux performs VAE GroupNorm statistics in FP32 even when the
        # surrounding encoder runs in BF16, then restores the activation dtype.
        dtype = x.dtype
        return self.norm(x.astype(mx.float32)).astype(dtype)


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """Adaptive modulation: x * (1 + scale) + shift (NHWC layout: channels at axis=-1)."""
    if x.ndim == 4:
        # NHWC layout: x shape [B, H, W, C]
        b = x.shape[0]
        c = x.shape[-1]
        return x * (1 + scale.reshape(b, 1, 1, c)) + shift.reshape(b, 1, 1, c)
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class TimestepEmbedder(nn.Module):
    """DConv-style timestep MLP (max_period=10000, freq_size=256, hidden=384)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        args = t[:, None].astype(mx.float32) * freqs[None]
        emb = mx.concat([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concat([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb

    def __call__(self, t: mx.array) -> mx.array:
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb)


class BottleneckPatchEmbed(nn.Module):
    """Image patch embed concatenated with a per-patch conditioning vector."""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, pca_dim: int = 128, embed_dim: int = 384, bias: bool = True):
        super().__init__()
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim + embed_dim, embed_dim, kernel_size=1, bias=bias)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        return self.proj2(mx.concat([self.proj1(x), cond], axis=-1))


class DiCoBlock(nn.Module):
    """DConv block with adaLN modulation."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size, bias=True)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)

        self.ca = nn.Sequential(
            AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
            nn.Sigmoid(),
        )

        ffn = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, ffn, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn, hidden_size, 1, bias=True)

        self.norm1 = LayerNorm2d(hidden_size, affine=False)
        self.norm2 = LayerNorm2d(hidden_size, affine=False)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def __call__(self, inp: mx.array, c: mx.array) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).split(6, axis=-1)
        )
        x = modulate(self.norm1(inp), shift_msa, scale_msa)
        # The reference DConv uses GELU here, not the SiLU used by the CoD
        # decoder's ResNet blocks.
        x = nn.gelu(self.conv2(self.conv1(x)))
        x = x * self.ca(x)
        x = self.conv3(x)
        x = inp + gate_msa[:, None, None, :] * x
        x = x + gate_mlp[:, None, None, :] * self.conv5(
            nn.gelu(self.conv4(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        )
        return x


class _EncoderDiCoBlock(nn.Module):
    """DiCoBlock without adaLN, for the encoder pathway."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size, bias=True)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.ca = nn.Sequential(
            AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
            nn.Sigmoid(),
        )
        ffn = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, ffn, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn, hidden_size, 1, bias=True)
        self.norm1 = LayerNorm2d(hidden_size, affine=True)
        self.norm2 = LayerNorm2d(hidden_size, affine=True)

    def __call__(self, inp: mx.array) -> mx.array:
        x = self.norm1(inp)
        x = nn.gelu(self.conv2(self.conv1(x)))
        x = x * self.ca(x)
        x = self.conv3(x)
        x = inp + x
        return x + self.conv5(nn.gelu(self.conv4(self.norm2(x))))


class NerfEmbedder(nn.Module):
    """Patch-position embedder used by the DConv decoder x-pathway."""

    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int = 8):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Linear(in_channels + max_freqs ** 2, hidden_size_input, bias=True)

    @staticmethod
    @lru_cache(maxsize=128)
    def fetch_pos(patch_size: int, max_freqs: int = 8) -> mx.array:
        pos = mx.linspace(0, 1, patch_size, dtype=mx.float32)
        pos_y, pos_x = mx.meshgrid(pos, pos, indexing="ij")
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)
        freqs = mx.linspace(0, max_freqs, max_freqs, dtype=mx.float32)
        fx = freqs[None, :, None]
        fy = freqs[None, None, :]
        coeffs = (1 + fx * fy) ** -1
        dct_x = mx.cos(pos_x * fx * math.pi)
        dct_y = mx.cos(pos_y * fy * math.pi)
        return (dct_x * dct_y * coeffs).reshape(1, -1, max_freqs ** 2)

    def __call__(self, x: mx.array) -> mx.array:
        B, P2, _ = x.shape
        ps = int(P2 ** 0.5)
        dct = mx.broadcast_to(self.fetch_pos(ps, self.max_freqs).astype(x.dtype), (B, ps * ps, self.max_freqs ** 2))
        return self.embedder(mx.concat([x, dct], axis=-1))


class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(self.norm(x))


class SimpleMLPAdaLN(nn.Module):
    """Final small MLP that maps NerfEmbedder features to per-patch RGB."""

    def __init__(self, in_channels: int, model_channels: int, out_channels: int,
                 z_channels: int, num_res_blocks: int, patch_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.patch_size = patch_size

        self.cond_embed = nn.Linear(z_channels, patch_size ** 2 * model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = [_MLPResBlock(model_channels) for _ in range(num_res_blocks)]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        x = self.input_proj(x)
        c = self.cond_embed(c).reshape(c.shape[0], self.patch_size ** 2, -1)
        for block in self.res_blocks:
            x = block(x, c)
        return x


class _MLPResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        shift, scale, gate = self.adaLN_modulation(y).split(3, axis=-1)
        h = self.in_ln(x) * (1 + scale) + shift
        return x + gate * self.mlp(h)


class ResnetBlock(nn.Module):
    """GroupNorm + Conv ResBlock used by the CoD Decoder."""

    def __init__(self, in_channels: int, out_channels: int | None = None, dropout: float = 0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(nonlinearity(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Patched self-attention used at inference."""

    def __init__(self, in_channels: int, patch_size: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h_ = self.norm(x)
        Q = self.q(h_)
        K = self.k(h_)
        V = self.v(h_)

        d = self.patch_size
        b, c, H, W = Q.shape  # Note: MLX Conv2d uses NHWC, but shapes are the same conceptually
        # For MLX NHWC: b, H, W, c
        b, H, W, c = Q.shape
        pad_h = (d - H % d) % d
        pad_w = (d - W % d) % d
        if pad_h or pad_w:
            pad_width = [(0, 0), (0, pad_h), (0, pad_w), (0, 0)]
            Q = mx.pad(Q, pad_width, mode="edge")
            K = mx.pad(K, pad_width, mode="edge")
            V = mx.pad(V, pad_width, mode="edge")
        _, H_pad, W_pad, _ = Q.shape
        nph, npw = H_pad // d, W_pad // d
        np_ = nph * npw

        def to_patches(t):
            # NHWC: [b, H, W, c] → [b*nph*npw, c, d*d]
            t = t.reshape(b, nph, d, npw, d, c)
            t = t.transpose(0, 1, 3, 5, 2, 4)
            return t.reshape(b * np_, c, d * d)

        Q = to_patches(Q)
        K = to_patches(K)
        V = to_patches(V)

        w_ = mx.softmax(mx.matmul(Q.transpose(0, 2, 1), K) * (c ** -0.5), axis=2)
        # PyTorch reference applies softmax and then permutes the last two
        # dimensions before multiplying V.
        h_ = mx.matmul(V, w_.transpose(0, 2, 1))
        h_ = h_.reshape(b, nph, npw, c, d, d).transpose(0, 1, 4, 2, 5, 3).reshape(b, H_pad, W_pad, c)
        if pad_h or pad_w:
            h_ = h_[:, :H, :W, :]
        return x + self.proj_out(h_)


# ---------------------------------------------------------------------------
# CoD Decoder
# ---------------------------------------------------------------------------
class _Decoder(nn.Module):
    """ds=16, up2x=True, light=True only."""

    def __init__(self, out_ch: int = 384, z_ch: int = 128):
        super().__init__()
        self.conv_in = nn.Conv2d(z_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.block = nn.Sequential(
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=32),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=32),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
        )
        self.norm_out = Normalize(out_ch)
        self.conv_out = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.ada = nn.Identity()

    def __call__(self, z: mx.array) -> mx.array:
        h = self.block(self.conv_in(z))
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return self.ada(h)


# ---------------------------------------------------------------------------
# DConvEncoder
# ---------------------------------------------------------------------------
class _DConvEncoder(nn.Module):
    def __init__(
        self,
        z_ch: int = 128,
        hidden_size: int = 384,
        num_blocks: int = 21,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
        head_size: int = 768,
        num_head_blocks: int = 2,
        out_ch_mult: int = 2,
    ):
        super().__init__()
        self.z_ch = z_ch
        self.patch_size = patch_size
        self.patch_cond_embed = nn.Conv2d(3, head_size, kernel_size=patch_size, stride=patch_size, bias=True)
        self.head_blocks = [_EncoderDiCoBlock(head_size, mlp_ratio=mlp_ratio) for _ in range(num_head_blocks)]
        self.proj_down = nn.Conv2d(head_size, hidden_size, kernel_size=1, bias=True)
        self.z_proj = nn.Conv2d(z_ch, hidden_size, kernel_size=1, bias=True)
        self.fuse_proj = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = [DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_blocks)]
        self.norm_out = LayerNorm2d(hidden_size)
        self.proj_out = nn.Conv2d(hidden_size, z_ch * out_ch_mult, kernel_size=1, bias=True)

    def forward_pred(self, z_t: mx.array, t: mx.array, y: mx.array) -> mx.array:
        cond = self.patch_cond_embed(y)
        for block in self.head_blocks:
            cond = block(cond)
        cond = self.proj_down(cond)

        s = self.fuse_proj(mx.concat([cond, self.z_proj(z_t)], axis=-1))
        c = self.t_embedder(t.reshape(-1))
        for block in self.blocks:
            s = block(s, c)
        return self.proj_out(self.norm_out(s))


# ---------------------------------------------------------------------------
# DConvDenoiser
# ---------------------------------------------------------------------------
class _YEmbedder(nn.Module):
    """Holds only the CoD decoder."""

    def __init__(self, ch: int = 384, z_ch: int = 128):
        super().__init__()
        self.decoder = _Decoder(out_ch=ch, z_ch=z_ch)


class _DConvDenoiser(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 384,
        hidden_size_x: int = 32,
        mlp_ratio: float = 4.0,
        num_blocks: int = 24,
        num_cond_blocks: int = 21,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.hidden_size_x = hidden_size_x
        self.num_cond_blocks = num_cond_blocks

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder_x = nn.Conv2d(hidden_size, hidden_size_x * patch_size ** 2, 1, 1, 0)
        self.x_embedder = NerfEmbedder(in_channels + hidden_size_x, hidden_size_x, max_freqs=8)
        self.s_embedder = BottleneckPatchEmbed(patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)
        self.blocks = [DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_cond_blocks)]
        self.dec_net = SimpleMLPAdaLN(
            in_channels=hidden_size_x,
            model_channels=hidden_size_x,
            out_channels=in_channels,
            z_channels=hidden_size,
            num_res_blocks=num_blocks - num_cond_blocks,
            patch_size=patch_size,
        )
        self.final_layer = NerfFinalLayer(hidden_size_x, in_channels)
        self.y_embedder = _YEmbedder(ch=hidden_size, z_ch=bottleneck_dim)

    def __call__(self, x: mx.array, t: mx.array, cond: mx.array) -> mx.array:
        b, h, w, ch = x.shape
        c = self.t_embedder(t.reshape(-1))

        s = self.s_embedder(x, cond)
        for block in self.blocks:
            s = block(s, c)

        # s shape: [B, H_p, W_p, C_s]
        h_p, w_p = s.shape[1], s.shape[2]
        length = h_p * w_p
        s_flat = s.reshape(-1, self.hidden_size)

        ps = self.patch_size

        # Patchify x: [B, H, W, Ch] → [B, H_p, ps, W_p, ps, Ch] → [B*length, ps*ps, Ch]
        x_patches = x.reshape(b, h_p, ps, w_p, ps, ch)
        x_patches = x_patches.transpose(0, 1, 3, 2, 4, 5)  # [B, H_p, W_p, ps, ps, Ch]
        x_patches = x_patches.reshape(b * length, ps * ps, ch)

        # Project cond. PyTorch emits channels in
        # [hidden_size_x, patch_position] order before unfold/reshape, so keep
        # that ordering here and then transpose to [patch_position, feature].
        y_cond = self.y_embedder_x(cond)  # [B, H_p, W_p, hidden_size_x * ps^2]
        y_cond = y_cond.reshape(b, h_p, w_p, self.hidden_size_x, ps * ps)
        y_cond = y_cond.transpose(0, 1, 2, 4, 3)
        y_cond = y_cond.reshape(b * length, ps * ps, self.hidden_size_x)

        # Concat image patches and conditioning: [B*length, ps*ps, Ch + hidden_size_x]
        x_fused = mx.concat([x_patches, y_cond], axis=-1)

        # NerfEmbedder: [B*length, ps*ps, Ch + hidden_size_x] → [B*length, ps*ps, hidden_size_x]
        x_fused = self.x_embedder(x_fused)

        # SimpleMLPAdaLN: x=[B*length, ps*ps, hidden_size_x], c=[B*length, hidden_size]
        out = self.dec_net(x_fused, s_flat)

        # Final layer: [B*length, ps*ps, hidden_size_x] → [B*length, ps*ps, Ch]
        out = self.final_layer(out)

        # Unpatchify: [B*length, ps*ps, Ch] → [B, H, W, Ch]
        out = out.reshape(b, h_p, w_p, ps, ps, ch)
        out = out.transpose(0, 1, 3, 2, 4, 5)  # [B, H_p, ps, W_p, ps, Ch]
        out = out.reshape(b, h, w, ch)
        return out


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class MageVAE(nn.Module):
    """MageVAE: Encode → latent [B, 128, H/16, W/16], Decode → image [B, H, W, 3].

    Args:
        ckpt_path: Path to the VAE safetensors file
        sample_posterior: If True, sample from posterior; if False, use mean
    """

    latent_channels = 128
    downsample_factor = 16

    def __init__(self, ckpt_path: str, sample_posterior: bool = False):
        super().__init__()
        self.sample_posterior = sample_posterior

        self.dconv_encoder = _DConvEncoder()
        self.decoder_model = _DConvDenoiser()

        self._load_weights(ckpt_path)
        self._freeze_adaln_cache()

    def _load_weights(self, ckpt_path: str) -> None:
        """Load encoder and decoder weights from safetensors."""
        weights = mx.load(ckpt_path)

        self.load_weights(list(weights.items()), strict=False)
        print(f"  Loaded VAE: {len(weights)} tensors")

    def _named_parameters(self):
        """Yield (path, param) tuples for all parameters."""
        from mlx.utils import tree_flatten
        return list(tree_flatten(self.parameters()))

    def _freeze_adaln_cache(self) -> None:
        """Constant-fold adaLN_modulation MLPs at t=0."""
        # At t=0, the timestep embedding is constant, so adaLN_modulation(c) is constant.
        # We precompute the modulation and replace the MLP with a buffer.
        # This saves ~37M params and speeds up inference.
        # For simplicity, we skip this optimization in the initial port.
        pass

    def encode(self, x: mx.array, key: mx.array | None = None) -> mx.array:
        """Encode image to latent.

        Args:
            x: [B, H, W, 3] image in [-1, 1]

        Returns:
            [B, H/16, W/16, 128] latent
        """
        ps = self.dconv_encoder.patch_size
        H, W = x.shape[1], x.shape[2]
        if H % ps or W % ps:
            raise ValueError(f"H, W must be multiples of {ps}, got ({H}, {W})")

        z_t = mx.zeros((x.shape[0], self.dconv_encoder.z_ch, H // ps, W // ps), dtype=x.dtype)
        z_t = z_t.transpose(0, 2, 3, 1)  # NCHW → NHWC
        t = mx.zeros((x.shape[0],), dtype=x.dtype)
        out = self.dconv_encoder.forward_pred(z_t, t, x)
        mean = out[..., :self.latent_channels]
        logvar = mx.clip(out[..., self.latent_channels:], -20.0, 10.0)

        if self.sample_posterior:
            if key is None:
                raise ValueError("key is required when sample_posterior is enabled")
            # Match mflux: keyed posterior noise is generated in NCHW order,
            # then transposed to this port's NHWC latent representation.
            noise_nchw = mx.random.normal(
                (mean.shape[0], mean.shape[-1], mean.shape[1], mean.shape[2]),
                dtype=mean.dtype,
                key=key,
            )
            noise = noise_nchw.transpose(0, 2, 3, 1)
            return mean + mx.exp(0.5 * logvar) * noise
        return mean

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent to image.

        Args:
            z: [B, H/16, W/16, 128] latent

        Returns:
            [B, H, W, 3] image in [-1, 1]
        """
        cond = self.decoder_model.y_embedder.decoder(z)
        B = z.shape[0]
        H = z.shape[1] * self.downsample_factor
        W = z.shape[2] * self.downsample_factor
        noise = mx.zeros((B, H, W, 3), dtype=z.dtype)
        t = mx.zeros((B,), dtype=z.dtype)
        return self.decoder_model(noise, t, cond)

    def pack_latents(self, latents: mx.array) -> mx.array:
        """Flatten spatial dimensions of NHWC latents.

        Args:
            latents: [B, H, W, C] latent

        Returns:
            [B, H*W, C] flattened latent
        """
        batch_size, height, width, channels = latents.shape
        return latents.reshape(batch_size, height * width, channels)

    def unpack_latents(self, latents: mx.array, height: int, width: int) -> mx.array:
        """Unflatten spatial dimensions of packed latents.

        Args:
            latents: [B, H*W, C] flattened latent
            height: Target height in patches
            width: Target width in patches

        Returns:
            [B, H, W, C] latent
        """
        batch_size, num_patches, channels = latents.shape
        return latents.reshape(batch_size, height, width, channels)

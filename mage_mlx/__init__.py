"""Mage-Flow MLX: Native Apple Silicon port of Microsoft's Mage-Flow (4B MMDiT).

Components:
    rope          — 2D multi-scale RoPE (MageFlowEmbedRope)
    timestep      — Sinusoidal timestep embedding (qwen_proj style)
    dit           — MageFlow DiT (12 double-stream MMDiT blocks)
    text_encoder  — Qwen3-VL text encoder (MLX)
    vae           — MageVAE (DConvEncoder + DConvDenoiser + CoD Decoder)
    scheduler     — FlowMatchEulerDiscreteScheduler
    pipeline      — MageFlowPipeline (end-to-end text-to-image)
"""

from .pipeline import MageFlowPipeline

__all__ = ["MageFlowPipeline"]
__version__ = "0.1.0"

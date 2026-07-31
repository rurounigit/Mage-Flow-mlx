"""Mage-Flow MLX: Native Apple Silicon port of Microsoft's Mage-Flow (4B MMDiT).

Components:
    rope          — 2D multi-scale RoPE (MageFlowEmbedRope)
    timestep      — Sinusoidal timestep embedding (qwen_proj style)
    dit           — MageFlow DiT (12 double-stream MMDiT blocks)
    text_encoder  — Qwen3-VL text encoder (MLX)
    vision_model  — Qwen3-VL vision tower (for edit)
    prompt_processor — Shared prompt templates (txt2img + edit)
    processor     — Qwen3-VL image processor + tokenizer wrapper
    vae           — MageVAE (DConvEncoder + DConvDenoiser + CoD Decoder)
    scheduler     — FlowMatchEulerDiscreteScheduler
    pipeline      — MageFlowPipeline (end-to-end text-to-image)
    profiler      — Phase-level timing and memory profiler
    embedding_cache — Prompt embedding cache (skip Qwen load on cache hit)
    worker        — Persistent JSONL worker (models stay resident)
    thermal       — macOS thermal state monitoring (notify framework + sysctl fallback)
"""

from .pipeline import MageFlowPipeline, MageFlowTokenizer
from .prompt_processor import MageFlowPromptProcessor
from .processor import MageFlowQwen3VLProcessor
from .text_encoder import MageFlowTextEncoder
from .vision_model import MageFlowQwen3VLVisionModel
from .rope import MageFlowEmbedRope
from .profiler import Profiler
from .embedding_cache import EmbeddingCache
from .thermal import get_thermal_state

__all__ = [
    "MageFlowPipeline",
    "MageFlowTokenizer",
    "MageFlowPromptProcessor",
    "MageFlowQwen3VLProcessor",
    "MageFlowTextEncoder",
    "MageFlowQwen3VLVisionModel",
    "MageFlowEmbedRope",
    "Profiler",
    "EmbeddingCache",
    "get_thermal_state",
]
__version__ = "0.2.0"

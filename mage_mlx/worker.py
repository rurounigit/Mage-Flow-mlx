"""Persistent JSONL worker for Mage-Flow MLX.

Keeps DiT, VAE, and tokenizer resident across multiple generations,
accepting prompts from a JSONL file. CLI flags set defaults; per-prompt
JSON fields override them.

Uses **prompt queue mode**: all prompts are text-encoded in a single
Qwen session (load once, encode all, unload once), then all images are
generated using cached embeddings. This amortizes the ~8 GB Qwen load
across the entire batch instead of paying it per-prompt.

Usage:
    .venv/bin/python generate.py --worker prompts.jsonl --metadata

JSONL format (one JSON object per line):
    {"prompt": "A cat", "seed": 42, "output": "cat.png"}
    {"prompt": "A dog", "seed": 43, "output": "dog.png", "steps": 8}
    {"prompt": "A bird", "seed": 44, "output": "bird.png", "width": 512, "height": 512}

Parameters requiring pipeline reload (model, quantize) trigger a reload.
Parameters requiring scheduler update (steps) trigger a scheduler reset.
All other parameters are passed directly to generate().
"""

from __future__ import annotations

import gc
import json
import os
from datetime import datetime
from typing import Any, Optional

import mlx.core as mx

from .profiler import Profiler


# Parameters that require a full pipeline reload
RELOAD_PARAMS = {"model", "quantize"}

# Parameters that require a scheduler reset (but no model reload)
SCHEDULER_PARAMS = {"steps"}

# All valid per-prompt parameters
VALID_PARAMS = {
    "prompt", "negative_prompt", "seed", "guidance",
    "width", "height", "output", "steps", "model", "quantize",
}


def _get_model_name(model_dir: str) -> str:
    """Convert model directory path to HuggingFace model name.

    e.g. 'models/microsoft_Mage-Flow-Turbo' -> 'microsoft/Mage-Flow-Turbo'
    e.g. 'microsoft/Mage-Flow-Turbo' -> 'microsoft/Mage-Flow-Turbo'
    """
    basename = os.path.basename(model_dir.rstrip("/"))
    return basename.replace("_", "/")


def _get_base_model(model_dir: str) -> str:
    """Read base_model from transformer_config.json, fallback to model_dir."""
    config_path = os.path.join(model_dir, "transformer_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            return config.get("_class_name", model_dir)
        except (json.JSONDecodeError, OSError):
            pass
    return model_dir


def load_prompts(jsonl_path: str) -> list[dict[str, Any]]:
    """Load prompts from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        List of prompt dictionaries
    """
    prompts = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                prompt = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping line {line_num}: invalid JSON: {e}")
                continue

            # Validate required fields
            if "prompt" not in prompt:
                print(f"  WARNING: Skipping line {line_num}: missing 'prompt' field")
                continue
            if "output" not in prompt:
                prompt["output"] = f"output_{line_num}.png"

            # Validate parameter names
            invalid = set(prompt.keys()) - VALID_PARAMS
            if invalid:
                print(f"  WARNING: Skipping line {line_num}: unknown parameters: {invalid}")
                continue

            prompts.append(prompt)

    return prompts


def merge_params(
    defaults: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Merge per-prompt parameters with CLI defaults.

    Per-prompt values take precedence over CLI defaults.

    Args:
        defaults: CLI default parameters
        override: Per-prompt override parameters

    Returns:
        Merged parameter dictionary
    """
    merged = dict(defaults)
    merged.update(override)
    return merged


def needs_reload(
    current: dict[str, Any],
    new: dict[str, Any],
) -> tuple[bool, bool]:
    """Check if a pipeline reload or scheduler reset is needed.

    Args:
        current: Current pipeline parameters
        new: New prompt parameters

    Returns:
        (needs_pipeline_reload, needs_scheduler_reset)
    """
    needs_pipeline = False
    needs_scheduler = False

    for param in RELOAD_PARAMS:
        if param in new and new[param] != current.get(param):
            needs_pipeline = True

    for param in SCHEDULER_PARAMS:
        if param in new and new[param] != current.get(param):
            needs_scheduler = True

    return needs_pipeline, needs_scheduler


def run_worker(
    jsonl_path: str,
    defaults: dict[str, Any],
    profiler: Optional[Profiler] = None,
    metadata_enabled: bool = False,
    report=None,
) -> tuple[Optional[dict], Optional[list]]:
    """Run the persistent JSONL worker in prompt queue mode.

    Phase 1: Load pipeline (DiT + VAE resident). Load Qwen once, encode all
    prompts (checking embedding cache first), save to cache, unload Qwen.
    Phase 2: Generate all images using cached embeddings.

    Args:
        jsonl_path: Path to the JSONL prompts file
        defaults: Default parameters from CLI
        profiler: Optional Profiler instance
        metadata_enabled: If True, save JSON + markdown metadata files
        report: Optional LiveReport instance for real-time terminal output

    Returns:
        (metadata dict, prompt_metadata list) if metadata_enabled, else (None, None).
    """
    from mage_mlx import MageFlowPipeline
    from mage_mlx.embedding_cache import EmbeddingCache

    # Load prompts
    prompts = load_prompts(jsonl_path)
    if not prompts:
        print("No valid prompts found in JSONL file.")
        return None

    print(f"  Loaded {len(prompts)} prompts from {jsonl_path}")

    # Set up real-time callback for LiveReport
    # The callback handles sub-phases (dit_step_N, vae_decode, etc.) in real-time.
    # Main phases are reported explicitly via stop_phase to add saved_file/metadata.
    _EXPLICIT_PREFIXES = ("pipeline_reload", "text_encoder_unload", "text_encode_", "generation_", "save_", "total_wall_clock", "python_startup")
    if report and profiler:
        def _on_phase_complete(name, elapsed, rss):
            if any(name.startswith(p) for p in _EXPLICIT_PREFIXES):
                return  # handled explicitly via stop_phase
            report.stop_phase(name, elapsed or 0.0, rss)
        profiler.on_phase_complete = _on_phase_complete

    # Initialize cache
    cache = EmbeddingCache(defaults.get("model", "models/microsoft_Mage-Flow-Turbo"))

    # Track current pipeline state
    current_params: dict[str, Any] = {}
    pipeline: Optional[MageFlowPipeline] = None
    te_path: Optional[str] = None

    # --- Phase 0: Load pipeline (once) ---
    first_params = merge_params(defaults, prompts[0])
    if profiler:
        profiler.start("pipeline_reload")
    pipeline = MageFlowPipeline.from_pretrained(
        model_dir=first_params["model"],
        num_steps=first_params.get("steps", 4),
        quantize=first_params.get("quantize"),
        profiler=profiler,
    )
    te_path = os.path.join(first_params["model"], "text_encoder.safetensors")
    if not os.path.exists(te_path):
        te_path = None
    current_params = dict(first_params)
    if profiler:
        profiler.stop("pipeline_reload")
    if report:
        report.stop_phase("pipeline_reload", profiler.get_elapsed("pipeline_reload") or 0.0, profiler._get_rss_gib())

    # --- Phase 1: Pre-encode all prompts (load Qwen once, encode all, unload) ---
    print(f"\n{'=' * 60}")
    print("Phase 1: Pre-encoding prompts (Qwen batch mode)")
    print(f"{'=' * 60}")

    # Store embeddings in memory for Phase 2
    prompt_embeds: dict[int, tuple[mx.array, Optional[mx.array]]] = {}

    for i, prompt_cfg in enumerate(prompts):
        params = merge_params(defaults, prompt_cfg)
        cache_key = cache.make_key(
            prompt=params["prompt"],
            negative_prompt=params.get("negative_prompt", " "),
            te_path=te_path,
        )
        cached_embeds = cache.get(cache_key)

        if cached_embeds is not None:
            print(f"  Prompt {i + 1}/{len(prompts)}: Cache HIT — skipping Qwen encode")
            # Still need to encode negative prompt if guidance > 1.0
            neg_embeds = None
            if params.get("guidance", 1.0) > 1.0:
                if profiler:
                    profiler.start(f"text_encode_neg_{i + 1}")
                neg_embeds, _ = pipeline.text_encoder.encode_text_to_image(
                    prompts=[params.get("negative_prompt", " ")],
                    tokenizer=pipeline.tokenizer,
                    max_sequence_length=2048,
                )
                mx.eval(neg_embeds)
                if profiler:
                    profiler.stop(f"text_encode_neg_{i + 1}")
                    profiler.set_metadata(f"text_encode_neg_{i + 1}", "cache", "HIT")
                    if report:
                        report.stop_phase(f"text_encode_neg_{i + 1}", profiler.get_elapsed(f"text_encode_neg_{i + 1}") or 0.0, profiler._get_rss_gib())
                        report.add_metadata(f"text_encode_neg_{i + 1}", "cache", "HIT")
            prompt_embeds[i] = (cached_embeds, neg_embeds)
        else:
            print(f"  Prompt {i + 1}/{len(prompts)}: Cache MISS — encoding with Qwen")
            if profiler:
                profiler.start(f"text_encode_{i + 1}")

            # Encode positive prompt (Qwen stays loaded)
            pos_embeds, _ = pipeline.text_encoder.encode_text_to_image(
                prompts=[params["prompt"]],
                tokenizer=pipeline.tokenizer,
                max_sequence_length=2048,
            )
            mx.eval(pos_embeds)

            # Save to cache for future runs
            cache.put(cache_key, pos_embeds)

            # Encode negative prompt if guidance > 1.0
            neg_embeds = None
            if params.get("guidance", 1.0) > 1.0:
                neg_embeds, _ = pipeline.text_encoder.encode_text_to_image(
                    prompts=[params.get("negative_prompt", " ")],
                    tokenizer=pipeline.tokenizer,
                    max_sequence_length=2048,
                )
                mx.eval(neg_embeds)

            prompt_embeds[i] = (pos_embeds, neg_embeds)

            if profiler:
                profiler.stop(f"text_encode_{i + 1}")
                profiler.set_metadata(f"text_encode_{i + 1}", "cache", "MISS")
                if report:
                    report.stop_phase(f"text_encode_{i + 1}", profiler.get_elapsed(f"text_encode_{i + 1}") or 0.0, profiler._get_rss_gib())
                    report.add_metadata(f"text_encode_{i + 1}", "cache", "MISS")

    # Unload Qwen once after all prompts are encoded
    if profiler:
        profiler.start("text_encoder_unload")
    pipeline.text_encoder.unload()
    gc.collect()
    mx.clear_cache()
    if profiler:
        profiler.stop("text_encoder_unload")
    if report:
        report.stop_phase("text_encoder_unload", profiler.get_elapsed("text_encoder_unload") or 0.0, profiler._get_rss_gib())
    print("  Qwen unloaded (batch encoding complete)")

    # --- Phase 2: Generate all images using cached embeddings ---
    print(f"\n{'=' * 60}")
    print("Phase 2: Generating images (DiT + VAE)")
    print(f"{'=' * 60}")

    # Collect per-prompt metadata for JSON output
    prompt_metadata: list[dict[str, Any]] = []

    for i, prompt_cfg in enumerate(prompts):
        params = merge_params(defaults, prompt_cfg)

        # Print prompt header via LiveReport
        if report:
            report.prompt_header(i + 1, len(prompts))

        # Check if pipeline reload or scheduler reset is needed
        needs_pipeline, needs_scheduler = needs_reload(current_params, params)

        # Handle pipeline reload (model/quantize change)
        if needs_pipeline:
            if profiler:
                profiler.start(f"pipeline_reload_{i + 1}")
            pipeline = MageFlowPipeline.from_pretrained(
                model_dir=params["model"],
                num_steps=params.get("steps", 4),
                quantize=params.get("quantize"),
                profiler=profiler,
            )
            # Unload text encoder from new pipeline (we use cached embeddings)
            if profiler:
                profiler.start(f"text_encoder_unload_{i + 1}")
            pipeline.text_encoder.unload()
            gc.collect()
            mx.clear_cache()
            if profiler:
                profiler.stop(f"text_encoder_unload_{i + 1}")
            current_params = dict(params)
            if profiler:
                profiler.stop(f"pipeline_reload_{i + 1}")
            if report:
                report.stop_phase(f"pipeline_reload_{i + 1}", profiler.get_elapsed(f"pipeline_reload_{i + 1}") or 0.0, profiler._get_rss_gib())
                report.stop_phase(f"text_encoder_unload_{i + 1}", profiler.get_elapsed(f"text_encoder_unload_{i + 1}") or 0.0, profiler._get_rss_gib())

        # Handle scheduler reset (steps change)
        if needs_scheduler:
            pipeline.scheduler.set_timesteps(params["steps"])
            pipeline.num_steps = params["steps"]
            current_params = dict(params)

        # Set per-prompt metadata BEFORE generation starts (so it appears right after prompt header)
        if report:
            report.add_metadata(f"generation_{i + 1}", "prompt", params["prompt"])
            report.add_metadata(f"generation_{i + 1}", "resolution", f"{params['width']}x{params['height']}")
            report.add_metadata(f"generation_{i + 1}", "steps", str(params["steps"]))
            report.add_metadata(f"generation_{i + 1}", "quantize", str(params.get("quantize")))
            print()  # Empty line after metadata, before generation steps

        if profiler:
            profiler.start(f"generation_{i + 1}")

        pos_embeds, neg_embeds = prompt_embeds[i]
        image = _generate_with_cached_embeds(
            pipeline,
            pos_embeds,
            neg_embeds,
            params,
            profiler,
        )

        if profiler:
            profiler.stop(f"generation_{i + 1}")
            # Also set metadata on profiler for JSON/markdown output
            profiler.set_metadata(f"generation_{i + 1}", "prompt", params["prompt"])
            profiler.set_metadata(f"generation_{i + 1}", "resolution", f"{params['width']}x{params['height']}")
            profiler.set_metadata(f"generation_{i + 1}", "steps", str(params["steps"]))
            profiler.set_metadata(f"generation_{i + 1}", "quantize", str(params.get("quantize")))
        if report:
            report.stop_phase(f"generation_{i + 1}", profiler.get_elapsed(f"generation_{i + 1}") or 0.0, profiler._get_rss_gib())

        # Save image
        if profiler:
            profiler.start(f"save_{i + 1}")
        image.save(params["output"])
        if profiler:
            profiler.stop(f"save_{i + 1}")
        if report:
            report.stop_phase(f"save_{i + 1}", profiler.get_elapsed(f"save_{i + 1}") or 0.0, profiler._get_rss_gib(), saved_file=params["output"])

        # Collect per-prompt metadata (with peak RAM)
        gen_elapsed = profiler.get_elapsed(f"generation_{i + 1}") if profiler else None
        gen_peak_rss = None
        if profiler:
            for rec in profiler.get_records():
                if rec.name == f"generation_{i + 1}":
                    gen_peak_rss = rec.peak_rss_gib
                    break
        prompt_metadata.append({
            "index": i + 1,
            "prompt": params["prompt"],
            "negative_prompt": params.get("negative_prompt"),
            "quantize": params.get("quantize"),
            "resolution": f"{params['width']}x{params['height']}",
            "steps": params["steps"],
            "generation_time_seconds": gen_elapsed,
            "peak_rss_gib": gen_peak_rss,
            "image_path": params["output"],
        })

        # Add to LiveReport for per-prompt summary table
        if report:
            report.add_prompt(
                index=i + 1,
                prompt=params["prompt"],
                resolution=f"{params['width']}x{params['height']}",
                steps=params["steps"],
                quantize=params.get("quantize"),
                generation_time=gen_elapsed,
                peak_rss_gib=gen_peak_rss,
                saved_file=params["output"],
            )

    # --- Stop total_wall_clock before saving metadata ---
    if profiler:
        profiler.stop("total_wall_clock")

    # --- Save metadata files ---
    if metadata_enabled and profiler:
        metadata = {
            "model": _get_model_name(defaults["model"]),
            "base_model": _get_base_model(defaults["model"]),
            "generation_time_seconds": profiler.get_elapsed("total_wall_clock"),
            "created_at": datetime.now().isoformat(),
            "image_path": None,
            "image_paths": None,
            "image_strength": None,
            "peak_memory_gib": profiler._get_rss_gib(),
        }
        base_path = os.path.splitext(jsonl_path)[0]
        profiler.save_metadata(base_path, metadata, prompts=prompt_metadata)
        print(f"\n  Metadata saved to {base_path}.json and {base_path}.md")
        return metadata, prompt_metadata

    return None, None


def _generate_with_cached_embeds(
    pipeline: "MageFlowPipeline",
    pos_embeds: "mx.array",
    neg_embeds: Optional["mx.array"],
    params: dict[str, Any],
    profiler: Optional[Profiler] = None,
):
    """Generate an image using cached text embeddings.

    Calls ``pipeline._generate_from_embeds()`` which bypasses text encoding
    and Qwen unloading entirely. Only DiT steps and VAE decode are run.

    Args:
        pipeline: MageFlowPipeline instance (Qwen already unloaded)
        pos_embeds: Cached positive prompt embeddings [1, seq_len, 2560]
        neg_embeds: Cached negative prompt embeddings, or None if no CFG
        params: Generation parameters
        profiler: Optional Profiler instance

    Returns:
        Generated PIL Image
    """
    return pipeline._generate_from_embeds(
        txt_embeds=pos_embeds,
        neg_txt_embeds=neg_embeds,
        height=params["height"],
        width=params["width"],
        seed=params["seed"],
        guidance_scale=params["guidance"],
        profiler=profiler,
    )

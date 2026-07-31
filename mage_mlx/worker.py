"""Persistent JSONL worker for Mage-Flow MLX.

Keeps DiT, VAE, and tokenizer resident across multiple generations,
accepting prompts from a JSONL file. CLI flags set defaults; per-prompt
JSON fields override them.

Uses **prompt queue mode**: all prompts are text-encoded in a single
Qwen session (load once, encode all, unload once), then all images are
generated using cached embeddings. This amortizes the ~8 GB Qwen load
across the entire batch instead of paying it per-prompt.

**Memory optimization**: Qwen (~7.5 GiB) is loaded and unloaded BEFORE
DiT + VAE (~7.9 GiB) are loaded, so peak RAM is max(Qwen, DiT+VAE)
instead of Qwen + DiT + VAE simultaneously (~15.4 GiB).

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
from .output_resolver import resolve_output_path, resolve_metadata_path
from mage_mlx.mflux_src.mflux.models.mage_flow.variants.conditioning import MageFlowConditioning


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
                prompt["output"] = None

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
        (needs_pipeline_load, needs_scheduler_reset)
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
    output: Optional[str] = None,
) -> tuple[Optional[dict], Optional[list]]:
    """Run the persistent JSONL worker in prompt queue mode.

    Phase 1: Load text encoder only. Encode all prompts (checking embedding
    cache first), save to cache, unload Qwen.
    Phase 1.5: Load DiT + VAE (after Qwen is unloaded — reduces peak RAM).
    Phase 2: Generate all images using cached embeddings.

    The key optimization: Qwen (~7.5 GiB) is loaded and unloaded BEFORE
    DiT + VAE (~7.9 GiB) are loaded, so peak RAM is max(Qwen, DiT+VAE)
    instead of Qwen + DiT + VAE simultaneously (~15.4 GiB).

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
        return None, None

    if report and report.verbose:
        print(f"  Loaded {len(prompts)} prompts from {jsonl_path}")
        if profiler:
            profiler.log(f"  Loaded {len(prompts)} prompts from {jsonl_path}")

    # Set up incremental save path on profiler (so files are written
    # after every phase completes, not just at the end)
    if metadata_enabled and profiler:
        base_path = resolve_metadata_path(output, jsonl_path)
        profiler.metadata_path = base_path
        profiler.metadata = {
            "model": _get_model_name(defaults["model"]),
            "base_model": _get_base_model(defaults["model"]),
            "generation_time_seconds": None,
            "created_at": datetime.now().isoformat(),
            "image_path": None,
            "image_paths": None,
            "image_strength": None,
            "peak_memory_gib": None,
        }

    # Set up real-time callback for LiveReport

    # The callback handles sub-phases (dit_step_N, vae_decode, etc.) in real-time.
    # Main phases are reported explicitly via stop_phase to add saved_file/metadata.
    _EXPLICIT_EXACT = {"dit_load", "vae_load"}
    _EXPLICIT_PREFIXES = ("pipeline_load", "text_encoder_unload", "text_encode_", "generation_", "save_", "total_wall_clock", "python_startup")
    if report and profiler:
        def _on_phase_complete(name, elapsed, rss):
            if name in _EXPLICIT_EXACT or any(name.startswith(p) for p in _EXPLICIT_PREFIXES):
                return  # handled explicitly via stop_phase
            # text_encoder_load is lazy — weights load during text_encode,
            # so the phase time is always ~0.0s. Show "lazy loading" instead.
            loading_mode = "lazy loading" if name == "text_encoder_load" else None
            report.stop_phase(name, elapsed or 0.0, rss, loading_mode=loading_mode)
        profiler.on_phase_complete = _on_phase_complete

    # Initialize cache
    cache = EmbeddingCache(defaults.get("model", "models/microsoft_Mage-Flow-Turbo"))

    # Track current pipeline state
    current_params: dict[str, Any] = {}
    pipeline: Optional[MageFlowPipeline] = None
    te_path: Optional[str] = None

    # --- Phase 0: Load text encoder + tokenizer (DiT + VAE deferred) ---
    # We load only the text encoder first, encode all prompts, unload it,
    # THEN load DiT + VAE. This reduces peak RAM from ~15.4 GiB to ~7.9 GiB
    # on cache miss, because Qwen (~7.5 GiB) is never resident alongside
    # DiT + VAE (~7.9 GiB) simultaneously.
    first_params = merge_params(defaults, prompts[0])
    if profiler:
        profiler.start("pipeline_load")
    pipeline = MageFlowPipeline.from_pretrained_text_encoder(
        model_dir=first_params["model"],
        num_steps=first_params.get("steps", 4),
        profiler=profiler,
    )
    te_path = os.path.join(first_params["model"], "text_encoder.safetensors")
    if not os.path.exists(te_path):
        te_path = None
    current_params = dict(first_params)
    if profiler:
        profiler.stop("pipeline_load")
    if report:
        report.stop_phase("pipeline_load", profiler.get_elapsed("pipeline_load") or 0.0, profiler.get_phase_rss("pipeline_load"), grey_separator=True)

    # --- Phase 1: Pre-encode all prompts (load Qwen once, encode all, unload) ---
    if report and report.verbose:
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
                        report.stop_phase(f"text_encode_neg_{i + 1}", profiler.get_elapsed(f"text_encode_neg_{i + 1}") or 0.0, profiler.get_phase_rss(f"text_encode_neg_{i + 1}"))
                        report.add_metadata(f"text_encode_neg_{i + 1}", "cache", "HIT")
            prompt_embeds[i] = (cached_embeds, neg_embeds)
            if report and report.verbose:
                print(f"  Prompt {i + 1}/{len(prompts)}: Cache HIT — skipping Qwen encode")
        else:
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
                    report.stop_phase(f"text_encode_{i + 1}", profiler.get_elapsed(f"text_encode_{i + 1}") or 0.0, profiler.get_phase_rss(f"text_encode_{i + 1}"))
                    report.add_metadata(f"text_encode_{i + 1}", "cache", "MISS")
            if report and report.verbose:
                print(f"  Prompt {i + 1}/{len(prompts)}: Cache MISS — encoding with Qwen")

    # Unload Qwen once after all prompts are encoded
    if profiler:
        profiler.start("text_encoder_unload")
    pipeline.text_encoder.unload()
    gc.collect()
    mx.clear_cache()
    if profiler:
        profiler.stop("text_encoder_unload")
    if report:
        report.stop_phase("text_encoder_unload", profiler.get_elapsed("text_encoder_unload") or 0.0, profiler.get_phase_rss("text_encoder_unload"))
    if report and report.verbose:
        print("  Qwen unloaded (batch encoding complete)")

    # --- Phase 1.5: Load DiT + VAE (after Qwen is unloaded) ---
    # This is the key optimization: DiT + VAE are loaded AFTER the text
    # encoder is unloaded, so peak RAM is max(Qwen, DiT+VAE) instead of
    # Qwen + DiT + VAE simultaneously.
    # load_dit_vae() handles profiler.start/stop for dit_load and vae_load internally
    pipeline.load_dit_vae(
        model_dir=current_params["model"],
        quantize=current_params.get("quantize"),
        profiler=profiler,
    )
    if report:
        report.stop_phase("dit_load", profiler.get_elapsed("dit_load") or 0.0, profiler.get_phase_rss("dit_load"))
        report.stop_phase("vae_load", profiler.get_elapsed("vae_load") or 0.0, profiler.get_phase_rss("vae_load"))

    # --- Phase 2: Generate all images using cached embeddings ---
    if report and report.verbose:
        print(f"\n{'=' * 60}")
        print("Phase 2: Generating images (DiT + VAE)")
        print(f"{'=' * 60}")

    # Collect per-prompt metadata for JSON output
    prompt_metadata: list[dict[str, Any]] = []

    for i, prompt_cfg in enumerate(prompts):
        params = merge_params(defaults, prompt_cfg)

        # Print thermal state at the start of each prompt (verbose mode only;
        # always captured for metadata output)
        thermal_state = None
        if report:
            thermal_state = Profiler.get_thermal_state()
            report.print_thermal_state(thermal_state)

        # Print prompt header via LiveReport
        if report and report.verbose:
            report.prompt_header(i + 1, len(prompts))

        # Check if pipeline reload or scheduler reset is needed
        needs_pipeline, needs_scheduler = needs_reload(current_params, params)

        # Handle pipeline reload (model/quantize change)
        if needs_pipeline:
            if profiler:
                profiler.start(f"pipeline_load_{i + 1}")
            # Only reload DiT + VAE — the text encoder is already unloaded
            # and we use cached embeddings for all prompts.
            pipeline.load_dit_vae(
                model_dir=params["model"],
                quantize=params.get("quantize"),
                profiler=profiler,
            )
            current_params = dict(params)
            if profiler:
                profiler.stop(f"pipeline_load_{i + 1}")
            if report:
                report.stop_phase(f"pipeline_load_{i + 1}", profiler.get_elapsed(f"pipeline_load_{i + 1}") or 0.0, profiler.get_phase_rss(f"pipeline_load_{i + 1}"))

        # Handle scheduler reset (steps change)
        if needs_scheduler:
            pipeline.scheduler.set_timesteps(params["steps"])
            pipeline.num_steps = params["steps"]
            current_params = dict(params)

        # Set per-prompt metadata BEFORE generation starts (so it appears right after prompt header)
        if report and report.verbose:
            report.add_metadata(f"generation_{i + 1}", "prompt", params["prompt"])
            report.add_metadata(f"generation_{i + 1}", "resolution", f"{params['width']}x{params['height']}")
            report.add_metadata(f"generation_{i + 1}", "steps", str(params["steps"]))
            report.add_metadata(f"generation_{i + 1}", "quantize", str(params.get("quantize")))
            report.add_metadata(f"generation_{i + 1}", "seed", str(params["seed"]))
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
            profiler.set_metadata(f"generation_{i + 1}", "seed", str(params["seed"]))
        if report:
            report.stop_phase(f"generation_{i + 1}", profiler.get_elapsed(f"generation_{i + 1}") or 0.0, profiler.get_phase_rss(f"generation_{i + 1}"))

        # Resolve output path (unified across all modes)
        params["output"] = resolve_output_path(
            output=params.get("output"),
            width=params["width"],
            height=params["height"],
            steps=params["steps"],
            seed=params["seed"],
            quantize=params.get("quantize"),
            mode="txt2img",
        )

        # Save image
        if profiler:
            profiler.start(f"save_{i + 1}")
        image.save(params["output"])
        if profiler:
            profiler.stop(f"save_{i + 1}")
        if report:
            report.stop_phase(f"save_{i + 1}", profiler.get_elapsed(f"save_{i + 1}") or 0.0, profiler.get_phase_rss(f"save_{i + 1}"), saved_file=params["output"])

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
            "seed": params["seed"],
            "generation_time_seconds": gen_elapsed,
            "peak_rss_gib": gen_peak_rss,
            "image_path": params["output"],
            "thermal_state": thermal_state,
        })

        # Add to LiveReport for per-prompt summary table
        if report:
            report.add_prompt(
                index=i + 1,
                prompt=params["prompt"],
                resolution=f"{params['width']}x{params['height']}",
                steps=params["steps"],
                quantize=params.get("quantize"),
                seed=params["seed"],
                generation_time=gen_elapsed,
                peak_rss_gib=gen_peak_rss,
                saved_file=params["output"],
                thermal_state=thermal_state,
            )

    # --- Stop total_wall_clock before saving metadata ---
    if profiler:
        profiler.stop("total_wall_clock")

    # --- Always build metadata dict for summary report ---
    metadata = {
        "model": _get_model_name(defaults["model"]),
        "base_model": _get_base_model(defaults["model"]),
        "generation_time_seconds": profiler.get_elapsed("total_wall_clock") if profiler else None,
        "created_at": datetime.now().isoformat(),
        "image_path": None,
        "image_paths": None,
        "image_strength": None,
        "peak_memory_gib": profiler.get_peak_rss_gib() if profiler else None,
    }

    if metadata_enabled and profiler:
        base_path = resolve_metadata_path(output, jsonl_path)
        # Update metadata with final values
        profiler.metadata["generation_time_seconds"] = profiler.get_elapsed("total_wall_clock")
        profiler.metadata["peak_memory_gib"] = profiler.get_peak_rss_gib()
        # Build overview from prompt_metadata
        profiler.overview = [
            {
                "index": pm["index"],
                "time": pm["generation_time_seconds"],
                "peak_rss_gib": pm["peak_rss_gib"],
                "resolution": pm["resolution"],
                "steps": pm["steps"],
                "seed": pm.get("seed"),
                "file": pm["image_path"],
                "thermal_state": pm.get("thermal_state"),
            }
            for pm in prompt_metadata
        ]
        # Build summary
        total_time = profiler.get_elapsed("total_wall_clock") or 0.0
        sum_gen = sum(
            pm["generation_time_seconds"]
            for pm in prompt_metadata
            if pm["generation_time_seconds"] is not None
        )
        text_encode_time = 0.0
        for rec in profiler.get_records():
            if rec.elapsed is None:
                continue
            if (
                rec.name.startswith("text_encode")
                or rec.name == "text_encoder_unload"
                or rec.name == "dit_load"
                or rec.name == "vae_load"
            ):
                text_encode_time += rec.elapsed
        overhead = total_time - sum_gen - text_encode_time
        profiler.summary = {
            "total_time": total_time,
            "peak_ram": profiler.get_peak_rss_gib() or 0.0,
            "prompts_count": len(prompts),
            "overhead": overhead,
            "text_encode_time": text_encode_time,
        }
        profiler.save_metadata(
            base_path,
            profiler.metadata,
            overview=profiler.overview,
            summary=profiler.summary,
        )

    return metadata, prompt_metadata


# ---------------------------------------------------------------------------
# Edit worker
# ---------------------------------------------------------------------------

# Valid per-prompt parameters for edit mode (extends txt2img with image fields)
EDIT_VALID_PARAMS = VALID_PARAMS | {"image", "ref_images"}


def load_edit_prompts(jsonl_path: str) -> list[dict[str, Any]]:
    """Load edit prompts from a JSONL file with image path validation.

    Each line must be a JSON object with at least a ``prompt`` field and
    either an ``image`` field (target image to edit) or a ``ref_images``
    field (list of reference image paths). If ``image`` is missing, the
    first entry in ``ref_images`` is used as the target.

    Image paths are validated for existence and openability. Prompts with
    missing or malformed image paths are skipped with a warning.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        List of validated prompt dictionaries with ``image`` and
        ``ref_images`` fields resolved to lists of strings.
    """
    from PIL import Image

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

            # Resolve image / ref_images
            image = prompt.get("image")
            ref_images = prompt.get("ref_images")

            if image is not None:
                image = str(image)
            if ref_images is not None:
                if isinstance(ref_images, str):
                    ref_images = [ref_images]
                ref_images = [str(r) for r in ref_images]

            # Determine the full reference list
            if image is not None and ref_images is not None:
                all_refs = [image] + ref_images
            elif image is not None:
                all_refs = [image]
            elif ref_images is not None and ref_images:
                all_refs = ref_images
                image = ref_images[0]
            else:
                print(f"  WARNING: Skipping line {line_num}: missing 'image' or 'ref_images' field")
                continue

            # Validate all image paths
            valid_refs = []
            skip = False
            for ref_path in all_refs:
                if not os.path.exists(ref_path):
                    print(f"  WARNING: Skipping line {line_num}: image not found: {ref_path}")
                    skip = True
                    break
                try:
                    with Image.open(ref_path) as img:
                        img.verify()
                except Exception as e:
                    print(f"  WARNING: Skipping line {line_num}: malformed image {ref_path}: {e}")
                    skip = True
                    break
                valid_refs.append(ref_path)

            if skip:
                continue

            # Normalize: image is the first ref, ref_images is the rest
            prompt["image"] = valid_refs[0]
            prompt["ref_images"] = valid_refs[1:] if len(valid_refs) > 1 else []

            # Default output
            if "output" not in prompt:
                prompt["output"] = None

            # Validate parameter names
            invalid = set(prompt.keys()) - EDIT_VALID_PARAMS
            if invalid:
                print(f"  WARNING: Skipping line {line_num}: unknown parameters: {invalid}")
                continue

            prompts.append(prompt)

    return prompts


def _hash_image_bytes(path: str) -> str:
    """Compute SHA-256 hash of raw image file bytes (no Pillow dependency)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_edit_worker(
    jsonl_path: str,
    defaults: dict[str, Any],
    profiler: Optional[Profiler] = None,
    metadata_enabled: bool = False,
    report=None,
    output: Optional[str] = None,
) -> tuple[Optional[dict], Optional[list]]:
    """Run the persistent JSONL edit worker in prompt queue mode.

    Mirrors ``run_worker()`` but for image editing:

    Phase 0: Load text encoder + tokenizer (edit model, DiT + VAE deferred).
    Phase 1: For each prompt — validate image paths, load reference images,
             compute image hashes, check embedding cache (keyed by prompt +
             ref image hashes), encode edit prompt via Qwen (multimodal),
             save to cache. Unload Qwen after all prompts.
    Phase 1.5: Load DiT + VAE (after Qwen is unloaded — reduces peak RAM).
    Phase 2: For each prompt — check vision cache for VAE-encoded reference
             latents, encode via VAE if miss, run edit denoising loop,
             decode, save image.

    The key optimization: Qwen (~7.5 GiB) is loaded and unloaded BEFORE
    DiT + VAE (~7.9 GiB) are loaded, so peak RAM is max(Qwen, DiT+VAE)
    instead of Qwen + DiT + VAE simultaneously (~15.4 GiB).

    Args:
        jsonl_path: Path to the JSONL prompts file
        defaults: Default parameters from CLI
        profiler: Optional Profiler instance
        metadata_enabled: If True, save JSON + markdown metadata files
        report: Optional LiveReport instance for real-time terminal output

    Returns:
        (metadata dict, prompt_metadata list) if metadata_enabled, else (None, None).
    """
    from mage_mlx.mflux_src.mflux.models.mage_flow.variants.edit.mage_flow_edit import (
        MageFlowEdit,
    )
    from mage_mlx.embedding_cache import EmbeddingCache
    from mage_mlx.vision_cache import VisionCache
    from mage_mlx.mflux_src.mflux.models.mage_flow.variants.edit.util import (
        MageFlowEditUtil,
    )
    from mage_mlx.mflux_src.mflux.models.mage_flow.latent_creator import (
        MageFlowLatentCreator,
    )
    from mage_mlx.mflux_src.mflux.models.mage_flow.variants.pipeline_helpers import (
        make_velocity_predictor,
        resolve_seed,
        resolve_generation_parameters,
    )
    from mage_mlx.mflux_src.mflux.models.common.config.config import Config
    from mage_mlx.mflux_src.mflux.utils.image_util import ImageUtil

    # Load prompts
    prompts = load_edit_prompts(jsonl_path)
    if not prompts:
        print("No valid prompts found in JSONL file.")
        return None, None

    if report and report.verbose:
        print(f"  Loaded {len(prompts)} edit prompts from {jsonl_path}")
        if profiler:
            profiler.log(f"  Loaded {len(prompts)} edit prompts from {jsonl_path}")

    # Set up incremental save path on profiler
    if metadata_enabled and profiler:
        base_path = resolve_metadata_path(output, jsonl_path)
        profiler.metadata_path = base_path
        profiler.metadata = {
            "model": _get_model_name(defaults["model"]),
            "base_model": _get_base_model(defaults["model"]),
            "generation_time_seconds": None,
            "created_at": datetime.now().isoformat(),
            "image_path": None,
            "image_paths": None,
            "image_strength": None,
            "peak_memory_gib": None,
        }

    # Real-time callback for LiveReport
    _EXPLICIT_EXACT = {"dit_load", "vae_load"}
    _EXPLICIT_PREFIXES = (
        "pipeline_load", "text_encoder_unload", "text_encode_",
        "vae_encode_ref_", "generation_", "save_", "total_wall_clock",
        "python_startup",
    )
    if report and profiler:
        def _on_phase_complete(name, elapsed, rss):
            if name in _EXPLICIT_EXACT or any(name.startswith(p) for p in _EXPLICIT_PREFIXES):
                return
            loading_mode = "lazy loading" if name == "text_encoder_load" else None
            report.stop_phase(name, elapsed or 0.0, rss, loading_mode=loading_mode)
        profiler.on_phase_complete = _on_phase_complete

    # Track current pipeline state
    current_params: dict[str, Any] = {}
    edit: Optional[MageFlowEdit] = None
    te_path: Optional[str] = None

    # --- Phase 0: Load text encoder + tokenizer (DiT + VAE deferred) ---
    first_params = merge_params(defaults, prompts[0])
    from mage_mlx.loader import ensure_mlx_model
    converted_model, _ = ensure_mlx_model(
        first_params["model"],
        quantize=first_params.get("quantize"),
        profiler=profiler,
    )
    first_params["model"] = converted_model
    defaults["model"] = converted_model
    if profiler:
        profiler.start("pipeline_load")
    # Load text encoder from shared path (text encoder is shared across all models)
    from mage_mlx.pipeline import resolve_text_encoder_path
    te_weights_path = resolve_text_encoder_path("models/microsoft_Mage-Flow-Turbo")
    from mage_mlx.text_encoder import MageFlowTextEncoder
    te = MageFlowTextEncoder(
        model_path=te_weights_path if os.path.exists(te_weights_path) else None,
    )

    edit = MageFlowEdit(
        quantize=first_params.get("quantize"),
        model_path=first_params["model"],
        load_dit_vae=False,
        text_encoder=te,
    )
    # Set tokenizer from Qwen/Qwen3-VL-8B-Instruct (cached from regular worker)
    from transformers import AutoTokenizer
    from mage_mlx.pipeline import MageFlowTokenizer
    raw_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    edit.tokenizers = {"mage": MageFlowTokenizer(raw_tokenizer)}
    te_path = te_weights_path if os.path.exists(te_weights_path) else None
    current_params = dict(first_params)

    # Initialize caches AFTER model is loaded to avoid creating stray
    # directories that intercept PathResolution for HF IDs
    cache = EmbeddingCache(converted_model)
    vision_cache = VisionCache(converted_model)
    if profiler:
        profiler.stop("pipeline_load")
    if report:
        report.stop_phase(
            "pipeline_load",
            profiler.get_elapsed("pipeline_load") or 0.0,
            profiler.get_phase_rss("pipeline_load"),
            grey_separator=True,
        )

    # --- Phase 1: Pre-encode all edit prompts (load Qwen once, encode all, unload) ---
    if report and report.verbose:
        print(f"\n{'=' * 60}")
        print("Phase 1: Pre-encoding edit prompts (Qwen batch mode)")
        print(f"{'=' * 60}")

    # Store embeddings in memory for Phase 2
    # Each entry: (pos_embeds, pos_mask, neg_embeds, neg_mask)
    prompt_embeds: dict[int, tuple] = {}

    for i, prompt_cfg in enumerate(prompts):
        params = merge_params(defaults, prompt_cfg)

        # Load reference images
        from PIL import Image
        ref_image_paths = [params["image"]] + params.get("ref_images", [])
        ref_images = [Image.open(p).convert("RGB") for p in ref_image_paths]

        # Compute image hashes for cache key
        ref_image_hashes = [_hash_image_bytes(p) for p in ref_image_paths]

        # Build cache key (includes ref image hashes for edit)
        cache_key = cache.make_key(
            prompt=params["prompt"],
            negative_prompt=params.get("negative_prompt", " "),
            te_path=te_path,
            ref_image_hashes=ref_image_hashes,
        )
        cached_conditioning = cache.get_conditioning(cache_key)

        if cached_conditioning is not None:
            cached_embeds, pos_mask = cached_conditioning
            # Cache HIT — still need to encode negative if guidance > 1.0
            neg_embeds = None
            neg_mask = None
            if params.get("guidance", 1.0) > 1.0:
                if profiler:
                    profiler.start(f"text_encode_neg_{i + 1}")
                neg_embeds, neg_mask = MageFlowConditioning.encode_edit(
                    prompts=[params.get("negative_prompt", " ")],
                    images_per_prompt=[ref_images],
                    tokenizer=edit.tokenizers["mage"],
                    text_encoder=edit.text_encoder,
                    max_sequence_length=2048,
                )
                mx.eval(neg_embeds, neg_mask)
                if profiler:
                    profiler.stop(f"text_encode_neg_{i + 1}")
                    profiler.set_metadata(f"text_encode_neg_{i + 1}", "cache", "HIT")
                    if report:
                        report.stop_phase(
                            f"text_encode_neg_{i + 1}",
                            profiler.get_elapsed(f"text_encode_neg_{i + 1}") or 0.0,
                            profiler.get_phase_rss(f"text_encode_neg_{i + 1}"),
                        )
                        report.add_metadata(f"text_encode_neg_{i + 1}", "cache", "HIT")
            prompt_embeds[i] = (cached_embeds, pos_mask, neg_embeds, neg_mask)
            if report and report.verbose:
                print(f"  Prompt {i + 1}/{len(prompts)}: Cache HIT — skipping Qwen encode")
        else:
            if profiler:
                profiler.start(f"text_encode_{i + 1}")

            # Encode positive edit prompt (Qwen stays loaded)
            pos_embeds, pos_mask = MageFlowConditioning.encode_edit(
                prompts=[params["prompt"]],
                images_per_prompt=[ref_images],
                tokenizer=edit.tokenizers["mage"],
                text_encoder=edit.text_encoder,
                max_sequence_length=2048,
            )
            mx.eval(pos_embeds, pos_mask)

            # Save to cache for future runs
            cache.put_conditioning(cache_key, pos_embeds, pos_mask)

            # Encode negative prompt if guidance > 1.0
            neg_embeds = None
            neg_mask = None
            if params.get("guidance", 1.0) > 1.0:
                neg_embeds, neg_mask = MageFlowConditioning.encode_edit(
                    prompts=[params.get("negative_prompt", " ")],
                    images_per_prompt=[ref_images],
                    tokenizer=edit.tokenizers["mage"],
                    text_encoder=edit.text_encoder,
                    max_sequence_length=2048,
                )
                mx.eval(neg_embeds, neg_mask)

            prompt_embeds[i] = (pos_embeds, pos_mask, neg_embeds, neg_mask)

            if profiler:
                profiler.stop(f"text_encode_{i + 1}")
                profiler.set_metadata(f"text_encode_{i + 1}", "cache", "MISS")
                if report:
                    report.stop_phase(
                        f"text_encode_{i + 1}",
                        profiler.get_elapsed(f"text_encode_{i + 1}") or 0.0,
                        profiler.get_phase_rss(f"text_encode_{i + 1}"),
                    )
                    report.add_metadata(f"text_encode_{i + 1}", "cache", "MISS")
            if report and report.verbose:
                print(f"  Prompt {i + 1}/{len(prompts)}: Cache MISS — encoding with Qwen")

    # Unload Qwen once after all prompts are encoded
    if profiler:
        profiler.start("text_encoder_unload")
    edit.text_encoder.unload()
    gc.collect()
    mx.clear_cache()
    if profiler:
        profiler.stop("text_encoder_unload")
    if report:
        report.stop_phase(
            "text_encoder_unload",
            profiler.get_elapsed("text_encoder_unload") or 0.0,
            profiler.get_phase_rss("text_encoder_unload"),
        )
    if report and report.verbose:
        print("  Qwen unloaded (batch encoding complete)")

    # --- Phase 1.5: Load DiT + VAE (after Qwen is unloaded) ---
    edit.load_dit_vae(profiler=profiler)
    if report:
        report.stop_phase("dit_load", profiler.get_elapsed("dit_load") or 0.0, profiler.get_phase_rss("dit_load"))
        report.stop_phase("vae_load", profiler.get_elapsed("vae_load") or 0.0, profiler.get_phase_rss("vae_load"))

    # --- Phase 2: Generate all images using cached embeddings ---
    if report and report.verbose:
        print(f"\n{'=' * 60}")
        print("Phase 2: Generating edited images (DiT + VAE)")
        print(f"{'=' * 60}")

    prompt_metadata: list[dict[str, Any]] = []

    for i, prompt_cfg in enumerate(prompts):
        params = merge_params(defaults, prompt_cfg)

        # Print thermal state at the start of each prompt (verbose mode only;
        # always captured for metadata output)
        thermal_state = None
        if report:
            thermal_state = Profiler.get_thermal_state()
            report.print_thermal_state(thermal_state)

        if report and report.verbose:
            report.prompt_header(i + 1, len(prompts))

        # Check if pipeline reload or scheduler reset is needed
        needs_pipeline, needs_scheduler = needs_reload(current_params, params)

        if needs_pipeline:
            if profiler:
                profiler.start(f"pipeline_load_{i + 1}")
            edit.load_dit_vae(
                model_dir=params["model"],
                quantize=params.get("quantize"),
                profiler=profiler,
            )
            current_params = dict(params)
            if profiler:
                profiler.stop(f"pipeline_load_{i + 1}")
            if report:
                report.stop_phase(
                    f"pipeline_load_{i + 1}",
                    profiler.get_elapsed(f"pipeline_load_{i + 1}") or 0.0,
                    profiler.get_phase_rss(f"pipeline_load_{i + 1}"),
                )

        if needs_scheduler:
            edit.scheduler.set_timesteps(params["steps"])
            edit.num_steps = params["steps"]
            current_params = dict(params)

        # Set per-prompt metadata
        if report and report.verbose:
            report.add_metadata(f"generation_{i + 1}", "prompt", params["prompt"])
            report.add_metadata(f"generation_{i + 1}", "resolution", f"{params['width']}x{params['height']}")
            report.add_metadata(f"generation_{i + 1}", "steps", str(params["steps"]))
            report.add_metadata(f"generation_{i + 1}", "quantize", str(params.get("quantize")))
            report.add_metadata(f"generation_{i + 1}", "seed", str(params["seed"]))
            print()

        if profiler:
            profiler.start(f"generation_{i + 1}")

        # Load reference images for this prompt
        ref_image_paths = [params["image"]] + params.get("ref_images", [])
        ref_images = [Image.open(p).convert("RGB") for p in ref_image_paths]

        # Resolve seed
        seed = resolve_seed(params["seed"])

        # Resolve generation parameters
        num_steps, guidance = resolve_generation_parameters(
            model_config=edit.model_config,
            num_inference_steps=params.get("steps", 4),
            guidance=params.get("guidance", 1.0),
        )

        # Resolve target size
        width, height = MageFlowEditUtil.resolve_target_size(
            ref_images[0],
            width=params["width"],
            height=params["height"],
        )

        # Build config
        config = Config(
            model_config=edit.model_config,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=params["image"],
            scheduler="mage_flow",
        )

        # Get cached embeddings
        pos_embeds, pos_mask, neg_embeds, neg_mask = prompt_embeds[i]

        # Stack for CFG if needed
        if neg_embeds is not None:
            # Pad to same length if needed
            pos_len = pos_embeds.shape[1]
            neg_len = neg_embeds.shape[1]
            if pos_len != neg_len:
                max_len = max(pos_len, neg_len)
                from mage_mlx.prompt_processor import MageFlowPromptProcessor
                pos_embeds, pos_mask = MageFlowPromptProcessor.trim_and_pad_hidden_states(
                    mx.concatenate([pos_embeds, neg_embeds], axis=0),
                    mx.concatenate([pos_mask, neg_mask], axis=0),
                    drop_tokens=0,
                    max_length=max_len,
                )
            else:
                pos_embeds = mx.concatenate([pos_embeds, neg_embeds], axis=0)
                pos_mask = mx.concatenate([pos_mask, neg_mask], axis=0)

        # Check vision cache for reference latents (VAE-encoded)
        vae_path = os.path.join(params["model"], "vae.safetensors")
        with_bytes = []
        for ref_path in ref_image_paths:
            with open(ref_path, "rb") as ref_file:
                with_bytes.append(ref_file.read())
        vision_key = vision_cache.make_key(
            image_bytes=with_bytes,
            size=(config.width, config.height),
            vae_path=vae_path,
            seed=seed,
        )
        if profiler:
            profiler.start(f"vae_encode_ref_{i + 1}")
        reference_latents = vision_cache.get(vision_key)
        if reference_latents is not None:
            cache_status = "HIT"
            if report and report.verbose:
                print(f"  Vision cache HIT — skipping VAE encode for ref image(s)")
        else:
            cache_status = "MISS"
            reference_latents = MageFlowEditUtil.encode_references(
                edit.vae,
                ref_images,
                width=config.width,
                height=config.height,
                seed=seed,
            )
            vision_cache.put(vision_key, reference_latents)
            if report and report.verbose:
                print(f"  Vision cache MISS — encoding ref image(s) with VAE")
        if profiler:
            profiler.stop(f"vae_encode_ref_{i + 1}")
            profiler.set_metadata(f"vae_encode_ref_{i + 1}", "cache", cache_status)
            if report:
                report.stop_phase(
                    f"vae_encode_ref_{i + 1}",
                    profiler.get_elapsed(f"vae_encode_ref_{i + 1}") or 0.0,
                    profiler.get_phase_rss(f"vae_encode_ref_{i + 1}"),
                )
                report.add_metadata(f"vae_encode_ref_{i + 1}", "cache", cache_status)

        # Create noise latents
        target_latents = MageFlowLatentCreator.create_noise(
            seed=seed,
            height=config.height,
            width=config.width,
            dtype=edit.model_config.precision,
        )
        mx.eval(target_latents, reference_latents, pos_embeds, pos_mask)

        # Build velocity predictor
        latent_height = config.height // 16
        latent_width = config.width // 16
        target_length = target_latents.shape[1]
        image_shapes = [(1, latent_height, latent_width)] * (1 + len(ref_images))
        predict = make_velocity_predictor(
            transformer=edit.transformer,
            text_embeddings=pos_embeds,
            text_attention_mask=pos_mask,
            image_shapes=image_shapes,
            guidance=guidance,
            target_length=target_length,
            renormalization=params.get("renormalization", False),
        )

        # Denoising loop (mirrors MageFlowEdit.generate_image for parity)
        from mage_mlx.mflux_src.mflux.utils.exceptions import StopImageGenerationException
        ctx = edit.callbacks.start(seed=seed, prompt=params["prompt"], config=config)
        ctx.before_loop(target_latents)
        for step in config.time_steps:
            try:
                if profiler:
                    profiler.start(f"edit_step_{step + 1}")
                model_input = mx.concatenate([target_latents, reference_latents], axis=1)
                velocity = predict(model_input, config.scheduler.sigmas[step])
                target_latents = config.scheduler.step(
                    noise=velocity,
                    timestep=step,
                    latents=target_latents,
                    sigmas=config.scheduler.sigmas,
                )
                ctx.in_loop(step, target_latents)
                mx.eval(target_latents)
                if profiler:
                    profiler.stop(f"edit_step_{step + 1}")
            except KeyboardInterrupt:
                ctx.interruption(step, target_latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {step + 1}/{config.num_inference_steps}"
                )
        # Release the predictor closure and final evaluated graph before
        # low-RAM callbacks evict the model (matches generate_image).
        del predict, velocity, model_input
        ctx.after_loop(target_latents)

        # Decode
        if profiler:
            profiler.start("vae_decode")
        decoded = edit.vae.decode(
            MageFlowLatentCreator.unpack_latents(
                target_latents,
                height=config.height,
                width=config.width,
            )
        )
        mx.eval(decoded)
        if profiler:
            profiler.stop("vae_decode")

        image = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=params["prompt"],
            negative_prompt=params.get("negative_prompt"),
            quantization=getattr(edit, "bits", None),
            lora_paths=getattr(edit, "lora_paths", None),
            lora_scales=getattr(edit, "lora_scales", None),
            image_path=params["image"],
            image_paths=params.get("ref_images") or None,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

        if profiler:
            profiler.stop(f"generation_{i + 1}")
            profiler.set_metadata(f"generation_{i + 1}", "prompt", params["prompt"])
            profiler.set_metadata(f"generation_{i + 1}", "resolution", f"{params['width']}x{params['height']}")
            profiler.set_metadata(f"generation_{i + 1}", "steps", str(params["steps"]))
            profiler.set_metadata(f"generation_{i + 1}", "quantize", str(params.get("quantize")))
            profiler.set_metadata(f"generation_{i + 1}", "seed", str(params["seed"]))
        if report:
            report.stop_phase(
                f"generation_{i + 1}",
                profiler.get_elapsed(f"generation_{i + 1}") or 0.0,
                profiler.get_phase_rss(f"generation_{i + 1}"),
            )

        # Resolve output path (unified across all modes)
        params["output"] = resolve_output_path(
            output=params.get("output"),
            width=params["width"],
            height=params["height"],
            steps=params["steps"],
            seed=params["seed"],
            quantize=params.get("quantize"),
            mode="edit",
        )

        # Save image
        if profiler:
            profiler.start(f"save_{i + 1}")
        image.save(params["output"])
        if profiler:
            profiler.stop(f"save_{i + 1}")
        if report:
            report.stop_phase(
                f"save_{i + 1}",
                profiler.get_elapsed(f"save_{i + 1}") or 0.0,
                profiler.get_phase_rss(f"save_{i + 1}"),
                saved_file=params["output"],
            )

        # Collect per-prompt metadata
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
            "seed": params["seed"],
            "generation_time_seconds": gen_elapsed,
            "peak_rss_gib": gen_peak_rss,
            "image_path": params["output"],
            "ref_images": ref_image_paths,
            "thermal_state": thermal_state,
        })

        if report:
            report.add_prompt(
                index=i + 1,
                prompt=params["prompt"],
                resolution=f"{params['width']}x{params['height']}",
                steps=params["steps"],
                quantize=params.get("quantize"),
                seed=params["seed"],
                generation_time=gen_elapsed,
                peak_rss_gib=gen_peak_rss,
                saved_file=params["output"],
                thermal_state=thermal_state,
            )

    # --- Stop total_wall_clock before saving metadata ---
    if profiler:
        profiler.stop("total_wall_clock")

    # --- Build metadata dict ---
    metadata = {
        "model": _get_model_name(defaults["model"]),
        "base_model": _get_base_model(defaults["model"]),
        "generation_time_seconds": profiler.get_elapsed("total_wall_clock") if profiler else None,
        "created_at": datetime.now().isoformat(),
        "image_path": None,
        "image_paths": None,
        "image_strength": None,
        "peak_memory_gib": profiler.get_peak_rss_gib() if profiler else None,
    }

    if metadata_enabled and profiler:
        base_path = resolve_metadata_path(output, jsonl_path)
        profiler.metadata["generation_time_seconds"] = profiler.get_elapsed("total_wall_clock")
        profiler.metadata["peak_memory_gib"] = profiler.get_peak_rss_gib()
        profiler.overview = [
            {
                "index": pm["index"],
                "time": pm["generation_time_seconds"],
                "peak_rss_gib": pm["peak_rss_gib"],
                "resolution": pm["resolution"],
                "steps": pm["steps"],
                "seed": pm.get("seed"),
                "file": pm["image_path"],
                "thermal_state": pm.get("thermal_state"),
            }
            for pm in prompt_metadata
        ]
        total_time = profiler.get_elapsed("total_wall_clock") or 0.0
        sum_gen = sum(
            pm["generation_time_seconds"]
            for pm in prompt_metadata
            if pm["generation_time_seconds"] is not None
        )
        text_encode_time = 0.0
        for rec in profiler.get_records():
            if rec.elapsed is None:
                continue
            if (
                rec.name.startswith("text_encode")
                or rec.name.startswith("vae_encode_ref_")
                or rec.name == "text_encoder_unload"
                or rec.name == "dit_load"
                or rec.name == "vae_load"
            ):
                text_encode_time += rec.elapsed
        overhead = total_time - sum_gen - text_encode_time
        profiler.summary = {
            "total_time": total_time,
            "peak_ram": profiler.get_peak_rss_gib() or 0.0,
            "prompts_count": len(prompts),
            "overhead": overhead,
            "text_encode_time": text_encode_time,
        }
        profiler.save_metadata(
            base_path,
            profiler.metadata,
            overview=profiler.overview,
            summary=profiler.summary,
        )

    return metadata, prompt_metadata


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

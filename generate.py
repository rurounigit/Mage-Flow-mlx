"""Generate images with Mage-Flow MLX.

CLI tool for text-to-image generation using the MLX port of Mage-Flow.

Usage:
    python generate.py --prompt "A futuristic cityscape at sunset"
    python generate.py --prompt "..." --steps 4 --height 1024 --width 1024 --seed 42
    python generate.py --prompt "..." --output output.png
    python generate.py --prompt "..." --model microsoft/Mage-Flow-Turbo
    python generate.py --prompt "..." --quantize 4
    python generate.py --prompt "..." --metadata
    python generate.py --worker prompts.jsonl --metadata
    python generate.py --benchmark-cleanup --prompt "test"
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Optional

import mlx.core as mx


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


def _collect_metadata(
    model: str,
    image_path: Optional[str],
    image_paths: Optional[list[str]],
    generation_time: Optional[float],
    peak_memory_gib: Optional[float],
) -> dict:
    """Collect run-level metadata for a generation run.

    Note: prompt, negative_prompt, and quantize are per-prompt values that
    appear in the profile table metadata, not in this run-level metadata dict.
    """
    return {
        "model": _get_model_name(model),
        "base_model": _get_base_model(model),
        "generation_time_seconds": generation_time,
        "created_at": datetime.now().isoformat(),
        "image_path": image_path,
        "image_paths": image_paths,
        "image_strength": None,
        "peak_memory_gib": peak_memory_gib,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with Mage-Flow MLX (Apple Silicon)"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to the BF16 MLX model directory or HuggingFace repo ID (e.g. microsoft/Mage-Flow-Turbo)"
    )
    parser.add_argument(
        "--steps", type=int, default=4,
        help="Number of denoising steps (4 for turbo)"
    )
    parser.add_argument(
        "--height", type=int, default=1024,
        help="Output image height (must be multiple of 16)"
    )
    parser.add_argument(
        "--width", type=int, default=1024,
        help="Output image width (must be multiple of 16)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--guidance", type=float, default=1.0,
        help="Classifier-free guidance scale (1 disables CFG)"
    )
    parser.add_argument(
        "--negative-prompt", type=str, default=" ",
        help="Negative prompt used by CFG"
    )
    parser.add_argument(
        "--output", type=str, default="output.png",
        help="Output image path"
    )
    parser.add_argument(
        "--quantize", type=int, default=None, choices=[4, 8],
        help="Quantize supported DiT layers to 4 or 8 bits in memory"
    )
    parser.add_argument(
        "--metadata", action="store_true",
        help="Enable phase-level profiling, print terminal report, and save JSON + markdown files"
    )
    parser.add_argument(
        "--worker", type=str, default=None, metavar="JSONL",
        help="Run in persistent JSONL worker mode: load models once, process prompts from JSONL file"
    )
    parser.add_argument(
        "--edit", action="store_true",
        help="Run in edit worker mode: same as --worker but for image editing (requires --worker JSONL with image/ref_images fields)"
    )

    parser.add_argument(
        "--benchmark-cleanup", action="store_true",
        help="Benchmark all 4 Qwen cleanup strategies (unload, gc, clear_cache, all)"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Target image path for editing (use with --ref-images)"
    )
    parser.add_argument(
        "--ref-images", type=str, default=None,
        help="Comma-separated reference image paths for editing"
    )
    parser.add_argument(
        "--renormalization", action="store_true",
        help="Renormalize velocity predictions during editing"
    )
    args = parser.parse_args()
    model_was_explicit = args.model is not None
    if args.model is None:
        args.model = (
            "microsoft/Mage-Flow-Edit-Turbo"
            if args.image is not None
            else "models/microsoft_Mage-Flow-Turbo"
        )

    # Validate: need either --prompt or --worker
    if args.worker is None and args.prompt is None:
        parser.error("either --prompt or --worker is required")

    # Validate: --edit requires --worker
    if args.edit and args.worker is None:
        parser.error("--edit requires --worker to be set")

    # Validate dimensions
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        print(f"Error: height and width must be positive multiples of 16, got {args.height}x{args.width}")
        sys.exit(1)
    if args.steps <= 0:
        print(f"Error: steps must be positive, got {args.steps}")
        sys.exit(1)
    if args.guidance < 1.0:
        print(f"Error: guidance must be at least 1.0, got {args.guidance}")
        sys.exit(1)

    # Edit requires the dedicated Mage-Flow-Edit checkpoint; txt2img Turbo
    # weights are not trained for multimodal editing.
    if args.image is not None and model_was_explicit and "edit" not in args.model.lower():
        parser.error(
            "the explicitly selected model is not a Mage-Flow-Edit checkpoint: "
            f"{args.model}. Omit --model to use Mage-Flow-Edit-Turbo automatically, "
            "or select a dedicated Edit checkpoint."
        )

    # --- Phase: Python/import startup ---
    from mage_mlx.profiler import Profiler, LiveReport, _C

    prof = Profiler(enabled=True, track_memory=True)

    # Create LiveReport for real-time terminal output
    verbose = bool(args.metadata)
    report = LiveReport(title="Mage-Flow MLX", verbose=verbose, profiler=prof)

    # Real-time callback: report sub-phases as they complete
    # Phases that are explicitly reported via stop_phase (skip in callback)
    # Use exact names for numbered phases, prefixes for generic ones
    # Phases reported explicitly via report.stop_phase (skip in callback).
    # "generation" and "save_png" are single-mode names; worker uses generation_N / save_N.
    _EXPLICIT_EXACT = {
        "total_wall_clock",
        "python_startup",
        "edit",
        "pipeline_load",

        "generation",
        "save_png",
        "dit_load",
        "vae_load",
    }
    # Phases that are explicitly reported via report.stop_phase (skip in callback).
    # "text_encoder_unload" is NOT in this list — it's profiled inside
    # MageFlowPipeline.generate() but never explicitly reported in single/edit
    # mode, so it must go through the callback. Worker handles it explicitly.
    _EXPLICIT_PREFIXES = (
        "pipeline_load",
        "text_encode_",
        "generation_",
        "save_",
    )
    def _on_phase_complete(name, elapsed, rss):
        if name in _EXPLICIT_EXACT or any(name.startswith(p) for p in _EXPLICIT_PREFIXES):
            return  # handled explicitly via stop_phase
        # text_encoder_load is lazy — weights load during text_encode,
        # so the phase time is always ~0.0s. Show "lazy loading" instead.
        loading_mode = "lazy loading" if name == "text_encoder_load" else None
        report.stop_phase(name, elapsed or 0.0, rss, loading_mode=loading_mode)
    prof.on_phase_complete = _on_phase_complete

    prof.start("total_wall_clock")
    prof.start("python_startup")

    # Load pipeline (auto-downloads and converts weights if needed)
    if verbose:
        print(f"  Importing Mage-Flow MLX pipeline from {args.model}...")
        prof.log(f"  Importing Mage-Flow MLX pipeline from {args.model}...")
        print(f"  Current working directory: {os.getcwd()}")
        prof.log(f"  Current working directory: {os.getcwd()}")
        print(f"  Python executable: {sys.executable}")
        prof.log(f"  Python executable: {sys.executable}")

    try:
        from mage_mlx import MageFlowPipeline
        if verbose:
            print("  Imported MageFlowPipeline successfully")
            prof.log("  Imported MageFlowPipeline successfully")
    except Exception as e:
        print(f"  ERROR importing MageFlowPipeline: {e}")
        prof.log(f"  ERROR importing MageFlowPipeline: {e}")
        traceback.print_exc()
        sys.exit(1)

    prof.stop("python_startup")
    if report:
        report.stop_phase(
            "python_startup",
            prof.get_elapsed("python_startup") or 0.0,
            prof.get_phase_rss("python_startup"),
        )

    # --- Worker mode ---
    if args.worker:
        from mage_mlx.worker import run_worker

        defaults = {
            "model": args.model,
            "steps": args.steps,
            "height": args.height,
            "width": args.width,
            "seed": args.seed,
            "guidance": args.guidance,
            "negative_prompt": args.negative_prompt,
            "quantize": args.quantize,
        }
        metadata, prompt_metadata = run_worker(
            args.worker,
            defaults,
            profiler=prof,
            metadata_enabled=args.metadata,
            report=report,
        )
        prof.stop("total_wall_clock")
        if report:
            report.stop_phase("total_wall_clock", prof.get_elapsed("total_wall_clock") or 0.0)
            # Prefer run peak from samples; fall back to metadata peak if present
            peak_ram = prof.get_peak_rss_gib()
            if peak_ram is None and metadata:
                peak_ram = metadata.get("peak_memory_gib")
            if metadata is not None and peak_ram is not None:
                metadata["peak_memory_gib"] = peak_ram
            report.print_summary(
                total_time=prof.get_elapsed("total_wall_clock") or 0.0,
                peak_ram=peak_ram or 0.0,
                show_text_encode=True,
            )
            report.print_run_metadata(metadata or {})
            if args.metadata:
                base_path = os.path.splitext(args.worker)[0]
                print(f"\n  Metadata saved to {_C.GREEN}{base_path}.json{_C.RESET} and {_C.GREEN}{base_path}.md{_C.RESET}")
        return

    # --- Edit worker mode ---
    if args.worker and args.edit:
        from mage_mlx.worker import run_edit_worker

        defaults = {
            "model": args.model,
            "steps": args.steps,
            "height": args.height,
            "width": args.width,
            "seed": args.seed,
            "guidance": args.guidance,
            "negative_prompt": args.negative_prompt,
            "quantize": args.quantize,
            "renormalization": args.renormalization,
        }
        metadata, prompt_metadata = run_edit_worker(
            args.worker,
            defaults,
            profiler=prof,
            metadata_enabled=args.metadata,
            report=report,
        )
        prof.stop("total_wall_clock")
        if report:
            report.stop_phase("total_wall_clock", prof.get_elapsed("total_wall_clock") or 0.0)
            peak_ram = prof.get_peak_rss_gib()
            if peak_ram is None and metadata:
                peak_ram = metadata.get("peak_memory_gib")
            if metadata is not None and peak_ram is not None:
                metadata["peak_memory_gib"] = peak_ram
            report.print_summary(
                total_time=prof.get_elapsed("total_wall_clock") or 0.0,
                peak_ram=peak_ram or 0.0,
                show_text_encode=True,
            )
            report.print_run_metadata(metadata or {})
            if args.metadata:
                base_path = os.path.splitext(args.worker)[0]
                print(f"\n  Metadata saved to {_C.GREEN}{base_path}.json{_C.RESET} and {_C.GREEN}{base_path}.md{_C.RESET}")
        return


    # --- Benchmark cleanup mode ---
    if args.benchmark_cleanup:
        _benchmark_cleanup(args, prof)
        prof.stop("total_wall_clock")
        if args.metadata:
            prof.print_report()
        return

    # --- Edit mode ---
    if args.image is not None:
        if args.metadata:
            base_path = os.path.splitext(args.output)[0]
            prof.metadata_path = base_path
            prof.metadata = _collect_metadata(
                model=args.model,
                image_path=args.image,
                image_paths=None,
                generation_time=None,
                peak_memory_gib=None,
            )
        _run_edit(args, prof, report)

        prof.stop("total_wall_clock")
        # For edit: image_path = target image, image_paths = reference images
        if args.ref_images is None:
            ref_paths = [args.image]
        else:
            ref_paths = [p.strip() for p in args.ref_images.split(",") if p.strip()]
        metadata = _collect_metadata(
            model=args.model,
            image_path=args.image,
            image_paths=ref_paths if len(ref_paths) > 1 else None,
            generation_time=prof.get_elapsed("total_wall_clock"),
            peak_memory_gib=prof.get_peak_rss_gib(),
        )
        if report:
            report.stop_phase("total_wall_clock", prof.get_elapsed("total_wall_clock") or 0.0)
            report.add_prompt(
                index=1,
                prompt=args.prompt,
                resolution=f"{args.width}x{args.height}",
                steps=args.steps,
                quantize=args.quantize,
                seed=args.seed,
                generation_time=prof.get_elapsed("edit") or 0.0,
                peak_rss_gib=prof.get_max_phase_rss(
                    "edit",
                    "text_encode",
                    "text_encoder_unload",
                    "dit_step_",
                    "edit_step_",
                    "vae_decode",
                ) or prof.get_peak_rss_gib(),
                saved_file=args.output,
            )
            report.print_summary(
                total_time=prof.get_elapsed("total_wall_clock") or 0.0,
                peak_ram=prof.get_peak_rss_gib() or 0.0,
            )
            report.print_run_metadata(metadata)
        if args.metadata:
            base_path = os.path.splitext(args.output)[0]
            prof.metadata["generation_time_seconds"] = prof.get_elapsed("total_wall_clock")
            prof.metadata["peak_memory_gib"] = prof.get_peak_rss_gib()
            prof.overview = [
                {
                    "index": 1,
                    "time": prof.get_elapsed("edit") or 0.0,
                    "peak_rss_gib": prof.get_max_phase_rss(
                        "edit",
                        "text_encode",
                        "text_encoder_unload",
                        "dit_step_",
                        "edit_step_",
                        "vae_decode",
                    ) or prof.get_peak_rss_gib(),
                    "resolution": f"{args.width}x{args.height}",
                    "steps": args.steps,
                    "seed": args.seed,
                    "file": args.output,
                }
            ]
            total_time = prof.get_elapsed("total_wall_clock") or 0.0
            gen_time = prof.get_elapsed("edit") or 0.0
            prof.summary = {
                "total_time": total_time,
                "peak_ram": prof.get_peak_rss_gib() or 0.0,
                "prompts_count": 1,
                "overhead": total_time - gen_time,
            }
            prof.save_metadata(base_path, prof.metadata, overview=prof.overview, summary=prof.summary)
            print(f"  Metadata saved to {_C.GREEN}{base_path}.json{_C.RESET} and {_C.GREEN}{base_path}.md{_C.RESET}")
        return


    # --- Set up incremental save path (so files are written after every phase) ---
    if args.metadata:
        base_path = os.path.splitext(args.output)[0]
        prof.metadata_path = base_path
        prof.metadata = _collect_metadata(
            model=args.model,
            image_path=None,
            image_paths=None,
            generation_time=None,
            peak_memory_gib=None,
        )

    # --- Phase: Load text encoder + tokenizer (DiT + VAE deferred) ---
    # We load only the text encoder first, encode the prompt, unload it,
    # THEN load DiT + VAE. This reduces peak RAM from ~15.4 GiB to ~7.9 GiB
    # on cache miss, because Qwen (~7.5 GiB) is never resident alongside
    # DiT + VAE (~7.9 GiB) simultaneously.
    prof.start("pipeline_load")

    try:
        pipeline = MageFlowPipeline.from_pretrained_text_encoder(
            model_dir=args.model,
            num_steps=args.steps,
            profiler=prof,
        )
    except Exception as e:
        print(f"  ERROR loading pipeline: {e}")
        traceback.print_exc()
        sys.exit(1)
    prof.stop("pipeline_load")
    if report:
        report.stop_phase(
            "pipeline_load",
            prof.get_elapsed("pipeline_load") or 0.0,
            prof.get_phase_rss("pipeline_load"),
            grey_separator=True,
        )

    # --- Phase: Embedding cache (single-prompt mode) ---
    # Create embedding cache so repeated prompts skip Qwen encoding entirely.
    # On cache hit, peak RAM drops from ~15.4 GiB to ~8.0 GiB.
    from mage_mlx.embedding_cache import EmbeddingCache
    from mage_mlx.loader import resolve_text_encoder_path

    embedding_cache = EmbeddingCache(model_dir=args.model)
    te_path = resolve_text_encoder_path(args.model)

    # --- Phase: Text encoding (Qwen) ---
    # Print prompt header BEFORE encoding starts (so it appears right before metadata)
    if report and report.verbose:
        report.prompt_header(1, 1)
        report.add_metadata("generation", "prompt", args.prompt)
        report.add_metadata("generation", "resolution", f"{args.width}x{args.height}")
        report.add_metadata("generation", "steps", str(args.steps))
        report.add_metadata("generation", "quantize", str(args.quantize))
        report.add_metadata("generation", "seed", str(args.seed))
        print()  # Empty line after metadata, before generation steps

    prof.start("generation")

    # Check embedding cache before encoding — on cache hit, Qwen is
    # never materialized in Metal memory, saving ~7.5 GiB peak RAM.
    cached_pos = None
    pos_key = None
    if embedding_cache is not None:
        pos_key = embedding_cache.make_key(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            te_path=te_path,
        )
        cached_pos = embedding_cache.get(pos_key)

    cached_neg = None
    neg_key = None
    if args.guidance > 1.0 and embedding_cache is not None:
        neg_key = embedding_cache.make_key(
            prompt=args.negative_prompt,
            negative_prompt=" ",
            te_path=te_path,
        )
        cached_neg = embedding_cache.get(neg_key)

    # Always start text_encode phase (even on cache hit, to record it in md/json)
    prof.start("text_encode")
    if cached_pos is not None:
        txt_embeds = cached_pos
        print("  Cache HIT — skipping Qwen encode")
    else:
        txt_embeds, _ = pipeline.text_encoder.encode_text_to_image(
            prompts=[args.prompt],
            tokenizer=pipeline.tokenizer,
            max_sequence_length=2048,
        )
        mx.eval(txt_embeds)
        if pos_key is not None:
            embedding_cache.put(pos_key, txt_embeds)

    neg_txt_embeds = None
    if args.guidance > 1.0:
        if cached_neg is not None:
            neg_txt_embeds = cached_neg
        else:
            neg_txt_embeds, _ = pipeline.text_encoder.encode_text_to_image(
                prompts=[args.negative_prompt],
                tokenizer=pipeline.tokenizer,
                max_sequence_length=2048,
            )
            mx.eval(neg_txt_embeds)
            if neg_key is not None:
                embedding_cache.put(neg_key, neg_txt_embeds)
    prof.stop("text_encode")
    cache_label = "HIT" if cached_pos is not None else "MISS"
    prof.set_metadata("text_encode", "cache", cache_label)

    # Unload Qwen — it's only needed for prompt encoding
    prof.start("text_encoder_unload")
    pipeline.text_encoder.unload()
    gc.collect()
    mx.clear_cache()
    prof.stop("text_encoder_unload")

    # --- Phase: Load DiT + VAE (after Qwen is unloaded) ---
    # load_dit_vae() handles profiler.start/stop for dit_load and vae_load internally
    pipeline.load_dit_vae(
        model_dir=args.model,
        quantize=args.quantize,
        profiler=prof,
    )
    if report:
        report.stop_phase("dit_load", prof.get_elapsed("dit_load") or 0.0, prof.get_phase_rss("dit_load"))
        report.stop_phase("vae_load", prof.get_elapsed("vae_load") or 0.0, prof.get_phase_rss("vae_load"))

    # --- Phase: Generation (DiT steps + VAE decode) ---
    image = pipeline._generate_from_embeds(
        txt_embeds=txt_embeds,
        neg_txt_embeds=neg_txt_embeds,
        height=args.height,
        width=args.width,
        seed=args.seed,
        guidance_scale=args.guidance,
        profiler=prof,
    )
    prof.stop("generation")
    if report:
        gen_rss = prof.get_max_phase_rss(
            "generation",
            "text_encode",
            "text_encoder_unload",
            "dit_load",
            "vae_load",
            "dit_step_",
            "vae_decode",
        ) or prof.get_phase_rss("generation")
        report.stop_phase(
            "generation",
            prof.get_elapsed("generation") or 0.0,
            gen_rss,
        )

    # Also set metadata on profiler for JSON/markdown output
    if args.metadata:
        prof.set_metadata("generation", "prompt", args.prompt)
        prof.set_metadata("generation", "resolution", f"{args.width}x{args.height}")
        prof.set_metadata("generation", "steps", str(args.steps))
        prof.set_metadata("generation", "quantize", str(args.quantize))
        prof.set_metadata("generation", "seed", str(args.seed))

    # --- Phase: Save ---
    prof.start("save_png")
    image.save(args.output)
    prof.stop("save_png")
    if report:
        report.stop_phase(
            "save_png",
            prof.get_elapsed("save_png") or 0.0,
            prof.get_phase_rss("save_png"),
            saved_file=args.output,
        )

    prof.stop("total_wall_clock")
    if report:
        report.stop_phase("total_wall_clock", prof.get_elapsed("total_wall_clock") or 0.0)
        # Add to LiveReport for per-prompt summary table
        report.add_prompt(
            index=1,
            prompt=args.prompt,
            resolution=f"{args.width}x{args.height}",
            steps=args.steps,
            quantize=args.quantize,
            seed=args.seed,
            generation_time=prof.get_elapsed("generation") or 0.0,
            peak_rss_gib=prof.get_max_phase_rss(
                "generation",
                "text_encode",
                "text_encoder_unload",
                "dit_load",
                "vae_load",
                "dit_step_",
                "vae_decode",
            ) or prof.get_peak_rss_gib(),
            saved_file=args.output,
        )

    # --- Report + Metadata ---
    metadata = _collect_metadata(
        model=args.model,
        image_path=None,
        image_paths=None,
        generation_time=prof.get_elapsed("total_wall_clock"),
        peak_memory_gib=prof.get_peak_rss_gib(),
        )
    report.print_summary(
        total_time=prof.get_elapsed("total_wall_clock") or 0.0,
        peak_ram=prof.get_peak_rss_gib() or 0.0,
    )
    report.print_run_metadata(metadata)
    if args.metadata:
        base_path = os.path.splitext(args.output)[0]
        prof.metadata["generation_time_seconds"] = prof.get_elapsed("total_wall_clock")
        prof.metadata["peak_memory_gib"] = prof.get_peak_rss_gib()
        prof.overview = [
            {
                "index": 1,
                "time": prof.get_elapsed("generation") or 0.0,
                "peak_rss_gib": prof.get_max_phase_rss(
                    "generation",
                    "text_encode",
                    "text_encoder_unload",
                    "dit_load",
                    "vae_load",
                    "dit_step_",
                    "vae_decode",
                ) or prof.get_peak_rss_gib(),
                "resolution": f"{args.width}x{args.height}",
                "steps": args.steps,
                "seed": args.seed,
                "file": args.output,
            }
        ]
        total_time = prof.get_elapsed("total_wall_clock") or 0.0
        gen_time = prof.get_elapsed("generation") or 0.0
        prof.summary = {
            "total_time": total_time,
            "peak_ram": prof.get_peak_rss_gib() or 0.0,
            "prompts_count": 1,
            "overhead": total_time - gen_time,
        }
        prof.save_metadata(base_path, prof.metadata, overview=prof.overview, summary=prof.summary)
        print(f"  Metadata saved to {_C.GREEN}{base_path}.json{_C.RESET} and {_C.GREEN}{base_path}.md{_C.RESET}")


def _run_edit(args, prof, report=None):

    """Run the image editing pipeline using mflux's MageFlowEdit."""
    from mage_mlx.mflux_src.mflux.models.mage_flow.variants.edit.mage_flow_edit import (
        MageFlowEdit,
    )

    if args.ref_images is None:
        # Use the target image as its own reference (mflux --image-paths semantics)
        ref_paths = [args.image]
    else:
        ref_paths = [p.strip() for p in args.ref_images.split(",") if p.strip()]
        if not ref_paths:
            print("Error: at least one reference image is required")
            sys.exit(1)

    try:
        prof.start("pipeline_load")
        edit = MageFlowEdit(
            quantize=args.quantize,
            model_path=args.model,
            load_dit_vae=False,
        )
        prof.stop("pipeline_load")
    except Exception as e:
        print(f"  ERROR loading edit pipeline: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Report pipeline_load phase to LiveReport
    if report:
        report.stop_phase(
            "pipeline_load",
            prof.get_elapsed("pipeline_load") or 0.0,
            prof.get_phase_rss("pipeline_load"),
            grey_separator=True,
        )

    # Print prompt header (magenta bold) before edit metadata
    if report and report.verbose:
        report.prompt_header(1, 1)
        report.add_metadata("edit", "prompt", args.prompt)
        report.add_metadata("edit", "resolution", f"{args.width}x{args.height}")
        report.add_metadata("edit", "steps", str(args.steps))
        report.add_metadata("edit", "quantize", str(args.quantize))
        report.add_metadata("edit", "seed", str(args.seed))
        print()  # Empty line after metadata, before generation steps

    prof.start("edit")
    generated = edit.generate_image(
        seed=args.seed,
        prompt=args.prompt,
        image_paths=[args.image] + ref_paths,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance=args.guidance,
        negative_prompt=args.negative_prompt,
        renormalization=args.renormalization,
        profiler=prof,
    )
    prof.stop("edit")
    if report:
        # Report dit_load and vae_load phases (started/stopped inside
        # MageFlowEdit.load_dit_vae(), which is called from generate_image).
        # These are in _EXPLICIT_EXACT so the callback skips them.
        report.stop_phase(
            "dit_load",
            prof.get_elapsed("dit_load") or 0.0,
            prof.get_phase_rss("dit_load"),
        )
        report.stop_phase(
            "vae_load",
            prof.get_elapsed("vae_load") or 0.0,
            prof.get_phase_rss("vae_load"),
        )
        edit_rss = prof.get_max_phase_rss(
            "edit",
            "text_encode",
            "text_encoder_unload",
            "dit_load",
            "vae_load",
            "dit_step_",
            "edit_step_",
            "vae_decode",
        ) or prof.get_phase_rss("edit")
        report.stop_phase(
            "edit",
            prof.get_elapsed("edit") or 0.0,
            edit_rss,
        )

    # Also set metadata on profiler for JSON/markdown output
    if prof.enabled:
        prof.set_metadata("edit", "prompt", args.prompt)
        prof.set_metadata("edit", "resolution", f"{args.width}x{args.height}")
        prof.set_metadata("edit", "steps", str(args.steps))
        prof.set_metadata("edit", "quantize", str(args.quantize))
        prof.set_metadata("edit", "seed", str(args.seed))

    # Extract PIL image from GeneratedImage
    image = generated.image

    # Save
    prof.start("save_png")
    image.save(args.output)
    prof.stop("save_png")
    if report:
        report.stop_phase(
            "save_png",
            prof.get_elapsed("save_png") or 0.0,
            prof.get_phase_rss("save_png"),
            saved_file=args.output,
        )


def _benchmark_cleanup(args, prof):
    """Benchmark all 4 Qwen cleanup strategies.

    Tests:
    - unload only
    - unload + gc.collect()
    - unload + mx.clear_cache()
    - all three (current default)
    """
    import gc
    import time

    import mlx.core as mx

    from mage_mlx import MageFlowPipeline

    strategies = [
        ("unload_only", lambda te: te.unload()),
        ("unload+gc", lambda te: (te.unload(), gc.collect())),
        ("unload+cache", lambda te: (te.unload(), mx.clear_cache())),
        ("all_three", lambda te: (te.unload(), gc.collect(), mx.clear_cache())),
    ]

    print("\n" + "=" * 60)
    print("  Cleanup Strategy Benchmark")
    print("=" * 60)
    print(f"  Prompt: {args.prompt}")
    print(f"  Steps: {args.steps}, Size: {args.width}x{args.height}")
    print()

    results = []
    for name, strategy in strategies:
        print(f"  Testing: {name}")

        # Fresh pipeline for each strategy
        prof.start(f"cleanup_{name}_load")
        pipeline = MageFlowPipeline.from_pretrained(
            model_dir=args.model,
            num_steps=args.steps,
            quantize=args.quantize,
            profiler=prof,
        )
        prof.stop(f"cleanup_{name}_load")

        # Generate image (this loads Qwen, encodes, then cleans up)
        prof.start(f"cleanup_{name}_gen")
        image = pipeline.generate(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            seed=args.seed,
            guidance_scale=args.guidance,
            negative_prompt=args.negative_prompt,
            profiler=prof,
            cleanup_strategy=name,
        )
        prof.stop(f"cleanup_{name}_gen")

        # Measure cleanup time
        # Re-load Qwen to measure cleanup time
        # Actually, the cleanup happens inside generate(), so we need to
        # measure it separately. Let's measure by re-encoding and timing
        # the cleanup call.
        prof.start(f"cleanup_{name}_time")
        # The cleanup already happened inside generate(). Let's just record
        # the total time for this strategy.
        prof.stop(f"cleanup_{name}_time")

        image.save(f"/tmp/cleanup_bench_{name}.png")
        print(f"    Saved to /tmp/cleanup_bench_{name}.png")
        print()

        results.append(name)

    # Print summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Tested {len(results)} strategies:")
    for name in results:
        print(f"    - {name}")
    print()
    print("  Use --metadata to see detailed phase timings for each strategy.")


if __name__ == "__main__":
    main()

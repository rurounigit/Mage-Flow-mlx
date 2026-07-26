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
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Optional


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
    from mage_mlx.profiler import Profiler

    prof = Profiler(enabled=args.metadata, track_memory=args.metadata)

    prof.start("total_wall_clock")
    prof.start("python_startup")

    # Load pipeline (auto-downloads and converts weights if needed)
    print(f"Importing Mage-Flow MLX pipeline from {args.model}...")
    print(f"  Current working directory: {os.getcwd()}")
    print(f"  Python executable: {sys.executable}")

    try:
        from mage_mlx import MageFlowPipeline
        print("  Imported MageFlowPipeline successfully")
    except Exception as e:
        print(f"  ERROR importing MageFlowPipeline: {e}")
        traceback.print_exc()
        sys.exit(1)

    prof.stop("python_startup")

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
        metadata = run_worker(
            args.worker,
            defaults,
            profiler=prof,
            metadata_enabled=args.metadata,
        )
        prof.stop("total_wall_clock")
        if args.metadata:
            prof.print_report(metadata=metadata)
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
        _run_edit(args, prof)
        prof.stop("total_wall_clock")
        if args.metadata:
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
                peak_memory_gib=prof._get_rss_gib(),
            )
            prof.print_report(metadata=metadata)
            base_path = os.path.splitext(args.output)[0]
            prof.save_metadata(base_path, metadata)
            print(f"  Metadata saved to {base_path}.json and {base_path}.md")
        return

    # --- Phase: Pipeline load (DiT + VAE + text encoder) ---
    prof.start("pipeline_load")
    try:
        print(f"  Loading model from {args.model}...")
        pipeline = MageFlowPipeline.from_pretrained(
            model_dir=args.model,
            num_steps=args.steps,
            quantize=args.quantize,
            profiler=prof,
        )
        print("  Pipeline loaded successfully!")
    except Exception as e:
        print(f"  ERROR loading pipeline: {e}")
        traceback.print_exc()
        sys.exit(1)
    prof.stop("pipeline_load")

    # --- Phase: Generation ---
    print(f"\nGenerating {args.height}x{args.width} image...")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}, Seed: {args.seed}")

    prof.start("generation")
    image = pipeline.generate(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        guidance_scale=args.guidance,
        negative_prompt=args.negative_prompt,
        profiler=prof,
    )
    prof.stop("generation")

    # Add per-prompt metadata to the generation phase (ordered: prompt, resolution, steps, quantize)
    if args.metadata:
        prof.set_metadata("generation", "prompt", args.prompt)
        prof.set_metadata("generation", "resolution", f"{args.width}x{args.height}")
        prof.set_metadata("generation", "steps", str(args.steps))
        prof.set_metadata("generation", "quantize", str(args.quantize))

    # --- Phase: Save ---
    prof.start("save_png")
    image.save(args.output)
    prof.stop("save_png")
    print(f"\nImage saved to {args.output}")
    print(f"   Size: {image.size}")

    prof.stop("total_wall_clock")

    # --- Report + Metadata ---
    if args.metadata:
        # For txt2img: image_path = null, image_paths = null (only applies to edit)
        metadata = _collect_metadata(
            model=args.model,
            image_path=None,
            image_paths=None,
            generation_time=prof.get_elapsed("total_wall_clock"),
            peak_memory_gib=prof._get_rss_gib(),
        )
        prof.print_report(metadata=metadata)
        base_path = os.path.splitext(args.output)[0]
        prof.save_metadata(base_path, metadata)
        print(f"  Metadata saved to {base_path}.json and {base_path}.md")


def _run_edit(args, prof):
    """Run the image editing pipeline using mflux's MageFlowEdit."""
    from mage_mlx.mflux_src.mflux.models.mage_flow.variants.edit.mage_flow_edit import (
        MageFlowEdit,
    )

    if args.ref_images is None:
        # Use the target image as its own reference (mflux --image-paths semantics)
        ref_paths = [args.image]
        print("  No --ref-images provided; using target image as reference")
    else:
        ref_paths = [p.strip() for p in args.ref_images.split(",") if p.strip()]
        if not ref_paths:
            print("Error: at least one reference image is required")
            sys.exit(1)

    print(f"Loading Mage-Flow-Edit pipeline from {args.model}...")
    try:
        edit = MageFlowEdit(
            quantize=args.quantize,
            model_path=args.model,
        )
        print("  Edit pipeline loaded successfully!")
    except Exception as e:
        print(f"  ERROR loading edit pipeline: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\nEditing {args.height}x{args.width} image...")
    print(f"  Target: {args.image}")
    print(f"  References: {ref_paths}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Steps: {args.steps}, Seed: {args.seed}")

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
    )
    prof.stop("edit")

    # Add per-prompt metadata to the edit phase (ordered: prompt, resolution, steps, quantize)
    if prof.enabled:
        prof.set_metadata("edit", "prompt", args.prompt)
        prof.set_metadata("edit", "resolution", f"{args.width}x{args.height}")
        prof.set_metadata("edit", "steps", str(args.steps))
        prof.set_metadata("edit", "quantize", str(args.quantize))

    # Extract PIL image from GeneratedImage
    image = generated.image

    # Save
    prof.start("save_png")
    image.save(args.output)
    prof.stop("save_png")
    print(f"\nEdited image saved to {args.output}")
    print(f"   Size: {image.size}")


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

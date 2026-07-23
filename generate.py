"""Generate images with Mage-Flow MLX.

CLI tool for text-to-image generation using the MLX port of Mage-Flow.

Usage:
    python generate.py --prompt "A futuristic cityscape at sunset"
    python generate.py --prompt "..." --steps 4 --height 1024 --width 1024 --seed 42
    python generate.py --prompt "..." --output output.png
    python generate.py --prompt "..." --model microsoft/Mage-Flow-Turbo
    python generate.py --prompt "..." --quantize 4
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with Mage-Flow MLX (Apple Silicon)"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--model", type=str, default="models/mage_flow_mlx",
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
    args = parser.parse_args()

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

    # Load pipeline (auto-downloads and converts weights if needed)
    print(f"Loading Mage-Flow MLX pipeline from {args.model}...")
    print(f"  Current working directory: {os.getcwd()}")
    print(f"  Python executable: {sys.executable}")

    try:
        from mage_mlx import MageFlowPipeline
        print("  Imported MageFlowPipeline successfully")
    except Exception as e:
        print(f"  ERROR importing MageFlowPipeline: {e}")
        traceback.print_exc()
        sys.exit(1)

    import time

    try:
        print(f"  Loading model from {args.model}...")
        pipeline = MageFlowPipeline.from_pretrained(
            model_dir=args.model,
            num_steps=args.steps,
            quantize=args.quantize,
        )
        print("  Pipeline loaded successfully!")
    except Exception as e:
        print(f"  ERROR loading pipeline: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Generate
    print(f"\nGenerating {args.height}x{args.width} image...")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}, Seed: {args.seed}")

    t_gen_start = time.time()
    image = pipeline.generate(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        guidance_scale=args.guidance,
        negative_prompt=args.negative_prompt,
    )
    t_gen_elapsed = time.time() - t_gen_start
    print(f"⏱️ Total generation time: {t_gen_elapsed:.2f}s")

    # Save
    image.save(args.output)
    print(f"\n✅ Image saved to {args.output}")
    print(f"   Size: {image.size}")


if __name__ == "__main__":
    main()

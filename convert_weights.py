"""Convert Mage-Flow PyTorch weights to MLX format with 4-bit quantization (Memory-Efficient Streaming).

This script:
1. Downloads weights from HuggingFace (microsoft/Mage-Flow-Turbo)
2. Converts PyTorch safetensors → MLX format tensor-by-tensor (low RAM usage)
3. Quantizes DiT transformer + text encoder to 4-bit (group_size=64)
4. Saves as MLX safetensors in models/mage_flow_mlx/

Usage:
    python convert_weights.py [--repo microsoft/Mage-Flow-Turbo] [--output models/mage_flow_mlx]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def convert_conv2d(arr: np.ndarray) -> np.ndarray:
    """Convert Conv2d weights from PyTorch NCHW to MLX NHWC.

    PyTorch: [Out, In, H, W] → MLX: [Out, H, W, In]
    """
    if arr.ndim == 4:
        return arr.transpose(0, 2, 3, 1)
    return arr


def quantize_single_tensor(key: str, arr: mx.array, quantize: bool = True, bits: int = 4, group_size: int = 64) -> dict[str, mx.array]:
    """Quantize a single tensor if it's 2D Linear weight."""
    if quantize and arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        q_w, scales, biases = mx.quantize(arr, group_size=group_size, bits=bits)
        # Fix MLX key naming convention for QuantizedLinear:
        # layer.weight -> layer.weight, layer.scales, layer.biases (NOT layer.weight.scales)
        base_key = key[:-7] if key.endswith(".weight") else key
        return {
            f"{base_key}.weight": q_w,
            f"{base_key}.scales": scales,
            f"{base_key}.biases": biases,
        }
    return {key: arr}


def process_and_convert_file(
    safetensors_path: str,
    key_mapper_fn,
    quantize: bool = True,
    bits: int = 4,
    group_size: int = 64,
) -> dict[str, mx.array]:
    """Streams and converts safetensors one tensor at a time to prevent RAM OOM."""
    converted = {}
    with safe_open(safetensors_path, framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            # Skip unused keys
            if key.startswith("_") or "pos_embed" in key or "freqs" in key:
                continue

            # Load single tensor
            tensor = f.get_tensor(key)
            if tensor.dtype == torch.bfloat16 or tensor.dtype == torch.float16:
                tensor = tensor.to(torch.float32)

            np_arr = tensor.numpy()
            del tensor

            # Conv2d transpose if 4D
            if np_arr.ndim == 4:
                np_arr = convert_conv2d(np_arr)

            mx_arr = mx.array(np_arr)
            del np_arr

            # Map key name
            mapped_key = key_mapper_fn(key)
            if mapped_key is None:
                continue

            # Quantize tensor if applicable
            q_dict = quantize_single_tensor(mapped_key, mx_arr, quantize=quantize, bits=bits, group_size=group_size)
            for k, v in q_dict.items():
                mx.eval(v)
                converted[k] = v

            gc.collect()

    return converted


def map_dit_key(key: str) -> str | None:
    new_key = key.replace(".img_mlp.net.0.proj.", ".img_mlp.fc1.")
    new_key = new_key.replace(".img_mlp.net.2.", ".img_mlp.fc2.")
    new_key = new_key.replace(".txt_mlp.net.0.proj.", ".txt_mlp.fc1.")
    new_key = new_key.replace(".txt_mlp.net.2.", ".txt_mlp.fc2.")
    return new_key


def map_vae_key(key: str) -> str | None:
    if "y_embedder.encoder." in key or "y_embedder.bottleneck." in key:
        return None
    if key.startswith("student.dconv_encoder."):
        return key[len("student.dconv_encoder."):]
    if key.startswith("pipeline."):
        return key[len("pipeline."):]
    return key


def save_mlx_safetensors(weights: dict[str, mx.array], path: str) -> None:
    """Save MLX weights as safetensors."""
    from safetensors.numpy import save_file

    np_weights = {}
    for key, arr in weights.items():
        np_weights[key] = np.array(arr)

    save_file(np_weights, path)
    print(f"  Saved: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def convert_text_encoder_weights(pt_weights: dict, quantize: bool = True, bits: int = 4) -> dict:
    from mlx.utils import tree_flatten, tree_unflatten

    weights = tree_unflatten(list(pt_weights.items()))
    weights.pop("vision_tower", None)
    weights = dict(tree_flatten(weights))

    sanitized = {}
    for key, value in weights.items():
        if not key.startswith("language_model."):
            key = "language_model." + key
        mx_arr = mx.array(value)
        q_dict = quantize_single_tensor(key, mx_arr, quantize=quantize, bits=bits)
        for k, v in q_dict.items():
            mx.eval(v)
            sanitized[k] = v
        gc.collect()

    return sanitized


def main():
    parser = argparse.ArgumentParser(description="Convert Mage-Flow PyTorch weights to MLX (Low Memory)")
    parser.add_argument("--repo", default="microsoft/Mage-Flow-Turbo", help="HuggingFace repo ID")
    parser.add_argument("--output", default="models/mage_flow_mlx", help="Output directory")
    parser.add_argument("--quantize", action="store_true", default=True, help="Quantize to 4-bit")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (4 or 8)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Downloading weights from {args.repo}...")
    repo_dir = snapshot_download(args.repo)

    # 1. DiT weights
    print("\nConverting DiT weights (low RAM streaming)...")
    dit_path = os.path.join(repo_dir, "transformer", "diffusion_pytorch_model.safetensors")
    mlx_dit = process_and_convert_file(
        dit_path,
        key_mapper_fn=map_dit_key,
        quantize=args.quantize,
        bits=args.bits,
    )
    save_mlx_safetensors(mlx_dit, os.path.join(args.output, "transformer.safetensors"))
    del mlx_dit
    gc.collect()

    # Save DiT config
    with open(os.path.join(repo_dir, "transformer", "config.json")) as f:
        dit_config = json.load(f)
    with open(os.path.join(args.output, "transformer_config.json"), "w") as f:
        json.dump(dit_config, f, indent=2)

    # 2. VAE weights
    print("\nConverting VAE weights...")
    vae_path = os.path.join(repo_dir, "vae", "diffusion_pytorch_model.safetensors")
    mlx_vae = process_and_convert_file(
        vae_path,
        key_mapper_fn=map_vae_key,
        quantize=False,  # VAE kept in FP32/FP16
    )
    save_mlx_safetensors(mlx_vae, os.path.join(args.output, "vae.safetensors"))
    del mlx_vae
    gc.collect()

    # 3. Text Encoder weights
    print("\nConverting Text Encoder weights (streaming)...")
    te_dir = os.path.join(repo_dir, "text_encoder")
    mlx_te = {}
    for shard in sorted(os.listdir(te_dir)):
        if shard.endswith(".safetensors"):
            shard_path = os.path.join(te_dir, shard)
            with safe_open(shard_path, framework="pt") as f:
                for key in list(f.keys()):
                    if "vision_tower" in key:
                        continue
                    t = f.get_tensor(key)
                    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
                        t = t.to(torch.float32)
                    np_arr = t.numpy()
                    del t

                    mapped_key = key if key.startswith("language_model.") else f"language_model.{key}"
                    mx_arr = mx.array(np_arr)
                    del np_arr

                    q_dict = quantize_single_tensor(mapped_key, mx_arr, quantize=args.quantize, bits=args.bits)
                    for k, v in q_dict.items():
                        mx.eval(v)
                        mlx_te[k] = v

                    gc.collect()

    save_mlx_safetensors(mlx_te, os.path.join(args.output, "text_encoder.safetensors"))
    del mlx_te
    gc.collect()

    print("\n✅ Conversion complete!")
    print(f"  Output directory: {args.output}")
    total_mb = sum(os.path.getsize(os.path.join(args.output, f)) for f in os.listdir(args.output) if f.endswith(".safetensors")) / 1e6
    print(f"  Total model size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()

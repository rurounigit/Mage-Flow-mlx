"""Convert Mage-Flow PyTorch weights to MLX format with quantization.

This script:
1. Downloads weights from HuggingFace (microsoft/Mage-Flow-Turbo)
2. Converts PyTorch safetensors → MLX format tensor-by-tensor (low RAM usage)
3. Quantizes DiT transformer + text encoder (8-bit by default, group_size=64)
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
        # ``transpose`` returns a strided view. MLX/safetensors conversion must
        # receive contiguous storage or the tensor can retain PyTorch's raw
        # OIHW memory order while merely advertising an OHWI shape.
        return np.ascontiguousarray(arr.transpose(0, 2, 3, 1))
    return arr


def quantize_single_tensor(key: str, arr: mx.array, quantize: bool = True, bits: int = 4, group_size: int = 64) -> dict[str, mx.array]:
    """Quantize a single tensor if it's 2D Linear weight."""
    if quantize and arr.ndim == 2:
        q_w, scales, biases = mx.quantize(arr, group_size=group_size, bits=bits)
        base_key = key[:-7] if key.endswith(".weight") else key
        return {
            f"{base_key}.weight": q_w,
            f"{base_key}.scales": scales,
            f"{base_key}.biases": biases,
        }
    return {key: arr}


def process_and_convert_file(
    safetensors_path: str,
    out_path: str,
    key_mapper_fn,
    quantize: bool = True,
    bits: int = 4,
    group_size: int = 64,
) -> None:
    """Streams and converts safetensors directly to disk as NumPy dict to minimize RAM peak."""
    from safetensors.numpy import save_file

    np_converted = {}
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

            mapped_key = key_mapper_fn(key)
            if mapped_key is None:
                del np_arr
                continue

            mx_arr = mx.array(np_arr)
            del np_arr

            # Quantize tensor if applicable
            q_dict = quantize_single_tensor(mapped_key, mx_arr, quantize=quantize, bits=bits, group_size=group_size)
            for k, v in q_dict.items():
                mx.eval(v)
                np_converted[k] = np.array(v)

            del mx_arr
            gc.collect()

    save_file(np_converted, out_path)
    print(f"  Saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    del np_converted
    gc.collect()


def map_dit_key(key: str) -> str | None:
    new_key = key.replace(".img_mlp.net.0.proj.", ".img_mlp.fc1.")
    new_key = new_key.replace(".img_mlp.net.2.", ".img_mlp.fc2.")
    new_key = new_key.replace(".txt_mlp.net.0.proj.", ".txt_mlp.fc1.")
    new_key = new_key.replace(".txt_mlp.net.2.", ".txt_mlp.fc2.")
    # Map diffusers img_mod.1 / txt_mod.1 to direct img_mod / txt_mod Linear
    new_key = new_key.replace(".img_mod.1.", ".img_mod.")
    new_key = new_key.replace(".txt_mod.1.", ".txt_mod.")
    return new_key


def map_text_encoder_key(key: str) -> str | None:
    """Map Hugging Face Qwen3-VL text keys to mlx-lm's module tree."""
    if "vision_tower" in key or "visual" in key:
        return None
    if key.startswith("model.language_model."):
        return "language_model.model." + key[len("model.language_model."):]
    if key.startswith("language_model."):
        return key
    return "language_model." + key


def map_vae_key(key: str) -> str | None:
    if "y_embedder.encoder." in key or "y_embedder.bottleneck." in key:
        return None

    # Map nn.Sequential sub-layers to .layers.N
    key = key.replace(".t_embedder.mlp.0.", ".t_embedder.mlp.layers.0.")
    key = key.replace(".t_embedder.mlp.2.", ".t_embedder.mlp.layers.2.")
    key = key.replace(".ca.1.", ".ca.layers.1.")
    key = key.replace(".adaLN_modulation.1.", ".adaLN_modulation.layers.1.")
    key = key.replace(".mlp.0.", ".mlp.layers.0.")
    key = key.replace(".mlp.2.", ".mlp.layers.2.")
    key = key.replace(".x_embedder.embedder.0.", ".x_embedder.embedder.")

    # Fix GroupNorm (Normalize) inside CoD Decoder ResnetBlock/AttnBlock
    key = key.replace(".decoder.block.0.norm1.", ".decoder.block.layers.0.norm1.norm.")
    key = key.replace(".decoder.block.0.norm2.", ".decoder.block.layers.0.norm2.norm.")
    key = key.replace(".decoder.block.1.norm.", ".decoder.block.layers.1.norm.norm.")
    key = key.replace(".decoder.block.2.norm1.", ".decoder.block.layers.2.norm1.norm.")
    key = key.replace(".decoder.block.2.norm2.", ".decoder.block.layers.2.norm2.norm.")
    key = key.replace(".decoder.block.3.norm.", ".decoder.block.layers.3.norm.norm.")
    key = key.replace(".decoder.block.4.norm1.", ".decoder.block.layers.4.norm1.norm.")
    key = key.replace(".decoder.block.4.norm2.", ".decoder.block.layers.4.norm2.norm.")

    # Remaining CoD Decoder Sequential blocks
    key = key.replace(".decoder.block.0.", ".decoder.block.layers.0.")
    key = key.replace(".decoder.block.1.", ".decoder.block.layers.1.")
    key = key.replace(".decoder.block.2.", ".decoder.block.layers.2.")
    key = key.replace(".decoder.block.3.", ".decoder.block.layers.3.")
    key = key.replace(".decoder.block.4.", ".decoder.block.layers.4.")


    # Fix LayerNorm2d (head_blocks) and Normalize (norm_out) wrapper keys
    key = key.replace(".head_blocks.0.norm1.", ".head_blocks.0.norm1.norm.")
    key = key.replace(".head_blocks.0.norm2.", ".head_blocks.0.norm2.norm.")
    key = key.replace(".head_blocks.1.norm1.", ".head_blocks.1.norm1.norm.")
    key = key.replace(".head_blocks.1.norm2.", ".head_blocks.1.norm2.norm.")
    key = key.replace(".norm_out.", ".norm_out.norm.")
    if key.startswith("student.dconv_encoder."):
        return "dconv_encoder." + key[len("student.dconv_encoder."):]
    if key.startswith("pipeline."):
        return "decoder_model." + key[len("pipeline."):]
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
    parser.add_argument("--quantize", action="store_true", default=True, help="Quantize model weights")
    parser.add_argument("--bits", type=int, default=8, choices=(4, 8), help="Quantization bits; 8 is recommended for image quality")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Downloading weights from {args.repo}...")
    repo_dir = snapshot_download(args.repo)

    # 1. DiT weights
    print("\nConverting DiT weights (low RAM streaming)...")
    dit_path = os.path.join(repo_dir, "transformer", "diffusion_pytorch_model.safetensors")
    process_and_convert_file(
        dit_path,
        os.path.join(args.output, "transformer.safetensors"),
        key_mapper_fn=map_dit_key,
        quantize=args.quantize,
        bits=args.bits,
    )

    # Save DiT config
    with open(os.path.join(repo_dir, "transformer", "config.json")) as f:
        dit_config = json.load(f)
    with open(os.path.join(args.output, "transformer_config.json"), "w") as f:
        json.dump(dit_config, f, indent=2)
    with open(os.path.join(args.output, "quantization_config.json"), "w") as f:
        json.dump(
            {
                "transformer_bits": args.bits,
                "text_encoder_bits": args.bits,
                "group_size": 64,
            },
            f,
            indent=2,
        )

    # 2. VAE weights
    print("\nConverting VAE weights...")
    vae_path = os.path.join(repo_dir, "vae", "diffusion_pytorch_model.safetensors")
    process_and_convert_file(
        vae_path,
        os.path.join(args.output, "vae.safetensors"),
        key_mapper_fn=map_vae_key,
        quantize=False,  # VAE kept in FP32/FP16
    )

    # 3. Text Encoder weights
    print("\nConverting Text Encoder weights (streaming shard by shard)...")
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

                    mapped_key = map_text_encoder_key(key)
                    if mapped_key is None:
                        continue
                    mx_arr = mx.array(np_arr)
                    del np_arr

                    q_dict = quantize_single_tensor(mapped_key, mx_arr, quantize=args.quantize, bits=args.bits)
                    for k, v in q_dict.items():
                        mx.eval(v)
                        # Save numpy array directly to dict
                        mlx_te[k] = np.array(v)

                    gc.collect()

    # Save safetensors directly from numpy dict
    from safetensors.numpy import save_file
    te_out_path = os.path.join(args.output, "text_encoder.safetensors")
    save_file(mlx_te, te_out_path)
    print(f"  Saved: {te_out_path} ({os.path.getsize(te_out_path) / 1e6:.1f} MB)")
    del mlx_te
    gc.collect()

    print("\n✅ Conversion complete!")
    print(f"  Output directory: {args.output}")
    total_mb = sum(os.path.getsize(os.path.join(args.output, f)) for f in os.listdir(args.output) if f.endswith(".safetensors")) / 1e6
    print(f"  Total model size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()

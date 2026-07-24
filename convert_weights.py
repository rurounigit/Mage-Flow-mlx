"""Convert Mage-Flow PyTorch weights to native BF16 MLX format.

This script:
1. Downloads weights from HuggingFace (microsoft/Mage-Flow-Turbo)
2. Converts PyTorch safetensors → MLX format tensor-by-tensor (low RAM usage)
3. Preserves the source BF16 precision for generation quality
4. Saves as MLX safetensors in models/microsoft_Mage-Flow-Turbo/

Usage:
    python convert_weights.py [--repo microsoft/Mage-Flow-Turbo] [--output models/microsoft_Mage-Flow-Turbo]
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    """Convert a CPU Torch tensor to MLX without losing BF16 precision."""
    if tensor.dtype == torch.bfloat16:
        storage = tensor.view(torch.uint16).numpy()
        return mx.array(storage).view(mx.bfloat16)
    return mx.array(tensor.numpy())


def process_and_convert_file(
    safetensors_path: str,
    out_path: str,
    key_mapper_fn,
) -> None:
    """Convert one source checkpoint while preserving BF16 tensors."""
    converted = {}
    with safe_open(safetensors_path, framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            # Skip unused keys
            if key.startswith("_") or "pos_embed" in key or "freqs" in key:
                continue

            # Load single tensor
            tensor = f.get_tensor(key)
            mapped_key = key_mapper_fn(key)
            if mapped_key is None:
                del tensor
                continue

            arr = torch_to_mlx(tensor)
            if arr.ndim == 4:
                arr = mx.transpose(arr, (0, 2, 3, 1))
            elif arr.ndim == 5:
                # PyTorch Conv3d [out, in, D, H, W] -> MLX channel-last
                # Conv3d [out, D, H, W, in].
                arr = mx.transpose(arr, (0, 2, 3, 4, 1))
            mx.eval(arr)
            converted[mapped_key] = arr
            del tensor, arr
            gc.collect()

    mx.save_safetensors(out_path, converted)
    print(f"  Saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    del converted
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
    """Map Qwen3-VL keys exactly like mflux's MageFlowWeightMapping."""
    if key in {"lm_head.weight", "model.visual.rotary_pos_emb.inv_freq"}:
        return None
    if key.startswith("model.language_model."):
        return "language_model." + key[len("model.language_model."):]
    if key.startswith("model.visual."):
        return "visual." + key[len("model.visual."):]
    raise ValueError(f"Unexpected Mage Flow text encoder weight: {key}")


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


def main():
    parser = argparse.ArgumentParser(description="Convert Mage-Flow PyTorch weights to MLX (Low Memory)")
    parser.add_argument("--repo", default="microsoft/Mage-Flow-Turbo", help="HuggingFace repo ID")
    parser.add_argument("--output", default="models/microsoft_Mage-Flow-Turbo", help="Output directory")
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
    )

    # Save DiT config
    with open(os.path.join(repo_dir, "transformer", "config.json")) as f:
        dit_config = json.load(f)
    with open(os.path.join(args.output, "transformer_config.json"), "w") as f:
        json.dump(dit_config, f, indent=2)
    with open(os.path.join(args.output, "precision_config.json"), "w") as f:
        json.dump(
            {
                "transformer_dtype": "bfloat16",
                "text_encoder_dtype": "bfloat16",
                "vae_dtype": "bfloat16",
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
                    mapped_key = map_text_encoder_key(key)
                    if mapped_key is None:
                        del t
                        continue
                    mx_arr = torch_to_mlx(t)
                    if mx_arr.ndim == 4:
                        mx_arr = mx.transpose(mx_arr, (0, 2, 3, 1))
                    elif mx_arr.ndim == 5:
                        mx_arr = mx.transpose(mx_arr, (0, 2, 3, 4, 1))
                    mx.eval(mx_arr)
                    mlx_te[mapped_key] = mx_arr
                    del t, mx_arr
                    gc.collect()

    te_out_path = os.path.join(args.output, "text_encoder.safetensors")
    mx.save_safetensors(te_out_path, mlx_te)
    print(f"  Saved: {te_out_path} ({os.path.getsize(te_out_path) / 1e6:.1f} MB)")
    del mlx_te
    gc.collect()

    print("\n✅ Conversion complete!")
    print(f"  Output directory: {args.output}")
    total_mb = sum(os.path.getsize(os.path.join(args.output, f)) for f in os.listdir(args.output) if f.endswith(".safetensors")) / 1e6
    print(f"  Total model size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()

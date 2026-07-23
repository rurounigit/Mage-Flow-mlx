"""Model loading, conversion, and caching utilities for Mage-Flow MLX."""

from __future__ import annotations

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
            if key.startswith("_") or "pos_embed" in key or "freqs" in key:
                continue

            tensor = f.get_tensor(key)
            mapped_key = key_mapper_fn(key)
            if mapped_key is None:
                del tensor
                continue

            arr = torch_to_mlx(tensor)
            if arr.ndim == 4:
                arr = mx.transpose(arr, (0, 2, 3, 1))
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

    key = key.replace(".t_embedder.mlp.0.", ".t_embedder.mlp.layers.0.")
    key = key.replace(".t_embedder.mlp.2.", ".t_embedder.mlp.layers.2.")
    key = key.replace(".ca.1.", ".ca.layers.1.")
    key = key.replace(".adaLN_modulation.1.", ".adaLN_modulation.layers.1.")
    key = key.replace(".mlp.0.", ".mlp.layers.0.")
    key = key.replace(".mlp.2.", ".mlp.layers.2.")
    key = key.replace(".x_embedder.embedder.0.", ".x_embedder.embedder.")

    key = key.replace(".decoder.block.0.norm1.", ".decoder.block.layers.0.norm1.norm.")
    key = key.replace(".decoder.block.0.norm2.", ".decoder.block.layers.0.norm2.norm.")
    key = key.replace(".decoder.block.1.norm.", ".decoder.block.layers.1.norm.norm.")
    key = key.replace(".decoder.block.2.norm1.", ".decoder.block.layers.2.norm1.norm.")
    key = key.replace(".decoder.block.2.norm2.", ".decoder.block.layers.2.norm2.norm.")
    key = key.replace(".decoder.block.3.norm.", ".decoder.block.layers.3.norm.norm.")
    key = key.replace(".decoder.block.4.norm1.", ".decoder.block.layers.4.norm1.norm.")
    key = key.replace(".decoder.block.4.norm2.", ".decoder.block.layers.4.norm2.norm.")

    key = key.replace(".decoder.block.0.", ".decoder.block.layers.0.")
    key = key.replace(".decoder.block.1.", ".decoder.block.layers.1.")
    key = key.replace(".decoder.block.2.", ".decoder.block.layers.2.")
    key = key.replace(".decoder.block.3.", ".decoder.block.layers.3.")
    key = key.replace(".decoder.block.4.", ".decoder.block.layers.4.")

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


def ensure_mlx_model(model_path_or_repo: str = "models/mage_flow_mlx") -> str:
    """Ensure that converted MLX model weights exist at model_path_or_repo.

    If model_path_or_repo is a Hugging Face repo ID or a local path lacking converted weights,
    downloads and converts weights automatically on-the-fly, caching them locally.
    """
    required_files = ["transformer.safetensors", "vae.safetensors", "transformer_config.json"]

    # Determine output directory and repo ID
    # If the path contains "/" and doesn't exist locally, treat it as an HF repo ID
    # A local path like "models/mage_flow_mlx" also contains "/" but exists locally
    # A HF repo ID like "microsoft/Mage-Flow-Turbo" contains "/" and doesn't exist locally
    # But we need to distinguish: "models/mage_flow_mlx" (local) vs "microsoft/Mage-Flow-Turbo" (HF)
    # HF repo IDs have a slash and don't exist as local paths
    # Local paths that don't exist yet should NOT be treated as HF repo IDs
    # Only treat as HF repo ID if it looks like an HF repo ID (contains "/" and doesn't start with "models/")
    if "/" in model_path_or_repo and not os.path.exists(model_path_or_repo) and not model_path_or_repo.startswith("models/"):
        # Treated as HF repo ID (e.g. "microsoft/Mage-Flow-Turbo")
        repo_id = model_path_or_repo
        # Use a sanitized local dir name based on the repo ID
        safe_name = model_path_or_repo.replace("/", "_")
        output_dir = os.path.join("models", safe_name)
    else:
        # Local path provided (may or may not have converted weights yet)
        output_dir = model_path_or_repo
        repo_id = "microsoft/Mage-Flow-Turbo"

    # Check if all required converted files exist
    all_exist = all(os.path.exists(os.path.join(output_dir, f)) for f in required_files)
    if all_exist:
        return output_dir

    print(f"🔄 Converted MLX weights not found in '{output_dir}'.")
    print(f"📥 Downloading and converting weights from {repo_id} (this happens only once)...")

    os.makedirs(output_dir, exist_ok=True)
    print(f"  Downloading from HuggingFace: {repo_id}...")
    print("  This may take a while (17GB of weights)...")
    repo_dir = snapshot_download(
        repo_id,
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
    )
    print(f"  Downloaded to: {repo_dir}")

    # 1. DiT weights & config
    print("Converting DiT weights...")
    dit_path = os.path.join(repo_dir, "transformer", "diffusion_pytorch_model.safetensors")
    if os.path.exists(dit_path):
        process_and_convert_file(
            dit_path,
            os.path.join(output_dir, "transformer.safetensors"),
            key_mapper_fn=map_dit_key,
        )

    dit_config_src = os.path.join(repo_dir, "transformer", "config.json")
    if os.path.exists(dit_config_src):
        with open(dit_config_src) as f:
            dit_config = json.load(f)
        with open(os.path.join(output_dir, "transformer_config.json"), "w") as f:
            json.dump(dit_config, f, indent=2)

    with open(os.path.join(output_dir, "precision_config.json"), "w") as f:
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
    print("Converting VAE weights...")
    vae_path = os.path.join(repo_dir, "vae", "diffusion_pytorch_model.safetensors")
    if os.path.exists(vae_path):
        process_and_convert_file(
            vae_path,
            os.path.join(output_dir, "vae.safetensors"),
            key_mapper_fn=map_vae_key,
        )

    # 3. Text Encoder weights
    print("Converting Text Encoder weights...")
    te_dir = os.path.join(repo_dir, "text_encoder")
    mlx_te = {}
    if os.path.exists(te_dir):
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
                        mx.eval(mx_arr)
                        mlx_te[mapped_key] = mx_arr
                        del t, mx_arr
                        gc.collect()

    te_out_path = os.path.join(output_dir, "text_encoder.safetensors")
    if mlx_te:
        mx.save_safetensors(te_out_path, mlx_te)
        print(f"  Saved: {te_out_path} ({os.path.getsize(te_out_path) / 1e6:.1f} MB)")
    del mlx_te
    gc.collect()

    print(f"✅ Auto-conversion complete! Weights cached at {output_dir}")
    return output_dir

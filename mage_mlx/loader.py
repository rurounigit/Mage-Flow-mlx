"""Model loading, conversion, and caching utilities for Mage-Flow MLX."""

from __future__ import annotations

from typing import Optional

import gc
import json
import os
import struct

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def _read_safetensors_header(path: str) -> dict:
    """Read safetensors metadata without loading checkpoint tensors."""
    with open(path, "rb") as f:
        header_length_bytes = f.read(8)
        if len(header_length_bytes) != 8:
            raise ValueError(f"Invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        # Safetensors headers are small JSON documents. Bound the value before
        # reading so arbitrary/non-safetensors files cannot request huge memory.
        if header_length <= 0 or header_length > 100 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header length in {path}")
        return json.loads(f.read(header_length))


def is_unquantized_transformer(path: str) -> bool:
    """Return whether a checkpoint contains the expected floating DiT weights."""
    if not os.path.exists(path):
        return False
    try:
        header = _read_safetensors_header(path)
        img_weight = header.get("img_in.weight", {})
        dtype = img_weight.get("dtype")
        shape = img_weight.get("shape")
        has_quantization_state = any(
            key.endswith((".scales", ".biases")) for key in header
        )
        return (
            dtype in {"BF16", "F16", "F32"}
            and shape == [3072, 128]
            and not has_quantization_state
        )
    except (OSError, ValueError, json.JSONDecodeError, struct.error):
        return False


def _cached_repo_snapshot(repo_id: str) -> str | None:
    """Resolve an existing Hugging Face snapshot even if optional files are absent."""
    cache_root = os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    repo_cache = os.path.join(cache_root, f"models--{repo_id.replace('/', '--')}")
    ref_path = os.path.join(repo_cache, "refs", "main")
    try:
        with open(ref_path) as f:
            commit = f.read().strip()
    except OSError:
        return None
    snapshot = os.path.join(repo_cache, "snapshots", commit)
    source = os.path.join(
        snapshot, "transformer", "diffusion_pytorch_model.safetensors"
    )
    return snapshot if os.path.exists(source) else None


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
    """Convert one source checkpoint atomically while preserving BF16 tensors."""
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
            elif arr.ndim == 5:
                # PyTorch Conv3d [out, in, D, H, W] -> MLX channel-last
                # Conv3d [out, D, H, W, in].
                arr = mx.transpose(arr, (0, 2, 3, 4, 1))
            mx.eval(arr)
            converted[mapped_key] = arr
            del tensor, arr
            gc.collect()

    root, extension = os.path.splitext(out_path)
    temp_path = f"{root}.tmp{extension}"
    mx.save_safetensors(temp_path, converted)
    os.replace(temp_path, out_path)
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
    """Map Hugging Face Qwen3-VL text keys to native MLX Qwen3-VL module tree.

    Keeps vision tower weights (mapped to ``visual.*``) so the text encoder
    can perform multi-modal encoding for image editing.
    """
    # Keep vision tower weights for the native MLX Qwen3-VL text encoder
    if key.startswith("model.visual."):
        return "visual." + key[len("model.visual."):]
    if key.startswith("model.vision_tower."):
        return "visual." + key[len("model.vision_tower."):]
    if key.startswith("vision_tower."):
        return "visual." + key[len("vision_tower."):]
    if key.startswith("model.language_model."):
        return "language_model." + key[len("model.language_model."):]
    if key.startswith("language_model."):
        return key
    return "language_model." + key



SHARED_TEXT_ENCODER_PATH = os.path.join(
    "models", "shared", "mage_flow_qwen3vl", "text_encoder.safetensors"
)


def _has_visual_weights(path: str) -> bool:
    """Return whether a converted encoder uses the canonical visual namespace."""
    if not os.path.exists(path):
        return False
    try:
        with safe_open(path, framework="pt") as f:
            return any(key.startswith("visual.") for key in f.keys())
    except (OSError, ValueError):
        return False


def resolve_text_encoder_path(model_dir: str) -> str | None:
    """Resolve and migrate to the shared Qwen3-VL text encoder cache."""
    shared = SHARED_TEXT_ENCODER_PATH
    candidates = [os.path.join(model_dir, "text_encoder.safetensors")]
    if os.path.isdir("models"):
        candidates.extend(
            os.path.join("models", name, "text_encoder.safetensors")
            for name in os.listdir("models") if name.startswith("microsoft_")
        )
    if not _has_visual_weights(shared):
        source = next((path for path in candidates if _has_visual_weights(path)), None)
        if source:
            os.makedirs(os.path.dirname(shared), exist_ok=True)
            temp = shared + ".tmp"
            with open(source, "rb") as src, open(temp, "wb") as dst:
                while chunk := src.read(16 * 1024 * 1024):
                    dst.write(chunk)
            os.replace(temp, shared)
    if _has_visual_weights(shared):
        return shared
    local = os.path.join(model_dir, "text_encoder.safetensors")
    return local if os.path.exists(local) else None


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


def ensure_mlx_model(
    model_path_or_repo: str = "models/microsoft_Mage-Flow-Turbo",
    quantize: int | None = None,
    profiler: Optional["object"] = None,
) -> tuple[str, int | None]:
    """Ensure that converted MLX model weights exist at model_path_or_repo.

    If model_path_or_repo is a Hugging Face repo ID or a local path lacking converted weights,
    downloads and converts weights automatically on-the-fly, caching them locally.

    Args:
        model_path_or_repo: Local path or HF repo ID
        quantize: If set (4 or 8), quantize transformer weights to N bits (applied at runtime via nn.quantize)

    Returns:
        Tuple of (model_dir, quantize) where quantize is the actual quantization
        level that will be used
    """
    required_files = ["transformer.safetensors", "vae.safetensors", "transformer_config.json"]

    # Determine output directory and repo ID
    if "/" in model_path_or_repo and not os.path.exists(model_path_or_repo) and not model_path_or_repo.startswith("models/"):
        repo_id = model_path_or_repo
        safe_name = model_path_or_repo.replace("/", "_")
        output_dir = os.path.join("models", safe_name)
    else:
        output_dir = model_path_or_repo
        repo_id = "microsoft/Mage-Flow-Turbo"

    # Migrate the original local default without copying multi-GB files.  Keep
    # a symlink at the legacy path so existing scripts remain compatible.
    canonical_default = os.path.join("models", "microsoft_Mage-Flow-Turbo")
    legacy_default = os.path.join("models", "mage_flow_mlx")
    if output_dir == canonical_default and not os.path.exists(output_dir):
        if os.path.isdir(legacy_default) and os.path.exists(os.path.join(legacy_default, "transformer.safetensors")):
            os.rename(legacy_default, canonical_default)
            os.symlink(os.path.basename(canonical_default), legacy_default)

    # A quantized checkpoint cannot be loaded into the model's nn.Linear layers.
    # Validate the base checkpoint's format, not just its presence.
    transformer_out = os.path.join(output_dir, "transformer.safetensors")
    transformer_valid = is_unquantized_transformer(transformer_out)
    other_files_exist = all(
        os.path.exists(os.path.join(output_dir, f))
        for f in required_files
        if f != "transformer.safetensors"
    )
    local_te_cache = os.path.join(output_dir, "text_encoder.safetensors")
    visual_weights_valid = _has_visual_weights(SHARED_TEXT_ENCODER_PATH) or _has_visual_weights(local_te_cache)
    all_exist = transformer_valid and other_files_exist and visual_weights_valid
    if not visual_weights_valid:
        # The cache predates visual-key conversion; force text-encoder rebuild.
        te_cache = os.path.join(output_dir, "text_encoder.safetensors")
        if os.path.exists(te_cache) and not _has_visual_weights(SHARED_TEXT_ENCODER_PATH):
            os.remove(te_cache)
    if not all_exist:
        print(f"🔄 Converted MLX weights not found in '{output_dir}'.")
        print(f"📥 Downloading and converting weights from {repo_id} (this happens only once)...")
        if profiler:
            profiler.log(f"📥 Downloading and converting weights from {repo_id} (this happens only once)...")

        os.makedirs(output_dir, exist_ok=True)
        print(f"  Downloading from HuggingFace: {repo_id}...")
        if profiler:
            profiler.log(f"  Downloading from HuggingFace: {repo_id}...")
        print("  This may take a while (17GB of weights)...")
        repo_dir = _cached_repo_snapshot(repo_id) or snapshot_download(
            repo_id,
            allow_patterns=["*.safetensors", "*.json", "*.txt"],
        )
        print(f"  Downloaded to: {repo_dir}")

        # 1. DiT weights & config
        print("Converting DiT weights...")
        if profiler:
            profiler.log("Converting DiT weights...")
        dit_path = os.path.join(repo_dir, "transformer", "diffusion_pytorch_model.safetensors")
        if not transformer_valid and os.path.exists(dit_path):
            process_and_convert_file(
                dit_path,
                transformer_out,
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
        if profiler:
            profiler.log("Converting VAE weights...")
        vae_path = os.path.join(repo_dir, "vae", "diffusion_pytorch_model.safetensors")
        vae_out_path = os.path.join(output_dir, "vae.safetensors")
        if not os.path.exists(vae_out_path) and os.path.exists(vae_path):
            process_and_convert_file(
                vae_path,
                vae_out_path,
                key_mapper_fn=map_vae_key,
            )

        # 3. Text Encoder weights
        print("Converting Text Encoder weights...")
        if profiler:
            profiler.log("Converting Text Encoder weights...")
        te_dir = os.path.join(repo_dir, "text_encoder")
        mlx_te = {}
        te_out_path = SHARED_TEXT_ENCODER_PATH
        if not _has_visual_weights(te_out_path) and os.path.exists(te_dir):
            os.makedirs(os.path.dirname(te_out_path), exist_ok=True)
            for shard in sorted(os.listdir(te_dir)):
                if shard.endswith(".safetensors"):
                    shard_path = os.path.join(te_dir, shard)
                    with safe_open(shard_path, framework="pt") as f:
                        for key in list(f.keys()):
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


        if mlx_te:
            mx.save_safetensors(te_out_path, mlx_te)
            print(f"  Saved: {te_out_path} ({os.path.getsize(te_out_path) / 1e6:.1f} MB)")
        del mlx_te
        gc.collect()

        print(f"Auto-conversion complete! Weights cached at {output_dir}")
    else:
        pass  # Weights already cached, no need to print

    return output_dir, quantize

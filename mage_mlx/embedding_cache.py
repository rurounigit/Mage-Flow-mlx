"""Prompt embedding cache for Mage-Flow MLX.

Caches text embeddings produced by the Qwen3-VL text encoder, keyed by:
- formatted prompt text (including chat template)
- negative prompt text
- text-encoder checkpoint signature (file size + mtime)
- tokenizer/template version

For a cache hit, Qwen loading and text encoding are skipped entirely.
A small embedding such as [1, 30, 2560] BF16 is ~240 KB — tiny compared
with 8 GiB of model weights.

Usage:
    from mage_mlx.embedding_cache import EmbeddingCache

    cache = EmbeddingCache(model_dir="models/mage_flow_mlx")
    key = cache.make_key(prompt="A cat", negative_prompt=" ", te_path="text_encoder.safetensors")
    embeds = cache.get(key)
    if embeds is None:
        embeds = text_encoder(prompt)
        cache.put(key, embeds)
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import mlx.core as mx


# Cache format version — bump when the embedding format or template changes
EMBEDDING_CACHE_VERSION = 1

# Tokenizer/template version — bump when MAGE_FLOW_TEMPLATE or START_IDX changes
TOKENIZER_VERSION = 1


class EmbeddingCache:
    """Persistent cache for text encoder embeddings.

    Args:
        model_dir: Directory containing the model (cache stored in
            ``model_dir/embedding_cache/``)
    """

    def __init__(self, model_dir: str = "models/mage_flow_mlx"):
        self.cache_dir = os.path.join(model_dir, "embedding_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def make_key(
        self,
        prompt: str,
        negative_prompt: str = " ",
        te_path: Optional[str] = None,
        template_version: int = TOKENIZER_VERSION,
    ) -> str:
        """Build a cache key from prompt content and encoder signature.

        The key incorporates:
        - The formatted prompt (with chat template applied)
        - The negative prompt
        - The text-encoder checkpoint signature (size + mtime)
        - The tokenizer/template version

        Args:
            prompt: Raw prompt text (before template)
            negative_prompt: Negative prompt text
            te_path: Path to text_encoder.safetensors (for signature)
            template_version: Version of the chat template

        Returns:
            SHA-256 hex digest string
        """
        from mage_mlx.text_encoder import MAGE_FLOW_TEMPLATE

        # Apply the same template the text encoder uses
        formatted = MAGE_FLOW_TEMPLATE.format(prompt)
        formatted_neg = MAGE_FLOW_TEMPLATE.format(negative_prompt)

        # Text-encoder checkpoint signature
        te_signature = "none"
        if te_path and os.path.exists(te_path):
            stat = os.stat(te_path)
            te_signature = f"{stat.st_size}:{stat.st_mtime_ns}"

        key_data = {
            "version": EMBEDDING_CACHE_VERSION,
            "template_version": template_version,
            "prompt": formatted,
            "negative_prompt": formatted_neg,
            "te_signature": te_signature,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        """Return the .safetensors path for a given cache key."""
        return os.path.join(self.cache_dir, f"{key}.safetensors")

    def _meta_path(self, key: str) -> str:
        """Return the .json metadata path for a given cache key."""
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str) -> Optional[mx.array]:
        """Retrieve cached embeddings, or None on cache miss.

        Args:
            key: Cache key from ``make_key()``

        Returns:
            Cached [1, seq_len, 2560] BF16 array, or None
        """
        safetensors_path = self._cache_path(key)
        meta_path = self._meta_path(key)

        if not os.path.exists(safetensors_path) or not os.path.exists(meta_path):
            return None

        try:
            # Validate metadata
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("version") != EMBEDDING_CACHE_VERSION:
                return None

            # Load the embedding
            arr = mx.load(safetensors_path)
            return arr
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def put(self, key: str, embeds: mx.array) -> None:
        """Store embeddings in the cache.

        Args:
            key: Cache key from ``make_key()``
            embeds: [1, seq_len, 2560] BF16 array to cache
        """
        safetensors_path = self._cache_path(key)
        meta_path = self._meta_path(key)

        # Atomic write: save data first, then metadata
        weights_temp = f"{safetensors_path}.tmp"
        meta_temp = f"{meta_path}.tmp"

        try:
            mx.save(weights_temp, embeds)
            os.replace(weights_temp, safetensors_path)

            meta = {
                "version": EMBEDDING_CACHE_VERSION,
                "shape": list(embeds.shape),
                "dtype": str(embeds.dtype),
            }
            with open(meta_temp, "w") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
            os.replace(meta_temp, meta_path)
        except OSError:
            # Clean up temp files on failure
            for path in (weights_temp, meta_temp):
                if os.path.exists(path):
                    os.remove(path)
            raise

    def clear(self) -> int:
        """Remove all cached embeddings. Returns the number of entries removed."""
        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".safetensors"):
                key = filename[:-len(".safetensors")]
                safetensors_path = self._cache_path(key)
                meta_path = self._meta_path(key)
                for path in (safetensors_path, meta_path):
                    if os.path.exists(path):
                        os.remove(path)
                count += 1
        return count
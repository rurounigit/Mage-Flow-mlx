"""VAE reference latent cache for Mage-Flow edit.

Caches VAE-encoded reference image latents on disk, keyed by:
- Raw image bytes hash (SHA-256)
- Image pixel dimensions (width x height)
- VAE checkpoint signature (file size + mtime)

On a cache hit, the expensive VAE encode step is skipped entirely.
A single reference latent such as [1, 4096, 128] BF16 is ~10 MB — small
compared with the ~1 GiB VAE weights and the ~7.9 GiB peak RAM.

Usage:
    from mage_mlx.vision_cache import VisionCache

    cache = VisionCache(model_dir="models/microsoft_Mage-Flow-Edit-Turbo")
    key = cache.make_key(image_bytes=raw_bytes, size=(1024, 1024), vae_path="vae.safetensors")
    latents = cache.get(key)
    if latents is None:
        latents = vae.encode(image_array)
        cache.put(key, latents)
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import mlx.core as mx


# Cache format version — bump when the latent format or VAE changes
VISION_CACHE_VERSION = 1


class VisionCache:
    """Persistent cache for VAE-encoded reference image latents.

    Args:
        model_dir: Directory containing the model (cache stored in
            ``model_dir/vision_cache/``)
    """

    def __init__(self, model_dir: str = "models/microsoft_Mage-Flow-Turbo"):
        self.cache_dir = os.path.join(model_dir, "vision_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def make_key(
        self,
        image_bytes: bytes,
        size: tuple[int, int],
        vae_path: Optional[str] = None,
    ) -> str:
        """Build a cache key from image content and VAE signature.

        The key incorporates:
        - SHA-256 hash of the raw image bytes
        - Image pixel dimensions (width x height)
        - VAE checkpoint signature (file size + mtime)

        Args:
            image_bytes: Raw image file bytes
            size: (width, height) of the image
            vae_path: Path to vae.safetensors (for signature)

        Returns:
            SHA-256 hex digest string
        """
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        vae_signature = "none"
        if vae_path and os.path.exists(vae_path):
            stat = os.stat(vae_path)
            vae_signature = f"{stat.st_size}:{stat.st_mtime_ns}"

        key_data = {
            "version": VISION_CACHE_VERSION,
            "image_hash": image_hash,
            "width": size[0],
            "height": size[1],
            "vae_signature": vae_signature,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        """Return the .npy path for a given cache key."""
        return os.path.join(self.cache_dir, f"{key}.npy")

    def _meta_path(self, key: str) -> str:
        """Return the .json metadata path for a given cache key."""
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str) -> Optional[mx.array]:
        """Retrieve cached VAE latents, or None on cache miss.

        Args:
            key: Cache key from ``make_key()``

        Returns:
            Cached [1, lat_h*lat_w, 128] BF16 array, or None
        """
        npy_path = self._cache_path(key)
        meta_path = self._meta_path(key)

        if not os.path.exists(npy_path) or not os.path.exists(meta_path):
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("version") != VISION_CACHE_VERSION:
                return None

            arr = mx.load(npy_path)
            return arr
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def put(self, key: str, latents: mx.array) -> None:
        """Store VAE latents in the cache.

        Uses atomic write: save to a temp .npy file, then rename.

        Args:
            key: Cache key from ``make_key()``
            latents: [1, lat_h*lat_w, 128] BF16 array to cache
        """
        npy_path = self._cache_path(key)
        meta_path = self._meta_path(key)

        weights_temp = os.path.join(self.cache_dir, f"{key}.tmp.npy")
        meta_temp = os.path.join(self.cache_dir, f"{key}.tmp.json")

        try:
            mx.save(weights_temp, latents)
            os.replace(weights_temp, npy_path)

            meta = {
                "version": VISION_CACHE_VERSION,
                "shape": list(latents.shape),
                "dtype": str(latents.dtype),
            }
            with open(meta_temp, "w") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
            os.replace(meta_temp, meta_path)
        except OSError:
            for path in (weights_temp, meta_temp):
                if os.path.exists(path):
                    os.remove(path)
            raise

    def clear(self) -> int:
        """Remove all cached latents. Returns the number of entries removed."""
        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".npy"):
                key = filename[:-len(".npy")]
                npy_path = self._cache_path(key)
                meta_path = self._meta_path(key)
                for path in (npy_path, meta_path):
                    if os.path.exists(path):
                        os.remove(path)
                count += 1
        return count

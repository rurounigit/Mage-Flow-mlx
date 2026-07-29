"""Unified output path resolution for Mage-Flow MLX.

Provides a single function, :func:`resolve_output_path`, that all generation
modes (single generation, edit, worker, edit worker) use to decide where to
save the output image.  The rules are:

1. **Filename only** (e.g. ``"image.png"``) — save into the ``output/``
   subfolder of the current working directory.
2. **Absolute path with filename** (e.g. ``"/tmp/img.png"``) — save there.
3. **Absolute path without filename** (e.g. ``"/tmp/"``) — construct a default
   filename from metadata (resolution, steps, seed, quantization) plus a short
   unique identifier, then save there.
4. **No output** (``None``) — same as case 3 but in the ``output/`` subfolder.

In every case, if a file with the resolved name already exists it is silently
overwritten.  The target directory is created if it does not exist.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

#: Subfolder used when the user provides only a bare filename or no output at all.
DEFAULT_OUTPUT_DIR = "output"

#: Image extension appended to auto-generated filenames.
DEFAULT_EXTENSION = ".png"


def _format_quantize(quantize: Optional[int]) -> str:
    """Return a short string representation of the quantization level.

    ``None`` (no quantization) becomes ``"f16"`` to indicate the canonical
    BF16/F16 checkpoint.
    """
    if quantize is None:
        return "f16"
    return f"q{quantize}"


def _default_filename(
    width: int,
    height: int,
    steps: int,
    seed: int,
    quantize: Optional[int],
) -> str:
    """Build a deterministic-ish filename from important metadata values.

    A short 8-character hex suffix from :func:`uuid.uuid4` guarantees
    uniqueness across runs that share the same metadata.
    """
    short_id = uuid.uuid4().hex[:8]
    return (
        f"{width}x{height}_s{steps}_seed{seed}"
        f"_{_format_quantize(quantize)}_{short_id}{DEFAULT_EXTENSION}"
    )


def resolve_output_path(
    output: Optional[str],
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    seed: int = 42,
    quantize: Optional[int] = None,
    mode: str = "txt2img",
) -> str:
    """Resolve the final image save path according to the unified rules.

    Args:
        output: The raw ``--output`` value from the CLI or JSONL prompt.
            ``None`` means the user did not specify an output.
        width: Generation width in pixels.
        height: Generation height in pixels.
        steps: Number of denoising steps.
        seed: Random seed.
        quantize: Quantization level (4, 8) or ``None`` for no quantization.
        mode: Generation mode (``"txt2img"`` or ``"edit"``).  Currently
            informational; included for future extensibility.

    Returns:
        An absolute or project-relative path string.  The parent directory is
        created if it does not exist.

    Examples:
        >>> resolve_output_path("image.png")
        'output/image.png'
        >>> resolve_output_path("/tmp/img.png")
        '/tmp/img.png'
        >>> resolve_output_path("/tmp/")  # doctest: +SKIP
        '/tmp/1024x1024_s4_seed42_f16_a3f7b2c1.png'
        >>> resolve_output_path(None)  # doctest: +SKIP
        'output/1024x1024_s4_seed42_f16_a3f7b2c1.png'
    """
    # --- Case 4: no output specified at all ---
    if output is None:
        filename = _default_filename(width, height, steps, seed, quantize)
        path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        return path

    # --- Case 1: bare filename (no directory separator) ---
    # os.path.split returns ("", "image.png") for a bare filename.
    # We also treat paths that are just a filename with no directory component.
    parent, name = os.path.split(output)
    if parent == "":
        # Bare filename — save into output/ subfolder.
        path = os.path.join(DEFAULT_OUTPUT_DIR, name)
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        return path

    # --- Case 2 & 3: path with a directory component ---
    # Determine whether the path ends with a separator (i.e. no filename).
    # os.path.split("/tmp/") returns ("/tmp", "") on POSIX.
    if name == "":
        # Absolute (or relative) path without a filename — generate one.
        filename = _default_filename(width, height, steps, seed, quantize)
        path = os.path.join(output, filename)
    else:
        # Path with a filename — use as-is.
        path = output

    # Ensure the directory portion exists.
    dir_part = os.path.dirname(path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)
    return path

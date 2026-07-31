"""Thermal state detection for macOS (Apple Silicon).

Reads the macOS thermal pressure level via the notify framework (ctypes),
which works on macOS 26 (Tahoe) and later. Falls back to sysctl on older
macOS versions where the notify key is unavailable.

The notify framework exposes ``com.apple.system.thermalpressurelevel``
as an integer 0-3:
    0 = NOMINAL   (no throttling)
    1 = FAIR      (light throttling)
    2 = SERIOUS    (moderate-to-heavy throttling)
    3 = CRITICAL  (severe throttling)

On older macOS, sysctl keys ``machdep.xcpm.cpu_thermal_level`` and
``machdep.xcpm.gpu_thermal_level`` provide 0-100 integer levels.

Usage:
    from mage_mlx.thermal import get_thermal_state, format_thermal_state

    state = get_thermal_state()
    print(format_thermal_state(state))
    # → "NOMINAL"  or  "CPU=42 GPU=38 (SERIOUS)"  or  "thermal state unavailable"
"""
from __future__ import annotations

import ctypes
import subprocess
from typing import Optional


# ── Label mappings ──────────────────────────────────────────────────────────

# Notify framework states (0-3) → human-readable labels
_NOTIFY_LABELS = {
    0: "NOMINAL",
    1: "FAIR",
    2: "SERIOUS",
    3: "CRITICAL",
}

# Sysctl levels (0-100) → same label scheme
# Thresholds chosen to align with macOS thermal management levels
_SYSCTL_THRESHOLDS = [
    (20, "NOMINAL"),
    (40, "FAIR"),
    (60, "SERIOUS"),
    (101, "CRITICAL"),  # anything >= 60 is CRITICAL
]


def _level_to_label(level: int) -> str:
    """Convert a sysctl thermal level (0-100) to a human-readable label.

    Maps to the same NOMINAL/FAIR/SERIOUS/CRITICAL scheme used by the
    notify framework, so both detection methods produce consistent labels.
    """
    for threshold, label in _SYSCTL_THRESHOLDS:
        if level < threshold:
            return label
    return "CRITICAL"


def _notify_state_to_label(state: int) -> str:
    """Convert a notify framework thermal state (0-3) to a label."""
    return _NOTIFY_LABELS.get(state, "unknown")


# ── Detection methods ────────────────────────────────────────────────────────

def _get_thermal_via_notify() -> Optional[int]:
    """Get thermal pressure level via the macOS notify framework (ctypes).

    Uses ``notify_register_check`` + ``notify_get_state`` with the
    ``com.apple.system.thermalpressurelevel`` key. This is the primary
    method on macOS 26 (Tahoe) where sysctl keys are no longer available.

    Returns:
        Integer 0-3 (NOMINAL/CRITICAL) or None if unavailable.
    """
    try:
        libc = ctypes.CDLL(None)

        libc.notify_register_check.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        libc.notify_register_check.restype = ctypes.c_int

        libc.notify_get_state.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        libc.notify_get_state.restype = ctypes.c_int

        token = ctypes.c_int()
        ret = libc.notify_register_check(
            b"com.apple.system.thermalpressurelevel",
            ctypes.byref(token),
        )
        if ret != 0:
            return None

        state = ctypes.c_uint64()
        ret = libc.notify_get_state(token.value, ctypes.byref(state))
        if ret != 0:
            return None

        return int(state.value)
    except Exception:
        return None


def _read_sysctl(key: str) -> Optional[int]:
    """Read an integer value from sysctl (fallback for older macOS).

    Args:
        key: The sysctl key to read (e.g. "machdep.xcpm.cpu_thermal_level")

    Returns:
        The integer value, or None if the key is not available.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (ValueError, OSError, subprocess.TimeoutExpired):
        pass
    return None


# ── Public API ───────────────────────────────────────────────────────────────

def get_thermal_state() -> dict:
    """Get current thermal state from macOS.

    Primary method: ctypes + notify framework (``com.apple.system.thermalpressurelevel``).
    Fallback: sysctl (``machdep.xcpm.cpu_thermal_level`` / ``gpu_thermal_level``).

    On non-macOS or when neither method is available, levels are None
    and the label is "unknown".

    Returns:
        dict with keys:
            - cpu_thermal_level: int or None
            - gpu_thermal_level: int or None
            - thermal_throttling: str ("NOMINAL", "FAIR", "SERIOUS",
              "CRITICAL", or "unknown")
    """
    # Try notify framework first (macOS 26+)
    notify_state = _get_thermal_via_notify()
    if notify_state is not None:
        return {
            "cpu_thermal_level": notify_state,
            "gpu_thermal_level": None,
            "thermal_throttling": _notify_state_to_label(notify_state),
        }

    # Fallback: sysctl (older macOS)
    cpu_level = _read_sysctl("machdep.xcpm.cpu_thermal_level")
    gpu_level = _read_sysctl("machdep.xcpm.gpu_thermal_level")

    if cpu_level is not None:
        throttling = _level_to_label(cpu_level)
    elif gpu_level is not None:
        throttling = _level_to_label(gpu_level)
    else:
        throttling = "unknown"

    return {
        "cpu_thermal_level": cpu_level,
        "gpu_thermal_level": gpu_level,
        "thermal_throttling": throttling,
    }


def format_thermal_state(state: dict) -> str:
    """Format a thermal state dict as a human-readable string.

    Examples:
        >>> state = {"cpu_thermal_level": 0, "gpu_thermal_level": None,
        ...          "thermal_throttling": "NOMINAL"}
        >>> format_thermal_state(state)
        'NOMINAL'

        >>> state = {"cpu_thermal_level": 42, "gpu_thermal_level": 38,
        ...          "thermal_throttling": "SERIOUS"}
        >>> format_thermal_state(state)
        'CPU=42 GPU=38 (SERIOUS)'

    If levels are unavailable, returns "thermal state unavailable".
    """
    parts = []
    if state.get("cpu_thermal_level") is not None:
        parts.append(f"CPU={state['cpu_thermal_level']}")
    if state.get("gpu_thermal_level") is not None:
        parts.append(f"GPU={state['gpu_thermal_level']}")
    if not parts:
        return "thermal state unavailable"
    return f"{' '.join(parts)} ({state['thermal_throttling']})"

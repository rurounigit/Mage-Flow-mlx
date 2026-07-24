"""Phase-level profiler for Mage-Flow MLX.

Provides fine-grained timing and memory tracking across the full generation
lifecycle — from Python startup through model loading, text encoding, DiT
denoising, VAE decode, and PNG save.

Usage:
    from mage_mlx.profiler import Profiler

    prof = Profiler(enabled=True)
    prof.start("total")
    prof.start("import_mlx")
    import mlx.core as mx
    prof.stop("import_mlx")
    # ... more phases ...
    prof.stop("total")
    prof.print_report()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhaseRecord:
    """Timing record for a single phase."""

    name: str
    elapsed: float
    peak_rss_gib: Optional[float] = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Profiler:
    """Phase-level profiler with optional memory tracking.

    Args:
        enabled: If False, all methods are no-ops (zero overhead).
        track_memory: If True, record peak RSS after each phase.
    """

    enabled: bool = True
    track_memory: bool = True
    _records: list[PhaseRecord] = field(default_factory=list)
    _timers: dict[str, list[float]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_rss_gib() -> Optional[float]:
        """Return current process RSS in GiB, or None if unavailable."""
        try:
            import platform
            import resource

            # ru_maxrss is in bytes on macOS, kilobytes on Linux.
            # Detect by platform rather than magnitude, since a 5 GB process
            # on macOS (~5e9 bytes) would be misread as ~4768 GiB if treated
            # as KB.
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss > 0:
                if platform.system() == "Darwin":
                    return rss / (1024 ** 3)  # bytes → GiB
                else:
                    return rss / (1024 ** 2)  # KB → GiB
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Timing API
    # ------------------------------------------------------------------
    def start(self, name: str) -> None:
        """Start timing a named phase.

        Supports nested phases with the same name via a stack.
        """
        if not self.enabled:
            return
        if name not in self._timers:
            self._timers[name] = []
        self._timers[name].append(time.perf_counter())

    def stop(self, name: str) -> float:
        """Stop timing a named phase and record the elapsed time.

        Pops the most recent start for the given name (LIFO), so nested
        phases with the same name are handled correctly.

        Returns the elapsed seconds (0.0 if disabled or no matching start).
        """
        if not self.enabled:
            return 0.0
        timers = self._timers.get(name)
        if not timers:
            return 0.0
        start = timers.pop()
        if not timers:
            del self._timers[name]
        elapsed = time.perf_counter() - start
        rss = self._get_rss_gib() if self.track_memory else None
        self._records.append(PhaseRecord(name=name, elapsed=elapsed, peak_rss_gib=rss))
        return elapsed

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def get_records(self) -> list[PhaseRecord]:
        """Return all recorded phases."""
        return list(self._records)

    def get_elapsed(self, name: str) -> Optional[float]:
        """Return elapsed time for a named phase, or None if not found."""
        for rec in self._records:
            if rec.name == name:
                return rec.elapsed
        return None

    def set_metadata(self, name: str, key: str, value: str) -> None:
        """Attach a metadata key-value pair to a named phase record.

        Useful for annotating phases with context like resolution or steps.
        """
        if not self.enabled:
            return
        for rec in self._records:
            if rec.name == name:
                rec.metadata[key] = value
                return

    def total_elapsed(self) -> float:
        """Return total elapsed time across all recorded phases."""
        return sum(r.elapsed for r in self._records)

    def print_report(self, title: str = "Mage-Flow MLX Profiler") -> str:
        """Print a formatted timing report and return it as a string."""
        if not self.enabled or not self._records:
            return ""

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"  {title}")
        lines.append("=" * 60)
        lines.append(f"  {'Phase':<40} {'Time (s)':>10}   {'Peak RSS (GiB)':>14}")
        lines.append("  " + "-" * 64)

        for rec in self._records:
            rss_str = f"{rec.peak_rss_gib:>14.2f}" if rec.peak_rss_gib is not None else "              N/A"
            meta_str = ""
            if rec.metadata:
                meta_str = "  " + " ".join(f"{k}={v}" for k, v in rec.metadata.items())
            lines.append(f"  {rec.name:<40} {rec.elapsed:>10.4f}   {rss_str}{meta_str}")

        # Show wall-clock total (not sum of phases, which double-counts
        # nested phases like generation_N that include dit_step_N children).
        wall_clock = self.get_elapsed("total_wall_clock")
        lines.append("  " + "-" * 64)
        if wall_clock is not None:
            lines.append(f"  {'total_wall_clock':<40} {wall_clock:>10.4f}")
        else:
            # Fallback: sum only top-level (non-nested) phases
            lines.append(f"  {'Sum of all phases':<40} {sum(r.elapsed for r in self._records):>10.4f}")
        lines.append("  Note: phase times are nested; child phases are subsets of parent phases.")
        lines.append("=" * 60)
        lines.append("")

        report = "\n".join(lines)
        print(report)
        return report

    def save_report(self, path: str, title: str = "Mage-Flow MLX Profiler") -> None:
        """Save the timing report to a file."""
        report = self.print_report(title)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(report)

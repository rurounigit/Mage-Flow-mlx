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
    _timers: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_rss_gib() -> Optional[float]:
        """Return current process RSS in GiB, or None if unavailable."""
        try:
            import resource

            # ru_maxrss is in bytes on macOS, kilobytes on Linux
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss > 0:
                # macOS reports bytes; Linux reports KB. Detect by magnitude.
                if rss > 10_000_000_000:  # >10 GB → already in bytes
                    return rss / (1024 ** 3)
                else:
                    return rss / (1024 ** 2)  # KB → GiB
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Timing API
    # ------------------------------------------------------------------
    def start(self, name: str) -> None:
        """Start timing a named phase."""
        if not self.enabled:
            return
        self._timers[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing a named phase and record the elapsed time.

        Returns the elapsed seconds (0.0 if disabled).
        """
        if not self.enabled:
            return 0.0
        start = self._timers.pop(name, None)
        if start is None:
            return 0.0
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

        total = 0.0
        for rec in self._records:
            rss_str = f"{rec.peak_rss_gib:>14.2f}" if rec.peak_rss_gib is not None else "              N/A"
            lines.append(f"  {rec.name:<40} {rec.elapsed:>10.4f}   {rss_str}")
            total += rec.elapsed

        lines.append("  " + "-" * 64)
        lines.append(f"  {'TOTAL':<40} {total:>10.4f}")
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

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

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional


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
        """Return current process RSS in GiB, or None if unavailable.

        Note: ru_maxrss is the peak RSS for the entire process lifetime —
        it never decreases. All phases after the peak will show the same
        value.
        """
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

    def to_dict(self) -> dict:
        """Return all phase records as a JSON-serializable dictionary."""
        return {
            "phases": [
                {
                    "name": rec.name,
                    "elapsed": rec.elapsed,
                    "peak_rss_gib": rec.peak_rss_gib,
                    "metadata": dict(rec.metadata),
                }
                for rec in self._records
            ],
            "total_wall_clock": self.get_elapsed("total_wall_clock"),
        }

    def to_markdown(
        self,
        title: str = "Mage-Flow MLX Profiler",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Return the report as a markdown string with optional metadata.

        Args:
            title: Title for the profile section.
            metadata: Optional dict of metadata to render as a table at the top.

        Returns:
            Markdown-formatted report string.
        """
        if not self.enabled or not self._records:
            return ""

        lines = []

        # Metadata section
        if metadata:
            lines.append("## Metadata")
            lines.append("")
            lines.append("| Field | Value |")
            lines.append("|-------|-------|")
            for key, value in metadata.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")

        # Profile data section
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Phase | Time (s) | Peak RSS (GiB) | Metadata |")
        lines.append("|-------|----------|----------------|----------|")

        for rec in self._records:
            # total_wall_clock is the final result, not an action — no peak RSS
            if rec.name == "total_wall_clock":
                rss_str = ""
            else:
                rss_str = f"{rec.peak_rss_gib:.2f}" if rec.peak_rss_gib is not None else "N/A"
            lines.append(f"| {rec.name} | {rec.elapsed:.4f} | {rss_str} | |")
            # Each metadata key-value pair on its own line, no indentation
            if rec.metadata:
                for k, v in rec.metadata.items():
                    lines.append(f"{k}={v}")

        # Show total only if total_wall_clock is NOT already in the records
        has_total = any(r.name == "total_wall_clock" for r in self._records)
        if not has_total:
            lines.append("|-------|----------|----------------|----------|")
            lines.append(
                f"| **Sum of all phases** | **{sum(r.elapsed for r in self._records):.4f}** | | |"
            )
        lines.append("")
        lines.append("*Note: phase times are nested; child phases are subsets of parent phases.*")

        return "\n".join(lines)

    def save_metadata(
        self,
        path: str,
        metadata: Optional[dict[str, Any]] = None,
        title: str = "Mage-Flow MLX Profiler",
        prompts: Optional[list[dict]] = None,
    ) -> None:
        """Save profile data with metadata as both JSON and markdown files.

        Creates two files:
            - ``path + ".json"`` — JSON with metadata, phases, and total
            - ``path + ".md"`` — markdown with metadata table + profile table

        Args:
            path: Base path (without extension) for the output files.
            metadata: Optional dict of run-level metadata.
            title: Title for the profile section in the markdown file.
            prompts: Optional list of per-prompt metadata dicts (worker mode).
        """
        data = {
            "metadata": metadata or {},
            "phases": [
                {
                    "name": rec.name,
                    "elapsed": rec.elapsed,
                    "peak_rss_gib": rec.peak_rss_gib,
                    "metadata": dict(rec.metadata),
                }
                for rec in self._records
            ],
            "total_wall_clock": self.get_elapsed("total_wall_clock"),
        }
        if prompts:
            data["prompts"] = prompts

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Save JSON
        json_path = path + ".json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Save markdown
        md_path = path + ".md"
        md_content = self.to_markdown(title, metadata)
        with open(md_path, "w") as f:
            f.write(md_content)

    def print_report(
        self,
        title: str = "Mage-Flow MLX Profiler",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Print a formatted timing report and return it as a string.

        Args:
            title: Title for the profile section.
            metadata: Optional dict of metadata to print before the table.
        """
        if not self.enabled or not self._records:
            return ""

        lines = []

        # Metadata section
        if metadata:
            lines.append("")
            lines.append("=" * 60)
            lines.append("  Metadata")
            lines.append("=" * 60)
            for key, value in metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        lines.append("")
        lines.append("=" * 60)
        lines.append(f"  {title}")
        lines.append("=" * 60)
        lines.append(f"  {'Phase':<40} {'Time (s)':>10}   {'Peak RSS (GiB)':>14}")
        lines.append("  " + "-" * 64)

        for rec in self._records:
            # total_wall_clock is the final result, not an action — no peak RSS
            if rec.name == "total_wall_clock":
                rss_str = " " * 14
            else:
                rss_str = f"{rec.peak_rss_gib:>14.2f}" if rec.peak_rss_gib is not None else "              N/A"
            lines.append(f"  {rec.name:<40} {rec.elapsed:>10.4f}   {rss_str}")
            # Each metadata key-value pair on its own line, no indentation
            if rec.metadata:
                for k, v in rec.metadata.items():
                    lines.append(f"{k}={v}")

        # Show total only if total_wall_clock is NOT already in the records
        has_total = any(r.name == "total_wall_clock" for r in self._records)
        if not has_total:
            lines.append("  " + "-" * 64)
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

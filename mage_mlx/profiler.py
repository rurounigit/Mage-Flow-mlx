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
import re
import shutil
import textwrap
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
    saved_file: Optional[str] = None


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
    _peak_rss_gib: Optional[float] = None  # max RSS observed across all samples
    on_phase_complete: Optional[callable] = None  # callback(name, elapsed, peak_rss)

    # ── Incremental save attributes ──────────────────────────────────────
    # When metadata_path is set, the profiler writes md/json files to disk
    # after every phase completes, so a crash mid-run still leaves a partial
    # report on disk.
    metadata_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    overview: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    log_messages: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Memory helpers

    # ------------------------------------------------------------------
    @staticmethod
    def _get_process_rss_gib() -> Optional[float]:
        """Return current process RSS in GiB via `ps` (KB on macOS)."""
        try:
            import subprocess

            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                rss_kb = int(result.stdout.strip())
                return rss_kb / (1024 ** 2)  # KB → GiB
        except Exception:
            pass
        return None

    @staticmethod
    def _get_mlx_memory_gib() -> Optional[float]:
        """Return MLX active device memory in GiB.

        After Qwen unload, process RSS can drop sharply while DiT/VAE weights
        remain in Metal memory. Use active memory only (not cache) so values
        stay comparable to worker-mode generation peaks.
        """
        try:
            import mlx.core as mx

            return float(mx.get_active_memory()) / (1024 ** 3)
        except Exception:
            return None

    @classmethod
    def _get_rss_gib(cls) -> Optional[float]:
        """Return best-effort current memory usage in GiB.

        Takes the max of process RSS and MLX device memory so values stay
        meaningful both during model load (high RSS) and after text-encoder
        unload (weights mostly in Metal, low process RSS).
        """
        samples = [
            s for s in (cls._get_process_rss_gib(), cls._get_mlx_memory_gib())
            if s is not None
        ]
        return max(samples) if samples else None

    def _sample_rss_gib(self) -> Optional[float]:
        """Sample current memory and update the run-level peak."""
        rss = self._get_rss_gib()
        if rss is not None:
            if self._peak_rss_gib is None or rss > self._peak_rss_gib:
                self._peak_rss_gib = rss
        return rss

    def get_peak_rss_gib(self) -> Optional[float]:
        """Return the highest memory observed during this profiling run."""
        return self._peak_rss_gib

    def get_phase_rss(self, name: str) -> Optional[float]:
        """Return memory recorded when a named phase stopped (last match)."""
        for rec in reversed(self._records):
            if rec.name == name:
                return rec.peak_rss_gib
        return None

    def get_max_phase_rss(self, *names: str) -> Optional[float]:
        """Return max memory among phases whose names equal or start with ``names``.

        Example::
            get_max_phase_rss("generation", "text_encode", "dit_step_", "vae_decode")
        """
        best: Optional[float] = None
        for rec in self._records:
            if rec.peak_rss_gib is None:
                continue
            for name in names:
                if rec.name == name or rec.name.startswith(name):
                    if best is None or rec.peak_rss_gib > best:
                        best = rec.peak_rss_gib
                    break
        return best

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
        rss = self._sample_rss_gib() if self.track_memory else None
        self._records.append(PhaseRecord(name=name, elapsed=elapsed, peak_rss_gib=rss))
        # Real-time callback (e.g. LiveReport.start_phase/stop_phase)
        if self.on_phase_complete is not None:
            self.on_phase_complete(name, elapsed, rss)
        # Incremental save: flush to disk after every phase so a crash
        # mid-run still leaves a partial report on disk.
        if self.metadata_path is not None:
            self.save_metadata(
                self.metadata_path,
                self.metadata,
                overview=self.overview,
                summary=self.summary,
            )
        return elapsed

    def set_saved_file(self, name: str, saved_file: str) -> None:
        """Set the saved_file for a named phase record (for the green arrow)."""
        if not self.enabled:
            return
        for rec in reversed(self._records):
            if rec.name == name:
                rec.saved_file = saved_file
                return

    def log(self, message: str) -> None:
        """Append a log message to log_messages for inclusion in md/json output."""
        if not self.enabled:
            return
        self.log_messages.append(message)

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

    def get_metadata(self, name: str) -> dict[str, str]:
        """Return all metadata key-value pairs for a named phase record."""
        for rec in self._records:
            if rec.name == name:
                return dict(rec.metadata)
        return {}

    def total_elapsed(self) -> float:
        """Return total elapsed time across all recorded phases."""
        return sum(r.elapsed for r in self._records)

    def to_dict(
        self,
        metadata: Optional[dict[str, Any]] = None,
        overview: Optional[list[dict[str, Any]]] = None,
        summary: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Return all phase records as a JSON-serializable dictionary.

        Args:
            metadata: Optional run-level metadata dict.
            overview: Optional list of per-prompt overview dicts.
            summary: Optional summary dict (total_time, peak_ram, etc.).
        """
        # Round timing values in metadata to 1 decimal place
        rounded_metadata = dict(metadata or {})
        if "generation_time_seconds" in rounded_metadata and rounded_metadata["generation_time_seconds"] is not None:
            rounded_metadata["generation_time_seconds"] = round(rounded_metadata["generation_time_seconds"], 1)
        if "peak_memory_gib" in rounded_metadata and rounded_metadata["peak_memory_gib"] is not None:
            rounded_metadata["peak_memory_gib"] = round(rounded_metadata["peak_memory_gib"], 2)
        data = {
            "metadata": rounded_metadata,
            "phases": [
                {
                    "name": rec.name,
                    "elapsed": rec.elapsed,
                    "peak_rss_gib": rec.peak_rss_gib,
                    "metadata": dict(rec.metadata),
                    "saved_file": rec.saved_file,
                }
                for rec in self._records
            ],
            "total_wall_clock": self.get_elapsed("total_wall_clock"),
        }
        if overview:
            data["overview"] = overview
        if summary:
            data["summary"] = summary
        if self.log_messages:
            data["log"] = self.log_messages
        return data


    def to_markdown(
        self,
        metadata: Optional[dict[str, Any]] = None,
        overview: Optional[list[dict[str, Any]]] = None,
        summary: Optional[dict[str, Any]] = None,
    ) -> str:
        """Return the report as a markdown string matching the terminal output.

        The markdown structure mirrors the terminal output order:
        1. Phase table (with saved_file column and per-phase metadata)
        2. Summary section (total time, peak RAM, prompts count)
        3. Overview table (per-prompt results)
        4. Overhead row (and text encode/decode row for worker mode)
        5. Run Metadata block

        Args:
            metadata: Optional run-level metadata dict (for the Run Metadata block).
            overview: Optional list of per-prompt overview dicts.
            summary: Optional summary dict (total_time, peak_ram, prompts_count,
                     overhead, text_encode_time).

        Returns:
            Markdown-formatted report string.
        """
        if not self.enabled or not self._records:
            return ""

        lines = []

        # ── Log messages ─────────────────────────────────────────────────
        if self.log_messages:
            lines.append("## Log")
            lines.append("")
            for msg in self.log_messages:
                lines.append(msg)
            lines.append("")

        # ── Phase table ──────────────────────────────────────────────
        lines.append("## Phases")
        lines.append("")
        # Check if any phase has a saved_file (to decide whether to show the column)
        has_saved_files = any(rec.saved_file for rec in self._records)
        if has_saved_files:
            lines.append("| Phase | Time (s) | Peak RSS (GiB) | Saved File | Metadata |")
            lines.append("|-------|----------|----------------|------------|----------|")
        else:
            lines.append("| Phase | Time (s) | Peak RSS (GiB) | Metadata |")
            lines.append("|-------|----------|----------------|----------|")

        for rec in self._records:
            if rec.name == "total_wall_clock":
                rss_str = ""
            else:
                rss_str = f"{rec.peak_rss_gib:.2f}" if rec.peak_rss_gib is not None else "N/A"
            saved_str = rec.saved_file or ""
            if rec.metadata:
                # Embed metadata inside the table cell (comma-separated key=value pairs)
                parts = []
                for k, v in rec.metadata.items():
                    v_str = str(v)
                    parts.append(f"{k}={v_str}")
                metadata_str = ", ".join(parts)
            else:
                metadata_str = ""
            if has_saved_files:
                lines.append(f"| {rec.name} | {rec.elapsed:.1f} | {rss_str} | {saved_str} | {metadata_str} |")
            else:
                lines.append(f"| {rec.name} | {rec.elapsed:.1f} | {rss_str} | {metadata_str} |")

        has_total = any(r.name == "total_wall_clock" for r in self._records)
        if not has_total:
            if has_saved_files:
                lines.append("|-------|----------|----------------|------------|----------|")
                lines.append(
                    f"| **Sum of all phases** | **{sum(r.elapsed for r in self._records):.1f}** | | | |"
                )
            else:
                lines.append("|-------|----------|----------------|----------|")
                lines.append(
                    f"| **Sum of all phases** | **{sum(r.elapsed for r in self._records):.1f}** | | |"
                )
        lines.append("")

        # ── Summary section ──────────────────────────────────────────
        if summary:
            lines.append("## Summary")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            total_time = summary.get("total_time", 0.0)
            peak_ram = summary.get("peak_ram", 0.0)
            prompts_count = summary.get("prompts_count", 0)
            lines.append(f"| Total time | {total_time:.1f} |")
            lines.append(f"| Peak RAM | {peak_ram:.2f} |")
            lines.append(f"| Prompts | {prompts_count} |")
            lines.append("")

        # ── Overview table ──────────────────────────────────────────
        if overview:
            lines.append("## Overview")
            lines.append("")
            lines.append("| # | Time (s) | Peak RSS (GiB) | Resolution | Steps | File |")
            lines.append("|---|----------|----------------|------------|-------|------|")
            for row in overview:
                idx = row.get("index", "—")
                t = row.get("time")
                t_str = f"{t:.1f}" if t is not None else "—"
                r = row.get("peak_rss_gib")
                r_str = f"{r:.2f}" if r is not None else "—"
                res = row.get("resolution", "—")
                steps = row.get("steps", "—")
                file = row.get("file", "—")
                lines.append(f"| {idx} | {t_str} | {r_str} | {res} | {steps} | {file} |")

            # Text encode / decode row (worker mode)
            if summary and summary.get("text_encode_time", 0) > 0:
                te_time = summary["text_encode_time"]
                te_ram = summary.get("text_encode_ram")
                te_ram_str = f"{te_ram:.2f}" if te_ram is not None else "—"
                lines.append(f"| — | {te_time:.1f} | {te_ram_str} | — | — | text encode / decode |")

            # Overhead row
            if summary and summary.get("overhead", 0) > 0:
                oh = summary["overhead"]
                lines.append(f"| — | {oh:.1f} | — | — | — | overhead (load + encode + decode) |")
            lines.append("")

        # ── Run Metadata block ──────────────────────────────────────
        if metadata:
            lines.append("## Run Metadata")
            lines.append("")
            lines.append("| Field | Value |")
            lines.append("|-------|-------|")
            for key, value in metadata.items():
                if key == "generation_time_seconds" and value is not None:
                    value = round(value, 1)
                elif key == "peak_memory_gib" and value is not None:
                    value = round(value, 2)
                lines.append(f"| {key} | {value} |")
            lines.append("")

        return "\n".join(lines)

    def save_metadata(
        self,
        path: str,
        metadata: Optional[dict[str, Any]] = None,
        overview: Optional[list[dict[str, Any]]] = None,
        summary: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save profile data with metadata as both JSON and markdown files.

        Creates two files:
            - ``path + ".json"`` — JSON with metadata, phases, overview, summary
            - ``path + ".md"`` — markdown matching the terminal output structure

        Args:
            path: Base path (without extension) for the output files.
            metadata: Optional dict of run-level metadata.
            overview: Optional list of per-prompt overview dicts.
            summary: Optional summary dict (total_time, peak_ram, etc.).
        """
        data = self.to_dict(metadata, overview, summary)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Save JSON
        json_path = path + ".json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Save markdown
        md_path = path + ".md"
        md_content = self.to_markdown(metadata, overview, summary)
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
            if rec.name == "total_wall_clock":
                rss_str = " " * 14
            else:
                rss_str = f"{rec.peak_rss_gib:>14.2f}" if rec.peak_rss_gib is not None else "              N/A"
            lines.append(f"  {rec.name:<40} {rec.elapsed:>10.4f}   {rss_str}")
            if rec.metadata:
                for k, v in rec.metadata.items():
                    lines.append(f"{k}={v}")

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


# ── Live terminal report renderer ────────────────────────────────────────
# Extracted from example_realtime_report.py — keeps the exact same styling.
# Colors: times use relative green/red (min/max midpoint), RAM is cyan,
# total time is cyan. Real-time phase output has no time coloring.
class _C:
    """ANSI color codes matching the template."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    GRAY    = "\033[90m"


# ── Column widths for the live report table ────────────────────────────────
# Phase name is narrower (36 vs 42) to make room for a wider time column (14
# vs 8) that can accommodate "lazy loading" (13 chars) without pushing the
# RAM column out of alignment. Total width stays 62 chars.
_PHASE_WIDTH = 36
_TIME_WIDTH = 14
_RAM_WIDTH = 10

# Regex to strip ANSI escape codes for visible-width calculations
_ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def _visible_len(s: str) -> int:
    """Return the visible length of a string, excluding ANSI escape codes."""
    return len(_ANSI_RE.sub('', s))


def _ansi_rjust(s: str, width: int, color: str = "") -> str:
    """Right-justify visible text to *width*, then optionally wrap in ANSI color.

    Unlike Python's built-in ``{:>width}`` format, this function only counts
    visible characters when calculating padding, so ANSI color codes don't
    push subsequent columns out of alignment.
    """
    pad = max(0, width - _visible_len(s))
    result = ' ' * pad + s
    if color:
        result = f"{color}{result}{_C.RESET}"
    return result


def _fmt_time(s: float) -> str:
    """Format seconds with one decimal place."""
    return f"{s:.1f}s"


def _colorize_time(s: float, min_val: float = 0.0, max_val: float = 0.0) -> str:
    """Color-code time relative to min/max of all phase times.

    Green = below midpoint, red = above midpoint.
    If min_val == max_val (no range), uses yellow.
    """
    ts = _fmt_time(s)
    if max_val <= min_val:
        return f"{_C.YELLOW}{ts}{_C.RESET}"
    midpoint = (min_val + max_val) / 2
    if s <= midpoint:
        return f"{_C.GREEN}{ts}{_C.RESET}"
    else:
        return f"{_C.RED}{ts}{_C.RESET}"


def _colorize_ram(gib: float) -> str:
    """Color-code peak RAM with a neutral color (cyan)."""
    rs = f"{gib:.2f}GiB"
    return f"{_C.CYAN}{rs}{_C.RESET}"


def _colorize_total(s: float) -> str:
    """Color-code total wall time with the same neutral color as RAM (cyan)."""
    ts = _fmt_time(s)
    return f"{_C.CYAN}{ts}{_C.RESET}"


@dataclass
class _PhaseRow:
    """A single phase row in the live report."""
    name: str
    elapsed: Optional[float] = None
    peak_rss_gib: Optional[float] = None
    metadata: dict[str, str] = field(default_factory=dict)
    saved_file: Optional[str] = None


@dataclass
class _PromptRow:
    """Per-prompt summary row."""
    index: int
    prompt: str
    resolution: str
    steps: int
    quantize: Optional[int]
    generation_time: Optional[float]
    peak_rss_gib: Optional[float]
    saved_file: Optional[str]


class LiveReport:
    """Real-time terminal report renderer for the profiler.

    Uses the same styling as the prototype template:
    - 70-char wide cyan bold separators
    - Dim column headers
    - Magenta bold prompt headers
    - Green arrow (->) for saved files
    - Relative green/red coloring for times in the summary
    - Cyan for RAM and total time
    - No redundant table — real-time output IS the table
    - Summary section with per-prompt results table
    - Run Metadata block at the end

    Real-time phase output uses plain text (no time coloring) because
    min/max are not known until all phases complete.
    """

    def __init__(self, title: str = "Mage-Flow MLX", verbose: bool = True, profiler=None):
        self.title = title
        self.verbose = verbose
        self.phases: list[_PhaseRow] = []
        self.prompts: list[_PromptRow] = []
        self._phase_times: list[float] = []  # for relative color scaling
        self.profiler = profiler  # reference for incremental saves
        self._print_header()

    # ── header ──
    def _print_header(self) -> None:
        print()
        print(f"{_C.BOLD}{_C.CYAN}{'=' * 70}{_C.RESET}")
        print(f"{_C.BOLD}{_C.CYAN}  {self.title}{_C.RESET}")
        print(f"{_C.BOLD}{_C.CYAN}{'=' * 70}{_C.RESET}")
        if self.verbose:
            print(f"{_C.DIM}  {'Phase':<36} {'Time':>14}   {'Peak RAM':>10}{_C.RESET}")
            print(f"{_C.DIM}  {'─' * 62}{_C.RESET}")
        # Capture header in profiler log for md/json output
        if self.profiler is not None:
            self.profiler.log("=" * 70)
            self.profiler.log(f"  {self.title}")
            self.profiler.log("=" * 70)

    # ── progress bar (non-verbose mode) ──
    def progress_bar(self, name: str) -> None:
        """Print a single in-place progress bar using carriage return.

        Overwrites itself on the same line using \\033[K to clear line.
        """
        count = len(self.phases)
        bar = '█' * count
        print(f"\r\033[K  [{bar}] {count} events — {name}", end="", flush=True)

    # ── phase lifecycle ──
    def start_phase(self, name: str) -> None:
        """Called when a phase starts — prints a live indicator."""
        row = _PhaseRow(name=name)
        self.phases.append(row)
        if self.verbose:
            print(f"{_C.GRAY}  ▸ {name:<34}{_C.RESET}", end="", flush=True)

    def stop_phase(
        self,
        name: str,
        elapsed: float,
        peak_rss_gib: Optional[float] = None,
        saved_file: Optional[str] = None,
        loading_mode: Optional[str] = None,
        grey_separator: bool = False,
    ) -> None:
        """Called when a phase completes — prints timing + RAM.

        Args:
            name: Phase name.
            elapsed: Elapsed seconds (ignored if loading_mode is set).
            peak_rss_gib: Peak memory in GiB.
            saved_file: Optional saved file path (shown with green arrow).
            loading_mode: If set (e.g. "lazy"), displayed instead of the
                time string. Used for phases where the actual work happens
                later (e.g. text_encoder_load where weights are lazy-loaded).
            grey_separator: If True, print a grey separator line instead of
                an empty line before the phase. Used to visually group
                phases (e.g. above pipeline_load in worker mode).
        """
        row = None
        for r in reversed(self.phases):
            if r.name == name and r.elapsed is None:
                row = r
                break
        if row is None:
            row = _PhaseRow(name=name)
            self.phases.append(row)

        row.elapsed = elapsed
        row.peak_rss_gib = peak_rss_gib
        row.saved_file = saved_file

        if elapsed is not None:
            self._phase_times.append(elapsed)

        # Sync saved_file to profiler record for md/json output
        if self.profiler is not None and saved_file is not None:
            self.profiler.set_saved_file(name, saved_file)

        if not self.verbose:
            self.progress_bar(name)
            return

        # Real-time: no time coloring (don't know min/max yet)
        if loading_mode is not None:
            time_str = _ansi_rjust(loading_mode, _TIME_WIDTH, _C.YELLOW)
        elif elapsed is not None:
            time_str = _ansi_rjust(_fmt_time(elapsed), _TIME_WIDTH)
        else:
            time_str = _ansi_rjust("—", _TIME_WIDTH, _C.GRAY)
        if peak_rss_gib is not None:
            ram_str = _ansi_rjust(f"{peak_rss_gib:.2f}GiB", _RAM_WIDTH, _C.CYAN)
        else:
            ram_str = _ansi_rjust("—", _RAM_WIDTH, _C.GRAY)

        # Print empty line before phase for visual separation
        # Block phases form a group with no empty lines between them:
        # - Pipeline loading phases: dit_load, vae_load, text_encoder_load
        # - Generation steps: text_encode, dit_step_N, edit_step_N, vae_decode
        _BLOCK_PHASES = {"dit_load", "vae_load", "text_encoder_load", "text_encode"}
        def _is_block(name):
            return name in _BLOCK_PHASES or name.startswith("dit_step_") or name.startswith("edit_step_")

        is_block_phase = _is_block(name)
        prev = self.phases[-1] if self.phases else None
        prev_was_block = prev and prev.elapsed is not None and _is_block(prev.name)

        should_print_separator = False
        if is_block_phase:
            if not prev_was_block:
                should_print_separator = True
        else:
            should_print_separator = True

        if should_print_separator:
            if grey_separator:
                print(f"{_C.DIM}  {'─' * 62}{_C.RESET}")
            else:
                print()

        print(f"  {name:<36} {time_str}   {ram_str}", end="")
        if saved_file:
            print(f"   {_C.GREEN}→ {saved_file}{_C.RESET}")
        else:
            print()

    def add_saved_file(self, name: str, saved_file: str) -> None:
        """Add a saved_file to an already-reported phase row (e.g. for the green arrow)."""
        for r in reversed(self.phases):
            if r.name == name and r.elapsed is not None:
                r.saved_file = saved_file
                return

    # ── bulk report from profiler records ──
    def report_profiler_phases(self, profiler: "Profiler", exclude: Optional[str] = None) -> None:
        """Report all phases from a Profiler that haven't been reported yet.

        Args:
            profiler: The Profiler instance to read records from.
            exclude: Phase name to skip (e.g. the parent phase that will be
                     reported separately via stop_phase).
        """
        reported = {r.name for r in self.phases if r.elapsed is not None}
        if exclude:
            reported.add(exclude)
        for rec in profiler.get_records():
            if rec.name in reported:
                continue
            self.stop_phase(rec.name, rec.elapsed or 0.0, rec.peak_rss_gib)
            for key, value in profiler.get_metadata(rec.name).items():
                self.add_metadata(rec.name, key, value)
            reported.add(rec.name)

    # ── metadata ──
    def add_metadata(self, phase_name: str, key: str, value: str) -> None:
        """Attach metadata to a phase — printed with 2-space indent, yellow key."""
        row = None
        for r in reversed(self.phases):
            if r.name == phase_name:
                row = r
                break
        if row is None:
            row = _PhaseRow(name=phase_name)
            self.phases.append(row)
        row.metadata[key] = value

        prefix = f"  {_C.YELLOW}{key}{_C.RESET}:"
        visible_prefix = f"  {key}:"
        indent_len = len(visible_prefix)

        try:
            term_width = shutil.get_terminal_size().columns
        except Exception:
            term_width = 80

        wrap_width = max(20, term_width - indent_len)
        wrapped_lines = textwrap.wrap(value, width=wrap_width)

        if wrapped_lines:
            print(f"{prefix}{wrapped_lines[0]}")
            for line in wrapped_lines[1:]:
                print(f"{' ' * indent_len}{line}")
        else:
            print(prefix)

    # ── prompt header ──
    def prompt_header(self, index: int, total: int) -> None:
        """Print a prompt header (magenta bold)."""
        print()
        print(f"{_C.BOLD}{_C.MAGENTA}  Prompt {index}/{total}{_C.RESET}")
        print(f"{_C.DIM}  {'─' * 62}{_C.RESET}")

    # ── prompt summary ──
    def add_prompt(
        self,
        index: int,
        prompt: str,
        resolution: str,
        steps: int,
        quantize: Optional[int],
        generation_time: Optional[float],
        peak_rss_gib: Optional[float],
        saved_file: Optional[str],
    ) -> None:
        """Add a per-prompt summary row and trigger incremental save."""
        pr = _PromptRow(
            index=index,
            prompt=prompt,
            resolution=resolution,
            steps=steps,
            quantize=quantize,
            generation_time=generation_time,
            peak_rss_gib=peak_rss_gib,
            saved_file=saved_file,
        )
        self.prompts.append(pr)

        # Update profiler overview and trigger incremental save
        if self.profiler is not None and self.profiler.metadata_path is not None:
            self.profiler.overview = [
                {
                    "index": p.index,
                    "time": p.generation_time,
                    "peak_rss_gib": p.peak_rss_gib,
                    "resolution": p.resolution,
                    "steps": p.steps,
                    "file": p.saved_file,
                }
                for p in self.prompts
            ]
            self.profiler.save_metadata(
                self.profiler.metadata_path,
                self.profiler.metadata,
                overview=self.profiler.overview,
                summary=self.profiler.summary,
            )

    # ── final summary ──
    def print_summary(self, total_time: float, peak_ram: float, show_text_encode: bool = False) -> None:
        """Print the final summary section with per-prompt results table.

        Args:
            total_time: Total wall clock time in seconds.
            peak_ram: Peak RAM in GiB.
            show_text_encode: If True (worker mode), show a separate
                "text encode / decode" row summing text_encode,
                text_encoder_unload, dit_load, and vae_load phases.
                The overhead row then reflects only the leftover
                (startup + save) time.
        """
        if not self.verbose:
            print()  # Final newline to complete the \r progress bar
        print()
        print(f"{_C.BOLD}{_C.CYAN}{'─' * 70}{_C.RESET}")
        print(f"{_C.BOLD}  Summary{_C.RESET}")
        print(f"{_C.BOLD}{_C.CYAN}{'─' * 70}{_C.RESET}")

        total_str = _colorize_total(total_time)
        ram_str = _colorize_ram(peak_ram)
        print(f"  {_C.BOLD}Total time:{_C.RESET}     {total_str}")
        print(f"  {_C.BOLD}Peak RAM:{_C.RESET}       {ram_str}")
        print(f"  {_C.BOLD}Prompts:{_C.RESET}        {len(self.prompts)}")
        print()

        # Overview table
        if self.prompts:
            print(f"{_C.BOLD}  Overview:{_C.RESET}")
            print(f"{_C.DIM}  {'#':>3}  {'Time':>8}   {'Peak RAM':>10}   {'Resolution':>12}   {'Steps':>5}   File{_C.RESET}")
            print(f"{_C.DIM}  {'─' * 62}{_C.RESET}")
            gen_times = [p.generation_time for p in self.prompts if p.generation_time is not None]
            if gen_times:
                t_min = min(gen_times)
                t_max = max(gen_times)
            else:
                t_min = t_max = 0.0
            for p in self.prompts:
                if p.generation_time is not None:
                    t_str = _colorize_time(p.generation_time, t_min, t_max)
                else:
                    t_str = f"{_C.GRAY}—{_C.RESET}"
                r_str = _colorize_ram(p.peak_rss_gib) if p.peak_rss_gib else f"{_C.GRAY}—{_C.RESET}"
                file_str = f"{_C.GREEN}{p.saved_file}{_C.RESET}" if p.saved_file else f"{_C.GRAY}—{_C.RESET}"
                print(
                    f"  {p.index:>3}  {_ansi_rjust(t_str, 8)}   {_ansi_rjust(r_str, 10)}   "
                    f"{p.resolution:>12}   {p.steps:>5}   {file_str}"
                )

            sum_gen = sum(p.generation_time for p in self.prompts if p.generation_time is not None)
            text_encode_time = 0.0
            text_encode_ram: Optional[float] = None
            if show_text_encode:
                for ph in self.phases:
                    if ph.elapsed is None:
                        continue
                    if (
                        ph.name.startswith("text_encode")
                        or ph.name == "text_encoder_unload"
                        or ph.name == "dit_load"
                        or ph.name == "vae_load"
                    ):
                        text_encode_time += ph.elapsed
                        if ph.peak_rss_gib is not None:
                            if text_encode_ram is None or ph.peak_rss_gib > text_encode_ram:
                                text_encode_ram = ph.peak_rss_gib
                if text_encode_time > 0:
                    print(f"{_C.DIM}  {'─' * 62}{_C.RESET}")
                    te_str = _colorize_total(text_encode_time)
                    if text_encode_ram is not None:
                        te_ram_str = _colorize_ram(text_encode_ram)
                    else:
                        te_ram_str = f"{_C.GRAY}—{_C.RESET}"
                    print(
                        f"  {'—':>3}  {_ansi_rjust(te_str, 8)}   {_ansi_rjust(te_ram_str, 10)}   "
                        f"{'—':>12}   {'—':>5}   {_C.GRAY}text encode / decode{_C.RESET}"
                    )

            overhead = total_time - sum_gen
            if show_text_encode:
                overhead -= text_encode_time
            if overhead > 0:
                print(f"{_C.DIM}  {'─' * 62}{_C.RESET}")
                oh_str = _colorize_total(overhead)
                print(
                    f"  {'—':>3}  {_ansi_rjust(oh_str, 8)}   {_ansi_rjust('—', 10)}   "
                    f"{'—':>12}   {'—':>5}   {_C.GRAY}overhead (load + encode + decode){_C.RESET}"
                )
            print()

        print(f"{_C.BOLD}{_C.CYAN}{'=' * 70}{_C.RESET}")
        print()

    # ── run metadata block ──
    def print_run_metadata(self, metadata: dict[str, Any]) -> None:
        """Print the run metadata block at the end."""
        print(f"{_C.BOLD}  Run Metadata{_C.RESET}")
        print(f"{_C.DIM}  {'─' * 62}{_C.RESET}")
        for key, value in metadata.items():
            if key == "generation_time_seconds":
                print(f"  {key}: {_colorize_total(value)}")
            elif key == "peak_memory_gib" and value is not None:
                print(f"  {key}: {_colorize_ram(value)}")
            else:
                print(f"  {key}: {value}")
        print()

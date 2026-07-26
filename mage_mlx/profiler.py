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
    on_phase_complete: Optional[callable] = None  # callback(name, elapsed, peak_rss)

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
        # Real-time callback (e.g. LiveReport.start_phase/stop_phase)
        if self.on_phase_complete is not None:
            self.on_phase_complete(name, elapsed, rss)
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

    def get_metadata(self, name: str) -> dict[str, str]:
        """Return all metadata key-value pairs for a named phase record."""
        for rec in self._records:
            if rec.name == name:
                return dict(rec.metadata)
        return {}

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
    - Green arrow (→) for saved files
    - Relative green/red coloring for times in the summary
    - Cyan for RAM and total time
    - No redundant table — real-time output IS the table
    - Summary section with per-prompt results table
    - Run Metadata block at the end

    Real-time phase output uses plain text (no time coloring) because
    min/max are not known until all phases complete.
    """

    def __init__(self, title: str = "Mage-Flow MLX"):
        self.title = title
        self.phases: list[_PhaseRow] = []
        self.prompts: list[_PromptRow] = []
        self._phase_times: list[float] = []  # for relative color scaling
        self._print_header()

    # ── header ──
    def _print_header(self) -> None:
        print()
        print(f"{_C.BOLD}{_C.CYAN}{'=' * 70}{_C.RESET}")
        print(f"{_C.BOLD}{_C.CYAN}  {self.title}{_C.RESET}")
        print(f"{_C.BOLD}{_C.CYAN}{'=' * 70}{_C.RESET}")
        print(f"{_C.DIM}  {'Phase':<42} {'Time':>8}   {'Peak RAM':>10}{_C.RESET}")
        print(f"{_C.DIM}  {'─' * 62}{_C.RESET}")

    # ── phase lifecycle ──
    def start_phase(self, name: str) -> None:
        """Called when a phase starts — prints a live indicator."""
        row = _PhaseRow(name=name)
        self.phases.append(row)
        print(f"{_C.GRAY}  ▸ {name:<40}{_C.RESET}", end="", flush=True)

    def stop_phase(
        self,
        name: str,
        elapsed: float,
        peak_rss_gib: Optional[float] = None,
        saved_file: Optional[str] = None,
    ) -> None:
        """Called when a phase completes — prints timing + RAM."""
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

        # Real-time: no time coloring (don't know min/max yet)
        time_str = _fmt_time(elapsed) if elapsed is not None else f"{_C.GRAY}—{_C.RESET}"
        ram_str = _colorize_ram(peak_rss_gib) if peak_rss_gib is not None else f"{_C.GRAY}—{_C.RESET}"

        # Print empty line before phase for visual separation
        # Block phases form a group with no empty lines between them:
        # - Pipeline loading phases: dit_load, vae_load, text_encoder_load
        # - Generation steps: text_encode, dit_step_N, edit_step_N, vae_decode
        _BLOCK_PHASES = {"dit_load", "vae_load", "text_encoder_load", "text_encode", "vae_decode"}
        def _is_block(name):
            return name in _BLOCK_PHASES or name.startswith("dit_step_") or name.startswith("edit_step_")

        is_block_phase = _is_block(name)
        prev = self.phases[-1] if self.phases else None
        prev_was_block = prev and prev.elapsed is not None and _is_block(prev.name)

        if is_block_phase:
            # Empty line before first block phase (if previous was not a block phase)
            if not prev_was_block:
                print()
        else:
            # Empty line before non-block phases (always, gives separation after blocks)
            print()

        print(f"  {name:<42} {time_str:>8}   {ram_str:>10}", end="")
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
            # Report any metadata set on this phase
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

        # Build the prefix: "  key:" with ANSI color codes
        prefix = f"  {_C.YELLOW}{key}{_C.RESET}:"
        # Visible prefix length (without ANSI codes) for indentation
        visible_prefix = f"  {key}:"
        indent_len = len(visible_prefix)

        # Get terminal width, fall back to 80 if unavailable
        try:
            term_width = shutil.get_terminal_size().columns
        except Exception:
            term_width = 80

        # Wrap the value to fit within the terminal width
        wrap_width = max(20, term_width - indent_len)
        wrapped_lines = textwrap.wrap(value, width=wrap_width)

        if wrapped_lines:
            # First line with the prefix
            print(f"{prefix}{wrapped_lines[0]}")
            # Subsequent lines indented to match the prefix
            for line in wrapped_lines[1:]:
                print(f"{' ' * indent_len}{line}")
        else:
            # Empty value
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
        """Add a per-prompt summary row."""
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

    # ── final summary ──
    def print_summary(self, total_time: float, peak_ram: float) -> None:
        """Print the final summary section with per-prompt results table."""
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

        # Per-prompt table
        if self.prompts:
            print(f"{_C.BOLD}  Per-Prompt Results:{_C.RESET}")
            print(f"{_C.DIM}  {'#':>3}  {'Time':>8}   {'Peak RAM':>10}   {'Resolution':>12}   {'Steps':>5}   File{_C.RESET}")
            print(f"{_C.DIM}  {'─' * 62}{_C.RESET}")
            # Use generation times only for relative coloring
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
                    f"  {p.index:>3}  {t_str:>8}   {r_str:>10}   "
                    f"{p.resolution:>12}   {p.steps:>5}   {file_str}"
                )
            print()

        # No redundant phase timings table — already shown in real-time

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
            elif key == "peak_memory_gib":
                print(f"  {key}: {_colorize_ram(value)}")
            else:
                print(f"  {key}: {value}")
        print()

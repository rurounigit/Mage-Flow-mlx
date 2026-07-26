#!/usr/bin/env python3
"""Prototype: integrated real-time colored profiler report.

This is a standalone example showing what the terminal output could look like
if the profiler table were integrated with real-time generation output, with
colors highlighting key metrics (peak RAM, generation time, etc.).

Run:  .venv/bin/python example_realtime_report.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ── ANSI colors ──────────────────────────────────────────────────────────
class C:
    """Minimal ANSI color helpers."""
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

    # backgrounds
    BG_RED     = "\033[41m"
    BG_GREEN   = "\033[42m"
    BG_YELLOW  = "\033[43m"
    BG_BLUE    = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_GRAY    = "\033[100m"


def fmt_time(s: float) -> str:
    """Format seconds with one decimal place."""
    return f"{s:.1f}s"


def colorize_time(s: float, min_val: float = 0.0, max_val: float = 0.0) -> str:
    """Color-code time relative to min/max of all phase times.

    Green = below midpoint, red = above midpoint.
    If min_val == max_val (no range), uses a default yellow.
    """
    ts = fmt_time(s)
    if max_val <= min_val:
        return f"{C.YELLOW}{ts}{C.RESET}"
    midpoint = (min_val + max_val) / 2
    if s <= midpoint:
        return f"{C.GREEN}{ts}{C.RESET}"
    else:
        return f"{C.RED}{ts}{C.RESET}"


def colorize_ram(gib: float) -> str:
    """Color-code peak RAM with a neutral color (cyan).

    RAM pressure is hard to measure precisely, so all values use the same
    color for consistency.
    """
    rs = f"{gib:.2f}GiB"
    return f"{C.CYAN}{rs}{C.RESET}"


def colorize_total(s: float) -> str:
    """Color-code total wall time with the same neutral color as RAM (cyan)."""
    ts = fmt_time(s)
    return f"{C.CYAN}{ts}{C.RESET}"


# ── Data structures ──────────────────────────────────────────────────────
@dataclass
class PhaseRow:
    """A single phase row in the live report."""
    name: str
    elapsed: Optional[float] = None
    peak_rss_gib: Optional[float] = None
    metadata: dict[str, str] = field(default_factory=dict)
    saved_file: Optional[str] = None


@dataclass
class PromptRow:
    """Per-prompt summary row."""
    index: int
    prompt: str
    resolution: str
    steps: int
    quantize: Optional[int]
    generation_time: Optional[float]
    peak_rss_gib: Optional[float]
    saved_file: Optional[str]


# ── Live report renderer ─────────────────────────────────────────────────
class LiveReport:
    """Renders a live, colorized profiler report to the terminal."""

    def __init__(self, title: str = "Mage-Flow MLX"):
        self.title = title
        self.phases: list[PhaseRow] = []
        self.prompts: list[PromptRow] = []
        self._phase_times: list[float] = []  # for relative color scaling
        self._print_header()

    # ── header ──
    def _print_header(self) -> None:
        print()
        print(f"{C.BOLD}{C.CYAN}{'=' * 70}{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}  {self.title}{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}{'=' * 70}{C.RESET}")
        print(f"{C.DIM}  {'Phase':<42} {'Time':>8}   {'Peak RAM':>10}{C.RESET}")
        print(f"{C.DIM}  {'─' * 62}{C.RESET}")

    # ── phase lifecycle ──
    def start_phase(self, name: str) -> None:
        """Called when a phase starts — prints a live indicator."""
        row = PhaseRow(name=name)
        self.phases.append(row)
        # Show the phase name with a spinner-like indicator
        print(f"{C.GRAY}  ▸ {name:<40}{C.RESET}", end="", flush=True)

    def stop_phase(
        self,
        name: str,
        elapsed: float,
        peak_rss_gib: Optional[float] = None,
        saved_file: Optional[str] = None,
    ) -> None:
        """Called when a phase completes — prints timing + RAM."""
        # Find the row
        row = None
        for r in reversed(self.phases):
            if r.name == name and r.elapsed is None:
                row = r
                break
        if row is None:
            row = PhaseRow(name=name)
            self.phases.append(row)

        row.elapsed = elapsed
        row.peak_rss_gib = peak_rss_gib
        row.saved_file = saved_file

        # Track phase times for relative color scaling (used in summary)
        if elapsed is not None:
            self._phase_times.append(elapsed)

        # During real-time output, don't color times (we don't know min/max yet)
        if elapsed is not None:
            time_str = f"{fmt_time(elapsed)}"
        else:
            time_str = f"{C.GRAY}—{C.RESET}"
        ram_str = colorize_ram(peak_rss_gib) if peak_rss_gib is not None else f"{C.GRAY}—{C.RESET}"

        # Move to next line (simulating the live update)
        print()  # newline after the ▸ indicator
        print(f"  {name:<42} {time_str:>8}   {ram_str:>10}", end="")
        if saved_file:
            print(f"   {C.GREEN}→ {saved_file}{C.RESET}")
        else:
            print()

    # ── metadata ──
    def add_metadata(self, phase_name: str, key: str, value: str) -> None:
        """Attach metadata to a phase — printed with 2-space indent matching table."""
        # Find existing row or create one
        row = None
        for r in reversed(self.phases):
            if r.name == phase_name:
                row = r
                break
        if row is None:
            row = PhaseRow(name=phase_name)
            self.phases.append(row)
        row.metadata[key] = value
        # Print immediately with 2-space indentation (matching table)
        # Yellow colon, white value
        print(f"  {C.YELLOW}{key}{C.RESET}:{value}")

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
        pr = PromptRow(
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
        """Print the final summary section."""
        print()
        print(f"{C.BOLD}{C.CYAN}{'─' * 70}{C.RESET}")
        print(f"{C.BOLD}  Summary{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}{'─' * 70}{C.RESET}")

        # Total time — neutral color (same as RAM)
        total_str = colorize_total(total_time)
        ram_str = colorize_ram(peak_ram)
        print(f"  {C.BOLD}Total time:{C.RESET}     {total_str}")
        print(f"  {C.BOLD}Peak RAM:{C.RESET}       {ram_str}")
        print(f"  {C.BOLD}Prompts:{C.RESET}        {len(self.prompts)}")
        print()

        # Per-prompt table
        if self.prompts:
            print(f"{C.BOLD}  Per-Prompt Results:{C.RESET}")
            print(f"{C.DIM}  {'#':>3}  {'Time':>8}   {'Peak RAM':>10}   {'Resolution':>12}   {'Steps':>5}   File{C.RESET}")
            print(f"{C.DIM}  {'─' * 62}{C.RESET}")
            for p in self.prompts:
                # Generation times use relative coloring (same as phase times)
                if p.generation_time and self._phase_times:
                    t_min = min(self._phase_times)
                    t_max = max(self._phase_times)
                    t_str = colorize_time(p.generation_time, t_min, t_max)
                else:
                    t_str = f"{C.GRAY}—{C.RESET}"
                r_str = colorize_ram(p.peak_rss_gib) if p.peak_rss_gib else f"{C.GRAY}—{C.RESET}"
                file_str = f"{C.GREEN}{p.saved_file}{C.RESET}" if p.saved_file else f"{C.GRAY}—{C.RESET}"
                print(
                    f"  {p.index:>3}  {t_str:>8}   {r_str:>10}   "
                    f"{p.resolution:>12}   {p.steps:>5}   {file_str}"
                )
            print()

        # Phase timings are already shown in real-time during generation,
        # so no need to repeat them here.

        print(f"{C.BOLD}{C.CYAN}{'=' * 70}{C.RESET}")
        print()


# ── Demo simulation ──────────────────────────────────────────────────────
def demo() -> None:
    """Simulate a worker-mode run with 2 prompts."""
    report = LiveReport("Mage-Flow MLX — Worker Mode")

    # Phase: pipeline load
    report.start_phase("pipeline_reload")
    time.sleep(0.1)
    report.stop_phase("pipeline_reload", elapsed=2.21, peak_rss_gib=0.44)

    # Phase: text encoding
    report.start_phase("text_encode_1")
    time.sleep(0.1)
    report.stop_phase("text_encode_1", elapsed=1.37, peak_rss_gib=7.71)

    report.start_phase("text_encode_2")
    time.sleep(0.1)
    report.stop_phase("text_encode_2", elapsed=0.18, peak_rss_gib=7.72)

    report.start_phase("text_encoder_unload")
    time.sleep(0.1)
    report.stop_phase("text_encoder_unload", elapsed=0.09, peak_rss_gib=7.72)

    # ── Prompt 1 ──
    print()
    print(f"{C.BOLD}{C.MAGENTA}  Prompt 1/2{C.RESET}")
    print(f"{C.DIM}  {'─' * 62}{C.RESET}")

    # Metadata (printed before generation starts, with 2-space indent)
    report.add_metadata("generation_1", "prompt", "black and white pencil drawing on rough paper in the style of monet, a steampunk airship battling a mechanical dragon over a Victorian city, copper and brass details, dramatic storm lighting.")
    report.add_metadata("generation_1", "resolution", "1024x1024")
    report.add_metadata("generation_1", "steps", "4")
    report.add_metadata("generation_1", "quantize", "None")

    report.start_phase("generation_1")
    # Simulate DiT steps
    for step in range(1, 5):
        report.start_phase(f"  dit_step_{step}")
        time.sleep(0.05)
        report.stop_phase(f"  dit_step_{step}", elapsed=3.14, peak_rss_gib=7.72)
    report.start_phase("  vae_decode")
    time.sleep(0.05)
    report.stop_phase("  vae_decode", elapsed=0.001, peak_rss_gib=7.72)
    report.stop_phase("generation_1", elapsed=14.48, peak_rss_gib=7.72)

    report.start_phase("save_1")
    time.sleep(0.05)
    report.stop_phase("save_1", elapsed=0.053, peak_rss_gib=7.72, saved_file="test_01_airship_dragon_new.png")

    report.add_prompt(
        index=1,
        prompt="black and white pencil drawing...steampunk airship...",
        resolution="1024x1024",
        steps=4,
        quantize=None,
        generation_time=14.48,
        peak_rss_gib=7.72,
        saved_file="test_01_airship_dragon_new.png",
    )

    # ── Prompt 2 ──
    print()
    print(f"{C.BOLD}{C.MAGENTA}  Prompt 2/2{C.RESET}")
    print(f"{C.DIM}  {'─' * 62}{C.RESET}")

    # Metadata (printed before generation starts, with 2-space indent)
    report.add_metadata("generation_2", "prompt", "Anime shonen style, an astronaut riding a cosmic wolf through a nebula, bioluminescent fur, star trails, ethereal purple and blue lighting.")
    report.add_metadata("generation_2", "resolution", "1024x1024")
    report.add_metadata("generation_2", "steps", "4")
    report.add_metadata("generation_2", "quantize", "None")

    report.start_phase("generation_2")
    for step in range(1, 5):
        report.start_phase(f"  dit_step_{step}")
        time.sleep(0.05)
        report.stop_phase(f"  dit_step_{step}", elapsed=3.18, peak_rss_gib=7.72)
    report.start_phase("  vae_decode")
    time.sleep(0.05)
    report.stop_phase("  vae_decode", elapsed=0.001, peak_rss_gib=7.72)
    report.stop_phase("generation_2", elapsed=14.92, peak_rss_gib=7.72)

    report.start_phase("save_2")
    time.sleep(0.05)
    report.stop_phase("save_2", elapsed=0.063, peak_rss_gib=7.72, saved_file="test_02_cosmic_wolf_new.png")

    report.add_prompt(
        index=2,
        prompt="Anime shonen style...cosmic wolf...",
        resolution="1024x1024",
        steps=4,
        quantize=None,
        generation_time=14.92,
        peak_rss_gib=7.72,
        saved_file="test_02_cosmic_wolf_new.png",
    )

    # ── Final summary ──
    report.print_summary(total_time=33.4, peak_ram=7.72)

    # Metadata block (like current output, but at the end)
    print(f"{C.BOLD}  Run Metadata{C.RESET}")
    print(f"{C.DIM}  {'─' * 62}{C.RESET}")
    print(f"  model: microsoft/Mage-Flow-Turbo")
    print(f"  base_model: MageFlow")
    print(f"  generation_time_seconds: {colorize_total(33.4)}")
    print(f"  created_at: 2026-07-26T14:18:02")
    print(f"  peak_memory_gib: {colorize_ram(7.72)}")
    print()


if __name__ == "__main__":
    demo()

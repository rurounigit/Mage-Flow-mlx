"""End-to-end test that runs all generate.py prompts sequentially.

This test actually invokes the full Mage-Flow MLX pipeline (loading real model
weights, generating images on the GPU) via subprocess — exactly as a user would
from the terminal.  Each prompt is run one at a time, waiting for it to finish
before starting the next.

After all prompts have run, the test checks for problems in:
  - Terminal output (exit codes, error messages, warnings)
  - Generated .json metadata files (valid JSON, expected fields)
  - Generated .md metadata files (valid markdown, expected sections)
  - Generated .png output files (exist, non-zero size)

The test is marked with a long timeout since each generation takes ~30-60s.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Project root (conftest.py inserts it into sys.path, but be explicit)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
GENERATE = str(PROJECT_ROOT / "generate.py")

# The shoe prompt shared by prompts 1-4
SHOE_PROMPT = (
    "An unbranded futuristic running shoe made from white technical mesh "
    "with a vivid orange sole, floating above a pale gray studio surface, "
    "dramatic softbox lighting, premium product photography, photorealistic."
)

# The edit prompt shared by prompts 3-4
EDIT_PROMPT = (
    "change the shoe to deep burgundy polished leather with a translucent "
    "smoke-gray sole and subtle chrome details. Preserve the exact silhouette, "
    "panel seams, laces, camera angle, floating pose, lighting, shadows, "
    "and background."
)

# All 8 prompts to run, in order.  Each entry is a dict with:
#   - name: human-readable label
#   - args: list of CLI arguments (without the python/generate.py prefix)
#   - metadata: whether --metadata is expected (so we know which .json/.md to check)
PROMPTS = [
    {
        "name": "1_single_txt2img_with_metadata",
        "args": [
            "--prompt", SHOE_PROMPT,
            "--width", "1024",
            "--height", "1024",
            "--output", "test_10_shoe.png",
            "--metadata",
        ],
        "metadata": True,
    },
    {
        "name": "2_single_txt2img_no_metadata",
        "args": [
            "--prompt", SHOE_PROMPT,
            "--width", "1024",
            "--height", "1024",
        ],
        "metadata": False,
    },
    {
        "name": "3_single_edit_with_metadata",
        "args": [
            "--prompt", EDIT_PROMPT,
            "--image", "test_10_shoe.png",
            "--output", "test_10_shoe_edited.png",
            "--metadata",
        ],
        "metadata": True,
    },
    {
        "name": "4_single_edit_no_metadata",
        "args": [
            "--prompt", EDIT_PROMPT,
            "--image", "test_10_shoe.png",
        ],
        "metadata": False,
    },
    {
        "name": "5_edit_worker_with_metadata",
        "args": [
            "--worker", "test_prompts_edit.jsonl",
            "--metadata",
            "--edit",
        ],
        "metadata": True,
    },
    {
        "name": "6_txt2img_worker_with_metadata",
        "args": [
            "--worker", "test_prompts.jsonl",
            "--metadata",
        ],
        "metadata": True,
    },
    {
        "name": "7_edit_worker_no_metadata",
        "args": [
            "--worker", "test_prompts_edit.jsonl",
            "--edit",
        ],
        "metadata": False,
    },
    {
        "name": "8_txt2img_worker_no_metadata",
        "args": [
            "--worker", "test_prompts.jsonl",
        ],
        "metadata": False,
    },
]


def _run_prompt(name: str, args: list[str]) -> subprocess.CompletedProcess:
    """Run a single generate.py prompt via subprocess and return the result.

    Captures stdout and stderr.  Uses the project's .venv python.
    """
    cmd = [PYTHON, GENERATE] + args
    print(f"\n{'=' * 70}")
    print(f"  Running prompt: {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=600,  # 10 minutes per prompt
    )

    # Print stdout/stderr for visibility
    if result.stdout:
        print(f"  [stdout]\n{result.stdout}")
    if result.stderr:
        print(f"  [stderr]\n{result.stderr}")

    return result


def _check_terminal_output(result: subprocess.CompletedProcess, name: str) -> list[str]:
    """Check terminal output for problems.  Returns list of issue strings."""
    issues = []

    if result.returncode != 0:
        issues.append(
            f"[{name}] Exit code {result.returncode} (expected 0). "
            f"stderr: {result.stderr[:500]}"
        )

    combined = (result.stdout or "") + (result.stderr or "")
    error_indicators = [
        "Traceback (most recent call last)",
        "Error:",
        "ERROR",
        "Exception",
        "Traceback",
        "FatalError",
    ]
    for indicator in error_indicators:
        if indicator in combined:
            issues.append(f"[{name}] Found '{indicator}' in terminal output")

    return issues


def _check_json_metadata(name: str, json_path: Path) -> list[str]:
    """Check a .json metadata file for problems.  Returns list of issue strings."""
    issues = []

    if not json_path.exists():
        issues.append(f"[{name}] JSON metadata file not found: {json_path}")
        return issues

    try:
        with open(json_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        issues.append(f"[{name}] JSON metadata is invalid: {e}")
        return issues

    # Check for expected top-level keys
    expected_keys = {"metadata", "phases"}
    missing = expected_keys - set(data.keys())
    if missing:
        issues.append(f"[{name}] JSON metadata missing keys: {missing}")

    # Check metadata sub-dict has expected fields
    meta = data.get("metadata", {})
    expected_meta = {"model", "generation_time_seconds", "created_at"}
    missing_meta = expected_meta - set(meta.keys())
    if missing_meta:
        issues.append(f"[{name}] JSON metadata.metadata missing: {missing_meta}")

    # Check phases list is non-empty
    phases = data.get("phases", [])
    if not phases:
        issues.append(f"[{name}] JSON metadata has no phases")

    return issues


def _check_md_metadata(name: str, md_path: Path) -> list[str]:
    """Check a .md metadata file for problems.  Returns list of issue strings."""
    issues = []

    if not md_path.exists():
        issues.append(f"[{name}] MD metadata file not found: {md_path}")
        return issues

    content = md_path.read_text()

    if not content.strip():
        issues.append(f"[{name}] MD metadata file is empty")

    expected_sections = ["## Phases", "## Run Metadata"]
    for section in expected_sections:
        if section not in content:
            issues.append(f"[{name}] MD metadata missing section: {section}")

    return issues


def _check_png_output(name: str, png_path: Path) -> list[str]:
    """Check a .png output file for problems.  Returns list of issue strings."""
    issues = []

    if not png_path.exists():
        issues.append(f"[{name}] PNG output file not found: {png_path}")
        return issues

    size = png_path.stat().st_size
    if size == 0:
        issues.append(f"[{name}] PNG output file is empty (0 bytes): {png_path}")

    # PNG files should start with the PNG magic bytes
    with open(png_path, "rb") as f:
        header = f.read(8)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        issues.append(f"[{name}] PNG output file has invalid header: {png_path}")

    return issues


class TestEndToEndPrompts:
    """Run all 8 generate.py prompts sequentially and check for problems."""

    # Use a long timeout since each generation loads models and runs inference
    pytestmark = pytest.mark.timeout(6000)

    def test_all_prompts_run_and_check_outputs(self):
        """Run all 8 prompts sequentially, then check all outputs for problems."""
        all_issues: list[str] = []

        # Track which files are expected to be generated
        # (for metadata checks — only prompts with --metadata produce .json/.md)
        metadata_files: list[tuple[str, Path, Path]] = []  # (name, json_path, md_path)
        png_files: list[tuple[str, Path]] = []  # (name, png_path)

        # --- Run each prompt sequentially ---
        for prompt_cfg in PROMPTS:
            name = prompt_cfg["name"]
            args = prompt_cfg["args"]

            result = _run_prompt(name, args)

            # Check terminal output immediately after each prompt
            issues = _check_terminal_output(result, name)
            all_issues.extend(issues)

            # Track expected output files for later checking
            # Determine the output path from args
            output_idx = None
            for i, a in enumerate(args):
                if a == "--output" and i + 1 < len(args):
                    output_idx = i + 1
                    break

            if output_idx is not None:
                output_name = args[output_idx]
                # Bare filename → goes to output/ subfolder
                png_path = PROJECT_ROOT / "output" / output_name
                png_files.append((name, png_path))

                if prompt_cfg["metadata"]:
                    base = png_path.with_suffix("")
                    json_path = base.with_suffix(".json")
                    md_path = base.with_suffix(".md")
                    metadata_files.append((name, json_path, md_path))
            elif prompt_cfg["metadata"] and "--worker" in args:
                # Worker mode: metadata goes to output/<jsonl_basename>
                worker_file = None
                for i, a in enumerate(args):
                    if a == "--worker" and i + 1 < len(args):
                        worker_file = args[i + 1]
                        break
                if worker_file:
                    base_name = Path(worker_file).stem
                    base_path = PROJECT_ROOT / "output" / base_name
                    json_path = base_path.with_suffix(".json")
                    md_path = base_path.with_suffix(".md")
                    metadata_files.append((name, json_path, md_path))

        # --- After all prompts have run, check all outputs ---
        print(f"\n{'=' * 70}")
        print(f"  Checking all outputs for problems...")
        print(f"{'=' * 70}\n")

        # Check PNG files
        for name, png_path in png_files:
            issues = _check_png_output(name, png_path)
            all_issues.extend(issues)
            if not issues:
                print(f"  ✓ {name}: PNG OK ({png_path})")

        # Check JSON metadata files
        for name, json_path, md_path in metadata_files:
            issues = _check_json_metadata(name, json_path)
            all_issues.extend(issues)
            if not issues:
                print(f"  ✓ {name}: JSON OK ({json_path})")

        # Check MD metadata files
        for name, json_path, md_path in metadata_files:
            issues = _check_md_metadata(name, md_path)
            all_issues.extend(issues)
            if not issues:
                print(f"  ✓ {name}: MD OK ({md_path})")

        # --- Report ---
        if all_issues:
            print(f"\n{'!' * 70}")
            print(f"  FOUND {len(all_issues)} ISSUE(S):")
            print(f"{'!' * 70}")
            for issue in all_issues:
                print(f"  • {issue}")
            pytest.fail(
                f"Found {len(all_issues)} issue(s) in end-to-end prompt run:\n"
                + "\n".join(f"  • {i}" for i in all_issues)
            )
        else:
            print(f"\n{'=' * 70}")
            print(f"  ALL CHECKS PASSED — no problems found!")
            print(f"  Ran {len(PROMPTS)} prompts, checked {len(png_files)} PNGs, "
                  f"{len(metadata_files)} JSON+MD metadata files.")
            print(f"{'=' * 70}")

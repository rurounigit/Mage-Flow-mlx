"""Tests for the edit worker (run_edit_worker / load_edit_prompts).

These tests verify the JSONL loading logic, image path validation,
and cache key construction without requiring a GPU or model weights.
"""

import json
import os
import tempfile
import hashlib
from unittest.mock import MagicMock, patch

import pytest

from mage_mlx.worker import (
    load_edit_prompts,
    load_prompts,
    merge_params,
    needs_reload,
    EDIT_VALID_PARAMS,
    VALID_PARAMS,
    _hash_image_bytes,
)


# ---------------------------------------------------------------------------
# load_edit_prompts
# ---------------------------------------------------------------------------

class TestLoadEditPrompts:
    """Tests for load_edit_prompts JSONL parsing and image validation."""

    def test_valid_prompt_with_image(self, tmp_path):
        """A valid prompt with an image field should load correctly."""
        img = tmp_path / "ref.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")  # Minimal PNG header
        # Patch PIL.Image.verify to avoid needing a real image
        with patch("PIL.Image.open") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock()
            mock_open.return_value.verify = MagicMock()
            jsonl = tmp_path / "prompts.jsonl"
            jsonl.write_text(json.dumps({
                "prompt": "make it red",
                "image": str(img),
            }) + "\n")
            prompts = load_edit_prompts(str(jsonl))
        assert len(prompts) == 1
        assert prompts[0]["prompt"] == "make it red"
        assert prompts[0]["image"] == str(img)
        assert prompts[0]["ref_images"] == []
        assert prompts[0]["output"] == "edit_output_1.png"

    def test_valid_prompt_with_ref_images(self, tmp_path):
        """A valid prompt with ref_images list should load correctly."""
        img1 = tmp_path / "ref1.png"
        img1.write_bytes(b"\x89PNG\r\n\x1a\n")
        img2 = tmp_path / "ref2.png"
        img2.write_bytes(b"\x89PNG\r\n\x1a\n")
        with patch("PIL.Image.open") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock()
            mock_open.return_value.verify = MagicMock()
            jsonl = tmp_path / "prompts.jsonl"
            jsonl.write_text(json.dumps({
                "prompt": "edit this",
                "ref_images": [str(img1), str(img2)],
            }) + "\n")
            prompts = load_edit_prompts(str(jsonl))
        assert len(prompts) == 1
        assert prompts[0]["image"] == str(img1)  # first ref becomes image
        assert prompts[0]["ref_images"] == [str(img2)]

    def test_missing_image_and_ref_images_skipped(self, tmp_path):
        """A prompt with neither image nor ref_images should be skipped."""
        jsonl = tmp_path / "prompts.jsonl"
        jsonl.write_text(json.dumps({"prompt": "no image"}) + "\n")
        prompts = load_edit_prompts(str(jsonl))
        assert len(prompts) == 0

    def test_missing_image_file_skipped(self, tmp_path):
        """A prompt with a non-existent image path should be skipped."""
        jsonl = tmp_path / "prompts.jsonl"
        jsonl.write_text(json.dumps({
            "prompt": "edit",
            "image": "/nonexistent/path.png",
        }) + "\n")
        prompts = load_edit_prompts(str(jsonl))
        assert len(prompts) == 0

    def test_malformed_image_skipped(self, tmp_path):
        """A prompt with a malformed image should be skipped."""
        img = tmp_path / "bad.png"
        img.write_text("not an image")
        jsonl = tmp_path / "prompts.jsonl"
        jsonl.write_text(json.dumps({
            "prompt": "edit",
            "image": str(img),
        }) + "\n")
        prompts = load_edit_prompts(str(jsonl))
        assert len(prompts) == 0

    def test_missing_prompt_field_skipped(self, tmp_path):
        """A line without a prompt field should be skipped."""
        jsonl = tmp_path / "prompts.jsonl"
        jsonl.write_text(json.dumps({"image": "foo.png"}) + "\n")
        prompts = load_edit_prompts(str(jsonl))
        assert len(prompts) == 0

    def test_invalid_json_skipped(self, tmp_path):
        """A line with invalid JSON should be skipped."""
        jsonl = tmp_path / "prompts.jsonl"
        jsonl.write_text("not valid json\n")
        prompts = load_edit_prompts(str(jsonl))
        assert len(prompts) == 0

    def test_comments_and_blank_lines_skipped(self, tmp_path):
        """Comments and blank lines should be skipped."""
        jsonl = tmp_path / "prompts.jsonl"
        jsonl.write_text("# comment\n\n")
        prompts = load_edit_prompts(str(jsonl))
        assert len(prompts) == 0

    def test_unknown_params_skipped(self, tmp_path):
        """A line with unknown parameters should be skipped."""
        img = tmp_path / "ref.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        with patch("PIL.Image.open") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock()
            mock_open.return_value.verify = MagicMock()
            jsonl = tmp_path / "prompts.jsonl"
            jsonl.write_text(json.dumps({
                "prompt": "edit",
                "image": str(img),
                "unknown_param": "bad",
            }) + "\n")
            prompts = load_edit_prompts(str(jsonl))
        assert len(prompts) == 0

    def test_custom_output_preserved(self, tmp_path):
        """A custom output path should be preserved."""
        img = tmp_path / "ref.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        with patch("PIL.Image.open") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock()
            mock_open.return_value.verify = MagicMock()
            jsonl = tmp_path / "prompts.jsonl"
            jsonl.write_text(json.dumps({
                "prompt": "edit",
                "image": str(img),
                "output": "custom_output.png",
            }) + "\n")
            prompts = load_edit_prompts(str(jsonl))
        assert prompts[0]["output"] == "custom_output.png"

    def test_image_as_string_ref_images(self, tmp_path):
        """ref_images as a single string should be converted to a list."""
        img1 = tmp_path / "ref1.png"
        img1.write_bytes(b"\x89PNG\r\n\x1a\n")
        img2 = tmp_path / "ref2.png"
        img2.write_bytes(b"\x89PNG\r\n\x1a\n")
        with patch("PIL.Image.open") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock()
            mock_open.return_value.verify = MagicMock()
            jsonl = tmp_path / "prompts.jsonl"
            jsonl.write_text(json.dumps({
                "prompt": "edit",
                "image": str(img1),
                "ref_images": str(img2),  # single string, not list
            }) + "\n")
            prompts = load_edit_prompts(str(jsonl))
        assert len(prompts) == 1
        assert prompts[0]["ref_images"] == [str(img2)]


# ---------------------------------------------------------------------------
# _hash_image_bytes
# ---------------------------------------------------------------------------

class TestHashImageBytes:
    """Tests for _hash_image_bytes."""

    def test_deterministic(self, tmp_path):
        """Same file should produce same hash."""
        img = tmp_path / "ref.png"
        img.write_bytes(b"test image data")
        h1 = _hash_image_bytes(str(img))
        h2 = _hash_image_bytes(str(img))
        assert h1 == h2

    def test_different_files_different_hashes(self, tmp_path):
        """Different files should produce different hashes."""
        img1 = tmp_path / "ref1.png"
        img1.write_bytes(b"image 1")
        img2 = tmp_path / "ref2.png"
        img2.write_bytes(b"image 2")
        assert _hash_image_bytes(str(img1)) != _hash_image_bytes(str(img2))

    def test_sha256(self, tmp_path):
        """Should produce a valid SHA-256 hex string."""
        img = tmp_path / "ref.png"
        img.write_bytes(b"test")
        h = _hash_image_bytes(str(img))
        assert len(h) == 64  # SHA-256 hex digest length
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# merge_params
# ---------------------------------------------------------------------------

class TestMergeParams:
    """Tests for merge_params."""

    def test_override_takes_precedence(self):
        """Per-prompt values should override defaults."""
        defaults = {"seed": 42, "steps": 4, "width": 1024}
        override = {"seed": 99}
        merged = merge_params(defaults, override)
        assert merged["seed"] == 99
        assert merged["steps"] == 4
        assert merged["width"] == 1024

    def test_empty_override(self):
        """Empty override should return defaults."""
        defaults = {"seed": 42}
        merged = merge_params(defaults, {})
        assert merged == defaults

    def test_does_not_mutate_defaults(self):
        """merge_params should not mutate the defaults dict."""
        defaults = {"seed": 42}
        override = {"seed": 99}
        merge_params(defaults, override)
        assert defaults["seed"] == 42


# ---------------------------------------------------------------------------
# needs_reload
# ---------------------------------------------------------------------------

class TestNeedsReload:
    """Tests for needs_reload."""

    def test_no_changes(self):
        """No parameter changes should not require reload."""
        current = {"model": "a", "quantize": None, "steps": 4}
        new = {"model": "a", "quantize": None, "steps": 4}
        needs_pipeline, needs_scheduler = needs_reload(current, new)
        assert not needs_pipeline
        assert not needs_scheduler

    def test_model_change_requires_pipeline(self):
        """Model change should require pipeline reload."""
        current = {"model": "a", "quantize": None, "steps": 4}
        new = {"model": "b", "quantize": None, "steps": 4}
        needs_pipeline, needs_scheduler = needs_reload(current, new)
        assert needs_pipeline
        assert not needs_scheduler

    def test_quantize_change_requires_pipeline(self):
        """Quantize change should require pipeline reload."""
        current = {"model": "a", "quantize": None, "steps": 4}
        new = {"model": "a", "quantize": 4, "steps": 4}
        needs_pipeline, needs_scheduler = needs_reload(current, new)
        assert needs_pipeline

    def test_steps_change_requires_scheduler(self):
        """Steps change should require scheduler reset."""
        current = {"model": "a", "quantize": None, "steps": 4}
        new = {"model": "a", "quantize": None, "steps": 8}
        needs_pipeline, needs_scheduler = needs_reload(current, new)
        assert not needs_pipeline
        assert needs_scheduler


# ---------------------------------------------------------------------------
# EDIT_VALID_PARAMS
# ---------------------------------------------------------------------------

class TestEditValidParams:
    """Tests for EDIT_VALID_PARAMS."""

    def test_includes_txt2img_params(self):
        """Edit params should include all txt2img params."""
        assert VALID_PARAMS.issubset(EDIT_VALID_PARAMS)

    def test_includes_image_fields(self):
        """Edit params should include image and ref_images."""
        assert "image" in EDIT_VALID_PARAMS
        assert "ref_images" in EDIT_VALID_PARAMS

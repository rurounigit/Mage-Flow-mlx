"""Tests for the thermal monitoring module (mage_mlx/thermal.py).

These tests verify the thermal state detection logic, label thresholds,
formatting, and graceful fallback when sysctl is unavailable — all without
requiring a real macOS system or elevated privileges.
"""

from unittest.mock import patch

from mage_mlx.thermal import (
    _level_to_label,
    _notify_state_to_label,
    _read_sysctl,
    get_thermal_state,
    format_thermal_state,
)


# ---------------------------------------------------------------------------
# _level_to_label (sysctl 0-100 scale)
# ---------------------------------------------------------------------------

class TestLevelToLabel:
    """Tests for the _level_to_label threshold function (sysctl 0-100)."""

    def test_nominal(self):
        """Levels 0-19 should be 'NOMINAL'."""
        assert _level_to_label(0) == "NOMINAL"
        assert _level_to_label(19) == "NOMINAL"

    def test_fair(self):
        """Levels 20-39 should be 'FAIR'."""
        assert _level_to_label(20) == "FAIR"
        assert _level_to_label(39) == "FAIR"

    def test_serious(self):
        """Levels 40-59 should be 'SERIOUS'."""
        assert _level_to_label(40) == "SERIOUS"
        assert _level_to_label(59) == "SERIOUS"

    def test_critical(self):
        """Levels 60-100 should be 'CRITICAL'."""
        assert _level_to_label(60) == "CRITICAL"
        assert _level_to_label(100) == "CRITICAL"


# ---------------------------------------------------------------------------
# _notify_state_to_label (notify 0-3 scale)
# ---------------------------------------------------------------------------

class TestNotifyStateToLabel:
    """Tests for the _notify_state_to_label function (notify 0-3)."""

    def test_nominal(self):
        assert _notify_state_to_label(0) == "NOMINAL"

    def test_fair(self):
        assert _notify_state_to_label(1) == "FAIR"

    def test_serious(self):
        assert _notify_state_to_label(2) == "SERIOUS"

    def test_critical(self):
        assert _notify_state_to_label(3) == "CRITICAL"

    def test_unknown(self):
        assert _notify_state_to_label(99) == "unknown"


# ---------------------------------------------------------------------------
# _read_sysctl
# ---------------------------------------------------------------------------

class TestReadSysctl:
    """Tests for the _read_sysctl helper."""

    def test_returns_int_on_success(self):
        """Should return the integer value from sysctl stdout."""
        mock_result = type("R", (), {"returncode": 0, "stdout": "42\n"})()
        with patch("subprocess.run", return_value=mock_result):
            assert _read_sysctl("machdep.xcpm.cpu_thermal_level") == 42

    def test_returns_none_on_nonzero_exit(self):
        """Should return None when sysctl returns non-zero exit code."""
        mock_result = type("R", (), {"returncode": 1, "stdout": ""})()
        with patch("subprocess.run", return_value=mock_result):
            assert _read_sysctl("machdep.xcpm.cpu_thermal_level") is None

    def test_returns_none_on_timeout(self):
        """Should return None when sysctl times out."""
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="sysctl", timeout=2)):
            assert _read_sysctl("machdep.xcpm.cpu_thermal_level") is None

    def test_returns_none_on_oserror(self):
        """Should return None when sysctl raises OSError."""
        with patch("subprocess.run", side_effect=OSError("command not found")):
            assert _read_sysctl("machdep.xcpm.cpu_thermal_level") is None

    def test_returns_none_on_invalid_output(self):
        """Should return None when sysctl output is not a valid integer."""
        mock_result = type("R", (), {"returncode": 0, "stdout": "not_a_number\n"})()
        with patch("subprocess.run", return_value=mock_result):
            assert _read_sysctl("machdep.xcpm.cpu_thermal_level") is None


# ---------------------------------------------------------------------------
# get_thermal_state
# ---------------------------------------------------------------------------

class TestGetThermalState:
    """Tests for get_thermal_state()."""

    def test_returns_dict_with_expected_keys(self):
        """Should return a dict with the three expected keys."""
        # Mock notify to return None so it falls through to sysctl
        with patch("mage_mlx.thermal._get_thermal_via_notify", return_value=None):
            with patch("mage_mlx.thermal._read_sysctl", return_value=0):
                state = get_thermal_state()
        assert isinstance(state, dict)
        assert "cpu_thermal_level" in state
        assert "gpu_thermal_level" in state
        assert "thermal_throttling" in state

    def test_notify_primary_method(self):
        """Should use notify framework as primary method when available."""
        with patch("mage_mlx.thermal._get_thermal_via_notify", return_value=0):
            state = get_thermal_state()
        assert state["cpu_thermal_level"] == 0
        assert state["gpu_thermal_level"] is None
        assert state["thermal_throttling"] == "NOMINAL"

    def test_notify_critical_state(self):
        """Notify state 3 should produce 'CRITICAL' label."""
        with patch("mage_mlx.thermal._get_thermal_via_notify", return_value=3):
            state = get_thermal_state()
        assert state["thermal_throttling"] == "CRITICAL"

    def test_falls_back_to_sysctl_when_notify_unavailable(self):
        """When notify returns None, should fall back to sysctl."""
        with patch("mage_mlx.thermal._get_thermal_via_notify", return_value=None):
            with patch("mage_mlx.thermal._read_sysctl", return_value=50):
                state = get_thermal_state()
        assert state["cpu_thermal_level"] == 50
        assert state["thermal_throttling"] == "SERIOUS"

    def test_falls_back_to_gpu_when_cpu_unavailable(self):
        """When CPU level is None, should use GPU level for the label."""
        call_count = [0]

        def mock_read(key):
            call_count[0] += 1
            if "cpu" in key:
                return None
            return 50  # GPU level

        with patch("mage_mlx.thermal._get_thermal_via_notify", return_value=None):
            with patch("mage_mlx.thermal._read_sysctl", side_effect=mock_read):
                state = get_thermal_state()
        assert state["cpu_thermal_level"] is None
        assert state["gpu_thermal_level"] == 50
        assert state["thermal_throttling"] == "SERIOUS"

    def test_unknown_when_both_unavailable(self):
        """When both notify and sysctl are unavailable, label should be 'unknown'."""
        with patch("mage_mlx.thermal._get_thermal_via_notify", return_value=None):
            with patch("mage_mlx.thermal._read_sysctl", return_value=None):
                state = get_thermal_state()
        assert state["cpu_thermal_level"] is None
        assert state["gpu_thermal_level"] is None
        assert state["thermal_throttling"] == "unknown"


# ---------------------------------------------------------------------------
# format_thermal_state
# ---------------------------------------------------------------------------

class TestFormatThermalState:
    """Tests for format_thermal_state()."""

    def test_formats_with_both_levels(self):
        """Should format both CPU and GPU levels with the throttling label."""
        state = {
            "cpu_thermal_level": 42,
            "gpu_thermal_level": 38,
            "thermal_throttling": "SERIOUS",
        }
        result = format_thermal_state(state)
        assert result == "CPU=42 GPU=38 (SERIOUS)"

    def test_formats_with_cpu_only(self):
        """Should format when only CPU level is available."""
        state = {
            "cpu_thermal_level": 10,
            "gpu_thermal_level": None,
            "thermal_throttling": "NOMINAL",
        }
        result = format_thermal_state(state)
        assert result == "CPU=10 (NOMINAL)"

    def test_formats_with_gpu_only(self):
        """Should format when only GPU level is available."""
        state = {
            "cpu_thermal_level": None,
            "gpu_thermal_level": 75,
            "thermal_throttling": "CRITICAL",
        }
        result = format_thermal_state(state)
        assert result == "GPU=75 (CRITICAL)"

    def test_unavailable_when_no_levels(self):
        """Should return 'thermal state unavailable' when no levels are present."""
        state = {
            "cpu_thermal_level": None,
            "gpu_thermal_level": None,
            "thermal_throttling": "unknown",
        }
        result = format_thermal_state(state)
        assert result == "thermal state unavailable"

    def test_empty_dict(self):
        """Should handle an empty dict gracefully."""
        result = format_thermal_state({})
        assert result == "thermal state unavailable"

    def test_notify_state_format(self):
        """Notify state (0-3) should format with CPU level and label."""
        state = {
            "cpu_thermal_level": 0,
            "gpu_thermal_level": None,
            "thermal_throttling": "NOMINAL",
        }
        result = format_thermal_state(state)
        assert result == "CPU=0 (NOMINAL)"


# ---------------------------------------------------------------------------
# Profiler integration
# ---------------------------------------------------------------------------

class TestProfilerThermalState:
    """Tests for the Profiler.thermal_state property and integration."""

    def test_thermal_state_property_returns_none_when_no_records(self):
        """Profiler.thermal_state should be None when no phases have thermal data."""
        from mage_mlx.profiler import Profiler

        prof = Profiler(enabled=True)
        assert prof.thermal_state is None

    def test_thermal_state_property_returns_latest(self):
        """Profiler.thermal_state should return the most recent thermal state."""
        from mage_mlx.profiler import Profiler

        prof = Profiler(enabled=True)
        prof.start("phase1")
        prof.stop("phase1")
        prof.set_thermal_state("phase1", {"cpu_thermal_level": 0, "gpu_thermal_level": None, "thermal_throttling": "NOMINAL"})

        prof.start("phase2")
        prof.stop("phase2")
        prof.set_thermal_state("phase2", {"cpu_thermal_level": 3, "gpu_thermal_level": None, "thermal_throttling": "CRITICAL"})

        assert prof.thermal_state["thermal_throttling"] == "CRITICAL"

    def test_thermal_state_in_to_dict(self):
        """to_dict() should include thermal_state in phase records."""
        from mage_mlx.profiler import Profiler

        prof = Profiler(enabled=True)
        prof.start("test_phase")
        prof.stop("test_phase")
        prof.set_thermal_state("test_phase", {"cpu_thermal_level": 2, "gpu_thermal_level": None, "thermal_throttling": "SERIOUS"})

        data = prof.to_dict()
        assert "phases" in data
        assert len(data["phases"]) == 1
        assert data["phases"][0]["thermal_state"] is not None
        assert data["phases"][0]["thermal_state"]["thermal_throttling"] == "SERIOUS"

    def test_thermal_state_in_to_markdown(self):
        """to_markdown() should include thermal state in the phase table."""
        from mage_mlx.profiler import Profiler

        prof = Profiler(enabled=True)
        prof.start("test_phase")
        prof.stop("test_phase")
        prof.set_thermal_state("test_phase", {"cpu_thermal_level": 2, "gpu_thermal_level": None, "thermal_throttling": "SERIOUS"})

        md = prof.to_markdown()
        assert "thermal=" in md
        assert "SERIOUS" in md

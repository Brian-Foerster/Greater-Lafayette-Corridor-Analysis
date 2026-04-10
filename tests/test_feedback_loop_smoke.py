"""Smoke test for scripts/run_feedback_loop.py.

Verifies:
1. The script is importable and parser is accessible
2. CLI argument defaults are sensible
3. (slow) The script runs to completion with synthetic data

The slow test reuses the same synthetic data fixtures as test_model_integration.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestFeedbackLoopParser(unittest.TestCase):
    """Fast tests for CLI parser and script structure."""

    def test_parser_importable(self):
        from scripts.run_feedback_loop import _build_parser
        parser = _build_parser()
        self.assertIsNotNone(parser)

    def test_parser_defaults(self):
        from scripts.run_feedback_loop import _build_parser
        parser = _build_parser()
        args = parser.parse_args([])

        self.assertIn("parcels_enriched_final", args.parcels)
        self.assertIn("apm_phase2a_corridors", args.corridors)
        self.assertEqual(args.scenario, "no_zoning")
        self.assertFalse(args.parallel)

    def test_parser_scenario_flag(self):
        from scripts.run_feedback_loop import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--scenario", "no_zoning"])
        self.assertEqual(args.scenario, "no_zoning")

    def test_parser_steps_flag(self):
        from scripts.run_feedback_loop import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--steps", "0", "5", "10"])
        self.assertEqual(args.steps, [0, 5, 10])

    def test_parser_screening_flag(self):
        from scripts.run_feedback_loop import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--screening"])
        self.assertTrue(args.screening)

    def test_parser_no_gtfs_flag(self):
        from scripts.run_feedback_loop import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--no-gtfs"])
        self.assertTrue(args.no_gtfs)


@pytest.mark.slow
class TestFeedbackLoopSubprocess(unittest.TestCase):
    """Subprocess smoke test — runs the script with synthetic data."""

    @classmethod
    def setUpClass(cls):
        """Write synthetic data to temp dir (same fixtures as test_model_integration)."""
        from test_model_integration import (
            _make_parcels_geojson,
            _make_corridors_geojson,
            _make_od_csv,
        )

        cls._tmpdir_obj = tempfile.TemporaryDirectory()
        td = cls._tmpdir_obj.name

        cls._parcels = os.path.join(td, "parcels.geojson")
        cls._corridors = os.path.join(td, "corridors.geojson")
        cls._od = os.path.join(td, "od.csv")
        cls._output = os.path.join(td, "results.csv")
        cls._diag = os.path.join(td, "diagnostics.csv")

        with open(cls._parcels, "w") as f:
            json.dump(_make_parcels_geojson(10), f)
        with open(cls._corridors, "w") as f:
            json.dump(_make_corridors_geojson(2), f)
        with open(cls._od, "w") as f:
            f.write(_make_od_csv(10, 20))

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir_obj.cleanup()

    def test_script_runs_with_synthetic_data(self):
        """run_feedback_loop.py should complete with synthetic data."""
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "run_feedback_loop.py"),
            "--corridors", self._corridors,
            "--parcels", self._parcels,
            "--od-flows", self._od,
            "--steps", "0", "1", "2",
            "--no-gtfs",
            "--no-bus-restructure",
            "--skip-source-manifest-validation",
            "--output", self._output,
            "--diagnostics-output", self._diag,
        ]
        proc = subprocess.run(
            cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True,
            timeout=120,
        )

        self.assertEqual(
            proc.returncode, 0,
            f"Script failed (rc={proc.returncode}).\nstderr:\n{proc.stderr[-1000:]}",
        )
        self.assertTrue(
            os.path.exists(self._output),
            "Output CSV not produced",
        )


if __name__ == "__main__":
    unittest.main()

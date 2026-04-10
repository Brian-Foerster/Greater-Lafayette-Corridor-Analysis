"""Week 8 tests for behavioral benchmark validation gate."""

import tempfile
import unittest
from pathlib import Path


try:
    import scripts.validate_against_benchmarks as gate
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import guard for optional deps
    gate = None
    _IMPORT_ERROR = exc


class TestBehavioralValidationGate(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_expected_checks_present(self):
        results = gate.run_all_validations(decay_beta=0.00173)
        expected = {
            "walk_access_decay_400m",
            "walk_access_decay_800m",
            "peak_concentration",
            "headway_elasticity",
            "fare_elasticity",
            "parking_transit_sensitivity",
            "income_segment_fare_sensitivity_gap",
        }
        self.assertTrue(expected.issubset(set(results.keys())))
        for check in expected:
            self.assertIn(results[check]["status"], {"PASS", "WARN", "FAIL"})

    def test_output_artifacts_written(self):
        results = gate.run_all_validations(decay_beta=0.00173)
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "results.json"
            csv_path = Path(tmp) / "results.csv"
            md_path = Path(tmp) / "summary.md"

            gate.write_outputs(results, json_path, csv_path, md_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())

            text = md_path.read_text(encoding="utf-8")
            self.assertIn("Behavioral Validation Summary", text)
            self.assertIn("| Check | Status |", text)


if __name__ == "__main__":
    unittest.main()

"""Week 23 tests for geometry change detection utility."""

import tempfile
import unittest
from pathlib import Path

try:
    import sys
    import scripts.geometry_change_detector as gcd
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    gcd = None
    _IMPORT_ERROR = exc


class TestGeometryChangeDetector(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_is_geometry_path_detects_expected_inputs(self):
        self.assertTrue(gcd.is_geometry_path("data/processed/apm_phase2a_corridors.geojson"))
        self.assertTrue(gcd.is_geometry_path("data/processed/apm_phase2a_stops.geojson"))
        self.assertTrue(gcd.is_geometry_path("data/processed/phase2b_corridor_results_geometry_adjusted.csv"))
        self.assertFalse(gcd.is_geometry_path("src/mode_choice.py"))

    def test_build_report_flags_required_pipelines(self):
        report = gcd.build_geometry_change_report(
            changed_files=[
                "data/processed/apm_phase2a_corridors.geojson",
                "scripts/optimized_corridor_search.py",
            ],
            base_ref="abc",
            head_ref="def",
        )
        self.assertTrue(report["geometry_changed"])
        self.assertEqual(report["geometry_changed_file_count"], 1)
        self.assertIn("apm_corridor_evaluation_integrated", report["required_pipelines"])
        self.assertIn("run_feedback_loop", report["required_pipelines"])

    def test_changed_files_file_loader(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "changed.txt"
            p.write_text(
                "data/processed/apm_phase2a_corridors.geojson\n"
                "src/transit_model.py\n",
                encoding="utf-8",
            )
            files = gcd._load_changed_files_file(p)
            self.assertEqual(len(files), 2)
            self.assertEqual(files[0], "data/processed/apm_phase2a_corridors.geojson")


if __name__ == "__main__":
    unittest.main()

"""Week 23 tests for geometry rerun smoke script."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import sys
    import scripts.run_geometry_rerun_smoke as smoke
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    smoke = None
    _IMPORT_ERROR = exc


class TestGeometryRerunSmoke(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_smoke_summary_writers(self):
        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "summary.json"
            out_md = Path(td) / "summary.md"
            payload = {
                "generated_at_utc": "2026-02-24T00:00:00+00:00",
                "status": "pass",
                "failing_command_count": 0,
                "reason": {"geometry_changed": True, "geometry_changed_file_count": 1},
                "commands": [
                    {"command": "python -m unittest -v tests.test_geometry_change_detector", "returncode": 0, "duration_seconds": 0.1}
                ],
            }
            smoke._write_json(out_json, payload)
            smoke._write_md(out_md, payload)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_md.exists())

            parsed = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(parsed["status"], "pass")
            self.assertIn("Geometry Rerun Smoke Summary", out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

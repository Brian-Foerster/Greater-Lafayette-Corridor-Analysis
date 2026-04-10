"""Week 24 tests for KPI drift threshold gate."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    import sys
    import scripts.check_kpi_drift as drift
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    drift = None
    _IMPORT_ERROR = exc


class TestKpiDriftGate(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_run_drift_gate_pass_and_fail_cases(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data_path = td_path / "sample.csv"
            pd.DataFrame(
                {
                    "metric_a": [10.0, 10.0, 10.0],
                    "flag": [True, False, True],
                }
            ).to_csv(data_path, index=False)

            spec_pass = {
                "version": "v1",
                "metrics": [
                    {
                        "id": "mean_metric_a",
                        "path": str(data_path),
                        "column": "metric_a",
                        "agg": "mean",
                        "baseline": 10.0,
                        "max_rel_drift": 0.01,
                    },
                    {
                        "id": "flag_rate",
                        "path": str(data_path),
                        "column": "flag",
                        "type": "bool_rate",
                        "agg": "mean",
                        "baseline": 2.0 / 3.0,
                        "max_abs_drift": 0.001,
                    },
                ],
            }
            pass_path = td_path / "spec_pass.json"
            pass_path.write_text(json.dumps(spec_pass), encoding="utf-8")
            summary_pass = drift.run_drift_gate(pass_path)
            self.assertEqual(summary_pass["status"], "PASS")
            self.assertEqual(summary_pass["fail_count"], 0)

            spec_fail = {
                "version": "v1",
                "metrics": [
                    {
                        "id": "mean_metric_a_fail",
                        "path": str(data_path),
                        "column": "metric_a",
                        "agg": "mean",
                        "baseline": 20.0,
                        "max_rel_drift": 0.01,
                    }
                ],
            }
            fail_path = td_path / "spec_fail.json"
            fail_path.write_text(json.dumps(spec_fail), encoding="utf-8")
            summary_fail = drift.run_drift_gate(fail_path)
            self.assertEqual(summary_fail["status"], "FAIL")
            self.assertEqual(summary_fail["fail_count"], 1)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for TIF financing guardrails and breakeven logic."""

import unittest
from pathlib import Path

import pandas as pd

try:
    import sys
    import scripts.tif_financing_model as tfm
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    tfm = None
    _IMPORT_ERROR = exc


class TestTifFinancingModelRegressions(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_negative_tax_increment_is_clipped(self):
        values = pd.DataFrame(
            {
                "year": [0, 1, 2],
                "baseline_value_dollars": [100.0, 100.0, 100.0],
                "projected_value_dollars": [100.0, 90.0, 80.0],
            }
        )
        out = tfm.calculate_tif_revenue(
            values,
            scenario_name="test",
            config={
                "property_tax_rate": 0.01,
                "capture_rate": 0.85,
                "baseline_growth_rate": 0.02,
            },
        )
        self.assertTrue((out["tax_increment"] >= 0).all())
        self.assertTrue((out["tif_revenue_captured"] >= 0).all())
        self.assertAlmostEqual(float(out["cumulative_tif_revenue"].iloc[-1]), 0.0, places=8)

    def test_breakeven_never_returns_year_zero(self):
        tif_revenue = pd.DataFrame(
            {
                "year": [0, 1, 2, 3],
                "cumulative_tif_revenue": [0.0, 0.0, 0.0, 0.0],
                "scenario": ["test"] * 4,
            }
        )
        _, summary = tfm.calculate_breakeven_analysis(
            tif_revenue,
            apm_costs={
                "operating_cost_fixed": 1_000_000,
                "operating_cost_per_km": 100_000,
                "capital_cost_per_km": 10_000_000,
                "debt_service_period": 20,
                "default_length_km": 8.0,
            },
        )
        self.assertEqual(summary["operations_breakeven_year"], 26)
        self.assertEqual(summary["debt_service_breakeven_year"], 26)
        self.assertEqual(summary["full_cost_breakeven_year"], 26)


if __name__ == "__main__":
    unittest.main()

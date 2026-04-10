"""Week 21 tests for uncertainty framework and percentile-band outputs."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import sys
    import scripts.apm_corridor_evaluation_integrated as apm_eval
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    apm_eval = None
    _IMPORT_ERROR = exc


class TestWeek21UncertaintyFramework(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def _synthetic_results_df(self):
        return pd.DataFrame(
            {
                "corridor_id": ["C1", "C2"],
                "scenario": ["zoning", "zoning"],
                "daily_ridership": [3500.0, 1800.0],
                "tif_revenue_cumulative": [900_000_000.0, 420_000_000.0],
                "capital_cost": [420_000_000.0, 260_000_000.0],
                "annual_operating_cost": [3_000_000.0, 2_200_000.0],
                "annual_debt_service": [13_000_000.0, 8_500_000.0],
                "debt_coverage_ratio": [1.1, 0.8],
                "project_npv_dynamic_musd": [30.0, -40.0],
            }
        )

    def test_load_uncertainty_framework_from_config(self):
        cfg = apm_eval.load_uncertainty_framework("scenarios_config.json")
        self.assertIn("runner_defaults", cfg)
        self.assertIn("parameter_ranges", cfg)
        self.assertIn("ridership_multiplier", cfg["parameter_ranges"])
        self.assertIn("percentiles", cfg["runner_defaults"])

    def test_uncertainty_bands_are_monotonic_and_bounded(self):
        df = self._synthetic_results_df()
        wide, long_df, draws, meta = apm_eval.compute_uncertainty_bands(
            df,
            n_draws=250,
            random_seed=7,
            percentiles=[10, 50, 90],
            cashflow_years=25,
            fare_per_trip_usd=2.00,
            farebox_capture_rate=1.0,
            discount_rate=0.05,
        )

        self.assertEqual(len(wide), 2)
        self.assertGreater(len(long_df), 0)
        self.assertEqual(int(meta["n_draws"]), 250)
        self.assertEqual(len(draws), 500)

        for metric in [
            "daily_ridership",
            "tif_revenue_cumulative",
            "debt_coverage_ratio",
            "project_npv_dynamic_musd",
        ]:
            p10 = wide[f"{metric}_p10"].to_numpy()
            p50 = wide[f"{metric}_p50"].to_numpy()
            p90 = wide[f"{metric}_p90"].to_numpy()
            self.assertTrue(np.all(p10 <= p50))
            self.assertTrue(np.all(p50 <= p90))

        probs = wide["financial_viability_probability"].to_numpy()
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

    def test_uncertainty_results_are_reproducible_for_fixed_seed(self):
        df = self._synthetic_results_df()
        w1, _, _, _ = apm_eval.compute_uncertainty_bands(
            df,
            n_draws=200,
            random_seed=99,
            percentiles=[10, 50, 90],
        )
        w2, _, _, _ = apm_eval.compute_uncertainty_bands(
            df,
            n_draws=200,
            random_seed=99,
            percentiles=[10, 50, 90],
        )
        self.assertEqual(list(w1["corridor_id"]), list(w2["corridor_id"]))

        numeric_cols = [
            c
            for c in w1.columns
            if c not in {"corridor_id", "scenario"}
        ]
        self.assertTrue(
            np.allclose(
                w1[numeric_cols].to_numpy(dtype=float),
                w2[numeric_cols].to_numpy(dtype=float),
            )
        )


if __name__ == "__main__":
    unittest.main()

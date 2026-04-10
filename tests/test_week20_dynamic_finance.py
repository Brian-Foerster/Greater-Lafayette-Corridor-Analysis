"""Week 20 regression tests for dynamic finance re-baselining and traceability."""

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

try:
    import sys
    import scripts.apm_corridor_evaluation_integrated as apm_eval
    import scripts.tif_financing_model as tif_model
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    apm_eval = None
    tif_model = None
    _IMPORT_ERROR = exc


class TestWeek20DynamicFinance(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_integrated_dynamic_finance_metrics_are_stable(self):
        annual_series, source = apm_eval._build_annual_ridership_series(
            daily_ridership=100.0,
            daily_ridership_series=[100.0, 200.0, 300.0],
            years=3,
        )
        self.assertEqual(source, "dynamic_daily_series")
        # 312 academic-calendar weighted operating days
        self.assertTrue(np.allclose(annual_series, np.array([31200.0, 62400.0, 93600.0])))

        kwargs = {
            "years": 3,
            "annual_tif_series_usd": np.array([3_000_000.0, 3_000_000.0, 3_000_000.0]),
            "capital_cost_usd": 120_000_000.0,
            "annual_debt_service_usd": 7_800_000.0,
            "annual_operating_cost_usd": 2_500_000.0,
            "discount_rate": 0.05,
            "fare_per_trip_usd": 2.5,
            "farebox_capture_rate": 1.0,
        }
        m1 = apm_eval._compute_dynamic_finance_metrics(annual_series, **kwargs)
        m2 = apm_eval._compute_dynamic_finance_metrics(annual_series, **kwargs)

        self.assertAlmostEqual(m1["annual_ridership_mean"], 62400.0, places=6)
        self.assertAlmostEqual(m1["annual_ridership_total"], 187200.0, places=6)
        # 62400 * 2.5 * 1.0 / 1e6 = 0.156
        self.assertAlmostEqual(m1["farebox_revenue_annual_mean_musd"], 0.156, places=6)
        self.assertAlmostEqual(m1["project_npv_dynamic_musd"], m2["project_npv_dynamic_musd"], places=10)
        # DSCR fields replace the old debt_coverage_ratio_dynamic
        self.assertAlmostEqual(m1["dscr_year5"], m2["dscr_year5"], places=10)
        self.assertAlmostEqual(m1["dscr_min"], m2["dscr_min"], places=10)
        self.assertIn("dscr_year25", m1)

    def test_tif_dynamic_overlay_has_traceability_and_interpolation(self):
        tif_df = pd.DataFrame(
            {
                "year": [0, 1, 2, 3, 4],
                "tif_revenue_captured": [1_000_000.0] * 5,
                "cumulative_tif_revenue": [1_000_000.0, 2_000_000.0, 3_000_000.0, 4_000_000.0, 5_000_000.0],
                "scenario": ["zoning"] * 5,
            }
        )
        dynamic_df = pd.DataFrame(
            {
                "corridor_id": ["C1", "C2", "C1", "C2"],
                "year": [0, 0, 4, 4],
                "daily_riders": [100.0, 50.0, 200.0, 100.0],
            }
        )
        apm_costs = {
            "operating_cost_fixed": 1_000_000,
            "operating_cost_per_km": 100_000,
            "capital_cost_per_km": 20_000_000,
            "debt_service_period": 20,
            "default_length_km": 8.0,
        }

        out1 = tif_model.build_dynamic_finance_scenario_output(
            tif_df,
            "zoning",
            dynamic_ridership_df=dynamic_df,
            fare_per_trip_usd=2.0,
            farebox_capture_rate=1.0,
            apm_costs=apm_costs,
            discount_rate=0.03,
        )
        out2 = tif_model.build_dynamic_finance_scenario_output(
            tif_df,
            "zoning",
            dynamic_ridership_df=dynamic_df,
            fare_per_trip_usd=2.0,
            farebox_capture_rate=1.0,
            apm_costs=apm_costs,
            discount_rate=0.03,
        )

        self.assertIn("demand_trace_source", out1.columns)
        self.assertIn("project_npv_dynamic_musd", out1.columns)
        self.assertEqual(out1["demand_trace_source"].iloc[0], "feedback_loop_dynamic")

        self.assertAlmostEqual(out1.loc[out1["year"] == 0, "daily_riders_modeled_total"].iloc[0], 150.0, places=6)
        self.assertAlmostEqual(out1.loc[out1["year"] == 2, "daily_riders_modeled_total"].iloc[0], 225.0, places=6)
        self.assertAlmostEqual(out1.loc[out1["year"] == 4, "daily_riders_modeled_total"].iloc[0], 300.0, places=6)

        self.assertTrue(np.allclose(
            out1["debt_coverage_ratio_dynamic"].to_numpy(),
            out2["debt_coverage_ratio_dynamic"].to_numpy(),
        ))
        self.assertAlmostEqual(
            out1["project_npv_dynamic_musd"].iloc[0],
            out2["project_npv_dynamic_musd"].iloc[0],
            places=10,
        )

    def test_macro_tif_profile_fallback_nonnegative(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                {
                    "year": [0, 1, 2, 0, 1, 2],
                    "baseline_value_dollars": [100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
                    "projected_value_dollars": [300.0, 340.0, 380.0, 600.0, 660.0, 720.0],
                }
            )
            df.to_csv(processed / "property_tax_uplift_zoning.csv", index=False)
            profile = apm_eval._load_macro_tif_profile(
                "zoning",
                years=2,
                data_dir=Path(tmpdir),
                baseline_growth_rate=0.02,
            )
            self.assertIsNotNone(profile)
            self.assertGreater(profile["base_total"], 0.0)
            self.assertGreater(profile["future_total"], profile["base_total"])
            self.assertGreaterEqual(profile["cumulative_tif_total"], 0.0)


if __name__ == "__main__":
    unittest.main()

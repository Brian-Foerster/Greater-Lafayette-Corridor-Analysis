"""Week 19 tests for dynamic ridership integration in financial ranking."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    import sys
    import scripts.financial_corridor_ranking as fcr
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    fcr = None
    _IMPORT_ERROR = exc


class TestFinancialRankingDynamic(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def _base_corridors(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "corridor_id": ["C1", "C2"],
                "length_km": [5.0, 5.0],
                "n_stops": [6, 6],
                "daily_ridership": [100.0, 100.0],
                "base_property_value": [100_000_000.0, 100_000_000.0],
                "future_property_value": [130_000_000.0, 130_000_000.0],
                "daily_ridership_series": [
                    [100.0, 150.0, 200.0, 250.0, 300.0],
                    [100.0, 100.0, 100.0, 100.0, 100.0],
                ],
            }
        )

    def test_ranker_uses_dynamic_riders_for_efficiency(self):
        ranker = fcr.FinancialCorridorRanker(
            ridership_weight=0.50,
            tif_weight=0.30,
            efficiency_weight=0.20,
            min_debt_coverage=1.0,
        )
        ranked = ranker.rank_corridors(
            self._base_corridors(),
            capital_cost_per_km=2_000_000.0,
            interest_rate=0.03,
            term_years=20,
            cashflow_years=5,
            fare_per_trip_usd=2.0,
            farebox_capture_rate=1.0,
        )
        self.assertTrue(bool(ranked["has_dynamic_ridership"].all()))

        c1 = ranked[ranked["corridor_id"] == "C1"].iloc[0]
        c2 = ranked[ranked["corridor_id"] == "C2"].iloc[0]

        # Same costs, higher dynamic ridership should improve cost per rider.
        self.assertLess(c1["cost_per_rider"], c2["cost_per_rider"])
        self.assertGreater(c1["annual_ridership_effective"], c2["annual_ridership_effective"])
        self.assertGreaterEqual(c1["farebox_revenue_npv_musd"], c2["farebox_revenue_npv_musd"])

    def test_invalid_dynamic_series_raises(self):
        ranker = fcr.FinancialCorridorRanker(
            ridership_weight=0.50,
            tif_weight=0.30,
            efficiency_weight=0.20,
            min_debt_coverage=1.0,
        )
        bad = self._base_corridors().copy()
        bad = bad.astype({"daily_ridership_series": "object"})
        bad.at[0, "daily_ridership_series"] = [100.0, 120.0]  # wrong length for 5-year cashflow

        with self.assertRaises(ValueError):
            ranker.rank_corridors(
                bad,
                capital_cost_per_km=2_000_000.0,
                interest_rate=0.03,
                term_years=20,
                cashflow_years=5,
            )

    def test_integrate_dynamic_ridership_data_from_feedback_file(self):
        corridors = pd.DataFrame(
            {
                "corridor_id": ["C1", "C2"],
                "length_km": [5.0, 5.0],
                "daily_ridership": [0.0, 0.0],
            }
        )
        dynamic = pd.DataFrame(
            {
                "corridor_id": ["C1", "C1", "C2", "C2"],
                "year": [0, 5, 0, 5],
                "daily_riders": [100.0, 200.0, 50.0, 80.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "feedback.csv"
            dynamic.to_csv(p, index=False)
            enriched = fcr.integrate_dynamic_ridership_data(
                corridors,
                dynamic_ridership_path=p,
                cashflow_years=5,
            )
            self.assertIn("daily_ridership_series", enriched.columns)
            self.assertEqual(len(enriched.loc[0, "daily_ridership_series"]), 5)


if __name__ == "__main__":
    unittest.main()

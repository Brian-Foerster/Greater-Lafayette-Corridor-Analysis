"""Week 19 tests for dynamic annual finance cashflows."""

import unittest

import numpy as np

try:
    from src.finance import (
        annual_ridership_to_revenue_musd,
        annualize_daily_ridership_series,
        npv_irr,
    )
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    annual_ridership_to_revenue_musd = None
    annualize_daily_ridership_series = None
    npv_irr = None
    _IMPORT_ERROR = exc


class TestDynamicFinanceCashflows(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_scalar_revenue_backward_compatible(self):
        fin = npv_irr(
            annual_revenue_musd=2.0,
            capex_musd_val=10.0,
            years=5,
            discount_rate=0.0,
        )
        self.assertAlmostEqual(fin["npv_musd"], 0.0, places=8)
        self.assertEqual(len(fin["annual_net_cashflow_musd"]), 5)
        self.assertTrue(np.allclose(fin["annual_net_cashflow_musd"], np.full(5, 2.0)))

    def test_sequence_length_validation(self):
        with self.assertRaises(ValueError):
            npv_irr(
                annual_revenue_musd=[1.0, 2.0, 3.0],
                capex_musd_val=5.0,
                years=5,
            )

    def test_mapping_year_keys_supported(self):
        fin_zero_based = npv_irr(
            annual_revenue_musd={0: 1.0, 1: 1.0, 2: 1.0},
            annual_cost_musd=0.0,
            capex_musd_val=1.0,
            years=3,
            discount_rate=0.0,
        )
        fin_one_based = npv_irr(
            annual_revenue_musd={1: 1.0, 2: 1.0, 3: 1.0},
            annual_cost_musd=0.0,
            capex_musd_val=1.0,
            years=3,
            discount_rate=0.0,
        )
        self.assertAlmostEqual(fin_zero_based["npv_musd"], fin_one_based["npv_musd"], places=8)

    def test_daily_to_annual_and_revenue_conversion(self):
        # Default operating_days_per_year is 312 (academic calendar weighted).
        annual = annualize_daily_ridership_series([100.0, 200.0], years=2)
        self.assertTrue(np.allclose(annual, np.array([31200.0, 62400.0])))

        revenue = annual_ridership_to_revenue_musd(
            annual,
            fare_per_trip_usd=2.0,
            capture_rate=0.5,
            years=2,
        )
        self.assertTrue(np.allclose(revenue, np.array([0.0312, 0.0624])))

    def test_negative_values_rejected(self):
        with self.assertRaises(ValueError):
            annualize_daily_ridership_series([-1.0, 2.0], years=2)
        with self.assertRaises(ValueError):
            npv_irr(
                annual_revenue_musd=[1.0, 2.0],
                annual_cost_musd=[0.0, -1.0],
                capex_musd_val=5.0,
                years=2,
            )


if __name__ == "__main__":
    unittest.main()

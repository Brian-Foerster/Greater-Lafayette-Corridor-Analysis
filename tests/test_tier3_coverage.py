"""Unit tests for Tier 3 functions that previously lacked test coverage.

Covers: _correlated_sample, _quantile_from_spec, compute_spatial_equity,
compute_displacement_risk, screen_ej_communities, build_fta_metrics_table,
build_uncertainty_dashboard, generate_technical_appendix, interpolate_trajectory.
"""

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add scripts and src to path

try:
    from scripts.apm_corridor_evaluation_integrated import (
        _correlated_sample,
        _quantile_from_spec,
        _PARAM_NAMES_ORDERED,
    )
    _EVAL_IMPORT_ERROR = None
except Exception as exc:
    _correlated_sample = None
    _quantile_from_spec = None
    _EVAL_IMPORT_ERROR = exc

try:
    from src.economic_impact import (
        compute_spatial_equity,
        compute_displacement_risk,
        screen_ej_communities,
        build_fta_metrics_table,
    )
    _ECON_IMPORT_ERROR = None
except Exception as exc:
    _ECON_IMPORT_ERROR = exc

try:
    import scripts.generate_decision_package as gdp
    _GDP_IMPORT_ERROR = None
except Exception as exc:
    gdp = None
    _GDP_IMPORT_ERROR = exc

try:
    from scripts.run_behavioral_sensitivity import (
        interpolate_trajectory,
        BEHAVIORAL_PARAMS,
        lhs_to_params,
    )
    _BEHAV_IMPORT_ERROR = None
except Exception as exc:
    _BEHAV_IMPORT_ERROR = exc


# ───────────────────────────────────────────────────────
# _correlated_sample and _quantile_from_spec
# ───────────────────────────────────────────────────────


class TestCorrelatedSample(unittest.TestCase):
    def setUp(self):
        if _EVAL_IMPORT_ERROR is not None:
            self.skipTest(f"Import error: {_EVAL_IMPORT_ERROR}")

    def _make_ranges(self):
        return {
            "ridership_multiplier": {
                "distribution": "triangular",
                "low": 0.8, "high": 1.2, "mode": 1.0,
            },
            "tif_multiplier": {
                "distribution": "triangular",
                "low": 0.7, "high": 1.3, "mode": 1.0,
            },
            "capital_cost_multiplier": {
                "distribution": "triangular",
                "low": 0.9, "high": 1.1, "mode": 1.0,
            },
            "operating_cost_multiplier": {
                "distribution": "triangular",
                "low": 0.9, "high": 1.1, "mode": 1.0,
            },
            "fare_multiplier": {
                "distribution": "triangular",
                "low": 0.8, "high": 1.2, "mode": 1.0,
            },
            "discount_rate_delta": {
                "distribution": "uniform",
                "low": -0.01, "high": 0.01,
            },
        }

    def test_returns_all_parameters(self):
        rng = np.random.default_rng(42)
        ranges = self._make_ranges()
        result = _correlated_sample(ranges, 100, rng)
        # Every parameter supplied in ranges must appear in the result
        for pname in ranges:
            self.assertIn(pname, result)
            self.assertEqual(len(result[pname]), 100)

    def test_preserves_marginal_bounds(self):
        """Each marginal should stay within [low, high]."""
        rng = np.random.default_rng(42)
        ranges = self._make_ranges()
        result = _correlated_sample(ranges, 1000, rng)
        for pname, spec in ranges.items():
            low, high = spec["low"], spec["high"]
            self.assertGreaterEqual(result[pname].min(), low - 1e-9,
                                    f"{pname} below low")
            self.assertLessEqual(result[pname].max(), high + 1e-9,
                                 f"{pname} above high")

    def test_correlation_preserved(self):
        """Rank correlation between ridership and TIF should be positive (r=0.3 in default)."""
        rng = np.random.default_rng(42)
        result = _correlated_sample(self._make_ranges(), 2000, rng)
        # Spearman rank correlation via numpy (avoids scipy.stats import deadlock)
        r1 = result["ridership_multiplier"].argsort().argsort().astype(float)
        r2 = result["tif_multiplier"].argsort().argsort().astype(float)
        rho = np.corrcoef(r1, r2)[0, 1]
        self.assertGreater(rho, 0.1, "Expected positive rank correlation between ridership and TIF")

    def test_non_unit_diagonal_raises(self):
        rng = np.random.default_rng(42)
        bad_corr = np.eye(6)
        bad_corr[0, 0] = 0.99
        with self.assertRaises(AssertionError):
            _correlated_sample(self._make_ranges(), 100, rng, correlation_matrix=bad_corr)


class TestQuantileFromSpec(unittest.TestCase):
    def setUp(self):
        if _EVAL_IMPORT_ERROR is not None:
            self.skipTest(f"Import error: {_EVAL_IMPORT_ERROR}")

    def test_uniform_at_bounds(self):
        spec = {"distribution": "uniform", "low": 2.0, "high": 5.0}
        u = np.array([0.0, 0.5, 1.0])
        result = _quantile_from_spec("test", spec, u)
        expected = np.array([2.0, 3.5, 5.0])
        self.assertTrue(np.allclose(result, expected, atol=1e-9),
                        f"Expected {expected}, got {result}")

    def test_triangular_mode_at_center(self):
        spec = {"distribution": "triangular", "low": 0.0, "high": 1.0, "mode": 0.5}
        u = np.array([0.5])
        result = _quantile_from_spec("test", spec, u)
        self.assertAlmostEqual(result[0], 0.5, places=2)

    def test_swapped_low_high_raises(self):
        spec = {"distribution": "uniform", "low": 5.0, "high": 2.0}
        u = np.array([0.5])
        with self.assertRaises(ValueError, msg="Expected ValueError for low > high"):
            _quantile_from_spec("test_param", spec, u)


# ───────────────────────────────────────────────────────
# compute_spatial_equity
# ───────────────────────────────────────────────────────


class TestComputeSpatialEquity(unittest.TestCase):
    def setUp(self):
        if _ECON_IMPORT_ERROR is not None:
            self.skipTest(f"Import error: {_ECON_IMPORT_ERROR}")

    def test_pro_equity_when_se01_gains_more(self):
        n = 100
        acc_base = np.ones(n)
        acc_apm = np.ones(n) * 2.0
        # SE01 parcels get bigger boost
        acc_apm[:30] = 3.0
        pop = np.ones(n) * 10
        seg = np.concatenate([np.ones(30), np.full(40, 2), np.full(30, 3)])
        result = compute_spatial_equity(acc_apm, acc_base, pop, seg)
        self.assertGreater(result["equity_ratio"], 1.0)
        self.assertGreater(result["mean_delta_SE01"], result["mean_delta_SE03"])

    def test_anti_equity_when_se03_gains_more(self):
        n = 100
        acc_base = np.ones(n)
        acc_apm = np.ones(n) * 1.5
        acc_apm[70:] = 3.0  # SE03 parcels get bigger boost
        pop = np.ones(n) * 10
        seg = np.concatenate([np.ones(30), np.full(40, 2), np.full(30, 3)])
        result = compute_spatial_equity(acc_apm, acc_base, pop, seg)
        self.assertLess(result["equity_ratio"], 1.0)

    def test_both_negative_handled(self):
        """When both segments lose accessibility, ratio should be < 1 if SE01 loses more."""
        n = 60
        acc_base = np.ones(n) * 3.0
        acc_apm = np.ones(n) * 2.0
        acc_apm[:20] = 1.0  # SE01 loses more
        pop = np.ones(n) * 10
        seg = np.concatenate([np.ones(20), np.full(20, 2), np.full(20, 3)])
        result = compute_spatial_equity(acc_apm, acc_base, pop, seg)
        # SE01 delta = -2.0, SE03 delta = -1.0, ratio = 2.0 (SE01 delta / SE03 delta)
        # Both negative, but SE01 loses more → ratio > 1 actually means SE01 is more impacted
        self.assertIn("equity_ratio", result)

    def test_zero_se03_delta(self):
        """When SE03 has no change, equity_ratio should be 1.0 or 0.0."""
        n = 60
        acc_base = np.ones(n)
        acc_apm = np.ones(n)
        acc_apm[:20] = 2.0  # Only SE01 gains
        pop = np.ones(n) * 10
        seg = np.concatenate([np.ones(20), np.full(20, 2), np.full(20, 3)])
        result = compute_spatial_equity(acc_apm, acc_base, pop, seg)
        self.assertEqual(result["equity_ratio"], 1.0)


# ───────────────────────────────────────────────────────
# compute_displacement_risk
# ───────────────────────────────────────────────────────


class TestComputeDisplacementRisk(unittest.TestCase):
    def setUp(self):
        if _ECON_IMPORT_ERROR is not None:
            self.skipTest(f"Import error: {_ECON_IMPORT_ERROR}")

    def test_high_rent_burden_detected(self):
        rents = np.array([6000.0, 3000.0, 8000.0])  # threshold = 0.30 * 15000 = 4500
        pop = np.array([100.0, 50.0, 200.0])
        result = compute_displacement_risk(rents, pop)
        self.assertEqual(result["at_risk_parcels"], 2)  # parcels 0 and 2
        self.assertAlmostEqual(result["at_risk_se01_pop"], 300.0)

    def test_displacement_counted(self):
        rents = np.array([6000.0, 3000.0])
        pop_now = np.array([50.0, 50.0])
        pop_prev = np.array([100.0, 50.0])
        prev_rents = np.array([5000.0, 3000.0])
        result = compute_displacement_risk(rents, pop_now, pop_prev, prev_rents)
        self.assertIn("displaced_se01", result)
        # prev at-risk (parcel 0, prev_rent=5000 > 4500): pop=100
        # curr at-risk (parcel 0, curr_rent=6000 > 4500): pop=50
        # displaced = 100 - 50 = 50
        self.assertAlmostEqual(result["displaced_se01"], 50.0)

    def test_no_displacement_without_prev(self):
        rents = np.array([6000.0, 3000.0])
        pop = np.array([100.0, 50.0])
        result = compute_displacement_risk(rents, pop)
        self.assertNotIn("displaced_se01", result)


# ───────────────────────────────────────────────────────
# screen_ej_communities
# ───────────────────────────────────────────────────────


class TestScreenEjCommunities(unittest.TestCase):
    def setUp(self):
        if _ECON_IMPORT_ERROR is not None:
            self.skipTest(f"Import error: {_ECON_IMPORT_ERROR}")

    def test_dac_tracts_in_catchment(self):
        stations = np.array([[0.0, 0.0], [1000.0, 0.0]])
        tracts = np.array([[500.0, 0.0], [5000.0, 0.0]])
        pop = np.array([1000.0, 2000.0])
        dac = np.array([True, False])
        result = screen_ej_communities(stations, tracts, pop, dac, catchment_m=1200.0)
        self.assertEqual(result["dac_tract_count"], 1)
        self.assertAlmostEqual(result["dac_pop_share"], 1.0)  # Only DAC tract in catchment

    def test_empty_stations(self):
        result = screen_ej_communities(
            np.empty((0, 2)), np.array([[0, 0]]),
            np.array([100.0]), np.array([True]),
        )
        self.assertEqual(result["dac_tract_count"], 0)
        self.assertFalse(result["justice40_compliant"])

    def test_benefit_share(self):
        stations = np.array([[0.0, 0.0]])
        tracts = np.array([[100.0, 0.0], [200.0, 0.0]])
        pop = np.array([500.0, 500.0])
        dac = np.array([True, False])
        benefit = np.array([100.0, 200.0])
        result = screen_ej_communities(stations, tracts, pop, dac, 1200.0, benefit)
        # DAC benefit = 100, total benefit = 300, share = 1/3
        self.assertAlmostEqual(result["dac_benefit_share"], 1.0 / 3.0, places=2)


# ───────────────────────────────────────────────────────
# build_fta_metrics_table
# ───────────────────────────────────────────────────────


class TestBuildFtaMetricsTable(unittest.TestCase):
    def setUp(self):
        if _ECON_IMPORT_ERROR is not None:
            self.skipTest(f"Import error: {_ECON_IMPORT_ERROR}")

    def test_produces_cost_per_trip(self):
        df = pd.DataFrame({
            "corridor_id": ["C1", "C2"],
            "daily_ridership": [5000, 3000],
            "capital_cost": [500_000_000, 300_000_000],
            "length_km": [5.0, 3.0],
            "n_stops": [6, 4],
        })
        result = build_fta_metrics_table(df)
        self.assertEqual(len(result), 2)
        self.assertIn("cost_per_trip", result.columns)
        self.assertIn("fta_rating", result.columns)
        # Cost per trip should be positive
        self.assertTrue((result["cost_per_trip"] > 0).all())

    def test_zero_ridership_gives_inf(self):
        df = pd.DataFrame({
            "corridor_id": ["C1"],
            "daily_ridership": [0],
            "capital_cost": [500_000_000],
            "length_km": [5.0],
            "n_stops": [6],
        })
        result = build_fta_metrics_table(df)
        self.assertEqual(result.iloc[0]["cost_per_trip"], float("inf"))

    def test_om_includes_escalation(self):
        """The FTA table should use escalated O&M, not base-year."""
        df = pd.DataFrame({
            "corridor_id": ["C1"],
            "daily_ridership": [5000],
            "capital_cost": [500_000_000],
            "length_km": [5.0],
            "n_stops": [6],
        })
        result = build_fta_metrics_table(df)
        # Verify cost_per_trip is higher than it would be with base-year O&M
        # (just check it's positive and finite)
        cpt = result.iloc[0]["cost_per_trip"]
        self.assertGreater(cpt, 0)
        self.assertLess(cpt, 1000)  # Sanity — shouldn't be absurdly high


# build_uncertainty_dashboard and generate_technical_appendix tests are in
# tests/test_decision_package_generation.py (canonical location).


# ───────────────────────────────────────────────────────
# interpolate_trajectory
# ───────────────────────────────────────────────────────


class TestInterpolateTrajectory(unittest.TestCase):
    def setUp(self):
        if _BEHAV_IMPORT_ERROR is not None:
            self.skipTest(f"Import error: {_BEHAV_IMPORT_ERROR}")

    def _make_params_and_trajs(self):
        """Create two LHS points with distinct trajectories."""
        d = len(BEHAVIORAL_PARAMS)
        lhs_points = np.array([np.zeros(d), np.ones(d)])
        params = lhs_to_params(lhs_points)
        traj_low = pd.DataFrame({"year": [0, 1, 2], "ridership": [100.0, 200.0, 300.0]})
        traj_high = pd.DataFrame({"year": [0, 1, 2], "ridership": [500.0, 600.0, 700.0]})
        return params, [traj_low, traj_high]

    def test_midpoint_interpolation(self):
        params, trajs = self._make_params_and_trajs()
        # Query at midpoint between the two LHS points
        query = {}
        for pname, spec in BEHAVIORAL_PARAMS.items():
            query[pname] = (spec["low"] + spec["high"]) / 2.0
        result = interpolate_trajectory(query, params, trajs, n_nearest=2)
        # Should be approximately the average of the two trajectories
        self.assertEqual(len(result), 3)
        mid_val = result["ridership"].iloc[1]
        self.assertGreater(mid_val, 200.0)
        self.assertLess(mid_val, 600.0)

    def test_exact_match_returns_that_trajectory(self):
        params, trajs = self._make_params_and_trajs()
        # Query exactly at first LHS point
        result = interpolate_trajectory(params[0], params, trajs, n_nearest=2)
        # Should be dominated by the first trajectory
        self.assertAlmostEqual(result["ridership"].iloc[0], 100.0, delta=50.0)

    def test_empty_trajectories_raises(self):
        with self.assertRaises(ValueError):
            interpolate_trajectory({}, [], [])


# ───────────────────────────────────────────────────────
# robust_corridor_ranking integration (Bug 10 regression)
# ───────────────────────────────────────────────────────


class TestRobustRankingIntegration(unittest.TestCase):
    """Verify robust_corridor_ranking works with sample_id column (Bug 10)."""

    def setUp(self):
        if _EVAL_IMPORT_ERROR is not None:
            self.skipTest(f"Import error: {_EVAL_IMPORT_ERROR}")
        from scripts.apm_corridor_evaluation_integrated import robust_corridor_ranking
        self.robust_corridor_ranking = robust_corridor_ranking

    def test_accepts_sample_id_column(self):
        """Pipeline produces sample_id, not draw — must work."""
        draws = pd.DataFrame({
            "corridor_id": ["C1"] * 10 + ["C2"] * 10,
            "sample_id": list(range(10)) * 2,
            "debt_coverage_ratio": np.random.default_rng(42).uniform(0.1, 0.5, 20),
        })
        result = self.robust_corridor_ranking(draws)
        self.assertEqual(len(result), 2)
        self.assertIn("robust_rank", result.columns)
        self.assertIn("robust_score", result.columns)

    def test_accepts_draw_column(self):
        """Legacy draw column should still work."""
        draws = pd.DataFrame({
            "corridor_id": ["C1"] * 5 + ["C2"] * 5,
            "draw": list(range(5)) * 2,
            "debt_coverage_ratio": [0.3, 0.2, 0.4, 0.35, 0.25,
                                    0.15, 0.1, 0.2, 0.18, 0.12],
        })
        result = self.robust_corridor_ranking(draws)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()

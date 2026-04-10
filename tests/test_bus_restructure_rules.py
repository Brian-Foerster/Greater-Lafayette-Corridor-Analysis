"""Week 9 unit tests for GTFS-informed bus restructure rules."""

import unittest

import pandas as pd

try:
    from src.gtfs_ridership import compute_route_competitiveness_metrics
    from src.land_use_transport_model import _restructure_pressure
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    compute_route_competitiveness_metrics = None
    _restructure_pressure = None
    _IMPORT_ERROR = exc


class TestGtfsCompetitivenessMetrics(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_lower_headway_yields_higher_competitiveness(self):
        route_headways = pd.DataFrame(
            {
                "route_id": ["R_fast", "R_slow"],
                "avg_headway_min": [10.0, 45.0],
                "trips_per_day": [40.0, 40.0],
            }
        )
        metrics = compute_route_competitiveness_metrics(route_headways)
        fast = float(metrics.loc[metrics["route_id"] == "R_fast", "competitiveness_score"].iloc[0])
        slow = float(metrics.loc[metrics["route_id"] == "R_slow", "competitiveness_score"].iloc[0])
        self.assertGreater(fast, slow)

    def test_productivity_signal_affects_scores(self):
        route_headways = pd.DataFrame(
            {
                "route_id": ["R1", "R2"],
                "avg_headway_min": [20.0, 20.0],
                "trips_per_day": [30.0, 30.0],
            }
        )
        observed = pd.DataFrame(
            {
                "route_id": ["R1", "R2"],
                "observed_passengers": [5000.0, 500.0],
            }
        )
        metrics = compute_route_competitiveness_metrics(
            route_headways,
            observed_productivity=observed,
        )
        r1 = float(metrics.loc[metrics["route_id"] == "R1", "competitiveness_score"].iloc[0])
        r2 = float(metrics.loc[metrics["route_id"] == "R2", "competitiveness_score"].iloc[0])
        self.assertGreater(r1, r2)


class TestRestructurePressure(unittest.TestCase):
    """Tests for the _restructure_pressure helper."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_zero_ridership_gives_low_pressure(self):
        p = _restructure_pressure(0.0, 3000.0, 0.5, 0.5)
        # 0.70*0 + 0.20*0.5 + 0.10*0.5 = 0.15
        self.assertAlmostEqual(p, 0.15, places=3)

    def test_mature_ridership_gives_high_pressure(self):
        p = _restructure_pressure(3000.0, 3000.0, 0.2, 0.3)
        # 0.70*1.0 + 0.20*0.8 + 0.10*0.7 = 0.93
        self.assertAlmostEqual(p, 0.93, places=3)

    def test_high_competitiveness_lowers_pressure(self):
        low_comp = _restructure_pressure(2000.0, 3000.0, 0.2, 0.5)
        high_comp = _restructure_pressure(2000.0, 3000.0, 0.9, 0.5)
        self.assertGreater(low_comp, high_comp)

    def test_clipped_to_unit_interval(self):
        p = _restructure_pressure(5000.0, 1000.0, 0.0, 0.0)
        self.assertLessEqual(p, 1.0)
        p2 = _restructure_pressure(0.0, 3000.0, 1.0, 1.0)
        self.assertGreaterEqual(p2, 0.0)


if __name__ == "__main__":
    unittest.main()

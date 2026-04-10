"""Tests for src/zoning_population.py — population/jobs allocation utilities."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


class TestCapAndRedistribute(unittest.TestCase):
    """Test the iterative cap-and-redistribute allocation."""

    def test_all_within_caps(self):
        from src.zoning_population import _cap_and_redistribute

        weights = np.array([1.0, 2.0, 3.0])
        caps = np.array([100.0, 200.0, 300.0])
        result = _cap_and_redistribute(60.0, weights, caps)

        np.testing.assert_allclose(result.sum(), 60.0, atol=0.1)
        # Proportional: 10, 20, 30
        np.testing.assert_allclose(result, [10.0, 20.0, 30.0], atol=0.1)

    def test_one_parcel_capped(self):
        from src.zoning_population import _cap_and_redistribute

        weights = np.array([1.0, 1.0, 1.0])
        caps = np.array([5.0, 100.0, 100.0])
        result = _cap_and_redistribute(30.0, weights, caps)

        # Parcel 0 capped at 5, remaining 25 split between 1 and 2
        self.assertAlmostEqual(result[0], 5.0, places=1)
        np.testing.assert_allclose(result.sum(), 30.0, atol=0.1)

    def test_zero_population(self):
        from src.zoning_population import _cap_and_redistribute

        weights = np.array([1.0, 2.0])
        caps = np.array([100.0, 200.0])
        result = _cap_and_redistribute(0.0, weights, caps)

        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_all_caps_exceeded(self):
        from src.zoning_population import _cap_and_redistribute

        weights = np.array([1.0, 1.0])
        caps = np.array([10.0, 10.0])
        result = _cap_and_redistribute(100.0, weights, caps)

        # Can only allocate 20 total
        np.testing.assert_allclose(result, [10.0, 10.0])


class TestChooseDensity(unittest.TestCase):
    """Test zone-to-density lookup with PD fallback."""

    def _make_density_df(self):
        return pd.DataFrame({
            "zone": ["R1", "R3", "GB", "PDMX"],
            "pop_density": [5.0, 15.0, 2.0, 10.0],
            "job_density": [1.0, 3.0, 8.0, 5.0],
        })

    def test_exact_match(self):
        from src.zoning_population import _choose_density

        df = self._make_density_df()
        pop, job = _choose_density("R3", df)
        self.assertEqual(pop, 15.0)
        self.assertEqual(job, 3.0)

    def test_pd_fallback_to_pdmx(self):
        from src.zoning_population import _choose_density

        df = self._make_density_df()
        pop, job = _choose_density("PDRS", df)
        # Should fall back to PDMX
        self.assertEqual(pop, 10.0)
        self.assertEqual(job, 5.0)

    def test_unknown_zone_gets_median(self):
        from src.zoning_population import _choose_density

        df = self._make_density_df()
        pop, job = _choose_density("UNKNOWN_ZONE", df)
        # Should get median of all zones
        expected_pop = df["pop_density"].median()
        expected_job = df["job_density"].median()
        self.assertEqual(pop, expected_pop)
        self.assertEqual(job, expected_job)


if __name__ == "__main__":
    unittest.main()

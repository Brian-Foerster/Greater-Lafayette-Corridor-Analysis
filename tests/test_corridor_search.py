"""Minimal tests for scripts/optimized_corridor_search.py.

Tests pure functions (validate_station_set, deduplicate_station_sets) with
synthetic station data. No real road graph or GIS files required.
"""
from __future__ import annotations

import unittest

import numpy as np


def _make_station_data(n_stations: int = 10, spacing_ft: float = 3000.0):
    """Create synthetic station_data dict with evenly-spaced stations.

    Coordinates are in EPSG:2965 feet (the CRS used by the corridor search).
    3000 ft ≈ 914m — within the valid stop spacing range (500-1500m).
    """
    # Linear arrangement along x-axis
    coords = np.zeros((n_stations, 2), dtype=np.float64)
    coords[:, 0] = np.arange(n_stations) * spacing_ft
    coords[:, 1] = 0.0

    return {
        "coords_proj": coords,
        "node_ids": list(range(n_stations)),
    }


class TestValidateStationSet(unittest.TestCase):
    """Test corridor validation logic."""

    def test_valid_corridor(self):
        from scripts.optimized_corridor_search import validate_station_set

        # 6 stations at 914m spacing → ~5.5km corridor (within 2-25km range)
        sd = _make_station_data(10, spacing_ft=3000.0)
        ok, msg = validate_station_set([0, 1, 2, 3, 4, 5], sd)
        self.assertTrue(ok, f"Should be valid: {msg}")

    def test_too_few_stations(self):
        from scripts.optimized_corridor_search import validate_station_set

        sd = _make_station_data(10)
        ok, msg = validate_station_set([0, 1, 2], sd)
        self.assertFalse(ok)
        self.assertIn("too few", msg)

    def test_duplicate_stations_rejected(self):
        from scripts.optimized_corridor_search import validate_station_set

        sd = _make_station_data(10)
        ok, msg = validate_station_set([0, 1, 2, 2, 3], sd)
        self.assertFalse(ok)
        self.assertIn("duplicate", msg)

    def test_too_short_corridor(self):
        from scripts.optimized_corridor_search import validate_station_set

        # 4 stations at tiny spacing → corridor too short (<2km)
        sd = _make_station_data(10, spacing_ft=500.0)  # 500ft ≈ 152m
        ok, msg = validate_station_set([0, 1, 2, 3], sd)
        self.assertFalse(ok)
        # Either "too short" or "spacing too close"
        self.assertTrue("too short" in msg or "spacing" in msg, msg)

    def test_stations_too_close(self):
        from scripts.optimized_corridor_search import validate_station_set

        # Adjacent stations at 100ft ≈ 30m — well below 500m minimum
        sd = _make_station_data(10, spacing_ft=100.0)
        ok, msg = validate_station_set([0, 1, 2, 3], sd)
        self.assertFalse(ok)
        self.assertIn("spacing", msg)


class TestDeduplicateStationSets(unittest.TestCase):
    """Test corridor deduplication."""

    def test_identical_corridors_deduped(self):
        from scripts.optimized_corridor_search import deduplicate_station_sets

        sd = _make_station_data(10, spacing_ft=3000.0)
        candidates = [
            {"stations": [0, 1, 2, 3, 4, 5], "score": 100},
            {"stations": [0, 1, 2, 3, 4, 5], "score": 90},  # identical
        ]
        result = deduplicate_station_sets(candidates, sd)
        self.assertEqual(len(result), 1)

    def test_distinct_corridors_kept(self):
        from scripts.optimized_corridor_search import deduplicate_station_sets

        # 20 stations spread far apart so two non-overlapping corridors exist
        sd = _make_station_data(20, spacing_ft=3000.0)
        candidates = [
            {"stations": [0, 1, 2, 3, 4, 5], "score": 100},
            {"stations": [10, 11, 12, 13, 14, 15], "score": 90},  # far away
        ]
        result = deduplicate_station_sets(candidates, sd)
        self.assertEqual(len(result), 2)

    def test_empty_input(self):
        from scripts.optimized_corridor_search import deduplicate_station_sets

        sd = _make_station_data(10)
        result = deduplicate_station_sets([], sd)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()

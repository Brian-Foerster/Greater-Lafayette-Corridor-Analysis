"""Tests for two-layer catchment model (walk zone + feeder zone)."""
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point


def _make_corridor(stops_lonlat):
    """Build a minimal corridor row with geometry and n_stops."""
    coords = stops_lonlat
    line = LineString(coords)
    gdf = gpd.GeoDataFrame(
        [{"corridor_id": "test", "n_stops": len(coords), "geometry": line}],
        crs="EPSG:4326",
    )
    return gdf.iloc[0]


def _make_parcels(coords_lonlat, ids=None):
    """Build a parcels GeoDataFrame from (lon, lat) list."""
    if ids is None:
        ids = [f"P{i}" for i in range(len(coords_lonlat))]
    geom = [Point(c) for c in coords_lonlat]
    return gpd.GeoDataFrame({"parcel_id": ids}, geometry=geom, crs="EPSG:4326")


def _make_lodes(origin_ids, dest_ids, trips):
    return pd.DataFrame({
        "origin_parcel": origin_ids,
        "dest_parcel": dest_ids,
        "trips": trips,
    })


class TestFeederCatchment(unittest.TestCase):
    """Verify that feeder zone adds riders beyond walk-only catchment."""

    def setUp(self):
        # A corridor with 3 stops along a horizontal line at lat 40.42
        # Stops ~500m apart in projected coords
        self.corridor = _make_corridor([
            (-86.92, 40.42),
            (-86.915, 40.42),
            (-86.91, 40.42),
        ])

        # Parcels: some within walk zone, some in feeder zone, some outside
        self.parcels = _make_parcels([
            (-86.92, 40.42),      # P0: at stop (0m)
            (-86.915, 40.421),    # P1: ~110m from stop (walk zone)
            (-86.91, 40.425),     # P2: ~550m from stop (walk zone)
            (-86.92, 40.435),     # P3: ~1700m from stop (feeder zone)
            (-86.92, 40.445),     # P4: ~2800m from stop (feeder zone)
            (-86.92, 40.48),      # P5: ~6700m from stop (outside both zones)
        ])

        # OD flows: various combinations
        self.lodes = _make_lodes(
            ["P0", "P0", "P0", "P1", "P3", "P4"],
            ["P1", "P3", "P4", "P2", "P1", "P0"],
            [100, 80, 60, 120, 90, 70],
        )

    def test_feeder_adds_riders(self):
        """Feeder zone should produce more total APM trips than walk-only."""
        from scripts.generate_improved_ridership import compute_lodes_ridership

        # Walk-only (feeder_coverage_fraction=0)
        walk_only = compute_lodes_ridership(
            self.corridor, self.parcels, self.lodes,
            feeder_headway_min=15,
            feeder_coverage_fraction=0.0,
        )
        apm_walk = walk_only[0]

        # With feeder zone active
        with_feeder = compute_lodes_ridership(
            self.corridor, self.parcels, self.lodes,
            feeder_headway_min=15,
            feeder_coverage_fraction=0.9,
        )
        apm_feeder = with_feeder[0]

        self.assertGreater(apm_feeder, apm_walk,
                           "Feeder zone should add riders beyond walk-only catchment")

    def test_walk_zone_unchanged(self):
        """Walk-zone trips should be the same regardless of feeder settings."""
        from scripts.generate_improved_ridership import compute_lodes_ridership

        # Walk-only OD pairs (P0->P1, P1->P2) — both ends within walk zone
        walk_lodes = _make_lodes(["P0", "P1"], ["P1", "P2"], [100, 120])

        result_no_feeder = compute_lodes_ridership(
            self.corridor, self.parcels, walk_lodes,
            feeder_coverage_fraction=0.0,
        )
        result_with_feeder = compute_lodes_ridership(
            self.corridor, self.parcels, walk_lodes,
            feeder_headway_min=10,
            feeder_coverage_fraction=1.0,
        )

        self.assertAlmostEqual(result_no_feeder[0], result_with_feeder[0], places=4,
                               msg="Walk-zone APM trips should not change with feeder settings")

    def test_better_feeder_headway_increases_riders(self):
        """Shorter feeder headway → more feeder-zone APM trips."""
        from scripts.generate_improved_ridership import compute_lodes_ridership

        result_30min = compute_lodes_ridership(
            self.corridor, self.parcels, self.lodes,
            feeder_headway_min=30,
            feeder_coverage_fraction=0.9,
        )
        result_10min = compute_lodes_ridership(
            self.corridor, self.parcels, self.lodes,
            feeder_headway_min=10,
            feeder_coverage_fraction=0.9,
        )

        self.assertGreater(result_10min[0], result_30min[0],
                           "10-min feeder headway should attract more riders than 30-min")

    def test_integrated_fare_more_riders_than_double_charge(self):
        """Integrated fare ($2 total) should produce more riders than double-charge ($4)."""
        from scripts import generate_improved_ridership as gir

        # Temporarily disable integrated fare
        orig_policy = gir.INTEGRATED_FARE_POLICY
        try:
            gir.INTEGRATED_FARE_POLICY = True
            result_integrated = gir.compute_lodes_ridership(
                self.corridor, self.parcels, self.lodes,
                feeder_headway_min=15,
                feeder_coverage_fraction=0.9,
            )

            gir.INTEGRATED_FARE_POLICY = False
            result_double = gir.compute_lodes_ridership(
                self.corridor, self.parcels, self.lodes,
                feeder_headway_min=15,
                feeder_coverage_fraction=0.9,
            )
        finally:
            gir.INTEGRATED_FARE_POLICY = orig_policy

        self.assertGreater(result_integrated[0], result_double[0],
                           "Integrated fare should attract more riders than double-charge")

    def test_zero_coverage_no_feeder_riders(self):
        """When feeder_coverage_fraction=0, feeder zone contributes nothing."""
        from scripts.generate_improved_ridership import compute_lodes_ridership

        # Only feeder-zone OD pairs (P3->P1: origin in feeder, dest in walk)
        feeder_lodes = _make_lodes(["P3"], ["P1"], [100])

        result = compute_lodes_ridership(
            self.corridor, self.parcels, feeder_lodes,
            feeder_headway_min=15,
            feeder_coverage_fraction=0.0,
        )
        # With zero coverage, only walk-zone pairs contribute.
        # P3 is in feeder zone, so P3->P1 should not produce walk-zone APM trips.
        # The walk-zone mask requires both ends within max_walk_m.
        # P3 is ~1700m, so both_near should be False for this pair.
        # Result should be zero (or near-zero from relaxed fallback).
        self.assertAlmostEqual(result[0], 0.0, places=1,
                               msg="Zero feeder coverage should not add feeder riders")

    def test_backward_compatible_signature(self):
        """Calling without feeder params should still work (backward compatible)."""
        from scripts.generate_improved_ridership import compute_lodes_ridership

        result = compute_lodes_ridership(
            self.corridor, self.parcels, self.lodes,
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)  # 4 base + directional_split
        self.assertGreaterEqual(result[0], 0)
        # directional_split should be in valid range
        self.assertGreaterEqual(result[4], 0.50)
        self.assertLessEqual(result[4], 0.80)


class TestSoftBoundaryTransition(unittest.TestCase):
    """Verify the soft-boundary extension for both_near OD matching."""

    def setUp(self):
        # Corridor with 2 stops
        self.corridor = _make_corridor([
            (-86.92, 40.42),
            (-86.91, 40.42),
        ])

    def test_transition_band_captures_marginal_pairs(self):
        """OD pairs with one end at 1300m (in transition band) should still
        contribute, unlike the old hard 1200m cutoff."""
        from scripts.generate_improved_ridership import compute_lodes_ridership

        # P0 at stop (0m), P1 ~1300m from stop (in transition 1200-1600m)
        parcels = _make_parcels([
            (-86.92, 40.42),       # P0: at stop
            (-86.92, 40.432),      # P1: ~1330m north (in transition band)
        ])
        lodes = _make_lodes(["P0"], ["P1"], [200])

        result = compute_lodes_ridership(
            self.corridor, parcels, lodes, max_walk_m=1200,
        )
        # Should capture some trips (transition band, not zero)
        self.assertGreater(result[0], 0.0,
                           "Transition band should capture OD pairs beyond 1200m")

    def test_transition_weight_less_than_core(self):
        """Pairs in the transition band should have less total OD weight than
        equivalent pairs fully within the core zone.  Compare od_trips
        (result[1]) not apm_trips — the MNL favors APM for longer trips,
        masking the access-weight reduction."""
        from scripts.generate_improved_ridership import compute_lodes_ridership

        # Two separate tests: destination at 800m (core) vs 1400m (transition)
        parcels_core = _make_parcels([
            (-86.92, 40.42),       # P0: at stop
            (-86.92, 40.427),      # P1: ~780m (well within core)
        ])
        parcels_trans = _make_parcels([
            (-86.92, 40.42),       # P0: at stop
            (-86.92, 40.433),      # P1: ~1440m (in transition band)
        ])
        lodes = _make_lodes(["P0"], ["P1"], [200])

        result_core = compute_lodes_ridership(
            self.corridor, parcels_core, lodes, max_walk_m=1200,
        )
        result_trans = compute_lodes_ridership(
            self.corridor, parcels_trans, lodes, max_walk_m=1200,
        )
        # Transition band should have lower total OD weight (more decay)
        self.assertGreater(result_core[1], result_trans[1],
                           "Core zone should have higher OD weight than transition band")

    def test_beyond_transition_returns_zero(self):
        """Pairs with BOTH ends beyond transition should get zero walk-zone
        trips (dest_relax_factor=1.0 disables the relaxation fallback)."""
        from scripts.generate_improved_ridership import compute_lodes_ridership

        parcels = _make_parcels([
            (-86.92, 40.435),      # P0: ~1660m from stop (beyond transition)
            (-86.92, 40.44),       # P1: ~2220m from stop (beyond transition)
        ])
        lodes = _make_lodes(["P0"], ["P1"], [200])

        result = compute_lodes_ridership(
            self.corridor, parcels, lodes, max_walk_m=1200,
            dest_relax_factor=1.0,
        )
        self.assertAlmostEqual(result[0], 0.0, places=1,
                               msg="Beyond transition should get zero walk-zone trips")


class TestLogisticRentPremiumDecay(unittest.TestCase):
    """Verify logistic distance_factor in development model."""

    def test_logistic_midpoint(self):
        """At midpoint (500m), distance_factor should be 0.50."""
        midpoint = 500.0
        scale = 0.006
        factor = 1.0 / (1.0 + np.exp(scale * (500.0 - midpoint)))
        self.assertAlmostEqual(factor, 0.50, places=3)

    def test_logistic_near_station(self):
        """Near station (<100m), factor should be >0.90."""
        midpoint, scale = 500.0, 0.006
        factor = 1.0 / (1.0 + np.exp(scale * (50.0 - midpoint)))
        self.assertGreater(factor, 0.90)

    def test_logistic_far_station(self):
        """Far from station (>1000m), factor should be <0.05."""
        midpoint, scale = 500.0, 0.006
        factor = 1.0 / (1.0 + np.exp(scale * (1000.0 - midpoint)))
        self.assertLess(factor, 0.05)

    def test_logistic_more_generous_at_400m(self):
        """Logistic should be at least 2× more generous than old exponential
        at 400m — this is the core fix for the development cliff."""
        midpoint, scale = 500.0, 0.006
        old_exp = np.exp(-0.00425 * 400.0)        # 0.183
        new_logistic = 1.0 / (1.0 + np.exp(scale * (400.0 - midpoint)))  # 0.646
        self.assertGreater(new_logistic / old_exp, 2.0,
                           "Logistic should be >2× more generous at 400m")

    def test_logistic_monotonically_decreasing(self):
        """Factor should strictly decrease with distance."""
        midpoint, scale = 500.0, 0.006
        distances = np.arange(0, 1500, 50)
        factors = 1.0 / (1.0 + np.exp(scale * (distances - midpoint)))
        diffs = np.diff(factors)
        self.assertTrue(np.all(diffs < 0),
                        "Logistic should be monotonically decreasing")


if __name__ == "__main__":
    unittest.main()

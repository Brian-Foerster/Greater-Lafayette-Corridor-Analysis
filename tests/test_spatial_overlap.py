"""C8: Real geometry spatial overlap and classification tests.

Tests classify_routes_for_corridor() and NetworkRedesignStrategy._route_overlap_fraction()
with explicit Shapely geometries in projected CRS (meters). No file I/O, no real bus data.
"""
from __future__ import annotations

import unittest

import numpy as np

from src.bus_network import (
    BusRoute,
    BusOperatingCostModel,
    CORRIDOR_BUFFER_M,
    FEEDER_OVERLAP_THRESHOLD,
    NetworkRedesignStrategy,
    PARALLEL_OVERLAP_THRESHOLD,
    STATION_CONNECTIVITY_M,
    classify_routes_for_corridor,
)


def _make_route(route_id, **kwargs):
    defaults = dict(
        name=f"Route {route_id}",
        length_km=10.0,
        baseline_headway_min=30.0,
        current_headway_min=30.0,
        cycle_time_min=70.0,
        n_stops=20,
        trips_per_day=40.0,
        observed_daily_riders=500.0,
        classification="independent",
    )
    defaults.update(kwargs)
    return BusRoute(route_id=route_id, **defaults)


class TestRouteOverlapFraction(unittest.TestCase):
    """Test spatial overlap classification with explicit geometries.

    Corridor: 3 APM stations along a north-south line at y=0, 1000, 2000 (meters).
    All coordinates are in projected CRS (metric, e.g. EPSG:32616).
    """

    def setUp(self):
        # North-south corridor: 3 stations at (500, 0), (500, 1000), (500, 2000)
        self.corridor_stops = np.array([
            [500.0, 0.0],
            [500.0, 1000.0],
            [500.0, 2000.0],
        ])

    def test_parallel_route_high_overlap(self):
        """Route A: runs parallel within 200m — should classify as parallel (>0.40)."""
        # 10 bus stops along x=700 (200m east of corridor), y evenly from 0 to 2000
        n = 10
        bus_stops = np.column_stack([
            np.full(n, 700.0),           # 200m east of corridor
            np.linspace(0, 2000, n),
        ])
        route = _make_route("A")
        bus_stops_by_route = {"A": bus_stops}

        classify_routes_for_corridor(
            [route], self.corridor_stops, bus_stops_by_route,
        )
        self.assertEqual(route.classification, "parallel")

        # Also verify the overlap fraction directly via NetworkRedesignStrategy
        cost_model = BusOperatingCostModel(annual_budget=10_000_000)
        strategy = NetworkRedesignStrategy(
            routes=[route],
            cost_model=cost_model,
            bus_stops_by_route=bus_stops_by_route,
            corridor_stops_xy=self.corridor_stops,
        )
        overlap = strategy._route_overlap_fraction(route)
        self.assertGreater(overlap, 0.7, f"Expected >0.7 overlap, got {overlap:.2f}")

    def test_perpendicular_feeder_route(self):
        """Route B: runs east-west crossing corridor at one station — feeder."""
        # 10 bus stops along y=1000 (crosses middle station), x from 300 to 3000
        n = 10
        bus_stops = np.column_stack([
            np.linspace(300, 3000, n),
            np.full(n, 1000.0),
        ])
        route = _make_route("B")
        bus_stops_by_route = {"B": bus_stops}

        classify_routes_for_corridor(
            [route], self.corridor_stops, bus_stops_by_route,
        )
        # Should be feeder: low overlap but has station connection at (500,1000)
        self.assertEqual(route.classification, "feeder")

        # Verify overlap fraction is in expected range
        cost_model = BusOperatingCostModel(annual_budget=10_000_000)
        strategy = NetworkRedesignStrategy(
            routes=[route],
            cost_model=cost_model,
            bus_stops_by_route=bus_stops_by_route,
            corridor_stops_xy=self.corridor_stops,
        )
        overlap = strategy._route_overlap_fraction(route)
        self.assertGreater(overlap, 0.1, f"Expected >0.1, got {overlap:.2f}")
        self.assertLess(overlap, PARALLEL_OVERLAP_THRESHOLD,
                        f"Should be below parallel threshold, got {overlap:.2f}")

    def test_distant_independent_route(self):
        """Route C: runs 3km away — should classify as independent (zero overlap)."""
        n = 10
        bus_stops = np.column_stack([
            np.full(n, 4000.0),            # 3.5km east of corridor
            np.linspace(0, 2000, n),
        ])
        route = _make_route("C")
        bus_stops_by_route = {"C": bus_stops}

        classify_routes_for_corridor(
            [route], self.corridor_stops, bus_stops_by_route,
        )
        self.assertEqual(route.classification, "independent")

        cost_model = BusOperatingCostModel(annual_budget=10_000_000)
        strategy = NetworkRedesignStrategy(
            routes=[route],
            cost_model=cost_model,
            bus_stops_by_route=bus_stops_by_route,
            corridor_stops_xy=self.corridor_stops,
        )
        overlap = strategy._route_overlap_fraction(route)
        self.assertAlmostEqual(overlap, 0.0, places=2)

    def test_all_three_routes_classified_together(self):
        """Classify all three routes in a single call."""
        routes = [_make_route("A"), _make_route("B"), _make_route("C")]
        bus_stops_by_route = {
            "A": np.column_stack([
                np.full(10, 700.0),
                np.linspace(0, 2000, 10),
            ]),
            "B": np.column_stack([
                np.linspace(300, 3000, 10),
                np.full(10, 1000.0),
            ]),
            "C": np.column_stack([
                np.full(10, 4000.0),
                np.linspace(0, 2000, 10),
            ]),
        }

        classify_routes_for_corridor(
            routes, self.corridor_stops, bus_stops_by_route,
        )
        by_id = {r.route_id: r.classification for r in routes}
        self.assertEqual(by_id["A"], "parallel")
        self.assertEqual(by_id["B"], "feeder")
        self.assertEqual(by_id["C"], "independent")


class TestPartialOverlapClassification(unittest.TestCase):
    """Edge cases: routes that start parallel then diverge."""

    def setUp(self):
        # Same corridor as above
        self.corridor_stops = np.array([
            [500.0, 0.0],
            [500.0, 1000.0],
            [500.0, 2000.0],
        ])

    def test_starts_parallel_then_diverges(self):
        """Route runs parallel for first half, then veers 2km east.

        With 5/10 stops near corridor (50%), should classify as parallel
        since 50% >= PARALLEL_OVERLAP_THRESHOLD (40%).
        """
        n = 10
        x_coords = np.array([
            600, 600, 600, 600, 600,        # first 5: 100m from corridor (within 400m)
            1500, 2500, 3500, 4500, 5500,    # last 5: diverging east
        ], dtype=float)
        y_coords = np.linspace(0, 2000, n)
        bus_stops = np.column_stack([x_coords, y_coords])

        route = _make_route("D")
        bus_stops_by_route = {"D": bus_stops}

        classify_routes_for_corridor(
            [route], self.corridor_stops, bus_stops_by_route,
        )
        # 5/10 stops within 400m ≈ 50% → parallel (>= 40%)
        self.assertEqual(route.classification, "parallel")

    def test_mostly_diverged_with_station_connection(self):
        """Route barely touches corridor — 2/10 stops near, but one is at a station.

        With 20% overlap and station connection → feeder (15-40% range + connected).
        """
        n = 10
        x_coords = np.array([
            500, 600,                                  # 2 stops near corridor
            1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400,  # 8 far away
        ], dtype=float)
        # First stop at y=0 is right at station (500, 0)
        y_coords = np.array([0, 200, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600],
                            dtype=float)
        bus_stops = np.column_stack([x_coords, y_coords])

        route = _make_route("E")
        bus_stops_by_route = {"E": bus_stops}

        classify_routes_for_corridor(
            [route], self.corridor_stops, bus_stops_by_route,
        )
        self.assertEqual(route.classification, "feeder")

    def test_low_overlap_no_station_connection_is_independent(self):
        """Route with ~20% geometric overlap but NO stop within 200m of a station.

        Should be independent despite overlap being in feeder range (15-40%),
        because there's no transfer opportunity.
        """
        n = 10
        # Place 2 stops near corridor LINE but not near any STATION
        # Stations are at y=0, 1000, 2000. Put the near-corridor stops at y=500.
        x_coords = np.array([
            600, 600,                                  # near corridor line at y=500
            2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500,
        ], dtype=float)
        y_coords = np.array([500, 600, 500, 600, 500, 600, 500, 600, 500, 600],
                            dtype=float)
        bus_stops = np.column_stack([x_coords, y_coords])

        route = _make_route("F")
        bus_stops_by_route = {"F": bus_stops}

        classify_routes_for_corridor(
            [route], self.corridor_stops, bus_stops_by_route,
        )
        # 2/10 = 20% overlap (in feeder range), but the closest stop to any
        # station is at (600, 500) which is ~510m from station (500, 0) —
        # beyond STATION_CONNECTIVITY_M (200m).
        self.assertEqual(route.classification, "independent")


class TestOverlapFractionValues(unittest.TestCase):
    """Verify overlap fraction numerics with known geometries."""

    def test_all_stops_inside_buffer_gives_1(self):
        """All bus stops directly on the corridor line → overlap = 1.0."""
        corridor = np.array([[0.0, 0.0], [0.0, 1000.0]])
        bus_stops = np.array([
            [0.0, 100.0],
            [0.0, 300.0],
            [0.0, 500.0],
            [0.0, 700.0],
            [0.0, 900.0],
        ])
        cost_model = BusOperatingCostModel(annual_budget=10_000_000)
        route = _make_route("X")
        strategy = NetworkRedesignStrategy(
            routes=[route],
            cost_model=cost_model,
            bus_stops_by_route={"X": bus_stops},
            corridor_stops_xy=corridor,
        )
        overlap = strategy._route_overlap_fraction(route)
        self.assertAlmostEqual(overlap, 1.0, places=2)

    def test_no_stops_inside_buffer_gives_0(self):
        """All bus stops far from corridor → overlap = 0.0."""
        corridor = np.array([[0.0, 0.0], [0.0, 1000.0]])
        bus_stops = np.array([
            [5000.0, 100.0],
            [5000.0, 500.0],
            [5000.0, 900.0],
        ])
        cost_model = BusOperatingCostModel(annual_budget=10_000_000)
        route = _make_route("Y")
        strategy = NetworkRedesignStrategy(
            routes=[route],
            cost_model=cost_model,
            bus_stops_by_route={"Y": bus_stops},
            corridor_stops_xy=corridor,
        )
        overlap = strategy._route_overlap_fraction(route)
        self.assertAlmostEqual(overlap, 0.0, places=2)

    def test_half_stops_gives_half_overlap(self):
        """Half the stops within buffer, half outside."""
        corridor = np.array([[0.0, 0.0], [0.0, 2000.0]])
        # 3 stops on-corridor, 3 stops 2km away
        bus_stops = np.array([
            [50.0, 200.0],     # within 400m
            [50.0, 1000.0],    # within 400m
            [50.0, 1800.0],    # within 400m
            [3000.0, 200.0],   # far
            [3000.0, 1000.0],  # far
            [3000.0, 1800.0],  # far
        ])
        cost_model = BusOperatingCostModel(annual_budget=10_000_000)
        route = _make_route("Z")
        strategy = NetworkRedesignStrategy(
            routes=[route],
            cost_model=cost_model,
            bus_stops_by_route={"Z": bus_stops},
            corridor_stops_xy=corridor,
        )
        overlap = strategy._route_overlap_fraction(route)
        self.assertAlmostEqual(overlap, 0.5, places=1)


class TestMissingRouteEdgeCases(unittest.TestCase):
    """Edge cases: empty stops, missing route, single-station corridor."""

    def test_empty_bus_stops_classified_independent(self):
        corridor = np.array([[0.0, 0.0], [0.0, 1000.0]])
        route = _make_route("EMPTY")
        classify_routes_for_corridor(
            [route], corridor, {"EMPTY": np.empty((0, 2))},
        )
        self.assertEqual(route.classification, "independent")

    def test_missing_route_in_stops_dict(self):
        corridor = np.array([[0.0, 0.0], [0.0, 1000.0]])
        route = _make_route("MISSING")
        classify_routes_for_corridor(
            [route], corridor, {},
        )
        self.assertEqual(route.classification, "independent")

    def test_single_station_corridor_skips_classification(self):
        """With <2 corridor stops, classification is skipped entirely."""
        corridor = np.array([[0.0, 0.0]])
        bus_stops = np.array([[10.0, 10.0], [20.0, 20.0]])
        route = _make_route("SOLO")
        classify_routes_for_corridor(
            [route], corridor, {"SOLO": bus_stops},
        )
        # Classification unchanged (stays at default "independent")
        self.assertEqual(route.classification, "independent")


if __name__ == "__main__":
    unittest.main()

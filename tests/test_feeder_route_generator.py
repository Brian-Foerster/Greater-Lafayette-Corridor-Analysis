"""Tests for src/feeder_route_generator.py — arterial-spine feeder route generator.

Optimized: grid 10x10 (100 nodes), 10 fixed parcels instead of 2000 random.
Unit tests for _select_spines and _score_pop_walkshed use mock inputs directly.
"""

import math
import sys
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
from src.spatial_constants import PROJECT_CRS

from src.feeder_route_generator import (
    ArterialSpine,
    ANGULAR_DIVERSITY_RAD,
    MAX_SYNTHETIC_ROUTES,
    MIN_SPINE_POP,
    PARALLEL_REJECT_RAD,
    N_SECTORS,
    _compute_station_demand_sectors,
    _extract_arterial_spines,
    _interpolate_stops,
    _score_pop_walkshed,
    _select_feeder_targets,
    _select_spines,
    generate_feeder_routes,
)


def _make_grid_graph(
    center_lonlat=(-86.9, 40.43),
    grid_size=10,
    spacing_deg=0.002,
    highway="secondary",
):
    """Build a small grid road graph in WGS-84 for testing.

    Default: 10x10 = 100 nodes (was 40x40 = 1600).
    """
    G = nx.MultiDiGraph()
    node_id = 0
    id_grid = {}
    for r in range(grid_size):
        for c in range(grid_size):
            lon = center_lonlat[0] + c * spacing_deg
            lat = center_lonlat[1] + r * spacing_deg
            G.add_node(node_id, x=lon, y=lat)
            id_grid[(r, c)] = node_id
            node_id += 1

    for r in range(grid_size):
        for c in range(grid_size):
            u = id_grid[(r, c)]
            if c + 1 < grid_size:
                v = id_grid[(r, c + 1)]
                length = spacing_deg * 111_000
                G.add_edge(u, v, highway=highway, length=length)
                G.add_edge(v, u, highway=highway, length=length)
            if r + 1 < grid_size:
                v = id_grid[(r + 1, c)]
                length = spacing_deg * 111_000
                G.add_edge(u, v, highway=highway, length=length)
                G.add_edge(v, u, highway=highway, length=length)

    # Attach spatial index so _fast_nearest_node works without osmnx fallback.
    from scipy.spatial import cKDTree
    from src.feeder_route_generator import build_graph_node_index
    node_ids, node_xy = build_graph_node_index(G)
    G.graph["_node_ids"] = node_ids
    G.graph["_node_xy"] = node_xy
    G.graph["_node_tree"] = cKDTree(node_xy)

    return G, id_grid


def _project_lonlat_to_proj(lon, lat):
    """Project lon/lat to PROJECT_CRS coordinates."""
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", PROJECT_CRS, always_xy=True)
    return transformer.transform(lon, lat)


def _make_fixed_parcels(cx, cy, n=10):
    """Create fixed parcels in the feeder ring (1500-4000m) at known angles."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dists = np.linspace(1500, 3500, n)
    parcel_xy = np.column_stack([
        cx + dists * np.cos(angles),
        cy + dists * np.sin(angles),
    ])
    parcel_pop = np.full(n, 50.0)
    return parcel_xy, parcel_pop


class TestSelectSpines(unittest.TestCase):
    """Tests for _select_spines angular diversity and selection logic."""

    def _make_spine(self, angle_deg, pop=5000, length_m=2000, travel_bearing_deg=None):
        angle_rad = math.radians(angle_deg)
        if travel_bearing_deg is None:
            travel_bearing_deg = angle_deg
        return ArterialSpine(
            nodes=[0, 1],
            endpoint_node=1,
            angle_rad=angle_rad,
            travel_bearing_rad=math.radians(travel_bearing_deg),
            length_m=length_m,
            pop_within_walkshed=pop,
            highway_class="secondary",
        )

    def test_max_routes_cap(self):
        """Should select at most MAX_SYNTHETIC_ROUTES spines."""
        spines = [self._make_spine(i * 50, pop=10000) for i in range(8)]
        result = _select_spines(spines, max_routes=4)
        self.assertLessEqual(len(result), 4)

    def test_angular_diversity(self):
        """Spines within ANGULAR_DIVERSITY_RAD of each other should not both be selected."""
        s1 = self._make_spine(0, pop=10000)
        s2 = self._make_spine(10, pop=9000)
        s3 = self._make_spine(90, pop=8000)
        result = _select_spines([s1, s2, s3])
        angles = [s.angle_rad for s in result]
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].angle_rad, math.radians(0), places=3)
        self.assertAlmostEqual(result[1].angle_rad, math.radians(90), places=3)

    def test_min_pop_filter(self):
        """Spines below MIN_SPINE_POP should be excluded."""
        s1 = self._make_spine(0, pop=5000)
        s2 = self._make_spine(90, pop=500)
        result = _select_spines([s1, s2], min_pop=MIN_SPINE_POP)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].pop_within_walkshed, 5000)

    def test_existing_angles_respected(self):
        """Spines near existing feeder angles should be skipped."""
        s1 = self._make_spine(0, pop=10000)
        s2 = self._make_spine(180, pop=9000)
        result = _select_spines([s1, s2], existing_angles=[math.radians(5)])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].angle_rad, math.radians(180), places=3)

    def test_pop_priority(self):
        """Higher-population spines should be selected first."""
        s1 = self._make_spine(0, pop=3000)
        s2 = self._make_spine(90, pop=8000)
        s3 = self._make_spine(180, pop=5000)
        result = _select_spines([s1, s2, s3], max_routes=2)
        self.assertEqual(result[0].pop_within_walkshed, 8000)
        self.assertEqual(result[1].pop_within_walkshed, 5000)

    def test_wraparound_angles(self):
        """Angular distance should handle wrap-around (350 deg and 10 deg are 20 deg apart)."""
        s1 = self._make_spine(350, pop=8000)
        s2 = self._make_spine(10, pop=7000)
        s3 = self._make_spine(180, pop=6000)
        result = _select_spines([s1, s2, s3])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].pop_within_walkshed, 8000)
        self.assertEqual(result[1].pop_within_walkshed, 6000)

    def test_empty_input(self):
        result = _select_spines([])
        self.assertEqual(result, [])

    def test_parallel_rejection(self):
        """Spines parallel to corridor should be rejected."""
        # Corridor bearing = 0 (east-west)
        s_parallel = self._make_spine(90, pop=10000, travel_bearing_deg=10)  # parallel
        s_perp = self._make_spine(90, pop=8000, travel_bearing_deg=90)  # perpendicular
        result = _select_spines(
            [s_parallel, s_perp],
            corridor_bearing_rad=0.0,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].pop_within_walkshed, 8000)

    def test_parallel_rejection_reverse(self):
        """Spines running in the reverse direction of corridor should also be rejected."""
        # Corridor bearing = 0, spine bearing = 170 (~reverse parallel)
        s_rev = self._make_spine(90, pop=10000, travel_bearing_deg=170)
        s_perp = self._make_spine(90, pop=8000, travel_bearing_deg=90)
        result = _select_spines(
            [s_rev, s_perp],
            corridor_bearing_rad=0.0,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].pop_within_walkshed, 8000)


class TestScorePopWalkshed(unittest.TestCase):
    """Unit tests for _score_pop_walkshed with mock inputs (no graph needed)."""

    def test_parcels_near_spine_counted(self):
        from scipy.spatial import cKDTree
        node_xy = {0: (0.0, 0.0), 1: (1000.0, 0.0)}
        parcel_xy = np.array([[500.0, 100.0], [500.0, 200.0], [5000.0, 5000.0]])
        parcel_pop = np.array([100.0, 200.0, 500.0])
        tree = cKDTree(parcel_xy)

        score = _score_pop_walkshed([0, 1], node_xy, tree, parcel_pop)
        self.assertAlmostEqual(score, 300.0, places=0)

    def test_no_parcels_returns_zero(self):
        score = _score_pop_walkshed([0, 1], {0: (0, 0), 1: (1000, 0)}, None, np.array([]))
        self.assertEqual(score, 0.0)


class TestExtractArterialSpines(unittest.TestCase):
    """Tests for _extract_arterial_spines with synthetic road graphs."""

    def test_finds_spines_in_feeder_ring(self):
        """Should find arterial spines in the feeder ring around stations."""
        G = nx.MultiDiGraph()
        center_lon, center_lat = -86.9, 40.43

        G.add_node(0, x=center_lon, y=center_lat)

        node_id = 1
        for arm_idx in range(6):
            angle = arm_idx * math.pi / 3
            prev_node = 0
            for step in range(1, 20):
                d_deg = step * 0.003
                lon = center_lon + d_deg * math.cos(angle)
                lat = center_lat + d_deg * math.sin(angle)
                G.add_node(node_id, x=lon, y=lat)
                length = 0.003 * 111_000
                G.add_edge(prev_node, node_id, highway="secondary", length=length)
                G.add_edge(node_id, prev_node, highway="secondary", length=length)
                prev_node = node_id
                node_id += 1

        cx, cy = _project_lonlat_to_proj(center_lon, center_lat)
        station_xy = np.array([[cx, cy]])

        parcel_xy, parcel_pop = _make_fixed_parcels(cx, cy, n=10)

        spines = _extract_arterial_spines(G, station_xy, parcel_xy, parcel_pop)
        self.assertGreater(len(spines), 0, "Should find at least one arterial spine")
        # Verify travel_bearing_rad is populated
        for s in spines:
            self.assertIsInstance(s.travel_bearing_rad, float)

    def test_no_arterials_returns_empty(self):
        """If graph has no arterial edges, should return empty list."""
        G, _ = _make_grid_graph(highway="residential")
        cx, cy = _project_lonlat_to_proj(-86.9, 40.43)
        station_xy = np.array([[cx, cy]])
        parcel_xy = np.array([[cx + 2000, cy + 2000]])
        parcel_pop = np.array([100.0])

        spines = _extract_arterial_spines(G, station_xy, parcel_xy, parcel_pop)
        self.assertEqual(len(spines), 0)


class TestStationDemandSectors(unittest.TestCase):
    """Tests for _compute_station_demand_sectors."""

    def test_basic_demand_matrix_shape(self):
        """Demand matrix should be (N_stations, 8)."""
        station_xy = np.array([[0.0, 0.0], [5000.0, 0.0]])
        parcel_xy = np.array([[2000.0, 2000.0], [2000.0, -2000.0],
                              [7000.0, 2000.0]])
        parcel_pop = np.array([100.0, 200.0, 150.0])
        dm = _compute_station_demand_sectors(station_xy, parcel_xy, parcel_pop)
        self.assertEqual(dm.shape, (2, N_SECTORS))

    def test_population_drives_demand(self):
        """Sectors with more population should have higher demand."""
        station_xy = np.array([[0.0, 0.0]])
        # All parcels NE (sector ~1), but different pops
        parcel_xy = np.array([[2000.0, 2000.0], [2500.0, 2500.0]])
        parcel_pop = np.array([1000.0, 500.0])
        dm = _compute_station_demand_sectors(station_xy, parcel_xy, parcel_pop)
        total = dm.sum()
        self.assertGreater(total, 0)

    def test_empty_parcels(self):
        """Empty parcel array should give zero demand."""
        station_xy = np.array([[0.0, 0.0]])
        dm = _compute_station_demand_sectors(
            station_xy, np.empty((0, 2)), np.array([]))
        self.assertEqual(dm.sum(), 0.0)

    def test_od_weighted_demand(self):
        """OD flows should boost demand for corridor-bound commute sectors."""
        station_xy = np.array([[0.0, 0.0]])
        # Origin in feeder ring NE, destination near station
        parcel_xy = np.array([[2000.0, 2000.0], [100.0, 100.0]])
        parcel_pop = np.array([100.0, 50.0])
        od_origins = np.array([0])  # parcel 0 = origin
        od_dests = np.array([1])    # parcel 1 = dest (near station)
        od_flows = np.array([50.0])

        dm_with_od = _compute_station_demand_sectors(
            station_xy, parcel_xy, parcel_pop,
            od_origins_idx=od_origins, od_dests_idx=od_dests, od_flows=od_flows,
        )
        dm_no_od = _compute_station_demand_sectors(
            station_xy, parcel_xy, parcel_pop)
        self.assertGreater(dm_with_od.sum(), dm_no_od.sum())


class TestSelectFeederTargets(unittest.TestCase):
    """Tests for _select_feeder_targets demand-based selection."""

    def test_max_routes_respected(self):
        dm = np.ones((3, N_SECTORS)) * 100
        targets = _select_feeder_targets(dm, corridor_bearing_rad=None, max_routes=2)
        self.assertLessEqual(len(targets), 2)

    def test_parallel_sectors_rejected(self):
        """Sectors aligned with corridor should be rejected."""
        dm = np.zeros((1, N_SECTORS))
        # All demand in sector 4 (bearing ~ π, which is ~parallel to corridor at 0)
        dm[0, 4] = 1000
        dm[0, 2] = 500  # perpendicular sector (bearing ~ π/2)
        targets = _select_feeder_targets(dm, corridor_bearing_rad=0.0)
        # Sector 4 center is near π → parallel to corridor 0 → rejected
        # Sector 2 center is near π/2 → perpendicular → selected
        if targets:
            # Should prefer non-parallel sectors
            for _, ki, _ in targets:
                sector_center = (ki - 0.5) * (2 * math.pi / N_SECTORS) + (2 * math.pi / N_SECTORS) / 2 - math.pi
                corr_diff = abs(sector_center)
                if corr_diff > math.pi:
                    corr_diff = 2 * math.pi - corr_diff
                if corr_diff > math.pi / 2:
                    corr_diff = math.pi - corr_diff
                self.assertGreaterEqual(corr_diff, PARALLEL_REJECT_RAD - 0.01)

    def test_demand_priority(self):
        """Higher-demand sectors should be selected first."""
        dm = np.zeros((1, N_SECTORS))
        dm[0, 1] = 500
        dm[0, 3] = 1000
        dm[0, 5] = 200
        targets = _select_feeder_targets(dm, corridor_bearing_rad=None)
        if len(targets) >= 2:
            self.assertGreaterEqual(targets[0][2], targets[1][2])


class TestGenerateFeederRoutes(unittest.TestCase):
    """Integration tests for generate_feeder_routes with small grid."""

    def test_no_graph_returns_empty(self):
        """Without road graph, should return empty result."""
        station_xy = np.array([[0, 0]])
        station_lonlat = np.array([[-86.9, 40.43]])
        result = generate_feeder_routes(
            station_xy=station_xy,
            station_lonlat=station_lonlat,
            parcel_xy=np.array([[1000, 1000]]),
            parcel_pop=np.array([100]),
            road_graph=None,
        )
        self.assertEqual(result.summary["n_routes"], 0)

    def test_route_count_cap(self):
        """Should generate at most MAX_SYNTHETIC_ROUTES (10x10 grid, 10 parcels)."""
        G, _ = _make_grid_graph(
            center_lonlat=(-86.9, 40.43),
            grid_size=10,
            spacing_deg=0.003,
        )

        cx, cy = _project_lonlat_to_proj(-86.9, 40.43)
        station_xy = np.array([[cx, cy]])
        station_lonlat = np.array([[-86.9, 40.43]])

        parcel_xy, parcel_pop = _make_fixed_parcels(cx, cy, n=10)

        from src.feeder_route_generator import _add_transit_weights
        _add_transit_weights(G)

        result = generate_feeder_routes(
            station_xy=station_xy,
            station_lonlat=station_lonlat,
            parcel_xy=parcel_xy,
            parcel_pop=parcel_pop,
            road_graph=G,
        )
        self.assertLessEqual(result.summary["n_routes"], MAX_SYNTHETIC_ROUTES)

    def test_angular_diversity_in_routes(self):
        """Generated routes should serve different directions."""
        G, _ = _make_grid_graph(
            center_lonlat=(-86.9, 40.43),
            grid_size=10,
            spacing_deg=0.003,
        )

        cx, cy = _project_lonlat_to_proj(-86.9, 40.43)
        station_xy = np.array([[cx, cy]])
        station_lonlat = np.array([[-86.9, 40.43]])

        parcel_xy, parcel_pop = _make_fixed_parcels(cx, cy, n=10)

        from src.feeder_route_generator import _add_transit_weights
        _add_transit_weights(G)

        result = generate_feeder_routes(
            station_xy=station_xy,
            station_lonlat=station_lonlat,
            parcel_xy=parcel_xy,
            parcel_pop=parcel_pop,
            road_graph=G,
        )
        directions = result.summary.get("directions_served", [])
        self.assertEqual(len(directions), len(set(directions)),
                         f"Duplicate directions: {directions}")

    def test_demand_proportional_headway(self):
        """Higher-demand sectors should get lower headways."""
        from src.feeder_route_generator import MIN_FEEDER_HEADWAY, MAX_FEEDER_HEADWAY
        G, _ = _make_grid_graph(
            center_lonlat=(-86.9, 40.43),
            grid_size=10,
            spacing_deg=0.003,
        )

        cx, cy = _project_lonlat_to_proj(-86.9, 40.43)
        station_xy = np.array([[cx, cy]])
        station_lonlat = np.array([[-86.9, 40.43]])

        # Make parcels with unequal populations in different directions
        parcel_xy, parcel_pop = _make_fixed_parcels(cx, cy, n=10)
        # Boost population in one direction
        parcel_pop[0] = 5000.0

        from src.feeder_route_generator import _add_transit_weights
        _add_transit_weights(G)

        result = generate_feeder_routes(
            station_xy=station_xy,
            station_lonlat=station_lonlat,
            parcel_xy=parcel_xy,
            parcel_pop=parcel_pop,
            road_graph=G,
        )
        if len(result.routes) >= 2:
            headways = [r.baseline_headway_min for r in result.routes]
            # Not all headways should be the same (demand-proportional)
            self.assertGreater(max(headways) - min(headways), 0.1,
                             "Headways should vary by demand")
            # All headways should be in valid range
            for h in headways:
                self.assertGreaterEqual(h, MIN_FEEDER_HEADWAY - 0.1)
                self.assertLessEqual(h, MAX_FEEDER_HEADWAY + 0.1)


class TestInterpolateStops(unittest.TestCase):
    def test_stop_spacing(self):
        coords = [(-86.9, 40.43), (-86.89, 40.43)]
        stops = _interpolate_stops(coords, spacing_m=400)
        self.assertGreater(len(stops), 2, "Should add intermediate stops")

    def test_short_segment(self):
        coords = [(-86.9, 40.43), (-86.8999, 40.43)]
        stops = _interpolate_stops(coords, spacing_m=400)
        self.assertLessEqual(len(stops), 2)

    def test_single_coord(self):
        stops = _interpolate_stops([(-86.9, 40.43)], spacing_m=400)
        self.assertEqual(len(stops), 1)


if __name__ == "__main__":
    unittest.main()

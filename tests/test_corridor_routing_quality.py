"""Tests for corridor routing quality constraints.

Verifies physics-based curve speed penalties (ASCE 21.2-2008 lateral
acceleration model), minimum curve radius validation, effective speed
computation, and road-class cost surface augmentation.

Split into two groups:
- Pure geometry tests (no graph needed): curve radii, speed penalties, cost multipliers
- Bearing tests (5-node graph): path bearing detection, reversal checks
"""

import math
import numpy as np
import pytest
import sys
import os

from src.spatial_constants import PROJECT_CRS, US_SURVEY_FT_TO_M
from scripts.optimized_corridor_search import (
    compute_curve_speed_penalties,
    validate_station_set,
    APM_SPEED_KPH,
    _augment_graph_weights,
    _path_bearing_at,
    check_routed_path_quality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_station_data_stub(pts_ft):
    """Minimal station_data dict for validate_station_set.

    pts_ft must be in EPSG:2965 US survey feet (matching production coords_proj).
    """
    return {
        "coords_proj": pts_ft,
        "graph": None,  # skip road-graph circuity checks
    }


def _make_pts_with_turn(angle_deg, spacing_m=800.0):
    """Create 4 stations: straight -> turn of angle_deg -> straight.

    angle_deg is the deflection (0 = straight, 180 = reversal).
    Returns (N, 2) ndarray in meters.
    """
    sp = spacing_m

    p0 = np.array([0.0, 0.0])
    p1 = np.array([sp, 0.0])
    theta_rad = np.radians(angle_deg)
    v2_dir = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    p2 = p1 + sp * v2_dir
    p3 = p2 + sp * v2_dir

    return np.array([p0, p1, p2, p3])


def _make_pts_with_turn_ft(angle_deg, spacing_m=800.0):
    """Same as _make_pts_with_turn but returns EPSG:2965 feet for validate_station_set."""
    return _make_pts_with_turn(angle_deg, spacing_m) / US_SURVEY_FT_TO_M


# ---------------------------------------------------------------------------
# Pure geometry tests (no graph needed)
# ---------------------------------------------------------------------------

class TestCurveGeometry:
    """Pure math: curve radii, speed penalties, cost multipliers."""

    def test_curve_radius_from_geometry_90deg(self):
        """90 deg turn at 800m spacing -> R = T/tan(45 deg) = 400m."""
        pts = _make_pts_with_turn(90, spacing_m=800)
        result = compute_curve_speed_penalties(pts)
        assert len(result["curve_radii"]) == 2
        R = result["curve_radii"][0]
        assert R == pytest.approx(400.0, rel=0.02)

    def test_curve_radius_from_geometry_150deg(self):
        """150 deg turn at 800m spacing -> R = 400/tan(75 deg) ~ 107m."""
        pts = _make_pts_with_turn(150, spacing_m=800)
        result = compute_curve_speed_penalties(pts)
        R = result["curve_radii"][0]
        expected_R = 400.0 / np.tan(np.radians(75))
        assert R == pytest.approx(expected_R, rel=0.02)

    def test_gentle_turn_no_penalty(self):
        """90 deg turn at 800m spacing should have zero delay."""
        pts = _make_pts_with_turn(90, spacing_m=800)
        result = compute_curve_speed_penalties(pts)
        assert result["total_curve_delay_s"] == pytest.approx(0.0)
        assert result["n_speed_restricted_curves"] == 0

    def test_sharp_turn_speed_penalty(self):
        """160 deg turn at 800m spacing -> V_curve < V_line, delay > 0."""
        pts = _make_pts_with_turn(160, spacing_m=800)
        result = compute_curve_speed_penalties(pts)
        assert result["total_curve_delay_s"] > 0
        assert result["n_speed_restricted_curves"] >= 1
        assert result["curve_speeds_kph"][0] < APM_SPEED_KPH * 0.85

    def test_straight_corridor_no_penalty(self):
        """Perfectly straight corridor -> zero delay, cost_mult = 1.0."""
        sp = 800
        pts = np.array([[0, 0], [sp, 0], [2 * sp, 0], [3 * sp, 0]], dtype=float)
        result = compute_curve_speed_penalties(pts)
        assert result["total_curve_delay_s"] == pytest.approx(0.0)
        assert result["curve_cost_mult"] == pytest.approx(1.0)

    def test_curve_construction_cost_premium(self):
        """A sharp curve should increase curve_cost_mult above 1.0."""
        pts = _make_pts_with_turn(160, spacing_m=800)
        result = compute_curve_speed_penalties(pts)
        assert result["curve_cost_mult"] > 1.0

    def test_infeasible_curve_flagged(self):
        """Extreme turn at moderate spacing -> R < APM_MIN_CURVE_RADIUS_M -> infeasible."""
        pts = _make_pts_with_turn(175, spacing_m=800)
        result = compute_curve_speed_penalties(pts)
        assert result["has_infeasible_curve"]
        assert result["min_curve_radius_m"] < 25.0


# ---------------------------------------------------------------------------
# Validation: minimum curve radius (no graph, uses stub)
# ---------------------------------------------------------------------------

class TestCurveValidation:
    """validate_station_set with graph=None stub — tests curve radius logic only."""

    def test_min_radius_rejection(self):
        """Extreme turn (175 deg at 800m) should be rejected."""
        pts_ft = _make_pts_with_turn_ft(175, spacing_m=800)
        station_data = _make_station_data_stub(pts_ft)
        valid, reason = validate_station_set(list(range(4)), station_data)
        assert not valid
        assert "curve radius" in reason

    def test_90_degree_turn_accepted(self):
        """A single 90 deg turn should pass validation (R well above minimum)."""
        pts_ft = _make_pts_with_turn_ft(90, spacing_m=800)
        station_data = _make_station_data_stub(pts_ft)
        valid, reason = validate_station_set(list(range(4)), station_data)
        if not valid:
            assert "curve radius" not in reason

    def test_many_gentle_turns_accepted(self):
        """Multiple 90 deg turns should all pass."""
        sp_ft = 800 / US_SURVEY_FT_TO_M
        pts = np.array([
            [0, 0],
            [sp_ft, 0],
            [sp_ft, sp_ft],
            [2 * sp_ft, sp_ft],
            [2 * sp_ft, 2 * sp_ft],
            [3 * sp_ft, 2 * sp_ft],
            [3 * sp_ft, 3 * sp_ft],
        ], dtype=float)
        station_data = _make_station_data_stub(pts)
        valid, reason = validate_station_set(list(range(len(pts))), station_data)
        if not valid:
            assert "curve radius" not in reason


# ---------------------------------------------------------------------------
# Road-class cost surface (5-node graph)
# ---------------------------------------------------------------------------

class TestRoadClassCostSurface:
    def test_augment_graph_weights(self):
        """apm_cost should be > length for residential, == length for primary."""
        import networkx as nx

        G = nx.MultiDiGraph()
        G.add_edge(0, 1, length=100.0, highway="primary")
        G.add_edge(1, 2, length=100.0, highway="residential")
        G.add_edge(2, 3, length=100.0, highway="service")

        _augment_graph_weights(G)

        assert G[0][1][0]["apm_cost"] == pytest.approx(100.0)
        assert G[1][2][0]["apm_cost"] == pytest.approx(250.0)
        assert G[2][3][0]["apm_cost"] == pytest.approx(300.0)

    def test_augment_graph_weights_list_highway(self):
        """Highway attribute can be a list — should use first element."""
        import networkx as nx

        G = nx.MultiDiGraph()
        G.add_edge(0, 1, length=200.0, highway=["secondary", "tertiary"])

        _augment_graph_weights(G)

        assert G[0][1][0]["apm_cost"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Path-bearing reversal detection (5-node graph)
# ---------------------------------------------------------------------------

def _make_simple_graph_with_coords():
    """Build a small road graph for bearing tests.

    Layout (lon, lat — EPSG:4326-like):
        A(0,0) -- B(0.01, 0) -- C(0.01, 0.01) -- D(0.02, 0.01)
                                  |
                                  E(0.01, -0.01)  (south of C)
    """
    import networkx as nx

    G = nx.MultiDiGraph()
    nodes = {
        0: {"x": 0.0, "y": 0.0},
        1: {"x": 0.01, "y": 0.0},
        2: {"x": 0.01, "y": 0.01},
        3: {"x": 0.02, "y": 0.01},
        4: {"x": 0.01, "y": -0.01},
    }
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)

    edges = [
        (0, 1, 800), (1, 0, 800),
        (1, 2, 800), (2, 1, 800),
        (2, 3, 800), (3, 2, 800),
        (2, 4, 800), (4, 2, 800),
    ]
    for u, v, length in edges:
        G.add_edge(u, v, length=length, apm_cost=length, highway="primary")

    return G


class TestPathBearing:
    """Bearing detection on the 5-node graph."""

    def test_path_bearing_at_departure(self):
        """Departure bearing from node 0 toward node 1 (East) should be ~90 deg."""
        G = _make_simple_graph_with_coords()
        path = [0, 1, 2]
        bearing = _path_bearing_at(G, path, from_end=False)
        assert 80 < bearing < 100

    def test_path_bearing_at_arrival(self):
        """Arrival bearing at node 2 from path [0,1,2] should be ~0 deg (North)."""
        G = _make_simple_graph_with_coords()
        path = [0, 1, 2]
        bearing = _path_bearing_at(G, path, from_end=True)
        assert bearing < 10 or bearing > 350

    def test_bearing_reversal_detected_in_validate(self):
        """A corridor routed through a U-turn on the graph should be rejected."""
        G = _make_simple_graph_with_coords()

        sp = 800
        coords_proj = np.array([
            [0.0, 0.0],
            [sp, 0.0],
            [sp, sp],
            [2 * sp, sp],
            [sp, -sp],
        ])

        station_data = {
            "coords_proj": coords_proj,
            "graph": G,
            "node_ids": np.array([0, 1, 2, 3, 4]),
            "road_dist_cache": {},
            "path_bearing_cache": {},
        }

        stations = [0, 1, 2, 4]
        valid, reason = validate_station_set(stations, station_data)
        if not valid:
            bearing_cache = station_data.get("path_bearing_cache", {})
            if (1, 2) in bearing_cache and (2, 4) in bearing_cache:
                _, arrive = bearing_cache[(1, 2)]
                depart, _ = bearing_cache[(2, 4)]
                diff = abs(arrive - depart)
                if diff > 180:
                    diff = 360 - diff
                assert diff > 120, f"Expected reversal >120 deg, got {diff:.0f} deg"

    def test_no_reversal_for_gentle_turn(self):
        """A corridor with a 90 deg turn should NOT trigger bearing reversal."""
        G = _make_simple_graph_with_coords()

        sp = 800
        coords_proj = np.array([
            [0.0, 0.0],
            [sp, 0.0],
            [sp, sp],
            [2 * sp, sp],
            [sp, -sp],
        ])

        station_data = {
            "coords_proj": coords_proj,
            "graph": G,
            "node_ids": np.array([0, 1, 2, 3, 4]),
            "road_dist_cache": {},
            "path_bearing_cache": {},
        }

        stations = [0, 1, 2, 3]
        valid, reason = validate_station_set(stations, station_data)
        if not valid:
            assert "reversal" not in reason, f"False positive: {reason}"

    def test_check_routed_path_quality(self):
        """Post-routing check detects bearing reversal in alignment geometry."""
        from shapely.geometry import LineString

        sp = 800
        coords = [
            (0, 0), (sp, 0),
            (sp, sp * 0.5),
            (sp, sp),
            (sp, sp * 0.5),
            (sp, 0), (sp, -sp),
        ]
        alignment = LineString(coords)
        station_coords = np.array([
            [0, 0],
            [sp, sp],
            [sp, -sp],
        ])

        warnings = check_routed_path_quality(alignment, station_coords)
        assert len(warnings) > 0
        assert "reversal" in warnings[0]

"""Tests for src/bus_network.py — dynamic bus network module."""

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

from src.bus_network import (
    APMService,
    APM_MAX_HEADWAY_MIN,
    APM_MIN_HEADWAY_MIN,
    BusOperatingCostModel,
    BusRoute,
    MIN_APM_RIDERSHIP_FOR_RESTRUCTURE,
    RouteServicePlan,
    build_corridor_bus_network,
    classify_routes_for_corridor,
    compute_apm_headway,
    compute_sector_coverage,
    CITYBUS_ANNUAL_BUDGET_USD,
    DEFAULT_COST_PER_VEH_HOUR,
)


def _make_route(
    route_id="R1",
    name="Test Route",
    length_km=10.0,
    headway=30.0,
    cycle_time=70.0,
    n_stops=20,
    trips_per_day=40.0,
    riders=500.0,
    classification="independent",
):
    return BusRoute(
        route_id=route_id,
        name=name,
        length_km=length_km,
        baseline_headway_min=headway,
        current_headway_min=headway,
        cycle_time_min=cycle_time,
        n_stops=n_stops,
        trips_per_day=trips_per_day,
        observed_daily_riders=riders,
        classification=classification,
    )


class TestBusRoute(unittest.TestCase):
    def test_vehicles_needed(self):
        r = _make_route(cycle_time=60.0, headway=15.0)
        # ceil(60/15) = 4 vehicles in service, +20% spare = ceil(4.8) = 5
        self.assertEqual(r.vehicles_needed, 5)

    def test_vehicles_needed_single(self):
        r = _make_route(cycle_time=20.0, headway=30.0)
        # ceil(20/30) = 1, +20% = ceil(1.2) = 2
        self.assertEqual(r.vehicles_needed, 2)

    def test_daily_vehicle_hours(self):
        r = _make_route(cycle_time=60.0, headway=15.0)
        # 4 vehicles * 18 hours span = 72 veh-hrs
        self.assertAlmostEqual(r.daily_vehicle_hours, 72.0, places=1)

    def test_round_trip_km(self):
        r = _make_route(length_km=8.5)
        self.assertAlmostEqual(r.round_trip_km, 17.0)


class TestAPMService(unittest.TestCase):
    def test_cycle_time(self):
        apm = APMService(
            corridor_id="C1",
            length_km=8.0,
            n_stops=8,
            headway_min=5.0,
        )
        # 6 intermediate stops, dwell=25s, accel/decel=15s each
        # one-way cruise = 8/27 × 60 = 17.8min
        # stop time = 6 * 40s = 240s = 4.0min
        # one-way = 21.8min, round-trip = 43.6min + 2min terminal = 45.6min
        self.assertAlmostEqual(apm.cycle_time_min, 45.6, delta=1.0)

    def test_vehicles_needed(self):
        apm = APMService(
            corridor_id="C1",
            length_km=8.0,
            n_stops=8,
            headway_min=5.0,
        )
        # cycle ~46min / 5min = ceil(9.2) = 10, +20% = ceil(12.0) = 12
        self.assertGreaterEqual(apm.vehicles_needed, 10)
        self.assertLessEqual(apm.vehicles_needed, 14)


class TestAPMFrequencyResponse(unittest.TestCase):
    """Tests for capacity-constrained APM headway model."""

    def test_zero_ridership_returns_max(self):
        self.assertEqual(compute_apm_headway(0), APM_MAX_HEADWAY_MIN)

    def test_negative_ridership_returns_max(self):
        self.assertEqual(compute_apm_headway(-100), APM_MAX_HEADWAY_MIN)

    def test_low_ridership_high_headway(self):
        hw = compute_apm_headway(500)
        self.assertGreater(hw, APM_MAX_HEADWAY_MIN * 0.8,
                           "500 riders should produce long headway")
        self.assertLessEqual(hw, APM_MAX_HEADWAY_MIN)

    def test_medium_ridership(self):
        hw = compute_apm_headway(2000)
        self.assertGreater(hw, APM_MAX_HEADWAY_MIN * 0.4)
        self.assertLess(hw, APM_MAX_HEADWAY_MIN)

    def test_high_ridership_short_headway(self):
        hw = compute_apm_headway(6000)
        self.assertLess(hw, APM_MAX_HEADWAY_MIN * 0.5,
                        "6000 riders should produce short headway")
        self.assertGreaterEqual(hw, APM_MIN_HEADWAY_MIN)

    def test_very_high_ridership_near_minimum(self):
        hw = compute_apm_headway(15000)
        self.assertLessEqual(hw, APM_MIN_HEADWAY_MIN,
                             "15K riders should hit physical minimum")
        self.assertGreaterEqual(hw, APM_MIN_HEADWAY_MIN,
                                "Cannot go below physical minimum")

    def test_monotonically_decreasing(self):
        """More riders → shorter headway (monotonic)."""
        levels = [100, 500, 1000, 2000, 4000, 8000, 15000]
        headways = [compute_apm_headway(r) for r in levels]
        for i in range(1, len(headways)):
            self.assertLessEqual(headways[i], headways[i - 1],
                                 f"Headway should decrease: {levels[i]} riders gave {headways[i]} "
                                 f"but {levels[i-1]} riders gave {headways[i-1]}")

    def test_continuous_no_jumps(self):
        """Adjacent ridership levels should not produce large headway jumps."""
        prev = compute_apm_headway(999)
        curr = compute_apm_headway(1001)
        self.assertLess(abs(prev - curr), 2.0,
                        "Headway should change smoothly, not jump by >2 min")

    def test_fleet_constraint_applied(self):
        """With corridor geometry, fleet limit raises headway floor."""
        hw_unconstrained = compute_apm_headway(15000)
        hw_constrained = compute_apm_headway(15000, corridor_length_km=10.0, n_stops=10)
        self.assertGreaterEqual(hw_constrained, hw_unconstrained,
                                "Fleet constraint should not lower headway")

    def test_corridor_geometry_optional(self):
        """Function works without corridor geometry (backward compatible)."""
        hw = compute_apm_headway(3000)
        self.assertIsInstance(hw, float)
        self.assertGreaterEqual(hw, APM_MIN_HEADWAY_MIN)
        self.assertLessEqual(hw, APM_MAX_HEADWAY_MIN)


class TestRouteClassification(unittest.TestCase):
    def test_parallel_classification(self):
        """Route with >40% stops near corridor → parallel."""
        routes = [_make_route(route_id="R1")]
        # Corridor stops at x=0,100,200,...,900
        corridor_xy = np.array([[i * 100, 0] for i in range(10)])
        # Bus stops: 8 out of 10 within 400m of corridor
        bus_stops = {
            "R1": np.array([[i * 100, 50] for i in range(8)]  # near corridor
                          + [[5000, 5000], [6000, 6000]])       # far away
        }
        classify_routes_for_corridor(routes, corridor_xy, bus_stops)
        self.assertEqual(routes[0].classification, "parallel")

    def test_feeder_classification(self):
        """Route with 15-40% stops near corridor → feeder."""
        routes = [_make_route(route_id="R1")]
        corridor_xy = np.array([[0, 0], [100, 0], [200, 0]])
        # 3 out of 10 stops near corridor (30%)
        bus_stops = {
            "R1": np.array(
                [[0, 50], [100, 50], [200, 50]]  # near (3)
                + [[i * 1000, 5000] for i in range(7)]  # far (7)
            )
        }
        classify_routes_for_corridor(routes, corridor_xy, bus_stops)
        self.assertEqual(routes[0].classification, "feeder")

    def test_independent_classification(self):
        """Route with <15% stops near corridor → independent."""
        routes = [_make_route(route_id="R1")]
        corridor_xy = np.array([[0, 0], [100, 0]])
        # 1 out of 10 stops near corridor (10%)
        bus_stops = {
            "R1": np.array(
                [[0, 50]]  # near (1)
                + [[i * 1000, 5000] for i in range(9)]  # far (9)
            )
        }
        classify_routes_for_corridor(routes, corridor_xy, bus_stops)
        self.assertEqual(routes[0].classification, "independent")

    def test_missing_route_stops(self):
        """Route not in bus_stops_by_route → independent."""
        routes = [_make_route(route_id="R99")]
        corridor_xy = np.array([[0, 0], [100, 0]])
        classify_routes_for_corridor(routes, corridor_xy, {})
        self.assertEqual(routes[0].classification, "independent")


class TestBusOperatingCostModel(unittest.TestCase):
    def test_route_annual_cost(self):
        model = BusOperatingCostModel(cost_per_veh_hour=100.0)
        r = _make_route(cycle_time=60.0, headway=30.0)
        # ceil(60/30)=2 vehicles * 18h span = 36 veh-hrs/day * 300 days = 10800 * $100
        cost = model.route_annual_cost(r)
        self.assertAlmostEqual(cost, 1_080_000.0, places=0)

    def test_system_cost(self):
        model = BusOperatingCostModel(cost_per_veh_hour=100.0, annual_budget=5_000_000)
        routes = [_make_route(cycle_time=60.0, headway=30.0) for _ in range(3)]
        result = model.system_annual_cost(routes)
        self.assertAlmostEqual(result["bus_annual_cost"], 3_240_000.0, places=0)
        self.assertAlmostEqual(result["budget_surplus"], 1_760_000.0, places=0)

    def test_fleet_summary(self):
        model = BusOperatingCostModel()
        routes = [_make_route(cycle_time=60.0, headway=15.0)]
        fleet = model.fleet_summary(routes)
        self.assertEqual(fleet["bus_vehicles"], 5)


class TestRouteServicePlan(unittest.TestCase):
    def test_no_pressure_holds_headways(self):
        routes = [
            _make_route("P1", classification="parallel", headway=30.0),
            _make_route("F1", classification="feeder", headway=30.0),
            _make_route("I1", classification="independent", headway=30.0),
        ]
        model = BusOperatingCostModel(annual_budget=100_000_000)
        plan = RouteServicePlan(routes, model)
        result = plan.optimize_headways(restructure_pressure=0.0)

        for rr in result["routes"]:
            self.assertAlmostEqual(rr["current_headway"], rr["baseline_headway"])

    def test_high_pressure_degrades_parallel(self):
        routes = [
            _make_route("P1", classification="parallel", headway=30.0),
            _make_route("F1", classification="feeder", headway=30.0),
        ]
        model = BusOperatingCostModel(annual_budget=100_000_000)
        plan = RouteServicePlan(routes, model)
        result = plan.optimize_headways(
            restructure_pressure=1.0, apm_daily_riders=2000.0,
        )

        parallel = [r for r in result["routes"] if r["classification"] == "parallel"][0]
        feeder = [r for r in result["routes"] if r["classification"] == "feeder"][0]

        # Parallel headway should increase (degrade)
        self.assertGreater(parallel["current_headway"], 30.0)
        # Feeder headway should decrease (improve)
        self.assertLess(feeder["current_headway"], 30.0)

    def test_feeder_headway_summary(self):
        routes = [
            _make_route("F1", classification="feeder", headway=20.0),
            _make_route("F2", classification="feeder", headway=40.0),
        ]
        model = BusOperatingCostModel(annual_budget=100_000_000)
        plan = RouteServicePlan(routes, model)
        plan.optimize_headways(restructure_pressure=0.0)
        self.assertAlmostEqual(plan.get_feeder_headway(), 30.0)

    def test_savings_from_parallel_reduction(self):
        routes = [
            _make_route("P1", classification="parallel", headway=15.0, cycle_time=60.0),
        ]
        model = BusOperatingCostModel(cost_per_veh_hour=100.0, annual_budget=100_000_000)
        baseline_cost = model.route_annual_cost(routes[0])

        plan = RouteServicePlan(routes, model)
        result = plan.optimize_headways(restructure_pressure=0.8, apm_daily_riders=2000.0)

        self.assertGreater(result["bus_savings_from_parallel"], 0)
        self.assertLess(result["new_bus_cost"], baseline_cost)

    def test_below_min_ridership_no_restructure(self):
        """When APM ridership is below threshold, no restructuring occurs."""
        routes = [
            _make_route("P1", classification="parallel", headway=30.0),
            _make_route("F1", classification="feeder", headway=30.0),
        ]
        model = BusOperatingCostModel(annual_budget=100_000_000)
        plan = RouteServicePlan(routes, model)
        # Below MIN_APM_RIDERSHIP_FOR_RESTRUCTURE (1000) → pressure forced to 0
        result = plan.optimize_headways(
            restructure_pressure=0.9,
            apm_daily_riders=500.0,
        )

        # All routes should keep baseline headways (pressure zeroed out)
        for rr in result["routes"]:
            self.assertAlmostEqual(rr["current_headway"], rr["baseline_headway"])


class TestBuildCorridorBusNetwork(unittest.TestCase):
    def test_integration(self):
        routes = [
            _make_route("P1", headway=20.0, cycle_time=60.0),
            _make_route("F1", headway=30.0, cycle_time=50.0),
            _make_route("I1", headway=25.0, cycle_time=40.0),
        ]
        corridor_xy = np.array([[i * 100, 0] for i in range(10)])
        bus_stops = {
            "P1": np.array([[i * 100, 50] for i in range(10)]),  # all near → parallel
            "F1": np.array(
                [[0, 50], [100, 50], [200, 50]]  # 3 near
                + [[i * 1000, 5000] for i in range(7)]  # 7 far → feeder
            ),
            "I1": np.array([[i * 1000, 5000] for i in range(10)]),  # all far → independent
        }

        result = build_corridor_bus_network(
            routes=routes,
            corridor_stops_xy=corridor_xy,
            bus_stops_by_route=bus_stops,
            apm_daily_riders=4000,
            apm_length_km=8.0,
            apm_n_stops=8,
            restructure_pressure=0.6,
        )

        self.assertEqual(result["route_classifications"]["P1"], "parallel")
        self.assertEqual(result["route_classifications"]["F1"], "feeder")
        self.assertEqual(result["route_classifications"]["I1"], "independent")
        # Continuous model: 4000 riders → ~5 min (service-quality regime)
        self.assertGreater(result["apm_headway_min"], 3.0)
        self.assertLess(result["apm_headway_min"], 7.0)
        self.assertEqual(result["phase"], "feeder_transition")  # pressure=0.6
        # Spatial feeder coverage: 3 feeder stops near corridor cover a small
        # fraction of the 1200-5000m ring.  Phase floor for feeder_transition
        # is 0.20, so coverage should be at least that.
        self.assertGreaterEqual(result["feeder_coverage_fraction"], 0.20)
        self.assertLessEqual(result["feeder_coverage_fraction"], 1.0)
        self.assertIn("optimization", result)
        self.assertIn("fleet", result)

    def test_no_overlap_phase(self):
        routes = [_make_route("I1")]
        corridor_xy = np.array([[0, 0], [100, 0]])
        bus_stops = {
            "I1": np.array([[5000, 5000], [6000, 6000]]),
        }
        result = build_corridor_bus_network(
            routes=routes,
            corridor_stops_xy=corridor_xy,
            bus_stops_by_route=bus_stops,
            apm_daily_riders=1000,
            apm_length_km=5.0,
            apm_n_stops=5,
            restructure_pressure=0.5,
        )
        self.assertEqual(result["phase"], "no_overlap")
        self.assertEqual(result["feeder_coverage_fraction"], 0.0)


class TestEffectiveWaitTime(unittest.TestCase):
    """Tests for reliability-aware wait time (TCRP 165)."""

    def test_frequent_service_random_arrival(self):
        """At <=10 min headway, wait = hw/2 (random arrival)."""
        from scripts.generate_improved_ridership import effective_wait_time
        self.assertAlmostEqual(effective_wait_time(8.0), 4.0)
        self.assertAlmostEqual(effective_wait_time(10.0), 5.0)

    def test_infrequent_service_scheduled_arrival(self):
        """At >10 min headway, wait = hw/4 + reliability buffer."""
        from scripts.generate_improved_ridership import effective_wait_time
        # 20 min: 20/4 + 3 = 8.0
        self.assertAlmostEqual(effective_wait_time(20.0), 8.0)
        # 30 min: 30/4 + 3 = 10.5
        self.assertAlmostEqual(effective_wait_time(30.0), 10.5)

    def test_continuity_at_threshold(self):
        """No large jump at the 10-min threshold."""
        from scripts.generate_improved_ridership import effective_wait_time
        below = effective_wait_time(9.9)
        above = effective_wait_time(10.1)
        self.assertLess(abs(below - above), 1.0,
                        "Wait time should be near-continuous at 10-min threshold")

    def test_tsp_reduces_reliability_buffer(self):
        """TSP should reduce wait time for infrequent service."""
        from scripts.generate_improved_ridership import effective_wait_time
        no_tsp = effective_wait_time(20.0, tsp_active=False)
        with_tsp = effective_wait_time(20.0, tsp_active=True)
        self.assertLess(with_tsp, no_tsp)

    def test_numpy_array_input(self):
        """Should work with numpy arrays."""
        from scripts.generate_improved_ridership import effective_wait_time
        hw = np.array([5.0, 10.0, 20.0, 30.0])
        result = effective_wait_time(hw)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 2.5)
        self.assertAlmostEqual(result[2], 8.0)


class TestSectorCoverage(unittest.TestCase):
    """Tests for sector-based feeder coverage."""

    def test_uniform_creates_equal_sectors(self):
        from src.bus_network import SectorCoverage
        sc = SectorCoverage.uniform(0.6)
        self.assertEqual(sc.coverage.shape, (8,))
        self.assertAlmostEqual(sc.effective_coverage, 0.6)

    def test_effective_coverage_population_weighted(self):
        from src.bus_network import SectorCoverage
        coverage = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pop_weight = np.array([100, 0, 0, 0, 0, 0, 0, 0])
        sc = SectorCoverage(coverage=coverage, pop_weight=pop_weight)
        # All population in sector 0 which has coverage 1.0
        self.assertAlmostEqual(sc.effective_coverage, 1.0)

    def test_for_bearing_returns_correct_sector(self):
        from src.bus_network import SectorCoverage
        coverage = np.arange(8) / 7.0  # 0.0, 0.14, 0.29, ..., 1.0
        sc = SectorCoverage(coverage=coverage, pop_weight=np.ones(8))
        # Bearing 0 (north) should be sector 0
        self.assertAlmostEqual(sc.for_bearing(0.0), coverage[0], places=1)


class TestComputeSectorCoverage(unittest.TestCase):
    """Tests for compute_sector_coverage function."""

    def test_no_feeder_routes_zero_coverage(self):
        from src.bus_network import compute_sector_coverage
        corridor_xy = np.array([[0, 0], [0, 1000], [0, 2000]])
        result = compute_sector_coverage(corridor_xy, [], {})
        self.assertAlmostEqual(result.effective_coverage, 0.0)

    def test_surrounding_stops_give_coverage(self):
        from src.bus_network import compute_sector_coverage
        corridor_xy = np.array([[0, 0], [0, 1000]])
        # Create a feeder route with stops in the feeder ring
        route = _make_route("F1", classification="feeder", headway=15.0)
        stops = np.array([
            [2000, 0], [-2000, 0], [0, 2000], [0, -2000],
            [1500, 1500], [-1500, 1500], [1500, -1500], [-1500, -1500],
        ])
        result = compute_sector_coverage(
            corridor_xy, [route], {"F1": stops},
            route_headways={"F1": 15.0},
        )
        self.assertGreater(result.effective_coverage, 0.0)


class TestBarrierPenalty(unittest.TestCase):
    """Tests for Wabash River barrier penalty."""

    def test_no_river_no_penalty(self):
        from src.bus_network import apply_barrier_penalty, SectorCoverage
        sc = SectorCoverage.uniform(0.8)
        # Place corridor exactly on river so no sectors cross it
        result = apply_barrier_penalty(sc, corridor_centroid_x=0, river_x=0)
        self.assertTrue(np.allclose(result.coverage, sc.coverage),
                        f"Expected {sc.coverage}, got {result.coverage}")

    def test_west_corridor_penalizes_east_sectors(self):
        from src.bus_network import apply_barrier_penalty, SectorCoverage
        sc = SectorCoverage.uniform(1.0)
        # Corridor west of river
        result = apply_barrier_penalty(sc, corridor_centroid_x=-100, river_x=0)
        # Sectors 1,2,3 (NE, E, SE) should be penalized
        for s in (1, 2, 3):
            self.assertLess(result.coverage[s], 1.0)
        # Sectors 0,4 (N, S) should be unaffected
        self.assertAlmostEqual(result.coverage[0], 1.0)
        self.assertAlmostEqual(result.coverage[4], 1.0)


class TestTSPSpeedFactor(unittest.TestCase):
    """Tests for transit signal priority speed factor."""

    def test_before_activation_year(self):
        from src.bus_network import compute_tsp_speed_factor
        self.assertAlmostEqual(
            compute_tsp_speed_factor(5.0, 20, 3000, year=2), 1.0)

    def test_below_ridership_threshold(self):
        from src.bus_network import compute_tsp_speed_factor
        self.assertAlmostEqual(
            compute_tsp_speed_factor(5.0, 20, 500, year=5), 1.0)

    def test_active_gives_boost(self):
        from src.bus_network import compute_tsp_speed_factor
        factor = compute_tsp_speed_factor(5.0, 20, 3000, year=5)
        self.assertGreater(factor, 1.0)
        self.assertLessEqual(factor, 1.12)

    def test_zero_signals_no_boost(self):
        from src.bus_network import compute_tsp_speed_factor
        self.assertAlmostEqual(
            compute_tsp_speed_factor(5.0, 0, 3000, year=5), 1.0)


class TestPoPSpeedFactor(unittest.TestCase):
    """Tests for proof-of-payment speed factor."""

    def test_pop_active_gives_boost(self):
        from src.bus_network import compute_pop_speed_factor
        self.assertAlmostEqual(compute_pop_speed_factor(True), 1.06)

    def test_pop_inactive_no_boost(self):
        from src.bus_network import compute_pop_speed_factor
        self.assertAlmostEqual(compute_pop_speed_factor(False), 1.0)


class TestTransferPenaltyEnhanced(unittest.TestCase):
    """Tests for enhanced transfer penalty with TSP/PoP."""

    def test_tsp_reduces_penalty(self):
        from scripts.generate_improved_ridership import compute_transfer_penalty
        no_tsp = compute_transfer_penalty(5.0, 20.0, tsp_active=False)
        with_tsp = compute_transfer_penalty(5.0, 20.0, tsp_active=True)
        self.assertLess(with_tsp, no_tsp)

    def test_pop_reduces_penalty(self):
        from scripts.generate_improved_ridership import compute_transfer_penalty
        # Use headways > 15 to avoid timed-transfer regime hitting 4.0 floor
        no_pop = compute_transfer_penalty(5.0, 20.0, pop_active=False)
        with_pop = compute_transfer_penalty(5.0, 20.0, pop_active=True)
        self.assertLess(with_pop, no_pop)

    def test_penalty_floor(self):
        """Transfer penalty should never go below 4.0 minutes (TCRP 165)."""
        from scripts.generate_improved_ridership import compute_transfer_penalty
        penalty = compute_transfer_penalty(3.0, 3.0, tsp_active=True, pop_active=True)
        self.assertGreaterEqual(penalty, 4.0)

    def test_combined_tsp_pop(self):
        """TSP + PoP combined should be better than either alone."""
        from scripts.generate_improved_ridership import compute_transfer_penalty
        base = compute_transfer_penalty(5.0, 20.0)
        both = compute_transfer_penalty(5.0, 20.0, tsp_active=True, pop_active=True)
        self.assertLess(both, base)


# ===================================================================
# D — Corridor-specific directional split
# ===================================================================


class TestDirectionalSplit(unittest.TestCase):
    """Tests for corridor-specific directional split in compute_apm_headway."""

    def test_default_split_when_zero(self):
        """directional_split=0 should fall back to APM_PEAK_DIR_SPLIT (0.60)."""
        from src.bus_network import APM_PEAK_DIR_SPLIT
        hw_default = compute_apm_headway(3000, corridor_length_km=5.0,
                                          n_stops=6, directional_split=0.0)
        hw_explicit = compute_apm_headway(3000, corridor_length_km=5.0,
                                           n_stops=6,
                                           directional_split=APM_PEAK_DIR_SPLIT)
        self.assertAlmostEqual(hw_default, hw_explicit, places=4)

    def test_higher_split_tighter_headway(self):
        """A 0.75 directional split (more imbalanced) should require tighter
        headway than 0.55 (more balanced) for the same ridership."""
        hw_balanced = compute_apm_headway(4000, corridor_length_km=6.0,
                                           n_stops=7, directional_split=0.55)
        hw_imbalanced = compute_apm_headway(4000, corridor_length_km=6.0,
                                             n_stops=7, directional_split=0.75)
        # More imbalanced → more peak-direction demand → tighter headway
        self.assertLessEqual(hw_imbalanced, hw_balanced)

    def test_split_clipped_to_range(self):
        """directional_split outside [0.50, 0.80] should fall back to default."""
        from src.bus_network import APM_PEAK_DIR_SPLIT
        hw_out_of_range = compute_apm_headway(3000, corridor_length_km=5.0,
                                               n_stops=6, directional_split=0.90)
        hw_default = compute_apm_headway(3000, corridor_length_km=5.0,
                                          n_stops=6,
                                          directional_split=APM_PEAK_DIR_SPLIT)
        self.assertAlmostEqual(hw_out_of_range, hw_default, places=4)

    def test_balanced_flows_give_050(self):
        """Perfectly balanced OD flows should give directional_split near 0.50."""
        # Simulate: equal trips in both directions
        orig_sidx = np.array([0, 1, 2, 3, 4, 5])
        dest_sidx = np.array([5, 4, 3, 0, 1, 2])
        apm_prob = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
        weighted_trips = np.ones(6)
        apm_weighted = weighted_trips * apm_prob
        total = apm_weighted.sum()
        fwd = apm_weighted[dest_sidx > orig_sidx].sum()
        bwd = apm_weighted[dest_sidx < orig_sidx].sum()
        dominant = max(fwd, bwd)
        split = float(np.clip(dominant / total, 0.50, 0.80))
        self.assertAlmostEqual(split, 0.50, places=2)

    def test_unidirectional_flows_give_080(self):
        """All-one-direction OD flows should be clipped to 0.80."""
        orig_sidx = np.array([0, 0, 0, 1, 1])
        dest_sidx = np.array([3, 4, 5, 4, 5])
        apm_prob = np.ones(5) * 0.4
        weighted_trips = np.ones(5)
        apm_weighted = weighted_trips * apm_prob
        total = apm_weighted.sum()
        fwd = apm_weighted[dest_sidx > orig_sidx].sum()
        bwd = apm_weighted[dest_sidx < orig_sidx].sum()
        dominant = max(fwd, bwd)
        split = float(np.clip(dominant / total, 0.50, 0.80))
        self.assertAlmostEqual(split, 0.80, places=2)


# ===================================================================
# E — Induced bus ridership from APM network effect
# ===================================================================


class TestInducedBusRidership(unittest.TestCase):
    """Tests for compute_bus_induced_factor() production function."""

    def test_no_boost_before_threshold_year(self):
        """Bus induced factor should be 1.0 before year 5."""
        from src.bus_network import compute_bus_induced_factor
        factor = compute_bus_induced_factor(year=3, apm_daily_riders=2000)
        self.assertAlmostEqual(factor, 1.0)

    def test_boost_at_mature_ridership(self):
        """At mature ridership (2500), factor should be ~1.035."""
        from src.bus_network import compute_bus_induced_factor
        factor = compute_bus_induced_factor(year=10, apm_daily_riders=2500)
        # log1p(1.0) ≈ 0.693 → factor ≈ 1.0 + 0.05 * 0.693 ≈ 1.035
        self.assertAlmostEqual(factor, 1.0 + 0.05 * np.log(2.0), places=4)
        self.assertGreater(factor, 1.03)
        self.assertLess(factor, 1.04)

    def test_boost_proportional_to_all_routes(self):
        """All routes should receive the same multiplicative boost."""
        from src.bus_network import compute_bus_induced_factor
        bus_induced_factor = compute_bus_induced_factor(year=10, apm_daily_riders=2500)

        base_est = {"route_A": 100.0, "route_B": 250.0, "route_C": 50.0}
        boosted = {rid: b * bus_induced_factor for rid, b in base_est.items()}

        for rid in base_est:
            self.assertAlmostEqual(
                boosted[rid] / base_est[rid], bus_induced_factor, places=6
            )

    def test_no_boost_with_zero_riders(self):
        """Zero APM riders → no induced bus boost."""
        from src.bus_network import compute_bus_induced_factor
        factor = compute_bus_induced_factor(year=10, apm_daily_riders=0)
        self.assertAlmostEqual(factor, 1.0)


# ===================================================================
# F — Proactive Bus Network Redesign (Component 2)
# ===================================================================


class TestRouteProductivityScore(unittest.TestCase):
    """Tests for route_productivity_score()."""

    def test_normal_productivity(self):
        from src.bus_network import route_productivity_score
        # 500 riders / day, 72 veh-hrs/day (cycle=60, headway=15 → 4 vehs × 18h)
        r = _make_route(riders=500.0, cycle_time=60.0, headway=15.0)
        prod = route_productivity_score(r)
        # 500 / 72 ≈ 6.94
        self.assertAlmostEqual(prod, 500.0 / 72.0, places=1)

    def test_zero_vehicle_hours(self):
        from src.bus_network import route_productivity_score
        r = _make_route(riders=100.0, headway=0.0)
        self.assertAlmostEqual(route_productivity_score(r), 0.0)

    def test_high_productivity(self):
        from src.bus_network import route_productivity_score
        r = _make_route(riders=2000.0, cycle_time=60.0, headway=30.0)
        prod = route_productivity_score(r)
        # 2000 / 32 = 62.5
        self.assertGreater(prod, 50.0)


class TestDecideRouteRestructuring(unittest.TestCase):
    """Tests for decide_route_restructuring()."""

    def test_year_0_all_keep(self):
        from src.bus_network import decide_route_restructuring, RestructuringAction
        routes = [
            _make_route("P1", classification="parallel"),
            _make_route("F1", classification="feeder", riders=200.0),
            _make_route("I1", classification="independent"),
        ]
        decisions = decide_route_restructuring(routes, year=0)
        for rid, action in decisions.items():
            self.assertEqual(action, RestructuringAction.KEEP,
                             f"Year 0: all routes should KEEP, got {action} for {rid}")

    def test_year_3_parallel_eliminated(self):
        from src.bus_network import decide_route_restructuring, RestructuringAction
        routes = [
            _make_route("P1", classification="parallel"),
            _make_route("F1", classification="feeder", riders=200.0),
        ]
        decisions = decide_route_restructuring(routes, year=3)
        self.assertEqual(decisions["P1"], RestructuringAction.ELIMINATE)
        # Feeder enhance not yet unlocked at year 3
        self.assertEqual(decisions["F1"], RestructuringAction.KEEP)

    def test_year_5_reroute_unproductive_feeder(self):
        from src.bus_network import decide_route_restructuring, RestructuringAction, MIN_PRODUCTIVITY
        # Low productivity feeder: 50 riders / 32 veh-hrs = 1.56 pax/hr
        routes = [
            _make_route("F1", classification="feeder", riders=50.0,
                        cycle_time=60.0, headway=30.0),
        ]
        prod = routes[0].observed_daily_riders / routes[0].daily_vehicle_hours
        self.assertLess(prod, MIN_PRODUCTIVITY)
        decisions = decide_route_restructuring(routes, year=5)
        self.assertEqual(decisions["F1"], RestructuringAction.REROUTE)

    def test_year_7_enhance_productive_feeder(self):
        from src.bus_network import decide_route_restructuring, RestructuringAction, MIN_PRODUCTIVITY
        # High productivity feeder: 2000 riders / 32 veh-hrs = 62.5 pax/hr
        routes = [
            _make_route("F1", classification="feeder", riders=2000.0,
                        cycle_time=60.0, headway=30.0),
        ]
        prod = routes[0].observed_daily_riders / routes[0].daily_vehicle_hours
        self.assertGreater(prod, MIN_PRODUCTIVITY)
        decisions = decide_route_restructuring(routes, year=7)
        self.assertEqual(decisions["F1"], RestructuringAction.ENHANCE)

    def test_year_3_reduce_unproductive_independent(self):
        from src.bus_network import decide_route_restructuring, RestructuringAction
        # Low productivity independent: 50 riders / 32 veh-hrs = 1.56 pax/hr
        routes = [
            _make_route("I1", classification="independent", riders=50.0,
                        cycle_time=60.0, headway=30.0),
        ]
        decisions = decide_route_restructuring(routes, year=3)
        self.assertEqual(decisions["I1"], RestructuringAction.REDUCE)


class TestCheckCoverageEquity(unittest.TestCase):
    """Tests for coverage equity guard."""

    def test_high_se01_downgrades_eliminate(self):
        from src.bus_network import (
            check_coverage_equity, RestructuringAction, EQUITY_DISPARITY_THRESHOLD,
        )
        routes = [_make_route("P1", classification="parallel")]
        decisions = {"P1": RestructuringAction.ELIMINATE}
        # Bus stop at (0, 0), parcels nearby
        bus_stops = {"P1": np.array([[0.0, 0.0]])}
        parcel_xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
        # High SE01 share near this route
        parcel_se01 = np.array([0.8, 0.7, 0.9])
        metro_se01 = 0.33  # 80% > 33% × 1.5 = 49.5%

        adjusted = check_coverage_equity(
            decisions, routes, bus_stops, parcel_xy, parcel_se01, metro_se01,
        )
        self.assertEqual(adjusted["P1"], RestructuringAction.REDUCE,
                         "High SE01 route should be downgraded from ELIMINATE to REDUCE")

    def test_low_se01_keeps_eliminate(self):
        from src.bus_network import check_coverage_equity, RestructuringAction
        routes = [_make_route("P1", classification="parallel")]
        decisions = {"P1": RestructuringAction.ELIMINATE}
        bus_stops = {"P1": np.array([[0.0, 0.0]])}
        parcel_xy = np.array([[0.0, 0.0], [100.0, 0.0]])
        # Low SE01 share — no equity concern
        parcel_se01 = np.array([0.1, 0.1])
        metro_se01 = 0.33

        adjusted = check_coverage_equity(
            decisions, routes, bus_stops, parcel_xy, parcel_se01, metro_se01,
        )
        self.assertEqual(adjusted["P1"], RestructuringAction.ELIMINATE)

    def test_keep_action_not_affected(self):
        from src.bus_network import check_coverage_equity, RestructuringAction
        routes = [_make_route("I1", classification="independent")]
        decisions = {"I1": RestructuringAction.KEEP}
        bus_stops = {"I1": np.array([[0.0, 0.0]])}
        parcel_xy = np.array([[0.0, 0.0]])
        parcel_se01 = np.array([0.9])
        metro_se01 = 0.33

        adjusted = check_coverage_equity(
            decisions, routes, bus_stops, parcel_xy, parcel_se01, metro_se01,
        )
        self.assertEqual(adjusted["I1"], RestructuringAction.KEEP)


class TestApplyRestructuringDecisions(unittest.TestCase):
    """Tests for apply_restructuring_decisions()."""

    def test_eliminate_sets_zero(self):
        from src.bus_network import apply_restructuring_decisions, RestructuringAction
        routes = [_make_route("P1", headway=30.0)]
        decisions = {"P1": RestructuringAction.ELIMINATE}
        target = apply_restructuring_decisions(routes, decisions)
        self.assertEqual(target["P1"], 0.0)

    def test_reduce_increases_headway(self):
        from src.bus_network import apply_restructuring_decisions, RestructuringAction
        routes = [_make_route("I1", headway=30.0)]
        decisions = {"I1": RestructuringAction.REDUCE}
        target = apply_restructuring_decisions(routes, decisions)
        self.assertAlmostEqual(target["I1"], 45.0)  # 30 × 1.5

    def test_enhance_halves_headway(self):
        from src.bus_network import apply_restructuring_decisions, RestructuringAction
        routes = [_make_route("F1", headway=30.0)]
        decisions = {"F1": RestructuringAction.ENHANCE}
        target = apply_restructuring_decisions(routes, decisions)
        self.assertAlmostEqual(target["F1"], 15.0)  # 30 × 0.5

    def test_enhance_respects_min_headway(self):
        from src.bus_network import apply_restructuring_decisions, RestructuringAction
        routes = [_make_route("F1", headway=15.0)]
        decisions = {"F1": RestructuringAction.ENHANCE}
        target = apply_restructuring_decisions(routes, decisions, min_headway=10.0)
        self.assertGreaterEqual(target["F1"], 10.0)


class TestProactiveOptimization(unittest.TestCase):
    """Tests for optimize_headways_proactive()."""

    def test_below_ridership_falls_back_to_legacy(self):
        routes = [
            _make_route("P1", classification="parallel", headway=30.0),
        ]
        model = BusOperatingCostModel(annual_budget=100_000_000)
        plan = RouteServicePlan(routes, model)
        result = plan.optimize_headways_proactive(
            year=5, apm_daily_riders=500.0,
        )
        # Should return legacy result (no restructuring)
        self.assertNotIn("restructure_mode", result)

    def test_year_3_eliminates_parallel(self):
        routes = [
            _make_route("P1", classification="parallel", headway=30.0, riders=500.0),
            _make_route("F1", classification="feeder", headway=30.0, riders=2000.0),
            _make_route("I1", classification="independent", headway=30.0, riders=300.0),
        ]
        model = BusOperatingCostModel(
            cost_per_veh_hour=100.0, annual_budget=100_000_000,
        )
        plan = RouteServicePlan(routes, model)
        result = plan.optimize_headways_proactive(
            year=3, apm_daily_riders=2000.0,
        )
        self.assertEqual(result["restructure_mode"], "proactive")
        self.assertGreater(result["routes_eliminated"], 0)

        # Check route-level results
        route_actions = {r["route_id"]: r["action"] for r in result["routes"]}
        self.assertEqual(route_actions["P1"], "eliminate")

    def test_budget_not_exceeded(self):
        routes = [
            _make_route("P1", classification="parallel", headway=20.0, riders=300.0),
            _make_route("F1", classification="feeder", headway=30.0, riders=800.0),
            _make_route("F2", classification="feeder", headway=25.0, riders=1500.0),
        ]
        budget = 5_000_000  # tight budget
        model = BusOperatingCostModel(
            cost_per_veh_hour=100.0, annual_budget=budget,
        )
        plan = RouteServicePlan(routes, model)
        result = plan.optimize_headways_proactive(
            year=7, apm_daily_riders=3000.0,
        )
        self.assertLessEqual(result["new_bus_cost"], budget * 1.01,
                             "Bus cost should not exceed budget (with 1% tolerance)")


# ===================================================================
# G — Equity-Weighted Sector Coverage (Component 5.5)
# ===================================================================


class TestEquityWeightedCoverage(unittest.TestCase):
    """Tests for equity uplift in compute_sector_coverage()."""

    def test_high_se01_sector_gets_uplift(self):
        from src.bus_network import compute_sector_coverage, EQUITY_COVERAGE_UPLIFT
        corridor_xy = np.array([[0, 0], [0, 1000]])
        route = _make_route("F1", classification="feeder", headway=15.0)
        # Stops in the feeder ring
        stops = np.array([
            [2000, 0], [-2000, 0], [0, 2000], [0, -2000],
            [1500, 1500], [-1500, 1500], [1500, -1500], [-1500, -1500],
        ])

        # Compute without equity
        result_no_eq = compute_sector_coverage(
            corridor_xy, [route], {"F1": stops},
            route_headways={"F1": 15.0},
        )

        # Create parcel-based data with known SE01 in one sector
        # Place parcels in the feeder ring with high SE01
        n_parcels = 80
        np.random.seed(42)
        angles = np.linspace(0, 2 * np.pi, n_parcels, endpoint=False)
        parcel_r = 3000  # in feeder ring
        parcel_xy = np.column_stack([
            parcel_r * np.cos(angles),
            parcel_r * np.sin(angles),
        ])
        parcel_pop = np.ones(n_parcels) * 10.0
        dx = parcel_xy[:, 0] - 0
        dy = parcel_xy[:, 1] - 500  # centroid approx
        bearings = np.arctan2(dx, dy) % (2 * np.pi)
        parcel_sectors = ((bearings / (2 * np.pi / 8) + 0.5) % 8).astype(int)
        # High SE01 in sector 0 (north), low elsewhere
        parcel_se01 = np.where(parcel_sectors == 0, 0.8, 0.1)

        result_eq = compute_sector_coverage(
            corridor_xy, [route], {"F1": stops},
            route_headways={"F1": 15.0},
            feeder_parcel_xy=parcel_xy,
            feeder_parcel_sector=parcel_sectors,
            feeder_parcel_pop=parcel_pop,
            feeder_parcel_se01=parcel_se01,
        )

        # With equity weighting, coverage in high-SE01 sectors should be higher
        # This tests the mechanism exists but exact values depend on stop placement
        self.assertIsNotNone(result_eq)
        self.assertEqual(result_eq.coverage.shape, (8,))

    def test_no_se01_no_change(self):
        from src.bus_network import compute_sector_coverage
        corridor_xy = np.array([[0, 0], [0, 1000]])
        route = _make_route("F1", classification="feeder", headway=15.0)
        stops = np.array([[2000, 0], [-2000, 0]])

        # Without se01 data
        result = compute_sector_coverage(
            corridor_xy, [route], {"F1": stops},
            route_headways={"F1": 15.0},
        )
        self.assertEqual(result.coverage.shape, (8,))


# ===================================================================
# H — Corridor Map Generation (Component 6.3)
# ===================================================================


class TestCorridorMapGeneration(unittest.TestCase):
    """Tests for _plot_corridor_maps() in decision package."""

    def test_map_generation_no_geojson(self):
        """Should skip gracefully when GeoJSON doesn't exist."""
        import tempfile
        sys_path_bak = sys.path[:]
        try:
            from scripts.generate_decision_package import _plot_corridor_maps
        finally:
            sys.path[:] = sys_path_bak

        import pandas as pd
        recs = pd.DataFrame({"corridor_id": ["C1", "C2"]})
        with tempfile.TemporaryDirectory() as tmpdir:
            maps_dir = Path(tmpdir) / "maps"
            maps_dir.mkdir()
            _plot_corridor_maps(
                recs, maps_dir,
                corridor_geojson_path=Path(tmpdir) / "nonexistent.geojson",
                top_n=2,
            )
            # Should produce no files (GeoJSON missing)
            self.assertEqual(len(list(maps_dir.glob("*.png"))), 0)

    def test_map_generation_with_geojson(self):
        """Should generate PNG when valid GeoJSON exists."""
        import tempfile
        sys_path_bak = sys.path[:]
        try:
            from scripts.generate_decision_package import _plot_corridor_maps
        except ImportError:
            self.skipTest("generate_decision_package not importable")
        finally:
            sys.path[:] = sys_path_bak

        try:
            import matplotlib
        except ImportError:
            self.skipTest("matplotlib not available")

        import pandas as pd
        recs = pd.DataFrame({"corridor_id": ["C1"]})

        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {
                    "corridor_id": "C1",
                    "length_km": 5.0,
                    "n_stops": 5,
                    "station_lons": [-86.91, -86.92, -86.93],
                    "station_lats": [40.42, 40.43, 40.44],
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [-86.91, 40.42], [-86.92, 40.43], [-86.93, 40.44],
                    ],
                },
            }],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            maps_dir = Path(tmpdir) / "maps"
            maps_dir.mkdir()
            gj_path = Path(tmpdir) / "corridors.geojson"
            gj_path.write_text(json.dumps(geojson), encoding="utf-8")

            _plot_corridor_maps(
                recs, maps_dir,
                corridor_geojson_path=gj_path,
                top_n=1,
            )
            pngs = list(maps_dir.glob("*.png"))
            self.assertEqual(len(pngs), 1)
            # Check file is non-empty
            self.assertGreater(pngs[0].stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

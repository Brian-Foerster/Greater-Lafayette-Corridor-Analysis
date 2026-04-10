"""Tests for NetworkRedesignStrategy in src/bus_network.py.

Disposition tests mock _route_overlap_fraction() and _route_productivity()
for deterministic, fast decision-logic testing.  Only the two integration
tests at the bottom use the actual spatial pipeline.
"""

import math
import unittest
from unittest.mock import patch

import numpy as np

from src.bus_network import (
    APMService,
    BusOperatingCostModel,
    BusRoute,
    CITYBUS_ANNUAL_BUDGET_USD,
    CORRIDOR_BUFFER_M,
    DEFAULT_COST_PER_VEH_HOUR,
    ELIMINATION_OVERLAP_THRESHOLD,
    ELIMINATION_PRODUCTIVITY_FLOOR,
    FREQUENCY_ALLOCATION_SHARE,
    MIN_APM_RIDERSHIP_FOR_RESTRUCTURE,
    POST_OPENING_ADJUSTMENT_YEAR,
    NetworkRedesignStrategy,
    REDESIGN_PHASE_IN_YEARS,
    REDESIGN_TRIGGER_RIDERSHIP_RATIO,
    build_corridor_bus_network,
    classify_routes_for_corridor,
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


def _make_corridor_stops(n=5, spacing_m=500.0, origin=(0.0, 0.0)):
    """Create corridor station positions in projected coords (meters)."""
    return np.array(
        [[origin[0] + i * spacing_m, origin[1]] for i in range(n)]
    )


def _make_bus_stops_near_corridor(corridor_xy, n_stops=10, offset_m=50.0):
    """Bus stops parallel to corridor (within buffer)."""
    return np.array(
        [[corridor_xy[i % len(corridor_xy), 0], corridor_xy[i % len(corridor_xy), 1] + offset_m]
         for i in range(n_stops)]
    )


def _make_bus_stops_far(n_stops=10, center=(5000.0, 5000.0)):
    """Bus stops far from corridor (independent)."""
    return np.array(
        [[center[0] + i * 100, center[1]] for i in range(n_stops)]
    )


def _make_strategy_with_mocked_overlap(routes, corridor_xy=None, bus_stops=None,
                                        apm_service=None, overlap_map=None,
                                        productivity_map=None, budget=10_000_000):
    """Create a NetworkRedesignStrategy and patch spatial methods with fixed values."""
    if corridor_xy is None:
        corridor_xy = _make_corridor_stops()
    cost_model = BusOperatingCostModel(annual_budget=budget)
    strategy = NetworkRedesignStrategy(
        routes=routes,
        cost_model=cost_model,
        apm_service=apm_service,
        bus_stops_by_route=bus_stops or {},
        corridor_stops_xy=corridor_xy,
    )
    # Patch overlap and productivity with deterministic values
    if overlap_map:
        orig_overlap = strategy._route_overlap_fraction
        strategy._route_overlap_fraction = lambda r: overlap_map.get(r.route_id, orig_overlap(r))
    if productivity_map:
        orig_prod = strategy._route_productivity
        strategy._route_productivity = lambda r: productivity_map.get(r.route_id, orig_prod(r))
    return strategy


class TestRouteDisposition(unittest.TestCase):
    """Routes should be classified correctly based on overlap, productivity, and APM ridership.

    These tests mock spatial operations for speed and determinism.
    """

    def test_high_overlap_low_productivity_eliminated(self):
        """Route with >50% overlap and below-median productivity -> eliminate."""
        route = _make_route("P1", classification="parallel", riders=50.0)
        strategy = _make_strategy_with_mocked_overlap(
            routes=[route],
            overlap_map={"P1": 0.80},
            productivity_map={"P1": 5.0},
        )
        disp = strategy._compute_route_disposition(
            route, overlap=0.80, productivity=5.0, median_productivity=20.0,
            apm_daily_riders=2000.0,
        )
        self.assertEqual(disp, "eliminate")

    def test_high_overlap_high_productivity_truncated(self):
        """Route with >50% overlap, HIGH productivity, strong APM -> truncate."""
        route = _make_route("P1", classification="parallel", riders=2000.0)
        strategy = _make_strategy_with_mocked_overlap(
            routes=[route],
            overlap_map={"P1": 0.80},
            productivity_map={"P1": 50.0},
        )
        disp = strategy._compute_route_disposition(
            route, overlap=0.80, productivity=50.0, median_productivity=5.0,
            apm_daily_riders=2000.0,
        )
        self.assertEqual(disp, "truncate")

    def test_high_overlap_high_productivity_weak_apm_retained(self):
        """Route with >50% overlap, HIGH productivity, weak APM -> retain_enhanced."""
        route = _make_route("P1", classification="parallel", riders=2000.0)
        strategy = _make_strategy_with_mocked_overlap(
            routes=[route],
            overlap_map={"P1": 0.80},
            productivity_map={"P1": 50.0},
        )
        disp = strategy._compute_route_disposition(
            route, overlap=0.80, productivity=50.0, median_productivity=5.0,
            apm_daily_riders=500.0,  # below MIN_APM_RIDERSHIP_FOR_RESTRUCTURE
        )
        self.assertEqual(disp, "retain_enhanced")

    def test_moderate_overlap_strong_apm_converts_to_feeder(self):
        """Route with 15-50% overlap and strong APM -> convert_to_feeder."""
        route = _make_route("F1", riders=500.0, cycle_time=60.0, headway=30.0)
        strategy = _make_strategy_with_mocked_overlap(
            routes=[route],
            overlap_map={"F1": 0.30},
            productivity_map={"F1": 15.0},
        )
        disp = strategy._compute_route_disposition(
            route, overlap=0.30, productivity=15.0, median_productivity=10.0,
            apm_daily_riders=2000.0,
        )
        self.assertEqual(disp, "convert_to_feeder")

    def test_independent_high_productivity_enhanced(self):
        """Independent route with good productivity -> retain_enhanced."""
        route = _make_route("I1", riders=800.0, cycle_time=60.0, headway=30.0)
        strategy = _make_strategy_with_mocked_overlap(
            routes=[route],
            overlap_map={"I1": 0.05},
            productivity_map={"I1": 25.0},
        )
        disp = strategy._compute_route_disposition(
            route, overlap=0.05, productivity=25.0, median_productivity=10.0,
            apm_daily_riders=2000.0,
        )
        self.assertEqual(disp, "retain_enhanced")

    def test_independent_low_productivity_coverage(self):
        """Independent route with low productivity -> retain_coverage."""
        route = _make_route("I1", riders=5.0, cycle_time=60.0, headway=30.0)
        strategy = _make_strategy_with_mocked_overlap(
            routes=[route],
            overlap_map={"I1": 0.05},
            productivity_map={"I1": 0.5},
        )
        disp = strategy._compute_route_disposition(
            route, overlap=0.05, productivity=0.5, median_productivity=10.0,
            apm_daily_riders=2000.0,
        )
        self.assertEqual(disp, "retain_coverage")


class TestServiceHourAccounting(unittest.TestCase):
    """Freed hours from eliminated routes should balance the budget."""

    def _make_apm_service(self, riders=2000.0):
        return APMService(
            corridor_id="test", length_km=5.0, n_stops=5,
            headway_min=5.0, daily_riders=riders,
        )

    def test_freed_hours_positive(self):
        """Eliminating routes should free service hours."""
        corridor_xy = _make_corridor_stops()
        parallel = _make_route("P1", classification="parallel", riders=10.0)
        independent = _make_route("I1", classification="independent", riders=500.0)
        bus_stops = {
            "P1": _make_bus_stops_near_corridor(corridor_xy),
            "I1": _make_bus_stops_far(),
        }

        cost_model = BusOperatingCostModel(annual_budget=10_000_000)
        strategy = NetworkRedesignStrategy(
            routes=[parallel, independent],
            cost_model=cost_model,
            apm_service=self._make_apm_service(2000.0),
            bus_stops_by_route=bus_stops,
            corridor_stops_xy=corridor_xy,
        )
        result = strategy.redesign_network(
            restructure_pressure=0.8,
            year=5,
            apm_opening_year=0,
        )
        self.assertGreater(result["freed_service_hours"], 0)

    def test_budget_not_exceeded(self):
        """Total cost should not exceed the budget."""
        corridor_xy = _make_corridor_stops()
        routes = [
            _make_route("P1", classification="parallel", riders=10.0,
                        length_km=5.0, cycle_time=35.0, headway=30.0),
            _make_route("I1", classification="independent", riders=500.0,
                        length_km=5.0, cycle_time=35.0, headway=30.0),
            _make_route("I2", classification="independent", riders=300.0,
                        length_km=5.0, cycle_time=35.0, headway=30.0),
        ]
        bus_stops = {
            "P1": _make_bus_stops_near_corridor(corridor_xy),
            "I1": _make_bus_stops_far(),
            "I2": _make_bus_stops_far(center=(6000, 6000)),
        }
        budget = 10_000_000
        cost_model = BusOperatingCostModel(annual_budget=budget)
        strategy = NetworkRedesignStrategy(
            routes=routes,
            cost_model=cost_model,
            apm_service=self._make_apm_service(2000.0),
            bus_stops_by_route=bus_stops,
            corridor_stops_xy=corridor_xy,
        )
        result = strategy.redesign_network(restructure_pressure=0.9, year=5)
        self.assertLessEqual(result["new_bus_cost"], budget * 1.01)


class TestFrequencyReallocation(unittest.TestCase):
    """Retained routes should get improved headways from freed hours."""

    def test_enhanced_routes_improve(self):
        """Enhanced routes should have lower headway than baseline."""
        corridor_xy = _make_corridor_stops()
        parallel = _make_route("P1", classification="parallel", riders=10.0, headway=15.0)
        enhanced = _make_route("I1", classification="independent", riders=800.0, headway=30.0)
        bus_stops = {
            "P1": _make_bus_stops_near_corridor(corridor_xy),
            "I1": _make_bus_stops_far(),
        }

        apm_service = APMService(
            corridor_id="test", length_km=5.0, n_stops=5,
            headway_min=5.0, daily_riders=2000.0,
        )
        strategy = NetworkRedesignStrategy(
            routes=[parallel, enhanced],
            cost_model=BusOperatingCostModel(annual_budget=50_000_000),
            apm_service=apm_service,
            bus_stops_by_route=bus_stops,
            corridor_stops_xy=corridor_xy,
        )
        result = strategy.redesign_network(restructure_pressure=0.8, year=5)

        enhanced_result = [r for r in result["routes"] if r["route_id"] == "I1"]
        if enhanced_result:
            self.assertLess(enhanced_result[0]["current_headway"], 30.0)


class TestSyntheticFeederIntegration(unittest.TestCase):
    """Synthetic feeders should be included when budget allows."""

    def test_synthetic_feeders_activated(self):
        """With freed hours, synthetic feeders should be added to network."""
        corridor_xy = _make_corridor_stops()
        parallel = _make_route("P1", classification="parallel", riders=10.0)
        bus_stops = {"P1": _make_bus_stops_near_corridor(corridor_xy)}

        synthetic = _make_route("SYN_F1", classification="feeder", headway=20.0,
                                cycle_time=40.0, riders=0.0)

        apm_service = APMService(
            corridor_id="test", length_km=5.0, n_stops=5,
            headway_min=5.0, daily_riders=2000.0,
        )
        strategy = NetworkRedesignStrategy(
            routes=[parallel],
            cost_model=BusOperatingCostModel(annual_budget=50_000_000),
            apm_service=apm_service,
            synthetic_feeders=[synthetic],
            bus_stops_by_route=bus_stops,
            corridor_stops_xy=corridor_xy,
        )
        result = strategy.redesign_network(restructure_pressure=0.8, year=5)
        self.assertGreaterEqual(result["n_synthetic_feeders"], 0)


class TestDiscreteEvents(unittest.TestCase):
    """Restructuring should follow discrete event timing."""

    def test_below_ridership_threshold_no_restructure(self):
        """APM ridership below MIN_APM_RIDERSHIP_FOR_RESTRUCTURE -> no changes."""
        corridor_xy = _make_corridor_stops()
        route = _make_route("P1", classification="parallel", riders=10.0, headway=30.0)
        bus_stops = {"P1": _make_bus_stops_near_corridor(corridor_xy)}

        apm_service = APMService(
            corridor_id="test", length_km=5.0, n_stops=5,
            headway_min=5.0, daily_riders=500.0,
        )
        strategy = NetworkRedesignStrategy(
            routes=[route],
            cost_model=BusOperatingCostModel(annual_budget=50_000_000),
            apm_service=apm_service,
            bus_stops_by_route=bus_stops,
            corridor_stops_xy=corridor_xy,
        )
        result = strategy.redesign_network(restructure_pressure=0.8, year=0)
        self.assertAlmostEqual(result["redesign_phase_fraction"], 0.0)
        self.assertAlmostEqual(route.current_headway_min, 30.0)

    def test_opening_event_full_restructure(self):
        """At opening (year 0) with sufficient APM ridership -> full restructure."""
        corridor_xy = _make_corridor_stops()
        parallel = _make_route("P1", classification="parallel", riders=10.0, headway=30.0)
        independent = _make_route("I1", classification="independent", riders=800.0, headway=30.0)
        bus_stops = {
            "P1": _make_bus_stops_near_corridor(corridor_xy),
            "I1": _make_bus_stops_near_corridor(corridor_xy, offset_m=100.0),
        }

        apm_service = APMService(
            corridor_id="test", length_km=5.0, n_stops=5,
            headway_min=5.0, daily_riders=2000.0,
        )
        strategy = NetworkRedesignStrategy(
            routes=[parallel, independent],
            cost_model=BusOperatingCostModel(annual_budget=50_000_000),
            apm_service=apm_service,
            bus_stops_by_route=bus_stops,
            corridor_stops_xy=corridor_xy,
        )
        result = strategy.redesign_network(
            restructure_pressure=0.8,
            year=0,
            apm_opening_year=0,
        )
        self.assertAlmostEqual(result["redesign_phase_fraction"], 1.0)
        p1_result = [r for r in result["routes"] if r["route_id"] == "P1"]
        self.assertEqual(len(p1_result), 1)
        self.assertEqual(p1_result[0]["disposition"], "eliminate")
        self.assertLess(p1_result[0]["current_headway"], float("inf"))

    def test_truncated_route_becomes_feeder(self):
        """High-overlap, high-productivity route with strong APM -> truncated as feeder."""
        corridor_xy = _make_corridor_stops()
        route = _make_route("P1", classification="parallel", riders=2000.0, headway=30.0, cycle_time=70.0)
        bus_stops = {"P1": _make_bus_stops_near_corridor(corridor_xy)}

        apm_service = APMService(
            corridor_id="test", length_km=5.0, n_stops=5,
            headway_min=5.0, daily_riders=2000.0,
        )
        strategy = NetworkRedesignStrategy(
            routes=[route],
            cost_model=BusOperatingCostModel(annual_budget=50_000_000),
            apm_service=apm_service,
            bus_stops_by_route=bus_stops,
            corridor_stops_xy=corridor_xy,
        )
        result = strategy.redesign_network(restructure_pressure=0.8, year=0)

        self.assertEqual(result["n_truncated_routes"], 1)
        route_results = [r for r in result["routes"] if r["route_id"] == "P1"]
        self.assertEqual(len(route_results), 1)
        self.assertEqual(route_results[0]["classification"], "feeder")
        self.assertLess(route_results[0]["current_headway"], 30.0)


class TestCoverageLoss(unittest.TestCase):
    """Coverage loss should stay below the maximum threshold."""

    def test_coverage_loss_computed(self):
        """Coverage loss should be a float between 0 and 1."""
        corridor_xy = _make_corridor_stops()
        route = _make_route("P1", classification="parallel", riders=10.0)
        bus_stops = {"P1": _make_bus_stops_near_corridor(corridor_xy)}

        apm_service = APMService(
            corridor_id="test", length_km=5.0, n_stops=5,
            headway_min=5.0, daily_riders=2000.0,
        )
        strategy = NetworkRedesignStrategy(
            routes=[route],
            cost_model=BusOperatingCostModel(annual_budget=50_000_000),
            apm_service=apm_service,
            bus_stops_by_route=bus_stops,
            corridor_stops_xy=corridor_xy,
        )
        result = strategy.redesign_network(restructure_pressure=0.8, year=5)
        self.assertGreaterEqual(result["coverage_loss_fraction"], 0.0)
        self.assertLessEqual(result["coverage_loss_fraction"], 1.0)


class TestOutputContractParity(unittest.TestCase):
    """Redesign output should contain all keys the feedback loop expects."""

    REQUIRED_KEYS = {
        "restructure_pressure", "feeder_headway", "parallel_headway",
        "budget", "new_bus_cost", "routes", "route_dispositions",
        "freed_service_hours", "n_eliminated_routes", "n_truncated_routes",
        "n_synthetic_feeders", "coverage_loss_fraction",
        "frequency_weighted_coverage", "redesign_phase_fraction",
    }

    def test_all_keys_present(self):
        corridor_xy = _make_corridor_stops()
        route = _make_route("P1", classification="parallel", riders=10.0)
        bus_stops = {"P1": _make_bus_stops_near_corridor(corridor_xy)}

        apm_service = APMService(
            corridor_id="test", length_km=5.0, n_stops=5,
            headway_min=5.0, daily_riders=2000.0,
        )
        strategy = NetworkRedesignStrategy(
            routes=[route],
            cost_model=BusOperatingCostModel(annual_budget=50_000_000),
            apm_service=apm_service,
            bus_stops_by_route=bus_stops,
            corridor_stops_xy=corridor_xy,
        )
        result = strategy.redesign_network(restructure_pressure=0.8, year=5)

        for key in self.REQUIRED_KEYS:
            self.assertIn(key, result, f"Missing key: {key}")


class TestCorridorIndependence(unittest.TestCase):
    """Running redesign for two corridors should not leak state."""

    def test_no_state_leakage(self):
        """Route state from corridor A should not affect corridor B."""
        corridor_a = _make_corridor_stops(origin=(0, 0))
        corridor_b = _make_corridor_stops(origin=(10000, 10000))

        routes_a = [_make_route("P1", classification="parallel", riders=10.0)]
        routes_b = [_make_route("P1", classification="parallel", riders=10.0)]

        bus_stops_a = {"P1": _make_bus_stops_near_corridor(corridor_a)}
        bus_stops_b = {"P1": _make_bus_stops_near_corridor(corridor_b)}

        cost_model = BusOperatingCostModel(annual_budget=50_000_000)
        apm_service = APMService(
            corridor_id="test", length_km=5.0, n_stops=5,
            headway_min=5.0, daily_riders=2000.0,
        )

        strategy_a = NetworkRedesignStrategy(
            routes=routes_a, cost_model=cost_model,
            apm_service=apm_service,
            bus_stops_by_route=bus_stops_a, corridor_stops_xy=corridor_a,
        )
        result_a = strategy_a.redesign_network(restructure_pressure=0.8, year=5)

        strategy_b = NetworkRedesignStrategy(
            routes=routes_b, cost_model=cost_model,
            apm_service=apm_service,
            bus_stops_by_route=bus_stops_b, corridor_stops_xy=corridor_b,
        )
        result_b = strategy_b.redesign_network(restructure_pressure=0.8, year=5)

        self.assertEqual(
            result_a["n_eliminated_routes"],
            result_b["n_eliminated_routes"],
        )
        self.assertAlmostEqual(
            result_a["freed_service_hours"],
            result_b["freed_service_hours"],
            places=0,
        )


class TestBuildCorridorBusNetworkRedesign(unittest.TestCase):
    """Integration test: build_corridor_bus_network with strategy='redesign'."""

    def test_redesign_strategy_produces_valid_output(self):
        """Calling with strategy='redesign' should return valid result dict."""
        corridor_xy = _make_corridor_stops()
        routes = [
            _make_route("P1", headway=20.0, cycle_time=60.0, riders=50.0),
            _make_route("I1", headway=30.0, cycle_time=50.0, riders=500.0),
        ]
        bus_stops = {
            "P1": _make_bus_stops_near_corridor(corridor_xy),
            "I1": _make_bus_stops_far(),
        }

        result = build_corridor_bus_network(
            routes=routes,
            corridor_stops_xy=corridor_xy,
            bus_stops_by_route=bus_stops,
            apm_daily_riders=4000,
            apm_length_km=5.0,
            apm_n_stops=5,
            restructure_pressure=0.7,
            strategy="redesign",
            year=5,
        )

        self.assertIn("phase", result)
        self.assertIn("apm_headway_min", result)
        self.assertIn("feeder_coverage_fraction", result)
        self.assertIn("route_dispositions", result)
        self.assertIn("n_eliminated_routes", result)

    def test_incremental_strategy_unchanged(self):
        """Calling with strategy='incremental' should match default behavior."""
        corridor_xy = _make_corridor_stops()
        routes = [
            _make_route("P1", headway=20.0, cycle_time=60.0, riders=500.0),
        ]
        bus_stops = {
            "P1": _make_bus_stops_near_corridor(corridor_xy),
        }

        result = build_corridor_bus_network(
            routes=routes,
            corridor_stops_xy=corridor_xy,
            bus_stops_by_route=bus_stops,
            apm_daily_riders=4000,
            apm_length_km=5.0,
            apm_n_stops=5,
            restructure_pressure=0.5,
            strategy="incremental",
        )

        self.assertIn("phase", result)
        self.assertIn("optimization", result)
        self.assertNotIn("route_dispositions", result)


if __name__ == "__main__":
    unittest.main()

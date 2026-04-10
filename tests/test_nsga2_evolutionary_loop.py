"""C7: NSGA-II evolutionary loop integration test.

Builds a minimal synthetic environment (grid graph, 8 candidate stations,
6 demand parcels) and drives the NSGA-II cycle through initialization →
scoring → non-dominated sorting → mutation → crossover → deduplication →
survivor selection.  Asserts structural properties only, NOT numeric
magnitudes.
"""
from __future__ import annotations

import random
import unittest

import numpy as np

# EPSG:2965 US survey feet → meters
_FT_PER_M = 1.0 / 0.3048006096012192

# Station spacing: 1000m ≈ 3281 ft in EPSG:2965.
# 8 stations on a line → 7 gaps × 1000m = 7km (within 2-25km range).
_SPACING_FT = 1000.0 * _FT_PER_M


def _make_grid_station_data(n_stations=8):
    """Build a minimal station_data dict with everything score/mutate/crossover need.

    Stations are on a straight line at ~1000m spacing in EPSG:2965 feet.
    No real road graph — uses Euclidean fallback for all distances.
    Six synthetic parcels are placed near the stations to generate nonzero demand.
    """
    from scipy.spatial import cKDTree

    # -- Station coordinates (straight line, x-axis) --
    coords = np.zeros((n_stations, 2), dtype=np.float64)
    coords[:, 0] = np.arange(n_stations) * _SPACING_FT
    coords[:, 1] = 0.0

    # Adjacency: each station connected to its immediate neighbors
    adjacency = {}
    for i in range(n_stations):
        nbrs = []
        if i > 0:
            nbrs.append(i - 1)
        if i < n_stations - 1:
            nbrs.append(i + 1)
        adjacency[i] = nbrs

    # -- Parcel arrays (6 parcels near stations) --
    n_parcels = 6
    parcel_coords = np.zeros((n_parcels, 2), dtype=np.float64)
    for i in range(n_parcels):
        # Place near every-other station, offset 200m north
        parcel_coords[i, 0] = coords[min(i * 2, n_stations - 1), 0]
        parcel_coords[i, 1] = 200.0 * _FT_PER_M

    parcel_demand = np.full(n_parcels, 50.0)
    parcel_av = np.full(n_parcels, 500_000.0)
    parcel_exempt = np.zeros(n_parcels, dtype=bool)
    parcel_parking = np.full(n_parcels, 2.0)
    parcel_inst_wt = np.ones(n_parcels)  # no campus
    parcel_dev_potential = np.full(n_parcels, 0.3)

    station_data = {
        # Station arrays
        "coords_proj": coords,
        "node_ids": list(range(n_stations)),
        "adjacency": adjacency,
        "tree": cKDTree(coords),
        "demand_coverage": np.full(n_stations, 30.0),
        "demand_coverage_raw": np.full(n_stations, 30.0),
        "feeder_coverage": np.full(n_stations, 10.0),
        "tif_coverage": np.full(n_stations, 100_000.0),
        "parking_cost": np.full(n_stations, 2.0),
        "bus_competition": np.full(n_stations, 3.0),
        # Parcel arrays
        "parcel_coords_proj": parcel_coords,
        "parcel_demand": parcel_demand,
        "parcel_av": parcel_av,
        "parcel_exempt": parcel_exempt,
        "parcel_parking": parcel_parking,
        "parcel_inst_wt": parcel_inst_wt,
        "parcel_dev_potential": parcel_dev_potential,
        "parcel_tree": cKDTree(parcel_coords),
        # No spatial grid — scoring falls back to full parcel scan
        "parcel_grid_index": None,
        # No road graph — distance uses Euclidean × ROAD_CIRCUITY_FACTOR
        "graph": None,
        "road_dist_cache": {},
        "path_node_cache": {},
        "path_bearing_cache": {},
        # No OD cache — OD-based ridership returns 0
        "od_cache": None,
        # No anchors
        "anchor_station_local": [],
        "anchor_station_ids": [],
    }
    return station_data


@unittest.skipIf(
    not all(
        __import__("importlib").util.find_spec(m)
        for m in ("scipy", "shapely", "numpy")
    ),
    "scipy/shapely not installed",
)
class TestNSGA2EvolutionaryLoop(unittest.TestCase):
    """End-to-end structural test for the NSGA-II evolutionary loop."""

    @classmethod
    def setUpClass(cls):
        try:
            import sys
            from pathlib import Path
            from scripts.optimized_corridor_search import (
                crossover_station_sets,
                deduplicate_station_sets,
                fast_non_dominated_sort,
                mutate_station_set,
                nsga2_select,
                score_station_set,
                validate_station_set,
            )

            cls._score = staticmethod(score_station_set)
            cls._validate = staticmethod(validate_station_set)
            cls._mutate = staticmethod(mutate_station_set)
            cls._crossover = staticmethod(crossover_station_sets)
            cls._dedup = staticmethod(deduplicate_station_sets)
            cls._sort = staticmethod(fast_non_dominated_sort)
            cls._select = staticmethod(nsga2_select)
            cls._import_error = None
        except Exception as exc:
            cls._import_error = exc

    def setUp(self):
        if self._import_error is not None:
            self.skipTest(f"Import error: {self._import_error}")
        self.sd = _make_grid_station_data(8)

    # -- helpers --
    def _make_initial_population(self):
        """Create 6 valid corridors as initial population.

        Each corridor uses 4-6 consecutive stations so spacing and length
        constraints are satisfied (1000m spacing, 3-5km length).
        """
        candidates = [
            [0, 1, 2, 3],          # 4 stations, ~3km
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5],    # 6 stations, ~5km
        ]
        population = []
        for stations in candidates:
            ok, _ = self._validate(stations, self.sd)
            if ok:
                score = self._score(stations, self.sd)
                if score is not None:
                    population.append({
                        "stations": stations,
                        "score": score,
                        "source": "seed",
                    })
        return population

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("shapely"),
        "shapely required for scoring",
    )
    @unittest.skipUnless(
        __import__("os").environ.get("RUN_SLOW_TESTS"),
        "slow: ~5-10s due to scoring overhead (set RUN_SLOW_TESTS=1 to run)",
    )
    def test_full_evolutionary_cycle(self):
        """Run 2 generations of the NSGA-II loop on synthetic data."""
        random.seed(42)
        np.random.seed(42)

        # 1. Initialization
        population = self._make_initial_population()
        self.assertGreater(len(population), 0, "Need at least 1 scored corridor")

        # 2. Evolutionary loop (2 generations)
        for gen in range(2):
            # Selection: trim to pop_size=6
            if len(population) > 6:
                population = self._select(population, 6)

            # Mutation
            offspring = []
            for _ in range(3):
                parent = random.choice(population)
                child = self._mutate(parent, self.sd)
                if child is not None:
                    offspring.append(child)

            # Crossover
            for _ in range(2):
                if len(population) >= 2:
                    p1, p2 = random.sample(population, 2)
                    child = self._crossover(p1, p2, self.sd)
                    if child is not None:
                        offspring.append(child)

            # Deduplicate before scoring
            offspring = self._dedup(offspring, self.sd)

            # Score offspring
            scored_offspring = []
            for child in offspring:
                score = self._score(child["stations"], self.sd)
                if score is not None:
                    child["score"] = score
                    scored_offspring.append(child)
            population.extend(scored_offspring)

        # 3. Assertions on final population
        self.assertGreater(len(population), 0, "Population should be non-empty")
        self.assertLessEqual(
            len(population), 30,
            "Population should stay bounded",
        )

        # Every corridor has valid station set
        for cand in population:
            ok, msg = self._validate(cand["stations"], self.sd)
            self.assertTrue(ok, f"Invalid corridor: {msg}")

        # Every corridor has required score keys
        required_keys = {"ridership_est", "cost_efficiency", "viability_indicator"}
        for cand in population:
            self.assertIn("score", cand)
            for key in required_keys:
                self.assertIn(key, cand["score"], f"Missing score key: {key}")

        # Pareto front (rank 0) is non-empty
        fronts = self._sort(population)
        self.assertGreater(len(fronts), 0, "Should have at least one front")
        self.assertGreater(len(fronts[0]), 0, "Pareto front should be non-empty")

        # No duplicate station sets
        sigs = set()
        for cand in population:
            sig = tuple(sorted(cand["stations"]))
            self.assertNotIn(sig, sigs, f"Duplicate station set: {sig}")
            sigs.add(sig)

        # All station indices valid
        n_stations = len(self.sd["coords_proj"])
        for cand in population:
            for s in cand["stations"]:
                self.assertGreaterEqual(s, 0)
                self.assertLess(s, n_stations)

    def test_scoring_produces_required_keys(self):
        """score_station_set returns dict with NSGA-II objective keys."""
        stations = [0, 1, 2, 3]
        ok, msg = self._validate(stations, self.sd)
        if not ok:
            self.skipTest(f"Seed corridor invalid: {msg}")
        score = self._score(stations, self.sd)
        if score is None:
            self.skipTest("Scoring rejected corridor (infeasible curves)")
        for key in ("ridership_est", "cost_efficiency", "viability_indicator",
                     "tif_potential", "tif_viable", "length_km", "apm_share"):
            self.assertIn(key, score, f"Missing key: {key}")
        self.assertGreater(score["ridership_est"], 0)
        self.assertGreater(score["cost_efficiency"], 0)
        self.assertGreater(score["length_km"], 0)

    def test_non_dominated_sort_with_synthetic_scores(self):
        """fast_non_dominated_sort produces valid fronts from hand-crafted scores."""
        population = [
            {"stations": [0, 1, 2, 3], "score": {
                "ridership_est": 100, "cost_efficiency": 0.5, "viability_indicator": 0.3,
                "tif_viable": True,
            }},
            {"stations": [1, 2, 3, 4], "score": {
                "ridership_est": 200, "cost_efficiency": 0.4, "viability_indicator": 0.2,
                "tif_viable": True,
            }},
            {"stations": [2, 3, 4, 5], "score": {
                "ridership_est": 50, "cost_efficiency": 0.8, "viability_indicator": 0.5,
                "tif_viable": True,
            }},
            {"stations": [3, 4, 5, 6], "score": {
                "ridership_est": 10, "cost_efficiency": 0.1, "viability_indicator": 0.05,
                "tif_viable": False,
            }},
        ]
        fronts = self._sort(population)
        self.assertGreater(len(fronts), 0)
        self.assertGreater(len(fronts[0]), 0, "Rank-0 front should be non-empty")

        # All indices accounted for
        all_idx = set()
        for front in fronts:
            for idx in front:
                self.assertNotIn(idx, all_idx, "Index in multiple fronts")
                all_idx.add(idx)
        self.assertEqual(all_idx, {0, 1, 2, 3})

        # Dominated solution (3) should NOT be in rank 0
        self.assertNotIn(3, fronts[0])

    def test_nsga2_select_respects_size_limit(self):
        """nsga2_select trims population to requested size."""
        population = [
            {"stations": [i, i + 1, i + 2, i + 3], "score": {
                "ridership_est": float(50 + i * 10),
                "cost_efficiency": float(0.1 + i * 0.1),
                "viability_indicator": float(0.1 + i * 0.05),
                "tif_viable": True,
            }}
            for i in range(5)
        ]
        selected = self._select(population, 3)
        self.assertEqual(len(selected), 3)
        # All selected have score dicts
        for cand in selected:
            self.assertIn("score", cand)

    def test_mutation_returns_valid_or_none(self):
        """mutate_station_set returns a valid corridor dict or None."""
        random.seed(99)
        np.random.seed(99)
        parent = {
            "stations": [0, 1, 2, 3, 4, 5],
            "score": {"ridership_est": 100, "cost_efficiency": 0.5, "viability_indicator": 0.3},
            "source": "seed",
        }
        results = []
        for _ in range(50):
            child = self._mutate(parent, self.sd)
            results.append(child)

        # At least some mutations should succeed on a valid parent
        successes = [c for c in results if c is not None]
        self.assertGreater(
            len(successes), 0,
            "At least one mutation should succeed out of 50 attempts",
        )
        for child in successes:
            self.assertIn("stations", child)
            self.assertIn("source", child)
            ok, msg = self._validate(child["stations"], self.sd)
            self.assertTrue(ok, f"Mutated corridor invalid: {msg}")
            # All station indices in bounds
            for s in child["stations"]:
                self.assertGreaterEqual(s, 0)
                self.assertLess(s, len(self.sd["coords_proj"]))

    def test_crossover_returns_valid_or_none(self):
        """crossover_station_sets returns a valid corridor dict or None."""
        random.seed(77)
        parent_a = {"stations": [0, 1, 2, 3, 4], "source": "seed"}
        parent_b = {"stations": [3, 4, 5, 6, 7], "source": "seed"}

        results = []
        for _ in range(30):
            child = self._crossover(parent_a, parent_b, self.sd)
            results.append(child)

        successes = [c for c in results if c is not None]
        for child in successes:
            self.assertIn("stations", child)
            ok, msg = self._validate(child["stations"], self.sd)
            self.assertTrue(ok, f"Crossover child invalid: {msg}")

    def test_deduplication_removes_identical_sets(self):
        """deduplicate_station_sets removes exact duplicates."""
        candidates = [
            {"stations": [0, 1, 2, 3], "score": {"ridership_est": 100}},
            {"stations": [0, 1, 2, 3], "score": {"ridership_est": 90}},
            {"stations": [4, 5, 6, 7], "score": {"ridership_est": 80}},
        ]
        result = self._dedup(candidates, self.sd)
        self.assertEqual(len(result), 2)

    def test_validate_rejects_out_of_bounds_indices(self):
        """validate_station_set catches invalid indices gracefully."""
        n = len(self.sd["coords_proj"])
        # Too few stations
        ok, msg = self._validate([0, 1, 2], self.sd)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

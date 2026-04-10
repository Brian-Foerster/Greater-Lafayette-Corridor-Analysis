"""Tests for src/relocation_model.py — household relocation MNL."""

import unittest

import numpy as np

from src.relocation_model import (
    AVG_HOUSEHOLD_SIZE,
    METRO_POPULATION,
    TARGET_VAC_RES,
    HouseholdRelocationModel,
    RelocationConfig,
)


class TestAnnualMovers(unittest.TestCase):
    def test_annual_movers_default(self):
        """Default config: (89844 - 14700) * 0.05 = 3757."""
        model = HouseholdRelocationModel()
        expected = int((int(METRO_POPULATION / AVG_HOUSEHOLD_SIZE) - 14_700) * 0.05)
        self.assertEqual(model.annual_movers(), expected)
        self.assertEqual(model.annual_movers(), 3757)

    def test_annual_movers_custom(self):
        config = RelocationConfig(metro_households=100_000, student_households=20_000, annual_mover_rate=0.10)
        model = HouseholdRelocationModel(config)
        self.assertEqual(model.annual_movers(), 8000)

    def test_annual_movers_no_negative(self):
        config = RelocationConfig(metro_households=1000, student_households=5000)
        model = HouseholdRelocationModel(config)
        self.assertEqual(model.annual_movers(), 0)

    def test_metro_households_consistent(self):
        """metro_households must derive from METRO_POPULATION / AVG_HOUSEHOLD_SIZE."""
        config = RelocationConfig()
        expected_hh = int(METRO_POPULATION / AVG_HOUSEHOLD_SIZE)
        self.assertEqual(int(config.metro_households), expected_hh)


class TestChoiceSet(unittest.TestCase):
    def test_choice_set_size_bounded(self):
        """Choice set should have at most near + region + random unique entries."""
        model = HouseholdRelocationModel()
        rng = np.random.default_rng(42)
        n_parcels = 10_000
        near_800_idx = np.arange(100)
        developable_mask = np.ones(n_parcels, dtype=bool)

        cs, weights = model.sample_choice_set(near_800_idx, developable_mask, n_parcels, rng)
        self.assertLessEqual(len(cs), model.choice_set_size)
        self.assertEqual(len(cs), len(np.unique(cs)))
        self.assertEqual(len(cs), len(weights))

    def test_choice_set_returns_weights(self):
        """Weights should be positive expansion factors."""
        model = HouseholdRelocationModel()
        rng = np.random.default_rng(42)
        n_parcels = 10_000
        near_800_idx = np.arange(100)
        developable_mask = np.ones(n_parcels, dtype=bool)

        cs, weights = model.sample_choice_set(near_800_idx, developable_mask, n_parcels, rng)
        self.assertTrue(np.all(weights > 0))
        # Near-corridor weight should be smaller than random weight
        # (smaller stratum population / same sample size)
        near_set = set(near_800_idx.tolist())
        near_mask = np.array([idx in near_set for idx in cs])
        non_near_mask = ~near_mask
        if near_mask.any() and non_near_mask.any():
            self.assertLess(weights[near_mask].mean(), weights[non_near_mask].mean())

    def test_choice_set_empty_near(self):
        """With no near-corridor parcels, should still return regional + random."""
        model = HouseholdRelocationModel()
        rng = np.random.default_rng(42)
        n_parcels = 1000
        near_800_idx = np.array([], dtype=int)
        developable_mask = np.ones(n_parcels, dtype=bool)

        cs, weights = model.sample_choice_set(near_800_idx, developable_mask, n_parcels, rng)
        self.assertGreater(len(cs), 0)
        self.assertEqual(len(cs), len(weights))

    def test_choice_set_no_developable(self):
        """With nothing developable, near/region are empty but random still works."""
        model = HouseholdRelocationModel()
        rng = np.random.default_rng(42)
        n_parcels = 100
        near_800_idx = np.arange(10)
        developable_mask = np.zeros(n_parcels, dtype=bool)

        cs, weights = model.sample_choice_set(near_800_idx, developable_mask, n_parcels, rng)
        self.assertGreater(len(cs), 0)
        self.assertEqual(len(cs), len(weights))


class TestLocationProbabilities(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        """For any inputs, probabilities should sum to 1.0."""
        model = HouseholdRelocationModel()
        rents = np.array([14.0, 20.0, 26.0, 9.0])
        commute = np.array([15.0, 25.0, 10.0, 40.0])
        access = np.array([5.0, 0.0, 10.0, 0.0])

        for seg in ["SE01", "SE02", "SE03"]:
            probs = model.location_probabilities(rents, commute, access, seg)
            self.assertAlmostEqual(probs.sum(), 1.0, places=10)
            self.assertTrue(np.all(probs >= 0))

    def test_probabilities_sum_to_one_with_weights(self):
        """With importance weights, probabilities should still sum to 1.0."""
        model = HouseholdRelocationModel()
        rents = np.array([14.0, 20.0, 26.0, 9.0])
        commute = np.array([15.0, 25.0, 10.0, 40.0])
        access = np.array([5.0, 0.0, 10.0, 0.0])
        weights = np.array([20.0, 500.0, 500.0, 4000.0])

        for seg in ["SE01", "SE02", "SE03"]:
            probs = model.location_probabilities(rents, commute, access, seg, weights=weights)
            self.assertAlmostEqual(probs.sum(), 1.0, places=10)
            self.assertTrue(np.all(probs >= 0))

    def test_weights_reduce_oversampled_stratum(self):
        """An oversampled alternative with low weight should get lower probability."""
        model = HouseholdRelocationModel()
        # Two identical parcels, but one has a much higher weight
        rents = np.array([14.0, 14.0])
        commute = np.array([20.0, 20.0])
        access = np.array([0.0, 0.0])

        # Without weights: equal probability
        probs_unweighted = model.location_probabilities(rents, commute, access, "SE02")
        self.assertAlmostEqual(probs_unweighted[0], probs_unweighted[1], places=10)

        # With weights: alternative 1 represents more of the population
        weights = np.array([10.0, 1000.0])
        probs_weighted = model.location_probabilities(rents, commute, access, "SE02", weights=weights)
        self.assertGreater(probs_weighted[1], probs_weighted[0])

    def test_high_accessibility_attracts_more(self):
        """Parcel with high transit accessibility should get higher probability."""
        model = HouseholdRelocationModel()
        rents = np.array([14.0, 14.0])
        commute = np.array([20.0, 20.0])
        access = np.array([10.0, 0.0])

        probs = model.location_probabilities(rents, commute, access, "SE02")
        self.assertGreater(probs[0], probs[1])

    def test_lower_rent_attracts_more(self):
        """Lower rent should attract more movers (all else equal)."""
        model = HouseholdRelocationModel()
        rents = np.array([10.0, 30.0])
        commute = np.array([20.0, 20.0])
        access = np.array([0.0, 0.0])

        probs = model.location_probabilities(rents, commute, access, "SE01")
        self.assertGreater(probs[0], probs[1])

    def test_shorter_commute_attracts_more(self):
        """Shorter commute should attract more movers (all else equal)."""
        model = HouseholdRelocationModel()
        rents = np.array([14.0, 14.0])
        commute = np.array([10.0, 40.0])
        access = np.array([0.0, 0.0])

        probs = model.location_probabilities(rents, commute, access, "SE02")
        self.assertGreater(probs[0], probs[1])

    def test_extreme_inputs_stable(self):
        """Very large/small inputs should not produce NaN or Inf."""
        model = HouseholdRelocationModel()
        rents = np.array([0.01, 1000.0])
        commute = np.array([0.0, 200.0])
        access = np.array([0.0, 1e6])

        probs = model.location_probabilities(rents, commute, access, "SE03")
        self.assertAlmostEqual(probs.sum(), 1.0, places=10)
        self.assertFalse(np.any(np.isnan(probs)))
        self.assertFalse(np.any(np.isinf(probs)))


class TestIncomeSegments(unittest.TestCase):
    def test_shares_sum_to_one(self):
        """Income segment shares should sum to 1.0."""
        config = RelocationConfig()
        total = sum(seg["share"] for seg in config.income_segments.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_low_income_more_price_sensitive(self):
        """SE01 (low income) should have higher beta_price_mult than SE03."""
        config = RelocationConfig()
        self.assertGreater(
            config.income_segments["SE01"]["beta_price_mult"],
            config.income_segments["SE03"]["beta_price_mult"],
        )


class TestVacancyTarget(unittest.TestCase):
    def test_vacancy_target_consistent(self):
        """TARGET_VAC_RES should match demand_driven_development.py."""
        from src.demand_driven_development import MarketParams
        self.assertAlmostEqual(TARGET_VAC_RES, MarketParams().target_vacancy_residential, places=4)

    def test_vacancy_target_reasonable(self):
        self.assertGreater(TARGET_VAC_RES, 0.03)
        self.assertLess(TARGET_VAC_RES, 0.15)


class TestAllocateMovers(unittest.TestCase):
    def test_allocate_returns_reasonable_range(self):
        """Corridor HH should be between 0 and annual_movers."""
        model = HouseholdRelocationModel()
        n_parcels = 1000
        near_800_idx = np.arange(50)
        developable_mask = np.ones(n_parcels, dtype=bool)
        rents = np.full(n_parcels, 14.0)
        commute = np.full(n_parcels, 20.0)
        access = np.zeros(n_parcels)
        access[near_800_idx] = 5.0  # corridor parcels have APM access

        hh = model.allocate_movers_to_corridor(
            near_800_idx, developable_mask, n_parcels,
            rents, commute, access, year=0,
        )
        self.assertGreater(hh, 0)
        self.assertLessEqual(hh, model.annual_movers())

    def test_allocate_corridor_share_reasonable(self):
        """With importance weights, corridor share should reflect utility, not sampling."""
        model = HouseholdRelocationModel()
        n_parcels = 10_000
        near_800_idx = np.arange(100)  # 1% of parcels
        developable_mask = np.ones(n_parcels, dtype=bool)
        rents = np.full(n_parcels, 14.0)
        commute = np.full(n_parcels, 20.0)
        access = np.zeros(n_parcels)
        access[near_800_idx] = 5.0

        hh = model.allocate_movers_to_corridor(
            near_800_idx, developable_mask, n_parcels,
            rents, commute, access, year=0,
        )
        corridor_share = hh / model.annual_movers()
        # With only 1% of parcels near corridor and moderate transit premium,
        # share should be well under 50% (old bug gave ~33%+)
        self.assertLess(corridor_share, 0.20)
        # Should be positive (transit premium attracts some movers)
        self.assertGreater(corridor_share, 0.0)

    def test_allocate_no_near_parcels(self):
        """With no near-corridor parcels, should return 0."""
        model = HouseholdRelocationModel()
        hh = model.allocate_movers_to_corridor(
            near_800_idx=np.array([], dtype=int),
            developable_mask=np.ones(100, dtype=bool),
            n_parcels=100,
            rents=np.full(100, 14.0),
            commute_times=np.full(100, 20.0),
            accessibility=np.zeros(100),
            year=0,
        )
        self.assertEqual(hh, 0.0)

    def test_allocate_deterministic(self):
        """Same inputs + same year should produce identical output."""
        model = HouseholdRelocationModel()
        n_parcels = 500
        near = np.arange(30)
        dev = np.ones(n_parcels, dtype=bool)
        rents = np.random.default_rng(0).uniform(10, 30, n_parcels)
        commute = np.random.default_rng(1).uniform(5, 45, n_parcels)
        access = np.zeros(n_parcels)
        access[near] = 8.0

        hh1 = model.allocate_movers_to_corridor(near, dev, n_parcels, rents, commute, access, year=5)
        hh2 = model.allocate_movers_to_corridor(near, dev, n_parcels, rents, commute, access, year=5)
        self.assertEqual(hh1, hh2)


if __name__ == "__main__":
    unittest.main()

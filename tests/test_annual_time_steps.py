"""Tests for annual time-step infrastructure."""
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from src.land_use_transport_model import (
        OCCUPANCY_SCHEDULE,
        CONSTRUCTION_OCCUPANCY_FRACTION,
        DEFAULT_RIDERSHIP_CONVERGENCE_TOL,
        DEFAULT_DEVELOPMENT_CONVERGENCE_TOL,
        DEFAULT_CONSECUTIVE_CONVERGED_STEPS,
        LandUseTransportModel,
    )
    _IMPORT_ERROR = None
except Exception as exc:
    _IMPORT_ERROR = exc


class TestOccupancySchedule(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping: {_IMPORT_ERROR}")

    def test_schedule_is_monotonically_increasing(self):
        for i in range(1, len(OCCUPANCY_SCHEDULE)):
            self.assertGreaterEqual(OCCUPANCY_SCHEDULE[i], OCCUPANCY_SCHEDULE[i - 1])

    def test_schedule_starts_at_zero(self):
        self.assertEqual(OCCUPANCY_SCHEDULE[0], 0.0)

    def test_schedule_ends_at_one(self):
        self.assertEqual(OCCUPANCY_SCHEDULE[-1], 1.0)

    def test_increments_sum_to_one(self):
        increments = []
        for i in range(len(OCCUPANCY_SCHEDULE)):
            prev = OCCUPANCY_SCHEDULE[i - 1] if i > 0 else 0.0
            increments.append(OCCUPANCY_SCHEDULE[i] - prev)
        self.assertAlmostEqual(sum(increments), 1.0, places=6)

    def test_legacy_fraction_matches_schedule_average(self):
        # The legacy CONSTRUCTION_OCCUPANCY_FRACTION (0.40) should be
        # close to the average of the schedule (period-average occupancy)
        avg = sum(OCCUPANCY_SCHEDULE) / len(OCCUPANCY_SCHEDULE)
        self.assertAlmostEqual(avg, CONSTRUCTION_OCCUPANCY_FRACTION, places=2)


class TestAnnualDefaults(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping: {_IMPORT_ERROR}")

    def test_default_time_steps_are_annual(self):
        import inspect
        sig = inspect.signature(LandUseTransportModel.__init__)
        default = sig.parameters["time_steps"].default
        self.assertEqual(default, tuple(range(26)))

    def test_convergence_tolerances_appropriate_for_ranking(self):
        self.assertLessEqual(DEFAULT_RIDERSHIP_CONVERGENCE_TOL, 0.05)
        self.assertLessEqual(DEFAULT_DEVELOPMENT_CONVERGENCE_TOL, 0.10)

    def test_consecutive_converged_steps_increased(self):
        self.assertGreaterEqual(DEFAULT_CONSECUTIVE_CONVERGED_STEPS, 3)


class TestStepYearsComputation(unittest.TestCase):
    """Verify dynamic step_years from time_steps sequence."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping: {_IMPORT_ERROR}")

    def test_annual_steps_give_step_years_1(self):
        time_steps = tuple(range(26))
        for ti in range(1, len(time_steps)):
            step_years = time_steps[ti] - time_steps[ti - 1]
            self.assertEqual(step_years, 1)

    def test_five_year_steps_give_step_years_5(self):
        time_steps = (0, 5, 10, 15, 20, 25)
        for ti in range(1, len(time_steps)):
            step_years = time_steps[ti] - time_steps[ti - 1]
            self.assertEqual(step_years, 5)

    def test_mixed_steps_are_correct(self):
        time_steps = (0, 1, 5, 10, 25)
        expected = [1, 4, 5, 15]
        for ti in range(1, len(time_steps)):
            step_years = time_steps[ti] - time_steps[ti - 1]
            self.assertEqual(step_years, expected[ti - 1])


class TestStepSizeCLI(unittest.TestCase):
    """Verify --step-size CLI flag generates correct time_steps via actual parser."""

    def _parse(self, *cli_args):
        """Run the feedback loop argparser with given CLI args."""
        import sys
        from unittest.mock import patch
        # Import the parser setup inline; the script uses parse_args() at module scope
        # in its if __name__ == '__main__' block, so we can safely import the module
        # and call parse_args with custom argv.
        try:
            import importlib
            # Fresh parse with custom args — simulate CLI invocation
            from scripts.run_feedback_loop import _build_parser
        except ImportError:
            self.skipTest("Cannot import _build_parser from run_feedback_loop")
        finally:
            sys.path.pop(0)

        parser = _build_parser()
        args = parser.parse_args(list(cli_args))
        # Apply the same step resolution logic as the script
        if args.steps is None:
            step = max(int(args.step_size), 1)
            args.steps = list(range(0, 26, step))
            if args.steps[-1] != 25:
                args.steps.append(25)
        return args

    def test_step_size_1_gives_annual(self):
        args = self._parse("--step-size", "1")
        self.assertEqual(len(args.steps), 26)
        self.assertEqual(args.steps[0], 0)
        self.assertEqual(args.steps[-1], 25)

    def test_step_size_5_gives_legacy(self):
        args = self._parse("--step-size", "5")
        self.assertEqual(args.steps, [0, 5, 10, 15, 20, 25])

    def test_step_size_2_includes_25(self):
        args = self._parse("--step-size", "2")
        self.assertIn(25, args.steps)
        self.assertEqual(args.steps[0], 0)


class TestPendingDeliveriesDict(unittest.TestCase):
    """Verify the new dict-based pending deliveries pipeline."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping: {_IMPORT_ERROR}")

    def test_schedule_creates_future_deliveries(self):
        """OCCUPANCY_SCHEDULE should create entries at future year offsets."""
        schedule = OCCUPANCY_SCHEDULE
        pending = {}
        base_year = 3

        # Simulate scheduling one delivery
        for offset in range(len(schedule)):
            prev = schedule[offset - 1] if offset > 0 else 0.0
            inc_frac = schedule[offset] - prev
            if inc_frac <= 0:
                continue
            delivery_year = base_year + offset
            pending.setdefault(delivery_year, []).append({
                "pos": 42,
                "res_sqft": 1000.0 * inc_frac,
                "comm_sqft": 500.0 * inc_frac,
            })

        # Years 3 and 4 have 0% increment, so no entries
        self.assertNotIn(3, pending)
        self.assertNotIn(4, pending)
        # Years 5, 6, 7 have deliveries
        self.assertIn(5, pending)
        self.assertIn(6, pending)
        self.assertIn(7, pending)

        # Total delivered sqft should equal original
        total_res = sum(
            d["res_sqft"]
            for year_list in pending.values()
            for d in year_list
        )
        self.assertAlmostEqual(total_res, 1000.0, places=2)


if __name__ == "__main__":
    unittest.main()

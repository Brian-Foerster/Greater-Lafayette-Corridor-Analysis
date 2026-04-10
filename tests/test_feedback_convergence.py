"""Unit tests for Week 1 convergence scaffolding."""
import math
import unittest

try:
    from src.land_use_transport_model import compute_relative_delta, evaluate_convergence
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    compute_relative_delta = None
    evaluate_convergence = None
    _IMPORT_ERROR = exc


class TestFeedbackConvergence(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_relative_delta_uses_previous_when_large_enough(self):
        delta = compute_relative_delta(current=103.0, previous=100.0, floor=25.0)
        self.assertAlmostEqual(delta, 0.03, places=8)

    def test_relative_delta_uses_floor_when_previous_near_zero(self):
        delta = compute_relative_delta(current=10.0, previous=0.0, floor=25.0)
        self.assertAlmostEqual(delta, 0.4, places=8)

    def test_evaluate_convergence_baseline(self):
        result = evaluate_convergence(
            current_metrics={"daily_riders": 100.0, "new_pop": 20.0, "new_jobs": 5.0},
            previous_metrics=None,
            ridership_tol=0.03,
            development_tol=0.05,
            floor=25.0,
        )
        self.assertFalse(result["is_converged"])
        self.assertEqual(result["convergence_state"], "baseline")
        self.assertTrue(math.isnan(result["ridership_rel_delta"]))
        self.assertTrue(math.isnan(result["new_pop_rel_delta"]))
        self.assertTrue(math.isnan(result["new_jobs_rel_delta"]))

    def test_evaluate_convergence_true_when_all_thresholds_met(self):
        result = evaluate_convergence(
            current_metrics={"daily_riders": 102.0, "new_pop": 51.0, "new_jobs": 20.8},
            previous_metrics={"daily_riders": 100.0, "new_pop": 50.0, "new_jobs": 20.0},
            ridership_tol=0.03,
            development_tol=0.05,
            floor=25.0,
        )
        self.assertTrue(result["is_converged"])
        self.assertEqual(result["convergence_state"], "converged")

    def test_evaluate_convergence_false_when_any_threshold_exceeded(self):
        result = evaluate_convergence(
            current_metrics={"daily_riders": 108.0, "new_pop": 51.0, "new_jobs": 20.8},
            previous_metrics={"daily_riders": 100.0, "new_pop": 50.0, "new_jobs": 20.0},
            ridership_tol=0.03,
            development_tol=0.05,
            floor=25.0,
        )
        self.assertFalse(result["is_converged"])
        self.assertEqual(result["convergence_state"], "not_converged")


if __name__ == "__main__":
    unittest.main()

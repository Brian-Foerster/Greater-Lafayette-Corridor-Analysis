"""Synthetic-sequence tests for Week 2 adaptive stopping behavior."""
import unittest

try:
    from src.land_use_transport_model import (
        evaluate_stop_conditions,
        summarize_year_convergence,
    )
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    evaluate_stop_conditions = None
    summarize_year_convergence = None
    _IMPORT_ERROR = exc


class TestFeedbackAdaptiveStop(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_stable_sequence_triggers_adaptive_stop(self):
        converged_streak = 0
        divergent_streak = 0
        stop_year = None

        synthetic_years = [
            # baseline-like period: not converged
            {"C1": {"is_converged": False, "ridership_rel_delta": 0.40, "new_pop_rel_delta": 0.30, "new_jobs_rel_delta": 0.20}},
            # converged period 1
            {"C1": {"is_converged": True, "ridership_rel_delta": 0.01, "new_pop_rel_delta": 0.01, "new_jobs_rel_delta": 0.01}},
            # converged period 2 -> should trigger stop for consecutive=2
            {"C1": {"is_converged": True, "ridership_rel_delta": 0.01, "new_pop_rel_delta": 0.01, "new_jobs_rel_delta": 0.01}},
        ]

        for i, by_corridor in enumerate(synthetic_years):
            summary = summarize_year_convergence(by_corridor, divergence_threshold=1.0)
            decision = evaluate_stop_conditions(
                year_all_converged=summary["all_converged"],
                year_divergent=summary["divergence_flag"],
                converged_streak=converged_streak,
                divergent_streak=divergent_streak,
                adaptive_stop=True,
                consecutive_converged_steps=2,
                stop_on_divergence=False,
                consecutive_divergent_steps=2,
            )
            converged_streak = decision["converged_streak"]
            divergent_streak = decision["divergent_streak"]
            if decision["stop_triggered"]:
                stop_year = i
                self.assertEqual(decision["stop_reason"], "adaptive_converged_stop")
                break

        self.assertEqual(stop_year, 2)

    def test_divergent_sequence_triggers_divergence_stop(self):
        converged_streak = 0
        divergent_streak = 0
        stop_year = None

        synthetic_years = [
            {"C1": {"is_converged": False, "ridership_rel_delta": 0.10, "new_pop_rel_delta": 0.10, "new_jobs_rel_delta": 0.10}},
            # divergent period 1 (max delta > 1.0)
            {"C1": {"is_converged": False, "ridership_rel_delta": 1.20, "new_pop_rel_delta": 0.20, "new_jobs_rel_delta": 0.20}},
            # divergent period 2 -> should trigger stop
            {"C1": {"is_converged": False, "ridership_rel_delta": 1.30, "new_pop_rel_delta": 0.20, "new_jobs_rel_delta": 0.20}},
        ]

        for i, by_corridor in enumerate(synthetic_years):
            summary = summarize_year_convergence(by_corridor, divergence_threshold=1.0)
            decision = evaluate_stop_conditions(
                year_all_converged=summary["all_converged"],
                year_divergent=summary["divergence_flag"],
                converged_streak=converged_streak,
                divergent_streak=divergent_streak,
                adaptive_stop=False,
                consecutive_converged_steps=2,
                stop_on_divergence=True,
                consecutive_divergent_steps=2,
            )
            converged_streak = decision["converged_streak"]
            divergent_streak = decision["divergent_streak"]
            if decision["stop_triggered"]:
                stop_year = i
                self.assertEqual(decision["stop_reason"], "divergence_stop")
                break

        self.assertEqual(stop_year, 2)


if __name__ == "__main__":
    unittest.main()

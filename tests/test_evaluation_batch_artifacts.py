"""Week 13 tests for batch artifact naming and ranking deltas."""

import unittest

import pandas as pd

try:
    from scripts.run_evaluation_batch import (
        build_batch_tag,
        build_mode_run_tag,
        compute_ranking_deltas,
        normalize_corridor_ranking,
    )
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    build_batch_tag = None
    build_mode_run_tag = None
    compute_ranking_deltas = None
    normalize_corridor_ranking = None
    _IMPORT_ERROR = exc


class TestEvaluationBatchArtifacts(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_batch_and_mode_tags(self):
        batch_tag = build_batch_tag(
            iterations=10,
            population=50,
            n_output=25,
            timestamp_token="20260222T120000Z",
            git_commit="abc1234",
        )
        self.assertEqual(
            batch_tag,
            "batch_v1_20260222T120000Z_it10_pop50_out25_abc1234",
        )
        self.assertEqual(build_mode_run_tag("network_aware"), "network_aware_v1")

    def test_normalize_ranking_is_deterministic_on_ties(self):
        df = pd.DataFrame(
            {
                "corridor_id": ["C2", "C1", "C3"],
                "ridership_est": [100.0, 100.0, 90.0],
                "evaluation_mode": ["isolated"] * 3,
            }
        )
        rank_df = normalize_corridor_ranking(df)
        self.assertEqual(rank_df["corridor_id"].tolist(), ["C1", "C2", "C3"])
        self.assertEqual(rank_df["rank"].tolist(), [1, 2, 3])

    def test_compute_ranking_deltas(self):
        isolated = pd.DataFrame(
            {
                "corridor_id": ["C1", "C2", "C3"],
                "rank": [1, 2, 3],
                "ridership_est": [120.0, 110.0, 90.0],
            }
        )
        network = pd.DataFrame(
            {
                "corridor_id": ["C1", "C2", "C3"],
                "rank": [2, 1, 3],
                "ridership_est": [118.0, 116.0, 92.0],
            }
        )
        delta = compute_ranking_deltas(isolated, network)
        self.assertIn("rank_delta", delta.columns)
        c2 = delta[delta["corridor_id"] == "C2"].iloc[0]
        self.assertEqual(int(c2["rank_delta"]), -1)
        self.assertEqual(str(c2["direction"]), "up")


if __name__ == "__main__":
    unittest.main()

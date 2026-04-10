"""Week 14 network validation gate tests."""

import json
import sys
import types

import pytest
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

try:
    from scripts import run_evaluation_batch as reb
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    reb = None
    _IMPORT_ERROR = exc

try:
    from src.transit_model import assign_ridership_multi_corridor
    _TRANSIT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    assign_ridership_multi_corridor = None
    _TRANSIT_IMPORT_ERROR = exc


def _stub_run_optimized_search(
    data,
    use_network,
    n_iterations,
    population_size,
    n_output,
    evaluation_mode,
    network_synergy_weight,
    network_anchor_top_k,
    network_transfer_radius_m,
):
    corridor_ids = ["C1", "C2", "C3"]
    if evaluation_mode == "network_aware":
        ridership = [118.0, 130.0, 100.0]
        synergy = [0.20, 0.60, 0.10]
    else:
        ridership = [120.0, 110.0, 100.0]
        synergy = [0.0, 0.0, 0.0]

    corridors = gpd.GeoDataFrame(
        {
            "corridor_id": corridor_ids,
            "length_km": [6.0, 7.2, 5.5],
            "n_stops": [8, 9, 7],
            "ridership_est": ridership,
            "tif_potential": [1.0, 1.0, 1.0],
            "cost_efficiency": [1.0, 1.0, 1.0],
            "weighted_demand": [1.0, 1.0, 1.0],
            "bus_competition": [0.5, 0.5, 0.5],
            "apm_share": [0.2, 0.2, 0.2],
            "network_synergy": synergy,
            "ridership_est_base": [120.0, 110.0, 100.0],
            "ridership_network_adjusted": ridership,
            "evaluation_mode": [evaluation_mode] * 3,
        },
        geometry=[
            LineString([(0.0, 0.0), (0.1, 0.1)]),
            LineString([(0.1, 0.0), (0.2, 0.1)]),
            LineString([(0.2, 0.0), (0.3, 0.1)]),
        ],
        crs="EPSG:4326",
    )

    stops = gpd.GeoDataFrame(
        {
            "corridor_id": ["C1", "C2", "C3"],
            "stop_id": ["C1_S1", "C2_S1", "C3_S1"],
            "sequence": [1, 1, 1],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(0.1, 0.0),
            Point(0.2, 0.0),
        ],
        crs="EPSG:4326",
    )

    final_selection = [
        {
            "id": cid,
            "score": {
                "ridership_est": r,
                "evaluation_mode": evaluation_mode,
                "network_synergy": s,
            },
        }
        for cid, r, s in zip(corridor_ids, ridership, synergy)
    ]
    return final_selection, corridors, stops


class TestNetworkValidationGate(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    @pytest.mark.slow
    def test_optimization_batch_smoke_outputs_expected_artifacts(self):
        fake_module = types.ModuleType("optimized_corridor_search")
        fake_module.load_all_data = lambda: {"ok": True}
        fake_module.run_optimized_search = _stub_run_optimized_search

        with TemporaryDirectory() as tmpdir:
            out_root = Path(tmpdir)
            with patch.dict(sys.modules, {"optimized_corridor_search": fake_module}):
                with patch.object(reb, "utc_timestamp_compact", return_value="20260101T000000Z"):
                    with patch.object(reb, "get_git_commit_short", return_value="abc1234"):
                        outputs = reb.run_optimization_batch(
                            evaluation_modes=["isolated", "network_aware"],
                            n_iterations=1,
                            population_size=8,
                            n_output=3,
                            use_network=False,
                            network_synergy_weight=0.2,
                            network_anchor_top_k=12,
                            network_transfer_radius_m=1200.0,
                            output_root=out_root,
                            validate_manifest=False,
                        )

            self.assertEqual(set(outputs.keys()), {"isolated", "network_aware"})

            batch_tag = reb.build_batch_tag(
                iterations=1,
                population=8,
                n_output=3,
                timestamp_token="20260101T000000Z",
                git_commit="abc1234",
            )
            batch_dir = out_root / batch_tag

            self.assertTrue((batch_dir / "isolated_v1" / "ranking_isolated_v1.csv").exists())
            self.assertTrue((batch_dir / "network_aware_v1" / "ranking_network_aware_v1.csv").exists())
            self.assertTrue((batch_dir / "batch_artifact_manifest_v1.csv").exists())

            delta_path = batch_dir / "ranking_delta_isolated_vs_network_aware_v1.csv"
            self.assertTrue(delta_path.exists())
            delta_df = pd.read_csv(delta_path)
            self.assertGreaterEqual(int((delta_df["rank_delta"] != 0).sum()), 1)
            self.assertIn("direction", delta_df.columns)
            self.assertTrue(set(delta_df["direction"]).issubset({"up", "down", "unchanged"}))

            summary_path = batch_dir / "ranking_delta_summary_v1.json"
            self.assertTrue(summary_path.exists())
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(int(summary["n_corridors"]), 3)

    def test_transfer_assignment_is_reproducible_for_identical_inputs(self):
        if _TRANSIT_IMPORT_ERROR is not None:
            self.skipTest(f"Skipping transit-model reproducibility test: {_TRANSIT_IMPORT_ERROR}")

        od = pd.DataFrame(
            [{"origin_parcel": "P1", "dest_parcel": "P2", "trips": 120.0}]
        )
        access = pd.DataFrame(
            [
                {"PARCEL_ID": "P1", "corridor_id": "C1", "access_weight": 0.95},
                {"PARCEL_ID": "P1", "corridor_id": "C2", "access_weight": 0.55},
                {"PARCEL_ID": "P2", "corridor_id": "C1", "access_weight": 0.55},
                {"PARCEL_ID": "P2", "corridor_id": "C2", "access_weight": 0.95},
            ]
        )
        travel_time = {
            ("C1", "C1"): 40.0,
            ("C2", "C2"): 40.0,
            ("C1", "C2"): 18.0,
            ("C2", "C1"): 18.0,
        }

        c1, p1 = assign_ridership_multi_corridor(
            od_df=od,
            parcel_corridor_access_df=access,
            corridor_travel_time=travel_time,
        )
        c2, p2 = assign_ridership_multi_corridor(
            od_df=od,
            parcel_corridor_access_df=access,
            corridor_travel_time=travel_time,
        )

        c1 = c1.sort_values(["corridor_id"]).reset_index(drop=True)
        c2 = c2.sort_values(["corridor_id"]).reset_index(drop=True)
        path_sort_cols = [c for c in ["origin_parcel", "dest_parcel", "corridor_from", "corridor_to", "path_type"] if c in p1.columns]
        p1 = p1.sort_values(path_sort_cols).reset_index(drop=True)
        p2 = p2.sort_values(path_sort_cols).reset_index(drop=True)

        self.assertTrue(c1.equals(c2))
        self.assertTrue(p1.equals(p2))
        self.assertAlmostEqual(float(c1["assigned_trips"].sum()), 120.0, places=6)


if __name__ == "__main__":
    unittest.main()

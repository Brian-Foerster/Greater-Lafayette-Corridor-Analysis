"""Week 11 tests for transfer-aware multi-corridor assignment."""

import sys
import unittest
from pathlib import Path

import pandas as pd
from src.spatial_constants import PROJECT_CRS

try:
    from src.transit_model import (
        TransferAssignmentParams,
        assign_ridership_multi_corridor,
        build_parcel_corridor_access,
        compute_transfer_path_utility,
    )
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    TransferAssignmentParams = None
    assign_ridership_multi_corridor = None
    build_parcel_corridor_access = None
    compute_transfer_path_utility = None
    _IMPORT_ERROR = exc


def _transfer_share(path_df: pd.DataFrame) -> float:
    total = float(path_df["assigned_trips"].sum()) if not path_df.empty else 0.0
    if total <= 0:
        return 0.0
    transfer = float(path_df.loc[path_df["n_transfers"] > 0, "assigned_trips"].sum())
    return transfer / total


class TestTransferAwareAssignment(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_supports_one_transfer_path(self):
        od = pd.DataFrame(
            [{"origin_parcel": "P1", "dest_parcel": "P2", "trips": 100.0}]
        )
        access = pd.DataFrame(
            [
                {"PARCEL_ID": "P1", "corridor_id": "C1", "access_weight": 0.95},
                {"PARCEL_ID": "P2", "corridor_id": "C2", "access_weight": 0.90},
            ]
        )
        tt = {("C1", "C2"): 18.0}

        corridor_df, path_df = assign_ridership_multi_corridor(
            od_df=od,
            parcel_corridor_access_df=access,
            corridor_travel_time=tt,
        )

        self.assertFalse(path_df.empty)
        self.assertEqual(int(path_df["n_transfers"].max()), 1)
        self.assertAlmostEqual(float(path_df["assigned_trips"].sum()), 100.0, places=6)
        self.assertAlmostEqual(float(corridor_df["assigned_trips"].sum()), 100.0, places=6)
        self.assertEqual(set(corridor_df["corridor_id"]), {"C1", "C2"})

    def test_transfer_penalty_reduces_transfer_share(self):
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
        tt = {
            ("C1", "C1"): 40.0,
            ("C2", "C2"): 40.0,
            ("C1", "C2"): 18.0,
            ("C2", "C1"): 18.0,
        }
        low_params = TransferAssignmentParams(transfer_penalty_min=2.0)
        high_params = TransferAssignmentParams(transfer_penalty_min=20.0)

        _, low_paths = assign_ridership_multi_corridor(
            od_df=od,
            parcel_corridor_access_df=access,
            corridor_travel_time=tt,
            params=low_params,
        )
        _, high_paths = assign_ridership_multi_corridor(
            od_df=od,
            parcel_corridor_access_df=access,
            corridor_travel_time=tt,
            params=high_params,
        )

        self.assertGreater(_transfer_share(low_paths), _transfer_share(high_paths))

    def test_custom_utility_hook_can_disable_transfers(self):
        od = pd.DataFrame(
            [{"origin_parcel": "P1", "dest_parcel": "P2", "trips": 100.0}]
        )
        access = pd.DataFrame(
            [
                {"PARCEL_ID": "P1", "corridor_id": "C1", "access_weight": 0.95},
                {"PARCEL_ID": "P1", "corridor_id": "C2", "access_weight": 0.55},
                {"PARCEL_ID": "P2", "corridor_id": "C1", "access_weight": 0.55},
                {"PARCEL_ID": "P2", "corridor_id": "C2", "access_weight": 0.95},
            ]
        )
        tt = {
            ("C1", "C1"): 40.0,
            ("C2", "C2"): 40.0,
            ("C1", "C2"): 18.0,
            ("C2", "C1"): 18.0,
        }

        def no_transfer_hook(
            origin_access: float,
            dest_access: float,
            in_vehicle_time_min: float,
            n_transfers: int,
            params: TransferAssignmentParams,
        ) -> float:
            base = compute_transfer_path_utility(
                origin_access,
                dest_access,
                in_vehicle_time_min,
                n_transfers=n_transfers,
                params=params,
            )
            return base - 1000.0 * n_transfers

        _, paths = assign_ridership_multi_corridor(
            od_df=od,
            parcel_corridor_access_df=access,
            corridor_travel_time=tt,
            params=TransferAssignmentParams(transfer_penalty_min=1.0),
            utility_hook=no_transfer_hook,
        )

        transfer_assigned = float(paths.loc[paths["n_transfers"] > 0, "assigned_trips"].sum())
        self.assertLess(transfer_assigned, 1e-6)

    def test_build_parcel_corridor_access_from_stops(self):
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except Exception as exc:
            self.skipTest(f"Skipping geopandas/shapely-dependent test: {exc}")

        parcels = gpd.GeoDataFrame(
            {"PARCEL_ID": ["P1", "P2"]},
            geometry=[Point(0, 0), Point(1000, 0)],
            crs=PROJECT_CRS,
        )
        c1_stops = gpd.GeoDataFrame(
            {"stop_id": ["S1"]},
            geometry=[Point(0, 0)],
            crs=PROJECT_CRS,
        )
        c2_stops = gpd.GeoDataFrame(
            {"stop_id": ["S2"]},
            geometry=[Point(1000, 0)],
            crs=PROJECT_CRS,
        )
        access = build_parcel_corridor_access(
            parcels_gdf=parcels,
            corridor_stops={"C1": c1_stops, "C2": c2_stops},
            beta_walk=0.01,
            max_walk_m=300.0,
            walk_speed_kph=5.0,
        )
        self.assertFalse(access.empty)
        self.assertIn("access_weight", access.columns)
        self.assertTrue({"C1", "C2"}.issubset(set(access["corridor_id"])))


if __name__ == "__main__":
    unittest.main()

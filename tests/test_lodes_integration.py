"""Tests for src/lodes_integration.py — LODES OD flow processing."""
from __future__ import annotations

import unittest

import pandas as pd


class TestCreateParcelODFlows(unittest.TestCase):
    """Test block-to-parcel OD flow distribution."""

    def test_single_block_pair(self):
        from src.lodes_integration import create_parcel_od_flows

        od_df = pd.DataFrame({
            "h_geocode": ["180010001001000"],
            "w_geocode": ["180010001002000"],
            "S000": [100],
            "SE01": [30],
            "SE02": [40],
            "SE03": [30],
        })
        block_parcel_map = pd.DataFrame({
            "PARCEL_ID": ["P001", "P002"],
            "geocode": ["180010001001000", "180010001002000"],
        })

        result = create_parcel_od_flows(od_df, block_parcel_map, min_jobs=0)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result["trips"].sum(), 100.0, places=1)
        self.assertEqual(result.iloc[0]["origin_parcel"], "P001")
        self.assertEqual(result.iloc[0]["dest_parcel"], "P002")

    def test_multiple_parcels_per_block(self):
        from src.lodes_integration import create_parcel_od_flows

        od_df = pd.DataFrame({
            "h_geocode": ["BLK_A"],
            "w_geocode": ["BLK_B"],
            "S000": [100],
            "SE01": [30],
            "SE02": [40],
            "SE03": [30],
        })
        # 2 parcels in origin block, 1 in dest
        block_parcel_map = pd.DataFrame({
            "PARCEL_ID": ["P001", "P002", "P003"],
            "geocode": ["BLK_A", "BLK_A", "BLK_B"],
        })

        result = create_parcel_od_flows(od_df, block_parcel_map, min_jobs=0)

        # 2 origin × 1 dest = 2 flows, each 100/(2*1) = 50
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result["trips"].sum(), 100.0, places=1)

    def test_min_jobs_filter(self):
        from src.lodes_integration import create_parcel_od_flows

        od_df = pd.DataFrame({
            "h_geocode": ["BLK_A", "BLK_C"],
            "w_geocode": ["BLK_B", "BLK_D"],
            "S000": [10, 1],
            "SE01": [3, 0],
            "SE02": [4, 1],
            "SE03": [3, 0],
        })
        block_parcel_map = pd.DataFrame({
            "PARCEL_ID": ["P001", "P002", "P003", "P004"],
            "geocode": ["BLK_A", "BLK_B", "BLK_C", "BLK_D"],
        })

        result = create_parcel_od_flows(od_df, block_parcel_map, min_jobs=5)
        # Only the first pair (10 trips) should survive
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result["trips"].iloc[0], 10.0, places=1)

    def test_no_matches_returns_empty(self):
        from src.lodes_integration import create_parcel_od_flows

        od_df = pd.DataFrame({
            "h_geocode": ["BLK_X"],
            "w_geocode": ["BLK_Y"],
            "S000": [100],
            "SE01": [30],
            "SE02": [40],
            "SE03": [30],
        })
        block_parcel_map = pd.DataFrame({
            "PARCEL_ID": ["P001"],
            "geocode": ["BLK_Z"],  # no match
        })

        result = create_parcel_od_flows(od_df, block_parcel_map, min_jobs=0)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()

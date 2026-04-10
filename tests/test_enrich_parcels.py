"""Tests for src/enrich_parcels.py — parcel enrichment utilities."""
from __future__ import annotations

import os
import tempfile
import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


class TestLoadAndPrepareAVLayer(unittest.TestCase):
    """Test AV layer loading and ID normalization."""

    def _make_av_gdf(self, id_col="PARCEL_ID", ids=None):
        if ids is None:
            ids = ["790719233009000026", "790719233009000027"]
        return gpd.GeoDataFrame(
            {id_col: ids, "CurLandAV": [50000, 75000]},
            geometry=[Point(-86.91, 40.42), Point(-86.92, 40.43)],
            crs="EPSG:4326",
        )

    def test_detects_stkey_column(self):
        from src.enrich_parcels import load_and_prepare_av_layer

        gdf = self._make_av_gdf("STKEY")
        path = os.path.join(tempfile.mkdtemp(), "av.geojson")
        gdf.to_file(path, driver="GeoJSON")

        result_df, val_col = load_and_prepare_av_layer(path)
        self.assertIn("PARCEL_ID", result_df.columns)
        self.assertEqual(len(result_df), 2)
        self.assertEqual(val_col, "CurLandAV")

    def test_detects_pin_column(self):
        from src.enrich_parcels import load_and_prepare_av_layer

        gdf = gpd.GeoDataFrame(
            {"PIN": [1001, 1002], "CurLandAV": [50000, 75000]},
            geometry=[Point(-86.91, 40.42), Point(-86.92, 40.43)],
            crs="EPSG:4326",
        )
        path = os.path.join(tempfile.mkdtemp(), "av.geojson")
        gdf.to_file(path, driver="GeoJSON")

        result_df, val_col = load_and_prepare_av_layer(path)
        self.assertIn("PARCEL_ID", result_df.columns)
        # PIN IDs should be prefixed
        self.assertTrue(result_df["PARCEL_ID"].str.startswith("PIN").all())

    def test_raises_on_missing_id(self):
        from src.enrich_parcels import load_and_prepare_av_layer

        gdf = gpd.GeoDataFrame(
            {"unknown_col": [1, 2]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326",
        )
        path = os.path.join(tempfile.mkdtemp(), "bad.geojson")
        gdf.to_file(path, driver="GeoJSON")

        with self.assertRaises(KeyError):
            load_and_prepare_av_layer(path)


if __name__ == "__main__":
    unittest.main()

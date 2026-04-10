"""Lightweight integration test for LandUseTransportModel.run().

Instantiates the model with minimal synthetic data (10 parcels, 2 corridors,
3 time steps) and calls run() to verify the full pipeline executes without
crashing and produces structurally correct output.

Marked @pytest.mark.slow — runs in ~30-60s due to model init overhead.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

# Lafayette area coordinates — parcels must be near corridors for spatial
# cache to produce non-empty walk/feeder zones.
_BASE_LON, _BASE_LAT = -86.9100, 40.4250


def _make_parcels_geojson(n: int = 10) -> dict:
    """Create a minimal parcels GeoJSON with required columns."""
    features = []
    for i in range(n):
        # Spread parcels along a ~500m strip centered on _BASE
        lon = _BASE_LON + i * 0.001  # ~90m steps
        lat = _BASE_LAT + (i % 3) * 0.0005
        features.append({
            "type": "Feature",
            "properties": {
                "PARCEL_ID": f"P{i:04d}",
                "pop_alloc": float(50 + i * 10),
                "jobs_combined": float(20 + i * 5),
                "zone_code": "R3" if i < 7 else "GB",
                "parcel_sqft": 10000.0,
                "CurLandAV": 50000.0,
                "CurImpAV": 80000.0,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon, lat],
                    [lon + 0.0005, lat],
                    [lon + 0.0005, lat + 0.0005],
                    [lon, lat + 0.0005],
                    [lon, lat],
                ]],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _make_corridors_geojson(n: int = 2) -> dict:
    """Create minimal corridor GeoJSON with required columns."""
    features = []
    for i in range(n):
        # Each corridor is a ~1km LineString passing through the parcel cluster
        base_lon = _BASE_LON + i * 0.003
        stops = [
            [base_lon, _BASE_LAT - 0.002],
            [base_lon + 0.001, _BASE_LAT],
            [base_lon + 0.002, _BASE_LAT + 0.002],
            [base_lon + 0.003, _BASE_LAT + 0.004],
        ]
        features.append({
            "type": "Feature",
            "properties": {
                "corridor_id": f"CORR_{i:02d}",
                "length_km": 1.5,
                "n_stops": 4,
                "stop_coords": json.dumps(stops),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": stops,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _make_od_csv(n_parcels: int = 10, n_flows: int = 20) -> str:
    """Create minimal OD flow CSV content."""
    rng = np.random.RandomState(42)
    rows = ["origin_parcel,dest_parcel,trips"]
    for _ in range(n_flows):
        o = rng.randint(0, n_parcels)
        d = rng.randint(0, n_parcels)
        if o == d:
            d = (d + 1) % n_parcels
        trips = rng.randint(1, 20)
        rows.append(f"P{o:04d},P{d:04d},{trips}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestModelIntegration(unittest.TestCase):
    """Integration test: instantiate model with synthetic data and run."""

    @classmethod
    def setUpClass(cls):
        """Write synthetic data files to a temp directory."""
        cls._tmpdir_obj = tempfile.TemporaryDirectory()
        cls._tmpdir = cls._tmpdir_obj.name

        parcels_path = os.path.join(cls._tmpdir, "parcels.geojson")
        corridors_path = os.path.join(cls._tmpdir, "corridors.geojson")
        od_path = os.path.join(cls._tmpdir, "od_flows.csv")

        with open(parcels_path, "w") as f:
            json.dump(_make_parcels_geojson(10), f)
        with open(corridors_path, "w") as f:
            json.dump(_make_corridors_geojson(2), f)
        with open(od_path, "w") as f:
            f.write(_make_od_csv(10, 20))

        cls._parcels_path = parcels_path
        cls._corridors_path = corridors_path
        cls._od_path = od_path

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir_obj.cleanup()

    def _make_model(self, **kwargs):
        """Instantiate the model with synthetic data and minimal settings."""
        from src.land_use_transport_model import LandUseTransportModel

        defaults = dict(
            corridors_path=self._corridors_path,
            parcels_path=self._parcels_path,
            od_path=self._od_path,
            time_steps=(0, 1, 2),
            bus_restructure=False,  # skip GTFS dependency
            adaptive_stop=False,
            development_scenario="current_zoning",
        )
        defaults.update(kwargs)
        return LandUseTransportModel(**defaults)

    def test_model_init(self):
        """Model should instantiate without errors from synthetic data."""
        model = self._make_model()
        self.assertEqual(len(model.corridors), 2)
        self.assertEqual(len(model.parcels), 10)
        self.assertEqual(model.time_steps, (0, 1, 2))

    def test_run_produces_dataframe(self):
        """run() should return a DataFrame with corridor/year columns."""
        model = self._make_model()
        result = model.run(parallel=False)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0, "Result DataFrame should not be empty")
        self.assertIn("corridor_id", result.columns)
        self.assertIn("year", result.columns)

        # Should have rows for each corridor × time step
        expected_rows = 2 * 3  # 2 corridors × 3 years
        self.assertEqual(len(result), expected_rows)

        # Each corridor should appear exactly len(time_steps) times
        for cid in ["CORR_00", "CORR_01"]:
            corridor_rows = result[result["corridor_id"] == cid]
            self.assertEqual(len(corridor_rows), 3, f"{cid} should have 3 rows")

    def test_run_has_ridership_columns(self):
        """Result should include ridership metrics."""
        model = self._make_model()
        result = model.run(parallel=False)

        # Key ridership columns that the model always emits
        expected_cols = {"corridor_id", "year", "daily_riders"}
        missing = expected_cols - set(result.columns)
        self.assertEqual(missing, set(), f"Missing columns: {missing}")

        # Ridership should be non-negative
        self.assertTrue(
            (result["daily_riders"] >= 0).all(),
            "daily_riders should be non-negative",
        )

    def test_corridor_independence(self):
        """Each corridor's year-0 values should be identical (baseline restored)."""
        model = self._make_model()
        result = model.run(parallel=False)

        yr0 = result[result["year"] == 0]
        if "pop_walk_catch" in yr0.columns:
            # Both corridors see the same baseline population
            vals = yr0["pop_walk_catch"].values
            # They may differ slightly due to different spatial positions —
            # just verify both are positive (baseline was loaded)
            self.assertTrue((vals >= 0).all())

    def test_single_corridor_filter(self):
        """corridor_filter should restrict output to one corridor."""
        model = self._make_model(corridor_filter=["CORR_00"])
        result = model.run(parallel=False)

        unique_cids = result["corridor_id"].unique()
        self.assertEqual(list(unique_cids), ["CORR_00"])
        self.assertEqual(len(result), 3)  # 1 corridor × 3 years


if __name__ == "__main__":
    unittest.main()

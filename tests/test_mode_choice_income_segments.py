"""Week 4 tests for income-segment mode-choice coefficients."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from src.spatial_constants import PROJECT_CRS

try:
    from src.mode_choice import calculate_mode_shares_vectorized, mode_split_for_od
    from src.transit_model import load_lodes_for_mode_choice
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    calculate_mode_shares_vectorized = None
    mode_split_for_od = None
    load_lodes_for_mode_choice = None
    _IMPORT_ERROR = exc


def _one_hot_segment(seg_name: str, n: int) -> dict:
    base = {
        "SE01": np.zeros(n, dtype=float),
        "SE02": np.zeros(n, dtype=float),
        "SE03": np.zeros(n, dtype=float),
    }
    base[seg_name] = np.ones(n, dtype=float)
    return base


class TestModeChoiceIncomeSegments(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_baseline_equals_se02_one_hot(self):
        distances = np.array([2.0, 4.0, 6.0], dtype=float)
        baseline = calculate_mode_shares_vectorized(
            distances=distances,
            apm_headway=5.0,
            bus_headway=30.0,
            apm_fare=0.0,
            bus_fare=1.0,
        )
        se02 = calculate_mode_shares_vectorized(
            distances=distances,
            apm_headway=5.0,
            bus_headway=30.0,
            apm_fare=0.0,
            bus_fare=1.0,
            income_segment_shares=_one_hot_segment("SE02", len(distances)),
        )
        self.assertTrue(np.allclose(baseline["apm_share"], se02["apm_share"], atol=1e-9))
        self.assertTrue(np.allclose(baseline["car_share"], se02["car_share"], atol=1e-9))

    def test_se01_more_cost_sensitive_than_se03(self):
        distances = np.array([3.0, 5.0, 8.0], dtype=float)

        # Base fares
        se01_base = calculate_mode_shares_vectorized(
            distances, 5.0, 30.0, 0.0, 1.0, income_segment_shares=_one_hot_segment("SE01", len(distances))
        )
        se03_base = calculate_mode_shares_vectorized(
            distances, 5.0, 30.0, 0.0, 1.0, income_segment_shares=_one_hot_segment("SE03", len(distances))
        )

        # Fare shock
        se01_shock = calculate_mode_shares_vectorized(
            distances, 5.0, 30.0, 2.5, 3.5, income_segment_shares=_one_hot_segment("SE01", len(distances))
        )
        se03_shock = calculate_mode_shares_vectorized(
            distances, 5.0, 30.0, 2.5, 3.5, income_segment_shares=_one_hot_segment("SE03", len(distances))
        )

        drop_se01 = float((se01_base["transit_share"] - se01_shock["transit_share"]).mean())
        drop_se03 = float((se03_base["transit_share"] - se03_shock["transit_share"]).mean())
        self.assertGreater(drop_se01, drop_se03)

    def test_se03_less_time_sensitive_than_se02(self):
        distances = np.array([3.0, 5.0, 8.0], dtype=float)

        # Base headways
        se02_base = calculate_mode_shares_vectorized(
            distances, 5.0, 30.0, 0.0, 1.0, income_segment_shares=_one_hot_segment("SE02", len(distances))
        )
        se03_base = calculate_mode_shares_vectorized(
            distances, 5.0, 30.0, 0.0, 1.0, income_segment_shares=_one_hot_segment("SE03", len(distances))
        )

        # Headway degradation
        se02_worse = calculate_mode_shares_vectorized(
            distances, 12.0, 45.0, 0.0, 1.0, income_segment_shares=_one_hot_segment("SE02", len(distances))
        )
        se03_worse = calculate_mode_shares_vectorized(
            distances, 12.0, 45.0, 0.0, 1.0, income_segment_shares=_one_hot_segment("SE03", len(distances))
        )

        drop_se02 = float((se02_base["transit_share"] - se02_worse["transit_share"]).mean())
        drop_se03 = float((se03_base["transit_share"] - se03_worse["transit_share"]).mean())
        self.assertGreater(drop_se02, drop_se03)

    def test_lodes_loader_adds_segment_share_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "od.csv"
            pd.DataFrame(
                {
                    "origin_parcel": ["A", "B"],
                    "dest_parcel": ["X", "Y"],
                    "trips": [10.0, 20.0],
                    "SE01": [8.0, 1.0],
                    "SE02": [1.0, 2.0],
                    "SE03": [1.0, 17.0],
                }
            ).to_csv(csv_path, index=False)

            out = load_lodes_for_mode_choice(csv_path, min_trips=0.0)
            for col in ["se01_share", "se02_share", "se03_share", "dominant_income_segment"]:
                self.assertIn(col, out.columns)

            self.assertEqual(out.loc[0, "dominant_income_segment"], "SE01")
            self.assertEqual(out.loc[1, "dominant_income_segment"], "SE03")
            expected_share = 8.0 / (8.0 + 1.0 + 1.0 + 0.001)
            self.assertAlmostEqual(float(out.loc[0, "se01_share"]), expected_share, places=8)

    def test_mode_split_for_od_uses_segment_composition(self):
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except Exception as exc:
            self.skipTest(f"Skipping geopandas/shapely-dependent test: {exc}")

        parcels = gpd.GeoDataFrame(
            {"PARCEL_ID": ["P1", "P2"]},
            geometry=[Point(0, 0), Point(400, 0)],
            crs=PROJECT_CRS,
        )
        stops = gpd.GeoDataFrame(
            {"stop_id": ["S1"]},
            geometry=[Point(200, 0)],
            crs=PROJECT_CRS,
        )
        # Same OD geometry and trips, different segment composition.
        od = pd.DataFrame(
            {
                "origin_parcel": ["P1", "P1"],
                "dest_parcel": ["P2", "P2"],
                "trips": [100.0, 100.0],
                "SE01": [100.0, 0.0],
                "SE02": [0.0, 0.0],
                "SE03": [0.0, 100.0],
            }
        )

        baseline = mode_split_for_od(
            od_df=od,
            parcels_gdf=parcels,
            bus_stops_gdf=stops,
            apm_stops_gdf=stops,
            apm_headway_min=8.0,
            bus_headway_min=20.0,
            apm_fare=2.0,
            bus_fare=2.0,
            use_income_segments=False,
        )
        self.assertAlmostEqual(
            float(baseline.loc[0, "transit_share"]),
            float(baseline.loc[1, "transit_share"]),
            places=10,
        )

        segmented = mode_split_for_od(
            od_df=od,
            parcels_gdf=parcels,
            bus_stops_gdf=stops,
            apm_stops_gdf=stops,
            apm_headway_min=8.0,
            bus_headway_min=20.0,
            apm_fare=2.0,
            bus_fare=2.0,
            use_income_segments=True,
        )
        self.assertNotAlmostEqual(
            float(segmented.loc[0, "transit_share"]),
            float(segmented.loc[1, "transit_share"]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()

"""Week 5 tests for parking-cost utility behavior."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from src.spatial_constants import PROJECT_CRS

try:
    from src.mode_choice import calculate_mode_shares_vectorized, mode_split_for_od
    from scripts.run_lodes_mode_choice import parse_parking_costs
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    calculate_mode_shares_vectorized = None
    mode_split_for_od = None
    parse_parking_costs = None
    _IMPORT_ERROR = exc


class TestModeChoiceParking(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_vectorized_parking_cost_reduces_car_share(self):
        distances = np.array([2.0, 4.0, 6.0], dtype=float)

        low = calculate_mode_shares_vectorized(
            distances=distances,
            apm_headway=5.0,
            bus_headway=20.0,
            apm_fare=0.0,
            bus_fare=1.0,
            car_parking_cost=0.0,
        )
        high = calculate_mode_shares_vectorized(
            distances=distances,
            apm_headway=5.0,
            bus_headway=20.0,
            apm_fare=0.0,
            bus_fare=1.0,
            car_parking_cost=8.0,
        )

        self.assertGreater(float(low["car_share"].mean()), float(high["car_share"].mean()))
        self.assertGreater(float(high["transit_share"].mean()), float(low["transit_share"].mean()))

    def test_mode_split_for_od_parking_cost_reduces_car_trips(self):
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
        od = pd.DataFrame(
            {
                "origin_parcel": ["P1", "P1", "P1"],
                "dest_parcel": ["P2", "P2", "P2"],
                "trips": [100.0, 100.0, 100.0],
            }
        )

        low = mode_split_for_od(
            od_df=od,
            parcels_gdf=parcels,
            bus_stops_gdf=stops,
            apm_stops_gdf=stops,
            apm_headway_min=10.0,
            bus_headway_min=25.0,
            apm_fare=0.0,
            bus_fare=1.0,
            car_parking_cost=0.0,
            use_income_segments=False,
        )
        high = mode_split_for_od(
            od_df=od,
            parcels_gdf=parcels,
            bus_stops_gdf=stops,
            apm_stops_gdf=stops,
            apm_headway_min=10.0,
            bus_headway_min=25.0,
            apm_fare=0.0,
            bus_fare=1.0,
            car_parking_cost=8.0,
            use_income_segments=False,
        )

        self.assertGreater(float(low["car_trips"].sum()), float(high["car_trips"].sum()))

    def test_parse_parking_costs(self):
        self.assertEqual(parse_parking_costs("0, 2.5,5"), [0.0, 2.5, 5.0])
        with self.assertRaises(ValueError):
            parse_parking_costs("")
        with self.assertRaises(ValueError):
            parse_parking_costs("-1,2")


if __name__ == "__main__":
    unittest.main()

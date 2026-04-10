"""Week 6 tests for institutional demand overlay ingestion and application."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from src.data.institutional_generators import load_institutional_overlay
    from src.transit_model import load_lodes_for_mode_choice
    from src.data.purdue_transit_demand import build_purdue_institutional_overlay
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    load_institutional_overlay = None
    load_lodes_for_mode_choice = None
    build_purdue_institutional_overlay = None
    _IMPORT_ERROR = exc


class TestInstitutionalOverlay(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_overlay_ingest_normalizes_alias_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            overlay_path = Path(tmp) / "overlay.csv"
            pd.DataFrame(
                {
                    "PARCEL_ID": ["P1", "P2"],
                    "demand_multiplier": [2.5, 1.8],
                    "category": ["campus_core", "campus_adjacent"],
                }
            ).to_csv(overlay_path, index=False)

            overlay = load_institutional_overlay(overlay_path)
            self.assertTrue({"parcel_id", "institutional_weight", "institutional_category", "institutional_source"}.issubset(overlay.columns))
            self.assertEqual(overlay.loc[0, "parcel_id"], "P1")
            self.assertAlmostEqual(float(overlay.loc[0, "institutional_weight"]), 2.5, places=6)

    def test_lodes_overlay_path_applies_trip_multiplier_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            od_path = Path(tmp) / "od.csv"
            overlay_path = Path(tmp) / "overlay.csv"

            pd.DataFrame(
                {
                    "origin_parcel": ["P1", "P2"],
                    "dest_parcel": ["P2", "P3"],
                    "trips": [10.0, 20.0],
                }
            ).to_csv(od_path, index=False)
            pd.DataFrame(
                {
                    "parcel_id": ["P1", "P3"],
                    "institutional_weight": [2.0, 3.0],
                }
            ).to_csv(overlay_path, index=False)

            out = load_lodes_for_mode_choice(
                lodes_path=od_path,
                min_trips=0.0,
                apply_institutional_overlay=True,
                institutional_overlay_path=overlay_path,
                institutional_strength=1.0,
                institutional_max_multiplier=10.0,
            )

            for col in [
                "trips_base",
                "institutional_origin_weight",
                "institutional_dest_weight",
                "institutional_multiplier",
                "trips_institutional_delta",
                "institutional_pair_flag",
                "institutional_overlay_applied",
                "institutional_overlay_source",
            ]:
                self.assertIn(col, out.columns)

            self.assertAlmostEqual(float(out.loc[0, "institutional_multiplier"]), (2.0 * 1.0) ** 0.5, places=8)
            self.assertAlmostEqual(float(out.loc[1, "institutional_multiplier"]), (1.0 * 3.0) ** 0.5, places=8)
            self.assertGreater(float(out["trips"].sum()), float(out["trips_base"].sum()))
            self.assertTrue(bool(out["institutional_overlay_applied"].all()))

    def test_lodes_overlay_from_parcels_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            od_path = Path(tmp) / "od.csv"
            pd.DataFrame(
                {
                    "origin_parcel": ["P1"],
                    "dest_parcel": ["P2"],
                    "trips": [10.0],
                }
            ).to_csv(od_path, index=False)
            parcels = pd.DataFrame(
                {
                    "PARCEL_ID": ["P1", "P2"],
                    "institutional_weight": [1.5, 1.0],
                }
            )

            out = load_lodes_for_mode_choice(
                lodes_path=od_path,
                min_trips=0.0,
                apply_institutional_overlay=True,
                parcels_gdf=parcels,  # DataFrame is sufficient for lookup columns
            )

            self.assertAlmostEqual(float(out.loc[0, "institutional_multiplier"]), (1.5 * 1.0) ** 0.5, places=8)
            self.assertEqual(str(out.loc[0, "institutional_overlay_source"]), "parcels_column")

    def test_build_purdue_institutional_overlay_schema(self):
        import unittest.mock as mock
        parcels = pd.DataFrame(
            {
                "PARCEL_ID": ["P1", "P2", "P3"],
                "institutional_weight": [2.0, 1.5, 1.0],
                "is_campus_core": [True, False, False],
                "is_campus_adjacent": [False, True, False],
            }
        )

        with mock.patch(
            "src.data.purdue_transit_demand.add_data_grounded_institutional_weights",
            return_value=parcels,
        ):
            overlay = build_purdue_institutional_overlay(
                parcels_gdf=parcels,
                include_non_institutional=False,
            )

        self.assertTrue({"parcel_id", "institutional_weight", "institutional_category", "institutional_source"}.issubset(overlay.columns))
        self.assertEqual(set(overlay["parcel_id"].tolist()), {"P1", "P2"})


if __name__ == "__main__":
    unittest.main()

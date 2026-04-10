import unittest

import geopandas as gpd
from shapely.geometry import Point

from scripts.generate_improved_ridership import _build_parcel_lookup, normalize_parcel_id


class TestParcelIdNormalization(unittest.TestCase):
    def test_normalize_parcel_id_examples(self) -> None:
        self.assertEqual(normalize_parcel_id("ST790601326011000034"), "790601326011000034")
        self.assertEqual(normalize_parcel_id("790601326011000034"), "790601326011000034")
        self.assertEqual(normalize_parcel_id(" 00790601326011000034 "), "790601326011000034")
        self.assertEqual(normalize_parcel_id("ST0"), "0")
        self.assertEqual(normalize_parcel_id(""), "")
        self.assertEqual(normalize_parcel_id(None), "")

    def test_lookup_supports_normalized_od_ids(self) -> None:
        parcels = gpd.GeoDataFrame(
            {"PARCEL_ID": ["ST790601326011000034"]},
            geometry=[Point(-86.9, 40.4)],
            crs="EPSG:4326",
        )
        _, _, lookup = _build_parcel_lookup(parcels)
        self.assertIn("ST790601326011000034", lookup)
        self.assertIn("790601326011000034", lookup)
        self.assertEqual(lookup["ST790601326011000034"], lookup["790601326011000034"])


if __name__ == "__main__":
    unittest.main()

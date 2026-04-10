import unittest
import pandas as pd

from src.normalize_parcels import _detect_id_field, coerce_parcel_id


class TestNormalizeParcels(unittest.TestCase):
    def test_detect_stkey_preferred(self):
        df = pd.DataFrame(columns=['ParcelID', 'PIN', 'STKEY', 'Other'])
        detected = _detect_id_field(df)
        self.assertEqual(detected, 'STKEY')

    def test_coerce_stkey_normalization(self):
        data = {
            'STKEY': ['79-10-25-200-020.000-020', '79 10 25 200 021.000.020', None]
        }
        gdf = pd.DataFrame(data)
        out = coerce_parcel_id(gdf, 'STKEY')
        # PARCEL_ID should exist and be normalized (alphanumeric, uppercase)
        self.assertIn('PARCEL_ID', out.columns)
        self.assertEqual(out['PARCEL_ID'].iloc[0], '791025200020000020')
        self.assertEqual(out['PARCEL_ID'].iloc[1], '791025200021000020')
        self.assertTrue(pd.isna(out['PARCEL_ID'].iloc[2]))


if __name__ == '__main__':
    unittest.main()

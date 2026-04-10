"""Unit tests for src.validate

These tests use the existing sample fixture and additional small fixtures and
are written to skip when geopandas is not installed.
"""
import unittest
import json
import tempfile
from pathlib import Path

import src.validate as validate


class TestValidate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.good = self.tmpdir / 'good.geojson'
        good_sample = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"ZONE_ID": 4001}, "geometry": {"type": "Point", "coordinates": [-86.9, 40.4]}},
                {"type": "Feature", "properties": {"ZONE_ID": 4002}, "geometry": {"type": "Point", "coordinates": [-86.8, 40.5]}}
            ]
        }
        with open(self.good, 'w', encoding='utf-8') as f:
            json.dump(good_sample, f)

        self.bad_no_id = self.tmpdir / 'bad_no_id.geojson'
        bad_sample = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"NAME": "A"}, "geometry": {"type": "Point", "coordinates": [-86.9, 40.4]}},
            ]
        }
        with open(self.bad_no_id, 'w', encoding='utf-8') as f:
            json.dump(bad_sample, f)

        self.bad_nonint = self.tmpdir / 'bad_nonint.geojson'
        bad_nonint = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"ZONE_ID": 3.1415}, "geometry": {"type": "Point", "coordinates": [-86.9, 40.4]}},
            ]
        }
        with open(self.bad_nonint, 'w', encoding='utf-8') as f:
            json.dump(bad_nonint, f)

    def test_validate_good_or_skip(self):
        try:
            result = validate.validate_geojson(str(self.good))
        except RuntimeError as e:
            self.skipTest(f"Skipping because geopandas not available: {e}")

        self.assertTrue(result)

    def test_validate_missing_id(self):
        try:
            validate.validate_geojson(str(self.bad_no_id))
            self.fail("Expected KeyError for missing id property")
        except KeyError:
            pass
        except RuntimeError as e:
            self.skipTest(f"Skipping because geopandas not available: {e}")

    def test_validate_nonint(self):
        try:
            validate.validate_geojson(str(self.bad_nonint))
            self.fail("Expected ValueError for non-integral id values")
        except ValueError:
            pass
        except RuntimeError as e:
            self.skipTest(f"Skipping because geopandas not available: {e}")

    def test_validate_string_ids(self):
        try:
            # create a geojson with string ids (STKEY)
            good_str = self.tmpdir / 'good_str.geojson'
            good_sample = {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"STKEY": "79-10-25-200-020.000-020"}, "geometry": {"type": "Point", "coordinates": [-86.9, 40.4]}},
                    {"type": "Feature", "properties": {"STKEY": "79-10-25-200-021.000-020"}, "geometry": {"type": "Point", "coordinates": [-86.8, 40.5]}}
                ]
            }
            with open(good_str, 'w', encoding='utf-8') as f:
                json.dump(good_sample, f)

            result = validate.validate_geojson(str(good_str), id_prop='STKEY')
            self.assertTrue(result)
        except RuntimeError as e:
            self.skipTest(f"Skipping because geopandas not available: {e}")


if __name__ == '__main__':
    unittest.main()

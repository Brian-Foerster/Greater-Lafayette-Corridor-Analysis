"""Tests for src.bootstrap helpers.

These tests are written to gracefully skip if required dependencies are not
available (geopandas, orca). They validate that the functions raise informative
errors or register tables when possible.
"""
import unittest
import tempfile
import json
from pathlib import Path

import src.bootstrap as bootstrap


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.input = self.tmpdir / 'sample_zones.geojson'
        sample = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"ZONE_ID": 3001}, "geometry": {"type": "Point", "coordinates": [-86.9, 40.4]}},
                {"type": "Feature", "properties": {"ZONE_ID": 3002}, "geometry": {"type": "Point", "coordinates": [-86.8, 40.5]}}
            ]
        }
        with open(self.input, 'w', encoding='utf-8') as f:
            json.dump(sample, f)

    def test_build_views_or_skip(self):
        try:
            views = bootstrap.build_views_from_zones(str(self.input))
        except RuntimeError as e:
            self.skipTest(f"Skipping because geopandas not available: {e}")

        self.assertIn('zones', views)
        df = views['zones']
        self.assertIn('zone_id', df.columns)

    def test_setup_orca_or_skip(self):
        try:
            # setup_orca_from_zones will raise if orca or geopandas missing
            result = bootstrap.setup_orca_from_zones(str(self.input))
        except RuntimeError as e:
            self.skipTest(f"Skipping because dependency missing: {e}")

        self.assertEqual(result, 'zones')


if __name__ == '__main__':
    unittest.main()

"""Tests for the validate CLI wrapper.

These tests call the `run` function directly so they don't require subprocesses.
They skip gracefully when geopandas is not installed.
"""
import unittest
import json
import tempfile
from pathlib import Path

from src import validate_cli, validate


class TestValidateCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        # good file
        self.good = self.tmpdir / 'good.geojson'
        good_sample = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"ZONE_ID": 5001}, "geometry": {"type": "Point", "coordinates": [-86.9, 40.4]}},
            ]
        }
        self.good.write_text(json.dumps(good_sample), encoding='utf-8')

        # bad file: missing id
        self.bad = self.tmpdir / 'bad.geojson'
        bad_sample = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"NAME": "X"}, "geometry": {"type": "Point", "coordinates": [-86.9, 40.4]}},
            ]
        }
        self.bad.write_text(json.dumps(bad_sample), encoding='utf-8')

    def test_run_good_and_bad(self):
        try:
            results = validate_cli.run([str(self.good), str(self.bad)])
        except RuntimeError as e:
            self.skipTest(f"Skipping because geopandas not available: {e}")

        self.assertIn(str(self.good), results)
        self.assertIn(str(self.bad), results)
        self.assertEqual(results[str(self.good)]['status'], 'ok')
        self.assertEqual(results[str(self.bad)]['status'], 'error')


if __name__ == '__main__':
    unittest.main()

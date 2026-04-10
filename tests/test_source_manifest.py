import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from src.source_manifest import (
    REQUIRED_SOURCE_MANIFEST_COLUMNS,
    load_source_manifest,
    validate_source_manifest_file,
)


class TestSourceManifest(unittest.TestCase):
    def test_manifest_file_exists_and_valid(self):
        path = Path("data/processed/source_manifest.csv")
        self.assertTrue(path.exists(), "source_manifest.csv is missing")

        df = validate_source_manifest_file(
            path,
            as_of_date=date(2026, 2, 21),
            max_source_age_days=3650,
        )
        for col in REQUIRED_SOURCE_MANIFEST_COLUMNS:
            self.assertIn(col, df.columns)
        self.assertGreater(len(df), 0)

    def test_missing_required_columns_fails(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "manifest.csv"
            pd.DataFrame(
                {
                    "dataset_id": ["x"],
                    "source_url": ["https://example.com"],
                    # missing required columns intentionally
                }
            ).to_csv(p, index=False)

            with self.assertRaises(ValueError):
                validate_source_manifest_file(p)

    def test_stale_required_source_fails(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "dataset_id": "required_old_source",
                        "source_url": "https://example.com/old",
                        "data_vintage": "2020",
                        "retrieval_date": "2000-01-01",
                        "refresh_cadence_days": 30,
                        "required_for_pipeline": True,
                        "status": "active",
                    }
                ]
            ).to_csv(p, index=False)

            with self.assertRaises(ValueError):
                validate_source_manifest_file(
                    p,
                    as_of_date=date(2026, 2, 21),
                    max_source_age_days=3650,
                )

    def test_stale_optional_source_does_not_fail(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "dataset_id": "optional_old_source",
                        "source_url": "https://example.com/old",
                        "data_vintage": "2020",
                        "retrieval_date": "2000-01-01",
                        "refresh_cadence_days": 30,
                        "required_for_pipeline": False,
                        "status": "active",
                    }
                ]
            ).to_csv(p, index=False)

            df = validate_source_manifest_file(
                p,
                as_of_date=date(2026, 2, 21),
                max_source_age_days=3650,
            )
            self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()

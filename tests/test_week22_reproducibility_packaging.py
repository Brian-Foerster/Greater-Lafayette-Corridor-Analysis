"""Week 22 tests for reproducibility metadata and artifact packaging."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from src.reproducibility import (
        build_run_id,
        check_required_artifacts,
        copy_artifacts_to_output_dir,
        initialize_run_directory,
        source_manifest_version_info,
        write_artifact_manifest,
        write_json,
    )

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    build_run_id = None
    check_required_artifacts = None
    copy_artifacts_to_output_dir = None
    initialize_run_directory = None
    source_manifest_version_info = None
    write_artifact_manifest = None
    write_json = None
    _IMPORT_ERROR = exc


class TestWeek22ReproducibilityPackaging(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_build_run_id_contains_trace_fields(self):
        run_id = build_run_id(
            workflow="uncertainty_sensitivity",
            timestamp_token="20260224T120000Z",
            git_commit="abc1234",
            scenario_ids=["zoning", "no_zoning"],
        )
        self.assertTrue(run_id.startswith("run_v1_uncertainty_sensitivity_20260224T120000Z_abc1234"))
        self.assertIn("scn_multi2", run_id)

    def test_source_manifest_version_info_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "source_manifest.csv"
            df = pd.DataFrame(
                {
                    "dataset_id": ["a", "b"],
                    "source_url": ["https://example.com/a", "https://example.com/b"],
                    "data_vintage": ["2024", "2025"],
                    "retrieval_date": ["2026-01-01", "2026-02-01"],
                    "refresh_cadence_days": [365, 365],
                    "required_for_pipeline": [True, True],
                    "status": ["active", "active"],
                }
            )
            df.to_csv(p, index=False)
            v1 = source_manifest_version_info(p)
            v2 = source_manifest_version_info(p)
            self.assertEqual(v1["source_manifest_sha256"], v2["source_manifest_sha256"])
            self.assertEqual(v1["source_manifest_version"], v2["source_manifest_version"])
            self.assertEqual(v1["source_manifest_latest_retrieval_date"], "2026-02-01")

    def test_required_artifact_completeness_check(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repro_runs"
            dirs = initialize_run_directory(run_root=root, run_id="run_v1_test")
            out = dirs["outputs_dir"]
            meta = dirs["metadata_dir"]

            # Create one source artifact and one metadata file.
            src_file = Path(td) / "result.csv"
            pd.DataFrame({"x": [1, 2]}).to_csv(src_file, index=False)
            write_json(meta / "run_metadata_v1.json", {"run_id": "run_v1_test"})

            copied = copy_artifacts_to_output_dir([src_file], out)
            self.assertEqual(len(copied), 1)
            out_file = copied[0]

            run_meta = {
                "run_id": "run_v1_test",
                "workflow": "uncertainty_sensitivity",
                "scenario_ids": ["zoning"],
                "git_commit": "abc1234",
                "source_manifest_version": "manifest_v1_x",
            }
            manifest_path = write_artifact_manifest(
                run_dir=dirs["run_dir"],
                artifact_paths=[out_file],
                run_metadata=run_meta,
            )
            self.assertTrue(manifest_path.exists())

            check_ok = check_required_artifacts(
                run_dir=dirs["run_dir"],
                required_relative_paths=["outputs/result.csv"],
            )
            self.assertTrue(check_ok["is_complete"])

            check_bad = check_required_artifacts(
                run_dir=dirs["run_dir"],
                required_relative_paths=["outputs/result.csv", "outputs/missing.csv"],
            )
            self.assertFalse(check_bad["is_complete"])
            self.assertEqual(check_bad["missing_count"], 1)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
"""Week 21 Monte Carlo uncertainty runner for corridor finance outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import logging
logger = logging.getLogger(__name__)

from scripts.apm_corridor_evaluation_integrated import compute_uncertainty_bands
from src.reproducibility import (
    build_run_id,
    check_required_artifacts,
    copy_artifacts_to_output_dir,
    get_git_commit_short,
    initialize_run_directory,
    source_manifest_version_info,
    utc_timestamp_compact,
    write_artifact_manifest,
    write_json,
)


def _default_inputs() -> list[Path]:
    return [
        Path("data/processed/corridors_integrated_financial_current_zoning.csv"),
        Path("data/processed/corridors_integrated_financial_no_zoning.csv"),
    ]


def _parse_percentiles(value: str | None):
    if not value:
        return None
    parts = [v.strip() for v in str(value).split(",") if v.strip()]
    out = []
    for token in parts:
        out.append(float(token))
    return out


def _package_reproducible_run(
    *,
    produced_paths: list[Path],
    config_path: Path,
    source_manifest_path: Path,
    scenario_ids: list[str],
    run_root: Path,
) -> dict:
    """Create standardized Week 22 run package with metadata and manifests."""
    if not source_manifest_path.exists():
        raise FileNotFoundError(
            f"Source manifest not found for reproducibility packaging: {source_manifest_path}"
        )

    manifest_info = source_manifest_version_info(source_manifest_path)
    timestamp_token = utc_timestamp_compact()
    git_commit = get_git_commit_short()
    run_id = build_run_id(
        workflow="uncertainty_sensitivity",
        timestamp_token=timestamp_token,
        git_commit=git_commit,
        scenario_ids=scenario_ids,
    )
    dirs = initialize_run_directory(run_root=run_root, run_id=run_id)

    # Preserve key provenance inputs in the run package.
    for p in [source_manifest_path, config_path]:
        if p.exists() and p.is_file():
            dst = dirs["inputs_dir"] / p.name
            dst.write_bytes(p.read_bytes())

    copied_outputs = copy_artifacts_to_output_dir(
        artifact_paths=produced_paths,
        output_dir=dirs["outputs_dir"],
    )

    run_metadata = {
        "workflow": "uncertainty_sensitivity",
        "version": "v1",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_token": timestamp_token,
        "git_commit": git_commit,
        "scenario_ids": sorted({str(s) for s in scenario_ids if str(s).strip()}),
        "config_path": str(config_path),
        **manifest_info,
    }

    run_meta_path = write_json(dirs["metadata_dir"] / "run_metadata_v1.json", run_metadata)
    manifest_path = write_artifact_manifest(
        run_dir=dirs["run_dir"],
        artifact_paths=copied_outputs,
        run_metadata=run_metadata,
        manifest_filename="artifact_manifest_v1.csv",
    )

    required_relpaths = [f"outputs/{p.name}" for p in copied_outputs]
    required_check = check_required_artifacts(
        run_dir=dirs["run_dir"],
        required_relative_paths=required_relpaths,
    )
    required_path = write_json(
        dirs["metadata_dir"] / "required_artifacts_check_v1.json",
        required_check,
    )

    return {
        "run_id": run_id,
        "run_dir": dirs["run_dir"],
        "run_metadata_path": run_meta_path,
        "artifact_manifest_path": manifest_path,
        "required_check_path": required_path,
        "is_complete": bool(required_check.get("is_complete", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 21 uncertainty/sensitivity analysis")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="*",
        help="Input corridor finance CSV(s). Defaults to integrated zoning and no-zoning outputs.",
    )
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default="scenarios_config.json")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--percentiles", type=str, default=None, help="Comma-separated list, e.g. 5,50,95")
    parser.add_argument("--years", type=int, default=25)
    parser.add_argument("--fare-per-trip", type=float, default=2.00)
    parser.add_argument("--farebox-capture-rate", type=float, default=1.0)
    parser.add_argument("--discount-rate", type=float, default=0.05)
    parser.add_argument(
        "--run-root",
        type=str,
        default="data/processed/repro_runs",
        help="Week 22 standardized reproducible-run root folder",
    )
    parser.add_argument(
        "--source-manifest",
        type=str,
        default="data/processed/source_manifest.csv",
        help="Source manifest used to compute run source-vintage version",
    )
    parser.add_argument(
        "--skip-packaging",
        action="store_true",
        help="Skip Week 22 reproducible artifact packaging",
    )

    args = parser.parse_args()

    if args.inputs:
        input_paths = [Path(p) for p in args.inputs]
    else:
        input_paths = _default_inputs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    source_manifest_path = Path(args.source_manifest)
    run_root = Path(args.run_root)

    summary_rows = []
    scenario_ids_seen: set[str] = set()
    produced_paths: list[Path] = []
    for path in input_paths:
        if not path.exists():
            logger.info(f"[SKIP] Missing input file: {path}")
            continue

        df = pd.read_csv(path)
        if df.empty:
            logger.info(f"[SKIP] Empty input file: {path}")
            continue
        if "corridor_id" not in df.columns:
            logger.info(f"[SKIP] Missing required column corridor_id in: {path}")
            continue
        if "scenario" not in df.columns:
            if "no_zoning" in path.stem.lower():
                df["scenario"] = "no_zoning"
            elif "zoning" in path.stem.lower():
                df["scenario"] = "zoning"
            else:
                df["scenario"] = "unknown"
        scenario_values = sorted({str(v) for v in df["scenario"].dropna().astype(str).tolist()})
        scenario_ids_seen.update(scenario_values)

        wide_df, long_df, draws_df, meta = compute_uncertainty_bands(
            df,
            config_path=config_path,
            cashflow_years=args.years,
            fare_per_trip_usd=args.fare_per_trip,
            farebox_capture_rate=args.farebox_capture_rate,
            discount_rate=args.discount_rate,
            n_draws=args.draws,
            random_seed=args.seed,
            percentiles=_parse_percentiles(args.percentiles),
        )

        merged = df.merge(wide_df, on=["corridor_id", "scenario"], how="left")
        stem = path.stem
        merged_path = output_dir / f"{stem}_with_uncertainty.csv"
        wide_path = output_dir / f"{stem}_uncertainty_bands.csv"
        long_path = output_dir / f"{stem}_uncertainty_percentiles_long.csv"
        draws_path = output_dir / f"{stem}_uncertainty_draws.csv"
        meta_path = output_dir / f"{stem}_uncertainty_metadata.json"

        merged.to_csv(merged_path, index=False)
        wide_df.to_csv(wide_path, index=False)
        long_df.to_csv(long_path, index=False)
        draws_df.to_csv(draws_path, index=False)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        produced_paths.extend([merged_path, wide_path, long_path, draws_path, meta_path])

        summary_rows.append(
            {
                "input_file": str(path),
                "scenario_ids": ",".join(scenario_values),
                "corridors": int(len(df)),
                "draws": int(meta.get("n_draws", 0)),
                "output_with_uncertainty": str(merged_path),
                "output_bands": str(wide_path),
                "output_percentiles_long": str(long_path),
                "output_draws": str(draws_path),
                "output_metadata": str(meta_path),
            }
        )
        logger.info(f"[DONE] {path.name}: uncertainty outputs written.")

    if not summary_rows:
        logger.info("No uncertainty outputs produced.")
        return

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "week21_uncertainty_run_summary.csv"
    summary.to_csv(summary_path, index=False)
    produced_paths.append(summary_path)
    logger.info(f"[DONE] Summary written: {summary_path}")

    if not args.skip_packaging:
        pkg = _package_reproducible_run(
            produced_paths=produced_paths,
            config_path=config_path,
            source_manifest_path=source_manifest_path,
            scenario_ids=sorted(scenario_ids_seen),
            run_root=run_root,
        )
        logger.info(f"[DONE] Week 22 package run_id: {pkg['run_id']}")
        logger.info(f"[DONE] Reproducible run directory: {pkg['run_dir']}")
        logger.info(f"[DONE] Artifact completeness: {'PASS' if pkg['is_complete'] else 'FAIL'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

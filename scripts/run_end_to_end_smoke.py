#!/usr/bin/env python
"""Week 24 end-to-end integration smoke target."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import logging
logger = logging.getLogger(__name__)



def _run_cmd(cmd: List[str]) -> dict:
    started = time.time()
    proc = subprocess.run(cmd, check=False)
    duration = time.time() - started
    return {
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "duration_seconds": float(round(duration, 3)),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_md(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Week 24 End-to-End Smoke Summary")
    lines.append("")
    lines.append(f"- Generated UTC: `{summary['generated_at_utc']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Commands: `{len(summary.get('commands', []))}`")
    lines.append(f"- Failing commands: `{summary['failing_command_count']}`")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    for row in summary.get("commands", []):
        status = "PASS" if row["returncode"] == 0 else "FAIL"
        lines.append(f"- `{status}` `{row['command']}` ({row['duration_seconds']}s)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week 24 end-to-end smoke target")
    parser.add_argument(
        "--summary-json",
        type=str,
        default="data/processed/week24_e2e_smoke_summary.json",
    )
    parser.add_argument(
        "--summary-md",
        type=str,
        default="data/processed/week24_e2e_smoke_summary.md",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=60,
        help="Uncertainty draws for smoke run (kept small for CI)",
    )
    parser.add_argument(
        "--kpi-spec",
        type=str,
        default="data/processed/kpi_baseline_week24.json",
    )
    args = parser.parse_args()

    commands = []

    # 1) Run uncertainty + reproducibility package smoke.
    commands.append(
        [
            sys.executable,
            "scripts/run_uncertainty_sensitivity.py",
            "--draws",
            str(int(args.draws)),
            "--seed",
            "24",
            "--percentiles",
            "10,50,90",
            "--output-dir",
            "data/processed",
            "--run-root",
            "data/processed/repro_runs",
        ]
    )

    # 2) Run geometry-change detector against recent commit diff.
    commands.append(
        [
            sys.executable,
            "scripts/geometry_change_detector.py",
            "--base-ref",
            "HEAD~1",
            "--head-ref",
            "HEAD",
            "--output-json",
            "data/processed/geometry_change_detection.json",
            "--output-csv",
            "data/processed/geometry_change_detection_files.csv",
        ]
    )

    # 3) Run geometry rerun smoke (lightweight test subset to keep CI stable).
    commands.append(
        [
            sys.executable,
            "scripts/run_geometry_rerun_smoke.py",
            "--reason-file",
            "data/processed/geometry_change_detection.json",
            "--summary-json",
            "data/processed/geometry_rerun_smoke_summary.json",
            "--summary-md",
            "data/processed/geometry_rerun_smoke_summary.md",
            "--tests",
            "tests.test_geometry_change_detector",
            "tests.test_geometry_rerun_smoke",
            "tests.test_week21_uncertainty_framework",
            "tests.test_week22_reproducibility_packaging",
        ]
    )

    # 4) Tight KPI drift gate.
    commands.append(
        [
            sys.executable,
            "scripts/check_kpi_drift.py",
            "--spec",
            str(args.kpi_spec),
            "--output-json",
            "data/processed/kpi_drift_summary.json",
            "--output-csv",
            "data/processed/kpi_drift_summary.csv",
            "--summary-md",
            "data/processed/kpi_drift_summary.md",
        ]
    )

    results = []
    for cmd in commands:
        results.append(_run_cmd(cmd))

    failing = sum(1 for r in results if r["returncode"] != 0)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if failing == 0 else "fail",
        "failing_command_count": int(failing),
        "commands": results,
    }

    summary_json = Path(args.summary_json)
    summary_md = Path(args.summary_md)
    _write_json(summary_json, summary)
    _write_md(summary_md, summary)

    logger.info(f"[DONE] Week 24 smoke summary JSON: {summary_json}")
    logger.info(f"[DONE] Week 24 smoke summary MD: {summary_md}")
    if failing:
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())

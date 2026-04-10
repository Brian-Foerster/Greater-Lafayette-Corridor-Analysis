#!/usr/bin/env python
"""Run a lightweight CI smoke target for geometry-triggered reruns."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import logging
logger = logging.getLogger(__name__)



DEFAULT_TEST_MODULES = [
    "tests.test_geometry_change_detector",
    "tests.test_evaluation_batch_artifacts",
    "tests.test_network_validation_gate",
    "tests.test_week20_dynamic_finance",
    "tests.test_week21_uncertainty_framework",
    "tests.test_week22_reproducibility_packaging",
]


def _run_cmd(cmd: list[str]) -> dict:
    started = time.time()
    proc = subprocess.run(cmd, check=False)
    duration = time.time() - started
    return {
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "duration_seconds": float(round(duration, 3)),
    }


def _load_reason(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_md(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Geometry Rerun Smoke Summary")
    lines.append("")
    lines.append(f"- Generated UTC: `{summary['generated_at_utc']}`")
    lines.append(f"- Overall status: `{summary['status']}`")
    lines.append(f"- Failing commands: `{summary['failing_command_count']}`")
    if summary.get("reason"):
        lines.append(f"- Geometry changed: `{summary['reason'].get('geometry_changed')}`")
        lines.append(f"- Geometry files: `{summary['reason'].get('geometry_changed_file_count')}`")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    for row in summary.get("commands", []):
        status = "PASS" if row["returncode"] == 0 else "FAIL"
        lines.append(f"- `{status}` `{row['command']}` ({row['duration_seconds']}s)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry rerun smoke target")
    parser.add_argument(
        "--reason-file",
        type=str,
        default=None,
        help="Optional geometry-change detection JSON for traceability",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="data/processed/geometry_rerun_smoke_summary.json",
    )
    parser.add_argument(
        "--summary-md",
        type=str,
        default="data/processed/geometry_rerun_smoke_summary.md",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        default=None,
        help="Optional unittest module list override",
    )
    args = parser.parse_args()

    test_modules = list(args.tests) if args.tests else list(DEFAULT_TEST_MODULES)
    cmd = [sys.executable, "-m", "unittest", "-v", *test_modules]
    result = _run_cmd(cmd)

    reason = _load_reason(Path(args.reason_file)) if args.reason_file else {}
    failing = 0 if result["returncode"] == 0 else 1
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if failing == 0 else "fail",
        "failing_command_count": int(failing),
        "reason": reason,
        "commands": [result],
    }

    summary_json = Path(args.summary_json)
    summary_md = Path(args.summary_md)
    _write_json(summary_json, summary)
    _write_md(summary_md, summary)

    logger.info(f"[DONE] Smoke summary JSON: {summary_json}")
    logger.info(f"[DONE] Smoke summary MD: {summary_md}")
    if failing:
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

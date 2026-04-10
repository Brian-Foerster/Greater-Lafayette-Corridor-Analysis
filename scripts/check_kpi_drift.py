#!/usr/bin/env python
"""Week 24 KPI drift gate for final QA hardening."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)



def _load_spec(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"KPI drift spec not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "metrics" not in payload or not isinstance(payload["metrics"], list):
        raise ValueError("Spec must include metrics list")
    return payload


def _compute_agg(series: pd.Series, agg: str) -> float:
    agg_norm = str(agg).strip().lower()
    s = pd.to_numeric(series, errors="coerce").dropna()
    if agg_norm == "mean":
        return float(s.mean())
    if agg_norm == "median":
        return float(s.median())
    if agg_norm == "sum":
        return float(s.sum())
    if agg_norm == "min":
        return float(s.min())
    if agg_norm == "max":
        return float(s.max())
    if agg_norm == "count":
        return float(s.count())
    raise ValueError(f"Unsupported aggregation: {agg}")


def _resolve_series(df: pd.DataFrame, metric: Dict[str, Any]) -> pd.Series:
    if "column" in metric:
        col = str(metric["column"])
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        if str(metric.get("type", "")).lower() == "bool_rate":
            raw = df[col].astype(str).str.strip().str.lower()
            mapped = raw.map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
            return pd.to_numeric(mapped, errors="coerce")
        return pd.to_numeric(df[col], errors="coerce")

    if str(metric.get("expression", "")).lower() == "row_count":
        return pd.Series([float(len(df))], dtype=float)

    raise ValueError(f"Metric must define column or supported expression: {metric.get('id', '<unknown>')}")


def _apply_where(df: pd.DataFrame, metric: Dict[str, Any]) -> pd.DataFrame:
    where = metric.get("where")
    if where is None:
        return df
    if not isinstance(where, dict):
        raise ValueError(f"'where' must be a dict for metric: {metric.get('id', '<unknown>')}")

    filtered = df
    for col, expected in where.items():
        if col not in filtered.columns:
            raise ValueError(f"Missing where-column '{col}' for metric: {metric.get('id', '<unknown>')}")
        if isinstance(expected, list):
            filtered = filtered[filtered[col].isin(expected)]
        else:
            filtered = filtered[filtered[col] == expected]
    return filtered


def evaluate_metric(metric: Dict[str, Any]) -> Dict[str, Any]:
    metric_id = str(metric.get("id", "<unknown>"))
    path = Path(metric["path"])
    if not path.exists():
        return {
            "id": metric_id,
            "status": "FAIL",
            "reason": f"Missing input file: {path}",
            "path": str(path),
        }

    df = pd.read_csv(path)
    df = _apply_where(df, metric)
    series = _resolve_series(df, metric)
    agg = str(metric.get("agg", "mean"))
    current = _compute_agg(series, agg)
    baseline = float(metric["baseline"])

    rel_drift = abs(current - baseline) / max(abs(baseline), 1e-12)
    abs_drift = abs(current - baseline)

    max_rel = metric.get("max_rel_drift")
    max_abs = metric.get("max_abs_drift")
    pass_rel = True if max_rel is None else rel_drift <= float(max_rel)
    pass_abs = True if max_abs is None else abs_drift <= float(max_abs)
    is_pass = bool(pass_rel and pass_abs)

    return {
        "id": metric_id,
        "status": "PASS" if is_pass else "FAIL",
        "path": str(path),
        "agg": agg,
        "baseline": baseline,
        "current": float(current),
        "abs_drift": float(abs_drift),
        "rel_drift": float(rel_drift),
        "max_abs_drift": None if max_abs is None else float(max_abs),
        "max_rel_drift": None if max_rel is None else float(max_rel),
        "note": str(metric.get("note", "")),
    }


def run_drift_gate(spec_path: Path) -> Dict[str, Any]:
    spec = _load_spec(spec_path)
    rows: List[Dict[str, Any]] = []
    for metric in spec.get("metrics", []):
        rows.append(evaluate_metric(metric))
    fail_count = sum(1 for r in rows if r.get("status") != "PASS")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec_path": str(spec_path),
        "version": str(spec.get("version", "v1")),
        "metric_count": len(rows),
        "fail_count": int(fail_count),
        "status": "PASS" if fail_count == 0 else "FAIL",
        "results": rows,
    }


def _write_outputs(summary: Dict[str, Any], json_path: Path, csv_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = pd.DataFrame(summary.get("results", []))
    rows.to_csv(csv_path, index=False)

    lines = [
        "# KPI Drift Summary",
        "",
        f"- Generated UTC: `{summary['generated_at_utc']}`",
        f"- Status: `{summary['status']}`",
        f"- Metrics: `{summary['metric_count']}`",
        f"- Fails: `{summary['fail_count']}`",
        "",
        "| Metric | Status | Current | Baseline | Abs Drift | Rel Drift | Max Abs | Max Rel |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("results", []):
        cur = row.get("current")
        base = row.get("baseline")
        abs_d = row.get("abs_drift")
        rel_d = row.get("rel_drift")
        max_abs = row.get("max_abs_drift")
        max_rel = row.get("max_rel_drift")
        lines.append(
            f"| {row.get('id')} | {row.get('status')} | "
            f"{'n/a' if cur is None else f'{float(cur):.6g}'} | "
            f"{'n/a' if base is None else f'{float(base):.6g}'} | "
            f"{'n/a' if abs_d is None else f'{float(abs_d):.6g}'} | "
            f"{'n/a' if rel_d is None else f'{float(rel_d):.6g}'} | "
            f"{'n/a' if max_abs is None else f'{float(max_abs):.6g}'} | "
            f"{'n/a' if max_rel is None else f'{float(max_rel):.6g}'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check KPI drift against Week 24 thresholds")
    parser.add_argument(
        "--spec",
        type=str,
        default="data/processed/kpi_baseline_week24.json",
        help="KPI baseline spec JSON path",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/processed/kpi_drift_summary.json",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/processed/kpi_drift_summary.csv",
    )
    parser.add_argument(
        "--summary-md",
        type=str,
        default="data/processed/kpi_drift_summary.md",
    )
    args = parser.parse_args()

    summary = run_drift_gate(Path(args.spec))
    _write_outputs(
        summary,
        json_path=Path(args.output_json),
        csv_path=Path(args.output_csv),
        md_path=Path(args.summary_md),
    )

    logger.info(
        f"[DONE] KPI drift gate: {summary['status']} "
        f"({summary['metric_count']} metrics, {summary['fail_count']} fails)"
    )
    logger.info(f"[DONE] Summary JSON: {args.output_json}")
    logger.info(f"[DONE] Summary CSV: {args.output_csv}")
    logger.info(f"[DONE] Summary MD: {args.summary_md}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())

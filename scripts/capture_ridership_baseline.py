"""Capture pre/post-fix ridership baseline diagnostics.

Writes machine-readable and human-readable snapshots for comparison.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pandas as pd

import logging
logger = logging.getLogger(__name__)



def _safe_ratio(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def build_snapshot(data_dir: Path) -> Dict[str, Any]:
    feedback = pd.read_csv(data_dir / "feedback_loop_results.csv")
    integrated = pd.read_csv(data_dir / "corridors_integrated_financial_current_zoning.csv")
    full = pd.read_csv(data_dir / "FULL_evaluation_50M_all40.csv")
    phase2b = pd.read_csv(data_dir / "phase2b_corridor_results.csv")

    y25 = feedback.loc[feedback["year"] == 25, ["corridor_id", "daily_riders"]].copy()
    compare = (
        phase2b[["corridor_id", "phase2b_daily_riders"]]
        .merge(y25, on="corridor_id", how="left")
        .merge(full[["corridor_id", "daily_ridership"]], on="corridor_id", how="left")
    )
    compare["ratio_y25_vs_phase2b"] = compare.apply(
        lambda r: _safe_ratio(r["daily_riders"], r["phase2b_daily_riders"]),
        axis=1,
    )
    compare["ratio_effective_vs_phase2b"] = compare.apply(
        lambda r: _safe_ratio(r["daily_ridership"], r["phase2b_daily_riders"]),
        axis=1,
    )

    y25_feedback = feedback.loc[feedback["year"] == 25].copy()

    out: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": {
            "feedback_loop_results": int(len(feedback)),
            "corridors_integrated_financial_current_zoning": int(len(integrated)),
            "FULL_evaluation_50M_all40": int(len(full)),
            "phase2b_corridor_results": int(len(phase2b)),
        },
        "feedback_ridership": {
            "year0_mean_daily_riders": float(feedback.loc[feedback["year"] == 0, "daily_riders"].mean()),
            "year25_mean_daily_riders": float(y25_feedback["daily_riders"].mean()),
            "year25_max_daily_riders": float(y25_feedback["daily_riders"].max()),
            "effective_mean_daily_riders_used_in_finance": float(full["daily_ridership"].mean()),
        },
        "dynamic_vs_static_ratios": {
            "mean_y25_vs_phase2b": float(compare["ratio_y25_vs_phase2b"].mean()),
            "median_y25_vs_phase2b": float(compare["ratio_y25_vs_phase2b"].median()),
            "mean_effective_vs_phase2b": float(compare["ratio_effective_vs_phase2b"].mean()),
            "median_effective_vs_phase2b": float(compare["ratio_effective_vs_phase2b"].median()),
            "min_effective_vs_phase2b": float(compare["ratio_effective_vs_phase2b"].min()),
            "max_effective_vs_phase2b": float(compare["ratio_effective_vs_phase2b"].max()),
        },
        "fallback_pattern_indicators": {
            "year25_unique_commute_mode_share_values": int(y25_feedback["commute_mode_share"].nunique()),
            "year25_unique_directional_fraction_values": int(y25_feedback["directional_fraction"].nunique()),
            "year25_commute_mode_share_values": sorted(float(v) for v in y25_feedback["commute_mode_share"].unique()),
            "year25_directional_fraction_values": sorted(float(v) for v in y25_feedback["directional_fraction"].unique()),
        },
        "development_activity": {
            "total_new_units": float(feedback["new_units"].sum()),
            "total_new_jobs": float(feedback["new_jobs"].sum()),
            "total_new_pop": float(feedback["new_pop"].sum()),
            "nonzero_new_units_rows": int((feedback["new_units"] > 0).sum()),
        },
        "finance_summary_50m": {
            "mean_debt_coverage_ratio": float(full["debt_coverage_ratio"].mean()),
            "viable_count": int((full["financially_viable"] == True).sum()),
            "top5_dcr": full.nlargest(5, "debt_coverage_ratio")[
                ["corridor_id", "debt_coverage_ratio", "daily_ridership"]
            ].to_dict(orient="records"),
        },
    }
    return out


def write_markdown(snapshot: Dict[str, Any], path: Path) -> None:
    rows = snapshot["rows"]
    fr = snapshot["feedback_ridership"]
    ratios = snapshot["dynamic_vs_static_ratios"]
    fb = snapshot["fallback_pattern_indicators"]
    dev = snapshot["development_activity"]
    fin = snapshot["finance_summary_50m"]

    lines = [
        "# Ridership Baseline Snapshot",
        "",
        f"- Generated UTC: `{snapshot['generated_at_utc']}`",
        "",
        "## Input Rows",
        "",
        f"- feedback_loop_results: `{rows['feedback_loop_results']}`",
        f"- corridors_integrated_financial_current_zoning: `{rows['corridors_integrated_financial_current_zoning']}`",
        f"- FULL_evaluation_50M_all40: `{rows['FULL_evaluation_50M_all40']}`",
        f"- phase2b_corridor_results: `{rows['phase2b_corridor_results']}`",
        "",
        "## Ridership Levels",
        "",
        f"- Year 0 mean daily riders: `{fr['year0_mean_daily_riders']:.2f}`",
        f"- Year 25 mean daily riders: `{fr['year25_mean_daily_riders']:.2f}`",
        f"- Year 25 max daily riders: `{fr['year25_max_daily_riders']:.2f}`",
        f"- Effective mean daily riders in finance: `{fr['effective_mean_daily_riders_used_in_finance']:.2f}`",
        "",
        "## Dynamic vs Static Ratios",
        "",
        f"- Mean Year25/Phase2B: `{ratios['mean_y25_vs_phase2b']:.6f}`",
        f"- Median Year25/Phase2B: `{ratios['median_y25_vs_phase2b']:.6f}`",
        f"- Mean Effective/Phase2B: `{ratios['mean_effective_vs_phase2b']:.6f}`",
        f"- Median Effective/Phase2B: `{ratios['median_effective_vs_phase2b']:.6f}`",
        f"- Min Effective/Phase2B: `{ratios['min_effective_vs_phase2b']:.6f}`",
        f"- Max Effective/Phase2B: `{ratios['max_effective_vs_phase2b']:.6f}`",
        "",
        "## Fallback Indicators",
        "",
        f"- Unique Year25 commute_mode_share values: `{fb['year25_unique_commute_mode_share_values']}`",
        f"- Unique Year25 directional_fraction values: `{fb['year25_unique_directional_fraction_values']}`",
        f"- Year25 commute_mode_share values: `{fb['year25_commute_mode_share_values']}`",
        f"- Year25 directional_fraction values: `{fb['year25_directional_fraction_values']}`",
        "",
        "## Development Activity",
        "",
        f"- Total new units: `{dev['total_new_units']:.2f}`",
        f"- Total new jobs: `{dev['total_new_jobs']:.2f}`",
        f"- Total new pop: `{dev['total_new_pop']:.2f}`",
        f"- Rows with nonzero new_units: `{dev['nonzero_new_units_rows']}`",
        "",
        "## Finance Summary (50M/km)",
        "",
        f"- Mean debt coverage ratio: `{fin['mean_debt_coverage_ratio']:.6f}`",
        f"- Viable corridors: `{fin['viable_count']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data_dir = Path("data/processed")
    snapshot = build_snapshot(data_dir)

    json_path = data_dir / "ridership_baseline_snapshot.json"
    md_path = data_dir / "ridership_baseline_snapshot.md"

    json_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    write_markdown(snapshot, md_path)

    logger.info(f"[DONE] Baseline JSON: {json_path}")
    logger.info(f"[DONE] Baseline MD: {md_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

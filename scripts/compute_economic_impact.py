"""Compute economic impact analysis for all APM corridors.

Reads feedback loop results and corridor metadata, runs the full CBA
framework (USDOT BCA + fiscal impact + economic activity + FTA metrics),
and outputs CSV summaries and taxpayer-facing narratives.

Usage:
    python scripts/compute_economic_impact.py \\
        --scenario current_zoning \\
        --feedback-results data/processed/feedback_loop_results.csv \\
        --corridor-metadata data/processed/apm_optimized_search_results.csv \\
        --output-dir data/processed/decision_package/ \\
        [--monte-carlo] [--mc-draws 500]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

from src.economic_impact import (
    compute_corridor_cba,
    format_taxpayer_summary,
    CorridorCBAResult,
)


def load_corridor_metadata(path: str) -> pd.DataFrame:
    """Load corridor search results CSV with length_km, n_stops."""
    df = pd.read_csv(path)
    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("id", "corridor_id"):
            col_map[c] = "corridor_id"
        elif cl in ("length", "length_km"):
            col_map[c] = "length_km"
        elif cl in ("stops", "n_stops", "stations"):
            col_map[c] = "n_stops"
    df = df.rename(columns=col_map)

    # Parse length_km if it has units like "10.1km"
    if df["length_km"].dtype == object:
        df["length_km"] = df["length_km"].str.replace("km", "").astype(float)

    return df


def main():
    parser = argparse.ArgumentParser(description="Compute economic impact for APM corridors")
    parser.add_argument("--scenario", default="current_zoning", help="Scenario name")
    parser.add_argument(
        "--feedback-results",
        default="data/processed/feedback_loop_results.csv",
        help="Feedback loop results CSV",
    )
    parser.add_argument(
        "--corridor-metadata",
        default="data/processed/apm_optimized_search_results.csv",
        help="Corridor search results CSV",
    )
    parser.add_argument(
        "--tif-results",
        default=None,
        help="Optional TIF revenue CSV (with corridor_id, cumulative_tif_revenue)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/decision_package",
        help="Output directory",
    )
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo uncertainty")
    parser.add_argument("--mc-draws", type=int, default=500, help="Number of MC draws")
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading feedback loop results: {args.feedback_results}")
    results_df = pd.read_csv(args.feedback_results)

    logger.info(f"Loading corridor metadata: {args.corridor_metadata}")
    meta_df = load_corridor_metadata(args.corridor_metadata)

    # Load TIF results if provided
    tif_lookup = {}
    if args.tif_results and os.path.exists(args.tif_results):
        logger.info(f"Loading TIF results: {args.tif_results}")
        tif_df = pd.read_csv(args.tif_results)
        for _, row in tif_df.iterrows():
            cid = str(row.get("corridor_id", ""))
            tif_lookup[cid] = float(row.get("cumulative_tif_revenue", 0))

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine corridor IDs
    corridor_ids = sorted(results_df["corridor_id"].unique())
    logger.info(f"\nProcessing {len(corridor_ids)} corridors for scenario '{args.scenario}'")
    logger.info(f"Monte Carlo: {'Yes' if args.monte_carlo else 'No'} ({args.mc_draws} draws)")
    logger.info()

    # Run CBA for each corridor
    all_results: list[CorridorCBAResult] = []

    for cid in corridor_ids:
        cid_str = str(cid)
        corridor_data = results_df[results_df["corridor_id"] == cid].copy()

        if corridor_data.empty:
            logger.debug(f"  {cid_str}: no feedback loop data, skipping")
            continue

        # Get metadata
        meta_row = meta_df[meta_df["corridor_id"] == cid_str]
        if meta_row.empty:
            # Try matching without prefix
            meta_row = meta_df[meta_df["corridor_id"].astype(str) == cid_str]
        if meta_row.empty:
            logger.debug(f"  {cid_str}: no metadata found, skipping")
            continue

        length_km = float(meta_row.iloc[0]["length_km"])
        n_stations = int(meta_row.iloc[0]["n_stops"])
        tif_usd = tif_lookup.get(cid_str, 0.0)

        # Ensure 'year' column exists
        if "year" not in corridor_data.columns:
            corridor_data["year"] = range(len(corridor_data))

        cba = compute_corridor_cba(
            results_df=corridor_data,
            corridor_id=cid_str,
            scenario=args.scenario,
            length_km=length_km,
            n_stations=n_stations,
            tif_cumulative_usd=tif_usd,
            run_monte_carlo=args.monte_carlo,
            n_mc_draws=args.mc_draws,
        )
        all_results.append(cba)

        logger.debug(
            f"  {cid_str}: BCR={cba.bcr_3pct:.2f} (3%) / {cba.bcr_7pct:.2f} (7%), "
            f"VMT avoided={cba.annual_vmt_avoided_miles:,.0f}, "
            f"jobs={cba.construction_jobs_apm + cba.permanent_tod_jobs:,.0f}"
        )

    if not all_results:
        logger.info("\nNo corridors processed. Check input data.")
        return

    # --- Output CSVs ---
    logger.info(f"\nWriting outputs to {args.output_dir}/")

    # 1. Economic impact summary (one row per corridor)
    summary_path = os.path.join(args.output_dir, f"economic_impact_summary_{args.scenario}.csv")
    summary_df = pd.DataFrame([r.to_dict() for r in all_results])
    summary_df.to_csv(summary_path, index=False)
    logger.debug(f"  {summary_path}")

    # 2. Taxpayer summaries for top-5 corridors (by BCR)
    sorted_results = sorted(all_results, key=lambda r: r.bcr_3pct, reverse=True)
    summary_lines = []
    for r in sorted_results[:5]:
        summary_lines.append(format_taxpayer_summary(r))
        summary_lines.append("")

    taxpayer_path = os.path.join(args.output_dir, f"taxpayer_summary_{args.scenario}.txt")
    with open(taxpayer_path, "w") as f:
        f.write("\n".join(summary_lines))
    logger.debug(f"  {taxpayer_path}")

    # 3. BCR comparison table
    bcr_rows = []
    for r in sorted_results:
        bcr_rows.append({
            "corridor_id": r.corridor_id,
            "bcr_3pct": r.bcr_3pct,
            "bcr_7pct": r.bcr_7pct,
            "bcr_3pct_p10": r.bcr_3pct_p10,
            "bcr_3pct_p90": r.bcr_3pct_p90,
            "total_benefits_3pct_musd": r.total_benefits_3pct,
            "total_costs_3pct_musd": r.total_costs_3pct,
            "leverage_ratio": r.leverage_ratio_gross,
            "fta_cost_per_trip": r.fta_cost_per_trip,
        })
    bcr_path = os.path.join(args.output_dir, f"bcr_comparison_{args.scenario}.csv")
    pd.DataFrame(bcr_rows).to_csv(bcr_path, index=False)
    logger.debug(f"  {bcr_path}")

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info(f"ECONOMIC IMPACT ANALYSIS — {args.scenario}")
    logger.info(f"{'='*60}")
    logger.debug(f"  Corridors analyzed: {len(all_results)}")
    bcr_vals = [r.bcr_3pct for r in all_results]
    logger.debug(f"  BCR range (3%): {min(bcr_vals):.2f} - {max(bcr_vals):.2f}")
    bcr_7_vals = [r.bcr_7pct for r in all_results]
    logger.debug(f"  BCR range (7%): {min(bcr_7_vals):.2f} - {max(bcr_7_vals):.2f}")
    top = sorted_results[0]
    logger.debug(f"  Best corridor: {top.corridor_id} (BCR={top.bcr_3pct:.2f})")
    total_jobs = sum(r.construction_jobs_apm + r.permanent_tod_jobs for r in all_results)
    logger.debug(f"  Total jobs across all corridors: {total_jobs:,.0f}")
    logger.info()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

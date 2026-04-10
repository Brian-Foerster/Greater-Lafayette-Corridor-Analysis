"""Full evaluation runner for all corridors at configurable capital cost.

By default this script now aligns with the dynamic-feedback pipeline:
- It uses feedback-loop ridership when available (`--ridership-source auto`).
- It falls back to static phase2b ridership only when dynamic data is unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.apm_corridor_evaluation_integrated import evaluate_all_corridors
from src.source_manifest import validate_source_manifest_file

import logging
logger = logging.getLogger(__name__)



def _resolve_dynamic_ridership_path(
    ridership_source: str,
    dynamic_ridership_path: Path,
) -> Path | None:
    mode = str(ridership_source).strip().lower()
    if mode not in {"auto", "dynamic", "static"}:
        raise ValueError(f"Unsupported ridership source mode: {ridership_source}")

    if mode == "static":
        return None

    if dynamic_ridership_path.exists():
        return dynamic_ridership_path

    if mode == "dynamic":
        raise FileNotFoundError(
            f"Dynamic ridership required but file was not found: {dynamic_ridership_path}"
        )
    return None


def _print_summary(results: pd.DataFrame, output_path: Path) -> None:
    if results.empty:
        logger.info("\nNo valid corridor outputs were produced. Aborting summary reporting.")
        return

    viable = int((results["financially_viable"] == True).sum())
    total = int(len(results))
    pct_viable = (viable / total * 100.0) if total else 0.0

    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS - ALL CORRIDORS")
    logger.info("=" * 80)
    logger.info(f"Total corridors evaluated: {total}")
    logger.info(f"Financially viable: {viable} ({pct_viable:.1f}%)")
    logger.info(f"Mean debt coverage: {results['debt_coverage_ratio'].mean():.2f}x")
    logger.info(f"Mean daily ridership: {results['daily_ridership'].mean():,.0f}")
    logger.info(f"Mean TIF revenue (25-yr): ${results['tif_revenue_cumulative'].mean() / 1e9:.2f}B")

    logger.info("\n" + "=" * 80)
    logger.info("TOP 10 CORRIDORS (by debt coverage)")
    logger.info("=" * 80)
    top10 = results.nlargest(10, "debt_coverage_ratio")
    for _, row in top10.iterrows():
        viable_status = "VIABLE" if bool(row["financially_viable"]) else "needs funding"
        logger.info(
            f"{row['corridor_id']}: {row['debt_coverage_ratio']:.2f}x "
            f"({viable_status}), {row['daily_ridership']:.0f} riders/day"
        )

    logger.info("\n" + "=" * 80)
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full evaluation of all corridors with unified ridership assumptions."
    )
    parser.add_argument(
        "--demand-results",
        default="data/processed/phase2b_corridor_results.csv",
        help="Input corridor demand CSV (typically phase2b_corridor_results.csv).",
    )
    parser.add_argument(
        "--dynamic-ridership-path",
        default="data/processed/feedback_loop_results.csv",
        help="Feedback-loop long-format ridership CSV used for dynamic runs.",
    )
    parser.add_argument(
        "--ridership-source",
        choices=["auto", "dynamic", "static"],
        default="auto",
        help=(
            "Ridership input mode: auto=use dynamic when available else static; "
            "dynamic=require feedback file; static=ignore feedback file."
        ),
    )
    parser.add_argument(
        "--capital-cost-per-km",
        type=float,
        default=100_000_000.0,
        help="Capital cost assumption in USD per km (default: $100M/km).",
    )
    parser.add_argument(
        "--scenario",
        default="current_zoning",
        choices=["current_zoning", "no_zoning"],
        help="Policy scenario: current_zoning (existing FAR), no_zoning (uncapped FAR).",
    )
    parser.add_argument(
        "--cashflow-years",
        type=int,
        default=25,
        help="Cashflow horizon in years.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/FULL_evaluation_50M_all40.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--source-manifest",
        default="data/processed/source_manifest.csv",
        help="Source manifest CSV path.",
    )
    parser.add_argument(
        "--max-source-age-days",
        type=int,
        default=3650,
        help="Upper bound on allowed source age in days.",
    )
    return parser.parse_args()


def _build_evaluation_overlay(
    results: pd.DataFrame, scenario: str, output_path: Path
) -> None:
    """Build / update evaluation_overlay.json for the corridor viewer.

    Merges this scenario's per-corridor uncertainty data (p10/p50/p90 ridership
    and DSCR, viability %, NPV, IRR) into a single overlay JSON that the viewer
    reads at Stage 3.
    """
    import json

    overlay_path = Path("data/processed/evaluation_overlay.json")
    overlay: dict = {}
    if overlay_path.exists():
        try:
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        except Exception:
            overlay = {}

    # Try loading uncertainty bands for richer data
    out = Path(output_path)
    bands_path = out.with_name(f"{out.stem}_uncertainty_bands.csv")
    bands_df = None
    if bands_path.exists():
        try:
            bands_df = pd.read_csv(bands_path)
        except Exception:
            pass

    scenario_data: dict = {}
    for _, row in results.iterrows():
        cid = str(row.get("corridor_id", ""))
        if not cid:
            continue
        entry: dict = {
            "p50_ridership": float(row.get("daily_ridership", 0)),
            "p50_dscr": float(row.get("debt_coverage_ratio", 0)),
            "npv_musd": float(row.get("project_npv_dynamic_musd", 0)),
            "irr": float(row.get("irr", 0)) if "irr" in row.index else None,
        }

        # Merge uncertainty bands if available
        if bands_df is not None and cid in bands_df["corridor_id"].values:
            brow = bands_df[bands_df["corridor_id"] == cid].iloc[0]
            entry["p10_ridership"] = float(brow.get("daily_ridership_p10", 0))
            entry["p90_ridership"] = float(brow.get("daily_ridership_p90", 0))
            entry["p10_dscr"] = float(brow.get("debt_coverage_ratio_p10", 0))
            entry["p90_dscr"] = float(brow.get("debt_coverage_ratio_p90", 0))
            entry["viability_pct"] = float(brow.get("financial_viability_probability", 0)) * 100
            entry["uncertainty_n_draws"] = int(brow.get("uncertainty_n_draws", 500))

        scenario_data[cid] = entry

    overlay[scenario] = scenario_data
    overlay_path.write_text(
        json.dumps(overlay, indent=2, allow_nan=False), encoding="utf-8"
    )
    logger.info(f"Evaluation overlay updated: {overlay_path} ({len(scenario_data)} corridors)")


def main() -> None:
    args = parse_args()
    demand_results_path = Path(args.demand_results)
    output_path = Path(args.output)
    dynamic_path = Path(args.dynamic_ridership_path)

    logger.info("\n" + "=" * 80)
    logger.info("FULL APM CORRIDOR EVALUATION - ALL CORRIDORS")
    logger.info(f"Capital Cost: ${args.capital_cost_per_km / 1e6:.1f}M/km")
    logger.info("=" * 80)

    validate_source_manifest_file(args.source_manifest, max_source_age_days=args.max_source_age_days)
    logger.info(f"Source manifest validated: {args.source_manifest}")

    dynamic_ridership_path = _resolve_dynamic_ridership_path(
        ridership_source=args.ridership_source,
        dynamic_ridership_path=dynamic_path,
    )
    ridership_mode = "dynamic" if dynamic_ridership_path is not None else "static"
    logger.info(f"Ridership source mode: {ridership_mode.upper()}")
    if dynamic_ridership_path is not None:
        logger.info(f"Dynamic ridership file: {dynamic_ridership_path}")
    else:
        logger.info("Dynamic ridership file: not used")

    # Use scenario-specific feedback loop results if available
    feedback_path = Path(f"data/processed/feedback_loop_results_{args.scenario}.csv")
    if not feedback_path.exists():
        feedback_path = dynamic_ridership_path

    results = evaluate_all_corridors(
        demand_results_path=demand_results_path,
        capital_cost_per_km=float(args.capital_cost_per_km),
        scenario=args.scenario,
        output_path=output_path,
        dynamic_ridership_path=dynamic_ridership_path,
        feedback_loop_path=feedback_path,
        cashflow_years=int(args.cashflow_years),
    )
    _print_summary(results, output_path=output_path)

    # Build evaluation_overlay.json for the viewer
    _build_evaluation_overlay(results, args.scenario, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

#!/usr/bin/env python
"""
Run the iterative land-use-transport feedback loop.

Usage:
    # Single scenario (default: no_zoning)
    python scripts/run_feedback_loop.py --scenario current_zoning

    # Both zoning scenarios
    python scripts/run_feedback_loop.py --all-scenarios

    # All scenarios + BRT comparison + viewer
    python scripts/run_feedback_loop.py --all-scenarios --brt-compare --serve

    # Custom time steps
    python scripts/run_feedback_loop.py --steps 0 5 10 15 20 25

    # Parallel scenario execution (memory-aware)
    python scripts/run_feedback_loop.py --all-scenarios --parallel-scenarios
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Suppress pandas FutureWarning about Series positional indexing (pandas >=2.1)
# These are triggered internally by pandas groupby/apply operations, not by our code.
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

from src.land_use_transport_model import LandUseTransportModel
from src.source_manifest import validate_source_manifest_file
from src.ensure_enriched import ensure_enriched_parcels

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Three development scenarios used by the pipeline
# ---------------------------------------------------------------------------
DEVELOPMENT_SCENARIOS = ["current_zoning", "no_zoning"]


# ---------------------------------------------------------------------------
# CLI parser (importable for tests)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Separated from main() for testability."""
    parser = argparse.ArgumentParser(
        description="Iterative land-use-transport feedback loop"
    )
    parser.add_argument(
        "--corridors",
        default="data/processed/apm_phase2a_corridors.geojson",
        help="Path to corridors GeoJSON",
    )
    parser.add_argument(
        "--parcels",
        default="data/processed/parcels_enriched_final.geojson",
        help="Path to enriched parcels GeoJSON",
    )
    parser.add_argument(
        "--od-flows",
        default="data/processed/od_parcel_flows_lodes.csv",
        help="Path to LODES OD flows CSV",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Time steps (years) to simulate. Default: annual 0-25",
    )
    parser.add_argument(
        "--no-bus-restructure",
        action="store_true",
        help="Disable bus network restructuring",
    )
    parser.add_argument(
        "--no-gtfs",
        action="store_true",
        help="Skip GTFS loading (faster startup, no dynamic bus network)",
    )
    parser.add_argument(
        "--gtfs-dir",
        default="data/raw/CityBus2025",
        help="GTFS directory for bus network integration (default: data/raw/CityBus2025)",
    )
    parser.add_argument(
        "--gtfs-productivity-csvs",
        nargs="*",
        default=[],
        help="Optional ridership CSV files (auto-discovered from GTFS dir if omitted)",
    )
    parser.add_argument(
        "--screening",
        action="store_true",
        help="Quick screening mode (fewer time steps, less output)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/feedback_loop_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--diagnostics-output",
        default="data/processed/feedback_loop_diagnostics.csv",
        help="Diagnostics CSV path",
    )
    parser.add_argument(
        "--scenario",
        default="no_zoning",
        choices=DEVELOPMENT_SCENARIOS,
        help="Development scenario (default: no_zoning)",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run all development scenarios (current_zoning + no_zoning)",
    )
    parser.add_argument(
        "--brt-compare",
        action="store_true",
        help="Re-run each scenario with transit_mode=brt for comparison",
    )
    parser.add_argument(
        "--transit-mode",
        default="apm",
        choices=["apm", "brt"],
        help="Transit mode (default: apm)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Open corridor viewer in browser after completion",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel corridor evaluation within the model",
    )
    parser.add_argument(
        "--parallel-scenarios",
        action="store_true",
        help="Run scenarios in parallel (memory-aware, uses ProcessPoolExecutor)",
    )
    parser.add_argument(
        "--skip-source-manifest-validation",
        action="store_true",
        help="Skip source manifest freshness check",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=1,
        help="Year step size (default: 1 = annual). Overridden by --steps.",
    )
    parser.add_argument(
        "--student-trip-rate-mult",
        type=float,
        default=1.0,
        help="Multiplier on student trip generation rate (default: 1.0)",
    )
    parser.add_argument(
        "--student-mode-bias",
        type=float,
        default=0.0,
        help="Additive bias on student APM mode-choice ASC (default: 0.0)",
    )
    parser.add_argument(
        "--adaptive-stop",
        action="store_true",
        help="Enable convergence-based early stopping",
    )
    # --- Tier 3 model_options flags ---
    parser.add_argument("--bus-restructuring", choices=["reactive", "proactive"],
        default=None, help="Bus restructuring mode (overrides model_options)")
    parser.add_argument("--alternative", choices=["build", "no-build", "tsm"],
        default=None, help="FTA alternatives: build, no-build, or tsm")
    parser.add_argument("--opening-delay-years", type=int, default=None,
        help="Delay APM opening by N years (construction lag)")
    parser.add_argument("--equity", action="store_true",
        help="Enable equity analysis in model_options")
    parser.add_argument("--displacement-tracking", action="store_true",
        help="Enable displacement tracking in model_options")
    parser.add_argument("--cejst-tracts", type=str, default=None,
        help="Path to CEJST disadvantaged tracts CSV")
    parser.add_argument("--scenario-config", default="scenarios_config.json",
        help="Path to scenarios_config.json for model_options and uncertainty")
    parser.add_argument("--bus-classification-method", default=None,
        help="Bus route classification method")
    parser.add_argument("--no-auto-parallel", action="store_true",
        help="Disable auto-parallel detection")
    parser.add_argument("--n-workers", type=int, default=None,
        help="Number of worker processes for --parallel (default: cpu_count - 1)")
    # --- Screening parameters ---
    parser.add_argument("--viability-threshold", type=float, default=500.0,
        help="Minimum daily ridership for screening (default: 500)")
    parser.add_argument("--screen-output", default="data/processed/screening_survivors.json",
        help="Path for screening survivors JSON")
    # --- Source manifest ---
    parser.add_argument("--source-manifest", default="data/processed/source_manifest.csv",
        help="Source manifest CSV path")
    parser.add_argument("--max-source-age-days", type=int, default=3650,
        help="Upper bound on allowed source age in days")
    parser.add_argument("--validate-sources", action="store_true",
        help="Validate source manifest before running")
    # --- Convergence tolerances ---
    parser.add_argument("--ridership-convergence-tol", type=float, default=None,
        help="Ridership convergence tolerance (default: 0.01)")
    parser.add_argument("--development-convergence-tol", type=float, default=None,
        help="Development convergence tolerance (default: 0.02)")
    parser.add_argument("--convergence-floor", type=float, default=None,
        help="Minimum ridership for convergence check (default: 25.0)")
    parser.add_argument("--max-time-steps", type=int, default=None,
        help="Maximum time steps before forced stop (default: 100)")
    parser.add_argument("--consecutive-converged-steps", type=int, default=None,
        help="Number of consecutive converged steps to stop (default: 3)")
    parser.add_argument("--stop-on-divergence", action="store_true", default=None,
        help="Stop if model diverges")
    parser.add_argument("--divergence-threshold", type=float, default=None,
        help="Relative change threshold for divergence detection (default: 1.0)")
    parser.add_argument("--consecutive-divergent-steps", type=int, default=None,
        help="Number of divergent steps before stopping (default: 2)")
    # --- Bus operating parameters ---
    parser.add_argument("--bus-service-hour-budget-multiplier", type=float, default=None,
        help="Bus service hour budget multiplier (default: 1.10)")
    parser.add_argument("--bus-service-span-hours", type=float, default=None,
        help="Bus daily service span in hours (default: 18.0)")
    parser.add_argument("--bus-parallel-route-equiv", type=float, default=None,
        help="Parallel route service-hour equivalence (default: 1.0)")
    parser.add_argument("--bus-feeder-route-equiv", type=float, default=None,
        help="Feeder route service-hour equivalence (default: 0.6)")
    parser.add_argument("--bus-max-parallel-headway", type=float, default=None,
        help="Maximum parallel bus headway minutes (default: 90.0)")
    parser.add_argument("--bus-min-feeder-headway", type=float, default=None,
        help="Minimum feeder bus headway minutes (default: 15.0)")
    parser.add_argument("--bus-max-feeder-headway", type=float, default=None,
        help="Maximum feeder bus headway minutes (default: 45.0)")
    parser.add_argument("--bus-network-strategy", choices=["incremental", "redesign"],
        default=None, help="Bus network restructuring strategy")
    # --- Ridership calibration ---
    parser.add_argument("--ridership-scale-multiplier", type=float, default=None,
        help="Global ridership scale multiplier")
    parser.add_argument("--commute-direction-min", type=float, default=None,
        help="Minimum commute direction split (default: 0.10)")
    parser.add_argument("--commute-direction-max", type=float, default=None,
        help="Maximum commute direction split (default: 0.80)")
    # --- Metro growth parameters ---
    parser.add_argument("--metro-population", type=float, default=None,
        help="Metro area population (default: 232000)")
    parser.add_argument("--metro-jobs", type=float, default=None,
        help="Metro area jobs (default: 95000)")
    parser.add_argument("--annual-pop-growth-rate", type=float, default=None,
        help="Annual population growth rate (default: 0.015)")
    parser.add_argument("--annual-job-growth-rate", type=float, default=None,
        help="Annual job growth rate (default: 0.018)")
    parser.add_argument("--corridor-capture-rate", type=float, default=None,
        help="Base corridor development capture rate (default: 0.10)")
    parser.add_argument("--max-corridor-capture-rate", type=float, default=None,
        help="Maximum corridor capture rate (default: 0.25)")
    # --- model_options CLI flags ---
    parser.add_argument("--uncertainty-correlation", action="store_true", default=None,
        help="Enable correlated (copula) Monte Carlo sampling.")
    parser.add_argument("--behavioral-sensitivity", action="store_true", default=None,
        help="Enable GP behavioral sensitivity integration.")
    parser.add_argument("--behavioral-lhs-points", type=int, default=None,
        help="Number of LHS points for behavioral sensitivity (default: 20).")
    parser.add_argument("--phase-1-stations", type=int, nargs="+", default=None,
        help="Station indices for phased opening (Phase 1).")
    parser.add_argument("--phase-2-start-year", type=int, default=None,
        help="Year when Phase 2 stations open.")
    parser.add_argument("--equity-feeder-weighting", action="store_true", default=None,
        help="Weight feeder routes toward equity communities.")
    parser.add_argument("--equity-financial-weight", type=float, default=None,
        help="Weight for equity in financial ranking (default: 0.60).")
    parser.add_argument("--equity-uplift", type=float, default=None,
        help="Equity uplift factor (default: 0.5).")
    parser.add_argument("--decision-package-maps", action="store_true", default=None,
        help="Generate map images in decision package output.")
    parser.add_argument("--fta-cost-effectiveness", action="store_true", default=None,
        help="Compute FTA cost-effectiveness metrics.")
    parser.add_argument("--robust-ranking", action="store_true", default=None,
        help="Enable robust corridor ranking (P(top-k), CVaR, max regret).")
    parser.add_argument("--robust-ranking-metric", choices=["cvar", "p10"], default=None,
        help="Tail metric for robust ranking.")
    parser.add_argument("--technical-appendix-auto", action="store_true", default=None,
        help="Auto-generate technical appendix.")
    parser.add_argument("--validation-gate", action="store_true", default=None,
        help="Run behavioral validation gate before uncertainty sampling.")
    parser.add_argument("--validation-damping", type=float, default=None,
        help="Damping factor for validation-driven calibration adjustments (0-1).")
    return parser


# ---------------------------------------------------------------------------
# Viewer data generation
# ---------------------------------------------------------------------------

def _safe_num(v, default=0):
    """Convert a value to a JSON-safe number (no inf/NaN)."""
    import math
    if v is None:
        return default
    try:
        f = float(v)
        if math.isinf(f) or math.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _compute_inline_financials(cdf, cid: str, corridor_props: dict | None) -> dict:
    """Compute financial metrics from feedback CSV data when evaluation is unavailable.

    Uses the feedback loop's per-year ridership, O&M, and fare revenue columns
    plus corridor properties (length_km, n_stops, barrier_cost_usd, curve_cost_mult)
    to produce the same financial fields the evaluation script would generate.
    """
    from src.financial_params import (
        BOND_RATE,
        DEBT_TERM_YEARS,
        FARE_PER_TRIP_USD,
        OPERATING_DAYS_PER_YEAR,
        compute_capital_cost,
        O_AND_M_FIXED_USD,
        O_AND_M_INFRA_PER_KM_USD as O_AND_M_PER_KM_USD,
        O_AND_M_PER_STATION_USD,
    )
    from src.finance import tif_cumulative_revenue

    # Corridor geometry from GeoJSON props or feedback CSV
    props = (corridor_props or {}).get(cid, {})
    length_km = _safe_num(props.get("length_km") or cdf["length_km"].iloc[0], 8.0)
    n_stops = int(_safe_num(props.get("n_stops") or cdf["n_stops"].iloc[0], 6))
    barrier_cost = _safe_num(props.get("barrier_cost_usd", 0))
    curve_mult = _safe_num(props.get("curve_cost_mult", 1.0), 1.0)

    # Capital cost (MECE decomposition + barriers + curves)
    base_capex = compute_capital_cost(length_km, n_stops)
    capex = base_capex * curve_mult + barrier_cost

    # Debt service (level annual payment)
    r, n = BOND_RATE, DEBT_TERM_YEARS
    if r > 0 and n > 0:
        debt_service = capex * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    else:
        debt_service = capex / max(n, 1)

    # Per-year series from feedback CSV
    years = sorted(cdf["year"].unique())
    n_years = len(years)

    # Farebox
    if "apm_fare_revenue_annual" in cdf.columns:
        farebox_series = np.array([_safe_num(v) for v in cdf.sort_values("year")["apm_fare_revenue_annual"].values])
    else:
        riders = np.array([_safe_num(v) for v in cdf.sort_values("year")["daily_riders"].values])
        farebox_series = riders * FARE_PER_TRIP_USD * OPERATING_DAYS_PER_YEAR
    farebox_mean = float(np.mean(farebox_series))

    # O&M
    if "apm_om_annual" in cdf.columns:
        om_series = np.array([_safe_num(v) for v in cdf.sort_values("year")["apm_om_annual"].values])
    else:
        om_base = O_AND_M_FIXED_USD + O_AND_M_PER_KM_USD * length_km + O_AND_M_PER_STATION_USD * n_stops
        om_series = np.full(n_years, om_base)
    om_mean = float(np.mean(om_series))

    # TIF
    tif_annual = 0.0
    if "new_units" in cdf.columns:
        cum_units = _safe_num(cdf["new_units"].sum())
        from src.financial_params import PROPERTY_TAX_RATE, TIF_CAPTURE_RATE_CONSERVATIVE
        increment = cum_units * 250_000
        tif_annual = increment * PROPERTY_TAX_RATE * TIF_CAPTURE_RATE_CONSERVATIVE / max(n_years, 1)

    # DSCR
    annual_revenue = farebox_mean + tif_annual
    dscr = annual_revenue / debt_service if debt_service > 0 else 0.0

    riders_sorted = np.array([_safe_num(v) for v in cdf.sort_values("year")["daily_riders"].values])
    if len(riders_sorted) >= 2:
        y5_idx = min(5, len(riders_sorted) - 1)
        fb_y5 = riders_sorted[y5_idx] * FARE_PER_TRIP_USD * OPERATING_DAYS_PER_YEAR
        fb_y25 = riders_sorted[-1] * FARE_PER_TRIP_USD * OPERATING_DAYS_PER_YEAR
        dscr_y5 = (fb_y5 + tif_annual) / debt_service if debt_service > 0 else 0.0
        dscr_y25 = (fb_y25 + tif_annual) / debt_service if debt_service > 0 else 0.0
    else:
        dscr_y5 = dscr
        dscr_y25 = dscr

    dscr_min = min(dscr, dscr_y5)

    final_riders = _safe_num(riders_sorted[-1]) if len(riders_sorted) > 0 else 0
    annual_riders = final_riders * OPERATING_DAYS_PER_YEAR
    annual_subsidy = debt_service + om_mean - annual_revenue
    cost_per_rider = annual_subsidy / annual_riders if annual_riders > 0 else 0.0

    disc_factors = np.array([(1 + BOND_RATE) ** -t for t in range(1, n_years + 1)])
    net_annual = farebox_series + tif_annual - om_series
    npv = -capex + float(np.sum(net_annual * disc_factors[:len(net_annual)]))

    total_rev = farebox_mean + tif_annual
    total_cost_annual = debt_service + om_mean
    self_sufficiency = total_rev / total_cost_annual if total_cost_annual > 0 else 0.0

    return {
        "dscr_min": round(_safe_num(dscr_min), 2),
        "dscr_year5": round(_safe_num(dscr_y5), 2),
        "dscr_year25": round(_safe_num(dscr_y25), 2),
        "self_sufficiency": round(_safe_num(self_sufficiency), 3),
        "capital_musd": round(capex / 1e6, 1),
        "annual_tif_musd": round(tif_annual / 1e6, 2),
        "farebox_musd": round(farebox_mean / 1e6, 2),
        "campus_payment_musd": 0.0,
        "cost_per_rider": round(_safe_num(cost_per_rider), 2),
        "npv_musd": round(npv / 1e6, 1),
        "annual_debt_service_musd": round(debt_service / 1e6, 2),
        "annual_om_musd": round(om_mean / 1e6, 2),
        "financially_viable": bool(dscr_min >= 1.0),
    }


def _build_scenario_data(
    results_path: str,
    evaluation_df=None,
    corridor_props: dict | None = None,
) -> dict:
    """Build corridor viewer data dict from one scenario's results CSV.

    Parameters
    ----------
    results_path : str
        Path to feedback loop results CSV.
    evaluation_df : pandas.DataFrame, optional
        Evaluation output DataFrame with financial columns.
    corridor_props : dict, optional
        Mapping of ``{corridor_id: {property_dict}}`` from the corridors GeoJSON.
    """
    df = pd.read_csv(results_path)
    data = {}
    for cid in df["corridor_id"].unique():
        cdf = df[df["corridor_id"] == cid].sort_values("year")
        y25 = cdf[cdf["year"] == cdf["year"].max()].iloc[0]
        y0 = cdf[cdf["year"] == cdf["year"].min()].iloc[0]

        # Per-year development trajectories
        units_traj = [round(_safe_num(v)) for v in cdf["new_units"].tolist()]
        pop_traj = [round(_safe_num(v)) for v in cdf["new_pop"].tolist()]
        jobs_traj = [round(_safe_num(v)) for v in cdf["new_jobs"].tolist()]
        comm_sqft_col = "new_comm_sqft" if "new_comm_sqft" in cdf.columns else None
        comm_sqft_traj = [round(_safe_num(v)) for v in cdf[comm_sqft_col].tolist()] if comm_sqft_col else [0] * len(cdf)

        # Cumulative trajectories (running sum)
        cum_units_traj, cum_pop_traj, cum_jobs_traj = [], [], []
        _cu, _cp, _cj = 0, 0, 0
        for u, p, j in zip(units_traj, pop_traj, jobs_traj):
            _cu += u; _cp += p; _cj += j
            cum_units_traj.append(_cu)
            cum_pop_traj.append(_cp)
            cum_jobs_traj.append(_cj)

        # Bus headway trajectories.
        # CSV column names: feeder_headway, bus_headway, feeder_coverage.
        # Check both old (bus_feeder_headway) and current names for compat.
        _feeder_hw_col = "feeder_headway" if "feeder_headway" in cdf.columns else (
            "bus_feeder_headway" if "bus_feeder_headway" in cdf.columns else None)
        feeder_hw_traj = [round(_safe_num(v, 30.0), 1) for v in cdf[_feeder_hw_col].tolist()] if _feeder_hw_col else []
        parallel_hw_traj = [round(_safe_num(v, 90.0), 1) for v in cdf["bus_headway"].tolist()] if "bus_headway" in cdf.columns else []
        _pressure_col = next((c for c in ["bus_restructure_pressure", "restructure_pressure"] if c in cdf.columns), None)
        pressure_traj = [round(_safe_num(v, 0.0), 3) for v in cdf[_pressure_col].tolist()] if _pressure_col else []
        _bus_riders_col = next((c for c in ["estimated_bus_ridership", "bus_ridership"] if c in cdf.columns), None)
        bus_ridership_traj = [round(_safe_num(v)) for v in cdf[_bus_riders_col].tolist()] if _bus_riders_col else []
        _coverage_traj = [round(_safe_num(v, 0.15), 2) for v in cdf["feeder_coverage"].tolist()] if "feeder_coverage" in cdf.columns else []

        data[cid] = {
            "years": [int(y) for y in cdf["year"].tolist()],
            "ridership_trajectory": [round(_safe_num(v)) for v in cdf["daily_riders"].tolist()],
            "year0_riders": round(_safe_num(y0["daily_riders"])),
            "year25_riders": round(_safe_num(y25["daily_riders"])),
            "ridership_growth": round(
                (_safe_num(y25["daily_riders"]) / max(_safe_num(y0["daily_riders"]), 1) - 1) * 100, 1
            ),
            "apm_mode_share": round(_safe_num(y25.get("lodes_commute_apm_share", y25.get("apm_mode_share", 0))), 3),
            "cum_units": round(_safe_num(cdf["new_units"].sum())),
            "cum_pop": round(_safe_num(cdf["new_pop"].sum())),
            "cum_jobs": round(_safe_num(cdf["new_jobs"].sum())),
            "cum_comm_sqft": round(_safe_num(cdf[comm_sqft_col].sum())) if comm_sqft_col else 0,
            "units_trajectory": units_traj,
            "pop_trajectory": pop_traj,
            "jobs_trajectory": jobs_traj,
            "comm_sqft_trajectory": comm_sqft_traj,
            "cum_units_trajectory": cum_units_traj,
            "cum_pop_trajectory": cum_pop_traj,
            "cum_jobs_trajectory": cum_jobs_traj,
            # Ridership components (final year) — awareness-adjusted, sum to daily_riders.
            # CSV column names: work_commute_daily, local_nonwork_daily, campus_daily,
            # destination_daily, induced_daily, latent_daily.
            "work_commute_daily": round(_safe_num(y25.get("work_commute_daily", 0))),
            "local_nonwork_daily": round(_safe_num(y25.get("local_nonwork_daily", 0))),
            "campus_daily": round(_safe_num(y25.get("campus_daily", 0))),
            "destination_daily": round(_safe_num(y25.get("destination_daily", 0))),
            "induced_daily": round(_safe_num(y25.get("induced_daily", 0))),
            "latent_daily": round(_safe_num(y25.get("latent_daily", 0))),
            "non_campus_daily": round(_safe_num(y25.get("non_campus_daily", 0))),
            "pop_catchment": round(_safe_num(y25.get("pop_catchment", 0))),
            "jobs_catchment": round(_safe_num(y25.get("jobs_catchment", 0))),
            "bus_parallel_headway": round(_safe_num(y25.get("bus_headway", 30.0), 90.0), 1),
            "bus_feeder_headway": round(_safe_num(
                y25.get("feeder_headway", y25.get("bus_feeder_headway", 30.0)), 30.0), 1),
            "feeder_coverage": round(_safe_num(y25.get("feeder_coverage", 0.15)), 2),
            "bus_pressure": round(_safe_num(
                y25.get("bus_restructure_pressure", y25.get("restructure_pressure", 0.0))), 3),
            "bus_phase_trajectory": (
                [str(v) for v in cdf[_phase_col].tolist()]
                if (_phase_col := next((c for c in ["bus_restructure_phase", "restructure_phase"]
                                        if c in cdf.columns), None))
                else []
            ),
            "bus_feeder_headway_trajectory": feeder_hw_traj,
            "feeder_coverage_trajectory": _coverage_traj,
            "bus_parallel_headway_trajectory": parallel_hw_traj,
            "bus_pressure_trajectory": pressure_traj,
            "bus_ridership_trajectory": bus_ridership_traj,
            "riders_SE01": round(_safe_num(y25.get("riders_SE01", 0))),
            "riders_SE02": round(_safe_num(y25.get("riders_SE02", 0))),
            "riders_SE03": round(_safe_num(y25.get("riders_SE03", 0))),
            "latent_SE01": round(_safe_num(y25.get("latent_SE01", 0))),
            "low_income_access_ratio": round(_safe_num(y25.get("low_income_access_ratio", 0)), 2),
        }

        # Merge financial fields from evaluation output
        _has_eval = False
        if evaluation_df is not None and "corridor_id" in evaluation_df.columns:
            eval_row = evaluation_df[evaluation_df["corridor_id"] == cid]
            if len(eval_row) > 0:
                _has_eval = True
                er = eval_row.iloc[0]
                data[cid]["dscr_min"] = round(_safe_num(er.get("dscr_min", er.get("debt_coverage_ratio", 0))), 2)
                data[cid]["dscr_year5"] = round(_safe_num(er.get("dscr_year5", 0)), 2)
                data[cid]["dscr_year25"] = round(_safe_num(er.get("dscr_year25", 0)), 2)
                data[cid]["annual_tif_musd"] = round(_safe_num(er.get("annual_tif_revenue", 0)) / 1e6, 2)
                data[cid]["farebox_musd"] = round(_safe_num(er.get("farebox_revenue_annual_mean_musd", 0)), 2)
                data[cid]["campus_payment_musd"] = round(_safe_num(er.get("campus_payment_annual_musd", 0)), 2)
                data[cid]["cost_per_rider"] = round(_safe_num(er.get("cost_per_rider", 0)), 2)
                # Capital cost and self-sufficiency for consistent viewer display
                _cap = _safe_num(er.get("capital_cost", er.get("capital_cost_total", er.get("gross_capital_cost", 0))))
                if _cap > 0:
                    data[cid]["capital_musd"] = round(_cap / 1e6, 1)
                _debt = _safe_num(er.get("annual_debt_service", 0))
                _om = _safe_num(er.get("annual_om_mean_usd", 0))
                _tif = _safe_num(er.get("annual_tif_revenue", 0))
                _fare = _safe_num(er.get("farebox_revenue_annual_mean_usd", er.get("farebox_revenue_annual_mean_musd", 0) * 1e6))
                _total_cost = _debt + _om
                _total_rev = _tif + _fare
                if _total_cost > 0:
                    data[cid]["self_sufficiency"] = round(_total_rev / _total_cost, 3)
                data[cid]["npv_musd"] = round(_safe_num(er.get("project_npv_dynamic_musd", 0)), 1)
                data[cid]["annual_debt_service_musd"] = round(_safe_num(er.get("annual_debt_service", 0)) / 1e6, 2)
                data[cid]["annual_om_musd"] = round(_safe_num(er.get("annual_om_mean_usd", 0)) / 1e6, 2)
                data[cid]["financially_viable"] = bool(er.get("financially_viable", False))

        # Merge BRT mode-compare fields if available
        if evaluation_df is not None and "transit_mode" in evaluation_df.columns:
            brt_row = evaluation_df[
                (evaluation_df["corridor_id"] == cid)
                & (evaluation_df["transit_mode"] == "BRT")
            ]
            if len(brt_row) > 0:
                br = brt_row.iloc[0]
                data[cid]["brt_dscr_min"] = round(_safe_num(br.get("dscr_min", 0)), 2)
                data[cid]["brt_daily_ridership"] = round(_safe_num(br.get("daily_ridership", 0)))
                data[cid]["brt_capital_musd"] = round(_safe_num(br.get("capital_cost", 0)) / 1e6, 1)
                data[cid]["brt_annual_debt_musd"] = round(_safe_num(br.get("annual_debt_service", 0)) / 1e6, 2)
                data[cid]["brt_annual_om_musd"] = round(_safe_num(br.get("annual_om_mean_usd", 0)) / 1e6, 2)
                data[cid]["brt_annual_tif_musd"] = round(_safe_num(br.get("annual_tif_revenue", 0)) / 1e6, 2)
                data[cid]["brt_farebox_musd"] = round(_safe_num(br.get("farebox_revenue_annual_mean_musd", 0)), 2)
                data[cid]["brt_self_sufficiency"] = round(_safe_num(br.get("self_sufficiency", 0)), 3)
                data[cid]["brt_cost_per_rider"] = round(_safe_num(br.get("cost_per_rider", 0)), 2)
                data[cid]["brt_npv_musd"] = round(_safe_num(br.get("project_npv_dynamic_musd", 0)), 1)
                data[cid]["brt_financially_viable"] = bool(br.get("financially_viable", False))
                data[cid]["brt_federal_share"] = round(_safe_num(br.get("federal_share", 0.5)), 2)

        # Fallback: compute financial metrics from feedback CSV + corridor props
        if not _has_eval:
            try:
                data[cid].update(_compute_inline_financials(cdf, cid, corridor_props))
            except Exception:
                pass  # Financial fallback is best-effort

    return data


def _embed_feedback_in_viewer(data: dict, output_dir: Path):
    """Embed multi-scenario feedback data directly into corridor_viewer.html.

    Embeds four categories of data:
    1. Feedback JSON (corridor metrics)
    2. Financial params (APM + BRT constants for viewer-side calculations)
    3. GeoJSON (corridors, stops, bus routes for offline viewing)
    4. Overlays (Stage 3/4 evaluation and economic impact)
    """
    viewer_path = output_dir / "corridor_viewer.html"
    if not viewer_path.exists():
        return

    html = viewer_path.read_text(encoding="utf-8")

    # --- 1. Embed feedback JSON ---
    minified = json.dumps(data, separators=(",", ":"), allow_nan=False)

    placeholder = "/*FEEDBACK_JSON_PLACEHOLDER*/null"
    if placeholder in html:
        html = html.replace(placeholder, minified)
    elif "EMBEDDED_FEEDBACK" in html:
        # Replace any existing EMBEDDED_FEEDBACK assignment (var or bare)
        html = re.sub(
            r"(?:var\s+)?EMBEDDED_FEEDBACK\s*=\s*\{.*?\};",
            f"var EMBEDDED_FEEDBACK = {minified};",
            html,
            count=1,
        )
    else:
        inject_point = "// -- Load data --"
        if inject_point not in html:
            print("  WARNING: Could not embed feedback data in viewer HTML")
            return
        scaffold = (
            f"// -- Embedded feedback loop data --\n"
            f"var EMBEDDED_FEEDBACK = null;\n"
            f"try {{\n"
            f"  EMBEDDED_FEEDBACK = {minified};\n"
            f"}} catch(e) {{ console.warn('Embedded feedback parse error', e); }}\n\n"
        )
        html = html.replace(inject_point, scaffold + inject_point)
        html = html.replace(
            "fetch('feedback_loop_viewer_data.json').then(r => r.json()).catch(() => ({}))",
            "typeof EMBEDDED_FEEDBACK !== 'undefined' && EMBEDDED_FEEDBACK "
            "? Promise.resolve(EMBEDDED_FEEDBACK) : "
            "fetch('feedback_loop_viewer_data.json').then(r => r.json()).catch(() => ({}))",
        )

    # --- 2. Embed financial params ---
    try:
        from src.financial_params import (
            CAPITAL_COST_GUIDEWAY_PER_KM, CAPITAL_COST_PER_STATION,
            CAPITAL_COST_PER_VEHICLE, DEFAULT_FLEET_VEHICLES,
            CAPITAL_COST_SYSTEMS_FIXED, PROFESSIONAL_SERVICES_RATE,
            CONSTRUCTION_COST_ESCALATION_RATE, CONSTRUCTION_PERIOD_YEARS,
            O_AND_M_FIXED_USD, O_AND_M_PER_STATION_USD,
            FARE_PER_TRIP_USD, OPERATING_DAYS_PER_YEAR,
            BRT_MODE, BRT_CAPITAL_COST_PER_STATION, BRT_CAPITAL_COST_VEHICLES,
            BRT_SYSTEMS_FIXED, BRT_O_AND_M_FIXED_USD,
            BRT_O_AND_M_INFRA_PER_KM_USD, BRT_O_AND_M_PER_STATION_USD,
            BRT_O_AND_M_VEH_HOUR_USD,
            O_AND_M_INFRA_PER_KM_USD as APM_OM_INFRA_KM,
            O_AND_M_VEH_HOUR_USD as APM_OM_VEH_HR,
        )
        fin_params = {
            "guideway_per_km": CAPITAL_COST_GUIDEWAY_PER_KM,
            "station_cost": CAPITAL_COST_PER_STATION,
            "vehicle_cost": CAPITAL_COST_PER_VEHICLE,
            "default_fleet": DEFAULT_FLEET_VEHICLES,
            "systems_fixed": CAPITAL_COST_SYSTEMS_FIXED,
            "prof_services_rate": PROFESSIONAL_SERVICES_RATE,
            "construction_escalation_rate": CONSTRUCTION_COST_ESCALATION_RATE,
            "construction_period_years": CONSTRUCTION_PERIOD_YEARS,
            "om_fixed": O_AND_M_FIXED_USD,
            "om_infra_per_km": APM_OM_INFRA_KM,
            "om_per_station": O_AND_M_PER_STATION_USD,
            "om_veh_hour": APM_OM_VEH_HR,
            "fare_per_trip": FARE_PER_TRIP_USD,
            "operating_days": OPERATING_DAYS_PER_YEAR,
            "brt_capital_per_km": BRT_MODE.capital_cost_per_km,
            "brt_station_cost": BRT_CAPITAL_COST_PER_STATION,
            "brt_vehicle_cost": BRT_CAPITAL_COST_VEHICLES,
            "brt_systems_fixed": BRT_SYSTEMS_FIXED,
            "brt_speed_kph": BRT_MODE.speed_kph,
            "brt_om_fixed": BRT_O_AND_M_FIXED_USD,
            "brt_om_infra_per_km": BRT_O_AND_M_INFRA_PER_KM_USD,
            "brt_om_per_station": BRT_O_AND_M_PER_STATION_USD,
            "brt_om_veh_hour": BRT_O_AND_M_VEH_HOUR_USD,
            "brt_ridership_discount": 0.70,
            "brt_federal_share_default": 0.0,
        }
        fin_json = json.dumps(fin_params, separators=(",", ":"))
        fin_marker = "// -- Embedded financial params --"
        if fin_marker not in html:
            inject = "// -- Load data --"
            if inject in html:
                html = html.replace(inject, f"{fin_marker}\nvar EMBEDDED_FINANCIAL_PARAMS = {fin_json};\n\n{inject}")
                print(f"  Embedded financial params for viewer sync")
        else:
            html = re.sub(
                r"(var EMBEDDED_FINANCIAL_PARAMS\s*=\s*).+?(;\n)",
                lambda m: m.group(1) + fin_json + m.group(2),
                html, count=1,
            )
    except ImportError:
        pass

    # --- 3. Embed GeoJSON data for offline viewing ---
    geojson_map = {
        "EMBEDDED_CORRIDORS": output_dir / "apm_phase2a_corridors.geojson",
        "EMBEDDED_STOPS": output_dir / "apm_phase2a_stops.geojson",
        "EMBEDDED_BUS_ROUTES": output_dir / "bus_routes.geojson",
        "EMBEDDED_FEEDER_ROUTES": output_dir / "feeder_routes_all.geojson",
    }
    geojson_marker = "// -- Embedded GeoJSON data (for file:// usage) --"
    if geojson_marker not in html:
        inject = "// -- Load data --"
        if inject in html:
            block = geojson_marker + "\n"
            for var_name, gpath in geojson_map.items():
                if gpath.exists():
                    gdata = json.loads(gpath.read_text(encoding="utf-8"))
                    gjson = json.dumps(gdata, separators=(",", ":"), allow_nan=False)
                    block += f"var {var_name} = {gjson};\n"
                else:
                    block += f"var {var_name} = null;\n"
            block += "\n"
            html = html.replace(inject, block + inject)
            for var_name, gpath in geojson_map.items():
                fname = gpath.name
                old_fetch = f"fetch('{fname}').then(r => r.json())"
                new_fetch = (
                    f"typeof {var_name} !== 'undefined' && {var_name} "
                    f"? Promise.resolve({var_name}) : {old_fetch}"
                )
                html = html.replace(old_fetch, new_fetch)
            print(f"  Embedded GeoJSON data for offline viewing")
    else:
        for var_name, gpath in geojson_map.items():
            if gpath.exists():
                gdata = json.loads(gpath.read_text(encoding="utf-8"))
                gjson = json.dumps(gdata, separators=(",", ":"), allow_nan=False)
                html = re.sub(
                    rf"(var {var_name}\s*=\s*).+?(;\n)",
                    lambda m, gj=gjson: m.group(1) + gj + m.group(2),
                    html, count=1,
                )

    # --- 4. Embed Stage 3/4 overlay files ---
    overlay_map = {
        "EMBEDDED_EVALUATION": output_dir / "evaluation_overlay.json",
        "EMBEDDED_ECONOMIC": output_dir / "economic_impact_overlay.json",
    }
    for var_name, opath in overlay_map.items():
        if opath.exists():
            odata = json.loads(opath.read_text(encoding="utf-8"))
            ojson = json.dumps(odata, separators=(",", ":"), allow_nan=False)
            old_decl = f"var {var_name} = null;"
            if old_decl in html:
                html = html.replace(old_decl, f"var {var_name} = {ojson};")
                print(f"  Embedded {var_name} overlay ({len(ojson)} bytes)")
            elif f"// -- Overlay data variables --" in html:
                html = html.replace(
                    "// -- Overlay data variables --",
                    f"// -- Overlay data variables --\nvar {var_name} = {ojson};",
                )
                print(f"  Injected {var_name} overlay ({len(ojson)} bytes)")

    viewer_path.write_text(html, encoding="utf-8")
    print(f"  Embedded feedback data in {viewer_path.name} ({len(minified)} bytes)")


def _generate_viewer_data(
    feedback_paths: Dict[str, str],
    evaluation_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    brt_paths: Optional[Dict[str, str]] = None,
) -> None:
    """Generate multi-scenario JSON for the corridor viewer.

    Parameters
    ----------
    feedback_paths : dict
        {scenario_name: csv_path} for APM results.
    evaluation_dfs : dict, optional
        {scenario_name: DataFrame} with financial columns to merge.
    brt_paths : dict, optional
        {scenario_name: brt_csv_path} from BRT feedback loop runs.
    """
    # Load corridor properties from GeoJSON for fallback financial computation
    corridor_props = None
    corridors_geojson = Path("data/processed/apm_phase2a_corridors.geojson")
    if corridors_geojson.exists():
        try:
            cg = json.loads(corridors_geojson.read_text(encoding="utf-8"))
            corridor_props = {
                f["properties"]["corridor_id"]: f["properties"]
                for f in cg.get("features", [])
            }
        except Exception:
            pass

    multi = {}
    eval_dfs = evaluation_dfs or {}
    for scenario, csv_path in feedback_paths.items():
        if not Path(csv_path).exists():
            continue
        multi[scenario] = _build_scenario_data(
            csv_path,
            evaluation_df=eval_dfs.get(scenario),
            corridor_props=corridor_props,
        )
        print(f"  Viewer data: {scenario} ({len(multi[scenario])} corridors)")

    # Merge BRT feedback loop results into the viewer data
    brt_results = dict(brt_paths or {})
    if brt_results:
        from src.financial_params import (
            BRT_MODE,
            BRT_O_AND_M_FIXED_USD,
            BRT_O_AND_M_INFRA_PER_KM_USD as BRT_O_AND_M_PER_KM_USD,
            BRT_O_AND_M_PER_STATION_USD,
            BRT_O_AND_M_VEH_HOUR_USD,
            PROPERTY_TAX_RATE, TIF_CAPTURE_RATE_CONSERVATIVE,
            OPERATING_DAYS_PER_YEAR,
            compute_brt_capital_cost,
            compute_brt_annual_vehicle_hours,
        )
        from src.bus_network import compute_apm_headway

        # BRT federal funding share — 0.0 = 100% local, matching APM assumption.
        BRT_FEDERAL_SHARE = 0.0

        for scenario, brt_csv in brt_results.items():
            if scenario not in multi:
                continue
            brt_path = Path(brt_csv)
            if not brt_path.exists():
                print(f"  Warning: BRT results not found: {brt_path}")
                continue
            brt_df = pd.read_csv(brt_path)
            n_merged = 0
            for cid in brt_df["corridor_id"].unique():
                if cid not in multi[scenario]:
                    continue
                brt_cdf = brt_df[brt_df["corridor_id"] == cid].sort_values("year")
                brt_y25 = brt_cdf[brt_cdf["year"] == brt_cdf["year"].max()].iloc[0]

                brt_riders_traj = [round(_safe_num(v)) for v in brt_cdf["daily_riders"].tolist()]
                brt_riders_y25 = round(_safe_num(brt_y25["daily_riders"]))

                cp = corridor_props.get(cid, {}) if corridor_props else {}
                length_km = cp.get("length_km", 5.0)
                n_stops = cp.get("n_stops", 6)
                gross_cap = compute_brt_capital_cost(length_km, n_stops)
                local_cap = gross_cap  # 100% local funding, same as APM

                r, n = 0.05, 25
                ann_debt = local_cap * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

                # Tier-3 frequency-sensitive BRT O&M
                _brt_hw = max(
                    BRT_MODE.min_headway_min,
                    min(
                        compute_apm_headway(brt_riders_y25, corridor_length_km=length_km, n_stops=n_stops),
                        BRT_MODE.max_headway_min,
                    ),
                )
                _brt_avh = compute_brt_annual_vehicle_hours(length_km, n_stops, _brt_hw)
                ann_om = BRT_MODE.compute_annual_om(length_km, n_stops, _brt_avh)

                # BRT-specific TIF from BRT's own development output
                brt_cum_units = _safe_num(brt_cdf["new_units"].sum())
                brt_n_years = max(len(brt_cdf["year"].unique()), 1)
                brt_tif_increment = brt_cum_units * 250_000  # assessed value per unit
                brt_tif_annual = brt_tif_increment * PROPERTY_TAX_RATE * TIF_CAPTURE_RATE_CONSERVATIVE / brt_n_years
                brt_tif_musd = brt_tif_annual / 1e6

                brt_fare_musd = brt_riders_y25 * 2.0 * OPERATING_DAYS_PER_YEAR / 1e6
                total_rev = (brt_tif_musd + brt_fare_musd) * 1e6
                total_cost = ann_debt + ann_om
                dscr = total_rev / ann_debt if ann_debt > 0 else 0
                ss = total_rev / total_cost if total_cost > 0 else 0
                cpr = (total_cost - total_rev) / max(brt_riders_y25 * OPERATING_DAYS_PER_YEAR, 1)

                # NPV: -capital + sum of discounted net annual cashflows
                _brt_net_annual = total_rev - ann_om
                _brt_disc = sum(_brt_net_annual / (1 + r) ** t for t in range(1, n + 1))
                brt_npv = -local_cap + _brt_disc

                multi[scenario][cid]["brt_ridership_trajectory"] = brt_riders_traj
                multi[scenario][cid]["brt_year25_riders"] = brt_riders_y25
                multi[scenario][cid]["brt_ridership_growth"] = round(
                    (_safe_num(brt_y25["daily_riders"]) / max(_safe_num(brt_cdf.iloc[0]["daily_riders"]), 1) - 1) * 100, 1
                )
                multi[scenario][cid]["brt_dscr_min"] = round(dscr, 2)
                multi[scenario][cid]["brt_daily_ridership"] = brt_riders_y25
                multi[scenario][cid]["brt_capital_musd"] = round(local_cap / 1e6, 1)
                multi[scenario][cid]["brt_annual_debt_musd"] = round(ann_debt / 1e6, 2)
                multi[scenario][cid]["brt_annual_om_musd"] = round(ann_om / 1e6, 2)
                multi[scenario][cid]["brt_annual_tif_musd"] = round(brt_tif_musd, 2)
                multi[scenario][cid]["brt_farebox_musd"] = round(brt_fare_musd, 2)
                multi[scenario][cid]["brt_self_sufficiency"] = round(ss, 3)
                multi[scenario][cid]["brt_cost_per_rider"] = round(cpr, 2)
                multi[scenario][cid]["brt_npv_musd"] = round(brt_npv / 1e6, 1)
                multi[scenario][cid]["brt_financially_viable"] = dscr >= 1.0
                multi[scenario][cid]["brt_federal_share"] = BRT_FEDERAL_SHARE
                multi[scenario][cid]["brt_cum_units"] = round(_safe_num(brt_cdf["new_units"].sum()))
                multi[scenario][cid]["brt_cum_pop"] = round(_safe_num(brt_cdf["new_pop"].sum()))
                multi[scenario][cid]["brt_cum_jobs"] = round(_safe_num(brt_cdf["new_jobs"].sum()))
                n_merged += 1
            print(f"  BRT viewer data: {scenario} ({n_merged} corridors merged from feedback loop)")

    out_path = Path("data/processed/feedback_loop_viewer_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(multi, f, indent=2, allow_nan=False)
    print(f"  Wrote {out_path}")

    _embed_feedback_in_viewer(multi, out_path.parent)


# ---------------------------------------------------------------------------
# Post-run summary
# ---------------------------------------------------------------------------
def _print_summary(results: pd.DataFrame, diagnostics: Optional[pd.DataFrame] = None) -> None:
    """Print structured post-run summary with 7 tables."""
    if results is None or results.empty:
        return

    corridors = sorted(results["corridor_id"].unique())
    years = sorted(results["year"].unique())
    milestone_years = [y for y in [0, 5, 10, 15, 20, 25] if y in years]
    if not milestone_years:
        milestone_years = years

    print(f"\n{'=' * 70}", flush=True)
    print("POST-RUN SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Table 1: Daily Ridership by Corridor and Year
    print("\n[1] Daily Ridership by Corridor and Year", flush=True)
    if "daily_riders" in results.columns:
        pivot = results.pivot_table(
            index="corridor_id", columns="year",
            values="daily_riders", aggfunc="first")
        milestone_cols = [c for c in milestone_years if c in pivot.columns]
        if milestone_cols:
            print(pivot[milestone_cols].round(0).to_string(), flush=True)

    # Table 2: Development Units by Corridor
    print("\n[2] Cumulative Development Units", flush=True)
    if "new_units" in results.columns:
        cum = results.groupby("corridor_id")["new_units"].sum().round(0)
        print(cum.to_string(), flush=True)

    # Table 3: Bus Parallel Headway
    print("\n[3] Parallel Bus Headway (minutes)", flush=True)
    if "bus_headway" in results.columns:
        pivot_bh = results.pivot_table(
            index="corridor_id", columns="year",
            values="bus_headway", aggfunc="first")
        milestone_cols = [c for c in milestone_years if c in pivot_bh.columns]
        if milestone_cols:
            print(pivot_bh[milestone_cols].round(1).to_string(), flush=True)

    # Table 4: Feeder Bus Headway
    print("\n[4] Feeder Bus Headway (minutes)", flush=True)
    if "feeder_headway" in results.columns:
        pivot_fh = results.pivot_table(
            index="corridor_id", columns="year",
            values="feeder_headway", aggfunc="first")
        milestone_cols = [c for c in milestone_years if c in pivot_fh.columns]
        if milestone_cols:
            print(pivot_fh[milestone_cols].round(1).to_string(), flush=True)

    # Table 5: Bus Restructure Phase
    print("\n[5] Final Bus Restructure Phase", flush=True)
    if "bus_restructure_phase" in results.columns:
        final = results[results["year"] == results["year"].max()]
        for _, row in final.iterrows():
            print(f"  {row['corridor_id']}: phase {row.get('bus_restructure_phase', '?')}",
                  flush=True)

    # Table 6: Convergence Summary
    print("\n[6] Convergence Summary", flush=True)
    if diagnostics is not None and not diagnostics.empty:
        for cid in corridors:
            cdiag = diagnostics[diagnostics["corridor_id"] == cid] if "corridor_id" in diagnostics.columns else diagnostics
            if not cdiag.empty:
                last = cdiag.iloc[-1]
                conv = last.get("converged", "?")
                print(f"  {cid}: converged={conv}", flush=True)

    # Table 7: Development Summary + Run Diagnostics
    print("\n[7] Development Summary", flush=True)
    final_year = results[results["year"] == results["year"].max()]
    for _, row in final_year.iterrows():
        cid = row["corridor_id"]
        riders = row.get("daily_riders", 0)
        units_cum = results[results["corridor_id"] == cid]["new_units"].sum() if "new_units" in results.columns else 0
        pop_cum = results[results["corridor_id"] == cid]["new_pop"].sum() if "new_pop" in results.columns else 0
        jobs_cum = results[results["corridor_id"] == cid]["new_jobs"].sum() if "new_jobs" in results.columns else 0
        print(f"  {cid}: {riders:,.0f} riders/day, "
              f"{units_cum:,.0f} units, {pop_cum:,.0f} pop, {jobs_cum:,.0f} jobs",
              flush=True)

    print(f"{'=' * 70}\n", flush=True)


# ---------------------------------------------------------------------------
# Feeder GeoJSON export (called after model.run())
# ---------------------------------------------------------------------------
def _export_feeder_geojson(model: "LandUseTransportModel", output_path: str) -> None:
    """Export feeder route GeoJSON from model's feeder cache or regenerate."""
    from src.feeder_route_generator import generate_feeder_routes, load_road_network
    from pyproj import Transformer
    from src.spatial_constants import US_SURVEY_FT_TO_M

    all_features: List = []

    # 1. Try the model's internal feeder cache first
    if hasattr(model, "_synthetic_feeder_cache"):
        for cid, cached in model._synthetic_feeder_cache.items():
            if hasattr(cached, "geojson_features") and cached.geojson_features:
                all_features.extend(cached.geojson_features)

    # 2. Regenerate from scratch if cache is empty
    if not all_features:
        road_graph = load_road_network(
            getattr(model, "road_graph_path", None)
            or "data/processed/lafayette_road_network.pkl"
        )
        if road_graph is None:
            return
        # Feeder generator expects EPSG:2965 feet (Indiana State Plane East).
        # Project both stations and parcels to 2965 so all KDTree queries
        # and road-graph routing use consistent coordinates.
        _tx = Transformer.from_crs("EPSG:4326", "EPSG:2965", always_xy=True)
        # parcel_cache is EPSG:3857 — reproject parcel centroids to 2965 (vectorized)
        _tx_3857_to_2965 = Transformer.from_crs("EPSG:3857", "EPSG:2965", always_xy=True)
        _, parcel_xy_3857, _ = model.parcel_cache
        _px, _py = _tx_3857_to_2965.transform(parcel_xy_3857[:, 0], parcel_xy_3857[:, 1])
        parcel_xy = np.column_stack([_px, _py])
        parcel_pop = model.parcels.get(
            "pop_alloc", pd.Series(0, index=model.parcels.index)
        ).values
        for cid, meta in model._corridor_meta.items():
            stop_coords = meta.get("stop_coords", [])
            if not stop_coords or len(stop_coords) < 2:
                continue
            station_lonlat = np.array(stop_coords)
            station_xy = np.array([
                _tx.transform(lon, lat) for lon, lat in station_lonlat
            ])
            fr_result = generate_feeder_routes(
                station_xy=station_xy,
                station_lonlat=station_lonlat,
                parcel_xy=parcel_xy,
                parcel_pop=parcel_pop,
                corridor_id=cid,
                road_graph=road_graph,
            )
            all_features.extend(fr_result.geojson_features)

    if all_features:
        feeder_path = Path(output_path).parent / "feeder_routes_all.geojson"
        feeder_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": all_features}),
            encoding="utf-8",
        )
        print(f"  Feeder routes: {len(all_features)} features -> {feeder_path}",
              flush=True)


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------
def _run_scenario(
    args: argparse.Namespace,
    scenario: str,
    transit_mode: str = "apm",
    model_options: Optional[Dict] = None,
) -> tuple:
    """Run a single scenario and return (output_path, diag_path, scenario_name).

    Returns
    -------
    (output_path, diag_path, scenario_name) on success, or None on failure.
    """
    # Build model_options
    _model_options = dict(model_options or {})
    _model_options["transit_mode"] = transit_mode

    # Load model_options and demand-driven development params from scenarios_config.json
    _cfg_path = Path(getattr(args, "scenario_config", "scenarios_config.json"))
    _ddd_cfg: Dict = {}  # demand_driven_development config from scenarios_config.json
    if _cfg_path.exists():
        try:
            with open(_cfg_path, encoding="utf-8") as _f:
                _cfg = json.load(_f)
            for k, v in _cfg.get("model_options", {}).items():
                _model_options.setdefault(k, v)
            _ddd_cfg = _cfg.get("metadata", {}).get("demand_driven_development", {})
        except Exception:
            pass

    # CLI flags override config values
    if getattr(args, "bus_restructuring", None) is not None:
        _model_options["bus_restructuring"] = args.bus_restructuring
    if getattr(args, "alternative", None) is not None:
        _model_options["alternative"] = args.alternative
    if getattr(args, "opening_delay_years", None) is not None:
        _model_options["opening_delay_years"] = args.opening_delay_years
    if getattr(args, "equity", False):
        _model_options["equity_analysis"] = True
    if getattr(args, "displacement_tracking", False):
        _model_options["displacement_tracking"] = True
    if getattr(args, "cejst_tracts", None) is not None:
        _model_options["cejst_tracts_path"] = args.cejst_tracts
    if getattr(args, "bus_classification_method", None) is not None:
        _model_options["bus_classification_method"] = args.bus_classification_method

    # Student sensitivity parameters (injected via CLI by run_student_sensitivity.py)
    if getattr(args, "student_trip_rate_mult", 1.0) != 1.0:
        _model_options["student_trip_rate_multiplier"] = args.student_trip_rate_mult
    if getattr(args, "student_mode_bias", 0.0) != 0.0:
        _model_options["student_mode_bias"] = args.student_mode_bias

    # Determine output paths
    parser = _build_parser()
    output_path = args.output
    diag_path = args.diagnostics_output

    # Fix 2: Single --scenario runs get suffixed output paths when using default
    if args.all_scenarios:
        output_path = f"data/processed/feedback_loop_results_{scenario}.csv"
        diag_path = f"data/processed/feedback_loop_diagnostics_{scenario}.csv"
    elif output_path == parser.get_default("output"):
        output_path = f"data/processed/feedback_loop_results_{scenario}.csv"
        diag_path = f"data/processed/feedback_loop_diagnostics_{scenario}.csv"

    # Fix 3: Auto-suffix with transit mode when not APM
    _tm = _model_options.get("transit_mode", "apm")
    if _tm != "apm" and f"_{_tm}" not in output_path:
        output_path = output_path.replace(".csv", f"_{_tm}.csv")
        diag_path = diag_path.replace(".csv", f"_{_tm}.csv")

    # Fix 4: Overwrite warning
    if os.path.exists(output_path):
        from datetime import datetime as _dt
        _mtime = _dt.fromtimestamp(os.path.getmtime(output_path))
        print(f"WARNING: {output_path} already exists "
              f"(modified {_mtime:%Y-%m-%d %H:%M}). Will be overwritten.",
              flush=True)

    # Time steps: --steps overrides --step-size which overrides default annual
    if args.steps is not None:
        time_steps = tuple(args.steps)
    elif args.screening:
        time_steps = (0, 5, 10, 15, 20, 25)
    else:
        step = max(int(getattr(args, "step_size", 1)), 1)
        steps = list(range(0, 26, step))
        if steps[-1] != 25:
            steps.append(25)
        time_steps = tuple(steps)

    print(f"\n{'=' * 70}", flush=True)
    print(f"SCENARIO: {scenario} | MODE: {transit_mode.upper()}", flush=True)
    print(f"Time steps: {time_steps}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"{'=' * 70}", flush=True)

    import time as _time
    _t0 = _time.monotonic()

    # Build metro growth params if any CLI args provided
    _metro_growth = {}
    if getattr(args, "metro_population", None) is not None:
        _metro_growth["metro_population"] = args.metro_population
    if getattr(args, "metro_jobs", None) is not None:
        _metro_growth["metro_jobs"] = args.metro_jobs
    if getattr(args, "annual_pop_growth_rate", None) is not None:
        _metro_growth["annual_pop_growth_rate"] = args.annual_pop_growth_rate
    if getattr(args, "annual_job_growth_rate", None) is not None:
        _metro_growth["annual_job_growth_rate"] = args.annual_job_growth_rate

    _corridor_capture = {}
    if getattr(args, "corridor_capture_rate", None) is not None:
        _corridor_capture["base_corridor_capture_rate"] = args.corridor_capture_rate
    if getattr(args, "max_corridor_capture_rate", None) is not None:
        _corridor_capture["max_corridor_capture_rate"] = args.max_corridor_capture_rate

    # Resolve GTFS paths: --no-gtfs disables; otherwise use --gtfs-dir
    gtfs_dir_resolved = None if getattr(args, "no_gtfs", False) else (
        getattr(args, "gtfs_dir", None) or None)
    gtfs_csvs = tuple(getattr(args, "gtfs_productivity_csvs", None) or [])

    # Build demand-driven development params from scenarios_config.json
    # These differentiate scenarios: zoning_cost_params controls per-scenario
    # construction cost multipliers (1.00 / 0.97 / 0.936).
    _zoning_cost_cfg = _ddd_cfg.get("zoning_cost_adjustment", {})
    _zoning_costs = {
        k: v for k, v in _zoning_cost_cfg.items()
        if k != "description" and k != "sources"
    } if _zoning_cost_cfg else {}

    _market_eq_cfg = _ddd_cfg.get("market_equilibrium", {})
    _market_params = dict(_market_eq_cfg) if _market_eq_cfg else {}

    _absorption_params = {}  # uses dataclass defaults (calibrated in code)

    model = LandUseTransportModel(
        corridors_path=args.corridors,
        parcels_path=args.parcels,
        od_path=args.od_flows,
        time_steps=time_steps,
        bus_restructure=not args.no_bus_restructure,
        adaptive_stop=getattr(args, "adaptive_stop", False),
        development_scenario=scenario,
        model_options=_model_options,
        transit_mode=transit_mode,
        # Convergence tolerances
        ridership_convergence_tol=getattr(args, "ridership_convergence_tol", None),
        development_convergence_tol=getattr(args, "development_convergence_tol", None),
        convergence_floor=getattr(args, "convergence_floor", None),
        max_time_steps=getattr(args, "max_time_steps", None),
        consecutive_converged_steps=getattr(args, "consecutive_converged_steps", None),
        stop_on_divergence=getattr(args, "stop_on_divergence", None),
        divergence_threshold=getattr(args, "divergence_threshold", None),
        consecutive_divergent_steps=getattr(args, "consecutive_divergent_steps", None),
        # Bus operating parameters
        bus_service_hour_budget_multiplier=getattr(args, "bus_service_hour_budget_multiplier", None),
        bus_service_span_hours=getattr(args, "bus_service_span_hours", None),
        bus_parallel_route_equiv=getattr(args, "bus_parallel_route_equiv", None),
        bus_feeder_route_equiv=getattr(args, "bus_feeder_route_equiv", None),
        bus_max_parallel_headway=getattr(args, "bus_max_parallel_headway", None),
        bus_min_feeder_headway=getattr(args, "bus_min_feeder_headway", None),
        bus_max_feeder_headway=getattr(args, "bus_max_feeder_headway", None),
        bus_network_strategy=getattr(args, "bus_network_strategy", None),
        # Ridership calibration
        ridership_scale_multiplier=getattr(args, "ridership_scale_multiplier", None) or 1.0,
        commute_direction_min=getattr(args, "commute_direction_min", None),
        commute_direction_max=getattr(args, "commute_direction_max", None),
        # Metro growth
        metro_growth_params=_metro_growth or None,
        corridor_capture_params=_corridor_capture or None,
        # Scenario differentiation: zoning cost, market equilibrium, absorption
        zoning_cost_params=_zoning_costs or None,
        market_params=_market_params or None,
        absorption_params=_absorption_params or None,
        # GTFS
        gtfs_dir=gtfs_dir_resolved,
        gtfs_productivity_csvs=gtfs_csvs,
    )

    # Auto-parallel: enable when 4+ corridors and multiple CPUs available
    _use_parallel = args.parallel
    if not _use_parallel and not getattr(args, "no_auto_parallel", False):
        _n_corridors = len(model._corridor_meta) if hasattr(model, "_corridor_meta") else 0
        _n_cpus = os.cpu_count() or 1
        if _n_corridors >= 4 and _n_cpus >= 2:
            _use_parallel = True
            print(f"  Auto-parallel: {_n_corridors} corridors, {_n_cpus} CPUs",
                  flush=True)

    results = model.run(parallel=_use_parallel)

    # Save results
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_p, index=False)

    # Save diagnostics
    diag_df = model.get_diagnostics()
    if diag_df is not None and not diag_df.empty:
        diag_p = Path(diag_path)
        diag_df.to_csv(diag_p, index=False)

    # Screening mode: write surviving corridors
    if getattr(args, "screening", False) and "daily_riders" in results.columns:
        _vt = getattr(args, "viability_threshold", 500.0)
        max_riders = results.groupby("corridor_id")["daily_riders"].max()
        survivors = sorted(max_riders[max_riders >= _vt].index.tolist())
        eliminated = sorted(max_riders[max_riders < _vt].index.tolist())
        screen_out = Path(getattr(args, "screen_output",
                                  "data/processed/screening_survivors.json"))
        screen_out.parent.mkdir(parents=True, exist_ok=True)
        screen_out.write_text(json.dumps({
            "scenario": scenario, "transit_mode": transit_mode,
            "viability_threshold": _vt,
            "survivors": survivors, "eliminated": eliminated,
        }, indent=2), encoding="utf-8")
        print(f"  Screening: {len(survivors)} survivors, "
              f"{len(eliminated)} eliminated -> {screen_out}", flush=True)

    _elapsed = _time.monotonic() - _t0
    print(f"Scenario {scenario} completed in {_elapsed:.1f}s", flush=True)

    # Structured post-run summary
    _print_summary(results, diag_df)

    # Export feeder route GeoJSON (regenerate — cache lost in parallel workers)
    try:
        print("  Generating feeder routes...", flush=True)
        _export_feeder_geojson(model, output_path)
    except Exception as _fe:
        import traceback
        print(f"  Feeder GeoJSON export failed: {_fe}", flush=True)
        traceback.print_exc()

    return output_path, diag_path, scenario


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Fix 1: Ensure stdout is line-buffered (prevents appearing stuck when piped)
    sys.stdout.reconfigure(line_buffering=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = _build_parser()
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("LAND-USE-TRANSPORT FEEDBACK LOOP", flush=True)
    print(f"Bus restructuring: {'OFF' if args.no_bus_restructure else 'ON'}", flush=True)
    print(f"GTFS: {'OFF' if args.no_gtfs else 'ON'}", flush=True)
    print("=" * 70, flush=True)

    # Source manifest validation
    if getattr(args, "validate_sources", False):
        args.skip_source_manifest_validation = False
    if not args.skip_source_manifest_validation:
        manifest_path = Path(getattr(args, "source_manifest", "data/processed/source_manifest.csv"))
        if manifest_path.exists():
            try:
                validate_source_manifest_file(
                    str(manifest_path),
                    max_source_age_days=getattr(args, "max_source_age_days", 3650),
                )
                print(f"Source manifest validated: {manifest_path}", flush=True)
            except Exception as e:
                print(f"WARNING: Source manifest validation failed: {e}", flush=True)

    # Ensure enriched parcels exist
    ensure_enriched_parcels(Path(args.parcels))

    # Clear stale feeder routes from previous runs (regenerated per-scenario)
    _stale_feeder = Path("data/processed/feeder_routes_all.geojson")
    if _stale_feeder.exists():
        _stale_feeder.unlink()
        print(f"Cleared stale {_stale_feeder.name}", flush=True)

    # Deterministic seed derived from project tag "GLAPM26"
    _seed = int(hashlib.sha256(b"GLAPM26").hexdigest()[:8], 16)  # 3187594324
    random.seed(_seed)
    np.random.seed(_seed % (2**32))

    # Load scenario config for model_options
    _config_path = Path(getattr(args, "scenario_config", "scenarios_config.json"))
    _global_model_options: Dict = {}
    if _config_path.exists():
        with open(_config_path, "r", encoding="utf-8") as f:
            _config = json.load(f)
        _global_model_options = _config.get("model_options", {})

    # Wire CLI model_options flags into global options
    if getattr(args, "uncertainty_correlation", None):
        _global_model_options["uncertainty_correlation"] = True
    if getattr(args, "behavioral_sensitivity", None):
        _global_model_options["behavioral_sensitivity"] = True
    if getattr(args, "behavioral_lhs_points", None) is not None:
        _global_model_options["behavioral_sensitivity_lhs_points"] = args.behavioral_lhs_points
    if getattr(args, "phase_1_stations", None) is not None:
        _global_model_options["phase_1_stations"] = args.phase_1_stations
    if getattr(args, "phase_2_start_year", None) is not None:
        _global_model_options["phase_2_start_year"] = args.phase_2_start_year
    if getattr(args, "equity_feeder_weighting", None):
        _global_model_options["equity_feeder_weighting"] = True
    if getattr(args, "equity_financial_weight", None) is not None:
        _global_model_options["equity_financial_weight"] = args.equity_financial_weight
    if getattr(args, "equity_uplift", None) is not None:
        _global_model_options["equity_uplift"] = args.equity_uplift
    if getattr(args, "decision_package_maps", None):
        _global_model_options["decision_package_maps"] = True
    if getattr(args, "fta_cost_effectiveness", None):
        _global_model_options["fta_cost_effectiveness"] = True
    if getattr(args, "robust_ranking", None):
        _global_model_options["robust_ranking"] = True
    if getattr(args, "robust_ranking_metric", None) is not None:
        _global_model_options["robust_ranking_metric"] = args.robust_ranking_metric
    if getattr(args, "technical_appendix_auto", None):
        _global_model_options["technical_appendix_auto"] = True
    if getattr(args, "validation_gate", None):
        _global_model_options["validation_gate"] = True
    if getattr(args, "validation_damping", None) is not None:
        _global_model_options["validation_damping"] = args.validation_damping

    # Build scenario list
    if args.all_scenarios:
        scenarios = list(DEVELOPMENT_SCENARIOS)
    else:
        scenarios = [args.scenario]

    # Build transit mode list
    transit_modes = [args.transit_mode]
    if args.brt_compare and "brt" not in transit_modes:
        transit_modes.append("brt")

    # Build run matrix: [(scenario, transit_mode), ...]
    run_matrix: List[tuple] = []
    for sc in scenarios:
        for tm in transit_modes:
            run_matrix.append((sc, tm))

    import time as _time
    _t_total = _time.monotonic()

    feedback_paths: Dict[str, str] = {}
    brt_paths: Dict[str, str] = {}

    # Fix 5: Parallel scenario execution
    if args.parallel_scenarios and len(run_matrix) > 1:
        try:
            import psutil
            avail_mb = psutil.virtual_memory().available // (1024 * 1024)
            max_workers = max(1, min(len(run_matrix), avail_mb // 500))
        except ImportError:
            max_workers = min(2, len(run_matrix))

        print(f"\nParallel scenario mode: {len(run_matrix)} runs, "
              f"{max_workers} workers", flush=True)

        from concurrent.futures import ProcessPoolExecutor, as_completed

        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for sc, tm in run_matrix:
                model_opts = dict(_global_model_options)
                if args.no_gtfs:
                    model_opts["no_gtfs"] = True
                fut = pool.submit(_run_scenario, args, sc, tm, model_opts)
                futures[fut] = (sc, tm)

            for fut in as_completed(futures):
                sc, tm = futures[fut]
                try:
                    result = fut.result()
                    if result:
                        if tm == "apm":
                            feedback_paths[sc] = result[0]
                        elif tm == "brt":
                            brt_paths[sc] = result[0]
                except Exception as e:
                    print(f"ERROR: {sc}/{tm} failed: {e}", flush=True)
    else:
        # Sequential execution
        for sc, tm in run_matrix:
            model_opts = dict(_global_model_options)
            if args.no_gtfs:
                model_opts["no_gtfs"] = True
            try:
                result = _run_scenario(args, sc, tm, model_opts)
                if result:
                    if tm == "apm":
                        feedback_paths[sc] = result[0]
                    elif tm == "brt":
                        brt_paths[sc] = result[0]
            except Exception as e:
                print(f"ERROR: {sc}/{tm} failed: {e}", flush=True)
                import traceback
                traceback.print_exc()

    _elapsed_total = _time.monotonic() - _t_total
    print(f"\nTotal elapsed: {_elapsed_total:.1f}s ({_elapsed_total/60:.1f}m)",
          flush=True)

    # Generate viewer data
    if feedback_paths:
        _generate_viewer_data(feedback_paths, brt_paths=brt_paths)

    # Serve viewer
    if args.serve:
        viewer_path = Path("data/processed/corridor_viewer.html")
        if viewer_path.exists():
            import webbrowser
            webbrowser.open(str(viewer_path.resolve()))
            print("Viewer opened in browser.", flush=True)
        else:
            print(f"WARNING: Viewer not found at {viewer_path}", flush=True)


if __name__ == "__main__":
    main()

"""
Standalone helper functions extracted from land_use_transport_model.py.

Pure structural refactoring — no behavior changes.  These functions have no
dependency on the LandUseTransportModel class and can be imported independently.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape as _shape
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# GeoJSON I/O
# ============================================================================

def _read_geojson_fast(path) -> gpd.GeoDataFrame:
    """Read GeoJSON via json module (bypasses fiona, 10-100x faster on Windows)."""
    path = Path(path)
    # Try parquet cache first
    pq = path.with_suffix(".parquet")
    if pq.exists() and pq.stat().st_mtime >= path.stat().st_mtime:
        try:
            return gpd.read_parquet(pq)
        except Exception:
            pass
    with open(path, "r") as fh:
        gj = json.load(fh)
    rows = [feat.get("properties", {}) for feat in gj.get("features", [])]
    geoms = [
        _shape(feat["geometry"]) if feat.get("geometry") else None
        for feat in gj["features"]
    ]
    crs = gj.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs=crs)


# ============================================================================
# Convergence functions
# ============================================================================

def compute_relative_delta(current: float, previous: float, floor: float) -> float:
    """Compute robust relative change with floor to avoid near-zero denominators."""
    denom = max(abs(float(previous)), float(floor))
    return abs(float(current) - float(previous)) / denom


def evaluate_convergence(
    current_metrics: Dict[str, float],
    previous_metrics: Optional[Dict[str, float]],
    ridership_tol: float,
    development_tol: float,
    floor: float,
) -> Dict[str, float]:
    """Evaluate per-corridor convergence status for one time step."""
    if previous_metrics is None:
        return {
            "ridership_rel_delta": np.nan,
            "new_pop_rel_delta": np.nan,
            "new_jobs_rel_delta": np.nan,
            "is_converged": False,
            "convergence_state": "baseline",
        }

    ridership_delta = compute_relative_delta(
        current_metrics.get("daily_riders", 0.0),
        previous_metrics.get("daily_riders", 0.0),
        floor=floor,
    )
    new_pop_delta = compute_relative_delta(
        current_metrics.get("new_pop", 0.0),
        previous_metrics.get("new_pop", 0.0),
        floor=floor,
    )
    new_jobs_delta = compute_relative_delta(
        current_metrics.get("new_jobs", 0.0),
        previous_metrics.get("new_jobs", 0.0),
        floor=floor,
    )

    converged = (
        ridership_delta <= ridership_tol
        and new_pop_delta <= development_tol
        and new_jobs_delta <= development_tol
    )

    return {
        "ridership_rel_delta": ridership_delta,
        "new_pop_rel_delta": new_pop_delta,
        "new_jobs_rel_delta": new_jobs_delta,
        "is_converged": converged,
        "convergence_state": "converged" if converged else "not_converged",
    }


def summarize_year_convergence(
    convergence_by_corridor: Dict[str, Dict[str, float]],
    divergence_threshold: float,
) -> Dict[str, float]:
    """Summarize year-level convergence and divergence diagnostics."""
    n_corridors = len(convergence_by_corridor)
    if n_corridors == 0:
        return {
            "n_corridors": 0,
            "n_converged": 0,
            "pct_converged": 0.0,
            "max_ridership_rel_delta": np.nan,
            "max_new_pop_rel_delta": np.nan,
            "max_new_jobs_rel_delta": np.nan,
            "all_converged": False,
            "divergence_flag": False,
        }

    def _valid_vals(key: str) -> List[float]:
        vals = []
        for entry in convergence_by_corridor.values():
            val = entry.get(key, np.nan)
            if pd.notna(val):
                vals.append(float(val))
        return vals

    ridership_vals = _valid_vals("ridership_rel_delta")
    new_pop_vals = _valid_vals("new_pop_rel_delta")
    new_jobs_vals = _valid_vals("new_jobs_rel_delta")
    n_converged = sum(1 for entry in convergence_by_corridor.values() if bool(entry.get("is_converged", False)))

    max_ridership = max(ridership_vals) if ridership_vals else np.nan
    max_new_pop = max(new_pop_vals) if new_pop_vals else np.nan
    max_new_jobs = max(new_jobs_vals) if new_jobs_vals else np.nan

    divergence_flag = any(
        val > divergence_threshold
        for val in [max_ridership, max_new_pop, max_new_jobs]
        if pd.notna(val)
    )

    return {
        "n_corridors": n_corridors,
        "n_converged": n_converged,
        "pct_converged": n_converged / n_corridors,
        "max_ridership_rel_delta": max_ridership,
        "max_new_pop_rel_delta": max_new_pop,
        "max_new_jobs_rel_delta": max_new_jobs,
        "all_converged": n_converged == n_corridors,
        "divergence_flag": divergence_flag,
    }


def evaluate_stop_conditions(
    year_all_converged: bool,
    year_divergent: bool,
    converged_streak: int,
    divergent_streak: int,
    adaptive_stop: bool,
    consecutive_converged_steps: int,
    stop_on_divergence: bool,
    consecutive_divergent_steps: int,
) -> Dict[str, object]:
    """Update convergence/divergence streaks and evaluate run-stop conditions."""
    if year_all_converged:
        converged_streak += 1
    else:
        converged_streak = 0

    if year_divergent:
        divergent_streak += 1
    else:
        divergent_streak = 0

    stop_triggered = False
    stop_reason = ""

    if stop_on_divergence and divergent_streak >= consecutive_divergent_steps:
        stop_triggered = True
        stop_reason = "divergence_stop"
    elif adaptive_stop and converged_streak >= consecutive_converged_steps:
        stop_triggered = True
        stop_reason = "adaptive_converged_stop"

    return {
        "converged_streak": converged_streak,
        "divergent_streak": divergent_streak,
        "stop_triggered": stop_triggered,
        "stop_reason": stop_reason,
    }


def update_capacity_state(
    total_capacity_sqft: float,
    remaining_capacity_sqft: float,
    theoretical_capacity_sqft: float,
    requested_delivery_sqft: float,
) -> Dict[str, float]:
    """Update parcel capacity state and return delivered sqft.

    - `theoretical_capacity_sqft` is the parcel's current modeled maximum capacity.
    - `requested_delivery_sqft` is desired delivery this step before depletion.
    """
    total = max(float(total_capacity_sqft), 0.0)
    remaining = max(float(remaining_capacity_sqft), 0.0)
    theoretical = max(float(theoretical_capacity_sqft), 0.0)
    requested = max(float(requested_delivery_sqft), 0.0)

    added_capacity = 0.0
    if theoretical > total:
        added_capacity = theoretical - total
        total = theoretical
        remaining += added_capacity

    remaining = min(remaining, total)
    delivered = min(requested, remaining)
    remaining_after = max(remaining - delivered, 0.0)
    consumed_after = max(total - remaining_after, 0.0)

    return {
        "total_capacity_sqft": total,
        "remaining_capacity_sqft": remaining_after,
        "delivered_sqft": delivered,
        "added_capacity_sqft": added_capacity,
        "consumed_capacity_sqft": consumed_after,
    }


# ============================================================================
# Bus restructuring helper
# ============================================================================

def _restructure_pressure(
    riders: float, mature_target: float, comp: float, prod: float,
) -> float:
    """Compute bus restructure pressure from demand and GTFS context.

    Higher APM maturity and weaker incumbent bus competitiveness/productivity
    increase pressure to shift from parallel routes to feeders.
    """
    rr = float(np.clip(riders / max(mature_target, 1.0), 0.0, 1.0))
    return float(np.clip(
        0.70 * rr + 0.20 * (1.0 - comp) + 0.10 * (1.0 - prod), 0.0, 1.0,
    ))


# ============================================================================
# Sparse math helpers — weights_1200 / weights_5000 are stored as
# (index, value) pairs instead of dense (61593,) arrays.  50-95% of
# entries are zero, so this saves ~30 MB across 40 corridors and speeds
# up the 6+ dot-product sites by only touching nonzero elements.
# ============================================================================

def _sparse_dot(dense_arr: np.ndarray, sp_idx: np.ndarray, sp_val: np.ndarray) -> float:
    """Dot product of a dense array with a sparse (index, value) vector."""
    if len(sp_idx) == 0:
        return 0.0
    return float(np.dot(dense_arr[sp_idx], sp_val))


def _sparse_accumulate(
    target: np.ndarray, sp_idx: np.ndarray, sp_val: np.ndarray, scale: float,
) -> None:
    """target[sp_idx] += scale * sp_val  (in-place)."""
    if len(sp_idx) > 0:
        target[sp_idx] += scale * sp_val


# ============================================================================
# Corridor-level parallelization — module-level worker functions
# ============================================================================
# ProcessPoolExecutor on Windows uses spawn, so each worker gets a fresh
# Python process.  The model is pickled once and sent via the initializer
# (not per-task), so the ~100 MB payload is deserialized N_WORKERS times,
# not 40 times.

_worker_model: Optional["LandUseTransportModel"] = None


def _init_corridor_worker(model_bytes: bytes) -> None:
    """Initialize a worker process with its own copy of the model."""
    global _worker_model
    _worker_model = pickle.loads(model_bytes)


def _run_corridor_batch(args: Tuple[List[str], bytes]) -> Tuple[List[dict], List[dict]]:
    """Run a batch of corridors on this worker's model copy.

    Returns (results_rows, diagnostics_rows).
    """
    corridor_ids, baseline_bytes = args
    model = _worker_model
    baseline = pickle.loads(baseline_bytes)
    all_results: List[dict] = []
    all_diagnostics: List[dict] = []
    n = len(corridor_ids)
    for ci, cid in enumerate(corridor_ids):
        # Suppress per-year print noise in workers — just show progress
        logger.debug(f"  [worker-{os.getpid()}] corridor {cid} ({ci+1}/{n})")
        results, diagnostics = model._run_single_corridor(cid, baseline)
        all_results.extend(results)
        all_diagnostics.extend(diagnostics)
    return all_results, all_diagnostics


# ============================================================================
# Tier 2 Stage 4: Lafayette-calibrated UrbanSim SqFtProForma
# ============================================================================

def _make_lafayette_proforma_config():
    """Create SqFtProForma config calibrated to Lafayette MSA.

    Costs: RSMeans 2024 Midwest × 0.89 Lafayette location factor.
    Rents/cap rates: from MARKET_CONFIG in realistic_developer_proforma.py.
    """
    from urbansim.developer.sqftproforma import SqFtProFormaConfig

    config = SqFtProFormaConfig()

    # Lafayette construction costs by height tier ($/sqft)
    # Wood frame (<15ft), concrete (15-55ft), steel (55-120ft), high-rise (120ft+)
    # RSMeans 2024 Midwest × 0.89 Lafayette location factor.
    # Steel-frame tier reflects 40-60% escalation over wood frame (RSMeans 2024):
    # structural steel + moment frames + fireproofing + crane costs.
    # High-rise tier adds curtain wall, elevator cores, wind bracing.
    config.costs = {
        "retail":      [155.0, 170.0, 215.0, 265.0],
        "industrial":  [125.0, 155.0, 195.0, 245.0],
        "office":      [150.0, 170.0, 215.0, 265.0],
        "residential": [155.0, 175.0, 220.0, 270.0],
    }
    config.heights_for_costs = [15, 55, 120, np.inf]

    # Developer profit factor: 17.5% margin (from financial_params.py)
    config.profit_factor = 1.175

    # Cap rate: 5.5% — weighted avg of tertiary-market residential (6%)
    # and mixed-use (5%), reflecting Lafayette MSA pricing.
    # NOTE: This is a flat cap rate applied to all use types.  In practice
    # retail/office cap rates are higher (6-8%) and industrial lower (5-6%).
    # Per-use cap rates would require SqFtProForma modifications.
    config.cap_rate = 0.055

    # Building efficiency and coverage (standard)
    config.building_efficiency = 0.70
    config.parcel_coverage = 0.80

    # Parking: Lafayette suburban rates
    config.parking_rates = {
        "retail": 2.0,
        "industrial": 0.6,
        "office": 1.0,
        "residential": 1.0,
    }

    # Only test surface and deck (underground rare in Lafayette)
    config.parking_configs = ["surface", "deck"]
    config.parking_cost_d = {"surface": 30, "deck": 90}
    config.parking_sqft_d = {"surface": 300.0, "deck": 250.0}

    return config

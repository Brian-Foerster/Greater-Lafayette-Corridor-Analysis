#!/usr/bin/env python
"""Behavioral parameter sensitivity via Latin Hypercube Sampling.

Generates N=20 LHS points in 3-dimensional behavioral parameter space
(mode choice distance sensitivity, employment growth, car ownership trend),
runs the feedback loop once per point, fits a GP surrogate, and stores
the surrogate for use in the main Monte Carlo uncertainty framework.

Usage:
    # Generate LHS design only (no runs)
    python scripts/run_behavioral_sensitivity.py --scenario current_zoning --generate-only

    # Run full pipeline: LHS -> feedback loops -> GP fit
    python scripts/run_behavioral_sensitivity.py --scenario current_zoning --n-points 20

    # Fast mode: top-5 corridors, coarse time steps
    python scripts/run_behavioral_sensitivity.py --scenario current_zoning --top-n-corridors 5 --lhs-steps 0 5 10 15 20 25
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Behavioral parameter ranges
# ---------------------------------------------------------------------------

BEHAVIORAL_PARAMS = {
    "beta_distance_mult": {
        "description": "Mode choice distance sensitivity multiplier",
        "low": 0.80,
        "high": 1.20,
        "baseline": 1.0,
    },
    "employment_growth_rate": {
        "description": "Annual employment growth rate",
        "low": -0.005,  # -0.5%/yr
        "high": 0.015,   # +1.5%/yr
        "baseline": 0.005,  # +0.5%/yr
    },
    "zero_car_share_mult": {
        "description": "Zero-car household share multiplier",
        "low": 0.95,
        "high": 1.05,
        "baseline": 1.0,
    },
    "rent_premium_mult": {
        "description": "Rent premium multiplier (scales MAX_RENT_PREMIUM 0.40)",
        "low": 0.50,
        "high": 1.25,
        "baseline": 1.0,
    },
    "commercial_accessibility_bonus_mult": {
        "description": "Commercial accessibility bonus multiplier (scales 50% bonus)",
        "low": 0.60,
        "high": 1.00,
        "baseline": 1.0,
    },
}

GP_SURROGATE_PATH = Path("data/processed/behavioral_gp_surrogate.pkl")
TRAJECTORY_DIR = Path("data/processed/behavioral_lhs_trajectories")


# ---------------------------------------------------------------------------
# LHS generation
# ---------------------------------------------------------------------------

def generate_lhs_points(
    n_points: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Generate Latin Hypercube Sample points in [0, 1]^d (d = len(BEHAVIORAL_PARAMS)).

    Uses scipy.stats.qmc.LatinHypercube for maximum space-filling.
    Falls back to stratified random sampling if scipy.stats.qmc is unavailable.

    Returns
    -------
    ndarray of shape (n_points, d) with values in [0, 1].
    """
    d = len(BEHAVIORAL_PARAMS)
    try:
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=d, seed=seed)
        return sampler.random(n=n_points)
    except ImportError:
        # Fallback: stratified random
        rng = np.random.default_rng(seed)
        n = n_points
        result = np.zeros((n, d))
        for dim in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                result[perm[i], dim] = (i + rng.random()) / n
        return result


def lhs_to_params(
    lhs_points: np.ndarray,
) -> List[Dict[str, float]]:
    """Convert LHS [0,1]^d points to behavioral parameter values.

    Maps each dimension uniformly between [low, high] for the corresponding
    behavioral parameter.
    """
    result = []
    for row in lhs_points:
        point = {}
        for dim, (pname, spec) in enumerate(BEHAVIORAL_PARAMS.items()):
            low, high = spec["low"], spec["high"]
            point[pname] = float(low + row[dim] * (high - low))
        result.append(point)
    return result


# ---------------------------------------------------------------------------
# Lookup helpers (retained for backward compatibility)
# ---------------------------------------------------------------------------

def find_nearest_lhs_point(
    query: Dict[str, float],
    lhs_params: List[Dict[str, float]],
) -> Tuple[int, float]:
    """Find the nearest LHS point to a query parameter set.

    Uses normalized Euclidean distance (each dimension scaled to [0,1]).

    Returns (index, distance).
    """
    query_vec = np.array([
        (query.get(pname, spec["baseline"]) - spec["low"]) / max(spec["high"] - spec["low"], 1e-12)
        for pname, spec in BEHAVIORAL_PARAMS.items()
    ])
    best_idx = 0
    best_dist = float("inf")
    for i, point in enumerate(lhs_params):
        point_vec = np.array([
            (point.get(pname, spec["baseline"]) - spec["low"]) / max(spec["high"] - spec["low"], 1e-12)
            for pname, spec in BEHAVIORAL_PARAMS.items()
        ])
        dist = float(np.linalg.norm(query_vec - point_vec))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx, best_dist


def interpolate_trajectory(
    query: Dict[str, float],
    lhs_params: List[Dict[str, float]],
    trajectories: List[pd.DataFrame],
    n_nearest: int = 2,
) -> pd.DataFrame:
    """Interpolate between nearest LHS trajectories for a query point.

    Uses inverse-distance weighting between the n_nearest LHS points.
    Retained for backward compatibility — prefer query_surrogate() with GP.
    """
    if not trajectories:
        raise ValueError("No trajectories available for interpolation")

    # Compute distances to all LHS points in one call
    query_vec = np.array([
        (query.get(pname, spec["baseline"]) - spec["low"]) / max(spec["high"] - spec["low"], 1e-12)
        for pname, spec in BEHAVIORAL_PARAMS.items()
    ])
    distances = []
    for i, point in enumerate(lhs_params):
        point_vec = np.array([
            (point.get(pname, spec["baseline"]) - spec["low"]) / max(spec["high"] - spec["low"], 1e-12)
            for pname, spec in BEHAVIORAL_PARAMS.items()
        ])
        dist = float(np.linalg.norm(query_vec - point_vec))
        distances.append((i, max(dist, 1e-6)))

    distances.sort(key=lambda x: x[1])
    nearest = distances[:n_nearest]

    inv_dists = [1.0 / d for _, d in nearest]
    total_inv = sum(inv_dists)
    weights = [w / total_inv for w in inv_dists]

    base = trajectories[nearest[0][0]].copy()
    numeric_cols = base.select_dtypes(include=[np.number]).columns
    result = base.copy()
    result[numeric_cols] = 0.0
    for (idx, _), w in zip(nearest, weights):
        traj = trajectories[idx]
        for col in numeric_cols:
            if col in traj.columns:
                result[col] += traj[col].values * w

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_behavioral_lookup(
    output_path: Path,
    lhs_points: np.ndarray,
    lhs_params: List[Dict[str, float]],
    trajectories: Optional[List[pd.DataFrame]] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """Save the behavioral sensitivity lookup table."""
    lookup = {
        "lhs_points": lhs_points.tolist(),
        "lhs_params": lhs_params,
        "n_points": len(lhs_params),
        "param_ranges": {k: {"low": v["low"], "high": v["high"]}
                         for k, v in BEHAVIORAL_PARAMS.items()},
    }
    if metadata:
        lookup["metadata"] = metadata
    if trajectories:
        lookup["n_trajectories"] = len(trajectories)
        lookup["trajectory_summaries"] = [
            {
                "final_ridership": float(t.iloc[-1].get("daily_ridership", 0))
                if "daily_ridership" in t.columns else 0.0,
                "n_years": len(t),
            }
            for t in trajectories
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2)
    logger.info(f"[LHS] Saved behavioral lookup to {output_path}")


def load_behavioral_lookup(
    path: Path,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Load behavioral lookup table (LHS points and parameter sets)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    lhs_points = np.array(data["lhs_points"])
    lhs_params = data["lhs_params"]
    return lhs_points, lhs_params


# ---------------------------------------------------------------------------
# BatchRunner: warm-start data sharing across LHS runs
# ---------------------------------------------------------------------------

class BatchRunner:
    """Run multiple behavioral parameter sets with shared data loading.

    Loads corridors, parcels, and OD cache once; reuses for each LHS point.
    """

    def __init__(
        self,
        corridors_path: str,
        parcels_path: str,
        scenario: str = "current_zoning",
        corridor_filter: Optional[List[str]] = None,
        steps: Optional[List[int]] = None,
    ):
        from src.land_use_transport_model import LandUseTransportModel

        self._corridors_path = corridors_path
        self._parcels_path = parcels_path
        self._scenario = scenario
        self._corridor_filter = corridor_filter
        self._steps = steps
        self._ModelClass = LandUseTransportModel

    def run_point(self, model_options: Dict) -> pd.DataFrame:
        """Run one behavioral parameter set through the feedback loop.

        Returns DataFrame with columns: corridor_id, year, daily_ridership,
        new_units, new_pop, new_jobs.
        """
        model = self._ModelClass(
            corridors_path=self._corridors_path,
            parcels_path=None,
            development_scenario=self._scenario,
            corridor_filter=self._corridor_filter,
            model_options=model_options,
        )
        results = model.run(
            time_steps=tuple(self._steps) if self._steps else None,
        )
        return results


# ---------------------------------------------------------------------------
# GP surrogate: fit, query, save/load
# ---------------------------------------------------------------------------

def _extract_metric(
    traj: pd.DataFrame,
    metric: str,
    target_year: int,
    corridor_id: Optional[str] = None,
) -> float:
    """Extract a scalar metric from a trajectory DataFrame."""
    df = traj
    if corridor_id is not None and "corridor_id" in df.columns:
        df = df[df["corridor_id"] == corridor_id]
    if df.empty:
        return 0.0
    # Use the row closest to target_year
    if "year" in df.columns:
        closest_idx = (df["year"] - target_year).abs().idxmin()
        row = df.loc[closest_idx]
    else:
        row = df.iloc[-1]
    return float(row.get(metric, 0.0))


def fit_behavioral_surrogate(
    lhs_points: np.ndarray,
    trajectories: List[pd.DataFrame],
    target_year: int = 25,
    target_corridor: Optional[str] = None,
) -> Dict:
    """Fit GP regressors for each output metric at a given year.

    Returns dict mapping metric name -> fitted GaussianProcessRegressor.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel

    metrics = ["daily_ridership", "new_units", "new_pop", "new_jobs"]
    surrogates = {}

    for metric in metrics:
        y = np.array([
            _extract_metric(traj, metric, target_year, target_corridor)
            for traj in trajectories
        ])
        # Skip metrics that are all zero (no signal to fit)
        if np.all(y == 0):
            continue
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=0.3)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            alpha=1e-6,
        )
        gp.fit(lhs_points, y)
        surrogates[metric] = gp
        logger.debug(f"  GP fit for {metric}: R²={gp.score(lhs_points, y):.4f}, "
              f"kernel={gp.kernel_}")

    return surrogates


def query_surrogate(
    query: Dict[str, float],
    surrogates: Dict,
) -> Dict[str, Tuple[float, float]]:
    """Query GP surrogate for a single parameter point.

    Returns dict mapping metric -> (mean, std) prediction.
    """
    query_vec = np.array([[
        (query.get(pname, spec["baseline"]) - spec["low"])
        / max(spec["high"] - spec["low"], 1e-12)
        for pname, spec in BEHAVIORAL_PARAMS.items()
    ]])

    results = {}
    for metric, gp in surrogates.items():
        mean, std = gp.predict(query_vec, return_std=True)
        results[metric] = (float(mean[0]), float(std[0]))
    return results


def query_surrogate_batch(
    query_points: np.ndarray,
    surrogates: Dict,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Query GP surrogate for multiple parameter points (vectorized).

    Parameters
    ----------
    query_points : ndarray of shape (n_draws, 3) in [0,1]^3 normalized space
    surrogates : dict from fit_behavioral_surrogate

    Returns dict mapping metric -> (mean_array, std_array).
    """
    results = {}
    for metric, gp in surrogates.items():
        mean, std = gp.predict(query_points, return_std=True)
        results[metric] = (mean, std)
    return results


def save_surrogate(surrogates: Dict, path: Path = GP_SURROGATE_PATH) -> None:
    """Persist fitted GP surrogates to disk."""
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(surrogates, path)
    logger.info(f"[GP] Saved surrogate to {path}")


def load_surrogate(path: Path = GP_SURROGATE_PATH) -> Optional[Dict]:
    """Load fitted GP surrogates from disk. Returns None if not found."""
    if not path.exists():
        return None
    import joblib
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate behavioral sensitivity LHS design and run feedback loops"
    )
    parser.add_argument("--scenario", default="current_zoning")
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/processed/behavioral_sensitivity_lookup.json",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate LHS design, don't run feedback loops",
    )
    parser.add_argument(
        "--top-n-corridors",
        type=int,
        default=None,
        help="Run only top N corridors by ridership (faster for exploration)",
    )
    parser.add_argument(
        "--lhs-steps",
        nargs="+",
        type=int,
        default=None,
        help="Time steps for LHS exploration (e.g., 0 5 10 15 20 25)",
    )
    parser.add_argument(
        "--corridors-path",
        default="data/processed/corridors_integrated_financial_current_zoning.geojson",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=25,
        help="Year to extract metrics for GP fitting",
    )
    args = parser.parse_args()

    logger.info(f"[LHS] Generating {args.n_points}-point Latin Hypercube design...")
    lhs_points = generate_lhs_points(n_points=args.n_points, seed=args.seed)
    lhs_params = lhs_to_params(lhs_points)

    logger.info(f"[LHS] Generated {len(lhs_params)} parameter combinations:")
    for i, p in enumerate(lhs_params):
        logger.debug(f"  Point {i+1}: " + ", ".join(f"{k}={v:.4f}" for k, v in p.items()))

    # Check minimum pairwise distance (LHS coverage quality)
    min_dist = float("inf")
    for i in range(len(lhs_points)):
        for j in range(i + 1, len(lhs_points)):
            d = float(np.linalg.norm(lhs_points[i] - lhs_points[j]))
            if d < min_dist:
                min_dist = d
    logger.info(f"[LHS] Minimum pairwise distance: {min_dist:.4f} "
          f"(ideal ≈ {1.0/args.n_points:.4f})")

    save_behavioral_lookup(
        Path(args.output), lhs_points, lhs_params,
        metadata={
            "scenario": args.scenario,
            "seed": args.seed,
            "n_points": args.n_points,
        },
    )

    if args.generate_only:
        logger.info("[LHS] Design generated. Use without --generate-only to run feedback loops.")
        return 0

    # -----------------------------------------------------------------------
    # Run each LHS point through the feedback loop
    # -----------------------------------------------------------------------
    logger.info(f"\n[LHS] Running {len(lhs_params)} feedback loop evaluations...")

    # Determine corridor filter (top-N by ridership from a quick sort)
    corridor_filter = None
    if args.top_n_corridors is not None:
        try:
            import geopandas as gpd
            corr_gdf = gpd.read_file(args.corridors_path)
            if "daily_ridership" in corr_gdf.columns:
                top = corr_gdf.nlargest(args.top_n_corridors, "daily_ridership")
                corridor_filter = top["corridor_id"].tolist()
                logger.info(f"[LHS] Filtering to top {args.top_n_corridors} corridors: {corridor_filter}")
            elif "corridor_id" in corr_gdf.columns:
                corridor_filter = corr_gdf["corridor_id"].tolist()[:args.top_n_corridors]
                logger.info(f"[LHS] Filtering to first {args.top_n_corridors} corridors: {corridor_filter}")
        except Exception as e:
            logger.info(f"[LHS] Warning: couldn't filter corridors: {e}")

    runner = BatchRunner(
        corridors_path=args.corridors_path,
        parcels_path=None,
        scenario=args.scenario,
        corridor_filter=corridor_filter,
        steps=args.lhs_steps,
    )

    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    trajectories = []
    for i, params in enumerate(lhs_params):
        logger.info(f"\n[LHS] === Point {i+1}/{len(lhs_params)} ===")
        logger.debug(f"  {params}")
        t0 = time.time()
        try:
            result_df = runner.run_point(model_options={
                "beta_distance_mult": params["beta_distance_mult"],
                "employment_growth_rate": params["employment_growth_rate"],
                "zero_car_share_mult": params["zero_car_share_mult"],
            })
            traj_path = TRAJECTORY_DIR / f"trajectory_{i:02d}_{args.scenario}.csv"
            result_df.to_csv(traj_path, index=False)
            trajectories.append(result_df)
            elapsed = time.time() - t0
            logger.debug(f"  Done in {elapsed:.0f}s, saved to {traj_path}")
        except Exception as e:
            logger.debug(f"  ERROR: {e}")
            # Append empty DataFrame to maintain index alignment
            trajectories.append(pd.DataFrame())

    # Update lookup with trajectory info
    save_behavioral_lookup(
        Path(args.output), lhs_points, lhs_params,
        trajectories=[t for t in trajectories if not t.empty],
        metadata={
            "scenario": args.scenario,
            "seed": args.seed,
            "n_points": args.n_points,
        },
    )

    # -----------------------------------------------------------------------
    # Fit GP surrogate
    # -----------------------------------------------------------------------
    valid_trajectories = [(i, t) for i, t in enumerate(trajectories) if not t.empty]
    if len(valid_trajectories) < 5:
        logger.info(f"\n[GP] Only {len(valid_trajectories)} valid trajectories — "
              f"need at least 5 for GP fitting. Skipping.")
        return 0

    valid_indices, valid_trajs = zip(*valid_trajectories)
    valid_lhs = lhs_points[list(valid_indices)]

    logger.info(f"\n[GP] Fitting surrogate on {len(valid_trajs)} LHS trajectories "
          f"(target year {args.target_year})...")
    surrogates = fit_behavioral_surrogate(
        valid_lhs, list(valid_trajs),
        target_year=args.target_year,
    )

    if surrogates:
        save_surrogate(surrogates)

        # Quick validation: predict at baseline
        baseline = {k: v["baseline"] for k, v in BEHAVIORAL_PARAMS.items()}
        baseline_pred = query_surrogate(baseline, surrogates)
        logger.info(f"\n[GP] Baseline prediction:")
        for metric, (mean, std) in baseline_pred.items():
            logger.debug(f"  {metric}: {mean:.1f} ± {std:.1f}")
    else:
        logger.info("[GP] No metrics with non-zero signal. No surrogate saved.")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())

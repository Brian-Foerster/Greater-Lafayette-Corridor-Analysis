"""GTFS-driven ridership calibration utilities.

This module augments the gravity-based ridership estimator using observed
route productivity data (passengers, miles, hours) and GTFS service patterns.

Workflow:
- Load GTFS (stops, trips, stop_times, routes) for a period.
- Derive stop→route frequency weights from stop_times/trips.
- Predict stop-level potential trips via gravity model (existing transit_model).
- Allocate to routes by frequency.
- Calibrate walk-time decay (beta) and global scaling to minimize error vs observed.

Outputs:
- Best beta, scale, RMSE, per-route comparison DataFrame.
- CSVs for QA (optional).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from src.transit_model import assign_boardings_to_routes, assign_ridership_gravity

# Module-level GTFS cache — avoids re-parsing the same CSV files
# every time the model is instantiated or a scenario reruns.
_gtfs_cache: dict = {}


def _fare_scaling(
    fare_base: float,
    fare_new: float,
    elasticity: float,
    elasticity_01: float = -0.25,
    elasticity_other: float = -0.07,
    use_piecewise: bool = False,
) -> float:
    """Compute demand scaling from fare change.

    If use_piecewise=True, apply elasticity_01 for a 0→1 fare introduction, and
    elasticity_other for any other fare change. Otherwise, use the provided
    constant elasticity.
    """
    base = max(fare_base, 0.01)
    new = max(fare_new, 0.01)

    if use_piecewise:
        if fare_base < 1.01 and fare_new >= 0.99 and fare_new <= 1.01:
            eff_elasticity = elasticity_01
        else:
            eff_elasticity = elasticity_other
    else:
        eff_elasticity = elasticity

    return (new / base) ** eff_elasticity


@dataclass
class CalibrationResult:
    beta: float
    scale: float
    rmse: float
    comparisons: pd.DataFrame


@dataclass(frozen=True)
class GtfsCompetitivenessSummary:
    """Summary of GTFS-derived route competitiveness metrics."""

    route_metrics: pd.DataFrame
    system_competitiveness: float
    system_productivity: float


def _load_gtfs_tables(gtfs_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gtfs_dir = Path(gtfs_dir)
    cache_key = str(gtfs_dir.resolve())
    if cache_key in _gtfs_cache:
        return _gtfs_cache[cache_key]
    stops = pd.read_csv(gtfs_dir / "stops.txt")
    routes = pd.read_csv(gtfs_dir / "routes.txt")
    trips = pd.read_csv(gtfs_dir / "trips.txt")
    stop_times = pd.read_csv(gtfs_dir / "stop_times.txt")
    day_cols = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    calendar_path = gtfs_dir / "calendar.txt"
    if calendar_path.exists():
        calendar = pd.read_csv(calendar_path)
    else:
        # Fallback for feeds that only provide calendar_dates or no calendar in fixtures.
        service_ids = trips["service_id"].dropna().astype(str).unique().tolist()
        calendar = pd.DataFrame({"service_id": service_ids})
        for col in day_cols:
            calendar[col] = 1

    if "service_id" not in calendar.columns:
        calendar["service_id"] = trips["service_id"].astype(str)
    for col in day_cols:
        if col not in calendar.columns:
            calendar[col] = 1

    result = (stops, routes, trips, stop_times, calendar)
    _gtfs_cache[cache_key] = result
    return result


def _build_stop_geometries(stops: pd.DataFrame, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        stops.copy(),
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs=crs,
    )
    return gdf


def _compute_actual_headways(
    trips: pd.DataFrame, 
    stop_times: pd.DataFrame, 
    calendar: pd.DataFrame
) -> pd.DataFrame:
    """Compute actual average headways (minutes) per route from GTFS scheduling.
    
    Analyzes trip departure times to calculate time between successive trips,
    weighted by service days. Returns dataframe with route_id and avg_headway_min.
    """
    # Merge trips with stop_times to get first stop departure for each trip
    trip_starts = stop_times.groupby('trip_id').agg(
        departure_time=('departure_time', 'first')
    ).reset_index()
    
    # Parse time strings to minutes since midnight
    def time_to_minutes(t):
        parts = str(t).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    
    trip_starts['departure_min'] = trip_starts['departure_time'].apply(time_to_minutes)
    
    # Merge with trips to get route_id and service_id (indexed for speed)
    _trips_idx = trips[['trip_id', 'route_id', 'service_id']].set_index('trip_id')
    trip_starts = trip_starts.join(_trips_idx, on='trip_id', how='inner')

    # Merge with calendar to get service days (indexed for speed)
    _cal_idx = calendar[['service_id', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']].set_index('service_id')
    trip_starts = trip_starts.join(_cal_idx, on='service_id', how='inner')
    
    # Compute days per week for each service pattern
    day_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    trip_starts['days_per_week'] = trip_starts[day_cols].sum(axis=1)
    
    headways = []
    for route_id in trip_starts['route_id'].unique():
        route_trips = trip_starts[trip_starts['route_id'] == route_id].copy()
        route_trips = route_trips.sort_values('departure_min')
        
        if len(route_trips) < 2:
            # Single trip per day, assign long headway
            headways.append({'route_id': route_id, 'avg_headway_min': 120.0, 'trips_per_day': 1.0})
            continue
        
        # Compute gaps between consecutive trips
        route_trips['next_departure'] = route_trips['departure_min'].shift(-1)
        route_trips['gap_min'] = route_trips['next_departure'] - route_trips['departure_min']
        
        # Filter out overnight gaps (>180 min indicates end-of-day)
        valid_gaps = route_trips[route_trips['gap_min'] <= 180]['gap_min']
        
        if len(valid_gaps) == 0:
            avg_headway = 120.0
        else:
            avg_headway = valid_gaps.mean()
        
        # Compute trips per day weighted by service days
        trips_per_day = (route_trips['days_per_week'] / 7.0).sum()
        
        headways.append({
            'route_id': route_id,
            'avg_headway_min': avg_headway,
            'trips_per_day': trips_per_day
        })
    
    return pd.DataFrame(headways)


def compute_route_competitiveness_metrics(
    route_headways: pd.DataFrame,
    observed_productivity: Optional[pd.DataFrame] = None,
    headway_reference_min: float = 15.0,
    headway_power: float = 1.5,
) -> pd.DataFrame:
    """Compute route-level competitiveness from GTFS frequency and productivity.

    Parameters
    ----------
    route_headways:
        DataFrame with `route_id`, `avg_headway_min`, and optionally `trips_per_day`.
    observed_productivity:
        Optional DataFrame with `route_id` and `observed_passengers`.
    headway_reference_min:
        Reference service level for headway scoring.
    headway_power:
        Curvature for headway score response.
    """
    required = {"route_id", "avg_headway_min"}
    missing = sorted(required - set(route_headways.columns))
    if missing:
        raise ValueError(
            f"route_headways missing required columns: {missing}"
        )

    metrics = route_headways.copy()
    metrics["route_id"] = metrics["route_id"].astype(str)
    metrics["avg_headway_min"] = pd.to_numeric(metrics["avg_headway_min"], errors="coerce")
    metrics["avg_headway_min"] = metrics["avg_headway_min"].fillna(120.0).clip(lower=5.0, upper=180.0)

    if "trips_per_day" in metrics.columns:
        metrics["trips_per_day"] = pd.to_numeric(metrics["trips_per_day"], errors="coerce").fillna(0.0)
    else:
        metrics["trips_per_day"] = 0.0

    metrics["trips_per_hour"] = 60.0 / metrics["avg_headway_min"].clip(lower=5.0)
    ratio = metrics["avg_headway_min"] / max(float(headway_reference_min), 1.0)
    metrics["headway_score"] = 1.0 / (1.0 + np.power(ratio, float(headway_power)))
    metrics["headway_score"] = metrics["headway_score"].clip(lower=0.0, upper=1.0)

    metrics["service_intensity_score"] = (metrics["trips_per_day"] / 24.0).clip(lower=0.0, upper=1.0)
    metrics["productivity_score"] = 0.5
    metrics["observed_passengers"] = np.nan

    if observed_productivity is not None and len(observed_productivity) > 0:
        prod = observed_productivity.copy()
        if "route_id" not in prod.columns:
            raise ValueError("observed_productivity must include route_id")
        if "observed_passengers" not in prod.columns:
            raise ValueError("observed_productivity must include observed_passengers")
        prod["route_id"] = prod["route_id"].astype(str)
        prod["observed_passengers"] = pd.to_numeric(prod["observed_passengers"], errors="coerce")
        prod = prod.dropna(subset=["observed_passengers"])
        prod = prod.groupby("route_id", as_index=False)["observed_passengers"].sum()

        metrics = metrics.merge(prod, on="route_id", how="left", suffixes=("", "_prod"))
        if "observed_passengers_prod" in metrics.columns:
            metrics["observed_passengers"] = metrics["observed_passengers_prod"]
            metrics = metrics.drop(columns=["observed_passengers_prod"])

        observed = metrics["observed_passengers"].fillna(0.0)
        max_obs = float(observed.max()) if len(observed) else 0.0
        if max_obs > 0:
            metrics["productivity_score"] = (observed / max_obs).clip(lower=0.0, upper=1.0)
        else:
            metrics["productivity_score"] = 0.5

    metrics["competitiveness_score"] = (
        0.55 * metrics["headway_score"]
        + 0.30 * metrics["service_intensity_score"]
        + 0.15 * metrics["productivity_score"]
    ).clip(lower=0.0, upper=1.0)

    metrics["competitiveness_band"] = pd.cut(
        metrics["competitiveness_score"],
        bins=[-0.001, 0.33, 0.66, 1.0],
        labels=["low", "medium", "high"],
    ).astype(str)

    ordered_cols = [
        "route_id",
        "avg_headway_min",
        "trips_per_day",
        "trips_per_hour",
        "observed_passengers",
        "headway_score",
        "service_intensity_score",
        "productivity_score",
        "competitiveness_score",
        "competitiveness_band",
    ]
    return metrics[ordered_cols].sort_values("route_id").reset_index(drop=True)


def load_gtfs_competitiveness_summary(
    gtfs_dir: Path,
    productivity_csvs: Optional[Sequence[Path]] = None,
) -> GtfsCompetitivenessSummary:
    """Load GTFS tables and return route/system competitiveness summary."""
    _, _, trips, stop_times, calendar = _load_gtfs_tables(Path(gtfs_dir))
    route_headways = _compute_actual_headways(trips, stop_times, calendar)

    observed = None
    if productivity_csvs:
        prod_frames = []
        for csv_path in productivity_csvs:
            csv_file = Path(csv_path)
            if csv_file.exists():
                try:
                    prod_frames.append(_load_productivity_csv(csv_file))
                except Exception:
                    # Keep calibration robust to partial/dirty monthly files.
                    continue
        if prod_frames:
            observed = pd.concat(prod_frames, ignore_index=True)
            observed = observed.groupby("route_id", as_index=False)["observed_passengers"].sum()

    route_metrics = compute_route_competitiveness_metrics(
        route_headways=route_headways,
        observed_productivity=observed,
    )

    weights = route_metrics["trips_per_day"].fillna(0.0).to_numpy()
    comp_vals = route_metrics["competitiveness_score"].fillna(0.0).to_numpy()
    prod_vals = route_metrics["productivity_score"].fillna(0.0).to_numpy()

    if len(weights) == 0:
        system_comp = 0.5
        system_prod = 0.5
    elif float(weights.sum()) > 0:
        system_comp = float(np.average(comp_vals, weights=weights))
        system_prod = float(np.average(prod_vals, weights=weights))
    else:
        system_comp = float(np.mean(comp_vals))
        system_prod = float(np.mean(prod_vals))

    return GtfsCompetitivenessSummary(
        route_metrics=route_metrics,
        system_competitiveness=system_comp,
        system_productivity=system_prod,
    )


def _compute_stop_route_frequency(trips: pd.DataFrame, stop_times: pd.DataFrame) -> pd.DataFrame:
    """Estimate relative frequency weights per stop-route pair.

    Uses count of stop-time records per (stop_id, route_id) as a proxy for frequency.
    This is sufficient for allocation weights; absolute scheduling not required.
    """
    merged = stop_times.merge(trips[["trip_id", "route_id"]], on="trip_id", how="left")
    freq = merged.groupby(["stop_id", "route_id"], as_index=False).size()
    freq = freq.rename(columns={"size": "freq"})
    return freq


def _load_productivity_csv(csv_path: Path) -> pd.DataFrame:
    """Load route productivity CSV and normalize columns.

    Supports both 2025 monthly files and older year-joined files.
    Expected columns: route identifier, Passengers, Total Miles, Total Hours.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Identify route column
    route_col = None
    for cand in ["Route", "route", "route_id"]:
        if cand in df.columns:
            route_col = cand
            break
    if route_col is None:
        raise ValueError(f"No route column found in {csv_path}")

    # Filter valid routes (numeric or non-empty)
    df = df[df[route_col].notna()]
    df[route_col] = df[route_col].astype(str).str.strip()
    df = df[df[route_col] != ""]

    # Keep passenger metrics
    keep_cols = [col for col in df.columns if col.lower().startswith("passenger")]
    if not keep_cols:
        keep_cols = ["Passengers"] if "Passengers" in df.columns else []
    passengers_col = keep_cols[0] if keep_cols else "Passengers"
    if passengers_col not in df.columns:
        raise ValueError(f"Passengers column not found in {csv_path}")

    out = df[[route_col, passengers_col]].copy()
    out = out.rename(columns={route_col: "route_id", passengers_col: "observed_passengers"})

    # Some files include grouping rows; drop non-numeric passenger rows
    out["observed_passengers"] = pd.to_numeric(out["observed_passengers"], errors="coerce")
    out = out.dropna(subset=["observed_passengers"])

    # Aggregate (some files have multiple months/rows per route)
    out = out.groupby("route_id", as_index=False)["observed_passengers"].sum()
    return out


def calibrate_ridership_from_gtfs(
    parcels_gdf: gpd.GeoDataFrame,
    gtfs_dir: Path,
    productivity_csvs: Iterable[Path],
    jobs_col: str = "jobs",
    pop_col: str = None,
    beta_grid: Iterable[float] = (
        0.004, 0.005, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.009, 0.010, 0.011, 0.012
    ),
    max_walk_m: float = 1000.0,
    walk_speed_kph: float = 5.0,
    fare_base: float = 2.0,   # CityBus 2026 fare
    fare_new: float = 2.0,   # CityBus 2026 integrated fare
    elasticity: float = -0.5,
    elasticity_01: float = -0.25,
    elasticity_other: float = -0.07,
    use_piecewise_elasticity: bool = True,
    use_institutional_weights: bool = True,
) -> CalibrationResult:
    """Calibrate gravity model beta and global scale using observed route productivity.

    Args:
        parcels_gdf: Parcels with geometry and jobs_col (and pop_col if available).
        gtfs_dir: Folder containing GTFS files (stops.txt, trips.txt, stop_times.txt, routes.txt, calendar.txt).
        productivity_csvs: One or more CSVs with observed passengers by route.
        jobs_col: Column representing trip generation (employment) (default jobs).
        pop_col: Optional column for population; if provided, used as secondary demand source.
        beta_grid: Iterable of beta values to search.
        max_walk_m: Maximum walk radius for parcel-stop association.
        walk_speed_kph: Walking speed.
        use_institutional_weights: If True and institutional_weight column exists, use weighted demand.
    Returns:
        CalibrationResult with best beta, scale, rmse, and per-route comparison.
    """
    stops, routes, trips, stop_times, calendar = _load_gtfs_tables(gtfs_dir)
    stops_gdf = _build_stop_geometries(stops)
    stop_to_route = _compute_stop_route_frequency(trips, stop_times)
    
    # Compute actual headways from GTFS scheduling
    route_headways = _compute_actual_headways(trips, stop_times, calendar)
    
    # Add headway to stop_to_route for frequency-weighted allocation
    # Headway → frequency: shorter headway = higher frequency
    # Use trips_per_hour = 60 / headway_min as weight
    route_headways['trips_per_hour'] = 60.0 / route_headways['avg_headway_min'].clip(lower=5.0)
    stop_to_route = stop_to_route.merge(route_headways[['route_id', 'trips_per_hour']], on='route_id', how='left')
    stop_to_route['trips_per_hour'] = stop_to_route['trips_per_hour'].fillna(1.0)
    
    # Weight frequency by trips per hour (replaces simple stop-time count weighting)
    stop_to_route['freq_weighted'] = stop_to_route['freq'] * stop_to_route['trips_per_hour']

    # Load observed productivity
    observed_list = [_load_productivity_csv(Path(p)) for p in productivity_csvs]
    observed = pd.concat(observed_list, ignore_index=True).groupby("route_id", as_index=False).sum()

    fare_scale = _fare_scaling(
        fare_base, fare_new, elasticity,
        elasticity_01=elasticity_01,
        elasticity_other=elasticity_other,
        use_piecewise=use_piecewise_elasticity,
    )
    
    # Check for institutional weights and prepare weighted columns if needed
    parcels_for_gravity = parcels_gdf.copy()
    actual_jobs_col = jobs_col
    actual_pop_col = pop_col
    
    if use_institutional_weights and 'institutional_weight' in parcels_gdf.columns:
        # Create weighted versions of jobs/pop columns
        if jobs_col in parcels_gdf.columns:
            parcels_for_gravity[f'{jobs_col}_weighted'] = parcels_gdf[jobs_col] * parcels_gdf['institutional_weight']
            actual_jobs_col = f'{jobs_col}_weighted'
        if pop_col and pop_col in parcels_gdf.columns:
            parcels_for_gravity[f'{pop_col}_weighted'] = parcels_gdf[pop_col] * parcels_gdf['institutional_weight']
            actual_pop_col = f'{pop_col}_weighted'

    # Use enhanced gravity if both pop and jobs available
    if actual_pop_col and actual_pop_col in parcels_for_gravity.columns:
        from src.transit_model import assign_ridership_enhanced
        gravity_fn = lambda p, s, b: assign_ridership_enhanced(p, s, pop_col=actual_pop_col, jobs_col=actual_jobs_col, beta_walk=b, max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph, alpha_pop=0.6, alpha_jobs=0.4)
    else:
        gravity_fn = lambda p, s, b: assign_ridership_gravity(p, s, jobs_col=actual_jobs_col, beta_walk=b, max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph)

    # Predict stop-level potential trips for each beta in grid
    best_rmse = math.inf
    best_beta = None
    best_scale = None
    best_comp = None

    for beta in beta_grid:
        # Gravity prediction using selected gravity function
        stop_potential = gravity_fn(parcels_for_gravity, stops_gdf, beta)

        # Apply fare elasticity scaling
        stop_potential["potential_trips"] *= fare_scale

        # Allocate to routes using frequency weights (now headway-adjusted)
        route_alloc = assign_boardings_to_routes(stop_potential, stop_to_route, route_freq_col="freq_weighted")
        route_alloc = route_alloc.rename(columns={"allocated_trips": "predicted"})
        route_alloc = route_alloc.reset_index().rename(columns={"route_id": "route_id"})

        # Merge with observed
        comp = observed.merge(route_alloc, on="route_id", how="inner")
        if comp.empty:
            continue

        # Solve for optimal global scaling (least squares): scale = argmin ||obs - scale*pred||
        pred = comp["predicted"].to_numpy()
        obs = comp["observed_passengers"].to_numpy()
        denom = np.dot(pred, pred)
        if denom == 0:
            continue
        scale = float(np.dot(obs, pred) / denom)
        comp["pred_scaled"] = comp["predicted"] * scale
        comp["abs_error"] = comp["pred_scaled"] - comp["observed_passengers"]
        rmse = float(np.sqrt(np.mean(np.square(comp["abs_error"]))))

        if rmse < best_rmse:
            best_rmse = rmse
            best_beta = beta
            best_scale = scale
            best_comp = comp.copy()

    if best_comp is None:
        raise RuntimeError("Calibration failed: no overlapping routes between GTFS and productivity data")

    # Final comparisons sorted by absolute error
    best_comp = best_comp.sort_values("abs_error", key=lambda s: s.abs(), ascending=False)

    return CalibrationResult(
        beta=best_beta,
        scale=best_scale,
        rmse=best_rmse,
        comparisons=best_comp,
    )


def apply_calibrated_ridership(
    parcels_gdf: gpd.GeoDataFrame,
    gtfs_dir: Path,
    calibration: CalibrationResult,
    jobs_col: str = "jobs",
    pop_col: str = None,
    max_walk_m: float = 1000.0,
    walk_speed_kph: float = 5.0,
    fare_base: float = 2.0,   # CityBus 2026 fare
    fare_new: float = 2.0,   # CityBus 2026 integrated fare
    elasticity: float = -0.5,
    elasticity_01: float = -0.25,
    elasticity_other: float = -0.07,
    use_piecewise_elasticity: bool = True,
    use_institutional_weights: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate calibrated ridership predictions (stops, routes) using fitted beta and scale.

    Returns:
        stop_ridership: DataFrame indexed by stop_id with potential_trips (scaled)
        route_boardings: DataFrame indexed by route_id with allocated_trips (scaled)
    """
    stops, routes, trips, stop_times, calendar = _load_gtfs_tables(gtfs_dir)
    stops_gdf = _build_stop_geometries(stops)
    stop_to_route = _compute_stop_route_frequency(trips, stop_times)
    
    # Compute actual headways and weight frequencies
    route_headways = _compute_actual_headways(trips, stop_times, calendar)
    route_headways['trips_per_hour'] = 60.0 / route_headways['avg_headway_min'].clip(lower=5.0)
    stop_to_route = stop_to_route.merge(route_headways[['route_id', 'trips_per_hour']], on='route_id', how='left')
    stop_to_route['trips_per_hour'] = stop_to_route['trips_per_hour'].fillna(1.0)
    stop_to_route['freq_weighted'] = stop_to_route['freq'] * stop_to_route['trips_per_hour']
    
    # Handle institutional weights
    parcels_for_gravity = parcels_gdf.copy()
    actual_jobs_col = jobs_col
    actual_pop_col = pop_col
    
    if use_institutional_weights and 'institutional_weight' in parcels_gdf.columns:
        if jobs_col in parcels_gdf.columns:
            parcels_for_gravity[f'{jobs_col}_weighted'] = parcels_gdf[jobs_col] * parcels_gdf['institutional_weight']
            actual_jobs_col = f'{jobs_col}_weighted'
        if pop_col and pop_col in parcels_gdf.columns:
            parcels_for_gravity[f'{pop_col}_weighted'] = parcels_gdf[pop_col] * parcels_gdf['institutional_weight']
            actual_pop_col = f'{pop_col}_weighted'
    
    # Use enhanced or simple gravity
    if actual_pop_col and actual_pop_col in parcels_for_gravity.columns:
        from src.transit_model import assign_ridership_enhanced
        stop_potential = assign_ridership_enhanced(
            parcels_for_gravity, stops_gdf,
            pop_col=actual_pop_col, jobs_col=actual_jobs_col,
            beta_walk=calibration.beta, max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph,
            alpha_pop=0.6, alpha_jobs=0.4
        )
    else:
        stop_potential = assign_ridership_gravity(
            parcels_for_gravity, stops_gdf,
            jobs_col=actual_jobs_col,
            beta_walk=calibration.beta, max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph
        )
    
    fare_scale = _fare_scaling(
        fare_base, fare_new, elasticity,
        elasticity_01=elasticity_01,
        elasticity_other=elasticity_other,
        use_piecewise=use_piecewise_elasticity,
    )
    stop_potential["potential_trips"] *= calibration.scale * fare_scale

    route_alloc = assign_boardings_to_routes(stop_potential, stop_to_route, route_freq_col="freq_weighted")
    route_alloc = route_alloc.rename(columns={"allocated_trips": "allocated_trips"})

    return stop_potential, route_alloc

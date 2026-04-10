"""
Transit ridership model with gravity-based and LODES-based demand estimation.

Walk access decay parameter (beta_walk) calibrated to TCRP Report 165:
- beta_walk = 0.0024 per second of walk time
- At 5 km/h walk speed: 400m = 288 seconds
- exp(-0.0024 * 288) = 0.50 (50% weight at 400m, matching TCRP benchmark)
- exp(-0.0024 * 576) = 0.25 (25% weight at 800m)

Previous beta_walk=0.01 was too aggressive, producing only 5.6% weight at 400m.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from src.spatial_constants import PROJECT_CRS, US_SURVEY_FT_TO_M as _US_FT_TO_M

# Optional network-based walk-time helper
try:
    from src.build_network import compute_walk_time_via_network
except Exception:
    compute_walk_time_via_network = None


# Default path for LODES OD parcel flows
DEFAULT_LODES_OD_PATH = Path("data/processed/od_parcel_flows_lodes.csv")


@dataclass(frozen=True)
class TransferAssignmentParams:
    """Parameters for transfer-aware multi-corridor assignment."""

    transfer_penalty_min: float = 8.0
    transfer_constant_penalty: float = 0.35
    utility_time_beta: float = 0.08
    access_floor: float = 1e-6
    max_transfers: int = 1
    direct_in_vehicle_time_min: float = 20.0
    split_transfer_boardings: bool = True


def _build_institutional_weight_map(
    parcels_gdf: Optional[gpd.GeoDataFrame] = None,
    institutional_overlay_path: Optional[Path] = None,
    parcel_id_col: str = "PARCEL_ID",
) -> Tuple[Dict[str, float], str]:
    """Build parcel -> institutional_weight mapping from overlay or parcel columns."""
    if institutional_overlay_path is not None:
        from src.data.institutional_generators import load_institutional_overlay

        overlay_df = load_institutional_overlay(institutional_overlay_path)
        if overlay_df.empty:
            return {}, "overlay_empty"
        weight_map = dict(
            zip(
                overlay_df["parcel_id"].astype(str),
                overlay_df["institutional_weight"].astype(float),
            )
        )
        return weight_map, f"overlay:{Path(institutional_overlay_path).name}"

    if parcels_gdf is not None and "institutional_weight" in parcels_gdf.columns:
        pid_col = parcel_id_col if parcel_id_col in parcels_gdf.columns else None
        if pid_col is None:
            pid_col = next(
                (c for c in ("PARCEL_ID", "parcel_id", "STKEY", "stkey") if c in parcels_gdf.columns),
                None,
            )
        if pid_col is None:
            return {}, "parcels_missing_id"

        overlay = parcels_gdf[[pid_col, "institutional_weight"]].copy()
        overlay[pid_col] = overlay[pid_col].astype(str)
        overlay["institutional_weight"] = pd.to_numeric(
            overlay["institutional_weight"], errors="coerce"
        ).fillna(1.0)
        overlay["institutional_weight"] = overlay["institutional_weight"].clip(lower=0.0)
        overlay = overlay.drop_duplicates(subset=[pid_col], keep="first")
        weight_map = dict(zip(overlay[pid_col], overlay["institutional_weight"]))
        return weight_map, "parcels_column"

    return {}, "none"


def apply_institutional_overlay_to_od(
    od_df: pd.DataFrame,
    institutional_weight_map: Dict[str, float],
    strength: float = 1.0,
    max_multiplier: float = 5.0,
    source: str = "none",
) -> pd.DataFrame:
    """Apply parcel-level institutional overlay multipliers to OD trips.

    Pair multiplier is geometric mean of origin/destination weights:
      pair = sqrt(origin_weight * dest_weight)
      multiplier = 1 + strength * (pair - 1)
      multiplier clipped to [0, max_multiplier]
    """
    out = od_df.copy()
    out["trips_base"] = out["trips"].astype(float)

    if "origin_parcel" not in out.columns or "dest_parcel" not in out.columns:
        out["institutional_origin_weight"] = 1.0
        out["institutional_dest_weight"] = 1.0
        out["institutional_multiplier"] = 1.0
        out["trips_institutional_delta"] = 0.0
        out["institutional_pair_flag"] = False
        out["institutional_overlay_applied"] = False
        out["institutional_overlay_source"] = source
        return out

    if not institutional_weight_map:
        out["institutional_origin_weight"] = 1.0
        out["institutional_dest_weight"] = 1.0
        out["institutional_multiplier"] = 1.0
        out["trips_institutional_delta"] = 0.0
        out["institutional_pair_flag"] = False
        out["institutional_overlay_applied"] = False
        out["institutional_overlay_source"] = source
        return out

    s = max(float(strength), 0.0)
    max_mult = max(float(max_multiplier), 0.0)

    origin_w = (
        out["origin_parcel"].astype(str).map(institutional_weight_map).fillna(1.0).astype(float)
    )
    dest_w = (
        out["dest_parcel"].astype(str).map(institutional_weight_map).fillna(1.0).astype(float)
    )
    pair_mult_raw = np.sqrt(origin_w * dest_w)
    pair_mult = 1.0 + s * (pair_mult_raw - 1.0)
    pair_mult = np.clip(pair_mult, 0.0, max_mult)

    out["institutional_origin_weight"] = origin_w
    out["institutional_dest_weight"] = dest_w
    out["institutional_multiplier"] = pair_mult
    out["institutional_pair_flag"] = (origin_w > 1.0) | (dest_w > 1.0)
    out["trips"] = out["trips_base"] * out["institutional_multiplier"]
    out["trips_institutional_delta"] = out["trips"] - out["trips_base"]
    out["institutional_overlay_applied"] = True
    out["institutional_overlay_source"] = source
    return out


def compute_parcel_stop_walk_times(parcels_gdf, stops_gdf, max_walk_m=1000, walk_speed_kph=5):
    """Compute walk times (seconds) between each parcel and stops within max_walk_m.

    Returns a DataFrame with columns: PARCEL_ID, stop_id, walk_dist_m, walk_time_s.
    """
    if parcels_gdf.crs is None or parcels_gdf.crs.to_string().startswith('EPSG:4326'):
        parcels = parcels_gdf.to_crs(PROJECT_CRS).copy()
    else:
        parcels = parcels_gdf.copy()
    if stops_gdf.crs is None or stops_gdf.crs.to_string().startswith('EPSG:4326'):
        stops = stops_gdf.to_crs(PROJECT_CRS).copy()
    else:
        stops = stops_gdf.copy()

    # EPSG:2965 coordinates are in US survey feet — convert to meters
    parcel_pts = np.vstack([parcels.geometry.centroid.x.values * _US_FT_TO_M,
                            parcels.geometry.centroid.y.values * _US_FT_TO_M]).T
    stop_pts = np.vstack([stops.geometry.x.values * _US_FT_TO_M,
                          stops.geometry.y.values * _US_FT_TO_M]).T

    tree = cKDTree(stop_pts)
    # query all stops within max_walk_m for each parcel
    neighbors = tree.query_ball_point(parcel_pts, r=max_walk_m)

    rows = []
    for i, nbrs in enumerate(neighbors):
        pid = parcels.iloc[i]['PARCEL_ID']
        if len(nbrs) == 0:
            continue
        for j in nbrs:
            sx, sy = stop_pts[j]
            px, py = parcel_pts[i]
            dist = np.hypot(sx - px, sy - py)
            speed_mps = (walk_speed_kph * 1000.0) / 3600.0
            tt = dist / speed_mps
            rows.append({'PARCEL_ID': pid, 'stop_id': stops.iloc[j].get('stop_id', j), 'walk_dist_m': float(dist), 'walk_time_s': float(tt)})

    return pd.DataFrame(rows)


def assign_ridership_gravity(parcels_gdf, stops_gdf, jobs_col='jobs', beta_walk=0.0024, max_walk_m=1000, walk_speed_kph=5, nodes_gdf=None, edges_gdf=None):
    """Assign potential ridership to stops using a gravity model.

    - For each parcel-stop pair within `max_walk_m`, compute attraction = jobs * exp(-beta_walk * walk_time_s).
    - If `nodes_gdf` and `edges_gdf` are provided and `compute_walk_time_via_network` is available, parcel-stop walk times are computed via the network (NetworkX/pandana); otherwise KDTree Euclidean walk times are used.
    - Sum attractions by stop to estimate potential ridership at stops.

    Returns a DataFrame keyed by stop_id with columns `potential_trips` and `num_parcels`.
    """
    if jobs_col not in parcels_gdf.columns:
        parcels = parcels_gdf.copy()
        parcels[jobs_col] = 1
    else:
        parcels = parcels_gdf.copy()

    # Use network-based walk times if possible and requested
    if nodes_gdf is not None and edges_gdf is not None and compute_walk_time_via_network is not None:
        try:
            # compute_walk_time_via_network returns DataFrame with PARCEL_ID, walk_time_to_stop_s, etc
            wt = compute_walk_time_via_network(nodes_gdf, edges_gdf, stops_gdf, parcels, walk_speed_kph=walk_speed_kph)
            # wt may contain 'walk_time_to_stop_s' and 'stop_id'
            if 'walk_time_to_stop_s' in wt.columns and 'stop_id' in wt.columns:
                stop_pairs = wt.rename(columns={'walk_time_to_stop_s': 'walk_time_s'})[['PARCEL_ID','stop_id','walk_time_s']].copy()
            else:
                # fallback to KDTree if network output not in expected form
                stop_pairs = compute_parcel_stop_walk_times(parcels, stops_gdf, max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph)
        except Exception:
            stop_pairs = compute_parcel_stop_walk_times(parcels, stops_gdf, max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph)
    else:
        stop_pairs = compute_parcel_stop_walk_times(parcels, stops_gdf, max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph)

    if stop_pairs.empty:
        # return stops with zero potential
        out = pd.DataFrame({'stop_id': stops_gdf.apply(lambda r: r.get('stop_id', None) if 'stop_id' in stops_gdf.columns else None, axis=1)})
        out['potential_trips'] = 0.0
        out['num_parcels'] = 0
        out['stop_id'] = out['stop_id'].fillna(pd.RangeIndex(len(out))).astype(object)
        return out.set_index('stop_id')

    # attach jobs to parcel-stop pairs
    jobs = parcels.set_index('PARCEL_ID')[jobs_col].to_dict()
    stop_pairs['jobs'] = stop_pairs['PARCEL_ID'].map(lambda pid: float(jobs.get(pid, 0)))
    # compute attraction with exponential decay on walk_time_s
    stop_pairs['attr'] = stop_pairs['jobs'] * np.exp(-beta_walk * stop_pairs['walk_time_s'])

    # normalize near-zero negative/NaN walk times
    stop_pairs['attr'] = stop_pairs['attr'].fillna(0.0)

    agg = stop_pairs.groupby('stop_id').agg(potential_trips=('attr', 'sum'), num_parcels=('PARCEL_ID', 'nunique')).reset_index()
    agg = agg.set_index('stop_id')
    return agg


def run_transit_scenario(parcels_gdf, stops_gdf, out_path=None, jobs_col='jobs', beta_walk=0.0024, max_walk_m=1000, walk_speed_kph=5):
    """Run a scenario and write stop-level potential ridership to CSV.

    Returns the aggregated DataFrame.
    """
    agg = assign_ridership_gravity(parcels_gdf, stops_gdf, jobs_col=jobs_col, beta_walk=beta_walk, max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        agg.reset_index().to_csv(out_path, index=False)
    return agg


def assign_boardings_to_routes(stop_agg_df, stop_to_route_df, route_freq_col='freq'):
    """Allocate stop-level potential trips to routes serving those stops.

    Parameters:
    - stop_agg_df: DataFrame indexed by `stop_id` with column `potential_trips`.
    - stop_to_route_df: DataFrame with columns `stop_id`, `route_id`, and `freq` (frequency). If `freq` missing, defaults to 1.
    - route_freq_col: name of the frequency column in `stop_to_route_df`.

    Returns a DataFrame indexed by `route_id` with columns `allocated_trips` and `num_stops_served`.
    """
    # Ensure inputs are DataFrames
    stops = stop_agg_df.reset_index().rename(columns={'index':'stop_id'}) if 'stop_id' not in stop_agg_df.columns else stop_agg_df.copy()
    if 'stop_id' not in stops.columns:
        stops = stops.reset_index().rename(columns={'index':'stop_id'})

    str_df = stop_to_route_df.copy()
    if route_freq_col not in str_df.columns:
        str_df[route_freq_col] = 1.0

    # Merge potential trips into stop->route mapping
    merged = str_df.merge(stops[['stop_id','potential_trips']], on='stop_id', how='left')
    merged['potential_trips'] = merged['potential_trips'].fillna(0.0)

    # Compute weights per stop: proportion per route = freq / sum(freq for stop)
    merged['total_freq'] = merged.groupby('stop_id')[route_freq_col].transform('sum')
    merged['n_routes'] = merged.groupby('stop_id')[route_freq_col].transform('count')
    # Avoid division by zero; if total_freq == 0, treat equal weights across routes for that stop
    merged['weight'] = merged.apply(lambda r: (r[route_freq_col] / r['total_freq']) if r['total_freq'] > 0 else (1.0 / r['n_routes'] if r['n_routes'] > 0 else 0.0), axis=1)

    # allocate trips
    merged['allocated'] = merged['potential_trips'] * merged['weight']

    agg = merged.groupby('route_id').agg(allocated_trips=('allocated','sum'), num_stops_served=('stop_id','nunique')).reset_index().set_index('route_id')
    return agg


def assign_ridership_enhanced(parcels_gdf, stops_gdf, pop_col='pop_alloc', jobs_col='jobs_alloc', beta_walk=0.0024, max_walk_m=1000, walk_speed_kph=5, alpha_pop=0.6, alpha_jobs=0.4, nodes_gdf=None, edges_gdf=None, mass_col: str | None = None):
    """Enhanced ridership estimator using both population and jobs.

    Attraction = (alpha_pop * pop + alpha_jobs * jobs) * exp(-beta_walk * walk_time_s).
    Returns DataFrame indexed by stop_id with `potential_trips` and `num_parcels`.
    """
    parcels = parcels_gdf.copy()
    if pop_col not in parcels.columns:
        parcels[pop_col] = 0.0
    if jobs_col not in parcels.columns:
        parcels[jobs_col] = 0.0

    # Compute walk times using network if available, else KDTree
    if nodes_gdf is not None and edges_gdf is not None and compute_walk_time_via_network is not None:
        try:
            wt = compute_walk_time_via_network(nodes_gdf, edges_gdf, stops_gdf, parcels, walk_speed_kph=walk_speed_kph)
            if 'walk_time_to_stop_s' in wt.columns and 'stop_id' in wt.columns:
                stop_pairs = wt.rename(columns={'walk_time_to_stop_s': 'walk_time_s'})[['PARCEL_ID','stop_id','walk_time_s']].copy()
            else:
                stop_pairs = compute_parcel_stop_walk_times(parcels, stops_gdf, max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph)
        except Exception:
            stop_pairs = compute_parcel_stop_walk_times(parcels, stops_gdf, max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph)
    else:
        stop_pairs = compute_parcel_stop_walk_times(parcels, stops_gdf, max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph)

    if stop_pairs.empty:
        out = pd.DataFrame({'stop_id': stops_gdf.apply(lambda r: r.get('stop_id', None) if 'stop_id' in stops_gdf.columns else None, axis=1)})
        out['potential_trips'] = 0.0
        out['num_parcels'] = 0
        out['stop_id'] = out['stop_id'].fillna(pd.RangeIndex(len(out))).astype(object)
        return out.set_index('stop_id')

    # Determine parcel mass (demand) either from OD column or pop+jobs
    if mass_col is not None and mass_col in parcels.columns:
        mass_lookup = parcels.set_index('PARCEL_ID')[mass_col].astype(float).fillna(0.0).to_dict()
        stop_pairs['mass'] = stop_pairs['PARCEL_ID'].map(lambda pid: float(mass_lookup.get(pid, 0.0)))
    else:
        pops = parcels.set_index('PARCEL_ID')[pop_col].to_dict()
        jobs = parcels.set_index('PARCEL_ID')[jobs_col].to_dict()
        stop_pairs['pop'] = stop_pairs['PARCEL_ID'].map(lambda pid: float(pops.get(pid, 0.0)))
        stop_pairs['jobs'] = stop_pairs['PARCEL_ID'].map(lambda pid: float(jobs.get(pid, 0.0)))
        stop_pairs['mass'] = alpha_pop * stop_pairs['pop'] + alpha_jobs * stop_pairs['jobs']

    stop_pairs['attr'] = stop_pairs['mass'] * np.exp(-beta_walk * stop_pairs['walk_time_s'])
    stop_pairs['attr'] = stop_pairs['attr'].fillna(0.0)

    # Aggregate stop-level metrics, including a mass-weighted average walk distance for downstream mode choice
    stop_pairs['mass_nonzero'] = stop_pairs['mass'].replace(0.0, np.nan)
    stop_pairs['walk_mass'] = stop_pairs['walk_dist_m'] * stop_pairs['mass']

    agg = stop_pairs.groupby('stop_id').agg(
        potential_trips=('attr', 'sum'),
        num_parcels=('PARCEL_ID', 'nunique'),
        mass_sum=('mass', 'sum'),
        walk_mass_sum=('walk_mass', 'sum')
    ).reset_index()

    # Compute weighted mean walk distance; fall back to 0 when no mass
    agg['distance_m'] = agg.apply(lambda r: (r['walk_mass_sum'] / r['mass_sum']) if r['mass_sum'] > 0 else 0.0, axis=1)
    agg = agg.drop(columns=['mass_sum', 'walk_mass_sum'])

    agg = agg.set_index('stop_id')
    return agg


def load_lodes_od_flows(
    lodes_path: Path = DEFAULT_LODES_OD_PATH,
) -> pd.DataFrame:
    """Load LODES-derived parcel OD flows.

    Returns DataFrame with columns: origin_parcel, dest_parcel, trips, SE01, SE02, SE03
    """
    if not lodes_path.exists():
        raise FileNotFoundError(
            f"LODES OD flows not found at {lodes_path}. "
            "Run src/lodes_integration.py first to generate the file."
        )
    return pd.read_csv(
        lodes_path,
        dtype={"origin_parcel": str, "dest_parcel": str},
    )


def load_lodes_for_mode_choice(
    lodes_path: Path = DEFAULT_LODES_OD_PATH,
    min_trips: float = 0.01,
    normalize_columns: bool = True,
    apply_institutional_overlay: bool = False,
    parcels_gdf: Optional[gpd.GeoDataFrame] = None,
    institutional_overlay_path: Optional[Path] = None,
    institutional_strength: float = 1.0,
    institutional_max_multiplier: float = 5.0,
    parcel_id_col: str = "PARCEL_ID",
) -> pd.DataFrame:
    """Load and prepare LODES OD flows for mode choice model.

    Handles column name normalization for compatibility with mode_choice functions.

    Parameters:
    - lodes_path: Path to LODES OD flows CSV
    - min_trips: Minimum trips to include (filters low-volume flows)
    - normalize_columns: If True, ensures standard column names
    - apply_institutional_overlay: If True, apply institutional demand overlay to trips.
    - parcels_gdf: Parcels frame with `institutional_weight` column (optional overlay source).
    - institutional_overlay_path: CSV/GeoJSON overlay file path (optional overlay source).
    - institutional_strength: Blend factor for overlay impact (0=no effect, 1=full).
    - institutional_max_multiplier: Hard cap on OD multiplier from overlay.
    - parcel_id_col: Parcel id column used when deriving overlay from parcels_gdf.

    Returns DataFrame with columns:
    - origin_parcel, dest_parcel: parcel IDs
    - trips: number of trips
    - SE01, SE02, SE03: income segment counts
    - se01_share, se02_share, se03_share: normalized segment shares
    - dominant_income_segment: dominant segment label per OD pair
    - low_income_share, high_income_share: compatibility aliases
    - if overlay applied: `trips_base`, `institutional_multiplier`,
      `trips_institutional_delta`, and related diagnostic fields.
    """
    od_df = load_lodes_od_flows(lodes_path)

    # Normalize column names if needed (support various naming conventions)
    if normalize_columns:
        col_map = {
            # Support alternative column names
            'origin_parcel_id': 'origin_parcel',
            'dest_parcel_id': 'dest_parcel',
            'destination_parcel': 'dest_parcel',
            'origin': 'origin_parcel',
            'destination': 'dest_parcel',
            'trip_count': 'trips',
            'flow': 'trips',
        }
        od_df = od_df.rename(columns={k: v for k, v in col_map.items() if k in od_df.columns})

    # Filter low-volume flows
    if min_trips > 0 and 'trips' in od_df.columns:
        od_df = od_df[od_df['trips'] >= min_trips].copy()

    # Add derived income segment columns if income data available
    if all(col in od_df.columns for col in ['SE01', 'SE02', 'SE03']):
        total_jobs = od_df['SE01'] + od_df['SE02'] + od_df['SE03'] + 0.001  # avoid div by zero
        od_df['se01_share'] = od_df['SE01'] / total_jobs
        od_df['se02_share'] = od_df['SE02'] / total_jobs
        od_df['se03_share'] = od_df['SE03'] / total_jobs
        od_df['low_income_share'] = od_df['se01_share']
        od_df['high_income_share'] = od_df['se03_share']
        od_df['dominant_income_segment'] = od_df[['SE01', 'SE02', 'SE03']].idxmax(axis=1)

    if apply_institutional_overlay:
        weight_map, overlay_source = _build_institutional_weight_map(
            parcels_gdf=parcels_gdf,
            institutional_overlay_path=institutional_overlay_path,
            parcel_id_col=parcel_id_col,
        )
        od_df = apply_institutional_overlay_to_od(
            od_df=od_df,
            institutional_weight_map=weight_map,
            strength=institutional_strength,
            max_multiplier=institutional_max_multiplier,
            source=overlay_source,
        )

    return od_df


def assign_ridership_lodes(
    parcels_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    lodes_od_path: Path = DEFAULT_LODES_OD_PATH,
    beta_walk: float = 0.01,
    max_walk_m: float = 1000,
    walk_speed_kph: float = 5,
    min_trips: float = 0.01,
) -> pd.DataFrame:
    """Assign ridership to stops using LODES origin-destination commute flows.

    For each OD pair in LODES data:
    1. Find stops within walk distance of origin parcel (boarding stops)
    2. Find stops within walk distance of destination parcel (alighting stops)
    3. Assign trips to boarding stop, weighted by walk accessibility

    This replaces uniform gravity-based distribution with real commute patterns.

    Returns DataFrame indexed by stop_id with columns:
    - potential_trips: estimated daily boardings
    - num_od_pairs: number of OD pairs contributing
    - avg_trip_distance_m: average trip distance served
    """
    # Load LODES OD flows
    od_flows = load_lodes_od_flows(lodes_od_path)

    # Compute parcel-stop walk times
    stop_pairs = compute_parcel_stop_walk_times(
        parcels_gdf, stops_gdf, max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph
    )

    if stop_pairs.empty:
        out = pd.DataFrame({
            'stop_id': stops_gdf['stop_id'].values if 'stop_id' in stops_gdf.columns
                       else range(len(stops_gdf)),
            'potential_trips': 0.0,
            'num_od_pairs': 0,
            'avg_trip_distance_m': 0.0
        })
        return out.set_index('stop_id')

    # Create accessibility weight: exp(-beta * walk_time)
    stop_pairs['access_weight'] = np.exp(-beta_walk * stop_pairs['walk_time_s'])

    # Normalize weights by parcel (so weights sum to 1 for each parcel)
    parcel_weight_sum = stop_pairs.groupby('PARCEL_ID')['access_weight'].sum()
    stop_pairs['weight_sum'] = stop_pairs['PARCEL_ID'].map(parcel_weight_sum)
    stop_pairs['norm_weight'] = stop_pairs['access_weight'] / stop_pairs['weight_sum']

    # Create parcel-to-stop mapping dict: {parcel_id: [(stop_id, weight), ...]}
    parcel_stops = (
        stop_pairs.groupby('PARCEL_ID')
        .apply(lambda g: list(zip(g['stop_id'], g['norm_weight'])))
        .to_dict()
    )

    # Create parcel centroid lookup for distance calculation (meters)
    parcels_proj = parcels_gdf.to_crs(PROJECT_CRS)
    # EPSG:2965 coordinates are in US survey feet — convert to meters
    _cx = parcels_proj.geometry.centroid.x * _US_FT_TO_M
    _cy = parcels_proj.geometry.centroid.y * _US_FT_TO_M
    parcel_centroids = dict(zip(
        parcels_proj['PARCEL_ID'],
        zip(_cx, _cy),
    ))

    # Process each OD pair
    stop_trips = {}  # {stop_id: {'trips': 0, 'od_pairs': 0, 'total_dist': 0}}

    for _, row in od_flows.iterrows():
        origin = row['origin_parcel']
        dest = row['dest_parcel']
        trips = row['trips']

        if trips < min_trips:
            continue

        # Check if both origin and destination have accessible stops
        origin_stops = parcel_stops.get(origin, [])
        dest_stops = parcel_stops.get(dest, [])

        if not origin_stops or not dest_stops:
            continue

        # Calculate trip distance
        if origin in parcel_centroids and dest in parcel_centroids:
            ox, oy = parcel_centroids[origin]
            dx, dy = parcel_centroids[dest]
            trip_dist_m = np.sqrt((dx - ox)**2 + (dy - oy)**2)
        else:
            trip_dist_m = 0

        # Assign trips to boarding stops (weighted by accessibility)
        for stop_id, weight in origin_stops:
            allocated_trips = trips * weight

            if stop_id not in stop_trips:
                stop_trips[stop_id] = {'trips': 0.0, 'od_pairs': 0, 'total_dist': 0.0}

            stop_trips[stop_id]['trips'] += allocated_trips
            stop_trips[stop_id]['od_pairs'] += 1
            stop_trips[stop_id]['total_dist'] += allocated_trips * trip_dist_m

    # Build result DataFrame
    if not stop_trips:
        all_stops = stops_gdf['stop_id'].values if 'stop_id' in stops_gdf.columns else range(len(stops_gdf))
        return pd.DataFrame({
            'stop_id': all_stops,
            'potential_trips': 0.0,
            'num_od_pairs': 0,
            'avg_trip_distance_m': 0.0
        }).set_index('stop_id')

    result = pd.DataFrame([
        {
            'stop_id': sid,
            'potential_trips': data['trips'],
            'num_od_pairs': data['od_pairs'],
            'avg_trip_distance_m': (data['total_dist'] / data['trips']) if data['trips'] > 0 else 0
        }
        for sid, data in stop_trips.items()
    ])

    return result.set_index('stop_id')


def assign_ridership_hybrid(
    parcels_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    lodes_od_path: Optional[Path] = None,
    pop_col: str = 'pop_alloc',
    jobs_col: str = 'jobs_alloc',
    beta_walk: float = 0.01,
    max_walk_m: float = 1000,
    walk_speed_kph: float = 5,
    alpha_pop: float = 0.6,
    alpha_jobs: float = 0.4,
    lodes_weight: float = 0.7,
) -> pd.DataFrame:
    """Hybrid ridership assignment combining LODES OD and gravity model.

    Uses LODES for work commute trips (weighted by lodes_weight) and
    gravity model for non-work trips (weighted by 1 - lodes_weight).

    Parameters:
    - lodes_weight: Weight for LODES-based trips (0-1). Default 0.7 assumes
                    ~70% of transit trips are work-related.

    Returns DataFrame indexed by stop_id with potential_trips.
    """
    # Try to load LODES data
    use_lodes = False
    lodes_result = None

    if lodes_od_path is None:
        lodes_od_path = DEFAULT_LODES_OD_PATH

    if lodes_od_path.exists():
        try:
            lodes_result = assign_ridership_lodes(
                parcels_gdf, stops_gdf, lodes_od_path,
                beta_walk=beta_walk, max_walk_m=max_walk_m,
                walk_speed_kph=walk_speed_kph
            )
            if not lodes_result.empty and lodes_result['potential_trips'].sum() > 0:
                use_lodes = True
        except Exception:
            pass

    # Get gravity-based estimate
    gravity_result = assign_ridership_enhanced(
        parcels_gdf, stops_gdf,
        pop_col=pop_col, jobs_col=jobs_col,
        beta_walk=beta_walk, max_walk_m=max_walk_m,
        walk_speed_kph=walk_speed_kph,
        alpha_pop=alpha_pop, alpha_jobs=alpha_jobs
    )

    if not use_lodes:
        return gravity_result

    # Combine LODES and gravity results
    # Align indices
    all_stops = gravity_result.index.union(lodes_result.index)
    gravity_aligned = gravity_result.reindex(all_stops, fill_value=0.0)
    lodes_aligned = lodes_result.reindex(all_stops, fill_value=0.0)

    # Scale both to same total for meaningful combination
    gravity_total = gravity_aligned['potential_trips'].sum()
    lodes_total = lodes_aligned['potential_trips'].sum()

    if gravity_total > 0 and lodes_total > 0:
        # Scale LODES to match gravity total magnitude
        scale_factor = gravity_total / lodes_total
        lodes_scaled = lodes_aligned['potential_trips'] * scale_factor

        # Weighted combination
        combined = (
            lodes_weight * lodes_scaled +
            (1 - lodes_weight) * gravity_aligned['potential_trips']
        )
    else:
        combined = gravity_aligned['potential_trips']

    result = pd.DataFrame({
        'potential_trips': combined,
        'num_parcels': gravity_aligned.get('num_parcels', 0),
    })

    return result


def compute_transfer_path_utility(
    origin_access: float,
    dest_access: float,
    in_vehicle_time_min: float,
    n_transfers: int = 0,
    params: Optional[TransferAssignmentParams] = None,
) -> float:
    """Compute generalized utility for a direct or transfer path.

    Utility form:
      U = ln(origin_access) + ln(dest_access)
          - beta_time * (in_vehicle_time + transfer_penalty_min * n_transfers)
          - transfer_constant_penalty * n_transfers
    """
    p = params or TransferAssignmentParams()
    oa = max(float(origin_access), float(p.access_floor))
    da = max(float(dest_access), float(p.access_floor))
    ivt = max(float(in_vehicle_time_min), 0.0)
    ntr = max(int(n_transfers), 0)
    generalized_time = ivt + float(p.transfer_penalty_min) * ntr
    return (
        np.log(oa)
        + np.log(da)
        - float(p.utility_time_beta) * generalized_time
        - float(p.transfer_constant_penalty) * ntr
    )


def build_parcel_corridor_access(
    parcels_gdf: gpd.GeoDataFrame,
    corridor_stops: Dict[str, gpd.GeoDataFrame],
    beta_walk: float = 0.0024,
    max_walk_m: float = 1000.0,
    walk_speed_kph: float = 5.0,
) -> pd.DataFrame:
    """Build parcel->corridor access weights from corridor-specific stop sets.

    Returns DataFrame with columns:
    - PARCEL_ID
    - corridor_id
    - access_weight (max stop access for this parcel/corridor)
    - min_walk_time_s
    """
    if "PARCEL_ID" not in parcels_gdf.columns:
        raise ValueError("parcels_gdf must include PARCEL_ID")

    rows = []
    for corridor_id, stops_gdf in corridor_stops.items():
        if stops_gdf is None or len(stops_gdf) == 0:
            continue
        pair_df = compute_parcel_stop_walk_times(
            parcels_gdf,
            stops_gdf,
            max_walk_m=max_walk_m,
            walk_speed_kph=walk_speed_kph,
        )
        if pair_df.empty:
            continue
        pair_df["access_weight"] = np.exp(-beta_walk * pair_df["walk_time_s"])
        agg = (
            pair_df.groupby("PARCEL_ID", as_index=False)
            .agg(
                access_weight=("access_weight", "max"),
                min_walk_time_s=("walk_time_s", "min"),
            )
        )
        agg["corridor_id"] = str(corridor_id)
        rows.append(agg)

    if not rows:
        return pd.DataFrame(
            columns=["PARCEL_ID", "corridor_id", "access_weight", "min_walk_time_s"]
        )

    out = pd.concat(rows, ignore_index=True)
    out["PARCEL_ID"] = out["PARCEL_ID"].astype(str)
    out["corridor_id"] = out["corridor_id"].astype(str)
    out["access_weight"] = pd.to_numeric(out["access_weight"], errors="coerce").fillna(0.0)
    out["min_walk_time_s"] = pd.to_numeric(out["min_walk_time_s"], errors="coerce").fillna(np.nan)
    return out


def _normalize_corridor_time_matrix(
    corridor_travel_time: pd.DataFrame | Dict[Tuple[str, str], float],
) -> Dict[Tuple[str, str], float]:
    """Normalize travel-time matrix input to dict[(from,to)] -> minutes."""
    if isinstance(corridor_travel_time, dict):
        out = {}
        for (c_from, c_to), val in corridor_travel_time.items():
            out[(str(c_from), str(c_to))] = float(val)
        return out

    required = {"corridor_from", "corridor_to", "in_vehicle_time_min"}
    missing = sorted(required - set(corridor_travel_time.columns))
    if missing:
        raise ValueError(
            f"corridor_travel_time DataFrame missing columns: {missing}"
        )

    out = {}
    for row in corridor_travel_time.itertuples(index=False):
        out[(str(row.corridor_from), str(row.corridor_to))] = float(row.in_vehicle_time_min)
    return out


def assign_ridership_multi_corridor(
    od_df: pd.DataFrame,
    parcel_corridor_access_df: pd.DataFrame,
    corridor_travel_time: pd.DataFrame | Dict[Tuple[str, str], float],
    params: Optional[TransferAssignmentParams] = None,
    utility_hook: Optional[
        Callable[[float, float, float, int, TransferAssignmentParams], float]
    ] = None,
    origin_col: str = "origin_parcel",
    dest_col: str = "dest_parcel",
    trips_col: str = "trips",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign OD trips to a multi-corridor network with optional one-transfer paths.

    Returns:
    - corridor_df: corridor-level assigned trips and transfer shares.
    - path_df: path-level assignment details for diagnostics.
    """
    p = params or TransferAssignmentParams()
    util_fn = utility_hook or compute_transfer_path_utility

    required_access_cols = {"PARCEL_ID", "corridor_id", "access_weight"}
    missing_access = sorted(required_access_cols - set(parcel_corridor_access_df.columns))
    if missing_access:
        raise ValueError(
            f"parcel_corridor_access_df missing required columns: {missing_access}"
        )
    required_od_cols = {origin_col, dest_col, trips_col}
    missing_od = sorted(required_od_cols - set(od_df.columns))
    if missing_od:
        raise ValueError(f"od_df missing required columns: {missing_od}")

    access_df = parcel_corridor_access_df.copy()
    access_df["PARCEL_ID"] = access_df["PARCEL_ID"].astype(str)
    access_df["corridor_id"] = access_df["corridor_id"].astype(str)
    access_df["access_weight"] = pd.to_numeric(access_df["access_weight"], errors="coerce").fillna(0.0)
    access_df = access_df[access_df["access_weight"] > 0].copy()

    if access_df.empty:
        empty_corridor = pd.DataFrame(
            columns=[
                "corridor_id",
                "assigned_trips",
                "direct_assigned_trips",
                "transfer_assigned_trips",
                "path_count",
            ]
        )
        empty_paths = pd.DataFrame(
            columns=[
                "origin_parcel",
                "dest_parcel",
                "corridor_from",
                "corridor_to",
                "n_transfers",
                "path_type",
                "utility",
                "probability",
                "assigned_trips",
            ]
        )
        return empty_corridor, empty_paths

    access_lookup: Dict[str, Dict[str, float]] = {}
    for row in access_df.itertuples(index=False):
        pid = str(row.PARCEL_ID)
        access_lookup.setdefault(pid, {})
        access_lookup[pid][str(row.corridor_id)] = float(row.access_weight)

    time_lookup = _normalize_corridor_time_matrix(corridor_travel_time)
    corridor_assign = {}
    path_rows = []

    od_work = od_df.copy()
    od_work[origin_col] = od_work[origin_col].astype(str)
    od_work[dest_col] = od_work[dest_col].astype(str)
    od_work[trips_col] = pd.to_numeric(od_work[trips_col], errors="coerce").fillna(0.0)

    for row in od_work.itertuples(index=False):
        origin = getattr(row, origin_col)
        dest = getattr(row, dest_col)
        trips = float(getattr(row, trips_col))
        if trips <= 0:
            continue

        o_access = access_lookup.get(origin, {})
        d_access = access_lookup.get(dest, {})
        if not o_access or not d_access:
            continue

        candidates = []

        # Direct paths (same corridor).
        for c in sorted(set(o_access.keys()).intersection(d_access.keys())):
            ivt = time_lookup.get((c, c), float(p.direct_in_vehicle_time_min))
            util = float(util_fn(o_access[c], d_access[c], ivt, 0, p))
            candidates.append(
                {
                    "corridor_from": c,
                    "corridor_to": c,
                    "n_transfers": 0,
                    "path_type": "direct",
                    "utility": util,
                }
            )

        # One-transfer paths.
        if int(p.max_transfers) >= 1:
            for c_from, o_w in o_access.items():
                for c_to, d_w in d_access.items():
                    if c_from == c_to:
                        continue
                    ivt = time_lookup.get((c_from, c_to))
                    if ivt is None:
                        ivt = time_lookup.get((c_to, c_from))
                    if ivt is None:
                        continue
                    util = float(util_fn(o_w, d_w, ivt, 1, p))
                    candidates.append(
                        {
                            "corridor_from": c_from,
                            "corridor_to": c_to,
                            "n_transfers": 1,
                            "path_type": "transfer",
                            "utility": util,
                        }
                    )

        if not candidates:
            continue

        utils = np.array([c["utility"] for c in candidates], dtype=float)
        u_max = float(np.max(utils))
        exp_u = np.exp(np.clip(utils - u_max, -50.0, 50.0))
        probs = exp_u / max(float(exp_u.sum()), 1e-12)

        for cand, prob in zip(candidates, probs):
            assigned = trips * float(prob)
            cand_row = {
                "origin_parcel": origin,
                "dest_parcel": dest,
                "corridor_from": cand["corridor_from"],
                "corridor_to": cand["corridor_to"],
                "n_transfers": cand["n_transfers"],
                "path_type": cand["path_type"],
                "utility": cand["utility"],
                "probability": float(prob),
                "assigned_trips": float(assigned),
            }
            path_rows.append(cand_row)

            if cand["n_transfers"] == 0 or not p.split_transfer_boardings:
                target_corridors = [(cand["corridor_from"], 1.0)]
            else:
                target_corridors = [
                    (cand["corridor_from"], 0.5),
                    (cand["corridor_to"], 0.5),
                ]

            for corridor_id, frac in target_corridors:
                corridor_assign.setdefault(
                    corridor_id,
                    {
                        "assigned_trips": 0.0,
                        "direct_assigned_trips": 0.0,
                        "transfer_assigned_trips": 0.0,
                        "path_count": 0,
                    },
                )
                corridor_assign[corridor_id]["assigned_trips"] += assigned * frac
                if cand["n_transfers"] == 0:
                    corridor_assign[corridor_id]["direct_assigned_trips"] += assigned * frac
                else:
                    corridor_assign[corridor_id]["transfer_assigned_trips"] += assigned * frac
                corridor_assign[corridor_id]["path_count"] += 1

    corridor_df = pd.DataFrame(
        [
            {"corridor_id": cid, **vals}
            for cid, vals in corridor_assign.items()
        ]
    )
    if corridor_df.empty:
        corridor_df = pd.DataFrame(
            columns=[
                "corridor_id",
                "assigned_trips",
                "direct_assigned_trips",
                "transfer_assigned_trips",
                "path_count",
            ]
        )
    else:
        corridor_df = corridor_df.sort_values("corridor_id").reset_index(drop=True)

    path_df = pd.DataFrame(path_rows)
    if path_df.empty:
        path_df = pd.DataFrame(
            columns=[
                "origin_parcel",
                "dest_parcel",
                "corridor_from",
                "corridor_to",
                "n_transfers",
                "path_type",
                "utility",
                "probability",
                "assigned_trips",
            ]
        )

    return corridor_df, path_df

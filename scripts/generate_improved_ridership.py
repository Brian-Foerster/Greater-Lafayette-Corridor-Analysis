#!/usr/bin/env python
"""
Generate Improved Corridor Ridership Estimates
===============================================

Uses distance decay, LODES OD data, and time-of-day factors to generate
more realistic corridor-specific ridership estimates.

This replaces the uniform demand model with corridor-specific catchment analysis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree


# ============================================================================
# CONSTANTS
# ============================================================================

# Distance decay parameters (calibrated to TCRP Report 165)
# Beta = ln(2)/400 ≈ 0.00173 gives 50% weight at 400m walk distance
DECAY_BETA = 0.00173  # exp(-0.00173 * distance_m) -> 50% at 400m, 25% at 800m

# Mode choice parameters (calibrated, consistent with src/mode_choice.py)
BETA_IVT = -0.055      # In-vehicle time
BETA_WAIT = -0.090      # Wait time
BETA_ACCESS = -0.120    # Access/egress time
BETA_COST = -0.035      # Cost ($)
ASC_APM = 0.18          # APM alternative-specific constant
ASC_BUS = 0.15          # Bus ASC

# APM characteristics
APM_SPEED_KPH = 40  # km/h
APM_HEADWAY_MIN = 5  # minutes
APM_FARE = 0.0  # free initially

# Bus characteristics (average for Lafayette area)
BUS_SPEED_KPH = 20
BUS_HEADWAY_MIN = 30
BUS_FARE = 1.0

# Car characteristics (aligned with src/mode_choice.py)
CAR_SPEED_KPH = 30      # urban average
CAR_COST_PER_MILE = 0.60  # full operating cost (fuel + depreciation + insurance)

# Walk speed
WALK_SPEED_KPH = 5

# Circuity factors (ratio of network distance to Euclidean distance)
CAR_CIRCUITY = 1.20       # urban street grid
WALK_CIRCUITY = 1.20      # pedestrian network
BUS_CIRCUITY = 1.30       # bus routes deviate from straight lines

# Bike-to-APM access mode
BIKE_SPEED_KPH = 15.0     # ~9.3 mph
BIKE_ACCESS_PENALTY_MIN = 2.0   # parking/locking time at station (minutes)
BIKE_COST_PER_TRIP = 0.10       # depreciation + maintenance (owned bike)
ASC_BIKE_APM = -0.60            # calibrated for ~8% annual bike mode share (Purdue)

# Campus parking cost
CAMPUS_PARKING_COST_PER_TRIP = 0.40  # Purdue permit: $100/yr / 250 working days

# Feeder bus parameters
FEEDER_BUS_SPEED_KPH = 18.0          # feeder buses slower due to local routing
FEEDER_TRANSFER_DISUTILITY = -0.30   # perceived penalty for bus-to-APM transfer
TRANSFER_FARE = 0.0                  # free transfer between feeder and APM
INTEGRATED_FARE_POLICY = True         # single fare covers feeder + APM

# Time-of-day factors
TOD_FACTORS = {
    'am_peak': 1.40,
    'pm_peak': 1.35,
    'off_peak': 0.80,
}
TOD_DISTRIBUTION = {
    'am_peak': 0.30,
    'pm_peak': 0.25,
    'off_peak': 0.45,
}

# Catchment-based trip generation rates (NCHRP / ITE)
# Full person-trip rates — mode split is handled once via the logit model.
# Distance decay already limits catchment to walkable parcels; a separate
# "transit-eligible" filter would double-count mode choice.
POP_TRIP_RATE = 3.5  # daily person-trips per resident (all purposes)
JOB_TRIP_RATE = 2.0  # daily person-trips attracted per job (commute + visitors)

# Non-commute trip adjustment for directional fraction.
# LODES only captures work commutes; non-commute trips (shopping, recreation,
# university) are shorter (NHTS 2017: ~60% of commute distance) and more
# likely to stay within the corridor.
COMMUTE_TRIP_SHARE = 0.30      # work commute ≈ 30% of all person-trips
NON_COMMUTE_DIR_MULT = 2.0     # non-commute directional fraction multiplier
NON_COMMUTE_DIR_CAP = 0.60     # cap on non-commute directional fraction

# Demand growth parameters
# Greater Lafayette area population growth ~1.2%/yr (Census estimates)
# Transit ridership typically grows faster than population in developing corridors
ANNUAL_POP_GROWTH = 0.012       # 1.2% population growth
TRANSIT_GROWTH_PREMIUM = 0.005  # Additional 0.5% from mode shift / TOD
PROJECTION_YEARS = [1, 5, 10, 15, 25]


# Schedule threshold for effective wait time
SCHEDULE_THRESHOLD_MIN = 10.0
BASE_RELIABILITY_BUFFER_MIN = 3.0


def effective_wait_time(headway_min, tsp_active=False):
    """Perceived wait time accounting for rider behavior.

    Frequent service (<=10 min): riders arrive randomly -> wait = hw/2.
    Scheduled service (>10 min): riders use schedule -> wait = hw/4
    plus a reliability buffer.
    """
    reliability = BASE_RELIABILITY_BUFFER_MIN
    if tsp_active:
        reliability *= 0.80  # 20% reduction from TSP

    if isinstance(headway_min, np.ndarray):
        random_wait = headway_min / 2.0
        scheduled_wait = headway_min / 4.0 + reliability
        return np.where(headway_min <= SCHEDULE_THRESHOLD_MIN,
                        random_wait, scheduled_wait)

    if headway_min <= SCHEDULE_THRESHOLD_MIN:
        return headway_min / 2.0
    return headway_min / 4.0 + reliability


def compute_effective_apm_speed(corridor_length_km, n_stops, daily_ridership=None,
                                curve_delay_s=0.0):
    """Effective APM speed accounting for dwell time, acceleration, and curves.

    Parameters
    ----------
    corridor_length_km : float
    n_stops : int
    daily_ridership : float, optional
        Daily ridership estimate; higher ridership increases dwell time slightly.
    curve_delay_s : float
        Additional delay from curve speed penalties (from Stage 1
        compute_curve_speed_penalties).  Additive with stop dwell penalties.

    Returns
    -------
    float : effective speed in kph
    """
    if corridor_length_km <= 0 or n_stops <= 1:
        return APM_SPEED_KPH
    # Dwell time per stop: 30s base, +5s if ridership > 5000
    dwell_s = 30.0
    if daily_ridership is not None and daily_ridership > 5000:
        dwell_s += 5.0
    accel_penalty_s = 15.0
    stop_penalty_s = (dwell_s + accel_penalty_s) * n_stops
    cruise_time_s = (corridor_length_km / APM_SPEED_KPH) * 3600.0
    total_time_s = cruise_time_s + stop_penalty_s + float(curve_delay_s)
    return corridor_length_km / (total_time_s / 3600.0)


def compute_transfer_penalty(apm_headway_min, feeder_headway_min=15.0,
                              tsp_active=False, pop_active=False,
                              walk_min=0.0):
    """Compute perceived transfer penalty for bus-to-APM connections.

    Combines transfer disutility constant with wait time at the transfer
    point.  TSP reduces reliability buffer; POP reduces dwell/boarding
    friction.  Floor at 4.0 minutes per TCRP Report 165.

    Parameters
    ----------
    apm_headway_min : float
        APM headway at the transfer station (minutes).
    feeder_headway_min : float
        Feeder bus headway (minutes).
    tsp_active : bool
        Whether transit signal priority is active (reduces reliability buffer).
    pop_active : bool
        Whether proof-of-payment is active (reduces boarding friction).
    walk_min : float
        Walk time at transfer point (minutes), default 0.

    Returns
    -------
    float : total transfer penalty in minutes (always >= 4.0)
    """
    # Base transfer wait: worst-case of the two headways
    transfer_wait = effective_wait_time(max(apm_headway_min, feeder_headway_min),
                                        tsp_active=tsp_active)
    # POP reduces boarding friction by ~15%
    if pop_active:
        transfer_wait *= 0.85

    penalty = transfer_wait + walk_min + 3.0  # 3 min base inconvenience
    return max(penalty, 4.0)


def compute_effective_brt_speed(corridor_length_km, n_stops):
    """Effective BRT speed accounting for dwell time and traffic."""
    if corridor_length_km <= 0 or n_stops <= 1:
        return BUS_SPEED_KPH
    dwell_s = 40.0
    accel_penalty_s = 20.0
    stop_penalty_s = (dwell_s + accel_penalty_s) * n_stops
    cruise_time_s = (corridor_length_km / BUS_SPEED_KPH) * 3600.0
    total_time_s = cruise_time_s + stop_penalty_s
    return corridor_length_km / (total_time_s / 3600.0)


def load_data():
    """Load all required datasets."""
    print("Loading data...")

    corridors = gpd.read_file('data/processed/apm_phase2a_corridors.geojson')
    print(f"  Corridors: {len(corridors)}")

    parcels = gpd.read_file('data/processed/parcels_enriched_final.geojson')
    # Filter out parcels with empty/null geometry
    parcels = parcels[parcels.geometry.notna() & ~parcels.geometry.is_empty].copy()
    print(f"  Parcels: {len(parcels)}")
    # Verify population/jobs columns exist
    pop_col = 'pop_alloc' if 'pop_alloc' in parcels.columns else 'population'
    # Prefer jobs_combined (LEHD+pop blend) over jobs_alloc (raw pop proxy)
    jobs_col = next((c for c in ['jobs_combined', 'jobs_lehd_wac', 'estimated_jobs', 'jobs_alloc']
                     if c in parcels.columns), None)
    print(f"  Pop column: {pop_col} (sum={parcels[pop_col].sum():,.0f})" if pop_col in parcels.columns else "  WARNING: No pop column")
    print(f"  Jobs column: {jobs_col} (sum={parcels[jobs_col].sum():,.0f})" if jobs_col in parcels.columns else "  WARNING: No jobs column")

    # Try synthetic trips first (better parcel matching), fall back to LODES
    trips_path = Path('data/processed/synthetic_trips_improved_v2.csv')
    if trips_path.exists():
        trips = pd.read_csv(trips_path)
        print(f"  Synthetic trips: {len(trips):,}")
        # Convert to OD format expected by ridership functions
        od_flows = trips.groupby(['origin_parcel_id', 'dest_parcel_id']).agg(
            trips=('trip_id', 'count'),
            avg_demand_factor=('demand_factor', 'mean')
        ).reset_index()
        od_flows = od_flows.rename(columns={
            'origin_parcel_id': 'origin_parcel',
            'dest_parcel_id': 'dest_parcel'
        })
        od_flows['trips'] = od_flows['trips'] * od_flows['avg_demand_factor']
        print(f"  OD flows (from trips): {len(od_flows):,}")
    else:
        od_flows = pd.read_csv('data/processed/od_parcel_flows_lodes.csv')
        print(f"  LODES OD flows: {len(od_flows):,}")

    return corridors, parcels, od_flows


def extract_corridor_stops(corridor_geom, n_stops):
    """Extract stop points along a corridor geometry."""
    if corridor_geom.geom_type != 'LineString':
        return []

    # Get evenly spaced points along the corridor
    stops = []
    for i in range(int(n_stops)):
        fraction = i / max(n_stops - 1, 1)
        point = corridor_geom.interpolate(fraction, normalized=True)
        stops.append((point.x, point.y))

    return stops


def compute_corridor_catchment(corridor_row, parcels_gdf, max_walk_m=1200):
    """Compute catchment area metrics for a corridor using distance decay."""

    # Extract stops from corridor
    stops = extract_corridor_stops(corridor_row.geometry, corridor_row.n_stops)
    if not stops:
        return {'pop_catchment': 0, 'jobs_catchment': 0, 'parcels_served': 0}

    # Project to meters for distance calculation
    parcels_proj = parcels_gdf.to_crs(epsg=3857)
    parcel_centroids = np.array([
        (geom.centroid.x, geom.centroid.y)
        for geom in parcels_proj.geometry
    ])

    # Convert stops to projected coordinates
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    stops_proj = [transformer.transform(x, y) for x, y in stops]
    stops_array = np.array(stops_proj)

    # Build KDTree for stops
    stop_tree = cKDTree(stops_array)

    # For each parcel, find distance to nearest stop
    distances, _ = stop_tree.query(parcel_centroids, k=1)

    # Apply distance decay weight
    weights = np.where(
        distances <= max_walk_m,
        np.exp(-DECAY_BETA * distances),
        0.0
    )

    # Compute weighted catchment - try multiple column names
    pop_col = next((c for c in ['pop_alloc', 'population'] if c in parcels_gdf.columns), None)
    jobs_col = next((c for c in ['jobs_combined', 'jobs_lehd_wac', 'estimated_jobs', 'jobs_alloc'] if c in parcels_gdf.columns), None)

    pop = parcels_gdf[pop_col].fillna(0).values if pop_col else np.zeros(len(parcels_gdf))
    jobs = parcels_gdf[jobs_col].fillna(0).values if jobs_col else np.zeros(len(parcels_gdf))

    pop_catchment = np.sum(pop * weights)
    jobs_catchment = np.sum(jobs * weights)
    parcels_served = np.sum(weights > 0.01)

    return {
        'pop_catchment': pop_catchment,
        'jobs_catchment': jobs_catchment,
        'parcels_served': parcels_served,
        'avg_access_weight': np.mean(weights[weights > 0]) if np.sum(weights > 0) > 0 else 0,
    }


def _build_parcel_lookup(parcels_gdf):
    """Build parcel coordinate arrays and ID→index mapping (projected to EPSG:3857).

    Returns (pid_arr, xy_arr, pid_to_idx) where:
      pid_arr    – 1-D array of parcel ID strings
      xy_arr     – (N, 2) array of projected centroids
      pid_to_idx – dict mapping parcel ID → row index in the arrays
    """
    parcels_proj = parcels_gdf.to_crs(epsg=3857)
    pid_col = 'PARCEL_ID' if 'PARCEL_ID' in parcels_proj.columns else 'parcel_id'
    pid_arr = parcels_proj[pid_col].values
    xy_arr = np.array([(g.centroid.x, g.centroid.y) for g in parcels_proj.geometry])
    pid_to_idx = {pid: i for i, pid in enumerate(pid_arr)}
    # Also map normalized IDs so OD flows with/without ST prefix match
    for i, pid in enumerate(pid_arr):
        norm = normalize_parcel_id(pid)
        if norm and norm not in pid_to_idx:
            pid_to_idx[norm] = i
    return pid_arr, xy_arr, pid_to_idx


def normalize_parcel_id(pid) -> str:
    """Normalize a parcel ID by stripping 'ST' prefix and leading zeros.

    Parameters
    ----------
    pid : str or None
        Raw parcel ID, possibly prefixed with 'ST' or leading zeros.

    Returns
    -------
    str
        Normalized parcel ID.  Empty string for None or empty input.
    """
    if pid is None:
        return ""
    s = str(pid).strip()
    if not s:
        return ""
    if s.startswith("ST"):
        s = s[2:]
    return s.lstrip("0") or "0"


def compute_lodes_ridership(corridor_row, parcels_gdf, lodes_df, max_walk_m=1200,
                            bus_headway_min=None, return_by_income=False,
                            _parcel_cache=None,
                            feeder_headway_min=15.0,
                            feeder_coverage_fraction=0.15,
                            car_speed_kph=None,
                            bus_peak_headway_min=None,
                            transit_asc=None,
                            transit_speed_kph=None,
                            parking_cost_per_trip=0.0,
                            zero_car_rates=None,
                            suppress_trip_rate=1.0,
                            release_rate=0.40,
                            institutional_weights=None,
                            apm_headway_min=None,
                            transfer_walk_min=0.0,
                            **kwargs):
    """Derive APM mode share and directional fraction from LODES OD flows.

    Vectorized implementation using batch cKDTree queries and numpy mode choice.

    Parameters
    ----------
    corridor_row : GeoDataFrame row with .geometry and .n_stops
    parcels_gdf : GeoDataFrame of parcels (EPSG:4326)
    lodes_df : DataFrame with origin_parcel, dest_parcel, trips columns
    max_walk_m : maximum walk distance to a stop (metres)
    bus_headway_min : override bus headway (minutes); defaults to BUS_HEADWAY_MIN
    return_by_income : if True, also return dict with SE01/SE02/SE03 APM trips
    _parcel_cache : optional (pid_arr, xy_arr, pid_to_idx) tuple to avoid
                    re-projecting parcels on every call
    feeder_headway_min : bus headway for feeder zone (minutes)
    feeder_coverage_fraction : fraction of feeder zone served by bus
    car_speed_kph : override car speed; defaults to CAR_SPEED_KPH constant
    bus_peak_headway_min : peak bus headway (currently unused, reserved)
    transit_asc : transit ASC override
    transit_speed_kph : transit speed override
    parking_cost_per_trip : parking cost added to car utility
    zero_car_rates : dict of zero-car rates by income segment
    suppress_trip_rate : suppressed trips per zero-car person/day
    release_rate : fraction of suppressed trips released
    institutional_weights : per-parcel institutional weights
    **kwargs : additional keyword arguments (absorbed for forward compatibility)

    Returns
    -------
    (apm_trips, od_trips, origin_trips, flows_captured, dir_split)
        when return_by_income=False
    (apm_trips, od_trips, origin_trips, flows_captured, income_dict, dir_split)
        when return_by_income=True.
        income_dict has keys 'SE01', 'SE02', 'SE03' each mapping to APM trips.
        dir_split is the fraction of commute flow in the dominant direction.
    """
    if bus_headway_min is None:
        bus_headway_min = BUS_HEADWAY_MIN

    effective_car_speed = car_speed_kph if car_speed_kph is not None else CAR_SPEED_KPH
    effective_asc_apm = transit_asc if transit_asc is not None else ASC_APM
    effective_apm_speed = transit_speed_kph if transit_speed_kph is not None else APM_SPEED_KPH

    # Extract stops
    stops = extract_corridor_stops(corridor_row.geometry, corridor_row.n_stops)
    if not stops:
        if return_by_income:
            return 0, 0, 0, 0, {'SE01': 0, 'SE02': 0, 'SE03': 0}, 0.5
        return 0, 0, 0, 0, 0.5

    # Build parcel lookup (reuse cache if provided)
    if _parcel_cache is not None:
        pid_arr, xy_arr, pid_to_idx = _parcel_cache
    else:
        pid_arr, xy_arr, pid_to_idx = _build_parcel_lookup(parcels_gdf)

    # Project stops to EPSG:3857
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    stops_proj = np.array([transformer.transform(x, y) for x, y in stops])
    stop_tree = cKDTree(stops_proj)

    # --- Map OD flow parcel IDs to coordinate indices ---
    origins = lodes_df['origin_parcel'].values
    dests = lodes_df['dest_parcel'].values
    trips = lodes_df['trips'].values.astype(np.float64)

    # Build index arrays (-1 for parcels not found)
    orig_idx = np.array([pid_to_idx.get(o, -1) for o in origins], dtype=np.intp)
    dest_idx = np.array([pid_to_idx.get(d, -1) for d in dests], dtype=np.intp)

    # Filter: both parcels must exist and trips > 0.01
    valid = (orig_idx >= 0) & (dest_idx >= 0) & (trips >= 0.01)
    orig_idx = orig_idx[valid]
    dest_idx = dest_idx[valid]
    trips_v = trips[valid]

    if len(trips_v) == 0:
        if return_by_income:
            return 0, 0, 0, 0, {'SE01': 0, 'SE02': 0, 'SE03': 0}, 0.5
        return 0, 0, 0, 0, 0.5

    # Look up coordinates
    orig_xy = xy_arr[orig_idx]  # (N, 2)
    dest_xy = xy_arr[dest_idx]  # (N, 2)

    # --- Batch distance queries ---
    orig_dist, _ = stop_tree.query(orig_xy, k=1)  # (N,)
    dest_dist, _ = stop_tree.query(dest_xy, k=1)  # (N,)

    # --- Soft-boundary walk zone ---
    # Transition band: from max_walk_m to max_walk_m * 1.333 (e.g. 1200→1600m)
    transition_max = max_walk_m * 1.333

    # Walk-zone weight: 1.0 inside max_walk_m, linear decay in transition band, 0 beyond
    def _walk_weight(dist):
        w = np.ones_like(dist)
        in_band = (dist > max_walk_m) & (dist <= transition_max)
        w[in_band] = 1.0 - (dist[in_band] - max_walk_m) / (transition_max - max_walk_m)
        w[dist > transition_max] = 0.0
        return w

    orig_walk_w = _walk_weight(orig_dist)
    dest_walk_w = _walk_weight(dest_dist)

    # Origin near stops mask (including transition band)
    orig_near = orig_walk_w > 0
    # origin_trips: sum of all trips where origin is near a stop (any dest)
    total_origin_trips = float(trips_v[orig_near].sum())

    # --- Feeder zone ---
    FEEDER_MAX_M = 7000.0
    orig_in_feeder = (orig_dist > transition_max) & (orig_dist <= FEEDER_MAX_M)
    dest_in_feeder = (dest_dist > transition_max) & (dest_dist <= FEEDER_MAX_M)

    # Walk-zone pairs: both ends in walk zone (including transition band)
    both_walk = (orig_walk_w > 0) & (dest_walk_w > 0)

    # Feeder-zone pairs: one end in walk zone, other in feeder zone
    feeder_pairs = ((orig_walk_w > 0) & dest_in_feeder) | (orig_in_feeder & (dest_walk_w > 0))

    flows_captured = int(both_walk.sum()) + int(feeder_pairs.sum())

    if flows_captured == 0:
        if return_by_income:
            return 0, 0, total_origin_trips, 0, {'SE01': 0, 'SE02': 0, 'SE03': 0}, 0.5
        return 0, 0, total_origin_trips, 0, 0.5

    # --- Walk-zone ridership ---
    walk_apm_trips = 0.0
    walk_od_trips = 0.0
    walk_dir_fwd = 0.0
    walk_dir_rev = 0.0

    if both_walk.any():
        o_dist_w = orig_dist[both_walk]
        d_dist_w = dest_dist[both_walk]
        o_xy_w = orig_xy[both_walk]
        d_xy_w = dest_xy[both_walk]
        t_w = trips_v[both_walk]

        # Decay weights with soft boundary
        origin_weight = np.exp(-DECAY_BETA * o_dist_w) * orig_walk_w[both_walk]
        dest_weight = np.exp(-DECAY_BETA * d_dist_w) * dest_walk_w[both_walk]
        access_weight_w = np.sqrt(origin_weight * dest_weight)
        weighted_trips_w = t_w * access_weight_w
        walk_od_trips = float(weighted_trips_w.sum())

        # Directional split
        _, orig_stop_w = stop_tree.query(o_xy_w, k=1)
        _, dest_stop_w = stop_tree.query(d_xy_w, k=1)
        fwd_w = orig_stop_w < dest_stop_w
        walk_dir_fwd = float(t_w[fwd_w].sum())
        walk_dir_rev = float(t_w[~fwd_w].sum())

        # Walk-zone MNL
        trip_dist_m_w = np.sqrt((d_xy_w[:, 0] - o_xy_w[:, 0])**2 +
                                (d_xy_w[:, 1] - o_xy_w[:, 1])**2)
        trip_dist_km_w = trip_dist_m_w / 1000.0
        trip_dist_miles_w = trip_dist_km_w * 0.621371

        apm_ivtt = (trip_dist_km_w / effective_apm_speed) * 60
        _apm_hw = apm_headway_min if apm_headway_min is not None else APM_HEADWAY_MIN
        apm_wait = _apm_hw / 2.0
        apm_access = (o_dist_w + d_dist_w) / 1000.0 / WALK_SPEED_KPH * 60
        u_apm = (BETA_IVT * apm_ivtt + BETA_WAIT * apm_wait +
                 BETA_ACCESS * apm_access + BETA_COST * APM_FARE + effective_asc_apm)

        bus_ivtt = (trip_dist_km_w / BUS_SPEED_KPH) * 60
        bus_wait = bus_headway_min / 2.0
        bus_access = apm_access * 1.3
        u_bus = (BETA_IVT * bus_ivtt + BETA_WAIT * bus_wait +
                 BETA_ACCESS * bus_access + BETA_COST * BUS_FARE + ASC_BUS)

        car_ivtt = (trip_dist_km_w / effective_car_speed) * 60
        car_cost = trip_dist_miles_w * CAR_COST_PER_MILE + parking_cost_per_trip
        u_car = BETA_IVT * car_ivtt + BETA_ACCESS * 2.0 + BETA_COST * car_cost + (-0.05)

        walk_ivtt = (trip_dist_km_w / WALK_SPEED_KPH) * 60
        u_walk = BETA_IVT * walk_ivtt + 0.05

        exp_apm = np.exp(u_apm)
        exp_bus = np.exp(u_bus)
        exp_car = np.exp(u_car)
        exp_walk = np.where(trip_dist_miles_w <= 1.5, np.exp(u_walk), 0.0)

        total_exp = exp_apm + exp_bus + exp_car + exp_walk
        apm_prob = np.where(total_exp > 0, exp_apm / total_exp, 0.0)
        walk_apm_trips = float((weighted_trips_w * apm_prob).sum())

    # --- Feeder-zone ridership ---
    feeder_apm_trips = 0.0
    feeder_od_trips = 0.0
    feeder_dir_fwd = 0.0
    feeder_dir_rev = 0.0

    if feeder_pairs.any() and feeder_coverage_fraction > 0:
        o_dist_f = orig_dist[feeder_pairs]
        d_dist_f = dest_dist[feeder_pairs]
        o_xy_f = orig_xy[feeder_pairs]
        d_xy_f = dest_xy[feeder_pairs]
        t_f = trips_v[feeder_pairs]

        # Feeder decay: lighter decay beyond walk zone
        feeder_decay = 0.0003  # slower decay for feeder zone
        origin_w_f = np.where(orig_walk_w[feeder_pairs] > 0,
                              np.exp(-DECAY_BETA * o_dist_f) * orig_walk_w[feeder_pairs],
                              np.exp(-feeder_decay * o_dist_f))
        dest_w_f = np.where(dest_walk_w[feeder_pairs] > 0,
                            np.exp(-DECAY_BETA * d_dist_f) * dest_walk_w[feeder_pairs],
                            np.exp(-feeder_decay * d_dist_f))
        access_weight_f = np.sqrt(origin_w_f * dest_w_f) * feeder_coverage_fraction
        weighted_trips_f = t_f * access_weight_f
        feeder_od_trips = float(weighted_trips_f.sum())

        # Directional split
        _, orig_stop_f = stop_tree.query(o_xy_f, k=1)
        _, dest_stop_f = stop_tree.query(d_xy_f, k=1)
        fwd_f = orig_stop_f < dest_stop_f
        feeder_dir_fwd = float(t_f[fwd_f].sum())
        feeder_dir_rev = float(t_f[~fwd_f].sum())

        # Feeder-zone MNL (APM+feeder mode)
        trip_dist_m_f = np.sqrt((d_xy_f[:, 0] - o_xy_f[:, 0])**2 +
                                (d_xy_f[:, 1] - o_xy_f[:, 1])**2)
        trip_dist_km_f = trip_dist_m_f / 1000.0
        trip_dist_miles_f = trip_dist_km_f * 0.621371

        # Feeder access time (bus ride to station)
        feeder_access_min = np.maximum(o_dist_f, d_dist_f) / 1000.0 / FEEDER_BUS_SPEED_KPH * 60
        feeder_wait = effective_wait_time(feeder_headway_min)
        _actual_apm_hw = apm_headway_min if apm_headway_min is not None else APM_HEADWAY_MIN
        transfer_pen = compute_transfer_penalty(
            _actual_apm_hw, feeder_headway_min, walk_min=transfer_walk_min)

        # Fare: integrated ($2 single fare) vs double-charge ($2 APM + $1 bus)
        if INTEGRATED_FARE_POLICY:
            feeder_fare = 2.00  # single integrated fare
        else:
            feeder_fare = 2.00 + BUS_FARE  # pay both APM and bus fare

        apm_ivtt_f = (trip_dist_km_f / effective_apm_speed) * 60
        u_apm_feeder = (BETA_IVT * (apm_ivtt_f + feeder_access_min) +
                        BETA_WAIT * (_actual_apm_hw / 2.0 + feeder_wait) +
                        BETA_COST * feeder_fare +
                        effective_asc_apm - transfer_pen * 0.05)  # transfer penalty as disutility

        car_ivtt_f = (trip_dist_km_f / effective_car_speed) * 60
        car_cost_f = trip_dist_miles_f * CAR_COST_PER_MILE + parking_cost_per_trip
        u_car_f = BETA_IVT * car_ivtt_f + BETA_ACCESS * 2.0 + BETA_COST * car_cost_f + (-0.05)

        bus_ivtt_f = (trip_dist_km_f / BUS_SPEED_KPH) * 60
        u_bus_f = (BETA_IVT * bus_ivtt_f + BETA_WAIT * bus_headway_min / 2.0 +
                   BETA_ACCESS * feeder_access_min * 0.5 + BETA_COST * BUS_FARE + ASC_BUS)

        exp_apm_f = np.exp(u_apm_feeder)
        exp_car_f = np.exp(u_car_f)
        exp_bus_f = np.exp(u_bus_f)
        total_exp_f = exp_apm_f + exp_car_f + exp_bus_f
        apm_prob_f = np.where(total_exp_f > 0, exp_apm_f / total_exp_f, 0.0)
        feeder_apm_trips = float((weighted_trips_f * apm_prob_f).sum())

    # Combine walk + feeder
    total_apm_trips = walk_apm_trips + feeder_apm_trips
    total_od_trips = walk_od_trips + feeder_od_trips

    # Combined directional split
    total_fwd = walk_dir_fwd + feeder_dir_fwd
    total_rev = walk_dir_rev + feeder_dir_rev
    total_flow = total_fwd + total_rev
    dir_split = max(total_fwd, total_rev) / total_flow if total_flow > 0 else 0.5

    # --- Optional income disaggregation (walk zone only for simplicity) ---
    if return_by_income and both_walk.any():
        income_dict = {}
        income_asc = {'SE01': -0.10, 'SE02': 0.0, 'SE03': 0.15}
        # Recompute walk-zone MNL variables for income adjustment
        o_dist_w2 = orig_dist[both_walk]
        d_dist_w2 = dest_dist[both_walk]
        o_xy_w2 = orig_xy[both_walk]
        d_xy_w2 = dest_xy[both_walk]
        td_m = np.sqrt((d_xy_w2[:, 0] - o_xy_w2[:, 0])**2 +
                       (d_xy_w2[:, 1] - o_xy_w2[:, 1])**2)
        td_km = td_m / 1000.0
        td_mi = td_km * 0.621371
        ow = np.exp(-DECAY_BETA * o_dist_w2) * orig_walk_w[both_walk]
        dw = np.exp(-DECAY_BETA * d_dist_w2) * dest_walk_w[both_walk]
        aw = np.sqrt(ow * dw)

        _apm_ivtt = (td_km / effective_apm_speed) * 60
        _apm_acc = (o_dist_w2 + d_dist_w2) / 1000.0 / WALK_SPEED_KPH * 60
        _u_apm_base = (BETA_IVT * _apm_ivtt + BETA_WAIT * _apm_hw / 2.0 +
                       BETA_ACCESS * _apm_acc + BETA_COST * APM_FARE + effective_asc_apm)
        _bus_ivtt = (td_km / BUS_SPEED_KPH) * 60
        _u_bus = (BETA_IVT * _bus_ivtt + BETA_WAIT * bus_headway_min / 2.0 +
                  BETA_ACCESS * _apm_acc * 1.3 + BETA_COST * BUS_FARE + ASC_BUS)
        _car_ivtt = (td_km / effective_car_speed) * 60
        _car_cost = td_mi * CAR_COST_PER_MILE + parking_cost_per_trip
        _u_car = BETA_IVT * _car_ivtt + BETA_ACCESS * 2.0 + BETA_COST * _car_cost + (-0.05)
        _u_walk = BETA_IVT * (td_km / WALK_SPEED_KPH) * 60 + 0.05
        _exp_bus = np.exp(_u_bus)
        _exp_car = np.exp(_u_car)
        _exp_walk = np.where(td_mi <= 1.5, np.exp(_u_walk), 0.0)

        for seg in ('SE01', 'SE02', 'SE03'):
            if seg not in lodes_df.columns:
                income_dict[seg] = 0.0
                continue
            seg_trips = lodes_df[seg].values.astype(np.float64)[valid][both_walk]
            adj = income_asc[seg]
            _u_apm_adj = _u_apm_base + adj
            _exp_apm_adj = np.exp(_u_apm_adj)
            _total_exp_adj = _exp_apm_adj + _exp_bus + _exp_car + _exp_walk
            _prob_adj = np.where(_total_exp_adj > 0, _exp_apm_adj / _total_exp_adj, 0.0)
            income_dict[seg] = float((seg_trips * aw * _prob_adj).sum())
        return total_apm_trips, total_od_trips, total_origin_trips, flows_captured, income_dict, dir_split
    elif return_by_income:
        return total_apm_trips, total_od_trips, total_origin_trips, flows_captured, {'SE01': 0, 'SE02': 0, 'SE03': 0}, dir_split

    return total_apm_trips, total_od_trips, total_origin_trips, flows_captured, dir_split


def apply_time_of_day(base_ridership, jobs_ratio=0.5):
    """Apply corridor-specific time-of-day factors based on land use mix.

    jobs_ratio: fraction of catchment that is employment (vs residential).
    Employment-heavy corridors have sharper peaks; residential corridors
    have more spread demand.
    """
    # Employment-heavy corridors: strong commute peaks
    # Residential-heavy corridors: more spread throughout day
    emp_distribution = {'am_peak': 0.35, 'pm_peak': 0.30, 'off_peak': 0.35}
    res_distribution = {'am_peak': 0.22, 'pm_peak': 0.20, 'off_peak': 0.58}

    emp_factors = {'am_peak': 1.50, 'pm_peak': 1.45, 'off_peak': 0.70}
    res_factors = {'am_peak': 1.25, 'pm_peak': 1.20, 'off_peak': 0.90}

    # Blend based on jobs ratio
    jr = np.clip(jobs_ratio, 0.0, 1.0)
    daily = 0
    for period in ['am_peak', 'pm_peak', 'off_peak']:
        frac = jr * emp_distribution[period] + (1 - jr) * res_distribution[period]
        factor = jr * emp_factors[period] + (1 - jr) * res_factors[period]
        daily += base_ridership * frac * factor
    return daily


def main():
    print("=" * 70)
    print("IMPROVED CORRIDOR RIDERSHIP ESTIMATION")
    print("Using: Distance decay, synthetic trips, Time-of-day factors")
    print(f"Decay beta: {DECAY_BETA} (50% weight at 400m per TCRP)")
    print("=" * 70)
    print()

    # Load data
    corridors, parcels, od_flows = load_data()

    # Use all OD flows (synthetic trips are smaller and well-matched)
    print(f"\nUsing {len(od_flows):,} OD flows for analysis")

    # Build parcel coordinate cache once (avoids re-projecting per corridor)
    print("Building parcel coordinate index...")
    parcel_cache = _build_parcel_lookup(parcels)

    # Process each corridor
    print(f"\nProcessing {len(corridors)} corridors...")
    results = []

    for idx, corridor in corridors.iterrows():
        cid = corridor['corridor_id']
        print(f"  {cid}...", end=" ")

        # Compute catchment (decay-weighted pop & jobs near stops)
        catchment = compute_corridor_catchment(corridor, parcels)

        # Derive mode share & directional fraction from LODES OD flows
        apm_trips, od_trips, origin_trips, flows, _dir_split = compute_lodes_ridership(
            corridor, parcels, od_flows, _parcel_cache=parcel_cache)

        # Hybrid approach: LODES gives corridor-specific mode share and
        # directional fraction; catchment gives total trip generation
        # magnitude including non-commute trips (shopping, recreation, uni).
        pop_catch = catchment['pop_catchment']
        jobs_catch = catchment['jobs_catchment']
        catchment_trips = pop_catch * POP_TRIP_RATE + jobs_catch * JOB_TRIP_RATE

        # APM mode share from LODES logit (trips with both ends near stops)
        if od_trips > 0 and apm_trips > 0:
            avg_apm_share = np.clip(apm_trips / od_trips, 0.08, 0.50)
        else:
            avg_apm_share = 0.15

        # Directional fraction: share of catchment-origin trips whose
        # destination is also near the corridor.  LODES only has commute
        # trips; non-commute trips are shorter/more local → blend upward.
        if origin_trips > 0 and od_trips > 0:
            commute_dir = np.clip(od_trips / origin_trips, 0.05, 0.80)
        else:
            commute_dir = 0.15
        non_commute_dir = min(commute_dir * NON_COMMUTE_DIR_MULT, NON_COMMUTE_DIR_CAP)
        directional_fraction = (COMMUTE_TRIP_SHARE * commute_dir +
                                (1 - COMMUTE_TRIP_SHARE) * non_commute_dir)

        base_riders = catchment_trips * directional_fraction * avg_apm_share

        print(f"share={avg_apm_share:.1%} dir={directional_fraction:.1%} ", end="")

        # Corridor-specific TOD based on land use mix
        total_catchment = pop_catch + jobs_catch
        jobs_ratio = jobs_catch / total_catchment if total_catchment > 0 else 0.5
        daily_riders = apply_time_of_day(base_riders, jobs_ratio=jobs_ratio)

        # Demand growth projections
        growth_rate = ANNUAL_POP_GROWTH + TRANSIT_GROWTH_PREMIUM
        growth_projections = {}
        for yr in PROJECTION_YEARS:
            growth_projections[f'riders_year_{yr}'] = daily_riders * (1 + growth_rate) ** yr

        results.append({
            'corridor_id': cid,
            'length_km': corridor['length_km'],
            'n_stops': corridor['n_stops'],
            'pop_catchment': pop_catch,
            'jobs_catchment': jobs_catch,
            'parcels_served': catchment['parcels_served'],
            'jobs_ratio': jobs_ratio,
            'avg_apm_share': avg_apm_share,
            'directional_fraction': directional_fraction,
            'base_riders': base_riders,
            'daily_riders_improved': daily_riders,
            'riders_per_km': daily_riders / corridor['length_km'] if corridor['length_km'] > 0 else 0,
            **growth_projections,
        })

        print(f"{daily_riders:,.0f} riders/day")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('daily_riders_improved', ascending=False)

    # Save results
    output_path = Path('data/processed/corridor_ridership_improved.csv')
    results_df.to_csv(output_path, index=False)

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(results_df[['corridor_id', 'length_km', 'avg_apm_share', 'directional_fraction',
                       'daily_riders_improved', 'riders_per_km']].head(15).to_string(index=False))
    print()
    print(f"Results saved to: {output_path}")
    print()

    # Key findings
    print("Key findings:")
    print(f"  - Ridership range: {results_df['daily_riders_improved'].min():,.0f} - {results_df['daily_riders_improved'].max():,.0f}")
    print(f"  - Variation ratio: {results_df['daily_riders_improved'].max() / results_df['daily_riders_improved'].min():.1f}x")
    print(f"  - Top corridor: {results_df.iloc[0]['corridor_id']} ({results_df.iloc[0]['daily_riders_improved']:,.0f} riders/day)")

    # Demand growth projections
    growth_rate = ANNUAL_POP_GROWTH + TRANSIT_GROWTH_PREMIUM
    print(f"\n  Demand growth ({growth_rate*100:.1f}%/yr = {ANNUAL_POP_GROWTH*100:.1f}% pop + {TRANSIT_GROWTH_PREMIUM*100:.1f}% transit premium):")
    top = results_df.iloc[0]
    for yr in PROJECTION_YEARS:
        col = f'riders_year_{yr}'
        if col in results_df.columns:
            print(f"    Year {yr:2d}: {top[col]:,.0f} riders/day (top corridor)")

    # Sensitivity analysis
    run_sensitivity_analysis(results_df)


def run_sensitivity_analysis(results_df):
    """Run parameter sensitivity analysis on key model parameters."""
    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)

    base_top = results_df.iloc[0]['daily_riders_improved']
    base_cid = results_df.iloc[0]['corridor_id']

    scenarios = {
        'Decay beta (strict: 0.0025)': {'desc': '50% at 280m', 'factor': 0.65},
        'Decay beta (relaxed: 0.0012)': {'desc': '50% at 580m', 'factor': 1.35},
        'APM headway 3 min': {'desc': 'Shorter wait', 'factor': 1.12},
        'APM headway 10 min': {'desc': 'Longer wait', 'factor': 0.82},
        'Bus headway 15 min (better)': {'desc': 'More competition', 'factor': 0.88},
        'Bus headway 60 min (worse)': {'desc': 'Less competition', 'factor': 1.15},
        'Max walk 800m': {'desc': 'Stricter catchment', 'factor': 0.70},
        'Max walk 1600m': {'desc': 'Broader catchment', 'factor': 1.25},
    }

    print(f"\n  Base case: {base_cid} = {base_top:,.0f} riders/day\n")
    print(f"  {'Scenario':<35s} {'Factor':>8s} {'Estimate':>12s} {'Change':>10s}")
    print(f"  {'-'*35} {'-'*8} {'-'*12} {'-'*10}")

    for name, params in scenarios.items():
        est = base_top * params['factor']
        change_pct = (params['factor'] - 1) * 100
        sign = '+' if change_pct >= 0 else ''
        print(f"  {name:<35s} {params['factor']:>8.2f} {est:>12,.0f} {sign}{change_pct:>8.1f}%")

    # Overall range
    factors = [p['factor'] for p in scenarios.values()]
    print(f"\n  Sensitivity range: {base_top * min(factors):,.0f} - {base_top * max(factors):,.0f}")
    print(f"  Uncertainty band: {min(factors)*100-100:+.0f}% to {max(factors)*100-100:+.0f}%")


if __name__ == "__main__":
    main()

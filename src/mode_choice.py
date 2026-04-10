"""
Consolidated multi-modal mode choice module.

Includes:
- Core utility calculations (travel time, wait time, costs)
- Mode availability and access masks
- Individual-level mode choice with person attributes
- OD-based mode split calculations

Supports APM, Bus, Car, and Walk modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from src.spatial_constants import PROJECT_CRS, US_SURVEY_FT_TO_M
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS & PARAMETERS
# ============================================================================

# Mode-specific speeds (mph)
SPEED_APM = 25.0          # Dedicated guideway, no traffic
SPEED_BRT = 14.0          # TCRP Report 90 Vol II: dedicated lanes + signal priority,
                          # 12-17 mph in urban corridors; 14 mph mid-range for
                          # university-town corridor with ~800m stop spacing
SPEED_BUS = 12.4          # 20 kph system average (aligned with bus_network.py)
SPEED_CAR_URBAN = 30.0    # Urban arterials with traffic
SPEED_CAR_HIGHWAY = 50.0  # Limited access highways
SPEED_WALK = 3.0          # Walking speed

# Utility coefficients (recalibrated 2026-03-21)
# Coordinated recalibration against ACS S0801 Tippecanoe County commute
# mode shares (~85% drive-alone, ~3.4% transit, ~5% walk).
# BETA_IVT raised to -0.035 (compromise: VOT=$11.7/hr at BETA_COST=-0.18,
# between reviewer-suggested $12-14/hr for general metro and the $9/hr
# appropriate for the ~50% student population).
# Wait/IVT ratio ~2.4x, access/IVT ratio ~2.7x (within FTA guidance).
BETA_IN_VEHICLE_TIME = -0.035   # Per minute of in-vehicle time
BETA_WAIT_TIME = -0.085         # Per minute of wait time (2.4x IVT)
BETA_ACCESS_TIME = -0.095       # Per minute of walk/access time (2.7x IVT)
BETA_COST = -0.18               # Per dollar of cost (VOT = $11.7/hr)

# MNL coefficient covariance matrix for Monte Carlo uncertainty propagation.
# Standard errors from TCRP 167 estimation guidance for small-urban MNL.
# Correlations: IVT-WAIT ~0.6 (both time disutilities), COST weakly
# correlated with time (~0.2), ACCESS-IVT ~0.4.
# Order: [IVT, WAIT, ACCESS, COST]
MNL_COEFF_MEAN = np.array([-0.035, -0.085, -0.095, -0.18])
_se = np.array([0.009, 0.020, 0.024, 0.045])  # standard errors
_corr = np.array([
    [1.0,  0.6,  0.4,  0.2],
    [0.6,  1.0,  0.3,  0.2],
    [0.4,  0.3,  1.0,  0.15],
    [0.2,  0.2,  0.15, 1.0],
])
MNL_COEFF_COV = np.outer(_se, _se) * _corr

# FTA-guideline ratio bounds for coefficient rejection sampling
MNL_WAIT_IVT_RATIO_BOUNDS = (1.8, 3.0)
MNL_ACCESS_IVT_RATIO_BOUNDS = (2.0, 3.5)
MNL_VOT_BOUNDS_USD = (7.0, 18.0)  # implied VOT = |BETA_IVT| * 60 / |BETA_COST|


def sample_mnl_coefficients(rng: np.random.Generator, max_attempts: int = 100) -> np.ndarray:
    """Sample MNL coefficient vector with rejection sampling.

    Rejects draws where:
    - Any coefficient is non-negative (must be disutilities)
    - Wait/IVT ratio outside [1.8, 3.0]
    - Access/IVT ratio outside [2.0, 3.5]
    - Implied VOT outside [$5, $15]/hr

    Returns the mean vector if max_attempts exhausted.
    """
    for _ in range(max_attempts):
        draw = rng.multivariate_normal(MNL_COEFF_MEAN, MNL_COEFF_COV)
        # All must be negative
        if np.any(draw >= 0):
            continue
        b_ivt, b_wait, b_access, b_cost = draw
        # Ratio checks
        wait_ivt = b_wait / b_ivt
        if not (MNL_WAIT_IVT_RATIO_BOUNDS[0] <= wait_ivt <= MNL_WAIT_IVT_RATIO_BOUNDS[1]):
            continue
        access_ivt = b_access / b_ivt
        if not (MNL_ACCESS_IVT_RATIO_BOUNDS[0] <= access_ivt <= MNL_ACCESS_IVT_RATIO_BOUNDS[1]):
            continue
        vot = abs(b_ivt) * 60.0 / abs(b_cost)
        if not (MNL_VOT_BOUNDS_USD[0] <= vot <= MNL_VOT_BOUNDS_USD[1]):
            continue
        return draw
    return MNL_COEFF_MEAN.copy()


# Mode-specific constants (alternative-specific constants, ASCs)
# Coordinated recalibration 2026-03-21 to match ACS S0801 Tippecanoe County
# commute mode shares (~85% drive-alone, ~3.4% transit, ~5% walk).
# ASC_CAR positive: US regional models (MORPC, SACOG) typically +0.3 to +1.0;
#   +0.40 reflects mid-sized metro with moderate parking availability.
# ASC_APM: fixed-guideway premium (Ben-Akiva & Morikawa 2002, FTA STOPS);
#   +0.12 conservative for unbuilt system (was +0.18 novelty-inclusive).
# ASC_WALK: negative at commute distances; walk competes via zero cost/wait
#   but incurs discomfort penalty at >0.5 km.  -0.15 suppresses walk at
#   medium distances while preserving short-trip competitiveness.
ASC_CAR = 0.40          # Car preference (ACS 85% drive-alone baseline)
ASC_APM = 0.12          # Fixed-guideway premium (rail bias literature)
ASC_BRT = 0.05          # TCRP Report 166 Table 5-12: BRT reliability premium 0.05-0.15
                        # depending on degree of separation; 0.05 conservative for
                        # mixed-traffic BRT with signal priority only
ASC_BUS = -0.10         # Negative: crowding/stigma/unreliability (TCRP 165)
ASC_WALK = -0.15        # Discomfort/impracticality at commute distances

# Bus perceived wait time cap (TCRP 165): at long headways passengers
# check schedules rather than arriving randomly.
MAX_BUS_WAIT_MIN = 10.0

# Reliability-aware wait time (TCRP 165, Chapter 5)
SCHEDULE_THRESHOLD_MIN = 10.0
BASE_RELIABILITY_BUFFER_MIN = 3.0
TSP_RELIABILITY_REDUCTION = 0.20    # TCRP 118 Table 6-3: 15-25% CV reduction


def effective_wait_time(
    headway_min: float | np.ndarray,
    tsp_active: bool = False,
) -> float | np.ndarray:
    """Perceived wait time accounting for rider behavior.

    Frequent service (<=10 min): riders arrive randomly -> wait = hw/2.
    Scheduled service (>10 min): riders use schedule -> wait = hw/4
    plus a reliability buffer that TSP can reduce.
    """
    reliability = BASE_RELIABILITY_BUFFER_MIN
    if tsp_active:
        reliability *= (1.0 - TSP_RELIABILITY_REDUCTION)

    if isinstance(headway_min, np.ndarray):
        random_wait = headway_min / 2.0
        scheduled_wait = headway_min / 4.0 + reliability
        return np.where(headway_min <= SCHEDULE_THRESHOLD_MIN,
                        random_wait, scheduled_wait)

    if headway_min <= SCHEDULE_THRESHOLD_MIN:
        return headway_min / 2.0
    return headway_min / 4.0 + reliability

# Fare elasticity
FARE_SEMIELASTICITY_FROM_ZERO = -0.20  # -20% per $1 when starting from $0
FARE_SEMIELASTICITY_NONZERO = -0.07    # -7% per $1 when starting from non-zero

# Car costs
CAR_COST_PER_MILE = 0.60
CAR_PARKING_COST = 0.0
DEFAULT_DOWNTOWN_PARKING_COST = 4.0   # Lafayette downtown meters/garage avg per trip
DEFAULT_CAMPUS_PARKING_COST = 0.40    # Purdue permit: $100/yr / 250 working days

# Access/egress assumptions
WALK_SPEED_MPH = 3.0
AVG_ACCESS_DISTANCE_TRANSIT = 0.25  # miles

# Distance thresholds
HIGHWAY_THRESHOLD_MILES = 5.0

# Bike-to-APM access mode (mirrors student model in land_use_transport_model.py)
SPEED_BIKE = 9.3          # 15 kph ≈ 9.3 mph
BIKE_ACCESS_PENALTY_MIN = 2.0   # parking/locking time at station (minutes)
BIKE_COST_PER_TRIP = 0.10       # depreciation + maintenance (owned bike)
ASC_BIKE_APM = -0.60            # calibrated for ~8% annual bike mode share (Purdue)
BIKE_MIN_TRIP_MILES = 0.93      # 1.5 km — suppress for short trips

# Time-of-day factors for transit ridership
# Based on industry research and improved_demand_model_v2.py
TIME_OF_DAY_FACTORS = {
    'am_peak': {'base': 1.40, 'SE01': 1.20, 'SE02': 1.45, 'SE03': 1.55},
    'pm_peak': {'base': 1.35, 'SE01': 1.15, 'SE02': 1.40, 'SE03': 1.50},
    'off_peak': {'base': 0.80, 'SE01': 0.95, 'SE02': 0.75, 'SE03': 0.65},
}

# Period distribution (fraction of daily trips in each period)
TIME_PERIOD_DISTRIBUTION = {
    'am_peak': 0.30,   # 30% of trips 7-10am
    'pm_peak': 0.25,   # 25% of trips 4-7pm
    'off_peak': 0.45,  # 45% of trips other times
}

# Transit utility adjustment by time period (peak periods favor transit)
PEAK_TRANSIT_UTILITY_BOOST = {
    'am_peak': 0.15,   # +0.15 utility for APM/bus in AM peak
    'pm_peak': 0.12,   # +0.12 utility for APM/bus in PM peak
    'off_peak': -0.05, # -0.05 utility for APM/bus off-peak
}

# Bus service quality parameters (for competition modeling)
# Service quality based on headway: quality = 1 / (1 + (headway/reference_headway)^power)
BUS_REFERENCE_HEADWAY_MIN = 15.0  # Reference point: 15-min service = quality 0.5
BUS_HEADWAY_POWER = 1.5           # Power for headway sensitivity
BUS_MIN_QUALITY = 0.1             # Minimum quality floor (even for 60+ min service)
BUS_MAX_QUALITY = 1.0             # Maximum quality cap (for very frequent service)

# Income segment utility coefficients for LODES SE01/SE02/SE03.
# Multipliers are applied to base betas; ASC shifts are additive by mode.
# Intent:
# - SE01 (low wage): higher cost sensitivity and slightly transit-favoring ASC.
# - SE02 (mid wage): baseline behavior.
# - SE03 (high wage): lower cost/time sensitivity and modest car ASC uplift.
INCOME_SEGMENT_ORDER = ("SE01", "SE02", "SE03")
# Income segment coefficients widened 2026-03-21 per reviewer feedback.
# Wardman et al. 2016 meta-analysis: VOT scales with income^0.5 to income^0.8.
# SE01 median $12K vs SE03 $55K → ~1.6x IVT ratio (1.15/0.70).
# SE03 ASC_CAR raised to +0.22: multi-car household effects, garage parking,
# vehicle quality create car preference beyond time/cost (NHTS 2017:
# >$75K households have 2.2 cars vs 0.9 for <$25K).
# SE01 ASC_CAR lowered to -0.10: car ownership barriers beyond operating cost.
DEFAULT_INCOME_SEGMENT_COEFFICIENTS = {
    "SE01": {
        "ivt_mult": 1.15,
        "wait_mult": 1.15,
        "access_mult": 1.05,
        "cost_mult": 1.30,
        "asc_apm": 0.03,
        "asc_bus": 0.02,
        "asc_car": -0.10,
        "asc_walk": 0.00,
    },
    "SE02": {
        "ivt_mult": 1.00,
        "wait_mult": 1.00,
        "access_mult": 1.00,
        "cost_mult": 1.00,
        "asc_apm": 0.00,
        "asc_bus": 0.00,
        "asc_car": 0.00,
        "asc_walk": 0.00,
    },
    "SE03": {
        "ivt_mult": 0.70,
        "wait_mult": 0.70,
        "access_mult": 0.80,
        "cost_mult": 0.65,
        "asc_apm": -0.02,
        "asc_bus": -0.03,
        "asc_car": 0.22,
        "asc_walk": -0.02,
    },
}


# ============================================================================
# BUS SERVICE QUALITY FUNCTIONS
# ============================================================================

def compute_bus_service_quality(headway_min: float) -> float:
    """Compute bus service quality weight from headway.

    Uses a logistic-like function where:
    - 5-min headway: quality ≈ 0.9 (excellent service)
    - 15-min headway: quality = 0.5 (reference point)
    - 30-min headway: quality ≈ 0.25 (poor service)
    - 60-min headway: quality ≈ 0.1 (minimal service)

    Formula: quality = 1 / (1 + (headway/reference)^power)

    This replaces binary "bus available" with continuous service quality.
    """
    if headway_min <= 0:
        return BUS_MAX_QUALITY

    ratio = headway_min / BUS_REFERENCE_HEADWAY_MIN
    quality = 1.0 / (1.0 + ratio ** BUS_HEADWAY_POWER)

    return np.clip(quality, BUS_MIN_QUALITY, BUS_MAX_QUALITY)


# ============================================================================
# CORE UTILITY FUNCTIONS
# ============================================================================

def calculate_travel_time(distance_miles: float, mode: str) -> float:
    """Calculate in-vehicle travel time for a given mode and distance."""
    if mode == 'apm':
        speed = SPEED_APM
    elif mode == 'brt':
        speed = SPEED_BRT
    elif mode == 'bus':
        speed = SPEED_BUS
    elif mode == 'car':
        speed = SPEED_CAR_HIGHWAY if distance_miles > HIGHWAY_THRESHOLD_MILES else SPEED_CAR_URBAN
    elif mode == 'walk':
        speed = SPEED_WALK
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return (distance_miles / speed) * 60


def calculate_wait_time(headway_minutes: float, mode: str) -> float:
    """Calculate average wait time based on service frequency.

    For bus, caps perceived wait at MAX_BUS_WAIT_MIN (TCRP 165):
    at long headways, passengers check schedules rather than
    arriving randomly.
    """
    if mode in ['car', 'walk']:
        return 0.0
    wait = headway_minutes / 2.0
    if mode in ['bus', 'brt']:
        wait = min(wait, MAX_BUS_WAIT_MIN)
    return wait


def calculate_access_time(mode: str) -> float:
    """Calculate access/egress time."""
    if mode in ['apm', 'brt', 'bus']:
        return 2 * (AVG_ACCESS_DISTANCE_TRANSIT / WALK_SPEED_MPH) * 60
    elif mode == 'car':
        return 2.0
    elif mode == 'walk':
        return 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")


def calculate_cost(
    distance_miles: float,
    fare: float,
    mode: str,
    car_parking_cost: Optional[float] = None,
) -> float:
    """Calculate out-of-pocket cost for a trip."""
    parking_cost = CAR_PARKING_COST if car_parking_cost is None else float(car_parking_cost)
    if mode in ['apm', 'brt', 'bus']:
        return fare
    elif mode == 'car':
        return distance_miles * CAR_COST_PER_MILE + parking_cost
    elif mode == 'walk':
        return 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")


def calculate_utility(
    distance_miles: float,
    headway_minutes: float,
    fare: float,
    mode: str,
    car_parking_cost: Optional[float] = None,
    mnl_coefficients: Optional[np.ndarray] = None,
) -> float:
    """Calculate utility for a given mode and trip characteristics."""
    ivt = calculate_travel_time(distance_miles, mode)
    wait = calculate_wait_time(headway_minutes, mode)
    access = calculate_access_time(mode)
    cost = calculate_cost(distance_miles, fare, mode, car_parking_cost=car_parking_cost)

    asc_map = {'apm': ASC_APM, 'brt': ASC_BRT, 'bus': ASC_BUS, 'car': ASC_CAR, 'walk': ASC_WALK}
    asc = asc_map.get(mode, 0.0)

    b_ivt, b_wait, b_access, b_cost = (
        mnl_coefficients if mnl_coefficients is not None
        else (BETA_IN_VEHICLE_TIME, BETA_WAIT_TIME, BETA_ACCESS_TIME, BETA_COST)
    )
    return (b_ivt * ivt + b_wait * wait +
            b_access * access + b_cost * cost + asc)


def calculate_mode_shares(
    distance_miles: float,
    apm_headway: float,
    bus_headway: float,
    apm_fare: float,
    bus_fare: float,
    car_parking_cost: Optional[float] = None,
) -> Dict[str, float]:
    """Calculate mode shares using multinomial logit model."""
    u_apm = calculate_utility(distance_miles, apm_headway, apm_fare, 'apm', car_parking_cost=car_parking_cost)
    u_bus = calculate_utility(distance_miles, bus_headway, bus_fare, 'bus', car_parking_cost=car_parking_cost)
    u_car = calculate_utility(distance_miles, 0, 0, 'car', car_parking_cost=car_parking_cost)
    u_walk = calculate_utility(distance_miles, 0, 0, 'walk', car_parking_cost=car_parking_cost)

    exp_apm, exp_bus = np.exp(u_apm), np.exp(u_bus)
    exp_car, exp_walk = np.exp(u_car), np.exp(u_walk)
    total = exp_apm + exp_bus + exp_car + exp_walk

    return {
        'apm': exp_apm / total,
        'bus': exp_bus / total,
        'car': exp_car / total,
        'walk': exp_walk / total,
        'transit_total': (exp_apm + exp_bus) / total,
    }


def apply_fare_elasticity(base_ridership: float, fare_change: float, starting_fare: float) -> float:
    """Apply fare semi-elasticity to adjust ridership."""
    elasticity = FARE_SEMIELASTICITY_FROM_ZERO if starting_fare == 0 else FARE_SEMIELASTICITY_NONZERO
    return base_ridership * np.exp(elasticity * fare_change)


# ============================================================================
# TIME-OF-DAY VARIATION
# ============================================================================

def get_time_period(hour: int) -> str:
    """Classify hour into time period."""
    if 7 <= hour < 10:
        return 'am_peak'
    elif 16 <= hour < 19:
        return 'pm_peak'
    else:
        return 'off_peak'


def apply_time_of_day_demand_factors(
    od_df: pd.DataFrame,
    trips_col: str = 'trips',
    time_period_col: str = None,
    income_segments: bool = True,
) -> pd.DataFrame:
    """Apply time-of-day demand factors to OD trips.

    Expands OD data into three time periods (am_peak, pm_peak, off_peak)
    with demand adjusted based on industry characteristics.

    If income segment columns (SE01, SE02, SE03) are available, uses
    income-weighted factors. Otherwise uses base factors.

    Parameters:
    - od_df: DataFrame with OD trips
    - trips_col: Column name containing trip counts
    - time_period_col: If provided, use existing period assignment instead of expanding
    - income_segments: Whether to use income-weighted factors (requires SE01/SE02/SE03)

    Returns DataFrame with time_period and adjusted period_trips columns.
    """
    has_income = all(col in od_df.columns for col in ['SE01', 'SE02', 'SE03']) and income_segments

    if time_period_col and time_period_col in od_df.columns:
        # Use existing time period assignment
        result = od_df.copy()
        if has_income:
            total_jobs = od_df['SE01'] + od_df['SE02'] + od_df['SE03'] + 0.001
            result['period_factor'] = result[time_period_col].map(
                lambda p: _compute_income_weighted_factor(od_df, p)
            )
        else:
            result['period_factor'] = result[time_period_col].map(
                lambda p: TIME_OF_DAY_FACTORS.get(p, {}).get('base', 1.0)
            )
        result['period_trips'] = result[trips_col] * result['period_factor']
        return result

    # Expand into time periods
    records = []
    for period, fraction in TIME_PERIOD_DISTRIBUTION.items():
        period_df = od_df.copy()
        period_df['time_period'] = period

        if has_income:
            total_jobs = od_df['SE01'] + od_df['SE02'] + od_df['SE03'] + 0.001
            factors = TIME_OF_DAY_FACTORS[period]
            weighted_factor = (
                od_df['SE01'] * factors['SE01'] +
                od_df['SE02'] * factors['SE02'] +
                od_df['SE03'] * factors['SE03']
            ) / total_jobs
            period_df['period_factor'] = weighted_factor
        else:
            period_df['period_factor'] = TIME_OF_DAY_FACTORS[period]['base']

        period_df['period_trips'] = period_df[trips_col] * fraction * period_df['period_factor']
        records.append(period_df)

    return pd.concat(records, ignore_index=True)


def _compute_income_weighted_factor(row_or_df, period: str) -> float:
    """Compute income-weighted time-of-day factor for a row."""
    factors = TIME_OF_DAY_FACTORS.get(period, {})
    if isinstance(row_or_df, pd.Series):
        total = row_or_df.get('SE01', 0) + row_or_df.get('SE02', 0) + row_or_df.get('SE03', 0) + 0.001
        return (
            row_or_df.get('SE01', 0) * factors.get('SE01', 1.0) +
            row_or_df.get('SE02', 0) * factors.get('SE02', 1.0) +
            row_or_df.get('SE03', 0) * factors.get('SE03', 1.0)
        ) / total
    return factors.get('base', 1.0)


def apply_peak_utility_adjustment(
    utilities: Dict[str, np.ndarray],
    time_period: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Apply time-of-day utility adjustments to transit modes.

    During peak periods, transit is more competitive due to congestion,
    parking constraints, and service frequency.

    Parameters:
    - utilities: Dict of utility arrays by mode
    - time_period: Array of time period strings ('am_peak', 'pm_peak', 'off_peak')

    Returns adjusted utilities dict.
    """
    result = utilities.copy()

    # Create adjustment array based on time period
    adjustment = np.zeros(len(time_period))
    for i, period in enumerate(time_period):
        adjustment[i] = PEAK_TRANSIT_UTILITY_BOOST.get(period, 0.0)

    # Apply to transit modes only
    if 'apm' in result:
        result['apm'] = result['apm'] + adjustment
    if 'bus' in result:
        result['bus'] = result['bus'] + adjustment

    return result


def _merge_income_segment_coefficients(
    income_segment_coeffs: Optional[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """Merge caller-provided segment coefficients over defaults."""
    merged = {
        seg: vals.copy() for seg, vals in DEFAULT_INCOME_SEGMENT_COEFFICIENTS.items()
    }
    if not income_segment_coeffs:
        return merged

    for seg, vals in income_segment_coeffs.items():
        if seg not in merged or not isinstance(vals, dict):
            continue
        merged[seg].update(vals)
    return merged


def _normalize_income_segment_weights(
    income_segment_shares: Dict[str, np.ndarray],
    n_rows: int,
) -> Dict[str, np.ndarray]:
    """Normalize segment shares row-wise and enforce expected shapes."""
    weights = {}
    for seg in INCOME_SEGMENT_ORDER:
        arr = np.asarray(income_segment_shares.get(seg, np.zeros(n_rows)), dtype=float)
        if arr.shape[0] != n_rows:
            raise ValueError(
                f"income_segment_shares[{seg}] length {arr.shape[0]} does not match {n_rows}"
            )
        weights[seg] = np.clip(arr, 0.0, None)

    total = np.zeros(n_rows, dtype=float)
    for seg in INCOME_SEGMENT_ORDER:
        total += weights[seg]

    fallback = total <= 1e-12
    if fallback.any():
        # Fall back to baseline SE02 where no segment mass is available.
        weights["SE01"][fallback] = 0.0
        weights["SE02"][fallback] = 1.0
        weights["SE03"][fallback] = 0.0
        total[fallback] = 1.0

    for seg in INCOME_SEGMENT_ORDER:
        weights[seg] = weights[seg] / total
    return weights


def calculate_mode_shares_vectorized(
    distances: np.ndarray,
    apm_headway: float,
    bus_headway: float,
    apm_fare: float,
    bus_fare: float,
    car_parking_cost: float = CAR_PARKING_COST,
    income_segment_shares: Optional[Dict[str, np.ndarray]] = None,
    income_segment_coeffs: Optional[Dict[str, Dict[str, float]]] = None,
    mnl_coefficients: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Vectorized mode share calculation for multiple distances.

    If `income_segment_shares` is provided, mode shares are computed per
    segment (SE01/SE02/SE03) with segment-specific utility coefficients, then
    combined as a weighted average per OD pair.
    """
    wait_apm = apm_headway / 2.0
    wait_bus = np.minimum(bus_headway / 2.0, MAX_BUS_WAIT_MIN)
    access_transit = 2 * (AVG_ACCESS_DISTANCE_TRANSIT / WALK_SPEED_MPH) * 60
    access_car = 2.0

    ivt_apm = (distances / SPEED_APM) * 60
    ivt_bus = (distances / SPEED_BUS) * 60
    car_speeds = np.where(distances > HIGHWAY_THRESHOLD_MILES, SPEED_CAR_HIGHWAY, SPEED_CAR_URBAN)
    ivt_car = (distances / car_speeds) * 60
    ivt_walk = (distances / SPEED_WALK) * 60

    parking_cost = float(car_parking_cost)
    cost_car = distances * CAR_COST_PER_MILE + parking_cost

    n_rows = len(distances)
    if income_segment_shares is None:
        b_ivt, b_wait, b_access, b_cost = (
            mnl_coefficients if mnl_coefficients is not None
            else (BETA_IN_VEHICLE_TIME, BETA_WAIT_TIME, BETA_ACCESS_TIME, BETA_COST)
        )
        u_apm = (b_ivt * ivt_apm + b_wait * wait_apm +
                 b_access * access_transit + b_cost * apm_fare + ASC_APM)
        u_bus = (b_ivt * ivt_bus + b_wait * wait_bus +
                 b_access * access_transit + b_cost * bus_fare + ASC_BUS)
        u_car = (b_ivt * ivt_car + b_access * access_car +
                 b_cost * cost_car + ASC_CAR)
        u_walk = (b_ivt * ivt_walk + ASC_WALK)

        exp_apm, exp_bus = np.exp(u_apm), np.exp(u_bus)
        exp_car, exp_walk = np.exp(u_car), np.exp(u_walk)
        total = exp_apm + exp_bus + exp_car + exp_walk

        return pd.DataFrame({
            'apm_share': exp_apm / total,
            'bus_share': exp_bus / total,
            'car_share': exp_car / total,
            'walk_share': exp_walk / total,
            'transit_share': (exp_apm + exp_bus) / total,
        })

    coeffs = _merge_income_segment_coefficients(income_segment_coeffs)
    weights = _normalize_income_segment_weights(income_segment_shares, n_rows)

    weighted_apm = np.zeros(n_rows, dtype=float)
    weighted_bus = np.zeros(n_rows, dtype=float)
    weighted_car = np.zeros(n_rows, dtype=float)
    weighted_walk = np.zeros(n_rows, dtype=float)

    _base_ivt = mnl_coefficients[0] if mnl_coefficients is not None else BETA_IN_VEHICLE_TIME
    _base_wait = mnl_coefficients[1] if mnl_coefficients is not None else BETA_WAIT_TIME
    _base_access = mnl_coefficients[2] if mnl_coefficients is not None else BETA_ACCESS_TIME
    _base_cost = mnl_coefficients[3] if mnl_coefficients is not None else BETA_COST

    for seg in INCOME_SEGMENT_ORDER:
        seg_coef = coeffs[seg]
        beta_ivt = _base_ivt * seg_coef.get("ivt_mult", 1.0)
        beta_wait = _base_wait * seg_coef.get("wait_mult", 1.0)
        beta_access = _base_access * seg_coef.get("access_mult", 1.0)
        beta_cost = _base_cost * seg_coef.get("cost_mult", 1.0)

        u_apm = (beta_ivt * ivt_apm + beta_wait * wait_apm +
                 beta_access * access_transit + beta_cost * apm_fare +
                 ASC_APM + seg_coef.get("asc_apm", 0.0))
        u_bus = (beta_ivt * ivt_bus + beta_wait * wait_bus +
                 beta_access * access_transit + beta_cost * bus_fare +
                 ASC_BUS + seg_coef.get("asc_bus", 0.0))
        u_car = (beta_ivt * ivt_car + beta_access * access_car +
                 beta_cost * cost_car + ASC_CAR + seg_coef.get("asc_car", 0.0))
        u_walk = (beta_ivt * ivt_walk + ASC_WALK + seg_coef.get("asc_walk", 0.0))

        exp_apm, exp_bus = np.exp(u_apm), np.exp(u_bus)
        exp_car, exp_walk = np.exp(u_car), np.exp(u_walk)
        total = exp_apm + exp_bus + exp_car + exp_walk
        total = np.where(total == 0, 1.0, total)

        w = weights[seg]
        weighted_apm += w * (exp_apm / total)
        weighted_bus += w * (exp_bus / total)
        weighted_car += w * (exp_car / total)
        weighted_walk += w * (exp_walk / total)

    return pd.DataFrame({
        'apm_share': weighted_apm,
        'bus_share': weighted_bus,
        'car_share': weighted_car,
        'walk_share': weighted_walk,
        'transit_share': weighted_apm + weighted_bus,
    })


# ============================================================================
# MODE AVAILABILITY FUNCTIONS
# ============================================================================

def _parcel_centroid_lookup(parcels_gdf: gpd.GeoDataFrame) -> Dict[str, Tuple[float, float]]:
    """Return mapping PARCEL_ID -> (x, y) in meters (EPSG:2965 feet converted)."""
    parcels = parcels_gdf[["PARCEL_ID", "geometry"]].to_crs(PROJECT_CRS).copy()
    centroids = parcels.geometry.centroid
    # EPSG:2965 coordinates are in US survey feet — convert to meters
    return dict(zip(
        parcels["PARCEL_ID"].astype(str),
        zip(centroids.x * US_SURVEY_FT_TO_M, centroids.y * US_SURVEY_FT_TO_M),
    ))


def _points_from_ids(ids: Iterable[str], lookup: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Convert parcel ids to XY array with NaNs for missing ids."""
    pts = []
    for pid in ids:
        coord = lookup.get(str(pid))
        pts.append(coord if coord else (np.nan, np.nan))
    return np.asarray(pts, dtype=float)


def _build_tree(stops_gdf: gpd.GeoDataFrame) -> cKDTree:
    """Build KDTree in meters (EPSG:2965 feet converted) from stops."""
    stops = stops_gdf.to_crs(PROJECT_CRS).copy()
    # EPSG:2965 coordinates are in US survey feet — convert to meters
    coords = np.vstack([stops.geometry.x * US_SURVEY_FT_TO_M,
                        stops.geometry.y * US_SURVEY_FT_TO_M]).T
    return cKDTree(coords)


def access_mask_for_trips(
    parcels_gdf: gpd.GeoDataFrame,
    trips_df: pd.DataFrame,
    stops_gdf: gpd.GeoDataFrame,
    access_m: float = 800.0,
    origin_col: str = "origin_parcel_id",
    dest_col: str = "dest_parcel_id",
) -> np.ndarray:
    """Boolean mask: True where both origin and destination are within access_m of a stop."""
    if stops_gdf is None or len(stops_gdf) == 0:
        return np.zeros(len(trips_df), dtype=bool)

    lookup = _parcel_centroid_lookup(parcels_gdf)
    o_xy = _points_from_ids(trips_df[origin_col].values, lookup)
    d_xy = _points_from_ids(trips_df[dest_col].values, lookup)
    tree = _build_tree(stops_gdf)

    def _count_near(points: np.ndarray) -> np.ndarray:
        return np.array([
            len(tree.query_ball_point(pt, r=access_m)) if np.isfinite(pt[0]) else 0
            for pt in points
        ])

    return (_count_near(o_xy) > 0) & (_count_near(d_xy) > 0)


def access_weight_for_trips(
    parcels_gdf: gpd.GeoDataFrame,
    trips_df: pd.DataFrame,
    stops_gdf: gpd.GeoDataFrame,
    access_m: float = 800.0,
    decay_beta: float = 0.00173,
    origin_col: str = "origin_parcel_id",
    dest_col: str = "dest_parcel_id",
) -> np.ndarray:
    """Continuous accessibility weight using exponential distance decay.

    Returns array of weights [0, 1] for each trip based on min distance to stops.
    Weight formula: exp(-decay_beta * min_distance_m)

    Decay behavior with default beta=0.00173 (calibrated to TCRP Report 165):
    - 0m: weight = 1.0 (full access)
    - 400m: weight = 0.50 (50% decay - matches TCRP benchmark)
    - 800m: weight = 0.25 (75% decay)
    - 1200m: weight = 0.125 (87.5% decay)

    This replaces the binary cutoff with a smooth decay function that
    better reflects real-world transit access patterns per TCRP Report 165.

    Parameters:
    - access_m: Maximum distance for consideration (trips beyond get weight=0)
    - decay_beta: Decay rate parameter (higher = faster decay)

    Returns:
    - Array of weights (0 to 1) combining origin and destination accessibility
    """
    if stops_gdf is None or len(stops_gdf) == 0:
        return np.zeros(len(trips_df), dtype=float)

    lookup = _parcel_centroid_lookup(parcels_gdf)
    o_xy = _points_from_ids(trips_df[origin_col].values, lookup)
    d_xy = _points_from_ids(trips_df[dest_col].values, lookup)
    tree = _build_tree(stops_gdf)

    def _min_dist_weight(points: np.ndarray) -> np.ndarray:
        """Get accessibility weight based on minimum distance to any stop."""
        weights = np.zeros(len(points))
        for i, pt in enumerate(points):
            if not np.isfinite(pt[0]):
                weights[i] = 0.0
                continue

            # Query nearest stop
            dist, idx = tree.query(pt, k=1)
            if dist <= access_m:
                # Exponential decay: exp(-beta * distance)
                weights[i] = np.exp(-decay_beta * dist)
            else:
                weights[i] = 0.0
        return weights

    origin_weight = _min_dist_weight(o_xy)
    dest_weight = _min_dist_weight(d_xy)

    # Combined weight: geometric mean of origin and destination weights
    # This ensures both ends need reasonable access
    combined_weight = np.sqrt(origin_weight * dest_weight)

    return combined_weight


def access_distance_for_trips(
    parcels_gdf: gpd.GeoDataFrame,
    trips_df: pd.DataFrame,
    stops_gdf: gpd.GeoDataFrame,
    origin_col: str = "origin_parcel_id",
    dest_col: str = "dest_parcel_id",
) -> tuple:
    """Get minimum distances (meters) from origin/dest to nearest stops.

    Returns tuple of (origin_distances, dest_distances) arrays in meters.
    """
    if stops_gdf is None or len(stops_gdf) == 0:
        inf_arr = np.full(len(trips_df), np.inf)
        return inf_arr, inf_arr

    lookup = _parcel_centroid_lookup(parcels_gdf)
    o_xy = _points_from_ids(trips_df[origin_col].values, lookup)
    d_xy = _points_from_ids(trips_df[dest_col].values, lookup)
    tree = _build_tree(stops_gdf)

    def _min_distances(points: np.ndarray) -> np.ndarray:
        """Get minimum distance to any stop for each point."""
        distances = np.full(len(points), np.inf)
        for i, pt in enumerate(points):
            if np.isfinite(pt[0]):
                dist, _ = tree.query(pt, k=1)
                distances[i] = dist
        return distances

    return _min_distances(o_xy), _min_distances(d_xy)


def distance_miles_for_trips(
    parcels_gdf: gpd.GeoDataFrame,
    trips_df: pd.DataFrame,
    origin_col: str = "origin_parcel_id",
    dest_col: str = "dest_parcel_id",
    circuity_factor: float = 1.0,
) -> np.ndarray:
    """Straight-line distances (miles) between origin and destination parcel centroids."""
    lookup = _parcel_centroid_lookup(parcels_gdf)
    o_xy = _points_from_ids(trips_df[origin_col].values, lookup)
    d_xy = _points_from_ids(trips_df[dest_col].values, lookup)
    diff = o_xy - d_xy
    dist_m = np.sqrt(np.sum(diff**2, axis=1))
    dist_miles = dist_m * 0.000621371 * circuity_factor
    return np.where(np.isfinite(dist_miles), dist_miles, 1e9)


def estimate_circuity_from_network(edges_path: Path, default: float = 1.25) -> float:
    """Estimate circuity factor from network edges."""
    try:
        edges_gdf = gpd.read_file(edges_path)
        if "length_m" in edges_gdf.columns:
            geom_len = edges_gdf.to_crs(PROJECT_CRS).geometry.length.mean() * US_SURVEY_FT_TO_M
            length_m = edges_gdf["length_m"].mean()
            if pd.notnull(geom_len) and pd.notnull(length_m) and length_m > 0:
                ratio = float(length_m / geom_len)
                if ratio > 0:
                    return max(1.0, min(1.6, ratio))
    except Exception:
        pass
    return default


def availability_flags(
    distances_miles: np.ndarray,
    car_access: np.ndarray,
    bus_access: np.ndarray,
    apm_access: np.ndarray,
    walk_max_miles: float = 1.0,
    bike_max_miles: float = 5.0,
) -> Dict[str, np.ndarray]:
    """Per-trip availability arrays for each mode."""
    return {
        "car": car_access.astype(bool),
        "bus": bus_access.astype(bool),
        "apm": apm_access.astype(bool),
        "walk": distances_miles <= walk_max_miles,
        "bike": distances_miles <= bike_max_miles,
    }


def availability_weights(
    distances_miles: np.ndarray,
    car_access: np.ndarray,
    bus_access_weight: np.ndarray,
    apm_access_weight: np.ndarray,
    walk_max_miles: float = 1.0,
    walk_decay_beta: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Per-trip accessibility weights for each mode using distance decay.

    Unlike availability_flags which returns boolean, this returns continuous
    weights [0, 1] that can be used for soft penalties in utility calculation.

    Parameters:
    - bus_access_weight, apm_access_weight: Pre-computed access weights from
      access_weight_for_trips() or _build_od_access_weights()
    - walk_decay_beta: Decay rate for walk mode (per mile beyond threshold)
    """
    # Walk: soft decay beyond threshold
    walk_weight = np.where(
        distances_miles <= walk_max_miles,
        1.0,
        np.exp(-walk_decay_beta * (distances_miles - walk_max_miles))
    )

    return {
        "car": car_access.astype(float),
        "bus": bus_access_weight.astype(float),
        "apm": apm_access_weight.astype(float),
        "walk": walk_weight,
    }


# ============================================================================
# INDIVIDUAL MODE CHOICE
# ============================================================================

@dataclass
class ModeChoiceParameters:
    apm_headway_min: float = 5.0
    bus_headway_min: float = 30.0
    apm_fare: float = 2.0
    bus_fare: float = 2.0
    car_parking_cost: float = CAR_PARKING_COST
    walk_max_miles: float = 1.0
    bike_max_miles: float = 5.0
    income_floor: float = 20000.0
    purpose_asc_adjustments: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        if self.purpose_asc_adjustments is None:
            self.purpose_asc_adjustments = {
                "work": {"car": 0.2, "bus": 0.1, "apm": 0.3, "walk": -0.2},
                "school": {"car": -0.1, "bus": 0.2, "apm": 0.3, "walk": 0.1},
                "shopping": {"car": 0.1, "bus": 0.1, "apm": 0.2, "walk": 0.05},
                "social": {"car": 0.0, "bus": 0.05, "apm": 0.1, "walk": 0.05},
                "personal": {"car": 0.1, "bus": 0.0, "apm": 0.1, "walk": 0.05},
                "meal": {"car": 0.05, "bus": 0.0, "apm": 0.05, "walk": 0.05},
                "other": {"car": 0.0, "bus": 0.0, "apm": 0.0, "walk": 0.0},
            }


def _income_scaler(income: np.ndarray, floor: float) -> np.ndarray:
    """Scale cost sensitivity by income."""
    scaled = np.maximum(income, floor)
    return scaled / floor


def _purpose_adjustments(purpose: np.ndarray, adjustments: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
    """Return additive ASC adjustments per purpose for each mode."""
    modes = ["car", "bus", "apm", "walk"]
    adj_arrays = {m: np.zeros_like(purpose, dtype=float) for m in modes}
    for name, delta in adjustments.items():
        mask = purpose == name
        if mask.any():
            for m in modes:
                adj_arrays[m][mask] = delta.get(m, 0.0)
    return adj_arrays


def _utilities_for_individual_trips(
    distances_miles: np.ndarray,
    params: ModeChoiceParameters,
    availability: Dict[str, np.ndarray],
    income: np.ndarray,
    purpose: np.ndarray,
    mnl_coefficients: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute utilities per mode with availability and purpose adjustments."""
    b_ivt, b_wait, b_access, b_cost = (
        mnl_coefficients if mnl_coefficients is not None
        else (BETA_IN_VEHICLE_TIME, BETA_WAIT_TIME, BETA_ACCESS_TIME, BETA_COST)
    )

    wait_apm = params.apm_headway_min / 2.0
    wait_bus = effective_wait_time(params.bus_headway_min)
    access_transit = 2 * (AVG_ACCESS_DISTANCE_TRANSIT / WALK_SPEED_MPH) * 60
    access_car = 2.0

    ivt_apm = (distances_miles / SPEED_APM) * 60
    ivt_bus = (distances_miles / SPEED_BUS) * 60
    car_speeds = np.where(distances_miles > HIGHWAY_THRESHOLD_MILES, SPEED_CAR_HIGHWAY, SPEED_CAR_URBAN)
    ivt_car = (distances_miles / car_speeds) * 60
    ivt_walk = (distances_miles / SPEED_WALK) * 60

    income_scale = _income_scaler(income, params.income_floor)
    cost_apm = np.full_like(distances_miles, params.apm_fare)
    cost_bus = np.full_like(distances_miles, params.bus_fare)
    cost_car = distances_miles * CAR_COST_PER_MILE + params.car_parking_cost

    purpose_adj = _purpose_adjustments(purpose, params.purpose_asc_adjustments)

    u_apm = (b_ivt * ivt_apm + b_wait * wait_apm +
             b_access * access_transit + b_cost * (cost_apm / income_scale) +
             ASC_APM + purpose_adj["apm"])
    u_bus = (b_ivt * ivt_bus + b_wait * wait_bus +
             b_access * access_transit + b_cost * (cost_bus / income_scale) +
             ASC_BUS + purpose_adj["bus"])
    u_car = (b_ivt * ivt_car + b_access * access_car +
             b_cost * (cost_car / income_scale) + ASC_CAR + purpose_adj["car"])
    u_walk = (b_ivt * ivt_walk + ASC_WALK + purpose_adj["walk"])

    # Bike-to-APM: cycle to station, ride APM
    bike_access_ivt = (AVG_ACCESS_DISTANCE_TRANSIT / SPEED_BIKE) * 60  # minutes
    u_bike = (b_ivt * (ivt_apm + bike_access_ivt) +
              b_wait * wait_apm +
              b_access * BIKE_ACCESS_PENALTY_MIN +
              b_cost * ((cost_apm + BIKE_COST_PER_TRIP) / income_scale) +
              ASC_BIKE_APM + purpose_adj["apm"])

    neg_inf = -1e9
    u_apm = np.where(availability["apm"], u_apm, neg_inf)
    u_bus = np.where(availability["bus"], u_bus, neg_inf)
    u_car = np.where(availability["car"], u_car, neg_inf)
    u_walk = np.where(availability["walk"], u_walk, neg_inf)
    # Suppress bike-to-APM for short trips and require APM + bike availability
    bike_avail = availability["apm"] & availability["bike"] & (distances_miles >= BIKE_MIN_TRIP_MILES)
    u_bike = np.where(bike_avail, u_bike, neg_inf)

    return {"apm": u_apm, "bus": u_bus, "car": u_car, "walk": u_walk, "bike_apm": u_bike}


def _utilities_with_distance_decay(
    distances_miles: np.ndarray,
    params: ModeChoiceParameters,
    access_weights: Dict[str, np.ndarray],
    access_distances_m: Dict[str, np.ndarray],
    income: np.ndarray,
    purpose: np.ndarray,
    unavailable_penalty: float = -10.0,
    mnl_coefficients: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute utilities using continuous distance decay instead of binary availability.

    This replaces the -infinity penalty for unavailable modes with a soft penalty
    based on actual walk distance to stops.

    Parameters:
    - access_weights: Dict from availability_weights() with continuous weights [0, 1]
    - access_distances_m: Dict with actual walk distances in meters for each mode
                          {'apm': origin_dist_m, 'bus': origin_dist_m}
    - unavailable_penalty: Utility penalty for modes with zero access weight
    - mnl_coefficients: Optional array [IVT, WAIT, ACCESS, COST] overriding module defaults

    The access time component is computed from actual walk distance rather than
    using a fixed average distance.
    """
    b_ivt, b_wait, b_access, b_cost = (
        mnl_coefficients if mnl_coefficients is not None
        else (BETA_IN_VEHICLE_TIME, BETA_WAIT_TIME, BETA_ACCESS_TIME, BETA_COST)
    )

    wait_apm = params.apm_headway_min / 2.0
    wait_bus = effective_wait_time(params.bus_headway_min)
    access_car = 2.0  # minutes

    # In-vehicle travel times
    ivt_apm = (distances_miles / SPEED_APM) * 60
    ivt_bus = (distances_miles / SPEED_BUS) * 60
    car_speeds = np.where(distances_miles > HIGHWAY_THRESHOLD_MILES, SPEED_CAR_HIGHWAY, SPEED_CAR_URBAN)
    ivt_car = (distances_miles / car_speeds) * 60
    ivt_walk = (distances_miles / SPEED_WALK) * 60

    # Compute actual access times from distances (both ends, so 2x)
    # Convert meters to miles, then to minutes at walk speed
    m_to_miles = 0.000621371
    apm_dist_m = access_distances_m.get('apm', np.full_like(distances_miles, 500))
    bus_dist_m = access_distances_m.get('bus', np.full_like(distances_miles, 500))

    access_apm = 2 * (apm_dist_m * m_to_miles / SPEED_WALK) * 60  # minutes
    access_bus = 2 * (bus_dist_m * m_to_miles / SPEED_WALK) * 60  # minutes

    # Cap access time at reasonable maximum (30 min walk = ~2.4 km)
    access_apm = np.clip(access_apm, 0, 30)
    access_bus = np.clip(access_bus, 0, 30)

    income_scale = _income_scaler(income, params.income_floor)
    cost_apm = np.full_like(distances_miles, params.apm_fare)
    cost_bus = np.full_like(distances_miles, params.bus_fare)
    cost_car = distances_miles * CAR_COST_PER_MILE + params.car_parking_cost

    purpose_adj = _purpose_adjustments(purpose, params.purpose_asc_adjustments)

    # Base utilities with actual access times
    u_apm = (b_ivt * ivt_apm + b_wait * wait_apm +
             b_access * access_apm + b_cost * (cost_apm / income_scale) +
             ASC_APM + purpose_adj["apm"])
    u_bus = (b_ivt * ivt_bus + b_wait * wait_bus +
             b_access * access_bus + b_cost * (cost_bus / income_scale) +
             ASC_BUS + purpose_adj["bus"])
    u_car = (b_ivt * ivt_car + b_access * access_car +
             b_cost * (cost_car / income_scale) + ASC_CAR + purpose_adj["car"])
    u_walk = (b_ivt * ivt_walk + ASC_WALK + purpose_adj["walk"])

    # Apply soft penalty based on access weight (instead of -infinity)
    # Penalty increases as weight decreases: penalty = unavailable_penalty * (1 - weight)
    apm_weight = access_weights.get("apm", np.ones_like(distances_miles))
    bus_weight = access_weights.get("bus", np.ones_like(distances_miles))
    walk_weight = access_weights.get("walk", np.ones_like(distances_miles))
    car_weight = access_weights.get("car", np.ones_like(distances_miles))

    u_apm = u_apm + unavailable_penalty * (1 - apm_weight)
    u_bus = u_bus + unavailable_penalty * (1 - bus_weight)
    u_walk = u_walk + unavailable_penalty * (1 - walk_weight)
    u_car = u_car + unavailable_penalty * (1 - car_weight)

    # Bike-to-APM: cycle to station, ride APM
    bike_access_ivt = (AVG_ACCESS_DISTANCE_TRANSIT / SPEED_BIKE) * 60
    u_bike = (b_ivt * (ivt_apm + bike_access_ivt) +
              b_wait * wait_apm +
              b_access * BIKE_ACCESS_PENALTY_MIN +
              b_cost * ((cost_apm + BIKE_COST_PER_TRIP) / income_scale) +
              ASC_BIKE_APM + purpose_adj["apm"])
    u_bike = u_bike + unavailable_penalty * (1 - apm_weight)

    # Still apply hard cutoff for truly unavailable (weight = 0)
    neg_inf = -1e9
    u_apm = np.where(apm_weight > 0.01, u_apm, neg_inf)
    u_bus = np.where(bus_weight > 0.01, u_bus, neg_inf)
    u_car = np.where(car_weight > 0.01, u_car, neg_inf)
    u_walk = np.where(walk_weight > 0.01, u_walk, neg_inf)
    # Suppress bike-to-APM for short trips
    u_bike = np.where((apm_weight > 0.01) & (distances_miles >= BIKE_MIN_TRIP_MILES), u_bike, neg_inf)

    return {"apm": u_apm, "bus": u_bus, "car": u_car, "walk": u_walk, "bike_apm": u_bike}


def _logit_probs(utils: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert utilities to probabilities with logit.

    Bike-to-APM probability is combined with walk-to-APM into a single
    ``apm`` share, matching the student model convention.
    """
    exp_apm, exp_bus = np.exp(utils["apm"]), np.exp(utils["bus"])
    exp_car, exp_walk = np.exp(utils["car"]), np.exp(utils["walk"])
    exp_bike = np.exp(utils.get("bike_apm", np.full_like(exp_apm, -1e9)))
    total = exp_apm + exp_bike + exp_bus + exp_car + exp_walk
    total = np.where(total == 0, 1.0, total)
    return {
        "apm": (exp_apm + exp_bike) / total,
        "bus": exp_bus / total,
        "car": exp_car / total,
        "walk": exp_walk / total,
    }


def run_individual_mode_choice(
    trips_df: pd.DataFrame,
    persons_df: pd.DataFrame,
    parcels_gdf: gpd.GeoDataFrame,
    bus_stops_gdf: gpd.GeoDataFrame,
    apm_stops_gdf: Optional[gpd.GeoDataFrame] = None,
    params: Optional[ModeChoiceParameters] = None,
) -> pd.DataFrame:
    """Compute per-trip mode probabilities with person attributes and availability."""
    if params is None:
        params = ModeChoiceParameters()

    trips = trips_df.merge(persons_df[["person_id", "income", "car_access"]], on="person_id", how="left")
    trips["income"] = trips["income"].fillna(params.income_floor)
    trips["car_access"] = trips["car_access"].fillna(False)

    edges_path = Path("data/processed/network_edges.geojson")
    circuity = estimate_circuity_from_network(edges_path) if edges_path.exists() else 1.0

    distances = distance_miles_for_trips(parcels_gdf, trips, circuity_factor=circuity)

    bus_access = access_mask_for_trips(parcels_gdf, trips, bus_stops_gdf)
    apm_access = access_mask_for_trips(parcels_gdf, trips, apm_stops_gdf) if apm_stops_gdf is not None else np.zeros(len(trips), dtype=bool)

    availability = availability_flags(
        distances_miles=distances,
        car_access=trips["car_access"].values.astype(bool),
        bus_access=bus_access,
        apm_access=apm_access,
        walk_max_miles=params.walk_max_miles,
        bike_max_miles=params.bike_max_miles,
    )

    utils = _utilities_for_individual_trips(
        distances_miles=distances,
        params=params,
        availability=availability,
        income=trips["income"].values.astype(float),
        purpose=trips.get("purpose", pd.Series(["other"] * len(trips))).astype(str).str.lower().values,
    )

    probs = _logit_probs(utils)

    trips_out = trips_df.copy()
    trips_out["apm_prob"] = probs["apm"]
    trips_out["bus_prob"] = probs["bus"]
    trips_out["car_prob"] = probs["car"]
    trips_out["walk_prob"] = probs["walk"]
    trips_out["transit_prob"] = trips_out["apm_prob"] + trips_out["bus_prob"]

    return trips_out


def summarize_mode_shares(result_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate mean probabilities to system-level shares."""
    shares = {
        "apm": float(result_df["apm_prob"].mean()),
        "bus": float(result_df["bus_prob"].mean()),
        "car": float(result_df["car_prob"].mean()),
        "walk": float(result_df["walk_prob"].mean()),
    }
    shares["transit"] = shares["apm"] + shares["bus"]
    return shares


# ============================================================================
# OD-BASED MODE SPLIT
# ============================================================================

def _build_od_access_mask(
    parcels_gdf: gpd.GeoDataFrame,
    od_df: pd.DataFrame,
    stops_gdf: Optional[gpd.GeoDataFrame],
    access_m: float = 800.0,
    origin_col: str = 'origin_parcel',
    dest_col: str = 'dest_parcel',
) -> np.ndarray:
    """Build boolean mask for OD pairs with stops accessible on both ends."""
    if stops_gdf is None or len(stops_gdf) == 0:
        return np.zeros(len(od_df), dtype=bool)

    parcels = parcels_gdf[['PARCEL_ID', 'geometry']].to_crs(PROJECT_CRS).copy()
    stops = stops_gdf.to_crs(PROJECT_CRS).copy()

    cent = parcels.copy()
    # EPSG:2965 coordinates are in US survey feet — convert to meters
    cent['x'] = cent.geometry.centroid.x * US_SURVEY_FT_TO_M
    cent['y'] = cent.geometry.centroid.y * US_SURVEY_FT_TO_M
    cent_lookup = cent.set_index('PARCEL_ID')[['x', 'y']].to_dict('index')

    stop_xy = np.vstack([stops.geometry.x * US_SURVEY_FT_TO_M,
                         stops.geometry.y * US_SURVEY_FT_TO_M]).T
    tree = cKDTree(stop_xy)

    origin_xy = np.array([
        (cent_lookup.get(o, {'x': np.nan, 'y': np.nan})['x'], cent_lookup.get(o, {'x': np.nan, 'y': np.nan})['y'])
        for o in od_df[origin_col].values
    ])
    dest_xy = np.array([
        (cent_lookup.get(d, {'x': np.nan, 'y': np.nan})['x'], cent_lookup.get(d, {'x': np.nan, 'y': np.nan})['y'])
        for d in od_df[dest_col].values
    ])

    origin_counts = np.array([len(tree.query_ball_point(pt, r=access_m)) if not np.isnan(pt[0]) else 0 for pt in origin_xy])
    dest_counts = np.array([len(tree.query_ball_point(pt, r=access_m)) if not np.isnan(pt[0]) else 0 for pt in dest_xy])

    return (origin_counts > 0) & (dest_counts > 0)


def _build_od_access_weights(
    parcels_gdf: gpd.GeoDataFrame,
    od_df: pd.DataFrame,
    stops_gdf: Optional[gpd.GeoDataFrame],
    access_m: float = 800.0,
    decay_beta: float = 0.00173,
    origin_col: str = 'origin_parcel',
    dest_col: str = 'dest_parcel',
) -> np.ndarray:
    """Build continuous accessibility weights for OD pairs using distance decay.

    Returns array of weights [0, 1] for each OD pair.
    Weight = sqrt(origin_weight * dest_weight) where each weight = exp(-beta * dist)

    Default beta=0.00173 calibrated to TCRP Report 165 (50% weight at 400m walk).
    """
    if stops_gdf is None or len(stops_gdf) == 0:
        return np.zeros(len(od_df), dtype=float)

    parcels = parcels_gdf[['PARCEL_ID', 'geometry']].to_crs(PROJECT_CRS).copy()
    stops = stops_gdf.to_crs(PROJECT_CRS).copy()

    cent = parcels.copy()
    # EPSG:2965 coordinates are in US survey feet — convert to meters
    cent['x'] = cent.geometry.centroid.x * US_SURVEY_FT_TO_M
    cent['y'] = cent.geometry.centroid.y * US_SURVEY_FT_TO_M
    cent_lookup = cent.set_index('PARCEL_ID')[['x', 'y']].to_dict('index')

    stop_xy = np.vstack([stops.geometry.x * US_SURVEY_FT_TO_M,
                         stops.geometry.y * US_SURVEY_FT_TO_M]).T
    tree = cKDTree(stop_xy)

    def get_weight(parcel_ids):
        weights = np.zeros(len(parcel_ids))
        for i, pid in enumerate(parcel_ids):
            pt_data = cent_lookup.get(pid, {'x': np.nan, 'y': np.nan})
            pt = np.array([pt_data['x'], pt_data['y']])
            if np.isnan(pt[0]):
                continue
            dist, _ = tree.query(pt, k=1)
            if dist <= access_m:
                weights[i] = np.exp(-decay_beta * dist)
        return weights

    origin_weights = get_weight(od_df[origin_col].values)
    dest_weights = get_weight(od_df[dest_col].values)

    # Combined weight: geometric mean ensures both ends need access
    return np.sqrt(origin_weights * dest_weights)


def _compute_od_distances_miles(
    parcels_gdf: gpd.GeoDataFrame,
    od_df: pd.DataFrame,
    origin_col: str = 'origin_parcel',
    dest_col: str = 'dest_parcel',
) -> np.ndarray:
    """Compute straight-line distance (miles) between parcel centroids for each OD pair."""
    parcels = parcels_gdf[['PARCEL_ID', 'geometry']].to_crs(PROJECT_CRS).copy()
    cent = parcels.copy()
    # EPSG:2965 coordinates are in US survey feet — convert to meters
    cent['x'] = cent.geometry.centroid.x * US_SURVEY_FT_TO_M
    cent['y'] = cent.geometry.centroid.y * US_SURVEY_FT_TO_M
    lookup = cent.set_index('PARCEL_ID')[['x', 'y']].to_dict('index')

    o_xy = np.array([
        (lookup.get(o, {'x': np.nan, 'y': np.nan})['x'], lookup.get(o, {'x': np.nan, 'y': np.nan})['y'])
        for o in od_df[origin_col].values
    ])
    d_xy = np.array([
        (lookup.get(d, {'x': np.nan, 'y': np.nan})['x'], lookup.get(d, {'x': np.nan, 'y': np.nan})['y'])
        for d in od_df[dest_col].values
    ])

    diff = o_xy - d_xy
    dist_m = np.sqrt(np.sum(diff**2, axis=1))
    dist_miles = dist_m * 0.000621371
    return np.where(np.isfinite(dist_miles), dist_miles, 9999.0)


def mode_split_for_od(
    od_df: pd.DataFrame,
    parcels_gdf: gpd.GeoDataFrame,
    bus_stops_gdf: Optional[gpd.GeoDataFrame] = None,
    apm_stops_gdf: Optional[gpd.GeoDataFrame] = None,
    apm_headway_min: float = 5.0,
    bus_headway_min: float = 30.0,
    apm_fare: float = 2.0,
    bus_fare: float = 2.0,
    car_parking_cost: float = CAR_PARKING_COST,
    access_m: float = 800.0,
    use_income_segments: bool = True,
    income_segment_cols: Tuple[str, str, str] = INCOME_SEGMENT_ORDER,
    income_segment_coeffs: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Compute per-OD mode shares and trips across walk, car, bus, APM."""
    dist_miles = _compute_od_distances_miles(parcels_gdf, od_df)

    bus_ok = _build_od_access_mask(parcels_gdf, od_df, bus_stops_gdf, access_m) if bus_stops_gdf is not None else np.zeros(len(od_df), dtype=bool)
    apm_ok = _build_od_access_mask(parcels_gdf, od_df, apm_stops_gdf, access_m) if apm_stops_gdf is not None else np.zeros(len(od_df), dtype=bool)

    apm_headways = np.where(apm_ok, apm_headway_min, 1e9)
    bus_headways = np.where(bus_ok, bus_headway_min, 1e9)

    combos = pd.DataFrame({'apm_h': apm_headways, 'bus_h': bus_headways})
    groups = combos.groupby(['apm_h', 'bus_h']).groups

    shares = np.zeros((len(od_df), 5))
    segment_shares = None

    if use_income_segments and all(col in od_df.columns for col in income_segment_cols):
        seg_arrays = {
            seg: od_df[col].fillna(0).astype(float).values
            for seg, col in zip(INCOME_SEGMENT_ORDER, income_segment_cols)
        }
        segment_shares = _normalize_income_segment_weights(seg_arrays, len(od_df))

    for (ah, bh), idx in groups.items():
        idx_arr = np.fromiter(idx, dtype=int)
        dist_slice = dist_miles[idx_arr]
        seg_slice = None
        if segment_shares is not None:
            seg_slice = {
                seg: vals[idx_arr]
                for seg, vals in segment_shares.items()
            }
        df_sh = calculate_mode_shares_vectorized(
            distances=dist_slice,
            apm_headway=float(ah),
            bus_headway=float(bh),
            apm_fare=apm_fare,
            bus_fare=bus_fare,
            car_parking_cost=car_parking_cost,
            income_segment_shares=seg_slice,
            income_segment_coeffs=income_segment_coeffs,
        )
        shares[idx_arr, 0] = df_sh['apm_share'].values
        shares[idx_arr, 1] = df_sh['bus_share'].values
        shares[idx_arr, 2] = df_sh['car_share'].values
        shares[idx_arr, 3] = df_sh['walk_share'].values
        shares[idx_arr, 4] = df_sh['transit_share'].values

    result = od_df.copy()
    result['apm_share'] = shares[:, 0]
    result['bus_share'] = shares[:, 1]
    result['car_share'] = shares[:, 2]
    result['walk_share'] = shares[:, 3]
    result['transit_share'] = shares[:, 4]

    trips = result['trips'].values
    result['apm_trips'] = trips * result['apm_share']
    result['bus_trips'] = trips * result['bus_share']
    result['car_trips'] = trips * result['car_share']
    result['walk_trips'] = trips * result['walk_share']

    return result


# ============================================================================
# DIAGNOSTIC UTILITIES
# ============================================================================

def print_mode_choice_breakdown(
    distance_miles: float,
    apm_headway: float = 5.0,
    bus_headway: float = 30.0,
    apm_fare: float = 2.0,
    bus_fare: float = 2.0,
    car_parking_cost: float = CAR_PARKING_COST,
):
    """Print detailed breakdown of mode choice calculation."""
    logger.info(f"\n{'='*70}")
    logger.info(f"MODE CHOICE BREAKDOWN - {distance_miles:.1f} mile trip")
    logger.info(f"{'='*70}")

    for mode in ['apm', 'bus', 'car', 'walk']:
        logger.info(f"\n{mode.upper()}:")
        headway = apm_headway if mode == 'apm' else (bus_headway if mode == 'bus' else 0)
        fare = apm_fare if mode == 'apm' else (bus_fare if mode == 'bus' else 0)

        ivt = calculate_travel_time(distance_miles, mode)
        wait = calculate_wait_time(headway, mode)
        access = calculate_access_time(mode)
        cost = calculate_cost(distance_miles, fare, mode, car_parking_cost=car_parking_cost)
        utility = calculate_utility(distance_miles, headway, fare, mode, car_parking_cost=car_parking_cost)

        logger.info(f"  In-vehicle time: {ivt:.1f} min")
        logger.info(f"  Wait time: {wait:.1f} min")
        logger.info(f"  Access time: {access:.1f} min")
        logger.info(f"  Cost: ${cost:.2f}")
        logger.info(f"  Utility: {utility:.3f}")

    shares = calculate_mode_shares(
        distance_miles,
        apm_headway,
        bus_headway,
        apm_fare,
        bus_fare,
        car_parking_cost=car_parking_cost,
    )

    logger.info(f"\n{'='*70}")
    logger.info("MODE SHARES:")
    for mode in ['apm', 'bus', 'car', 'walk']:
        logger.info(f"  {mode.upper()}: {shares[mode]*100:.1f}%")
    logger.info(f"  Transit total: {shares['transit_total']*100:.1f}%")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    print("Testing consolidated mode choice model...")
    print_mode_choice_breakdown(distance_miles=3.0)

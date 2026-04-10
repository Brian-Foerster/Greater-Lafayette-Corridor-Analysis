"""
Dynamic Bus Network Module
===========================

Per-route service planning, operating costs, fleet sizing, and headway
optimization for CityBus integration with APM corridors.

Architecture:
- BusRoute: per-route state (geometry, headway, cycle time, ridership)
- BusOperatingCostModel: $/vehicle-hour costs, fleet sizing, annual budget
- RouteAPMClassifier: classifies routes as parallel/feeder/independent per corridor
- RouteServicePlan: budget-constrained per-route headway optimization
- APMFrequencyResponse: demand-responsive APM headway lookup

Data sources:
- GTFS (routes, shapes, stop_times, trips) for baseline network
- CityBus ridership CSVs for observed productivity
- Corridor geometry for spatial overlap classification
"""
from __future__ import annotations

import copy
import enum
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Operating cost (NTD 2023 small-urban fixed-route bus)
DEFAULT_COST_PER_VEH_HOUR = 135.0  # $/veh-hr (CityBus-scale: $120-160 range)
SPARE_RATIO = 0.20  # 20% spare vehicles on top of peak pull-out

# Import service span and APM cost from canonical source (financial_params)
from src.financial_params import (
    SERVICE_SPAN_HOURS as _FP_SERVICE_SPAN,
    SERVICE_DAYS_PER_YEAR as _FP_SERVICE_DAYS,
    O_AND_M_VEH_HOUR_USD as _FP_APM_VEH_HR,
    APM_AVG_SPEED_KPH as _FP_APM_SPEED,
    APM_DWELL_TIME_S as _FP_APM_DWELL,
    BRT_AVG_SPEED_KPH as _FP_BRT_SPEED,
    BRT_DWELL_TIME_S as _FP_BRT_DWELL,
    CARS_PER_TRAIN as _FP_CARS_PER_TRAIN,
)
DEFAULT_SERVICE_SPAN_HOURS = _FP_SERVICE_SPAN
APM_COST_PER_VEH_HOUR = _FP_APM_VEH_HR
APM_VEHICLES_PER_TRAIN = _FP_CARS_PER_TRAIN

# APM vehicle/train parameters
APM_PASSENGERS_PER_VEHICLE = 50   # seated 12 + standing 38 (Innovia APM 100)
APM_TRAIN_CAPACITY = APM_VEHICLES_PER_TRAIN * APM_PASSENGERS_PER_VEHICLE  # 100 pax
APM_PEAK_HOUR_FACTOR = 0.14       # ~14% of daily in peak hour (university peaking)
APM_PEAK_DIR_SPLIT = 0.60         # 60/40 directional split in peak (campus commute)
APM_TARGET_LOAD_FACTOR = 0.72     # ASCE 21.2: 70-75% of crush for standing-pax APM
APM_MIN_HEADWAY_MIN = 1.5         # 90s practical minimum (AGT signal/dwell limit)
APM_MAX_HEADWAY_MIN = 10.0        # policy max: maintain service attractiveness
APM_MAX_FLEET_SIZE = 20           # capital constraint: max trains purchasable
# Service-quality regime: logarithmic headway target based on demand
# At 0 riders: MAX_HEADWAY. Headway decreases as log(1+riders/REF) grows.
# Calibrated: ~1K → 8 min, ~3K → 6 min, ~6K → 3.8 min, ~8K → 2.8 min
APM_SQ_HEADWAY_SCALE = 4.5       # controls steepness of frequency response
APM_SQ_DEMAND_REF = 2000.0       # demand normalization reference

# Mode-choice default headways (used when dynamic bus network is unavailable)
DEFAULT_BUS_HEADWAY_MIN = 30       # CityBus average headway (minutes)
DEFAULT_APM_HEADWAY_MIN = 5        # APM initial headway assumption (minutes)
DEFAULT_BUS_SPEED_KPH = 20         # CityBus system average speed (km/h)
FEEDER_BUS_SPEED_KPH = 18          # feeder buses in lower-density residential areas

# Route-APM overlap thresholds
PARALLEL_OVERLAP_THRESHOLD = 0.40  # ≥40% stops within 400m of corridor → parallel
FEEDER_OVERLAP_THRESHOLD = 0.15   # 15-40% overlap → feeder candidate
CORRIDOR_BUFFER_M = 400.0         # buffer around corridor for overlap test

# Budget
CITYBUS_ANNUAL_BUDGET_USD = 13_500_000.0  # $13.5M (CityBus FY2024 operating budget)

# --- Operating budget scenarios ---
# Three institutional arrangements for who pays APM operating costs.
# The choice dramatically affects how much bus service survives.
BUDGET_MODE_SEPARATE = "separate"      # Status quo: APM funded independently (TIF + fares)
BUDGET_MODE_COMBINED = "combined"      # Single transit authority pays both
BUDGET_MODE_EXPANDED = "expanded"      # Combined, but with new federal/state revenue

# New revenue sources triggered by APM (expanded mode only)
# Lafayette UZA (~120K pop) currently receives ~$3.5M in 5307;
# fixed-guideway bonus + expanded service-area formula → ~$1.5M increment.
# FTA 5307 for operating: 50% federal / 50% local match.
# The $1.5M is the federal share; local match assumed covered by TIF/fares.
FTA_5307_OPERATING_INCREMENT_USD = 1_500_000

# Indiana Code 36-9-4: state operating assistance = 15% of eligible ops.
# Applied to both bus and APM operating costs in expanded mode.
INDIANA_PMTF_RATE = 0.15

# Fare inflation: fares escalate at general CPI (~2.5%/yr)
FARE_ESCALATION_RATE = 0.025

DEFAULT_BUDGET_MODE = BUDGET_MODE_COMBINED  # 100% local funding = single fiscal entity

# In combined/expanded modes, bus savings from parallel route degradation
# are split between feeder reinvestment and APM O&M offset.  In separate
# mode, all savings go to feeders (APM is a different fiscal entity).
# 30% APM offset is conservative: operational overlap (parallel routes
# duplicating APM) represents ~30-50% of savings value, but transit
# authorities typically favor maintaining bus service quality.
BUS_SAVINGS_APM_OFFSET_FRACTION = 0.30

# ---------------------------------------------------------------------------
# Proactive bus network redesign (Component 2)
# ---------------------------------------------------------------------------

# Route productivity threshold — passengers per revenue-hour.
# NTD 2023 small-urban (50K-200K UZA) median is ~10; CityBus routes range
# 5-25.  Routes below this threshold are candidates for restructuring.
MIN_PRODUCTIVITY = 8.0  # passengers / revenue-hour

# Phased restructuring timeline: APM year → set of allowed actions.
# No changes during construction/ramp-up (years 0-2).
# Parallel elimination at year 3, rerouting at year 5, feeder enhancement
# at year 7.  Each year threshold enables all actions at or below that level.
RESTRUCTURING_PHASES = {
    3: {"eliminate"},
    5: {"eliminate", "reroute"},
    7: {"eliminate", "reroute", "enhance"},
}

# Title VI equity guard: if a route's catchment SE01 share exceeds the
# metro average by this multiplier, downgrade any elimination/reduction
# to protect low-income transit-dependent riders.
EQUITY_DISPARITY_THRESHOLD = 1.5  # 50% above metro average

# Equity uplift for feeder coverage scoring: sectors with higher SE01
# share get up-weighted in coverage calculation.
EQUITY_COVERAGE_UPLIFT = 0.5  # 100% SE01 → 1.5× weight

# Minimum APM ridership to trigger bus network restructuring.
# Below this threshold, bus network operates as if no APM existed.
MIN_APM_RIDERSHIP_FOR_RESTRUCTURE = 1000  # daily riders

# Year after opening when post-opening adjustments begin (legacy constant).
POST_OPENING_ADJUSTMENT_YEAR = 3


# ---------------------------------------------------------------------------
# Induced bus ridership
# ---------------------------------------------------------------------------

BUS_INDUCED_ELASTICITY = 0.05     # half of APM elasticity (weaker indirect effect)
BUS_INDUCED_THRESHOLD_YEAR = 5    # no boost before year 5


def compute_bus_induced_factor(
    year: int,
    apm_daily_riders: float,
    mature_ridership_target: float = 2500.0,
) -> float:
    """Compute the multiplicative induced-ridership boost for bus routes.

    Better feeder service + APM-driven density → new local bus trips.
    Returns 1.0 (no boost) before threshold year or with zero APM riders.
    """
    if year < BUS_INDUCED_THRESHOLD_YEAR or apm_daily_riders <= 0:
        return 1.0
    apm_quality = min(apm_daily_riders / mature_ridership_target, 1.0)
    return 1.0 + BUS_INDUCED_ELASTICITY * np.log1p(apm_quality)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ServiceProfile:
    """Time-of-day headway profile for a bus route (TCRP 165 small-urban)."""
    am_peak_headway_min: float    # 7-10am
    midday_headway_min: float     # 10am-4pm
    pm_peak_headway_min: float    # 4-7pm
    evening_headway_min: float    # 7-10pm
    saturday_headway_min: float
    sunday_headway_min: float

    @classmethod
    def from_single_headway(cls, headway: float) -> "ServiceProfile":
        """Generate a typical profile from a single daily-average headway.

        Ratios from TCRP Report 165 (small-urban defaults):
        AM peak = 0.67x, midday = 1.0x, PM peak = 0.75x,
        evening = 1.5x, saturday = 1.5x, sunday = 2.0x.
        """
        return cls(
            am_peak_headway_min=round(headway * 0.67, 1),
            midday_headway_min=round(headway, 1),
            pm_peak_headway_min=round(headway * 0.75, 1),
            evening_headway_min=round(headway * 1.5, 1),
            saturday_headway_min=round(headway * 1.5, 1),
            sunday_headway_min=round(headway * 2.0, 1),
        )

    @property
    def weekday_average(self) -> float:
        """Weighted weekday average headway (using mode_choice period distribution)."""
        # am_peak=0.30, pm_peak=0.25, off_peak=0.45 (midday+evening blend)
        off_peak_hw = 0.7 * self.midday_headway_min + 0.3 * self.evening_headway_min
        return (0.30 * self.am_peak_headway_min
                + 0.25 * self.pm_peak_headway_min
                + 0.45 * off_peak_hw)

    def headway_for_period(self, period: str) -> float:
        """Return headway for a named period (am_peak, pm_peak, off_peak)."""
        if period == "am_peak":
            return self.am_peak_headway_min
        elif period == "pm_peak":
            return self.pm_peak_headway_min
        else:  # off_peak: blend midday + evening
            return 0.7 * self.midday_headway_min + 0.3 * self.evening_headway_min


# ---------------------------------------------------------------------------
# Sector-based feeder coverage
# ---------------------------------------------------------------------------

N_SECTORS = 8
SECTOR_WIDTH = 2 * np.pi / N_SECTORS
FEEDER_WALK_M = 400        # max walk from bus stop to parcel
FEEDER_INNER_M = 1200      # walk-zone boundary
from src.spatial_constants import FEEDER_CATCHMENT_M as _FEEDER_CATCHMENT_M
FEEDER_OUTER_M = _FEEDER_CATCHMENT_M  # feeder-zone outer boundary (from spatial_constants)
FREQUENCY_REF_MIN = 15.0   # 15-min headway = quality 1.0

# Wabash River barrier (only significant barrier in study area)
# X-coordinate of the Wabash River in meters (converted from EPSG:2965 feet).
# Original EPSG:2965 value: -14,603 ft × 0.3048006096 = -4,451 m
WABASH_APPROX_X = -14_603.0 * 0.3048006096012192  # ~-4451 m
WABASH_BRIDGE_PENALTY = 0.50


@dataclass
class SectorCoverage:
    """Per-sector feeder coverage for a corridor's stations.

    Sectors are 45-degree wedges: 0=N, 1=NE, 2=E, ..., 7=NW.
    Each sector's coverage value is frequency-weighted: a stop with
    15-min headway counts as 1.0, a stop with 60-min headway counts
    as 0.25.
    """
    coverage: np.ndarray   # shape (8,), values in [0, 1]
    pop_weight: np.ndarray  # shape (8,), population in each sector

    @property
    def effective_coverage(self) -> float:
        """Population-weighted average across sectors."""
        total = self.pop_weight.sum()
        if total == 0:
            return float(self.coverage.mean())
        return float((self.coverage * self.pop_weight).sum() / total)

    def for_bearing(self, bearing_rad: float) -> float:
        """Coverage fraction for a specific bearing from corridor centroid."""
        sector = int((bearing_rad / (2 * np.pi) * 8 + 0.5) % 8)
        return float(self.coverage[sector])

    @classmethod
    def uniform(cls, scalar: float) -> "SectorCoverage":
        """Create uniform coverage from a scalar (backward-compatible fallback)."""
        return cls(
            coverage=np.full(N_SECTORS, scalar),
            pop_weight=np.ones(N_SECTORS),
        )


def compute_sector_coverage(
    corridor_stops_xy: np.ndarray,
    feeder_routes: List["BusRoute"],
    bus_stops_by_route: Dict[str, np.ndarray],
    route_headways: Optional[Dict[str, float]] = None,
    # Pre-computed spatial cache data (preferred — avoids grid sampling
    # and redundant KDTree queries).
    feeder_parcel_sector: Optional[np.ndarray] = None,
    feeder_parcel_pop: Optional[np.ndarray] = None,
    feeder_parcel_xy: Optional[np.ndarray] = None,
    # Legacy parameters (kept for backward compatibility but ignored when
    # spatial cache data is provided).
    parcel_xy: Optional[np.ndarray] = None,
    parcel_pop: Optional[np.ndarray] = None,
    # Demand weighting: per-parcel OD demand relevance [0, 1].
    # When provided, multiplies population weight by the fraction of each
    # parcel's commute flows whose destinations are APM-served.  A feeder
    # serving parcels whose workers commute TO APM-accessible jobs is more
    # valuable than one serving parcels with irrelevant destinations.
    feeder_parcel_od_weight: Optional[np.ndarray] = None,
    # Equity weighting: per-parcel SE01 share for equity-weighted coverage.
    feeder_parcel_se01: Optional[np.ndarray] = None,
) -> SectorCoverage:
    """Compute per-sector feeder coverage from actual bus stop locations.

    Uses parcel-based computation when spatial cache data is provided
    (feeder_parcel_xy, feeder_parcel_sector, feeder_parcel_pop).  This
    is more accurate (population-weighted by construction) and cheaper
    (no grid generation, no redundant APM KDTree query) than the legacy
    grid-sampling fallback.

    When ``feeder_parcel_se01`` is provided, applies an equity uplift to
    coverage scoring: sectors with higher SE01 concentration get up-weighted
    by ``(1 + EQUITY_COVERAGE_UPLIFT * se01_share)``.  This nudges feeder
    optimization toward better serving low-income areas.

    For each 45-degree sector of the feeder ring:
    1. For each feeder-ring parcel within 400m of a feeder bus stop,
       assign a frequency-weighted quality score (15-min = 1.0, 60-min = 0.25)
    2. Sector coverage = pop-weighted mean quality of served parcels
    3. Sector population = sum of parcel pop in that sector
    """
    from scipy.spatial import cKDTree

    if corridor_stops_xy is None or len(corridor_stops_xy) < 2:
        return SectorCoverage(coverage=np.zeros(N_SECTORS),
                              pop_weight=np.zeros(N_SECTORS))

    # --- Collect feeder stops with frequency-weighted quality ---
    if route_headways is None:
        route_headways = {}
    feeder_stop_coords = []
    feeder_stop_quality = []
    for route in feeder_routes:
        rid = str(route.route_id)
        stops = bus_stops_by_route.get(rid)
        if stops is None or len(stops) == 0:
            continue
        hw = route_headways.get(rid, route.current_headway_min)
        quality = float(np.clip(FREQUENCY_REF_MIN / max(hw, 1.0), 0.1, 1.0))
        feeder_stop_coords.append(stops)
        feeder_stop_quality.extend([quality] * len(stops))

    # --- Determine query points and sectors ---
    # Prefer pre-computed spatial cache (parcel-based, population-weighted).
    use_parcel_path = (
        feeder_parcel_xy is not None
        and feeder_parcel_sector is not None
        and feeder_parcel_pop is not None
        and len(feeder_parcel_xy) > 0
    )

    if use_parcel_path:
        query_pts = feeder_parcel_xy
        sectors = feeder_parcel_sector.astype(int)
        pop = feeder_parcel_pop.astype(float)
        # Demand weighting: multiply population by OD relevance so sectors
        # whose residents commute to APM-served destinations get higher weight.
        if feeder_parcel_od_weight is not None and len(feeder_parcel_od_weight) == len(pop):
            _od_w = np.asarray(feeder_parcel_od_weight, dtype=float)
            # Floor at 0.2 so no sector is completely zeroed out — even
            # parcels with no APM-relevant destinations still benefit from
            # transit access for non-work trips.
            _od_w = np.clip(_od_w, 0.2, 1.0)
            pop = pop * _od_w
    else:
        # Legacy grid-sampling fallback
        centroid = corridor_stops_xy.mean(axis=0)
        apm_tree = cKDTree(corridor_stops_xy)

        extent = FEEDER_OUTER_M
        spacing = 200
        xs = np.arange(centroid[0] - extent, centroid[0] + extent, spacing)
        ys = np.arange(centroid[1] - extent, centroid[1] + extent, spacing)
        gx, gy = np.meshgrid(xs, ys)
        grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

        apm_dists, _ = apm_tree.query(grid_pts, k=1)
        in_ring = (apm_dists >= FEEDER_INNER_M) & (apm_dists <= FEEDER_OUTER_M)
        ring_pts = grid_pts[in_ring]

        if len(ring_pts) == 0:
            return SectorCoverage(coverage=np.zeros(N_SECTORS),
                                  pop_weight=np.zeros(N_SECTORS))

        dx = ring_pts[:, 0] - centroid[0]
        dy = ring_pts[:, 1] - centroid[1]
        bearings = np.arctan2(dx, dy) % (2 * np.pi)
        sectors = ((bearings / SECTOR_WIDTH + 0.5) % N_SECTORS).astype(int)
        query_pts = ring_pts
        pop = np.ones(len(ring_pts))  # uniform when no parcel data

    # --- Compute per-point quality scores ---
    coverage = np.zeros(N_SECTORS)
    pop_weight = np.zeros(N_SECTORS)
    if feeder_stop_coords:
        all_stops = np.vstack(feeder_stop_coords)
        stop_quality = np.array(feeder_stop_quality)
        feeder_tree = cKDTree(all_stops)

        bus_dists, bus_idx = feeder_tree.query(query_pts, k=1)
        served = bus_dists <= FEEDER_WALK_M
        served_quality = np.where(served, stop_quality[bus_idx], 0.0)

        for s in range(N_SECTORS):
            mask = sectors == s
            sector_pop = pop[mask]
            total_pop = sector_pop.sum()
            pop_weight[s] = total_pop
            if total_pop > 0:
                # Population-weighted mean quality
                coverage[s] = float((served_quality[mask] * sector_pop).sum() / total_pop)
            elif mask.sum() > 0:
                coverage[s] = float(served_quality[mask].mean())
    else:
        # No feeder stops: coverage is zero, but still compute pop weights
        for s in range(N_SECTORS):
            pop_weight[s] = pop[sectors == s].sum()

    # Ensure pop_weight is never all-zero (fallback to uniform)
    if pop_weight.sum() < 1e-9:
        pop_weight = np.ones(N_SECTORS)

    # --- Equity uplift: up-weight sectors with higher SE01 concentration ---
    if feeder_parcel_se01 is not None and use_parcel_path and len(feeder_parcel_se01) == len(pop):
        se01 = feeder_parcel_se01.astype(float)
        for s in range(N_SECTORS):
            mask = sectors == s
            sector_pop = pop[mask]
            total_pop = sector_pop.sum()
            if total_pop > 0:
                sector_se01 = float((se01[mask] * sector_pop).sum() / total_pop)
                coverage[s] *= (1.0 + EQUITY_COVERAGE_UPLIFT * sector_se01)
        # Clamp to [0, 1] — equity uplift can push coverage above 1.0
        coverage = np.clip(coverage, 0.0, 1.0)

    return SectorCoverage(coverage=coverage, pop_weight=pop_weight)


def apply_barrier_penalty(
    sector_coverage: SectorCoverage,
    corridor_centroid_x: float,
    river_x: Optional[float] = None,
) -> SectorCoverage:
    """Penalize sectors requiring a Wabash River crossing.

    If the corridor is west of the river (Purdue side), sectors pointing
    east across the river get a 0.5 multiplier. Vice versa for corridors
    east of the river (Lafayette side).

    Parameters
    ----------
    corridor_centroid_x : float
        X-coordinate of corridor centroid in projected CRS.
    river_x : float or None
        X-coordinate of river centerline. Falls back to module-level
        ``WABASH_APPROX_X`` when None.
    """
    if river_x is None:
        river_x = WABASH_APPROX_X
    if river_x is None:
        return sector_coverage

    adjusted = sector_coverage.coverage.copy()
    for s in range(N_SECTORS):
        # Sectors 1,2,3 (NE, E, SE) point east; 5,6,7 (SW, W, NW) point west
        points_east = s in (1, 2, 3)
        points_west = s in (5, 6, 7)

        if corridor_centroid_x < river_x and points_east:
            adjusted[s] *= WABASH_BRIDGE_PENALTY
        elif corridor_centroid_x > river_x and points_west:
            adjusted[s] *= WABASH_BRIDGE_PENALTY

    return SectorCoverage(coverage=adjusted, pop_weight=sector_coverage.pop_weight)


# ---------------------------------------------------------------------------
# Proactive Bus Network Redesign — decision engine
# ---------------------------------------------------------------------------

class RestructuringAction(enum.Enum):
    """Discrete restructuring actions for bus routes."""
    KEEP = "keep"
    ELIMINATE = "eliminate"
    REDUCE = "reduce"
    REROUTE = "reroute"
    ENHANCE = "enhance"


def route_productivity_score(route: "BusRoute") -> float:
    """Passengers per revenue-hour for a bus route.

    Standard transit productivity metric (NTD / APTA).  Uses observed
    daily ridership and computed daily vehicle-hours.
    """
    veh_hrs = route.daily_vehicle_hours
    if veh_hrs <= 0:
        return 0.0
    return route.observed_daily_riders / veh_hrs


def decide_route_restructuring(
    routes: List["BusRoute"],
    year: int = 0,
) -> Dict[str, RestructuringAction]:
    """Map each route to a restructuring action based on classification and
    productivity.

    Decision matrix:
    - Parallel / any productivity → ELIMINATE (savings → feeder pool)
    - Feeder / productive (≥MIN_PRODUCTIVITY) → ENHANCE (reduce headway)
    - Feeder / unproductive → REROUTE (same veh-hrs, better alignment)
    - Independent / productive → KEEP
    - Independent / unproductive → REDUCE (increase headway 1.5×)

    Actions are gated by RESTRUCTURING_PHASES — only actions unlocked by
    the current year are allowed.  Before year 3, all routes get KEEP.
    """
    # Determine which actions are allowed at this year
    allowed: set = set()
    for threshold_year, actions in sorted(RESTRUCTURING_PHASES.items()):
        if year >= threshold_year:
            allowed |= actions

    decisions: Dict[str, RestructuringAction] = {}
    for route in routes:
        rid = route.route_id
        prod = route_productivity_score(route)
        cls = route.classification

        if cls == "parallel":
            if "eliminate" in allowed:
                decisions[rid] = RestructuringAction.ELIMINATE
            else:
                decisions[rid] = RestructuringAction.KEEP
        elif cls == "feeder":
            if prod >= MIN_PRODUCTIVITY:
                if "enhance" in allowed:
                    decisions[rid] = RestructuringAction.ENHANCE
                else:
                    decisions[rid] = RestructuringAction.KEEP
            else:
                if "reroute" in allowed:
                    decisions[rid] = RestructuringAction.REROUTE
                else:
                    decisions[rid] = RestructuringAction.KEEP
        else:  # independent
            if prod < MIN_PRODUCTIVITY:
                # Reduction is available whenever eliminate is
                if "eliminate" in allowed:
                    decisions[rid] = RestructuringAction.REDUCE
                else:
                    decisions[rid] = RestructuringAction.KEEP
            else:
                decisions[rid] = RestructuringAction.KEEP

    return decisions


def check_coverage_equity(
    decisions: Dict[str, RestructuringAction],
    routes: List["BusRoute"],
    bus_stops_by_route: Dict[str, np.ndarray],
    parcel_xy: np.ndarray,
    parcel_se01_share: np.ndarray,
    metro_se01_share: float,
) -> Dict[str, RestructuringAction]:
    """Title VI equity guard — prevent disproportionate service cuts to
    low-income areas.

    For each route slated for ELIMINATE or REDUCE, compute the SE01 share
    of parcels within 400m of the route's stops.  If the route serves
    ≥ EQUITY_DISPARITY_THRESHOLD × metro average, downgrade the action:
    ELIMINATE → REDUCE, REDUCE → KEEP.

    Parameters
    ----------
    decisions : dict
        route_id → RestructuringAction from decide_route_restructuring().
    routes : list of BusRoute
        Full route list.
    bus_stops_by_route : dict
        route_id → (N, 2) stop coordinates in meters.
    parcel_xy : ndarray
        (M, 2) parcel coordinates in meters.
    parcel_se01_share : ndarray
        (M,) SE01 fraction per parcel (0-1).
    metro_se01_share : float
        Metro-wide SE01 average share (for disparity comparison).
    """
    from scipy.spatial import cKDTree

    if len(parcel_xy) == 0 or metro_se01_share <= 0:
        return decisions

    parcel_tree = cKDTree(parcel_xy)
    adjusted = dict(decisions)

    for route in routes:
        rid = route.route_id
        action = adjusted.get(rid, RestructuringAction.KEEP)
        if action not in (RestructuringAction.ELIMINATE, RestructuringAction.REDUCE):
            continue

        stops = bus_stops_by_route.get(str(rid))
        if stops is None or len(stops) == 0:
            continue

        # Find ALL parcels within 400m of any stop (not just nearest)
        hit_lists = parcel_tree.query_ball_point(stops, r=FEEDER_WALK_M)
        nearby_parcel_idx = set()
        for hits in hit_lists:
            nearby_parcel_idx.update(hits)
        if not nearby_parcel_idx:
            continue

        unique_idx = np.array(sorted(nearby_parcel_idx))
        route_se01 = float(parcel_se01_share[unique_idx].mean())

        if route_se01 > metro_se01_share * EQUITY_DISPARITY_THRESHOLD:
            # Downgrade action to protect low-income riders
            if action == RestructuringAction.ELIMINATE:
                adjusted[rid] = RestructuringAction.REDUCE
            elif action == RestructuringAction.REDUCE:
                adjusted[rid] = RestructuringAction.KEEP

    return adjusted


def apply_restructuring_decisions(
    routes: List["BusRoute"],
    decisions: Dict[str, RestructuringAction],
    min_headway: float = 10.0,
    max_headway: float = 120.0,
) -> Dict[str, float]:
    """Apply restructuring decisions to route headways.

    Returns a dict mapping route_id → target headway (minutes).
    Does NOT modify routes in place — caller applies the returned
    headways to route objects.

    Actions:
    - KEEP: headway unchanged (baseline)
    - ELIMINATE: headway set to 0.0 (route removed from service)
    - REDUCE: headway increased by 1.5× (less frequent)
    - REROUTE: headway unchanged (same veh-hrs, different alignment)
    - ENHANCE: headway reduced by 0.5× (more frequent, subject to budget)
    """
    target_headways: Dict[str, float] = {}

    for route in routes:
        rid = route.route_id
        action = decisions.get(rid, RestructuringAction.KEEP)
        baseline_hw = route.baseline_headway_min

        if action == RestructuringAction.ELIMINATE:
            target_headways[rid] = 0.0  # route removed
        elif action == RestructuringAction.REDUCE:
            target_headways[rid] = min(baseline_hw * 1.5, max_headway)
        elif action == RestructuringAction.ENHANCE:
            target_headways[rid] = max(baseline_hw * 0.5, min_headway)
        elif action == RestructuringAction.REROUTE:
            target_headways[rid] = baseline_hw  # same veh-hrs
        else:  # KEEP
            target_headways[rid] = baseline_hw

    return target_headways


# ---------------------------------------------------------------------------
# Transit Signal Priority (TSP)
# ---------------------------------------------------------------------------

TSP_BASE_SPEED_IMPROVEMENT = 0.12    # 12% speed gain at reference signal density (TCRP 118)
TSP_SIGNAL_DENSITY_REF = 4.0         # signals/km (urban arterial reference)
TSP_ACTIVATION_YEAR = 3              # available from early operations phase
TSP_RIDERSHIP_THRESHOLD = 1500       # daily riders to justify TSP investment


def compute_tsp_speed_factor(
    route_length_km: float,
    n_signals_on_route: int,
    daily_apm_riders: float,
    year: int,
) -> float:
    """Speed improvement factor from TSP for a feeder route.

    Returns a multiplier >= 1.0 to apply to feeder bus speed.
    Only active after year 3 and when APM ridership justifies investment.
    """
    if year < TSP_ACTIVATION_YEAR:
        return 1.0
    if daily_apm_riders < TSP_RIDERSHIP_THRESHOLD:
        return 1.0
    signal_density = n_signals_on_route / max(route_length_km, 0.1)
    density_factor = min(signal_density / TSP_SIGNAL_DENSITY_REF, 1.0)
    return 1.0 + TSP_BASE_SPEED_IMPROVEMENT * density_factor


def count_signals_along_route(
    route_stops_xy: np.ndarray,
    signal_nodes_xy: np.ndarray,
    buffer_m: float = 100.0,
) -> int:
    """Count traffic signals within buffer of a bus route's path.

    Uses route stops as a proxy for route path (stops are typically
    at or near intersections). A signal within 100m of any stop is
    counted once.
    """
    from scipy.spatial import cKDTree

    if len(signal_nodes_xy) == 0 or len(route_stops_xy) == 0:
        return 0
    tree = cKDTree(signal_nodes_xy)
    dists, _ = tree.query(route_stops_xy, k=1)
    return int((dists <= buffer_m).sum())


# ---------------------------------------------------------------------------
# Proof of Payment (PoP) / All-Door Boarding
# ---------------------------------------------------------------------------

POP_SPEED_IMPROVEMENT = 0.06         # 6% system-wide speed gain (TCRP 165 midpoint)
POP_TRANSFER_PENALTY_REDUCTION = 0.5  # minutes saved at transfer point
POP_ACTIVATION_YEAR = 0              # policy decision, active from opening


def compute_pop_speed_factor(pop_active: bool) -> float:
    """Speed improvement from proof-of-payment / all-door boarding."""
    return 1.0 + POP_SPEED_IMPROVEMENT if pop_active else 1.0


@dataclass
class BusRoute:
    """Per-route state for the dynamic bus network."""
    route_id: str
    name: str
    length_km: float
    baseline_headway_min: float
    current_headway_min: float
    cycle_time_min: float
    n_stops: int
    trips_per_day: float
    observed_daily_riders: float = 0.0
    avg_speed_kph: float = DEFAULT_BUS_SPEED_KPH  # computed from GTFS if available
    classification: str = "independent"  # parallel | feeder | independent
    is_express: bool = False
    service_profile: Optional[ServiceProfile] = None

    @property
    def round_trip_km(self) -> float:
        return self.length_km * 2.0

    @property
    def vehicles_needed(self) -> int:
        """Peak vehicles required = ceil(cycle_time / headway) + spares."""
        if self.current_headway_min <= 0:
            return 0
        raw = math.ceil(self.cycle_time_min / self.current_headway_min)
        return max(1, int(math.ceil(raw * (1.0 + SPARE_RATIO))))

    @property
    def daily_vehicle_hours(self) -> float:
        """Vehicle-hours per day, using ServiceProfile periods when available.

        Period spans (TCRP 165 small-urban weekday):
          AM peak  3h (07-10), Midday 6h (10-16), PM peak 3h (16-19), Evening 3h (19-22)
        When no profile exists, falls back to flat headway × 16h span.
        Profile headways are scaled by the ratio current_headway / baseline_headway
        so that restructuring (headway changes) are reflected in period costs.
        """
        if self.current_headway_min <= 0:
            return 0.0
        if self.service_profile is not None and self.baseline_headway_min > 0:
            ct = self.cycle_time_min
            hw_ratio = self.current_headway_min / self.baseline_headway_min
            # Vehicles needed in each period = ceil(cycle_time / scaled_period_headway)
            periods = [
                (3.0, self.service_profile.am_peak_headway_min * hw_ratio),
                (6.0, self.service_profile.midday_headway_min * hw_ratio),
                (3.0, self.service_profile.pm_peak_headway_min * hw_ratio),
                (3.0, self.service_profile.evening_headway_min * hw_ratio),
            ]
            total = 0.0
            for span_h, hw in periods:
                if hw > 0:
                    vehs = math.ceil(ct / max(hw, 1.0))
                    total += vehs * span_h
            if total > 0:
                return total
        vehicles_in_service = math.ceil(self.cycle_time_min / self.current_headway_min)
        return vehicles_in_service * DEFAULT_SERVICE_SPAN_HOURS

    @property
    def annual_vehicle_hours(self) -> float:
        """Weekday-equivalent annual vehicle-hours (300 days)."""
        return self.daily_vehicle_hours * 300.0


@dataclass
class APMService:
    """APM line operating parameters."""
    corridor_id: str
    length_km: float
    n_stops: int
    headway_min: float = 5.0
    line_speed_kph: float = _FP_APM_SPEED   # 27 km/h avg (was 40 cruise)
    dwell_time_s: float = _FP_APM_DWELL     # 25s (platform doors, level boarding)
    daily_riders: float = 0.0

    @property
    def cycle_time_min(self) -> float:
        """Round-trip time including dwell at intermediate stops and terminals."""
        cruise_time_h = self.length_km / max(self.line_speed_kph, 1.0)
        intermediate = max(self.n_stops - 2, 0)
        stop_time_h = intermediate * (self.dwell_time_s + 15.0) / 3600.0
        terminal_h = 2.0 * 60.0 / 3600.0  # 60s turnaround at each terminal
        one_way_h = cruise_time_h + stop_time_h
        return (one_way_h * 2.0 + terminal_h) * 60.0  # round-trip minutes

    @property
    def vehicles_needed(self) -> int:
        if self.headway_min <= 0:
            return 0
        raw = math.ceil(self.cycle_time_min / self.headway_min)
        return max(1, int(math.ceil(raw * (1.0 + SPARE_RATIO))))

    @property
    def daily_vehicle_hours(self) -> float:
        """Daily vehicle-hours for the APM service.

        ``veh`` is the number of *trains* needed.  We multiply by
        APM_VEHICLES_PER_TRAIN (2) to get individual vehicle count
        because APM_COST_PER_VEH_HOUR is a per-vehicle-hour cost,
        not per-train-hour (consistent with NTD reporting for AGT where
        each car is a "vehicle").
        """
        if self.headway_min <= 0:
            return 0.0
        veh = math.ceil(self.cycle_time_min / self.headway_min)
        return veh * APM_VEHICLES_PER_TRAIN * DEFAULT_SERVICE_SPAN_HOURS

    @property
    def annual_vehicle_hours(self) -> float:
        return self.daily_vehicle_hours * _FP_SERVICE_DAYS


# ---------------------------------------------------------------------------
# Operating Cost Model
# ---------------------------------------------------------------------------

class BusOperatingCostModel:
    """Computes bus and APM operating costs from vehicle-hours.

    Supports three budget modes controlling how APM O&M interacts with the
    bus operating budget:

    - **separate** (default): APM funded independently via TIF + fares.
      Bus gets full $13.5M budget.
    - **combined**: Single transit authority pays both.  APM O&M (net of
      fare revenue) is deducted from the bus budget.
    - **expanded**: Combined, but with incremental FTA 5307 formula funds
      and Indiana PMTF state operating assistance.

    When ``budget_mode`` is *combined* or *expanded*, ``annual_budget``
    becomes a derived property that shrinks for longer / more-station
    corridors.  The existing ``RouteServicePlan`` and
    ``NetworkRedesignStrategy`` budget enforcement automatically tightens.

    Parameters
    ----------
    cost_per_veh_hour : $/vehicle-hour for bus operations (NTD-derived)
    apm_cost_per_veh_hour : $/vehicle-hour for APM (automated, lower labor)
    base_budget : total annual transit operating budget before APM draw ($)
    budget_mode : 'separate', 'combined', or 'expanded'
    apm_om_annual : corridor-specific APM O&M cost ($/yr)
    apm_fare_revenue_annual : APM fare revenue directed toward O&M ($/yr)
    year : simulation year (for fare escalation)
    """

    def __init__(
        self,
        cost_per_veh_hour: float = DEFAULT_COST_PER_VEH_HOUR,
        apm_cost_per_veh_hour: float = APM_COST_PER_VEH_HOUR,
        annual_budget: float = CITYBUS_ANNUAL_BUDGET_USD,
        budget_mode: str = DEFAULT_BUDGET_MODE,
        apm_om_annual: float = 0.0,
        apm_fare_revenue_annual: float = 0.0,
        year: int = 0,
    ):
        self.cost_per_veh_hour = cost_per_veh_hour
        self.apm_cost_per_veh_hour = apm_cost_per_veh_hour
        self._base_budget = annual_budget
        self.budget_mode = budget_mode
        self.apm_om_annual = apm_om_annual
        self.apm_fare_revenue_annual = apm_fare_revenue_annual
        self.year = year

    @property
    def annual_budget(self) -> float:
        """Effective bus operating budget after APM cost draw.

        In combined/expanded modes, APM O&M (net of fare revenue directed
        toward O&M) is deducted from the base bus budget.  Fare revenue
        offsets APM O&M first — any surplus stays with bus operations.
        """
        if self.budget_mode == BUDGET_MODE_SEPARATE:
            return self._base_budget

        # Net APM cost after fare revenue offset
        net_apm_cost = max(self.apm_om_annual - self.apm_fare_revenue_annual, 0.0)

        if self.budget_mode == BUDGET_MODE_COMBINED:
            return max(self._base_budget - net_apm_cost, 0.0)

        if self.budget_mode == BUDGET_MODE_EXPANDED:
            # Incremental FTA 5307 + Indiana PMTF on the combined system
            total_ops = self._base_budget + self.apm_om_annual
            expanded_total = (
                self._base_budget
                + FTA_5307_OPERATING_INCREMENT_USD
                + total_ops * INDIANA_PMTF_RATE
            )
            return max(expanded_total - net_apm_cost, 0.0)

        return self._base_budget  # fallback

    @annual_budget.setter
    def annual_budget(self, value: float) -> None:
        """Allow direct assignment for backward compatibility."""
        self._base_budget = value

    def route_annual_cost(self, route: BusRoute) -> float:
        """Annual operating cost for a single bus route."""
        return route.annual_vehicle_hours * self.cost_per_veh_hour

    def apm_annual_cost(self, apm: APMService) -> float:
        """Annual operating cost for APM line."""
        return apm.annual_vehicle_hours * self.apm_cost_per_veh_hour

    def system_annual_cost(
        self,
        routes: List[BusRoute],
        apm_services: Optional[List[APMService]] = None,
    ) -> Dict[str, float]:
        """Total system cost breakdown.

        Budget reflects the effective amount available for bus operations
        under the current budget_mode (separate/combined/expanded).
        """
        bus_cost = sum(self.route_annual_cost(r) for r in routes)
        apm_cost = 0.0
        if apm_services:
            apm_cost = sum(self.apm_annual_cost(a) for a in apm_services)
        eff_budget = self.annual_budget
        return {
            "bus_annual_cost": bus_cost,
            "apm_annual_cost": apm_cost,
            "total_annual_cost": bus_cost + apm_cost,
            "budget": eff_budget,
            "budget_surplus": eff_budget - bus_cost,
            "budget_utilization": bus_cost / max(eff_budget, 1.0),
            "budget_mode": self.budget_mode,
            "apm_om_annual": self.apm_om_annual,
            "apm_fare_revenue_annual": self.apm_fare_revenue_annual,
            "net_apm_cost": max(self.apm_om_annual - self.apm_fare_revenue_annual, 0.0),
            "bus_budget_reduction_pct": (
                (1.0 - eff_budget / max(self._base_budget, 1.0)) * 100.0
            ),
        }

    def fleet_summary(
        self,
        routes: List[BusRoute],
        apm_services: Optional[List[APMService]] = None,
    ) -> Dict[str, object]:
        """Fleet sizing summary."""
        bus_vehicles = sum(r.vehicles_needed for r in routes)
        bus_veh_hours = sum(r.annual_vehicle_hours for r in routes)
        apm_vehicles = 0
        apm_veh_hours = 0.0
        if apm_services:
            apm_vehicles = sum(a.vehicles_needed for a in apm_services)
            apm_veh_hours = sum(a.annual_vehicle_hours for a in apm_services)
        return {
            "bus_vehicles": bus_vehicles,
            "bus_annual_veh_hours": bus_veh_hours,
            "apm_vehicles": apm_vehicles,
            "apm_annual_veh_hours": apm_veh_hours,
            "total_vehicles": bus_vehicles + apm_vehicles,
        }


# ---------------------------------------------------------------------------
# Route-APM Classification
# ---------------------------------------------------------------------------

STATION_CONNECTIVITY_M = 200.0  # max distance from bus stop to APM station for transfer


def _interpolate_corridor_points(
    corridor_stops_xy: np.ndarray,
    spacing_m: float = 100.0,
) -> np.ndarray:
    """Interpolate points along the corridor polyline at regular spacing.

    This produces a dense set of points representing the full corridor
    alignment between stations, not just station locations.  Used for
    more accurate route-overlap classification.
    """
    if len(corridor_stops_xy) < 2:
        return corridor_stops_xy
    _xy = np.asarray(corridor_stops_xy) if not isinstance(corridor_stops_xy, np.ndarray) else corridor_stops_xy
    pts = [_xy[0]]
    for i in range(len(_xy) - 1):
        a, b = _xy[i], _xy[i + 1]
        seg_len = float(np.linalg.norm(b - a))
        if seg_len < 1e-6:
            continue
        n_interp = max(1, int(seg_len / spacing_m))
        for j in range(1, n_interp + 1):
            t = j / n_interp
            pts.append(a + t * (b - a))
    return np.array(pts)


def classify_routes_for_corridor(
    routes: List[BusRoute],
    corridor_stops_xy: np.ndarray,
    bus_stops_by_route: Dict[str, np.ndarray],
    buffer_m: float = CORRIDOR_BUFFER_M,
    station_connectivity_m: float = STATION_CONNECTIVITY_M,
) -> List[BusRoute]:
    """Classify each bus route as parallel, feeder, or independent.

    Parameters
    ----------
    routes : list of BusRoute objects
    corridor_stops_xy : (N, 2) array of APM stop coordinates in projected CRS (meters)
    bus_stops_by_route : dict mapping route_id → (M, 2) array of bus stop coords
    buffer_m : buffer distance for overlap test
    station_connectivity_m : max distance for a bus stop to count as connected to an
        APM station.  A route must have at least one stop within this distance of a
        station to be classified as "feeder"; otherwise it is demoted to "independent"
        even if geometric overlap is in the feeder range (15-40%).

    Returns
    -------
    routes with updated `classification` field
    """
    if len(corridor_stops_xy) < 2:
        return routes

    from scipy.spatial import cKDTree

    # Interpolate corridor alignment at 100m spacing for overlap test
    corridor_line_pts = _interpolate_corridor_points(corridor_stops_xy, spacing_m=100.0)
    line_tree = cKDTree(corridor_line_pts)

    # Station-only tree for connectivity check (transfers happen at stations)
    station_tree = cKDTree(corridor_stops_xy)

    for route in routes:
        rid = route.route_id
        if rid not in bus_stops_by_route:
            route.classification = "independent"
            continue

        bus_xy = bus_stops_by_route[rid]
        if len(bus_xy) == 0:
            route.classification = "independent"
            continue

        # Overlap: fraction of bus stops within buffer_m of the corridor *line*
        dists_line, _ = line_tree.query(bus_xy, k=1)
        within = float(np.sum(dists_line <= buffer_m))
        overlap_frac = within / len(bus_xy)

        # Station connectivity: can passengers actually transfer?
        # At least one bus stop must be within station_connectivity_m of an APM station.
        dists_station, _ = station_tree.query(bus_xy, k=1)
        has_station_connection = bool(np.min(dists_station) <= station_connectivity_m)

        if overlap_frac >= PARALLEL_OVERLAP_THRESHOLD:
            route.classification = "parallel"
        elif overlap_frac >= FEEDER_OVERLAP_THRESHOLD and has_station_connection:
            route.classification = "feeder"
        else:
            route.classification = "independent"

    return routes


# ---------------------------------------------------------------------------
# OD-based route classification
# ---------------------------------------------------------------------------

# Thresholds for OD-based classification
OD_PARALLEL_THRESHOLD = 0.50  # >50% of route's OD pairs both in APM catchment -> parallel
OD_FEEDER_THRESHOLD = 0.20   # 20-50% -> feeder (complementary), <20% -> independent


def classify_routes_od_based(
    routes: List[BusRoute],
    bus_stops_by_route: Dict[str, np.ndarray],
    corridor_stops_xy: np.ndarray,
    od_origins: np.ndarray,
    od_dests: np.ndarray,
    od_flows_s000: np.ndarray,
    parcel_xy: np.ndarray,
    apm_catchment_m: float = 1200.0,
    bus_catchment_m: float = 400.0,
    station_connectivity_m: float = STATION_CONNECTIVITY_M,
) -> List[BusRoute]:
    """Classify routes using LODES OD passenger-trip overlap with APM catchment.

    For each bus route, computes what fraction of the route's passenger-trips
    have BOTH origin and destination within the APM catchment.  This correctly
    handles crosstown routes that geometrically overlap APM but serve different
    OD markets (classified as feeder/independent, not parallel).

    A route must also have at least one stop within ``station_connectivity_m``
    of an APM station to be classified as "feeder" (passengers must be able to
    actually transfer).

    Parameters
    ----------
    routes : list of BusRoute
    bus_stops_by_route : route_id -> (n_stops, 2) array
    corridor_stops_xy : (n_apm_stops, 2) projected APM stop coords
    od_origins : (n_flows,) origin parcel indices
    od_dests : (n_flows,) destination parcel indices
    od_flows_s000 : (n_flows,) total workers per OD pair
    parcel_xy : (n_parcels, 2) projected parcel coords
    apm_catchment_m : APM walk-zone catchment radius
    bus_catchment_m : bus stop walk catchment radius
    station_connectivity_m : max bus-stop-to-station distance for transfer feasibility
    """
    if len(corridor_stops_xy) < 2 or len(od_origins) == 0:
        # Fall back to geometric if OD data unavailable
        return routes

    from scipy.spatial import cKDTree

    apm_tree = cKDTree(corridor_stops_xy)
    parcel_tree = cKDTree(parcel_xy)

    # Pre-compute which parcels are in APM catchment
    apm_parcel_dists, _ = apm_tree.query(parcel_xy, k=1)
    in_apm_catchment = apm_parcel_dists <= apm_catchment_m

    for route in routes:
        rid = str(route.route_id)
        if rid not in bus_stops_by_route or len(bus_stops_by_route[rid]) == 0:
            route.classification = "independent"
            continue

        bus_xy = bus_stops_by_route[rid]

        # Find parcels within bus_catchment_m of any stop on this route.
        # Query each bus stop against the parcel tree (avoids building a
        # per-route KDTree — the parcel tree is built once outside the loop).
        parcel_near_lists = parcel_tree.query_ball_point(bus_xy, r=bus_catchment_m)
        served_parcels = set()
        for matches in parcel_near_lists:
            if isinstance(matches, (list, np.ndarray)):
                served_parcels.update(matches)

        if not served_parcels:
            route.classification = "independent"
            continue

        # Find OD pairs where origin is served by this bus route
        served_indices = np.array(list(served_parcels), dtype=np.intp)
        served_mask = np.zeros(len(parcel_xy), dtype=bool)
        served_mask[served_indices] = True

        # OD pairs where origin is served by this bus route
        origin_served = served_mask[od_origins]
        # Filter to flows that touch this route
        relevant = origin_served
        if not np.any(relevant):
            route.classification = "independent"
            continue

        # Of those flows, what fraction have BOTH origin AND dest in APM catchment?
        rel_origins = od_origins[relevant]
        rel_dests = od_dests[relevant]
        rel_flows = od_flows_s000[relevant]

        both_in_apm = in_apm_catchment[rel_origins] & in_apm_catchment[rel_dests]
        total_flow = float(np.sum(rel_flows))
        if total_flow <= 0:
            route.classification = "independent"
            continue

        apm_overlap_frac = float(np.sum(rel_flows[both_in_apm])) / total_flow

        # Station connectivity check: at least one bus stop must be within
        # station_connectivity_m of an APM station for a transfer to be feasible
        bus_to_apm_dists, _ = apm_tree.query(bus_xy, k=1)
        has_station_connection = bool(np.min(bus_to_apm_dists) <= station_connectivity_m)

        if apm_overlap_frac >= OD_PARALLEL_THRESHOLD:
            route.classification = "parallel"
        elif apm_overlap_frac >= OD_FEEDER_THRESHOLD and has_station_connection:
            route.classification = "feeder"
        else:
            route.classification = "independent"

    return routes


# ---------------------------------------------------------------------------
# Route truncation at APM stations
# ---------------------------------------------------------------------------

def truncate_route_at_station(
    route: BusRoute,
    bus_stops_xy: np.ndarray,
    corridor_stops_xy: np.ndarray,
    station_buffer_m: float = 200.0,
) -> Optional[BusRoute]:
    """Truncate a bus route at the nearest APM station, returning the outer segment.

    The outer segment (beyond the station) becomes a feeder route that
    terminates at the APM station.  The inner segment (duplicating APM) is
    dropped.

    Parameters
    ----------
    route : BusRoute to truncate
    bus_stops_xy : (n_stops, 2) stops of this route in projected coords
    corridor_stops_xy : (n_apm_stops, 2) APM station coords
    station_buffer_m : max distance for a bus stop to be "at" an APM station

    Returns
    -------
    A new BusRoute representing the truncated feeder segment, or None if
    truncation isn't feasible (route doesn't pass near enough to a station).
    """
    if len(bus_stops_xy) < 3 or len(corridor_stops_xy) < 1:
        return None

    from scipy.spatial import cKDTree
    apm_tree = cKDTree(corridor_stops_xy)

    # Find the bus stop closest to any APM station
    dists, apm_idx = apm_tree.query(bus_stops_xy, k=1)
    nearest_stop = int(np.argmin(dists))
    min_dist = float(dists[nearest_stop])

    if min_dist > station_buffer_m:
        return None  # route doesn't pass near enough to an APM station

    # Split the route at the nearest station
    # Keep the longer segment as the feeder
    n_stops = len(bus_stops_xy)
    left_len = nearest_stop
    right_len = n_stops - nearest_stop - 1

    if left_len >= right_len and left_len >= 2:
        # Left segment is longer — feeder serves left side
        retained_stops = nearest_stop + 1  # include the station stop
        retained_frac = retained_stops / n_stops
    elif right_len >= 2:
        # Right segment is longer
        retained_stops = right_len + 1
        retained_frac = retained_stops / n_stops
    else:
        return None  # both segments too short

    # Create truncated feeder route
    import copy
    feeder = copy.deepcopy(route)
    feeder.route_id = f"{route.route_id}_trunc"
    feeder.name = f"{route.name} (truncated feeder)" if route.name else f"Route {route.route_id} feeder"
    feeder.classification = "feeder"

    # Shorter route -> shorter cycle time -> same fleet runs more trips
    feeder.cycle_time_min = route.cycle_time_min * retained_frac
    # Headway improves proportionally (same vehicles, shorter route)
    feeder.baseline_headway_min = max(route.baseline_headway_min * retained_frac, 10.0)
    feeder.current_headway_min = feeder.baseline_headway_min

    return feeder


def truncate_parallel_routes(
    routes: List[BusRoute],
    bus_stops_by_route: Dict[str, np.ndarray],
    corridor_stops_xy: np.ndarray,
) -> tuple:
    """Truncate all parallel-classified routes at APM stations.

    Returns (truncated_feeders, truncated_stops) where each is a list/dict
    of new feeder routes created from truncation.
    """
    truncated_feeders = []
    truncated_stops = {}

    for route in routes:
        if route.classification != "parallel":
            continue

        stops_xy = bus_stops_by_route.get(str(route.route_id), np.empty((0, 2)))
        feeder = truncate_route_at_station(route, stops_xy, corridor_stops_xy)
        if feeder is not None:
            truncated_feeders.append(feeder)
            # Store only the retained segment's stops so sector coverage
            # doesn't count stops in the dropped part of the route.
            retained_stops = _extract_retained_stops(stops_xy, corridor_stops_xy)
            truncated_stops[feeder.route_id] = retained_stops

    return truncated_feeders, truncated_stops


def _extract_retained_stops(
    bus_stops_xy: np.ndarray,
    corridor_stops_xy: np.ndarray,
) -> np.ndarray:
    """Return the subset of bus stops belonging to the retained feeder segment.

    Mirrors the split logic in ``truncate_route_at_station``: find the bus
    stop closest to any APM station, then keep the longer segment.
    """
    if len(bus_stops_xy) < 3 or len(corridor_stops_xy) < 1:
        return bus_stops_xy

    from scipy.spatial import cKDTree
    apm_tree = cKDTree(corridor_stops_xy)
    dists, _ = apm_tree.query(bus_stops_xy, k=1)
    nearest = int(np.argmin(dists))

    n = len(bus_stops_xy)
    left_len = nearest
    right_len = n - nearest - 1

    if left_len >= right_len and left_len >= 2:
        return bus_stops_xy[: nearest + 1]
    elif right_len >= 2:
        return bus_stops_xy[nearest:]
    else:
        return bus_stops_xy


# ---------------------------------------------------------------------------
# APM Frequency Response
# ---------------------------------------------------------------------------

def compute_apm_headway(
    daily_riders: float,
    corridor_length_km: float = 0.0,
    n_stops: int = 0,
    directional_split: float = 0.0,
    train_capacity: int | None = None,
) -> float:
    """Two-regime APM headway: capacity constraint AND service-quality target.

    The headway is the **tighter** (shorter) of two regimes:

    1. **Capacity regime**: headway set so peak-hour peak-direction demand
       does not exceed train capacity × load factor.
    2. **Service-quality regime**: headway decreases logarithmically with
       demand — agencies invest in frequency to maintain ridership
       attractiveness, not just to avoid overcrowding.

    Both regimes are then constrained by:
    - Signal/dwell minimum (90 s)
    - Fleet size capital limit (if corridor geometry is provided)
    - Policy maximum (10 min)

    Parameters
    ----------
    daily_riders : float
        Total daily boardings (both directions).
    corridor_length_km : float, optional
        Corridor length for cycle-time / fleet constraint.
    n_stops : int, optional
        Number of stations (for cycle-time calculation).
    directional_split : float, optional
        Peak-direction share (0.50-0.80).  When 0 or not provided, falls
        back to the global APM_PEAK_DIR_SPLIT (0.60).
    train_capacity : int, optional
        Override for APM_TRAIN_CAPACITY (default 100 pax for 2-car train).
        Set to 200 for 4-car consists.

    Returns
    -------
    float
        Headway in minutes, clipped to [APM_MIN_HEADWAY_MIN, APM_MAX_HEADWAY_MIN].
    """
    if daily_riders <= 0:
        return APM_MAX_HEADWAY_MIN

    # --- Regime 1: Capacity constraint ---
    # Peak-hour, peak-direction demand (corridor-specific split if available)
    _dir_split = directional_split if 0.50 <= directional_split <= 0.80 else APM_PEAK_DIR_SPLIT
    peak_pphd = daily_riders * APM_PEAK_HOUR_FACTOR * _dir_split
    _train_cap = train_capacity if train_capacity is not None else APM_TRAIN_CAPACITY
    effective_capacity = _train_cap * APM_TARGET_LOAD_FACTOR
    if effective_capacity > 0 and peak_pphd > 0:
        trains_per_hour = peak_pphd / effective_capacity
        capacity_headway = 60.0 / trains_per_hour
    else:
        capacity_headway = APM_MAX_HEADWAY_MIN

    # --- Regime 2: Service-quality target ---
    # Logarithmic: headway = MAX - scale * ln(1 + riders/ref)
    # Calibrated for 2-car (100 pax): ~1K → ~8 min, ~3K → ~5 min, ~6K → ~3 min
    # Consist-size adjustment: larger trains can maintain acceptable service
    # at longer headways because throughput = capacity / headway.  A 4-car
    # train at 6 min delivers the same capacity as 2-car at 3 min.
    # Scale the demand reference by train_cap/100 so that 4-car trains reach
    # the same SQ headway at 2× the ridership (i.e., SQ curve shifts right).
    _cap_ratio = _train_cap / APM_TRAIN_CAPACITY  # 1.0 for 2-car, 2.0 for 4-car
    _adj_ref = APM_SQ_DEMAND_REF * _cap_ratio
    log_term = math.log(1.0 + daily_riders / _adj_ref)
    sq_headway = max(APM_MIN_HEADWAY_MIN, APM_MAX_HEADWAY_MIN - APM_SQ_HEADWAY_SCALE * log_term)

    # Take the tighter (shorter) of the two regimes
    headway = min(capacity_headway, sq_headway)

    # --- Physical constraints ---
    # Fleet constraint: can't run more trains than we own
    if corridor_length_km > 0 and n_stops > 0:
        cruise_h = corridor_length_km / max(_FP_APM_SPEED, 1.0)
        intermediate = max(n_stops - 2, 0)
        stop_h = intermediate * (_FP_APM_DWELL + 15.0) / 3600.0  # dwell + accel/decel
        terminal_h = 2.0 * 60.0 / 3600.0  # 60s turnaround at each terminal × 2
        cycle_min = (cruise_h + stop_h + terminal_h) * 2.0 * 60.0
        fleet_min_headway = cycle_min / max(APM_MAX_FLEET_SIZE, 1)
        headway = max(headway, fleet_min_headway)

    # Clip to physical and policy bounds
    headway = max(APM_MIN_HEADWAY_MIN, min(APM_MAX_HEADWAY_MIN, headway))
    return round(headway, 2)


# BRT headway parameters
BRT_SQ_MIN_HEADWAY_MIN = 5.0   # Service-quality floor (soft) — desired minimum
BRT_HARD_MIN_HEADWAY_MIN = 2.0 # Physical floor — dedicated-lane BRT with TSP
BRT_MIN_HEADWAY_MIN = BRT_SQ_MIN_HEADWAY_MIN  # Back-compat alias
BRT_MAX_HEADWAY_MIN = 10.0     # FTA service standard for high-frequency BRT
BRT_VEHICLE_CAPACITY = 60      # 60-ft articulated bus
BRT_TARGET_LOAD_FACTOR = 0.85  # Standing load acceptable for BRT
BRT_PEAK_HOUR_FACTOR = 0.12    # Same peaking as APM
BRT_PEAK_DIR_SPLIT = 0.60      # Same directional split
BRT_MAX_FLEET_SIZE = 25        # Typical BRT fleet for a single corridor
BRT_SQ_DEMAND_REF = 1500.0     # Service-quality reference ridership
BRT_SQ_HEADWAY_SCALE = 2.0     # Gentler frequency response than APM


def compute_brt_headway(
    daily_riders: float,
    corridor_length_km: float = 0.0,
    n_stops: int = 0,
    directional_split: float = 0.0,
) -> float:
    """Two-regime BRT headway: capacity + service-quality.

    Same structure as compute_apm_headway but with BRT parameters:
    - 60 pax/vehicle (articulated bus), 0.85 load factor
    - Service-quality floor 5 min (soft — capacity can push below this)
    - Hard physical minimum 2 min (dedicated-lane BRT with TSP)
    - Max headway 10 min (FTA high-frequency BRT standard)
    - Service-quality curve recalibrated for BRT ridership levels

    Returns headway in minutes, clipped to [2.0, 10.0].
    """
    if daily_riders <= 0:
        return BRT_MAX_HEADWAY_MIN

    # --- Regime 1: Capacity constraint ---
    _dir_split = directional_split if 0.50 <= directional_split <= 0.80 else BRT_PEAK_DIR_SPLIT
    peak_pphd = daily_riders * BRT_PEAK_HOUR_FACTOR * _dir_split
    effective_capacity = BRT_VEHICLE_CAPACITY * BRT_TARGET_LOAD_FACTOR
    if effective_capacity > 0 and peak_pphd > 0:
        vehicles_per_hour = peak_pphd / effective_capacity
        capacity_headway = 60.0 / vehicles_per_hour
    else:
        capacity_headway = BRT_MAX_HEADWAY_MIN

    # --- Regime 2: Service-quality target ---
    # Service-quality uses the soft floor (5 min) — frequency won't improve
    # past 5 min for service reasons alone, only capacity can push below.
    log_term = math.log(1.0 + daily_riders / BRT_SQ_DEMAND_REF)
    sq_headway = max(BRT_SQ_MIN_HEADWAY_MIN, BRT_MAX_HEADWAY_MIN - BRT_SQ_HEADWAY_SCALE * log_term)

    # Take the tighter (shorter) of the two regimes.
    # Capacity can push below the 5-min service floor down to the hard minimum.
    headway = min(capacity_headway, sq_headway)

    # Fleet constraint
    if corridor_length_km > 0 and n_stops > 0:
        cruise_h = corridor_length_km / max(_FP_BRT_SPEED, 1.0)
        intermediate = max(n_stops - 2, 0)
        stop_h = intermediate * (_FP_BRT_DWELL + 10.0) / 3600.0  # dwell + accel/decel
        terminal_h = 2.0 * 120.0 / 3600.0  # 120s layover at each terminal × 2
        cycle_min = (cruise_h + stop_h + terminal_h) * 2.0 * 60.0
        fleet_min_headway = cycle_min / max(BRT_MAX_FLEET_SIZE, 1)
        headway = max(headway, fleet_min_headway)

    # Hard physical minimum (2 min) — not the soft service floor (5 min)
    headway = max(BRT_HARD_MIN_HEADWAY_MIN, min(BRT_MAX_HEADWAY_MIN, headway))
    return round(headway, 2)


def compute_transit_headway(
    daily_riders: float,
    corridor_length_km: float = 0.0,
    n_stops: int = 0,
    directional_split: float = 0.0,
    transit_mode: str = "apm",
    train_capacity: int | None = None,
) -> float:
    """Dispatch to APM or BRT headway function based on transit mode.

    ``train_capacity`` is passed through to APM only (BRT uses fixed
    60-pax articulated bus capacity).
    """
    if transit_mode == "brt":
        return compute_brt_headway(daily_riders, corridor_length_km, n_stops, directional_split)
    return compute_apm_headway(
        daily_riders, corridor_length_km, n_stops, directional_split,
        train_capacity=train_capacity,
    )


# ---------------------------------------------------------------------------
# Lazy re-exports from extracted modules (backward compatibility)
# ---------------------------------------------------------------------------
# bus_service_planning imports from bus_network at module level, so eager
# re-imports here would trigger a circular import.  Use __getattr__ instead.

_GTFS_NAMES = {"load_bus_routes_from_gtfs"}
_SERVICE_PLANNING_NAMES = {
    "RouteServicePlan", "NetworkRedesignStrategy",
    "build_corridor_bus_network", "get_service_phase",
    "is_phase_transition_year", "get_elimination_fraction",
    "estimate_bus_route_ridership", "estimate_all_route_ridership",
    "PHASE_TRANSITION_YEARS", "PHASE_CONFIGS",
    "FREQUENCY_ALLOCATION_SHARE", "COVERAGE_ALLOCATION_SHARE",
    "PULSE_HEADWAY_THRESHOLD_MIN", "TRANSFER_TIME_SAVINGS_MIN",
    "REDESIGN_PHASE_IN_YEARS", "REDESIGN_TRIGGER_RIDERSHIP_RATIO",
    "ELIMINATION_OVERLAP_THRESHOLD", "ELIMINATION_PRODUCTIVITY_FLOOR",
    "DEFAULT_TRUNCATION_RETAINED_FRACTION",
    "BUS_STOP_CATCHMENT_M", "BUS_TRIP_RATE",
}


def __getattr__(name):  # noqa: E302
    if name in _GTFS_NAMES:
        from src.bus_gtfs import load_bus_routes_from_gtfs  # noqa: F811
        globals()[name] = load_bus_routes_from_gtfs
        return load_bus_routes_from_gtfs
    if name in _SERVICE_PLANNING_NAMES:
        import src.bus_service_planning as _sp
        for _n in _SERVICE_PLANNING_NAMES:
            globals()[_n] = getattr(_sp, _n)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

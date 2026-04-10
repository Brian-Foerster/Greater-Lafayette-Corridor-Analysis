"""
Bus Service Planning Module
============================

Service planning classes and orchestration for APM corridor bus integration.

Extracted from ``bus_network.py`` — contains:
- RouteServicePlan: budget-constrained per-route headway optimization
- NetworkRedesignStrategy: complete bus network redesign strategy
- Phase management: discrete service phase functions and constants
- Bus ridership estimation: gravity-model + MNL per-route ridership
- build_corridor_bus_network: main public API orchestrator
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports from bus_network.py (data classes, cost model, helpers)
# ---------------------------------------------------------------------------
from src.bus_network import (
    BusRoute,
    APMService,
    ServiceProfile,
    SectorCoverage,
    RestructuringAction,
    BusOperatingCostModel,
    compute_sector_coverage,
    check_coverage_equity,
    apply_restructuring_decisions,
    decide_route_restructuring,
    classify_routes_for_corridor,
    classify_routes_od_based,
    compute_apm_headway,
    compute_brt_headway,
    compute_transit_headway,
    truncate_parallel_routes,
    compute_tsp_speed_factor,
    compute_pop_speed_factor,
    route_productivity_score,
    apply_barrier_penalty,
    _interpolate_corridor_points,
    # Constants
    CORRIDOR_BUFFER_M,
    DEFAULT_BUS_SPEED_KPH,
    FEEDER_OVERLAP_THRESHOLD,
    FARE_ESCALATION_RATE,
    DEFAULT_BUDGET_MODE,
    MIN_PRODUCTIVITY,
    BUDGET_MODE_SEPARATE,
    BUDGET_MODE_COMBINED,
    BUDGET_MODE_EXPANDED,
    BUS_SAVINGS_APM_OFFSET_FRACTION,
    MIN_APM_RIDERSHIP_FOR_RESTRUCTURE,
    POST_OPENING_ADJUSTMENT_YEAR,
)

# ---------------------------------------------------------------------------
# Import from bus_gtfs.py
# ---------------------------------------------------------------------------
from src.bus_gtfs import load_bus_routes_from_gtfs

# ---------------------------------------------------------------------------
# Mode-choice coefficients for bus ridership estimation
# ---------------------------------------------------------------------------
from src.mode_choice import (
    BETA_IN_VEHICLE_TIME as _BETA_IVT,
    BETA_WAIT_TIME as _BETA_WAIT,
    BETA_ACCESS_TIME as _BETA_ACCESS,
    BETA_COST as _BETA_COST,
    ASC_CAR as _ASC_CAR,
    ASC_BUS as _ASC_BUS,
    ASC_WALK as _ASC_WALK,
    SCHEDULE_THRESHOLD_MIN as _SCHEDULE_THRESHOLD_MIN,
    BASE_RELIABILITY_BUFFER_MIN as _BASE_RELIABILITY_BUFFER_MIN,
    effective_wait_time as _effective_wait_time_impl,
)
import logging

logger = logging.getLogger(__name__)

__all__ = [
    # Classes
    "RouteServicePlan",
    "NetworkRedesignStrategy",
    # Phase management
    "get_service_phase",
    "is_phase_transition_year",
    "get_elimination_fraction",
    # Phase constants
    "PHASE_TRANSITION_YEARS",
    "PHASE_CONFIGS",
    "FREQUENCY_ALLOCATION_SHARE",
    "COVERAGE_ALLOCATION_SHARE",
    "PULSE_HEADWAY_THRESHOLD_MIN",
    "TRANSFER_TIME_SAVINGS_MIN",
    "REDESIGN_PHASE_IN_YEARS",
    "REDESIGN_TRIGGER_RIDERSHIP_RATIO",
    "ELIMINATION_OVERLAP_THRESHOLD",
    "ELIMINATION_PRODUCTIVITY_FLOOR",
    "DEFAULT_TRUNCATION_RETAINED_FRACTION",
    # Ridership estimation
    "estimate_bus_route_ridership",
    "estimate_all_route_ridership",
    "BUS_STOP_CATCHMENT_M",
    "BUS_TRIP_RATE",
    # Orchestrator
    "build_corridor_bus_network",
]

# ---------------------------------------------------------------------------
# Network Redesign Constants
# ---------------------------------------------------------------------------

# Route elimination: routes with high parallel overlap AND low productivity
# are candidates for elimination.  NTD small-urban 25th percentile ~10 r/vh.
ELIMINATION_OVERLAP_THRESHOLD = 0.50     # >50% stops near corridor → eliminate
ELIMINATION_PRODUCTIVITY_FLOOR = 10.0    # riders/veh-hr minimum to avoid elimination

# Frequency-coverage tradeoff (Jarrett Walker 2012 "Human Transit",
# Houston METRO 2015 network redesign achieved ~75/25 split)
FREQUENCY_ALLOCATION_SHARE = 0.75   # 75% of freed hours → frequency on retained
COVERAGE_ALLOCATION_SHARE = 0.25    # 25% → synthetic feeder / coverage routes

# Transfer connectivity (Caltrans "Designing Transit Transfer Facilities" 2015)
PULSE_HEADWAY_THRESHOLD_MIN = 15.0  # routes ≤15 min get timed transfers at APM
TRANSFER_TIME_SAVINGS_MIN = 3.0     # pulse scheduling saves ~3 min vs random

# Phase-in timeline (IndyGo Red Line 2019: 18-month phased transition)
REDESIGN_PHASE_IN_YEARS = 3
REDESIGN_TRIGGER_RIDERSHIP_RATIO = 0.50  # redesign begins at 50% of mature target

# Coverage loss is computed as a diagnostic but no longer used as a backstop.
# The prior equity backstop (MAX_ACCEPTABLE_COVERAGE_LOSS = 0.15) was removed:
# route disposition decisions should be driven by ridership productivity and
# budget constraints, not an arbitrary area-coverage floor.

# ---------------------------------------------------------------------------
# Discrete Restructuring Model (replaces continuous pressure)
# ---------------------------------------------------------------------------
# Real transit agencies restructure as planned events, not continuous tuning.
# Based on IndyGo Red Line (2019), Houston METRO New Network (2015),
# LA Metro NextGen (2020).
#
# Two events:
#   Year 0 (opening): full restructuring plan applied simultaneously
#   Year 3 (adjustment): feeder frequency tuning based on observed transfers
# Between events the bus network is frozen.

# ---------------------------------------------------------------------------
# Discrete service phases (replacing continuous pressure optimization)
# Modeled after real transit agency restructuring timelines:
#   - IndyGo Red Line: Day 1 restructure, Year 2-3 adjustments, Year 5+ mature
#   - Houston METRO: Opening day + 3-year tuning + long-range plan
# Most agencies do major service reviews every 5-7 years, so the network
# is redesigned at years 0, 3, 8, 15, and 20 (max gap = 7 years).
# Years 15 and 20 respond to TOD maturation and long-range planning.
# ---------------------------------------------------------------------------

# Phase transition years (years since APM opening)
PHASE_TRANSITION_YEARS = {
    "opening_day": 0,         # Day-1 restructure: truncate parallels, create feeders
    "early_operations": 3,    # Tune feeders based on observed demand, cut underperformers
    "mature_network": 8,      # Full hub-and-spoke, timed transfers, maximum reallocation
    "second_generation": 15,  # Respond to TOD maturation, new feeder routes to developed areas
    "long_range_plan": 20,    # Final optimization for long-range horizon
}

# Per-phase configuration parameters
PHASE_CONFIGS = {
    "pre_apm": {
        "label": "pre_apm",
        "parallel_action": "retain",       # no changes to existing routes
        "feeder_headway_target": None,      # N/A
        "truncation_fraction": None,        # N/A
        "frequency_realloc_share": 0.0,     # no hour reallocation
        "feeder_improvement_pct": 0.0,
        "cut_underperformers": False,
        "timed_transfers": False,
    },
    "opening_day": {
        "label": "opening_day",
        "parallel_action": "truncate",      # truncate at APM stations
        "feeder_headway_target": 15.0,      # 15-min initial feeder target
        "truncation_fraction": 0.55,
        "frequency_realloc_share": FREQUENCY_ALLOCATION_SHARE,
        "feeder_improvement_pct": 0.0,      # baseline headways
        "cut_underperformers": False,        # too early to judge
        "timed_transfers": False,
    },
    "early_operations": {
        "label": "early_operations",
        "parallel_action": "truncate",
        "feeder_headway_target": 12.0,      # tightened from 15 to 12 min
        "truncation_fraction": 0.55,
        "frequency_realloc_share": FREQUENCY_ALLOCATION_SHARE,
        "feeder_improvement_pct": 10.0,     # 10% headway improvement on feeders
        "cut_underperformers": True,         # enough data to judge performance
        "timed_transfers": True,             # pulse scheduling where possible
    },
    "mature_network": {
        "label": "mature_network",
        "parallel_action": "eliminate",      # full hub-and-spoke
        "feeder_headway_target": 10.0,       # 10-min feeder target
        "truncation_fraction": 0.55,
        "frequency_realloc_share": 0.90,     # max reallocation
        "feeder_improvement_pct": 20.0,      # 20% headway improvement
        "cut_underperformers": True,
        "timed_transfers": True,
    },
    "second_generation": {
        "label": "second_generation",
        "parallel_action": "eliminate",      # hub-and-spoke maintained
        "feeder_headway_target": 8.0,        # 8-min target — TOD density supports higher freq
        "truncation_fraction": 0.55,
        "frequency_realloc_share": 0.92,     # slight increase from mature
        "feeder_improvement_pct": 25.0,      # 25% improvement — new routes to TOD areas
        "cut_underperformers": True,
        "timed_transfers": True,
    },
    "long_range_plan": {
        "label": "long_range_plan",
        "parallel_action": "eliminate",      # fully optimized network
        "feeder_headway_target": 8.0,        # maintain 8-min target
        "truncation_fraction": 0.55,
        "frequency_realloc_share": 0.95,     # maximum reallocation
        "feeder_improvement_pct": 30.0,      # 30% improvement — full TOD maturity
        "cut_underperformers": True,
        "timed_transfers": True,
    },
}


def get_service_phase(year: int, apm_opening_year: int = 0) -> str:
    """Determine the discrete service phase for a given year.

    Returns one of: "pre_apm", "opening_day", "early_operations",
    "mature_network", "second_generation", "long_range_plan"
    """
    years_since = year - apm_opening_year
    if years_since < PHASE_TRANSITION_YEARS["opening_day"]:
        return "pre_apm"
    elif years_since < PHASE_TRANSITION_YEARS["early_operations"]:
        return "opening_day"
    elif years_since < PHASE_TRANSITION_YEARS["mature_network"]:
        return "early_operations"
    elif years_since < PHASE_TRANSITION_YEARS["second_generation"]:
        return "mature_network"
    elif years_since < PHASE_TRANSITION_YEARS["long_range_plan"]:
        return "second_generation"
    else:
        return "long_range_plan"


def is_phase_transition_year(year: int, apm_opening_year: int = 0) -> bool:
    """Check if this year triggers a bus network phase transition.

    Returns True when ``year - apm_opening_year`` equals one of the
    transition thresholds (0 = opening day, 3 = early operations,
    8 = mature network, 15 = second generation, 20 = long-range plan).
    The feedback loop uses this to decide when to unfreeze the bus
    network and run a full redesign.

    Note: this fires at the *start* of the transition year, so phase N's
    configuration applies from year N onward until the next transition.
    """
    years_since = year - apm_opening_year
    return years_since in PHASE_TRANSITION_YEARS.values()


def get_elimination_fraction(year: int, apm_opening_year: int = 0) -> float:
    """Continuous parallel-route elimination ramp.

    Returns a value in [0, 1] that controls the blend between truncation
    (early phases) and full elimination (mature phases).  Ramps linearly
    from 0.0 to 1.0 over years 8-12, preventing a single event year from
    producing a large discrete jump in feeder coverage.

    - Before year 8: 0.0 (truncate only, per early_operations config).
    - Years 8-12: linear ramp 0.0 -> 1.0.
    - After year 12: 1.0 (full elimination, per mature_network config).
    """
    y = year - apm_opening_year
    if y < 8:
        return 0.0
    if y >= 12:
        return 1.0
    return (y - 8) / 4.0


# Route truncation: when a parallel route is truncated at an APM station,
# the retained segment keeps its original headway (or better, since the
# shortened cycle time frees vehicles).
# Typical retained fraction of route length after truncation:
DEFAULT_TRUNCATION_RETAINED_FRACTION = 0.55  # 55% of route retained on average


# ---------------------------------------------------------------------------
# Route Service Plan (budget-constrained optimization)
# ---------------------------------------------------------------------------

class RouteServicePlan:
    """Budget-constrained per-route headway optimization.

    Given a corridor's APM ridership and the current bus network, adjusts
    per-route headways to:
    1. Degrade parallel routes (push riders to APM)
    2. Enhance feeder routes (expand APM catchment)
    3. Hold independent routes constant
    4. Stay within operating budget

    The optimization uses a simple priority-based allocation:
    - Feeder routes get headway improvements first (higher priority)
    - Parallel routes get headway degradation (service reduction saves $)
    - Savings from parallel reductions fund feeder improvements
    - APM is funded independently (TIF + fares); bus budget is bus-only
    - CityBus always fully utilises its budget
    """

    def __init__(
        self,
        routes: List[BusRoute],
        cost_model: BusOperatingCostModel,
        apm_service: Optional[APMService] = None,
    ):
        self.routes = routes
        self.cost_model = cost_model
        self.apm_service = apm_service

    def optimize_headways(
        self,
        restructure_pressure: float = 0.0,
        min_headway: float = 10.0,
        max_headway: float = 120.0,
        max_parallel_degradation: float = 3.0,
        max_feeder_improvement: float = 0.5,
        apm_daily_riders: float = 0.0,
    ) -> Dict[str, object]:
        """Adjust per-route headways based on restructure pressure.

        Parameters
        ----------
        restructure_pressure : 0-1 scalar from corridor ridership maturity
        min_headway : minimum allowed headway (minutes)
        max_headway : maximum allowed headway (minutes)
        max_parallel_degradation : max multiplier on parallel route headway (e.g. 3x)
        max_feeder_improvement : min multiplier on feeder route headway (e.g. 0.5x)
        apm_daily_riders : current APM daily ridership (for minimum threshold check)

        Returns
        -------
        dict with per-route headways, costs, and budget diagnostics
        """
        pressure = float(np.clip(restructure_pressure, 0.0, 1.0))

        # Below minimum APM ridership → no bus restructuring justified
        if apm_daily_riders < MIN_APM_RIDERSHIP_FOR_RESTRUCTURE:
            pressure = 0.0

        budget = self.cost_model.annual_budget  # bus-only budget

        # Snapshot baseline costs at current (pre-restructure) headways
        baseline_bus_cost = sum(
            self.cost_model.route_annual_cost(r) for r in self.routes
        )
        baseline_parallel_cost = sum(
            self.cost_model.route_annual_cost(r)
            for r in self.routes
            if r.classification == "parallel"
        )

        # --- Step 1: Degrade parallel routes & hold independents ---
        for route in self.routes:
            if route.classification == "parallel":
                multiplier = 1.0 + pressure * (max_parallel_degradation - 1.0)
                route.current_headway_min = min(
                    route.baseline_headway_min * multiplier, max_headway
                )
            elif route.classification == "feeder":
                # Initial feeder improvement (will be adjusted in step 2)
                multiplier = 1.0 - pressure * (1.0 - max_feeder_improvement)
                route.current_headway_min = max(
                    route.baseline_headway_min * multiplier, min_headway
                )
            else:
                route.current_headway_min = route.baseline_headway_min

        # --- Step 2: Reinvest parallel savings into feeders (and APM offset) ---
        # Savings freed by degrading parallel routes are redistributed.
        # In combined/expanded modes, a fraction offsets APM O&M (same fiscal
        # entity); the rest improves feeder frequency.  In separate mode,
        # all savings go to feeders (APM is independently funded).
        new_parallel_cost = sum(
            self.cost_model.route_annual_cost(r)
            for r in self.routes
            if r.classification == "parallel"
        )
        parallel_savings = max(baseline_parallel_cost - new_parallel_cost, 0.0)
        feeder_routes = [r for r in self.routes if r.classification == "feeder"]

        # Split savings by budget mode
        _budget_mode = getattr(self.cost_model, "budget_mode", BUDGET_MODE_SEPARATE)
        if _budget_mode in (BUDGET_MODE_COMBINED, BUDGET_MODE_EXPANDED):
            apm_om_offset = parallel_savings * BUS_SAVINGS_APM_OFFSET_FRACTION
            feeder_reinvestment = parallel_savings - apm_om_offset
        else:
            apm_om_offset = 0.0
            feeder_reinvestment = parallel_savings

        if feeder_routes and feeder_reinvestment > 0:
            current_feeder_cost = sum(
                self.cost_model.route_annual_cost(r) for r in feeder_routes
            )
            if current_feeder_cost > 0:
                # Reinvest feeder portion: scale feeder service up
                target_feeder_cost = current_feeder_cost + feeder_reinvestment
                scale = target_feeder_cost / current_feeder_cost
                for r in feeder_routes:
                    r.current_headway_min = max(
                        r.current_headway_min / scale, min_headway
                    )

        # Budget enforcement: if total bus cost exceeds budget, scale back feeders
        new_bus_cost_check = sum(
            self.cost_model.route_annual_cost(r) for r in self.routes
        )
        if new_bus_cost_check > budget and budget > 0 and feeder_routes:
            overshoot = new_bus_cost_check - budget
            current_feeder_cost = sum(
                self.cost_model.route_annual_cost(r) for r in feeder_routes
            )
            if current_feeder_cost > overshoot:
                # scale_back < 1.0 means keep that fraction of feeder cost.
                # Dividing headway by scale_back (<1) increases headway,
                # reducing frequency and cost. E.g., 0.7 → headway ×1.43 → cost -30%.
                scale_back = (current_feeder_cost - overshoot) / current_feeder_cost
                if scale_back > 1e-6:
                    for r in feeder_routes:
                        r.current_headway_min = min(
                            r.current_headway_min / scale_back, max_headway
                        )
                else:
                    # Savings insufficient — push feeders to max headway
                    for r in feeder_routes:
                        r.current_headway_min = max_headway

        # --- Build results ---
        new_bus_cost = sum(self.cost_model.route_annual_cost(r) for r in self.routes)

        route_results = []
        for route in self.routes:
            route_results.append({
                "route_id": route.route_id,
                "name": route.name,
                "classification": route.classification,
                "baseline_headway": route.baseline_headway_min,
                "current_headway": route.current_headway_min,
                "headway_change_pct": (
                    (route.current_headway_min / route.baseline_headway_min - 1.0)
                    * 100 if route.baseline_headway_min > 0 else 0.0
                ),
                "vehicles_needed": route.vehicles_needed,
                "annual_cost": self.cost_model.route_annual_cost(route),
            })

        return {
            "restructure_pressure": pressure,
            "baseline_bus_cost": baseline_bus_cost,
            "new_bus_cost": new_bus_cost,
            "total_cost": new_bus_cost,
            "bus_savings_from_parallel": parallel_savings,
            "feeder_reinvestment": feeder_reinvestment,
            "apm_om_offset_from_bus": apm_om_offset,
            "budget": budget,
            "budget_surplus": budget - new_bus_cost,
            "budget_utilization": new_bus_cost / max(budget, 1.0),
            "routes": route_results,
        }

    def optimize_headways_proactive(
        self,
        year: int = 0,
        min_headway: float = 10.0,
        max_headway: float = 120.0,
        apm_daily_riders: float = 0.0,
        bus_stops_by_route: Optional[Dict[str, np.ndarray]] = None,
        parcel_xy: Optional[np.ndarray] = None,
        parcel_se01_share: Optional[np.ndarray] = None,
        metro_se01_share: float = 0.0,
    ) -> Dict[str, object]:
        """Proactive restructuring using the decision engine.

        Replaces the reactive pressure-ramp with explicit per-route
        cut/keep/reroute/enhance decisions based on productivity ranking,
        phased by year, and guarded by Title VI equity checks.

        Falls back to the legacy ``optimize_headways()`` when APM ridership
        is below the restructuring threshold.
        """
        if apm_daily_riders < MIN_APM_RIDERSHIP_FOR_RESTRUCTURE:
            return self.optimize_headways(
                restructure_pressure=0.0,
                min_headway=min_headway,
                max_headway=max_headway,
                apm_daily_riders=apm_daily_riders,
            )

        budget = self.cost_model.annual_budget

        # Snapshot baseline costs
        baseline_bus_cost = sum(
            self.cost_model.route_annual_cost(r) for r in self.routes
        )

        # --- Step 1: Decide per-route actions ---
        decisions = decide_route_restructuring(self.routes, year=year)

        # --- Step 2: Equity guard ---
        if (
            bus_stops_by_route is not None
            and parcel_xy is not None
            and parcel_se01_share is not None
            and metro_se01_share > 0
        ):
            decisions = check_coverage_equity(
                decisions, self.routes, bus_stops_by_route,
                parcel_xy, parcel_se01_share, metro_se01_share,
            )

        # --- Step 3: Capture baseline headways, then apply decisions ---
        # Save baseline headways BEFORE mutation so savings computation is explicit
        _baseline_hw = {r.route_id: r.baseline_headway_min for r in self.routes}

        target_headways = apply_restructuring_decisions(
            self.routes, decisions, min_headway, max_headway,
        )
        _eliminated_ids = set()
        for route in self.routes:
            hw = target_headways.get(route.route_id, route.baseline_headway_min)
            if hw <= 0:
                # Route eliminated — headway=0 makes daily_vehicle_hours=0
                # via the property guard (current_headway_min <= 0 → 0.0 veh-hrs)
                route.current_headway_min = 0.0
                _eliminated_ids.add(route.route_id)
            else:
                route.current_headway_min = hw

        # --- Step 4: Budget-constrained feeder enhancement ---
        # Compute savings from eliminations and reductions
        eliminated_routes = [
            r for r in self.routes
            if decisions.get(r.route_id) == RestructuringAction.ELIMINATE
        ]
        reduced_routes = [
            r for r in self.routes
            if decisions.get(r.route_id) == RestructuringAction.REDUCE
        ]
        feeder_routes = [
            r for r in self.routes
            if decisions.get(r.route_id) == RestructuringAction.ENHANCE
        ]

        # Savings = baseline cost of eliminated + (baseline - reduced cost) of reduced
        # Uses explicitly captured _baseline_hw to avoid relying on mutation side effects
        savings = 0.0
        for r in eliminated_routes:
            # Full baseline cost is freed
            cur_hw = r.current_headway_min
            r.current_headway_min = _baseline_hw[r.route_id]
            savings += self.cost_model.route_annual_cost(r)
            r.current_headway_min = cur_hw  # restore to eliminated state (0.0)
        for r in reduced_routes:
            cur_hw = r.current_headway_min  # the new reduced headway from Step 3
            r.current_headway_min = _baseline_hw[r.route_id]
            baseline_cost = self.cost_model.route_annual_cost(r)
            r.current_headway_min = cur_hw  # restore to reduced headway
            new_cost = self.cost_model.route_annual_cost(r)
            savings += max(baseline_cost - new_cost, 0.0)

        # Split savings by budget mode
        _budget_mode = getattr(self.cost_model, "budget_mode", BUDGET_MODE_SEPARATE)
        if _budget_mode in (BUDGET_MODE_COMBINED, BUDGET_MODE_EXPANDED):
            apm_om_offset = savings * BUS_SAVINGS_APM_OFFSET_FRACTION
            feeder_pool = savings - apm_om_offset
        else:
            apm_om_offset = 0.0
            feeder_pool = savings

        # Allocate feeder pool to enhancements (greedy by productivity)
        if feeder_routes and feeder_pool > 0:
            # Sort feeders by productivity descending (best feeders first)
            feeders_ranked = sorted(
                feeder_routes,
                key=lambda r: route_productivity_score(r),
                reverse=True,
            )
            remaining_pool = feeder_pool
            for r in feeders_ranked:
                if remaining_pool <= 0:
                    # No more budget — revert to baseline headway
                    r.current_headway_min = r.baseline_headway_min
                    continue
                # Cost of enhanced service - cost of baseline service
                baseline_hw = r.current_headway_min
                r.current_headway_min = r.baseline_headway_min
                baseline_cost = self.cost_model.route_annual_cost(r)
                r.current_headway_min = max(r.baseline_headway_min * 0.5, min_headway)
                enhanced_cost = self.cost_model.route_annual_cost(r)
                extra = enhanced_cost - baseline_cost
                if extra <= remaining_pool:
                    remaining_pool -= extra
                    # keep the enhanced headway
                else:
                    # Partial enhancement: scale between baseline and enhanced
                    if extra > 0:
                        frac = remaining_pool / extra
                        target = r.baseline_headway_min * (1.0 - 0.5 * frac)
                        r.current_headway_min = max(target, min_headway)
                    else:
                        r.current_headway_min = r.baseline_headway_min
                    remaining_pool = 0.0

        # Ensure eliminated routes have zero service (headway=0 → 0 veh-hrs via property)
        for r in eliminated_routes:
            r.current_headway_min = 0.0

        # Budget enforcement
        active_routes = [
            r for r in self.routes
            if decisions.get(r.route_id) != RestructuringAction.ELIMINATE
        ]
        new_bus_cost = sum(
            self.cost_model.route_annual_cost(r) for r in active_routes
        )
        if new_bus_cost > budget and budget > 0 and feeder_routes:
            overshoot = new_bus_cost - budget
            current_feeder_cost = sum(
                self.cost_model.route_annual_cost(r) for r in feeder_routes
            )
            if current_feeder_cost > overshoot:
                # scale_back < 1.0 means keep that fraction of feeder cost.
                # Dividing headway by scale_back (<1) increases headway,
                # reducing frequency and cost. E.g., 0.7 → headway ×1.43 → cost -30%.
                scale_back = (current_feeder_cost - overshoot) / current_feeder_cost
                if scale_back > 1e-6:
                    for r in feeder_routes:
                        r.current_headway_min = min(
                            r.current_headway_min / scale_back, max_headway
                        )

        # --- Build results ---
        new_bus_cost = sum(
            self.cost_model.route_annual_cost(r) for r in active_routes
        )
        route_results = []
        for route in self.routes:
            action = decisions.get(route.route_id, RestructuringAction.KEEP)
            route_results.append({
                "route_id": route.route_id,
                "name": route.name,
                "classification": route.classification,
                "action": action.value,
                "productivity": round(route_productivity_score(route), 2),
                "baseline_headway": route.baseline_headway_min,
                "current_headway": route.current_headway_min,
                "headway_change_pct": (
                    (route.current_headway_min / route.baseline_headway_min - 1.0)
                    * 100 if route.baseline_headway_min > 0 else 0.0
                ),
                "vehicles_needed": route.vehicles_needed if action != RestructuringAction.ELIMINATE else 0,
                "annual_cost": (
                    self.cost_model.route_annual_cost(route)
                    if action != RestructuringAction.ELIMINATE else 0.0
                ),
            })

        return {
            "year": year,
            "restructure_mode": "proactive",
            "baseline_bus_cost": baseline_bus_cost,
            "new_bus_cost": new_bus_cost,
            "total_cost": new_bus_cost,
            "bus_savings_total": savings,
            "feeder_reinvestment": feeder_pool,
            "apm_om_offset_from_bus": apm_om_offset,
            "routes_eliminated": len(eliminated_routes),
            "routes_reduced": len(reduced_routes),
            "routes_enhanced": len(feeder_routes),
            "budget": budget,
            "budget_surplus": budget - new_bus_cost,
            "budget_utilization": new_bus_cost / max(budget, 1.0),
            "routes": route_results,
        }

    def get_feeder_headway(self) -> float:
        """Average feeder headway across active feeder-classified routes."""
        feeders = [
            r for r in self.routes
            if r.classification == "feeder" and r.current_headway_min > 0
        ]
        if not feeders:
            return 30.0
        return float(np.mean([r.current_headway_min for r in feeders]))

    def get_parallel_headway(self) -> float:
        """Average parallel headway across active parallel-classified routes."""
        parallels = [
            r for r in self.routes
            if r.classification == "parallel" and r.current_headway_min > 0
        ]
        if not parallels:
            return 30.0
        return float(np.mean([r.current_headway_min for r in parallels]))

    def get_route_headways_dict(self) -> Dict[str, float]:
        """Return dict of route_id → current_headway_min."""
        return {r.route_id: r.current_headway_min for r in self.routes}


# ---------------------------------------------------------------------------
# Complete Network Redesign Strategy
# ---------------------------------------------------------------------------

class NetworkRedesignStrategy:
    """Complete bus network redesign for APM corridor integration.

    Unlike incremental ``RouteServicePlan`` which adjusts headways on existing
    routes, this strategy models a complete network restructure:

    1. Eliminate parallel routes that duplicate APM service
    2. Convert viable parallel routes to feeders
    3. Integrate synthetic feeder routes (arterial-spine algorithm)
    4. Reallocate freed service hours to improve frequency on retained routes
    5. Phase in the redesign over ``REDESIGN_PHASE_IN_YEARS``

    Produces the same output interface as ``build_corridor_bus_network`` so the
    feedback loop consumes it without changes.
    """

    def __init__(
        self,
        routes: List[BusRoute],
        cost_model: BusOperatingCostModel,
        apm_service: Optional[APMService] = None,
        synthetic_feeders: Optional[List[BusRoute]] = None,
        synthetic_feeder_stops: Optional[Dict[str, np.ndarray]] = None,
        bus_stops_by_route: Optional[Dict[str, np.ndarray]] = None,
        corridor_stops_xy: Optional[np.ndarray] = None,
        parcel_xy: Optional[np.ndarray] = None,
        parcel_pop: Optional[np.ndarray] = None,
        parking_costs: Optional[np.ndarray] = None,
        institutional_weights: Optional[np.ndarray] = None,
    ):
        self.routes = routes
        self.cost_model = cost_model
        self.apm_service = apm_service
        self.synthetic_feeders = synthetic_feeders or []
        self.synthetic_feeder_stops = synthetic_feeder_stops or {}
        self.bus_stops_by_route = bus_stops_by_route or {}
        self.corridor_stops_xy = corridor_stops_xy
        self.parcel_xy = parcel_xy
        self.parcel_pop = parcel_pop
        self.parking_costs = parking_costs
        self.institutional_weights = institutional_weights
        # Pre-compute estimated ridership (MNL-based) when parcel data available
        self._estimated_ridership: Dict[str, float] = {}
        if parcel_xy is not None and parcel_pop is not None:
            self._estimated_ridership = estimate_all_route_ridership(
                routes, bus_stops_by_route or {}, parcel_xy, parcel_pop,
                parking_costs=parking_costs,
                institutional_weights=institutional_weights,
            )

    # ---- helpers -----------------------------------------------------------

    def _route_overlap_fraction(self, route: BusRoute) -> float:
        """Fraction of a route's stops within CORRIDOR_BUFFER_M of corridor line."""
        if self.corridor_stops_xy is None or len(self.corridor_stops_xy) < 2:
            return 0.0
        rid = route.route_id
        if rid not in self.bus_stops_by_route:
            return 0.0
        bus_xy = self.bus_stops_by_route[rid]
        if len(bus_xy) == 0:
            return 0.0
        from scipy.spatial import cKDTree
        corridor_line_pts = _interpolate_corridor_points(self.corridor_stops_xy)
        tree = cKDTree(corridor_line_pts)
        dists, _ = tree.query(bus_xy, k=1)
        return float(np.sum(dists <= CORRIDOR_BUFFER_M)) / len(bus_xy)

    def _geometric_retained_fraction(self, route: BusRoute) -> Optional[float]:
        """Compute retained fraction from geometric split at nearest APM station.

        Returns the fraction of stops in the longer segment (the feeder
        portion), or None if the route doesn't pass near an APM station.
        """
        if self.corridor_stops_xy is None or len(self.corridor_stops_xy) < 1:
            return None
        bus_xy = self.bus_stops_by_route.get(str(route.route_id), np.empty((0, 2)))
        if len(bus_xy) < 3:
            return None
        from scipy.spatial import cKDTree
        apm_tree = cKDTree(self.corridor_stops_xy)
        dists, _ = apm_tree.query(bus_xy, k=1)
        nearest_stop = int(np.argmin(dists))
        if float(dists[nearest_stop]) > 200.0:  # station_buffer_m
            return None
        n_stops = len(bus_xy)
        left_len = nearest_stop
        right_len = n_stops - nearest_stop - 1
        if left_len >= right_len and left_len >= 2:
            retained_stops = nearest_stop + 1
        elif right_len >= 2:
            retained_stops = right_len + 1
        else:
            return None
        return retained_stops / n_stops

    def _route_productivity(self, route: BusRoute) -> float:
        """Riders per vehicle-hour.

        Prefers estimated ridership (from gravity model) when parcel data
        was provided at init time, falling back to static GTFS
        ``observed_daily_riders`` otherwise.  Estimated ridership responds
        to headway changes (sqrt elasticity) and local population, giving
        more realistic disposition decisions for routes whose GTFS
        ridership field is zero or stale.
        """
        dvh = route.daily_vehicle_hours
        if dvh <= 0:
            return 0.0
        riders = self._estimated_ridership.get(route.route_id, route.observed_daily_riders)
        return riders / dvh

    def _compute_route_disposition(
        self,
        route: BusRoute,
        overlap: float,
        productivity: float,
        median_productivity: float,
        apm_daily_riders: float = 0.0,
    ) -> str:
        """Classify a route's fate using a decision matrix.

        Based on IndyGo Red Line, Houston METRO New Network, LA Metro NextGen:
        no agency operates a 90-min parallel route alongside a 5-min APM line.
        Routes are either kept, truncated, eliminated, or converted to feeder.

        Decision matrix:
        ┌──────────┬─────────────────┬───────────────┬──────────────────┐
        │ Overlap  │ Productivity    │ APM Ridership │ Decision         │
        ├──────────┼─────────────────┼───────────────┼──────────────────┤
        │ >50%     │ Below median    │ Any           │ eliminate        │
        │ >50%     │ Above median    │ >1000/day     │ truncate         │
        │ >50%     │ Above median    │ <1000/day     │ retain_enhanced  │
        │ 15-50%   │ Any             │ >1000/day     │ convert_to_feeder│
        │ 15-50%   │ Any             │ <1000/day     │ retain_enhanced  │
        │ <15%     │ Any             │ Any           │ no_change        │
        └──────────┴─────────────────┴───────────────┴──────────────────┘
        """
        apm_strong = apm_daily_riders >= MIN_APM_RIDERSHIP_FOR_RESTRUCTURE

        if overlap >= ELIMINATION_OVERLAP_THRESHOLD:
            if productivity < median_productivity or productivity < ELIMINATION_PRODUCTIVITY_FLOOR:
                return "eliminate"
            elif apm_strong:
                return "truncate"
            else:
                # APM too weak to justify restructuring a productive parallel route
                return "retain_enhanced"
        elif overlap >= FEEDER_OVERLAP_THRESHOLD:
            if apm_strong:
                return "convert_to_feeder"
            else:
                return "retain_enhanced"
        else:
            # Independent route — no change regardless of APM ridership
            if productivity >= ELIMINATION_PRODUCTIVITY_FLOOR:
                return "retain_enhanced"
            else:
                return "retain_coverage"

    # ---- main method -------------------------------------------------------

    def redesign_network(
        self,
        restructure_pressure: float = 0.0,
        year: int = 0,
        apm_opening_year: int = 0,
        min_headway: float = 10.0,
        max_headway: float = 120.0,
    ) -> Dict[str, object]:
        """Compute the redesigned network state using discrete service phases.

        Models bus restructuring as phased events matching real agency practice
        (IndyGo Red Line / Houston METRO pattern):

        - **Pre-APM**: No changes to existing routes.
        - **Opening Day** (year 0): Parallel routes truncated at APM stations,
          feeders created from truncated segments, hours reallocated.
        - **Early Operations** (year 3): Feeders tightened to 12 min,
          underperformers cut, timed transfers where headways allow.
        - **Mature Network** (year 8+): Full hub-and-spoke redesign, maximum
          hour reallocation, 10-min feeder target.

        Between phase transitions the bus network is frozen. APM headway
        always updates (demand-responsive).

        Parameters
        ----------
        restructure_pressure : 0-1 scalar (retained for interface compat).
        year : current simulation year.
        apm_opening_year : year APM service began.
        min_headway : minimum allowed headway (minutes).
        max_headway : maximum allowed headway (minutes).
        """
        pressure = float(np.clip(restructure_pressure, 0.0, 1.0))
        budget = self.cost_model.annual_budget
        apm_riders = self.apm_service.daily_riders if self.apm_service else 0.0

        # --- Determine current phase ---
        phase_name = get_service_phase(year, apm_opening_year)
        phase_cfg = PHASE_CONFIGS[phase_name]
        is_active = phase_name != "pre_apm"

        # Below ridership threshold -> no restructuring at all
        if apm_riders < MIN_APM_RIDERSHIP_FOR_RESTRUCTURE:
            is_active = False
            phase_name = "pre_apm"
            phase_cfg = PHASE_CONFIGS["pre_apm"]

        # --- Step 1: Route disposition (decision matrix) ---
        overlaps = {r.route_id: self._route_overlap_fraction(r) for r in self.routes}
        productivities = {r.route_id: self._route_productivity(r) for r in self.routes}
        prod_values = [p for p in productivities.values() if p > 0]
        median_prod = float(np.median(prod_values)) if prod_values else ELIMINATION_PRODUCTIVITY_FLOOR

        dispositions: Dict[str, str] = {}
        for route in self.routes:
            dispositions[route.route_id] = self._compute_route_disposition(
                route,
                overlaps[route.route_id],
                productivities[route.route_id],
                median_prod,
                apm_daily_riders=apm_riders,
            )

        # --- Step 2: Apply dispositions based on phase config ---
        # Save original cycle times so mutations don't compound across calls.
        _original_cycle_times = {r.route_id: r.cycle_time_min for r in self.routes}

        baseline_total_hours = sum(r.annual_vehicle_hours for r in self.routes)
        freed_hours = 0.0

        for route in self.routes:
            disp = dispositions[route.route_id]
            # Always reset cycle_time to the unmodified original before applying
            # any retained-fraction reduction, preventing compounding.
            route.cycle_time_min = _original_cycle_times[route.route_id]

            if not is_active:
                # Pre-APM or below threshold: all routes at baseline
                route.current_headway_min = route.baseline_headway_min
                continue

            if disp == "eliminate":
                # Continuous elimination ramp (Fix 2): instead of a binary
                # truncate/eliminate switch at mature_network, ramp from
                # 0.0 to 1.0 over years 8-12.  Routes with the highest
                # corridor overlap are eliminated first.
                _elim_frac = get_elimination_fraction(year, apm_opening_year)
                _route_overlap = overlaps.get(route.route_id, 0.0)
                if _elim_frac >= 1.0 or _route_overlap > (1.0 - _elim_frac):
                    # Fully eliminate this route
                    freed_hours += route.annual_vehicle_hours
                    route.current_headway_min = float("inf")
                else:
                    # Not yet eliminated — truncate instead.
                    default_frac = phase_cfg["truncation_fraction"] or DEFAULT_TRUNCATION_RETAINED_FRACTION
                    geo_frac = self._geometric_retained_fraction(route)
                    retained_frac = (
                        min(geo_frac, default_frac)
                        if geo_frac is not None
                        else default_frac
                    )
                    original_hours = route.annual_vehicle_hours
                    route.cycle_time_min = _original_cycle_times[route.route_id] * retained_frac
                    new_hw = route.baseline_headway_min * retained_frac
                    route.current_headway_min = max(new_hw, min_headway)
                    route.classification = "feeder"
                    freed_hours += max(original_hours - route.annual_vehicle_hours, 0)

            elif disp == "truncate":
                default_frac = phase_cfg["truncation_fraction"] or DEFAULT_TRUNCATION_RETAINED_FRACTION
                geo_frac = self._geometric_retained_fraction(route)
                retained_frac = (
                    min(geo_frac, default_frac)
                    if geo_frac is not None
                    else default_frac
                )
                original_hours = route.annual_vehicle_hours
                route.cycle_time_min = _original_cycle_times[route.route_id] * retained_frac
                new_hw = route.baseline_headway_min * retained_frac
                route.current_headway_min = max(new_hw, min_headway)
                route.classification = "feeder"
                freed_hours += max(original_hours - route.annual_vehicle_hours, 0)

            elif disp == "convert_to_feeder":
                route.classification = "feeder"
                retained_frac = 0.70
                original_hours = route.annual_vehicle_hours
                route.cycle_time_min = _original_cycle_times[route.route_id] * retained_frac
                route.current_headway_min = max(
                    route.baseline_headway_min * 0.90, min_headway
                )
                freed_hours += max(original_hours - route.annual_vehicle_hours, 0)

            elif disp == "retain_enhanced":
                route.current_headway_min = route.baseline_headway_min

            elif disp == "retain_coverage":
                route.current_headway_min = route.baseline_headway_min

            else:
                route.current_headway_min = route.baseline_headway_min

        # --- Step 3: Frequency reallocation from freed hours ---
        # Continuous ramp (Fix 2): interpolate realloc share between
        # early_operations (0.75) and mature_network (0.90) values
        # using the same elimination fraction for a smooth transition.
        _elim_f = get_elimination_fraction(year, apm_opening_year)
        _early_share = PHASE_CONFIGS["early_operations"]["frequency_realloc_share"]
        _mature_share = phase_cfg["frequency_realloc_share"]
        if _elim_f > 0.0 and _elim_f < 1.0 and _mature_share > _early_share:
            realloc_share = _early_share + (_mature_share - _early_share) * _elim_f
        else:
            realloc_share = phase_cfg["frequency_realloc_share"]
        frequency_hours = freed_hours * realloc_share
        enhanced_routes = [
            r for r in self.routes
            if dispositions[r.route_id] in ("retain_enhanced", "retain_coverage")
            and r.current_headway_min < float("inf")
        ]

        if enhanced_routes and frequency_hours > 0 and is_active:
            total_riders = sum(r.observed_daily_riders for r in enhanced_routes)
            if total_riders <= 0:
                total_riders = float(len(enhanced_routes))

            for route in enhanced_routes:
                rider_share = (
                    route.observed_daily_riders / total_riders
                    if total_riders > 0
                    else 1.0 / len(enhanced_routes)
                )
                bonus_hours = frequency_hours * rider_share
                current_annual = route.annual_vehicle_hours
                if current_annual > 0:
                    improvement_factor = current_annual / (current_annual + bonus_hours)
                    new_hw = route.baseline_headway_min * improvement_factor
                    route.current_headway_min = max(new_hw, min_headway)

        # --- Step 3b: Phase-specific feeder improvements ---
        # Continuous ramp (Fix 2): interpolate feeder improvement %
        # between early_operations and mature_network during transition.
        _early_imp = PHASE_CONFIGS["early_operations"]["feeder_improvement_pct"]
        _cfg_imp = phase_cfg["feeder_improvement_pct"]
        if _elim_f > 0.0 and _elim_f < 1.0 and _cfg_imp > _early_imp:
            feeder_improve_pct = _early_imp + (_cfg_imp - _early_imp) * _elim_f
        else:
            feeder_improve_pct = _cfg_imp
        if is_active and feeder_improve_pct > 0:
            feeder_routes = [
                r for r in self.routes
                if r.classification == "feeder" and r.current_headway_min < float("inf")
            ]
            improve_factor = 1.0 - feeder_improve_pct / 100.0
            for r in feeder_routes:
                r.current_headway_min = max(r.current_headway_min * improve_factor, min_headway)

        # --- Step 3c: Cut underperforming routes (early ops + mature) ---
        if is_active and phase_cfg["cut_underperformers"]:
            # Routes with very low productivity in feeder role get cut
            for route in self.routes:
                if (route.classification == "feeder"
                        and route.current_headway_min < float("inf")
                        and productivities.get(route.route_id, 0) < ELIMINATION_PRODUCTIVITY_FLOOR * 0.5):
                    freed_hours += route.annual_vehicle_hours * 0.5  # partial recovery
                    route.current_headway_min = float("inf")

        # --- Step 4: Synthetic feeder integration ---
        coverage_share = 1.0 - realloc_share if realloc_share < 1.0 else COVERAGE_ALLOCATION_SHARE
        feeder_hours_budget = freed_hours * coverage_share
        active_synthetic = []

        if self.synthetic_feeders and is_active:
            for sf in self.synthetic_feeders:
                sf_cost = self.cost_model.route_annual_cost(sf)
                if sf_cost <= 0:
                    continue
                sf_hours = sf.annual_vehicle_hours
                if feeder_hours_budget >= sf_hours * 0.5:
                    sf.current_headway_min = max(sf.baseline_headway_min, min_headway)
                    feeder_hours_budget -= sf.annual_vehicle_hours
                    active_synthetic.append(sf)
                else:
                    break

        all_routes = list(self.routes) + active_synthetic

        # --- Step 5: Coverage loss (diagnostic) ---
        coverage_loss = self._compute_coverage_loss(
            dispositions, 1.0 if is_active else 0.0
        )

        # --- Budget enforcement ---
        total_cost = sum(self.cost_model.route_annual_cost(r) for r in all_routes
                         if r.current_headway_min < float("inf"))
        if budget > 0 and total_cost > budget:
            overshoot = total_cost - budget
            enhanced_cost = sum(
                self.cost_model.route_annual_cost(r) for r in enhanced_routes
            )
            if enhanced_cost > 0 and overshoot < enhanced_cost:
                scale_back = max((enhanced_cost - overshoot) / enhanced_cost, 0.10)
                for r in enhanced_routes:
                    # Divide by scale_back < 1 → increases headway (saves money).
                    # Clip at 120 min (max_headway), not baseline — baseline cap
                    # was too aggressive and forced enhanced routes back to pre-APM.
                    r.current_headway_min = min(
                        r.current_headway_min / scale_back,
                        120.0,
                    )
            else:
                # Overshoot exceeds enhanced cost: revert all enhanced to baseline
                for r in enhanced_routes:
                    r.current_headway_min = r.baseline_headway_min

        # --- Output metrics ---
        active_routes = [r for r in all_routes if r.current_headway_min < float("inf")]
        new_total_cost = sum(self.cost_model.route_annual_cost(r) for r in active_routes)
        freq_weighted_cov = self._compute_frequency_weighted_coverage(active_routes)
        transfer_opps = sum(
            1 for r in active_routes
            if r.current_headway_min <= PULSE_HEADWAY_THRESHOLD_MIN
        )

        freq_improvements = []
        for r in enhanced_routes:
            if r.baseline_headway_min > 0:
                pct = (r.baseline_headway_min - r.current_headway_min) / r.baseline_headway_min * 100
                freq_improvements.append(pct)
        avg_freq_improvement = float(np.mean(freq_improvements)) if freq_improvements else 0.0

        feeders = [r for r in active_routes if r.classification == "feeder"]
        parallels = [r for r in active_routes if r.classification == "parallel"]
        feeder_hw = float(np.mean([r.current_headway_min for r in feeders])) if feeders else 30.0
        # Cap at 60 min (FTA minimum reporting threshold) instead of inf
        # when all parallel routes have been eliminated or converted.
        MAX_PARALLEL_DEGRADATION_HW = 60.0
        parallel_hw = float(np.mean([r.current_headway_min for r in parallels])) if parallels else MAX_PARALLEL_DEGRADATION_HW

        n_eliminated = sum(1 for d in dispositions.values() if d == "eliminate")
        n_converted = sum(1 for d in dispositions.values() if d == "convert_to_feeder")
        n_truncated = sum(1 for d in dispositions.values() if d == "truncate")

        route_results = []
        for route in active_routes:
            route_results.append({
                "route_id": route.route_id,
                "name": route.name,
                "classification": route.classification,
                "disposition": dispositions.get(route.route_id, "synthetic_feeder"),
                "baseline_headway": route.baseline_headway_min,
                "current_headway": route.current_headway_min,
                "headway_change_pct": (
                    (route.current_headway_min / route.baseline_headway_min - 1.0) * 100
                    if route.baseline_headway_min > 0 else 0.0
                ),
                "annual_cost": self.cost_model.route_annual_cost(route),
                "productivity": self._route_productivity(route),
            })

        # Backward-compat: phase_frac = 1.0 when active, 0.0 when pre_apm
        phase_frac = 1.0 if is_active else 0.0

        return {
            "restructure_pressure": pressure,
            "redesign_phase_fraction": phase_frac,
            "service_phase": phase_name,
            "baseline_bus_cost": sum(self.cost_model.route_annual_cost(r) for r in self.routes),
            "new_bus_cost": new_total_cost,
            "total_cost": new_total_cost,
            "budget": budget,
            "budget_surplus": budget - new_total_cost,
            "budget_utilization": new_total_cost / max(budget, 1.0),
            "freed_service_hours": freed_hours,
            "frequency_improvement_pct": avg_freq_improvement,
            "n_eliminated_routes": n_eliminated,
            "n_converted_routes": n_converted,
            "n_truncated_routes": n_truncated,
            "n_synthetic_feeders": len(active_synthetic),
            "n_enhanced_routes": len(enhanced_routes),
            "coverage_loss_fraction": coverage_loss,
            "frequency_weighted_coverage": freq_weighted_cov,
            "transfer_opportunities": transfer_opps,
            "feeder_headway": feeder_hw,
            "parallel_headway": parallel_hw,
            "route_dispositions": dispositions,
            "routes": route_results,
        }

    # ---- coverage metrics --------------------------------------------------

    def _compute_coverage_loss(
        self,
        dispositions: Dict[str, str],
        phase_frac: float,
    ) -> float:
        """Fraction of previously bus-served area that loses all service.

        Uses a 200m grid sample around the corridor within 5km.
        """
        if self.corridor_stops_xy is None or len(self.corridor_stops_xy) < 2:
            return 0.0

        from scipy.spatial import cKDTree

        # Build baseline stop set (all routes)
        baseline_stops = []
        for rid, stops_xy in self.bus_stops_by_route.items():
            if len(stops_xy) > 0:
                baseline_stops.append(stops_xy)
        if not baseline_stops:
            return 0.0
        baseline_all = np.vstack(baseline_stops)
        baseline_tree = cKDTree(baseline_all)

        # Build redesigned stop set (exclude fully eliminated routes)
        redesign_stops = []
        for route in self.routes:
            disp = dispositions.get(route.route_id, "retain_enhanced")
            if disp == "eliminate" and phase_frac >= 1.0:
                continue  # this route is gone
            rid = route.route_id
            if rid in self.bus_stops_by_route and len(self.bus_stops_by_route[rid]) > 0:
                redesign_stops.append(self.bus_stops_by_route[rid])
        # Add synthetic feeder stops
        for rid, stops_xy in self.synthetic_feeder_stops.items():
            if len(stops_xy) > 0:
                redesign_stops.append(stops_xy)

        if not redesign_stops:
            return 1.0  # all service lost

        redesign_all = np.vstack(redesign_stops)
        redesign_tree = cKDTree(redesign_all)

        # Sample grid within 5km of corridor
        cx, cy = self.corridor_stops_xy.mean(axis=0)
        extent = 5000.0
        spacing = 200
        xs = np.arange(cx - extent, cx + extent, spacing)
        ys = np.arange(cy - extent, cy + extent, spacing)
        gx, gy = np.meshgrid(xs, ys)
        grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

        # Points served by baseline (within 400m of a bus stop).
        # Note: 400m is the overlap classification buffer (CORRIDOR_BUFFER_M).
        # Typical bus-stop walk access is 250-400m, so this is a generous
        # upper bound and may slightly understate coverage loss.
        base_dists, _ = baseline_tree.query(grid_pts, k=1)
        served_baseline = base_dists <= CORRIDOR_BUFFER_M

        # Points served by redesign
        redesign_dists, _ = redesign_tree.query(grid_pts, k=1)
        served_redesign = redesign_dists <= CORRIDOR_BUFFER_M

        # Coverage loss = baseline-served points that lose service
        lost = served_baseline & ~served_redesign
        if served_baseline.sum() == 0:
            return 0.0
        return float(lost.sum()) / float(served_baseline.sum())

    def _compute_frequency_weighted_coverage(
        self,
        active_routes: List[BusRoute],
    ) -> float:
        """Fraction of service area with frequency ≤ 15 min (high-frequency)."""
        if self.corridor_stops_xy is None or len(self.corridor_stops_xy) < 2:
            return 0.0

        from scipy.spatial import cKDTree

        # Collect stops from high-frequency routes
        hf_stops = []
        all_stops = []
        for route in active_routes:
            rid = route.route_id
            stops = self.bus_stops_by_route.get(rid)
            if stops is None:
                stops = self.synthetic_feeder_stops.get(rid)
            if stops is None or len(stops) == 0:
                continue
            all_stops.append(stops)
            if route.current_headway_min <= PULSE_HEADWAY_THRESHOLD_MIN:
                hf_stops.append(stops)

        if not all_stops:
            return 0.0

        all_xy = np.vstack(all_stops)
        all_tree = cKDTree(all_xy)

        # Sample grid
        cx, cy = self.corridor_stops_xy.mean(axis=0)
        extent = 5000.0
        spacing = 200
        xs = np.arange(cx - extent, cx + extent, spacing)
        ys = np.arange(cy - extent, cy + extent, spacing)
        gx, gy = np.meshgrid(xs, ys)
        grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

        all_dists, _ = all_tree.query(grid_pts, k=1)
        served = all_dists <= CORRIDOR_BUFFER_M
        n_served = served.sum()
        if n_served == 0:
            return 0.0

        if not hf_stops:
            return 0.0

        hf_xy = np.vstack(hf_stops)
        hf_tree = cKDTree(hf_xy)
        hf_dists, _ = hf_tree.query(grid_pts, k=1)
        hf_served = hf_dists <= CORRIDOR_BUFFER_M

        return float((served & hf_served).sum()) / float(n_served)

    # ---- convenience -------------------------------------------------------

    def get_feeder_headway(self) -> float:
        """Average feeder headway across existing + synthetic feeders."""
        feeders = [r for r in self.routes
                   if r.classification == "feeder" and r.current_headway_min > 0]
        feeders += [sf for sf in self.synthetic_feeders
                    if sf.current_headway_min > 0 and sf.current_headway_min < float("inf")]
        if not feeders:
            return 30.0
        return float(np.mean([r.current_headway_min for r in feeders]))

    def get_route_headways_dict(self) -> Dict[str, float]:
        """Return dict of route_id → current_headway_min for all active routes."""
        result = {}
        for r in self.routes:
            if 0 < r.current_headway_min < float("inf"):
                result[r.route_id] = r.current_headway_min
        for sf in self.synthetic_feeders:
            if 0 < sf.current_headway_min < float("inf"):
                result[sf.route_id] = sf.current_headway_min
        return result


# ---------------------------------------------------------------------------
# Bus ridership estimation (gravity model per route)
# ---------------------------------------------------------------------------

# Bus ridership gravity model parameters.
# Catchment: TCRP 165 recommends 800m (0.5 mi) for fixed-route bus in
# urban areas.  Was 400m, which excluded ~75% of population and produced
# unrealistically low system totals (~102 vs actual ~5,982 boardings).
BUS_STOP_CATCHMENT_M = 800  # max walk distance to bus stop (meters)

# Legacy flat boarding rate (kept for backward-compatible imports only).
# The MNL-based estimation below supersedes this.
_BUS_BOARDING_RATE = 0.050
BUS_TRIP_RATE = _BUS_BOARDING_RATE

# ---------------------------------------------------------------------------
# 3-mode MNL bus ridership estimation
# ---------------------------------------------------------------------------

# Trip generation
_DAILY_TRIP_RATE = 3.4    # NHTS 2017 Midwest small-metro trips/person/day

# Travel parameters
_WALK_SPEED_KPH = 5.0
_WALK_CIRCUITY = 1.20
_CAR_CIRCUITY = 1.20
_BUS_FARE = 2.00          # CityBus 2026 fare
_CAR_COST_PER_MILE = 0.60
_CAR_SPEED_KPH = 30.0     # urban average
_BUS_SPEED_KPH = 20.0     # CityBus system average

# Wait time parameters imported from mode_choice.py above

# Campus population (same as land_use_transport_model.py)
_PURDUE_ENROLLMENT = 50_000
_PURDUE_FACULTY_STAFF = 20_500
_CAMPUS_TOTAL = _PURDUE_ENROLLMENT * 0.25 + _PURDUE_FACULTY_STAFF * 0.10  # ~14,560

# CityBus observed system daily boardings (calibration target)
_CITYBUS_SYSTEM_DAILY_BOARDINGS = 5_982.0


def _bus_effective_wait_time(headway_min: float) -> float:
    """Perceived wait time — delegates to mode_choice.effective_wait_time."""
    return float(_effective_wait_time_impl(headway_min))


def estimate_bus_route_ridership(
    route: BusRoute,
    bus_stops_xy: np.ndarray,
    parcel_xy: np.ndarray,
    parcel_pop: np.ndarray,
    *,
    bus_speed_kph: Optional[float] = None,
    car_speed_kph: Optional[float] = None,
    parking_costs: Optional[np.ndarray] = None,
    institutional_weights: Optional[np.ndarray] = None,
    parcel_tree: Optional["cKDTree"] = None,
) -> float:
    """Estimate daily bus ridership for a single route using 3-mode MNL.

    For each parcel within 800m of a route stop, computes a 3-mode logit
    (bus, car, walk) to determine the probability of choosing bus.  Trips
    are generated at NHTS daily trip rate.  Campus parcels get boosted
    population from institutional_weights and student ASC adjustments.

    Distance decay is handled naturally by the walk-access term in bus
    utility — parcels farther from stops have higher access time, lower
    bus probability.

    Parameters
    ----------
    route : BusRoute with current_headway_min
    bus_stops_xy : (n_stops, 2) projected coordinates of this route's stops
    parcel_xy : (n_parcels, 2) projected parcel coordinates
    parcel_pop : (n_parcels,) population per parcel
    bus_speed_kph : override bus speed (defaults to _BUS_SPEED_KPH)
    car_speed_kph : override car speed (defaults to _CAR_SPEED_KPH)
    parking_costs : (n_parcels,) per-parcel parking cost for car mode
    institutional_weights : (n_parcels,) campus weights for student boost
    parcel_tree : pre-built cKDTree of parcel_xy (avoids per-route rebuild)

    Returns
    -------
    Estimated daily boardings for this route (uncalibrated — apply K-factor
    in estimate_all_route_ridership).
    """
    if route.current_headway_min >= float("inf") or len(bus_stops_xy) == 0:
        return 0.0

    from scipy.spatial import cKDTree

    _bus_speed = bus_speed_kph if bus_speed_kph is not None else _BUS_SPEED_KPH
    _car_speed = car_speed_kph if car_speed_kph is not None else _CAR_SPEED_KPH

    # Find all parcels within catchment of any stop on this route.
    # If a parcel tree is provided, use query_ball_point per stop (faster
    # than building a per-route stop tree and querying all 61K parcels).
    if parcel_tree is not None and len(bus_stops_xy) > 0:
        _catchment_idx_set: set = set()
        for _stop in bus_stops_xy:
            _catchment_idx_set.update(
                parcel_tree.query_ball_point(_stop, BUS_STOP_CATCHMENT_M)
            )
        idx = np.array(sorted(_catchment_idx_set), dtype=np.intp)
        if len(idx) == 0:
            return 0.0
        # Compute distance to nearest stop for the subset
        stop_tree = cKDTree(bus_stops_xy)
        walk_dist_m, _ = stop_tree.query(parcel_xy[idx], k=1)
    else:
        stop_tree = cKDTree(bus_stops_xy)
        dists, _ = stop_tree.query(parcel_xy, k=1)
        in_catchment = dists <= BUS_STOP_CATCHMENT_M
        idx = np.where(in_catchment)[0]
        if len(idx) == 0:
            return 0.0
        walk_dist_m = dists[idx]
    pop = parcel_pop[idx].copy()

    # Campus population boost: add campus-affiliated people to parcels
    # with institutional_weight > 1.0 (same logic as land_use_transport_model)
    if institutional_weights is not None:
        iw = institutional_weights[idx]
        campus_mask = iw > 1.0
        if campus_mask.any():
            raw_weights = np.where(campus_mask, iw - 1.0, 0.0)
            total_weight = raw_weights.sum()
            if total_weight > 0:
                campus_pop_add = _CAMPUS_TOTAL * (raw_weights / total_weight)
                pop = pop + campus_pop_add

    # Representative trip distance: average bus trip ~ 1/3 of route length
    rep_trip_km = max(route.length_km / 3.0, 0.5) if route.length_km > 0 else 2.0
    rep_trip_miles = rep_trip_km * 0.621371

    # --- Bus utility ---
    bus_ivt_min = (rep_trip_km / _bus_speed) * 60.0
    bus_wait_min = _bus_effective_wait_time(route.current_headway_min)
    bus_access_min = (walk_dist_m * _WALK_CIRCUITY / 1000.0 / _WALK_SPEED_KPH) * 60.0
    u_bus = (_BETA_IVT * bus_ivt_min
             + _BETA_WAIT * bus_wait_min
             + _BETA_ACCESS * bus_access_min
             + _BETA_COST * _BUS_FARE
             + _ASC_BUS)

    # --- Car utility ---
    car_ivt_min = (rep_trip_km * _CAR_CIRCUITY / _car_speed) * 60.0
    car_cost = rep_trip_miles * _CAR_CIRCUITY * _CAR_COST_PER_MILE
    if parking_costs is not None:
        car_cost = car_cost + parking_costs[idx]
    u_car = (_BETA_IVT * car_ivt_min
             + _BETA_ACCESS * 2.0  # walk to/from parked car: ~2 min
             + _BETA_COST * car_cost
             + _ASC_CAR)

    # --- Walk utility (only viable for short trips) ---
    walk_ivt_min = (rep_trip_km * _WALK_CIRCUITY / _WALK_SPEED_KPH) * 60.0
    u_walk = _BETA_IVT * walk_ivt_min + _ASC_WALK

    # Student ASC adjustments for campus parcels
    if institutional_weights is not None:
        from src.data.purdue_transit_demand import STUDENT_ASC_ADJUSTMENTS
        iw = institutional_weights[idx]
        campus_frac = np.clip((iw - 1.0) / 4.0, 0.0, 1.0)
        u_bus = u_bus + campus_frac * STUDENT_ASC_ADJUSTMENTS["bus"]
        u_car = u_car + campus_frac * STUDENT_ASC_ADJUSTMENTS["car"]
        u_walk = u_walk + campus_frac * STUDENT_ASC_ADJUSTMENTS["walk"]

    # MNL probability of choosing bus
    # Clip utilities to avoid overflow in exp()
    max_u = np.maximum(np.maximum(u_bus, u_car), u_walk)
    exp_bus = np.exp(np.clip(u_bus - max_u, -500, 0))
    exp_car = np.exp(np.clip(u_car - max_u, -500, 0))
    exp_walk = np.exp(np.clip(u_walk - max_u, -500, 0))
    denom = exp_bus + exp_car + exp_walk
    p_bus = exp_bus / denom

    daily_boardings = float(np.sum(pop * _DAILY_TRIP_RATE * p_bus))
    return max(daily_boardings, 0.0)


def estimate_all_route_ridership(
    routes: List[BusRoute],
    bus_stops_by_route: Dict[str, np.ndarray],
    parcel_xy: np.ndarray,
    parcel_pop: np.ndarray,
    corridor_relevant_only: bool = False,
    *,
    bus_speed_kph: Optional[float] = None,
    car_speed_kph: Optional[float] = None,
    parking_costs: Optional[np.ndarray] = None,
    institutional_weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Estimate daily ridership for bus routes using MNL + K-factor calibration.

    For each route with observed ridership, computes a per-route K-factor
    (observed / estimated) to calibrate the MNL.  Routes without observed
    data get a system-level K-factor (sum observed / sum estimated).

    Parameters
    ----------
    corridor_relevant_only : if True, only estimate ridership for routes
        classified as "parallel" or "feeder" to the active corridor.
    bus_speed_kph : override bus speed for MNL (defaults to 20 km/h)
    car_speed_kph : override car speed for MNL (defaults to 30 km/h)
    parking_costs : (n_parcels,) per-parcel parking cost for car mode
    institutional_weights : (n_parcels,) campus weights for student boost

    Returns dict of route_id -> calibrated daily boardings.
    """
    active_routes = []
    skip_routes = []
    for route in routes:
        if corridor_relevant_only and route.classification == "independent":
            skip_routes.append(route)
        else:
            active_routes.append(route)

    # Phase 1: compute raw MNL estimates for all active routes.
    # Build parcel KDTree once — shared across all per-route ridership calls
    # to avoid rebuilding 30+ per-route stop trees each querying 61K parcels.
    from scipy.spatial import cKDTree as _cKDTree
    _parcel_tree = _cKDTree(parcel_xy) if len(parcel_xy) > 0 else None
    raw_estimates: Dict[str, float] = {}
    for route in active_routes:
        stops_xy = bus_stops_by_route.get(str(route.route_id), np.empty((0, 2)))
        raw_estimates[route.route_id] = estimate_bus_route_ridership(
            route, stops_xy, parcel_xy, parcel_pop,
            bus_speed_kph=bus_speed_kph,
            car_speed_kph=car_speed_kph,
            parking_costs=parking_costs,
            institutional_weights=institutional_weights,
            parcel_tree=_parcel_tree,
        )

    # Phase 2: K-factor calibration using observed CityBus data
    # Per-route K = observed / estimated (for routes with observed data)
    # System K = sum(observed) / sum(estimated) (fallback for routes without data)
    sum_observed = 0.0
    sum_estimated = 0.0
    per_route_k: Dict[str, float] = {}
    for route in active_routes:
        est = raw_estimates[route.route_id]
        obs = route.observed_daily_riders
        if obs > 0 and est > 0:
            per_route_k[route.route_id] = obs / est
            sum_observed += obs
            sum_estimated += est

    system_k = (sum_observed / sum_estimated) if sum_estimated > 0 else 1.0

    # Phase 3: apply K-factors
    result: Dict[str, float] = {}
    for route in active_routes:
        est = raw_estimates[route.route_id]
        k = per_route_k.get(route.route_id, system_k)
        result[route.route_id] = est * k

    for route in skip_routes:
        result[route.route_id] = 0.0
    return result


# ---------------------------------------------------------------------------
# Integration helper: build network state for a corridor
# ---------------------------------------------------------------------------

def build_corridor_bus_network(
    routes: List[BusRoute],
    corridor_stops_xy: np.ndarray,
    bus_stops_by_route: Dict[str, np.ndarray],
    apm_daily_riders: float,
    apm_length_km: float,
    apm_n_stops: int,
    cost_model: Optional[BusOperatingCostModel] = None,
    restructure_pressure: float = 0.0,
    strategy: str = "incremental",
    synthetic_feeders: Optional[List[BusRoute]] = None,
    synthetic_feeder_stops: Optional[Dict[str, np.ndarray]] = None,
    year: int = 0,
    apm_opening_year: int = 0,
    classification_method: str = "geometric",
    od_cache: Optional[tuple] = None,
    parcel_xy: Optional[np.ndarray] = None,
    budget_mode: str = DEFAULT_BUDGET_MODE,
    directional_split: float = 0.0,
    bus_restructuring_mode: str = "reactive",
    **kwargs,
) -> Dict[str, object]:
    """End-to-end: classify routes, set APM frequency, optimize headways.

    Parameters
    ----------
    strategy : "incremental" (default) or "redesign".
    classification_method : "geometric" (400m buffer) or "od_based" (LODES OD overlap).
    od_cache : tuple from _build_lodes_od_cache (needed for od_based classification).
    parcel_xy : projected parcel coords (needed for od_based classification).
    year : simulation year (redesign phase-in logic).
    apm_opening_year : when APM opened (redesign phase-in logic).
    river_x : float, optional. X-coordinate of river for barrier penalty.
    budget_mode : 'separate', 'combined', or 'expanded'.
    directional_split : peak directional split for APM headway capacity calc.
    bus_restructuring_mode : 'reactive' (default, pressure-ramp) or 'proactive'
        (productivity-ranked decision engine with equity guard).

    Returns a dict with all network state needed by the feedback loop.
    """
    # Compute corridor-specific APM O&M
    from src.financial_params import (
        O_AND_M_FIXED_USD, O_AND_M_PER_KM_USD, O_AND_M_PER_STATION_USD,
        O_AND_M_ESCALATION_RATE, FARE_PER_TRIP_USD, OPERATING_DAYS_PER_YEAR,
    )
    apm_om_annual = (
        O_AND_M_FIXED_USD
        + apm_length_km * O_AND_M_PER_KM_USD
        + apm_n_stops * O_AND_M_PER_STATION_USD
    ) * (1.0 + O_AND_M_ESCALATION_RATE) ** year

    # APM fare revenue directed toward O&M, escalated with fare inflation
    escalated_fare = FARE_PER_TRIP_USD * (1.0 + FARE_ESCALATION_RATE) ** year
    apm_fare_revenue_annual = (
        apm_daily_riders * escalated_fare * OPERATING_DAYS_PER_YEAR
    )

    if cost_model is None:
        cost_model = BusOperatingCostModel(
            budget_mode=budget_mode,
            apm_om_annual=apm_om_annual,
            apm_fare_revenue_annual=apm_fare_revenue_annual,
            year=year,
        )
    else:
        # Update existing cost model with corridor-specific APM cost
        cost_model.budget_mode = budget_mode
        cost_model.apm_om_annual = apm_om_annual
        cost_model.apm_fare_revenue_annual = apm_fare_revenue_annual
        cost_model.year = year

    # Deep-copy routes so mutations (classification, headway) don't leak
    # between corridor evaluations when routes list is shared.
    routes = copy.deepcopy(routes)

    # 1. Classify routes relative to this corridor
    if classification_method == "od_based" and od_cache is not None and parcel_xy is not None:
        od_origins = od_cache["orig_idx_all"]
        od_dests = od_cache["dest_idx_all"]
        od_flows_s000 = od_cache["trips"]
        valid = od_cache.get("valid_trip", od_flows_s000 > 0)
        # Also filter out unmatched parcels (idx == -1)
        valid = valid & (od_origins >= 0) & (od_dests >= 0)
        n_parcels = len(parcel_xy)
        valid = valid & (od_origins < n_parcels) & (od_dests < n_parcels)
        classify_routes_od_based(
            routes, bus_stops_by_route, corridor_stops_xy,
            od_origins[valid], od_dests[valid], od_flows_s000[valid], parcel_xy,
        )
    else:
        classify_routes_for_corridor(routes, corridor_stops_xy, bus_stops_by_route)

    # 2. APM frequency response (capacity-constrained)
    apm_headway = compute_apm_headway(
        apm_daily_riders,
        corridor_length_km=apm_length_km,
        n_stops=apm_n_stops,
        directional_split=directional_split,
    )
    apm_service = APMService(
        corridor_id="active",
        length_km=apm_length_km,
        n_stops=apm_n_stops,
        headway_min=apm_headway,
        daily_riders=apm_daily_riders,
    )

    # 3. Strategy dispatch
    if bus_restructuring_mode == "proactive":
        # Proactive path: productivity-ranked decision engine with equity guard.
        # Build a RouteServicePlan and delegate to optimize_headways_proactive().
        service_plan = RouteServicePlan(
            routes=routes,
            cost_model=cost_model,
            apm_service=apm_service,
        )
        proactive_result = service_plan.optimize_headways_proactive(
            year=year,
            min_headway=kwargs.get("bus_min_feeder_headway", 10.0),
            max_headway=kwargs.get("bus_max_feeder_headway", 120.0),
            apm_daily_riders=apm_daily_riders,
            bus_stops_by_route=bus_stops_by_route,
            parcel_xy=parcel_xy,
            parcel_se01_share=kwargs.get("parcel_se01_share"),
            metro_se01_share=kwargs.get("metro_se01_share", 0.0),
        )

        # Extract feeder/parallel headways from the result
        feeder_routes_list = [r for r in routes if r.classification == "feeder"]
        parallel_routes_list = [r for r in routes if r.classification == "parallel"]
        _feeder_hws = [r.current_headway_min for r in feeder_routes_list
                       if 0 < r.current_headway_min < float("inf")]
        feeder_hw = float(np.mean(_feeder_hws)) if _feeder_hws else 30.0
        _parallel_hws = [r.current_headway_min for r in parallel_routes_list
                         if 0 < r.current_headway_min < float("inf")]
        parallel_hw = float(np.mean(_parallel_hws)) if _parallel_hws else 90.0

        # Sector coverage — include synthetic feeders alongside CityBus feeders
        _cov_stops_pro = dict(bus_stops_by_route)
        if synthetic_feeders:
            for sf in synthetic_feeders:
                if sf.current_headway_min > 0 and sf.current_headway_min < float("inf"):
                    feeder_routes_list.append(sf)
            if synthetic_feeder_stops:
                _cov_stops_pro.update(synthetic_feeder_stops)
        _route_hws = {str(r.route_id): r.current_headway_min for r in routes}
        if synthetic_feeders:
            for sf in synthetic_feeders:
                _route_hws[str(sf.route_id)] = sf.current_headway_min
        sector_cov = compute_sector_coverage(
            corridor_stops_xy=corridor_stops_xy,
            feeder_routes=feeder_routes_list,
            bus_stops_by_route=_cov_stops_pro,
            route_headways=_route_hws,
            feeder_parcel_sector=kwargs.get("feeder_parcel_sector"),
            feeder_parcel_pop=kwargs.get("feeder_parcel_pop"),
            feeder_parcel_xy=kwargs.get("feeder_parcel_xy"),
            feeder_parcel_se01=kwargs.get("feeder_parcel_se01"),
        )
        cx_centroid = corridor_stops_xy.mean(axis=0)[0]
        sector_cov = apply_barrier_penalty(
            sector_cov, cx_centroid, river_x=kwargs.get("river_x"),
        )

        phase = get_service_phase(year)
        n_parallel = sum(1 for r in routes if r.classification == "parallel")
        n_feeder = sum(1 for r in routes if r.classification == "feeder")
        n_independent = sum(1 for r in routes if r.classification == "independent")

        return {
            "phase": phase,
            "restructure_pressure": restructure_pressure,
            "bus_restructuring_mode": "proactive",
            "parallel_headway": parallel_hw,
            "feeder_headway": feeder_hw,
            "feeder_coverage_fraction": sector_cov.effective_coverage,
            "sector_coverage": sector_cov,
            "apm_headway_min": apm_headway,
            "apm_om_annual": apm_om_annual,
            "apm_fare_revenue_annual": apm_fare_revenue_annual,
            "weighted_bus_speed_kph": float(np.mean([r.avg_speed_kph for r in routes])) if routes else DEFAULT_BUS_SPEED_KPH,
            "tsp_speed_factor": 1.0,
            "n_parallel": n_parallel,
            "n_feeder": n_feeder,
            "n_independent": n_independent,
            "optimization": proactive_result,
            "proactive_decisions": proactive_result.get("route_actions", []),
        }

    if strategy == "redesign":
        redesign = NetworkRedesignStrategy(
            routes=routes,
            cost_model=cost_model,
            apm_service=apm_service,
            synthetic_feeders=synthetic_feeders,
            synthetic_feeder_stops=synthetic_feeder_stops or {},
            bus_stops_by_route=bus_stops_by_route,
            corridor_stops_xy=corridor_stops_xy,
            parcel_xy=parcel_xy,
            parcel_pop=kwargs.get("parcel_pop"),
            parking_costs=kwargs.get("parking_costs"),
            institutional_weights=kwargs.get("institutional_weights"),
        )
        optimization = redesign.redesign_network(
            restructure_pressure=restructure_pressure,
            year=year,
            apm_opening_year=apm_opening_year,
        )
        feeder_hw = optimization["feeder_headway"]
        parallel_hw = optimization["parallel_headway"]
        fleet = cost_model.fleet_summary(routes, [apm_service])

        # Use the discrete service phase from the redesign strategy
        phase = optimization.get("service_phase", "unknown")

        # Sector-based coverage: use the same metric as incremental strategy.
        # The old ``1.0 - coverage_loss_fraction`` conflated "area retaining
        # any bus" with "area with feeder-to-APM connectivity."  Sector
        # coverage accounts for station proximity and frequency weighting.
        feeder_routes_list = [r for r in routes if r.classification == "feeder"]
        # Include synthetic feeders in coverage
        _cov_stops_rd = dict(bus_stops_by_route)
        if synthetic_feeders:
            for sf in synthetic_feeders:
                if sf.current_headway_min > 0 and sf.current_headway_min < float("inf"):
                    feeder_routes_list.append(sf)
            if synthetic_feeder_stops:
                _cov_stops_rd.update(synthetic_feeder_stops)
        _route_hws = {str(r.route_id): r.current_headway_min for r in routes}
        if synthetic_feeders:
            for sf in synthetic_feeders:
                _route_hws[str(sf.route_id)] = sf.current_headway_min
        sector_cov = compute_sector_coverage(
            corridor_stops_xy=corridor_stops_xy,
            feeder_routes=feeder_routes_list,
            bus_stops_by_route=_cov_stops_rd,
            route_headways=_route_hws,
            feeder_parcel_sector=kwargs.get("feeder_parcel_sector"),
            feeder_parcel_pop=kwargs.get("feeder_parcel_pop"),
            feeder_parcel_xy=kwargs.get("feeder_parcel_xy"),
        )
        cx_centroid = corridor_stops_xy.mean(axis=0)[0]
        sector_cov = apply_barrier_penalty(
            sector_cov, cx_centroid, river_x=kwargs.get("river_x"),
        )
        # Apply phase-based floor (same as incremental strategy)
        _redesign_phase_floors = {
            "pre_apm": 0.0, "opening_day": 0.10,
            "early_operations": 0.20, "mature_network": 0.40,
            "second_generation": 0.50, "long_range_plan": 0.60,
        }
        _floor = _redesign_phase_floors.get(phase, 0.0)
        sector_cov.coverage = np.maximum(sector_cov.coverage, _floor)
        feeder_coverage = sector_cov.effective_coverage

        parallel_routes = [r for r in routes if r.classification == "parallel"]
        if parallel_routes:
            total_riders = sum(r.observed_daily_riders for r in parallel_routes)
            if total_riders > 0:
                weighted_speed = sum(
                    r.avg_speed_kph * r.observed_daily_riders for r in parallel_routes
                ) / total_riders
            else:
                weighted_speed = float(np.mean([r.avg_speed_kph for r in parallel_routes]))
        else:
            weighted_speed = DEFAULT_BUS_SPEED_KPH

        n_parallel = sum(1 for r in routes if r.classification == "parallel")
        n_feeder = sum(1 for r in routes if r.classification == "feeder")
        n_independent = sum(1 for r in routes if r.classification == "independent")

        return {
            "phase": phase,
            "restructure_pressure": restructure_pressure,
            "apm_headway_min": apm_headway,
            "apm_daily_riders": apm_daily_riders,
            "apm_vehicles": apm_service.vehicles_needed,
            "feeder_headway": feeder_hw,
            "parallel_headway": parallel_hw,
            "feeder_coverage_fraction": feeder_coverage,
            "sector_coverage": sector_cov,
            "weighted_bus_speed_kph": float(np.clip(weighted_speed, 8.0, 50.0)),
            "tsp_speed_factor": kwargs.get("tsp_speed_factor", 1.0),
            "pop_active": kwargs.get("pop_active", True),
            "n_parallel_routes": n_parallel,
            "n_feeder_routes": n_feeder,
            "n_independent_routes": n_independent,
            "route_headways": redesign.get_route_headways_dict(),
            "route_classifications": {r.route_id: r.classification for r in routes},
            "optimization": optimization,
            "fleet": fleet,
            # Redesign-specific metrics
            "n_eliminated_routes": optimization["n_eliminated_routes"],
            "n_synthetic_feeders": optimization["n_synthetic_feeders"],
            "coverage_loss_fraction": optimization["coverage_loss_fraction"],
            "frequency_improvement_pct": optimization["frequency_improvement_pct"],
            "freed_service_hours": optimization["freed_service_hours"],
            "redesign_phase_fraction": optimization["redesign_phase_fraction"],
            "frequency_weighted_coverage": optimization["frequency_weighted_coverage"],
            "transfer_opportunities": optimization["transfer_opportunities"],
            "route_dispositions": optimization["route_dispositions"],
        }

    # --- Incremental strategy (existing behavior) ---

    # 3. Per-route headway optimization
    plan = RouteServicePlan(routes, cost_model, apm_service)
    optimization = plan.optimize_headways(
        restructure_pressure=restructure_pressure,
        apm_daily_riders=apm_daily_riders,
    )

    # 4. Summary headways for ridership model
    feeder_hw = plan.get_feeder_headway()
    parallel_hw = plan.get_parallel_headway()

    # 5. Fleet summary
    fleet = cost_model.fleet_summary(routes, [apm_service])

    n_parallel = sum(1 for r in routes if r.classification == "parallel")
    n_feeder = sum(1 for r in routes if r.classification == "feeder")
    n_independent = sum(1 for r in routes if r.classification == "independent")

    # Determine restructure phase from classification counts and pressure
    if n_parallel == 0 and n_feeder == 0:
        phase = "no_overlap"
    elif restructure_pressure < 0.25:
        phase = "retain_parallel"
    elif restructure_pressure < 0.55:
        phase = "hybrid"
    elif restructure_pressure < 0.80:
        phase = "feeder_transition"
    else:
        phase = "feeder_dominant"

    # Compute sector-based feeder coverage: directional quality scores
    # for the feeder ring around the corridor.
    feeder_routes_list = [r for r in routes if r.classification == "feeder"]
    # Include synthetic feeders in coverage calculation alongside CityBus feeders
    _cov_stops = dict(bus_stops_by_route)
    if synthetic_feeders:
        for sf in synthetic_feeders:
            if sf.current_headway_min > 0 and sf.current_headway_min < float("inf"):
                feeder_routes_list.append(sf)
        if synthetic_feeder_stops:
            _cov_stops.update(synthetic_feeder_stops)
    _route_hws = {str(r.route_id): r.current_headway_min for r in routes}
    if synthetic_feeders:
        for sf in synthetic_feeders:
            _route_hws[str(sf.route_id)] = sf.current_headway_min
    sector_cov = compute_sector_coverage(
        corridor_stops_xy=corridor_stops_xy,
        feeder_routes=feeder_routes_list,
        bus_stops_by_route=_cov_stops,
        route_headways=_route_hws,
        feeder_parcel_sector=kwargs.get("feeder_parcel_sector"),
        feeder_parcel_pop=kwargs.get("feeder_parcel_pop"),
        feeder_parcel_xy=kwargs.get("feeder_parcel_xy"),
    )

    # Apply Wabash River barrier penalty (uses WABASH_APPROX_X if river_x not provided)
    cx_centroid = corridor_stops_xy.mean(axis=0)[0]
    sector_cov = apply_barrier_penalty(
        sector_cov, cx_centroid, river_x=kwargs.get("river_x"),
    )

    # Blend with phase-based floor to avoid zero coverage
    # Includes both pressure-based labels (from incremental path) and
    # discrete phase names (from PHASE_TRANSITION_YEARS / get_service_phase)
    phase_coverage_floor = {
        "no_overlap": 0.0,
        "retain_parallel": 0.0,
        "hybrid": 0.10,
        "feeder_transition": 0.20,
        "feeder_dominant": 0.40,
        # Discrete phase labels (from PHASE_TRANSITION_YEARS)
        "pre_apm": 0.0,
        "opening_day": 0.10,
        "early_operations": 0.20,
        "mature_network": 0.40,
        "second_generation": 0.50,
        "long_range_plan": 0.60,
    }
    floor = phase_coverage_floor.get(phase, 0.0)
    sector_cov.coverage = np.maximum(sector_cov.coverage, floor)
    feeder_coverage = sector_cov.effective_coverage

    # Weighted bus speed: ridership-weighted average of parallel-route speeds
    # (these are the routes competing with APM in the walk zone)
    parallel_routes = [r for r in routes if r.classification == "parallel"]
    if parallel_routes:
        total_riders = sum(r.observed_daily_riders for r in parallel_routes)
        if total_riders > 0:
            weighted_speed = sum(
                r.avg_speed_kph * r.observed_daily_riders for r in parallel_routes
            ) / total_riders
        else:
            weighted_speed = float(np.mean([r.avg_speed_kph for r in parallel_routes]))
    else:
        weighted_speed = DEFAULT_BUS_SPEED_KPH

    # Aggregate service profiles for feeder and parallel routes
    _feeder_profiles = [r.service_profile for r in routes
                        if r.classification == "feeder" and r.service_profile]
    _parallel_profiles = [r.service_profile for r in routes
                          if r.classification == "parallel" and r.service_profile]

    def _avg_profile(profiles: list, fallback_hw: float) -> Optional[ServiceProfile]:
        if not profiles:
            return ServiceProfile.from_single_headway(fallback_hw)
        n = len(profiles)
        return ServiceProfile(
            am_peak_headway_min=sum(p.am_peak_headway_min for p in profiles) / n,
            midday_headway_min=sum(p.midday_headway_min for p in profiles) / n,
            pm_peak_headway_min=sum(p.pm_peak_headway_min for p in profiles) / n,
            evening_headway_min=sum(p.evening_headway_min for p in profiles) / n,
            saturday_headway_min=sum(p.saturday_headway_min for p in profiles) / n,
            sunday_headway_min=sum(p.sunday_headway_min for p in profiles) / n,
        )

    return {
        "phase": phase,
        "restructure_pressure": restructure_pressure,
        "apm_headway_min": apm_headway,
        "apm_daily_riders": apm_daily_riders,
        "apm_vehicles": apm_service.vehicles_needed,
        "feeder_headway": feeder_hw,
        "parallel_headway": parallel_hw,
        "feeder_coverage_fraction": feeder_coverage,
        "sector_coverage": sector_cov,
        "feeder_service_profile": _avg_profile(_feeder_profiles, feeder_hw),
        "parallel_service_profile": _avg_profile(_parallel_profiles, parallel_hw),
        "weighted_bus_speed_kph": float(np.clip(weighted_speed, 8.0, 50.0)),
        "tsp_speed_factor": kwargs.get("tsp_speed_factor", 1.0),
        "pop_active": kwargs.get("pop_active", True),
        "n_parallel_routes": n_parallel,
        "n_feeder_routes": n_feeder,
        "n_independent_routes": n_independent,
        "route_headways": plan.get_route_headways_dict(),
        "route_classifications": {r.route_id: r.classification for r in routes},
        "optimization": optimization,
        "fleet": fleet,
    }

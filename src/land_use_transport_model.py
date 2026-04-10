"""
Iterative Land-Use-Transport Feedback Loop
===========================================
Runs an annual-step equilibrium model (default 0-25) that links:
  A. Ridership estimation (vectorized LODES mode choice)
  B. Endogenous development (pro forma feasibility x accessibility)
  C. Bus network restructuring (parallel -> feeder as APM matures)
  D. Temporal ridership ramp (logistic awareness S-curve)
"""
from __future__ import annotations

import copy
import io
import json
import logging
import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

# ---- Project imports ----
from scripts.generate_improved_ridership import (
    _build_parcel_lookup,
    compute_lodes_ridership,
    extract_corridor_stops,
    DECAY_BETA,
    POP_TRIP_RATE,
    JOB_TRIP_RATE,
    COMMUTE_TRIP_SHARE,
    BUS_HEADWAY_MIN,
    BUS_SPEED_KPH,
    APM_HEADWAY_MIN,
    CAR_SPEED_KPH,
)

try:
    from scripts.generate_improved_ridership import DEFAULT_APM_HEADWAY_MIN
except ImportError:
    from src.bus_network import DEFAULT_APM_HEADWAY_MIN

from src.spatial_constants import PROJECT_CRS, WALK_CATCHMENT_M, FEEDER_CATCHMENT_M, US_SURVEY_FT_TO_M
from src.developer_proforma import ZONING_MATRIX, MARKET_CONFIG, NO_ZONING_FAR_CAP
from src.demand_driven_development import (
    DemandDrivenDevelopmentModel,
    MetroGrowthParams,
    MarketParams,
    CorridorCaptureParams,
    ZoningCostParams,
    AbsorptionParams,
    DEFAULT_SEGMENTS,
)
from src.gtfs_ridership import load_gtfs_competitiveness_summary
from src.bus_network import (
    APM_PASSENGERS_PER_VEHICLE,
    BusOperatingCostModel,
    BusRoute,
    APMService,
    RouteServicePlan,
    ServiceProfile,
    build_corridor_bus_network,
    classify_routes_for_corridor,
    compute_apm_headway,
    compute_transit_headway,
    decide_route_restructuring,
    check_coverage_equity,
    apply_restructuring_decisions,
    estimate_all_route_ridership,
    get_service_phase,
    is_phase_transition_year,
    load_bus_routes_from_gtfs,
    truncate_parallel_routes,
    POST_OPENING_ADJUSTMENT_YEAR,
    SectorCoverage,
    CITYBUS_ANNUAL_BUDGET_USD,
    DEFAULT_COST_PER_VEH_HOUR,
    DEFAULT_SERVICE_SPAN_HOURS,
)
from src.financial_params import CARS_PER_TRAIN, SERVICE_DAYS_PER_YEAR

# Re-export everything from model_constants so tests can still
# ``from src.land_use_transport_model import OCCUPANCY_SCHEDULE`` etc.
from src.model_constants import *                          # noqa: F401,F403
from src.model_constants import (
    _LOW_DENSITY_ZONES,
    _PERIODS,
    _resolve_congestion_profile,
    _phasing_gate,
    _clip_param,
)

# Re-export helpers so test files work unchanged
from src.model_helpers import (                             # noqa: F401
    _read_geojson_fast,
    compute_relative_delta,
    evaluate_convergence,
    summarize_year_convergence,
    evaluate_stop_conditions,
    update_capacity_state,
    _restructure_pressure,
    _sparse_dot,
    _sparse_accumulate,
    _init_corridor_worker,
    _run_corridor_batch,
    _make_lafayette_proforma_config,
)

from src.ridership_engine import RidershipMixin

BASE_BUS_HEADWAY = BUS_HEADWAY_MIN

logger = logging.getLogger(__name__)

# Scenario-dependent rent signal: deregulation reduces investor uncertainty.
# no_zoning gets a small premium (2%) reflecting lower regulatory risk.
# current_zoning is the baseline (0%).
_SCENARIO_RENT_SIGNAL = {
    "current_zoning": 0.00,
    "no_zoning": 0.02,
}


# ============================================================================
# LandUseTransportModel
# ============================================================================

class LandUseTransportModel(RidershipMixin):
    """Iterative land-use-transport equilibrium model.

    Parameters
    ----------
    corridors_path : path to corridors GeoJSON
    parcels_path : path to enriched parcels GeoJSON (or None to auto-generate)
    od_path : path to LODES OD flows CSV
    time_steps : sequence of simulation years (default annual 0-25)
    bus_restructure : whether to model bus network restructuring
    adaptive_stop : enable convergence-based early stop
    development_scenario : one of 'current_zoning', 'no_zoning'
    corridor_filter : restrict to these corridor IDs (None = all)
    model_options : dict of behavioral overrides for sensitivity analysis
    ridership_scale_multiplier : overall ridership scaling factor
    transit_mode : 'apm' or 'brt'
    """

    def __init__(
        self,
        corridors_path: str = "data/processed/apm_phase2a_corridors.geojson",
        parcels_path: Optional[str] = "data/processed/parcels_enriched_final.geojson",
        od_path: str = "data/processed/od_parcel_flows_lodes.csv",
        time_steps: Tuple[int, ...] = tuple(range(26)),
        bus_restructure: bool = True,
        adaptive_stop: bool = DEFAULT_ADAPTIVE_STOP,
        development_scenario: str = "current_zoning",
        corridor_filter: Optional[List[str]] = None,
        model_options: Optional[Dict] = None,
        ridership_scale_multiplier: float = RIDERSHIP_SCALE_MULTIPLIER_DEFAULT,
        transit_mode: str = "apm",
        # Convergence tolerances
        ridership_convergence_tol: Optional[float] = None,
        development_convergence_tol: Optional[float] = None,
        convergence_floor: Optional[float] = None,
        max_time_steps: Optional[int] = None,
        consecutive_converged_steps: Optional[int] = None,
        stop_on_divergence: Optional[bool] = None,
        divergence_threshold: Optional[float] = None,
        consecutive_divergent_steps: Optional[int] = None,
        # Bus operating parameters
        bus_service_span_hours: Optional[float] = None,
        bus_parallel_route_equiv: Optional[float] = None,
        bus_feeder_route_equiv: Optional[float] = None,
        bus_service_hour_budget_multiplier: Optional[float] = None,
        bus_max_parallel_headway: Optional[float] = None,
        bus_min_feeder_headway: Optional[float] = None,
        bus_max_feeder_headway: Optional[float] = None,
        bus_network_strategy: Optional[str] = None,
        # Ridership calibration
        commute_direction_min: Optional[float] = None,
        commute_direction_max: Optional[float] = None,
        # Metro growth
        metro_growth_params: Optional[Dict[str, float]] = None,
        corridor_capture_params: Optional[Dict[str, float]] = None,
        # Development model params
        market_params: Optional[Dict[str, float]] = None,
        zoning_cost_params: Optional[Dict[str, float]] = None,
        absorption_params: Optional[Dict[str, float]] = None,
        # GTFS
        gtfs_dir: Optional[str] = None,
        gtfs_productivity_csvs: Optional[Sequence[str]] = None,
    ):
        self.time_steps = time_steps
        self.bus_restructure = bus_restructure
        self.adaptive_stop = adaptive_stop
        self.development_scenario = development_scenario
        self.corridor_filter = corridor_filter
        self._model_options: Dict = model_options or {}
        self.ridership_scale_multiplier = ridership_scale_multiplier

        # Transit mode configuration
        self._transit_mode_name = transit_mode.lower()
        if self._transit_mode_name == "brt":
            from scripts.generate_improved_ridership import ASC_BUS
            self._transit_asc = 0.05  # BRT ASC lower than APM
        else:
            from scripts.generate_improved_ridership import ASC_APM
            self._transit_asc = ASC_APM  # 0.18

        # Convergence parameters (CLI overrides > model_options > defaults)
        self._ridership_tol = float(
            ridership_convergence_tol if ridership_convergence_tol is not None
            else self._model_options.get("ridership_tol", DEFAULT_RIDERSHIP_CONVERGENCE_TOL))
        self._development_tol = float(
            development_convergence_tol if development_convergence_tol is not None
            else self._model_options.get("development_tol", DEFAULT_DEVELOPMENT_CONVERGENCE_TOL))
        self._convergence_floor = float(
            convergence_floor if convergence_floor is not None
            else DEFAULT_CONVERGENCE_FLOOR)
        self._consecutive_converged_steps = int(
            consecutive_converged_steps if consecutive_converged_steps is not None
            else DEFAULT_CONSECUTIVE_CONVERGED_STEPS)
        self._stop_on_divergence = bool(
            stop_on_divergence if stop_on_divergence is not None
            else DEFAULT_STOP_ON_DIVERGENCE)
        self._divergence_threshold = float(
            divergence_threshold if divergence_threshold is not None
            else DEFAULT_DIVERGENCE_THRESHOLD)
        self._consecutive_divergent_steps = int(
            consecutive_divergent_steps if consecutive_divergent_steps is not None
            else DEFAULT_CONSECUTIVE_DIVERGENT_STEPS)

        # Store bus operating params for restructure engine (fallback to module defaults)
        self._bus_service_span_hours = (
            max(float(bus_service_span_hours), 1.0) if bus_service_span_hours is not None
            else DEFAULT_BUS_SERVICE_SPAN_HOURS)
        self._bus_parallel_route_equiv = (
            max(float(bus_parallel_route_equiv), 0.0) if bus_parallel_route_equiv is not None
            else DEFAULT_BUS_PARALLEL_ROUTE_EQUIV)
        self._bus_feeder_route_equiv = (
            max(float(bus_feeder_route_equiv), 0.0) if bus_feeder_route_equiv is not None
            else DEFAULT_BUS_FEEDER_ROUTE_EQUIV)
        self._bus_service_hour_budget_multiplier = (
            max(float(bus_service_hour_budget_multiplier), 0.0) if bus_service_hour_budget_multiplier is not None
            else DEFAULT_BUS_SERVICE_HOUR_BUDGET_MULTIPLIER)
        self._bus_max_parallel_headway = (
            max(float(bus_max_parallel_headway), BASE_BUS_HEADWAY) if bus_max_parallel_headway is not None
            else MAX_PARALLEL_BUS_HEADWAY)
        self._bus_min_feeder_headway = (
            max(float(bus_min_feeder_headway), 1.0) if bus_min_feeder_headway is not None
            else MIN_FEEDER_BUS_HEADWAY)
        self._bus_max_feeder_headway = (
            max(float(bus_max_feeder_headway), self._bus_min_feeder_headway)
            if bus_max_feeder_headway is not None
            else DEFAULT_BUS_MAX_FEEDER_HEADWAY)
        self._bus_network_strategy = bus_network_strategy

        # max_time_steps: cap the number of time steps in run()
        self._max_time_steps = (
            max(int(max_time_steps), 1) if max_time_steps is not None
            else DEFAULT_MAX_TIME_STEPS)

        # Ridership calibration — directional split bounds
        if commute_direction_min is None:
            self._commute_direction_min = COMMUTE_DIRECTION_MIN_DEFAULT
        else:
            self._commute_direction_min = float(np.clip(commute_direction_min, 0.0, 0.95))
        if commute_direction_max is None:
            self._commute_direction_max = COMMUTE_DIRECTION_MAX_DEFAULT
        else:
            self._commute_direction_max = float(np.clip(commute_direction_max, 0.05, 1.0))
        if self._commute_direction_max < self._commute_direction_min:
            self._commute_direction_min, self._commute_direction_max = (
                self._commute_direction_max, self._commute_direction_min)

        # Metro growth params — passed to DemandDrivenDevelopmentModel
        self._metro_growth_params = metro_growth_params or {}
        self._corridor_capture_params = corridor_capture_params or {}

        # GTFS paths
        self.gtfs_dir = Path(gtfs_dir) if gtfs_dir else None
        self.gtfs_productivity_csvs = tuple(gtfs_productivity_csvs or ())

        # Instantiate DemandDrivenDevelopmentModel with scenario-dependent params
        _growth_kw = dict(metro_growth_params or {})
        _market_kw = dict(market_params or {})
        _capture_kw = dict(corridor_capture_params or {})
        _zoning_cost_kw = dict(zoning_cost_params or {})
        _absorption_kw = dict(absorption_params or {})
        self.demand_dev_model = DemandDrivenDevelopmentModel(
            growth_params=MetroGrowthParams(**_growth_kw) if _growth_kw else None,
            market_params=MarketParams(**_market_kw) if _market_kw else None,
            capture_params=CorridorCaptureParams(**_capture_kw) if _capture_kw else None,
            zoning_cost_params=ZoningCostParams(**_zoning_cost_kw) if _zoning_cost_kw else None,
            absorption_params=AbsorptionParams(**_absorption_kw) if _absorption_kw else None,
        )

        # Zero-car multiplier for sensitivity analysis
        self._zero_car_mult = float(self._model_options.get("zero_car_mult", 1.0))

        # Pop-active flag (Tier 3 proactive restructuring)
        self._pop_active = bool(self._model_options.get("pop_active", False))

        # ---- Load data ----
        logger.info("Loading data...")

        # Corridors
        self.corridors = _read_geojson_fast(corridors_path)

        # Parcels: auto-generate if path is None or missing
        if parcels_path is None:
            from src.ensure_enriched import ensure_enriched_parcels
            parcels_path = str(ensure_enriched_parcels(
                Path("data/processed/parcels_enriched_final.geojson")))
        parcels_p = Path(parcels_path)
        if not parcels_p.exists():
            from src.ensure_enriched import ensure_enriched_parcels
            parcels_path = str(ensure_enriched_parcels(parcels_p))
        self.parcels = _read_geojson_fast(parcels_path)
        self.parcels = self.parcels[
            self.parcels.geometry.notna() & ~self.parcels.geometry.is_empty
        ].copy()

        # OD flows
        od_p = Path(od_path)
        if od_p.exists():
            self.od_flows = pd.read_csv(od_path, dtype={"origin_parcel": str, "dest_parcel": str})
            # Strip "ST" prefix from synthetic parcel IDs so they match
            # enriched parcel STKEY IDs (e.g. "ST790601..." → "790601...")
            for _col in ("origin_parcel", "dest_parcel"):
                if _col in self.od_flows.columns:
                    self.od_flows[_col] = self.od_flows[_col].apply(
                        lambda x: x[2:] if isinstance(x, str) and x.startswith("ST") else x
                    )
        else:
            self.od_flows = pd.DataFrame(
                columns=["origin_parcel", "dest_parcel", "trips"])

        # Corridor filter
        if corridor_filter is not None:
            self.corridors = self.corridors[
                self.corridors["corridor_id"].isin(corridor_filter)
            ].copy()

        # Identify columns
        self.pop_col = (
            "pop_alloc" if "pop_alloc" in self.parcels.columns else "population"
        )
        self.jobs_col = next(
            (c for c in ["jobs_combined", "jobs_lehd_wac", "estimated_jobs", "jobs_alloc"]
             if c in self.parcels.columns),
            None,
        )
        self.zone_col = next(
            (c for c in ["zone_code", "RefName", "ZONE"] if c in self.parcels.columns),
            None,
        )

        n_corr = len(self.corridors)
        n_parc = len(self.parcels)
        pop_total = self.parcels[self.pop_col].sum() if self.pop_col in self.parcels.columns else 0
        jobs_total = self.parcels[self.jobs_col].sum() if self.jobs_col and self.jobs_col in self.parcels.columns else 0
        logger.info(
            f"  Corridors: {n_corr}, Parcels: {n_parc}, "
            f"OD flows: {len(self.od_flows):,}"
        )
        logger.info(f"  Pop: {pop_total:,.0f}  Jobs: {jobs_total:,.0f}")

        # ---- Build corridor metadata ----
        self._corridor_meta: Dict[str, dict] = {}
        self._corridor_rows: Dict[str, object] = {}
        for _, row in self.corridors.iterrows():
            cid = row["corridor_id"]
            # Parse stop_coords from GeoJSON property (JSON string → list)
            _raw_sc = row.get("stop_coords", None)
            if isinstance(_raw_sc, str):
                try:
                    _raw_sc = json.loads(_raw_sc)
                except (json.JSONDecodeError, TypeError):
                    _raw_sc = []
            elif not isinstance(_raw_sc, list):
                _raw_sc = []
            self._corridor_meta[cid] = {
                "length_km": float(row.get("length_km", 1.0)),
                "n_stops": int(row.get("n_stops", 4)),
                "stop_coords": _raw_sc,
            }
            self._corridor_rows[cid] = row

        # ---- Build parcel coordinate cache ----
        logger.info("Building parcel coordinate index...")
        self.parcel_cache = _build_parcel_lookup(self.parcels)

        # ---- OD cache ----
        self.od_cache = self._load_or_build_od_cache()

        # ---- Per-corridor bus/feeder headways ----
        cid_list = list(self._corridor_meta.keys())
        self._bus_headways: Dict[str, float] = {c: BASE_BUS_HEADWAY for c in cid_list}
        self._feeder_headways: Dict[str, float] = {c: BASE_BUS_HEADWAY for c in cid_list}
        self._apm_headways: Dict[str, float] = {c: APM_HEADWAY_MIN for c in cid_list}
        self._bus_speeds: Dict[str, float] = {c: BUS_SPEED_KPH for c in cid_list}
        self._feeder_coverage: Dict[str, float] = {c: 0.15 for c in cid_list}
        self._restructure_pressure: Dict[str, float] = {c: 0.0 for c in cid_list}
        self._feeder_budget_utilization: Dict[str, float] = {c: 0.0 for c in cid_list}
        self._transfer_walk_min: Dict[str, float] = {c: 0.0 for c in cid_list}
        self._sector_coverage: Dict[str, Optional[SectorCoverage]] = {c: None for c in cid_list}
        self._tsp_speed_factors: Dict[str, float] = {c: 1.0 for c in cid_list}
        self._bus_service_profiles: Dict[str, Optional[ServiceProfile]] = {c: None for c in cid_list}
        self._feeder_service_profiles: Dict[str, Optional[ServiceProfile]] = {c: None for c in cid_list}
        self._congestion_profiles: Dict[str, dict] = {
            c: _resolve_congestion_profile(c) for c in cid_list
        }
        self._directional_split: Dict[str, float] = {c: 0.60 for c in cid_list}

        # ---- Base catchment tracking ----
        self._base_catchment_pop: Dict[str, float] = {}
        self._base_catchment_jobs: Dict[str, float] = {}
        self._base_feeder_pop_catch: Dict[str, float] = {}
        self._congestion_factor: float = 1.0

        # ---- Campus coordinates ----
        self._CAMPUS_LON = -86.9213
        self._CAMPUS_LAT = 40.4259

        # ---- Spatial cache ----
        self._corridor_spatial_cache: Dict[str, dict] = {}
        self._precompute_corridor_spatial_cache()

        # ---- Parking and institutional weights ----
        self._parking_costs: Optional[np.ndarray] = None
        self._institutional_weights: Optional[np.ndarray] = None
        self._build_parking_costs()
        self._build_institutional_weights()

        # ---- Development model inputs ----
        self._current_rents: Optional[np.ndarray] = None
        self._total_res_units: Optional[np.ndarray] = None
        self._occupied_res_units: Optional[np.ndarray] = None
        self._total_comm_sqft: Optional[np.ndarray] = None
        self._occupied_comm_sqft: Optional[np.ndarray] = None
        self._pending_deliveries: Dict[int, List[dict]] = {}
        self._cumulative_corridor_pop: Dict[str, float] = {}
        self._cumulative_corridor_jobs: Dict[str, float] = {}
        self._initialize_parcel_development_inputs()

        # ---- Proforma ----
        self._proforma = None
        self._proforma_config = None
        self._proforma_cache: Dict[str, object] = {}  # scenario → SqFtProForma

        # ---- Bus restructure context ----
        self._bus_routes: Optional[List] = None
        self._gtfs_competitiveness: Optional[pd.DataFrame] = None
        self._last_restructure_ridership: Dict[str, float] = {}
        if bus_restructure:
            self._initialize_bus_restructure_context()

        # Storage for results
        self._results: List[dict] = []
        self._diagnostics: List[dict] = []

        # Destination coverage cache (projected coords for student off-campus destinations)
        try:
            from pyproj import Transformer
            _to_proj = Transformer.from_crs("EPSG:4326", PROJECT_CRS, always_xy=True)
            _pts = []
            from src.model_constants import STUDENT_OFFCAMPUS_DESTINATIONS
            for _lon, _lat in STUDENT_OFFCAMPUS_DESTINATIONS.values():
                px, py = _to_proj.transform(_lon, _lat)
                _pts.append((px * US_SURVEY_FT_TO_M, py * US_SURVEY_FT_TO_M))
            self._dest_proj_m = np.array(_pts) if _pts else None
        except Exception:
            self._dest_proj_m = None

        # Warning flags for one-shot warnings
        self._d1_warned = False
        self._d4_warned = False

        logger.info("Model initialized.")

    # ------------------------------------------------------------------
    # Commute time precomputation
    # ------------------------------------------------------------------

    def _precompute_commute_times(self) -> np.ndarray:
        """Compute flow-weighted average commute time per parcel (minutes).

        Uses the OD flow matrix and car speed to estimate a representative
        commute time per parcel.  This feeds into the relocation MNL so
        that station-proximate parcels with shorter commutes attract more
        household relocations.

        Vectorized: accumulates (time × trips) and (trips) per origin,
        then divides.  Falls back to 15 min default for parcels with no OD.
        """
        n = len(self.parcels)
        commute_times = np.full(n, 15.0)  # default 15 min
        if len(self.od_flows) == 0:
            return commute_times

        _, parcel_xy, pid_to_idx = self.parcel_cache
        try:
            # Accumulate flow-weighted commute times
            commute_sum = np.zeros(n, dtype=np.float64)
            flow_sum = np.zeros(n, dtype=np.float64)

            origins = self.od_flows.get("origin_parcel")
            dests = self.od_flows.get("dest_parcel")
            trips_col = self.od_flows.get("trips")
            if origins is None or dests is None:
                return commute_times

            for i in range(len(self.od_flows)):
                o_raw = origins.iat[i]
                d_raw = dests.iat[i]
                # Normalize: strip ST prefix for synthetic persons
                o_key = str(o_raw).lstrip("ST") if isinstance(o_raw, str) else o_raw
                d_key = str(d_raw).lstrip("ST") if isinstance(d_raw, str) else d_raw
                o_idx = pid_to_idx.get(o_key)
                d_idx = pid_to_idx.get(d_key)
                if o_idx is None or d_idx is None:
                    continue
                dist_m = np.sqrt(((parcel_xy[o_idx] - parcel_xy[d_idx]) ** 2).sum())
                dist_km = dist_m / 1000.0
                time_min = (dist_km / max(CAR_SPEED_KPH, 1.0)) * 60.0
                trips = float(trips_col.iat[i]) if trips_col is not None else 1.0
                commute_sum[o_idx] += time_min * trips
                flow_sum[o_idx] += trips

            # Weighted average where we have flow data
            has_flow = flow_sum > 0
            commute_times[has_flow] = commute_sum[has_flow] / flow_sum[has_flow]
        except Exception:
            pass
        return commute_times

    # ------------------------------------------------------------------
    # Parcel FAR limits
    # ------------------------------------------------------------------

    def _get_parcel_max_far(self, zone_code: str) -> float:
        """Get maximum FAR for a parcel given zone and scenario."""
        entry = ZONING_MATRIX.get(zone_code)
        if entry is None:
            return 0.0
        base_far = entry[0]
        if self.development_scenario == "no_zoning":
            # no_zoning removes caps but never reduces below current zoning
            return max(base_far, NO_ZONING_FAR_CAP)
        return base_far

    # ------------------------------------------------------------------
    # Proforma initialization (scenario-dependent profit hurdle)
    # ------------------------------------------------------------------

    # Developers accept lower profit margins under less regulatory risk.
    # Base: 17.5% margin.  no_zoning: -10% (no entitlement delays).
    _SCENARIO_PROFIT_FACTOR = {
        "current_zoning": 1.175,
        "no_zoning": 1.175 * 0.90,
    }

    def _ensure_proforma(self):
        """Lazily create SqFtProForma for the current scenario.

        profit_factor is baked into the pre-computed lookup table at
        __init__, so we need a separate instance per scenario.  Cached
        in self._proforma_cache to avoid re-creation.
        """
        _key = self.development_scenario
        if _key not in self._proforma_cache:
            from urbansim.developer.sqftproforma import SqFtProForma
            config = _make_lafayette_proforma_config()
            config.profit_factor = self._SCENARIO_PROFIT_FACTOR.get(_key, 1.175)
            self._proforma_cache[_key] = SqFtProForma(config)
        self._proforma = self._proforma_cache[_key]
        self._proforma_config = self._proforma.config

    # ------------------------------------------------------------------
    # Proforma preparation
    # ------------------------------------------------------------------

    def _prepare_proforma_df(
        self,
        corridor_positions: np.ndarray,
        accessibility: np.ndarray,
        riders: float,
        year: int,
    ) -> Optional[pd.DataFrame]:
        """Build a DataFrame for SqFtProForma.lookup() from parcel data.

        Applies accessibility-driven rent premiums and temporal rent growth.
        Returns None if no developable parcels.
        """
        if len(corridor_positions) == 0:
            return None

        self._ensure_proforma()

        pos = corridor_positions.astype(int)
        zone_arr = (
            self.parcels[self.zone_col].values if self.zone_col else
            np.full(len(self.parcels), "R3")
        )

        # Speculative discount: developers discount transit rent premiums
        # before the system proves itself.  Only the transit premium portion
        # is discounted; base rents are unaffected.
        # FTA before-after studies: realized premiums average 60-80% of
        # projections in years 1-5, converging to 90-100% by year 8-10.
        _PREMIUM_MATURATION_YEARS = 10
        _SPECULATIVE_FLOOR = 0.50 if self._transit_mode_name != "brt" else 0.40
        if year < _PREMIUM_MATURATION_YEARS and _PREMIUM_MATURATION_YEARS > 0:
            maturation = _SPECULATIVE_FLOOR + (1.0 - _SPECULATIVE_FLOOR) * (year / _PREMIUM_MATURATION_YEARS)
        else:
            maturation = 1.0

        # Demolition cost: CurImpAV-scaled, $4/sqft floor to $10/sqft ceiling.
        # Low-value parcels (vacant/simple) cost less to demolish; high-value
        # parcels (substantial structures) cost more.  $50/sqft CurImpAV is
        # midpoint on scale.  RSMeans 2024 Midwest avg is ~$6/sqft.
        _DEMO_FLOOR_PSF = 4.0
        _DEMO_CEIL_PSF = 10.0
        _DEMO_REF_IMP_PSF = 50.0  # CurImpAV/sqft at which cost = ceiling

        rows = []
        valid_idx = []
        for p in pos:
            # Skip exempt parcels (PropClass 600-series: govt/educational/church)
            if not self._parcel_developable_mask[p]:
                continue

            zone = str(zone_arr[p])
            entry = ZONING_MATRIX.get(zone)
            if entry is None:
                continue
            _, use_type, _max_dua = entry
            if use_type == "undevelopable":
                continue
            max_far = self._get_parcel_max_far(zone)
            # DUA constraint from UZO (R1=4.35, R2=10.9, R3+=None)
            # no_zoning removes DUA restrictions along with FAR caps
            parcel_max_dua = (
                None if self.development_scenario == "no_zoning" else _max_dua
            )

            # Deduct already-built sqft from max FAR
            parcel_sqft = float(self._parcel_sqft_arr[p]) if self._parcel_sqft_arr is not None else 5000.0
            if parcel_sqft < 2000:
                continue
            if parcel_sqft > 0:
                built_far = self._parcel_built_sqft[p] / parcel_sqft
                max_far = max(max_far - built_far, 0.0)
            if max_far < 0.1:
                continue  # effectively built out

            # Land cost from assessor data + demolition for existing structures
            land_av = self.parcels.iloc[p].get("CurLandAV", 0)
            if land_av > 0 and parcel_sqft > 0:
                land_cost = float(np.clip(land_av, 1000, 5_000_000))
            else:
                land_cost = parcel_sqft * 10.0

            # Demolition cost: scaled by CurImpAV intensity
            imp_av = float(self._parcel_imp_av[p])
            if imp_av > 0:
                imp_psf = imp_av / max(parcel_sqft, 100.0)
                demo_psf = _DEMO_FLOOR_PSF + (_DEMO_CEIL_PSF - _DEMO_FLOOR_PSF) * min(imp_psf / _DEMO_REF_IMP_PSF, 1.0)
                land_cost += demo_psf * parcel_sqft

            # Rent premium from accessibility + temporal growth + mode premium
            acc = float(accessibility[p]) if p < len(accessibility) else 0.0
            # Service quality: sqrt of riders/reference, concave
            quality = min(np.sqrt(max(riders, 0.0) / 20000.0), 1.0)
            rent_premium = MAX_RENT_PREMIUM * acc * quality
            temporal_premium = (1 + ANNUAL_STATION_RENT_GROWTH) ** year - 1
            total_premium = rent_premium + temporal_premium * acc * quality

            # Apply speculative discount to the premium portion only
            total_premium = total_premium * maturation

            # Mode-specific rent multiplier
            if self._transit_mode_name == "brt":
                mode_mult = BRT_RENT_MULT
            else:
                mode_mult = FIXED_GUIDEWAY_RENT_MULT

            # Base rents from current_rents array (per-parcel, updated by vacancy feedback)
            base_rent = float(self._current_rents[p]) if self._current_rents is not None else 18.0
            adj_rent = base_rent * (1.0 + total_premium) * mode_mult

            # Scenario commitment signal: deregulation reduces investor risk,
            # raising year-0 rents slightly.  Portland MAX: 3-5% at announcement.
            adj_rent *= (1.0 + _SCENARIO_RENT_SIGNAL.get(self.development_scenario, 0.0))

            # Height from FAR × stories
            max_height = max_far * 12.0  # 12 ft per story

            row_dict = {
                "residential": adj_rent,
                "retail": adj_rent * 1.1,
                "office": adj_rent * 0.95,
                "industrial": adj_rent * 0.55,
                "land_cost": land_cost,
                "parcel_size": parcel_sqft,
                "max_far": max_far,
                "max_height": max_height,
            }
            # UZO dwelling-unit-per-acre cap (R1=4.35, R2=10.9, R3+=uncapped).
            # SqFtProForma natively supports max_dua + ave_unit_size columns
            # and will convert DUA to an effective FAR cap.
            if parcel_max_dua is not None:
                row_dict["max_dua"] = parcel_max_dua
                row_dict["ave_unit_size"] = float(AVG_UNIT_SQFT)
            rows.append(row_dict)
            valid_idx.append(p)

        if not rows:
            return None

        df = pd.DataFrame(rows, index=[f"P{i}" for i in valid_idx])
        df.attrs["valid_positions"] = np.array(valid_idx, dtype=int)
        return df

    # ------------------------------------------------------------------
    # Run proforma developer
    # ------------------------------------------------------------------

    def _run_proforma_developer(
        self,
        proforma_df: pd.DataFrame,
        target_units: int,
        use_form: str = "residential",
        cost_escalation_factor: float = 1.0,
        zoning_rent_factor: float = 1.0,
    ) -> Tuple[float, float, np.ndarray]:
        """Run SqFtProForma feasibility + Developer.pick.

        Parameters
        ----------
        cost_escalation_factor : capacity-pressure multiplier (>1.0 = scarce).
            Applied as a rent divisor: dividing rent by the factor reduces
            effective revenue uniformly across all building types, equivalent
            to raising the cap rate.  This correctly models construction-cost
            pressure (labor/materials scarcity), unlike the prior approach of
            multiplying land_cost which disproportionately penalized land-
            intensive low-rise development.
        zoning_rent_factor : regulatory-cost multiplier (<1.0 = relaxed).
            Also applied as a rent divisor (dividing rent by 0.97 ≈ +3.1%
            effective revenue, equivalent to 3% total cost reduction).

        Both factors reduce proforma feasibility by reducing the revenue
        the proforma attributes to each building, making marginal projects
        infeasible.  They are applied separately for traceability.

        Returns (new_res_sqft, new_comm_sqft, developed_positions).
        """
        self._ensure_proforma()

        needs_copy = (cost_escalation_factor != 1.0 or zoning_rent_factor != 1.0)
        if needs_copy:
            proforma_df = proforma_df.copy()
            _rent_cols = [c for c in ("residential", "retail", "office", "industrial")
                          if c in proforma_df.columns]

            # Capacity pressure → rent divisor (uniform across building types)
            if cost_escalation_factor != 1.0 and _rent_cols:
                for _col in _rent_cols:
                    proforma_df[_col] = proforma_df[_col] / cost_escalation_factor

            # Zoning cost reduction → rent divisor (same mechanism)
            if zoning_rent_factor != 1.0 and _rent_cols:
                for _col in _rent_cols:
                    proforma_df[_col] = proforma_df[_col] / zoning_rent_factor

        try:
            feasibility = self._proforma.lookup(use_form, proforma_df, only_built=True)
        except Exception:
            return 0.0, 0.0, np.array([], dtype=int)

        if feasibility is None or len(feasibility) == 0:
            return 0.0, 0.0, np.array([], dtype=int)

        from urbansim.developer.developer import Developer
        dev = Developer(feasibility)
        try:
            buildings = dev.pick(
                form=None,
                target_units=max(target_units, 1),
                parcel_size=proforma_df["parcel_size"].reindex(
                    feasibility.index, fill_value=10000),
                ave_unit_size=pd.Series(AVG_UNIT_SQFT, index=feasibility.index),
                current_units=pd.Series(0, index=feasibility.index),
                residential=(use_form == "residential"),
            )
        except Exception:
            return 0.0, 0.0, np.array([], dtype=int)

        if buildings is None or len(buildings) == 0:
            return 0.0, 0.0, np.array([], dtype=int)

        # Per-building unit cap: largest apartment project in Lafayette MSA
        # in the last decade is ~250 units (The Hub, student housing). Typical
        # market-rate is 80-150 units.  Cap at 200 to prevent the proforma
        # from producing 400-unit towers no developer would finance
        # speculatively in a 230K metro.
        _MAX_UNITS_PER_BUILDING = 200
        if (use_form in ("residential", "mixedresidential")
                and "residential_units" in buildings.columns):
            _over = buildings["residential_units"] > _MAX_UNITS_PER_BUILDING
            if _over.any():
                buildings.loc[_over, "residential_units"] = _MAX_UNITS_PER_BUILDING
                if "net_units" in buildings.columns:
                    buildings["net_units"] = (
                        buildings["residential_units"]
                        - buildings.get("current_units", 0)
                    )
                if "building_sqft" in buildings.columns and "ave_unit_size" in buildings.columns:
                    buildings.loc[_over, "building_sqft"] = (
                        buildings.loc[_over, "residential_units"]
                        * buildings.loc[_over, "ave_unit_size"]
                    )

        # Extract delivered sqft
        new_res_sqft = 0.0
        new_comm_sqft = 0.0
        positions = []

        for _, row in buildings.iterrows():
            # Developer.pick() returns a RangeIndex with original parcel IDs
            # in the 'parcel_id' column (format "P{position_index}")
            p_str = str(row.get("parcel_id", ""))
            try:
                if p_str.startswith("P"):
                    p_idx = int(p_str[1:])
                else:
                    continue
            except (ValueError, IndexError):
                continue

            building_sqft = float(row.get(
                "building_sqft", row.get("residential_sqft", 0)))

            if use_form == "residential":
                new_res_sqft += building_sqft
            else:
                new_comm_sqft += building_sqft
            positions.append(p_idx)

            # Track built sqft for capacity depletion
            self._parcel_built_sqft[p_idx] += building_sqft

        return new_res_sqft, new_comm_sqft, np.array(positions, dtype=int)

    # ------------------------------------------------------------------
    # Vacancy-rent feedback
    # ------------------------------------------------------------------

    _NEW_STOCK_WINDOW_YEARS = 3  # parcels built within this window have sticky rents

    def _update_rents_from_vacancy(
        self,
        corridor_positions: np.ndarray,
        step_years: int = 1,
        current_year: int = 0,
    ) -> dict:
        """Delegate to standalone vacancy_rent_feedback module."""
        from src.vacancy_rent_feedback import update_rents_from_vacancy
        recently_built = (
            (current_year - self._parcel_last_delivery_year) <= self._NEW_STOCK_WINDOW_YEARS
        ) if current_year > 0 else None
        return update_rents_from_vacancy(
            self._current_rents,
            self._total_res_units,
            self._occupied_res_units,
            corridor_positions,
            step_years=step_years,
            total_comm_sqft=self._total_comm_sqft,
            occupied_comm_sqft=self._occupied_comm_sqft,
            recently_built=recently_built,
            initial_rents=self._initial_rents,
            station_area_mask=self._station_area_mask,
        )

    # ------------------------------------------------------------------
    # Initialize per-parcel development arrays
    # ------------------------------------------------------------------

    def _initialize_parcel_development_inputs(self):
        """Set up per-parcel arrays for development tracking."""
        n = len(self.parcels)
        # Base rents from market config zone characteristics
        base_rent = 18.0  # default
        self._current_rents = np.full(n, base_rent, dtype=np.float64)
        if self.zone_col and self.zone_col in self.parcels.columns:
            zone_arr = self.parcels[self.zone_col].values
            for zone_type, cfg in MARKET_CONFIG.get("zone_characteristics", {}).items():
                for z in cfg.get("zones", []):
                    mask = zone_arr == z
                    if mask.any():
                        self._current_rents[mask] = cfg.get("rent_psf_year", base_rent)

        # ACS tract-level rent override (B25064 median gross rent).
        # When data/raw/acs_tract_rents.csv exists, overrides flat zone-type
        # rents with tract-level spatial variation, clipped to 0.7x-1.5x of
        # the zone base to preserve proforma structure.
        # CSV format: GEOID (str), median_gross_rent (float, monthly $).
        _acs_path = Path("data/raw/acs_tract_rents.csv")
        if _acs_path.exists():
            try:
                _acs = pd.read_csv(_acs_path, dtype={"GEOID": str})
                _tract_rent_map = dict(zip(_acs["GEOID"], _acs["median_gross_rent"]))
                # Spatial join: assign tract to each parcel
                if "tract_geoid" in self.parcels.columns:
                    _tract_ids = self.parcels["tract_geoid"].astype(str).values
                else:
                    # On-the-fly spatial join with TIGER tract boundaries
                    _tract_shp = Path("data/raw/tl_2024_18157_tract.shp")
                    _tract_ids = None
                    if _tract_shp.exists():
                        _tracts = gpd.read_file(_tract_shp)[["GEOID", "geometry"]]
                        _tracts = _tracts.to_crs(self.parcels.crs)
                        _joined = gpd.sjoin(
                            self.parcels[["geometry"]],
                            _tracts, how="left", predicate="intersects",
                        )
                        # De-duplicate: keep first match per parcel
                        _joined = _joined[~_joined.index.duplicated(keep="first")]
                        _tract_ids = _joined.reindex(self.parcels.index)["GEOID"].astype(str).values
                if _tract_ids is not None:
                    # Subtract estimated utility allowance ($150/mo) for
                    # contract rent approximation (HUD Tippecanoe County)
                    _UTILITY_ALLOWANCE = 150.0
                    _n_override = 0
                    for i in range(n):
                        _monthly = _tract_rent_map.get(str(_tract_ids[i]))
                        if _monthly is None or _monthly <= 0:
                            continue
                        _contract = max(_monthly - _UTILITY_ALLOWANCE, _monthly * 0.80)
                        _annual_psf = _contract * 12.0 / AVG_UNIT_SQFT
                        _zone_base = self._current_rents[i]
                        self._current_rents[i] = float(np.clip(
                            _annual_psf, _zone_base * 0.7, _zone_base * 1.5))
                        _n_override += 1
                    if _n_override > 0:
                        logger.info(f"  ACS tract rents: overrode {_n_override:,} parcels")
            except Exception as e:
                logger.warning(f"  ACS tract rent loading failed: {e}")

        # Snapshot initial rents — used as floor for transit-adjacent parcels.
        # Empirical evidence: rents within 800m of transit stations never fall
        # below ~82% of initial values even during vacancy spikes (Singer 2025,
        # Jiang et al. 2020).  Transit proximity creates captive demand that
        # prevents the full rent collapse the vacancy model would otherwise produce.
        self._initial_rents = self._current_rents.copy()
        # Metro-average rent for rent-responsive migration adjustment.
        _positive_rents = self._current_rents[self._current_rents > 0]
        self._mean_metro_rent = float(np.mean(_positive_rents)) if len(_positive_rents) > 0 else 18.0

        self._total_res_units = np.zeros(n, dtype=np.float64)
        self._occupied_res_units = np.zeros(n, dtype=np.float64)
        self._total_comm_sqft = np.zeros(n, dtype=np.float64)
        self._occupied_comm_sqft = np.zeros(n, dtype=np.float64)

        # Seed from existing building stock (CurImpAV as proxy)
        if "CurImpAV" in self.parcels.columns:
            imp_av = self.parcels["CurImpAV"].fillna(0).values.astype(float)
            # Rough: $100K improvement ≈ 1 unit
            existing_units = np.clip(imp_av / 100_000.0, 0, 500)
            self._total_res_units = existing_units.copy()
            # Assume at target vacancy
            from src.relocation_model import TARGET_VAC_RES
            self._occupied_res_units = existing_units * (1.0 - TARGET_VAC_RES)

        self._pending_deliveries = {}
        self._pending_comm_demand: Dict[str, float] = {}
        self._cumulative_corridor_pop = {}
        self._cumulative_corridor_jobs = {}

        # Per-parcel delivery year tracking (for new-stock rent stickiness)
        self._parcel_last_delivery_year = np.full(n, -999, dtype=np.int32)

        # Cross-year capacity tracking: cumulative built sqft per parcel
        self._parcel_built_sqft = np.zeros(n, dtype=np.float64)

        # Parcel-level developable mask (exempt parcels excluded)
        self._parcel_developable_mask = np.ones(n, dtype=bool)
        # Tenure classification from Indiana DLGF PriorPropClass:
        #   Homestead (owner-occupied):
        #     - 1xx codes (SF dwellings, explicitly classified)
        #     - 5xx codes with CurImpAV > $50k (residential land with a house;
        #       Indiana assessor data often classifies parcels by land type rather
        #       than improvement type — 50,311 of ~60,919 5xx parcels in
        #       Tippecanoe County are SF homes, median AV $238k)
        #   Rental (non-homestead residential): 4xx codes (apartments, MF)
        #   All other developed parcels default to rental for new TOD
        #   construction, since station-area zoning (R3/R4/MU) produces MF.
        _IMP_AV_THRESHOLD = 50_000  # min improvement AV to classify 5xx as homestead
        self._parcel_is_homestead = np.zeros(n, dtype=bool)
        if "PriorPropClass" in self.parcels.columns:
            ppc = self.parcels["PriorPropClass"].astype(str)
            _exempt_mask = ppc.str.startswith("6").to_numpy()
            n_exempt = int(_exempt_mask.sum())
            if n_exempt > 0:
                logger.info(f"  Exempt parcels (PropClass 6xx): {n_exempt} marked undevelopable")
            self._parcel_developable_mask &= ~_exempt_mask
            # 1xx = single-family dwellings (explicitly homestead)
            _is_1xx = ppc.str.startswith("1").to_numpy()
            # 5xx with significant improvements = SF homes on "land" parcels
            _is_5xx = ppc.str.startswith("5").to_numpy()
            _imp_av = np.zeros(n, dtype=float)
            if "CurImpAV" in self.parcels.columns:
                _imp_av = self.parcels["CurImpAV"].fillna(0).values.astype(float)
            _is_5xx_with_house = _is_5xx & (_imp_av >= _IMP_AV_THRESHOLD)
            self._parcel_is_homestead = _is_1xx | _is_5xx_with_house
            # 4xx = apartments/MF residential (non-homestead)
            _is_rental = ppc.str.startswith("4").to_numpy()
            # Compute empirical homestead share by assessed value
            _res_mask = self._parcel_is_homestead | _is_rental
            if _res_mask.any() and "CurTotAV" in self.parcels.columns:
                _av = self.parcels["CurTotAV"].fillna(0).values.astype(float)
                _hs_av = float((_av * self._parcel_is_homestead).sum())
                _res_av = float((_av * _res_mask).sum())
                self._empirical_homestead_share = (
                    _hs_av / _res_av if _res_av > 0 else 0.0)
                _n_hs = int(self._parcel_is_homestead.sum())
                _n_1xx = int(_is_1xx.sum())
                _n_5xx_h = int(_is_5xx_with_house.sum())
                logger.info(
                    f"  Homestead parcels: {_n_hs:,} "
                    f"({_n_1xx:,} explicit 1xx + {_n_5xx_h:,} inferred 5xx)")
                logger.info(
                    f"  Empirical homestead share (by AV): "
                    f"{self._empirical_homestead_share:.3f}")
            else:
                self._empirical_homestead_share = 0.0
        else:
            self._empirical_homestead_share = 0.0

        # Cache parcel land cost per sqft and improvement AV for demolition
        self._parcel_sqft_arr = np.zeros(n, dtype=np.float64)
        self._parcel_imp_av = np.zeros(n, dtype=np.float64)
        for col in ["parcel_sqft", "Shape_Area"]:
            if col in self.parcels.columns:
                self._parcel_sqft_arr = self.parcels[col].fillna(0).values.astype(float)
                break
        if "CurImpAV" in self.parcels.columns:
            self._parcel_imp_av = self.parcels["CurImpAV"].fillna(0).values.astype(float)

        # Commute times for relocation MNL
        self._commute_times = self._precompute_commute_times()

    # ------------------------------------------------------------------
    # OD cache
    # ------------------------------------------------------------------

    def _compute_cache_hash(self) -> str:
        """Compute a hash for cache invalidation based on data dimensions."""
        import hashlib
        h = hashlib.md5()
        h.update(str(len(self.parcels)).encode())
        h.update(str(len(self.corridors)).encode())
        h.update(str(len(self.od_flows)).encode())
        return h.hexdigest()[:12]

    def _load_or_build_od_cache(self):
        """Load or build the LODES OD parcel-to-parcel cache."""
        cache_path = Path("data/processed/od_cache.pkl")
        cache_hash = self._compute_cache_hash()
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("hash") == cache_hash:
                    logger.info("  OD cache loaded from disk.")
                    return cached.get("data")
            except Exception:
                pass

        # Build fresh cache (just store the OD flows for fast reuse)
        logger.info("  Building OD cache...")
        od_data = self.od_flows
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"hash": cache_hash, "data": od_data}, f)
        except Exception:
            pass
        return od_data

    # ------------------------------------------------------------------
    # Corridor spatial cache
    # ------------------------------------------------------------------

    def _precompute_corridor_spatial_cache(self):
        """Precompute per-corridor spatial data (distances, weights, stop projections).

        Builds walk-zone and feeder-zone weight vectors stored in sparse
        (index, value) format for each corridor.  Also caches stop projections,
        effective APM speed, destination coverage, etc.
        """
        from pyproj import Transformer
        try:
            from scripts.generate_improved_ridership import (
                compute_effective_apm_speed,
                compute_effective_brt_speed,
            )
        except ImportError:
            compute_effective_apm_speed = lambda length_km, n_stops, **kw: 30.0
            compute_effective_brt_speed = lambda length_km, n_stops: 20.0

        _to_proj = Transformer.from_crs("EPSG:4326", PROJECT_CRS, always_xy=True)

        # Project parcel centroids to meters
        pid_arr, parcel_xy_3857, pid_to_idx = self.parcel_cache
        n_parcels = len(self.parcels)

        # Project parcels to EPSG:2965 (feet) then convert to meters
        if not hasattr(self, "_parcel_xy_m"):
            parcels_proj = self.parcels.to_crs(PROJECT_CRS)
            centroids = parcels_proj.geometry.centroid
            self._parcel_xy_m = np.column_stack([
                centroids.x.values * US_SURVEY_FT_TO_M,
                centroids.y.values * US_SURVEY_FT_TO_M,
            ])

        parcel_xy_m = self._parcel_xy_m

        # Accumulate station-area mask across all corridors (for transit rent floor)
        self._station_area_mask = np.zeros(n_parcels, dtype=bool)

        for cid, meta in self._corridor_meta.items():
            # Use DP-selected station coordinates from Stage 1, not
            # interpolated positions along the display LineString.
            stops_wgs = meta.get("stop_coords", [])
            if not stops_wgs or len(stops_wgs) < 2:
                # Fallback: interpolate from geometry (legacy path)
                corridor = self._corridor_rows[cid]
                stops_wgs = extract_corridor_stops(corridor.geometry, meta["n_stops"])
            if not stops_wgs:
                self._corridor_spatial_cache[cid] = {}
                continue

            # Project stops to meters
            stops_proj = []
            for lon, lat in stops_wgs:
                x, y = _to_proj.transform(lon, lat)
                stops_proj.append((x * US_SURVEY_FT_TO_M, y * US_SURVEY_FT_TO_M))
            stops_proj = np.array(stops_proj)

            # Distance from each parcel to nearest stop (meters)
            stop_tree = cKDTree(stops_proj)
            parcel_dist, parcel_stop_idx = stop_tree.query(parcel_xy_m, k=1)

            # Walk-zone weights (exponential decay within WALK_CATCHMENT_M)
            walk_mask = parcel_dist <= WALK_CATCHMENT_M
            self._station_area_mask |= walk_mask  # accumulate for transit rent floor
            walk_indices = np.where(walk_mask)[0]
            if len(walk_indices) > 0:
                walk_values = np.exp(-DECAY_BETA * parcel_dist[walk_indices])
            else:
                walk_values = np.array([])

            # Feeder-zone weights (between walk catchment and feeder catchment)
            feeder_mask = (parcel_dist > WALK_CATCHMENT_M) & (parcel_dist <= FEEDER_CATCHMENT_M)
            feeder_indices = np.where(feeder_mask)[0]
            if len(feeder_indices) > 0:
                # Decay from the feeder inner boundary
                feeder_dist_beyond = parcel_dist[feeder_indices] - WALK_CATCHMENT_M
                feeder_beta = 0.0005  # half at ~1400m beyond walk zone
                feeder_values = np.exp(-feeder_beta * feeder_dist_beyond)
            else:
                feeder_values = np.array([])

            # Sector assignment for feeder-zone parcels (8 sectors)
            feeder_sectors = np.zeros(len(feeder_indices), dtype=int)
            if len(feeder_indices) > 0:
                # Compute angle from corridor centroid to each feeder parcel
                corr_center = stops_proj.mean(axis=0)
                dx = parcel_xy_m[feeder_indices, 0] - corr_center[0]
                dy = parcel_xy_m[feeder_indices, 1] - corr_center[1]
                angles = np.arctan2(dy, dx)  # -pi to pi
                angles = (angles + 2 * np.pi) % (2 * np.pi)  # 0 to 2pi
                feeder_sectors = np.clip((angles / (2 * np.pi / 8)).astype(int), 0, 7)

            # Stop distance matrix (N_stops × N_stops)
            n_stops = len(stops_proj)
            stop_dist_matrix = np.zeros((n_stops, n_stops))
            for i in range(n_stops):
                for j in range(n_stops):
                    stop_dist_matrix[i, j] = np.sqrt(
                        ((stops_proj[i] - stops_proj[j]) ** 2).sum())

            # Effective speed (includes curve delay from Stage 1 if available)
            length_km = meta["length_km"]
            _curve_delay = float(meta.get("curve_delay_s", 0.0))
            if self._transit_mode_name == "brt":
                apm_speed = compute_effective_brt_speed(length_km, n_stops)
            else:
                apm_speed = compute_effective_apm_speed(
                    length_km, n_stops, curve_delay_s=_curve_delay)

            # APM circuity (corridor route length / straight-line distance)
            if n_stops >= 2:
                straight_dist = np.sqrt(((stops_proj[0] - stops_proj[-1]) ** 2).sum())
                route_dist = sum(
                    np.sqrt(((stops_proj[i] - stops_proj[i + 1]) ** 2).sum())
                    for i in range(n_stops - 1)
                )
                apm_circuity = route_dist / max(straight_dist, 1.0)
            else:
                apm_circuity = 1.0

            # Destination coverage for student trips
            dest_coverage = self._compute_destination_coverage(stops_proj)

            self._corridor_spatial_cache[cid] = {
                "parcel_dist": parcel_dist,
                "parcel_stop_idx": parcel_stop_idx,
                "w_walk_idx": walk_indices,
                "w_walk_val": walk_values,
                "w5000_idx": feeder_indices,
                "w5000_val": feeder_values,
                "w5000_sector": feeder_sectors,
                "stops_proj": stops_proj,
                "stop_distance_matrix": stop_dist_matrix,
                "corridor_length_km": length_km,
                "apm_effective_speed": apm_speed,
                "apm_circuity": apm_circuity,
                "dest_coverage": dest_coverage,
            }

        logger.info(f"  Spatial cache built for {len(self._corridor_spatial_cache)} corridors.")

    # ------------------------------------------------------------------
    # Parking costs
    # ------------------------------------------------------------------

    def _build_parking_costs(self):
        """Build per-parcel parking cost array.

        Campus parcels: $0.40/trip (Purdue permit amortized)
        Downtown parcels: $4.00/trip
        Suburban: $0.00
        """
        n = len(self.parcels)
        self._parking_costs = np.zeros(n, dtype=np.float64)

        if self.zone_col and self.zone_col in self.parcels.columns:
            zone_arr = self.parcels[self.zone_col].values
            # Downtown zones
            downtown_zones = {"CB", "CBW", "NBU", "PDCC", "PDMX"}
            for z in downtown_zones:
                mask = zone_arr == z
                self._parking_costs[mask] = 4.0

        # Campus parcels (identified by institutional weights later)
        # Will be updated after _build_institutional_weights
        self._parking_campus_initialized = False

    def _update_parking_scarcity(
        self,
        corridor_positions: np.ndarray,
        developed_fraction: float,
    ):
        """Update parking costs based on development intensity.

        As TOD densifies, surface parking redevelops and remaining supply tightens.
        """
        if self._parking_costs is None or len(corridor_positions) == 0:
            return
        pos = corridor_positions.astype(int)
        for p in pos:
            zone = ""
            if self.zone_col and self.zone_col in self.parcels.columns:
                zone = str(self.parcels.iloc[p].get(self.zone_col, ""))
            if zone in {"CB", "CBW", "NBU", "PDCC", "PDMX"}:
                elasticity = PARKING_SCARCITY_ELASTICITY_DOWNTOWN
            elif (self._institutional_weights is not None and
                  p < len(self._institutional_weights) and
                  self._institutional_weights[p] >= 3.0):
                elasticity = PARKING_SCARCITY_ELASTICITY_CAMPUS
            else:
                elasticity = PARKING_SCARCITY_ELASTICITY_SUBURBAN
            scarcity_mult = 1.0 + elasticity * developed_fraction
            self._parking_costs[p] *= scarcity_mult

    # ------------------------------------------------------------------
    # Institutional weights
    # ------------------------------------------------------------------

    def _build_institutional_weights(self):
        """Build per-parcel institutional weight array.

        Identifies campus and institutional parcels by zone code and
        proximity to Purdue campus center.  Weights:
          0.0 = non-institutional
          0.5 = campus-adjacent (within 800m of campus center)
          2.0 = student housing zone
          3.0 = campus fringe buildings
          4.0 = core campus (university buildings)
        """
        n = len(self.parcels)
        self._institutional_weights = np.zeros(n, dtype=np.float64)

        # Campus proximity
        from pyproj import Transformer
        _to_proj = Transformer.from_crs("EPSG:4326", PROJECT_CRS, always_xy=True)
        cx, cy = _to_proj.transform(self._CAMPUS_LON, self._CAMPUS_LAT)
        cx *= US_SURVEY_FT_TO_M
        cy *= US_SURVEY_FT_TO_M

        if hasattr(self, "_parcel_xy_m"):
            parcel_xy_m = self._parcel_xy_m
        else:
            return

        campus_dists = np.sqrt(((parcel_xy_m - [cx, cy]) ** 2).sum(axis=1))

        # Core campus: within 500m
        core_mask = campus_dists <= 500
        self._institutional_weights[core_mask] = 4.0

        # Campus fringe: 500-1000m
        fringe_mask = (campus_dists > 500) & (campus_dists <= 1000)
        self._institutional_weights[fringe_mask] = 3.0

        # Student housing: 1000-1500m
        housing_mask = (campus_dists > 1000) & (campus_dists <= 1500)
        self._institutional_weights[housing_mask] = 2.0

        # Campus-adjacent: 1500-2000m
        adjacent_mask = (campus_dists > 1500) & (campus_dists <= 2000)
        self._institutional_weights[adjacent_mask] = 0.5

        # Zone-based overrides
        if self.zone_col and self.zone_col in self.parcels.columns:
            zone_arr = self.parcels[self.zone_col].values
            # PD zones near campus get higher weight
            for z in ["PDRS", "PDNR", "R3U", "R4W"]:
                z_mask = (zone_arr == z) & (campus_dists <= 2000)
                current = self._institutional_weights[z_mask]
                self._institutional_weights[z_mask] = np.maximum(current, 2.0)

        # Update campus parking costs
        if self._parking_costs is not None:
            campus_park_mask = self._institutional_weights >= 3.0
            self._parking_costs[campus_park_mask] = np.maximum(
                self._parking_costs[campus_park_mask], 0.40)

    def _build_weights_from_buildings(self):
        """Build institutional weights using building footprint data if available.

        Falls back to proximity-based method if building data unavailable.
        Uses CurImpAV (improvement assessed value) as a proxy for building
        intensity, combined with zone classification.
        """
        n = len(self.parcels)
        if "CurImpAV" not in self.parcels.columns:
            return self._build_weights_from_proximity()

        imp_av = self.parcels["CurImpAV"].fillna(0).values.astype(float)

        # Campus proximity
        from pyproj import Transformer
        _to_proj = Transformer.from_crs("EPSG:4326", PROJECT_CRS, always_xy=True)
        cx, cy = _to_proj.transform(self._CAMPUS_LON, self._CAMPUS_LAT)
        cx *= US_SURVEY_FT_TO_M
        cy *= US_SURVEY_FT_TO_M

        if not hasattr(self, "_parcel_xy_m"):
            return
        parcel_xy_m = self._parcel_xy_m
        campus_dists = np.sqrt(((parcel_xy_m - [cx, cy]) ** 2).sum(axis=1))

        weights = np.zeros(n, dtype=np.float64)

        # High improvement value near campus = institutional buildings
        high_imp_campus = (imp_av > 1_000_000) & (campus_dists <= 800)
        weights[high_imp_campus] = 4.0

        # Medium improvement near campus = student housing / academic support
        med_imp_campus = (imp_av > 200_000) & (campus_dists <= 1200) & ~high_imp_campus
        weights[med_imp_campus] = 3.0

        # Any building near campus
        near_campus = (campus_dists <= 1500) & (weights < 2.0)
        weights[near_campus] = np.maximum(weights[near_campus], 2.0)

        # Adjacent
        adjacent = (campus_dists > 1500) & (campus_dists <= 2500) & (weights < 0.5)
        weights[adjacent] = 0.5

        # Zone overrides
        if self.zone_col and self.zone_col in self.parcels.columns:
            zone_arr = self.parcels[self.zone_col].values
            for z in ["PDRS", "PDNR", "R3U", "R4W"]:
                z_mask = (zone_arr == z) & (campus_dists <= 2000)
                weights[z_mask] = np.maximum(weights[z_mask], 2.0)

        self._institutional_weights = weights

    def _build_weights_from_proximity(self):
        """Fallback: build institutional weights from distance only."""
        # Already implemented in _build_institutional_weights
        pass

    # ------------------------------------------------------------------
    # Bus restructure context initialization
    # ------------------------------------------------------------------

    def _initialize_bus_restructure_context(self):
        """Load GTFS data and competitiveness metrics for bus restructuring."""
        try:
            self._gtfs_competitiveness = load_gtfs_competitiveness_summary(
                gtfs_dir=self.gtfs_dir or Path("data/raw/CityBus2025"),
            )
            logger.info("  GTFS competitiveness data loaded.")
        except Exception as e:
            logger.warning(f"  GTFS competitiveness data unavailable: {e}")
            self._gtfs_competitiveness = None

        try:
            gtfs_dir = Path("data/raw/CityBus2025")
            if gtfs_dir.exists():
                self._bus_routes = load_bus_routes_from_gtfs(gtfs_dir)
                logger.info(f"  Loaded {len(self._bus_routes)} bus routes from GTFS.")
            else:
                self._bus_routes = None
                logger.info("  GTFS directory not found; bus restructuring uses defaults.")
        except Exception as e:
            self._bus_routes = None
            logger.warning(f"  Failed to load GTFS bus routes: {e}")

    def _initialize_dynamic_bus_network(self, cid: str, year: int):
        """Initialize or update dynamic bus network for a corridor.

        Classifies bus routes relative to the corridor and computes initial
        service profiles (headways by period).
        """
        if self._bus_routes is None:
            return

        spatial = self._corridor_spatial_cache.get(cid, {})
        stops_proj = spatial.get("stops_proj")
        if stops_proj is None:
            return

        try:
            # Classify routes as parallel/feeder/independent
            corridor_row = self._corridor_rows[cid]
            classifications = classify_routes_for_corridor(
                self._bus_routes, corridor_row, stops_proj)

            # Build bus network model for this corridor
            network = build_corridor_bus_network(
                self._bus_routes, classifications,
                corridor_row, year)

            if network is not None:
                # Extract service profiles
                if hasattr(network, "parallel_profile"):
                    self._bus_service_profiles[cid] = network.parallel_profile
                if hasattr(network, "feeder_profile"):
                    self._feeder_service_profiles[cid] = network.feeder_profile
        except Exception as e:
            logger.debug(f"  Dynamic bus network init failed for {cid}: {e}")

    # ------------------------------------------------------------------
    # Snapshot/restore for corridor independence
    # ------------------------------------------------------------------

    def _snapshot_baseline(self) -> dict:
        """Capture mutable state before corridor evaluation."""
        snap = {
            "pop": self.parcels[self.pop_col].values.copy(),
            "jobs": self.parcels[self.jobs_col].values.copy() if self.jobs_col else None,
            "rents": self._current_rents.copy() if self._current_rents is not None else None,
            "res_units": self._total_res_units.copy() if self._total_res_units is not None else None,
            "occ_units": self._occupied_res_units.copy() if self._occupied_res_units is not None else None,
            "comm_sqft": self._total_comm_sqft.copy() if self._total_comm_sqft is not None else None,
            "occ_comm": self._occupied_comm_sqft.copy() if self._occupied_comm_sqft is not None else None,
            "parking": self._parking_costs.copy() if self._parking_costs is not None else None,
            "built_sqft": self._parcel_built_sqft.copy(),
            "delivery_year": self._parcel_last_delivery_year.copy(),
            "commute_times": self._commute_times.copy() if self._commute_times is not None else None,
            # DemandDrivenDevelopmentModel metro state (corridor independence)
            "metro_population": self.demand_dev_model.growth.metro_population,
            "metro_jobs": self.demand_dev_model.growth.metro_jobs,
        }
        return snap

    def _restore_baseline(self, snap: dict):
        """Restore mutable state from snapshot (corridor independence)."""
        self.parcels[self.pop_col] = snap["pop"]
        if self.jobs_col and snap["jobs"] is not None:
            self.parcels[self.jobs_col] = snap["jobs"]
        if snap["rents"] is not None:
            self._current_rents[:] = snap["rents"]
        if snap["res_units"] is not None:
            self._total_res_units[:] = snap["res_units"]
        if snap["occ_units"] is not None:
            self._occupied_res_units[:] = snap["occ_units"]
        if snap["comm_sqft"] is not None:
            self._total_comm_sqft[:] = snap["comm_sqft"]
        if snap["occ_comm"] is not None:
            self._occupied_comm_sqft[:] = snap["occ_comm"]
        if snap["parking"] is not None:
            self._parking_costs[:] = snap["parking"]

        # Restore cross-year capacity tracking
        if snap.get("built_sqft") is not None:
            self._parcel_built_sqft[:] = snap["built_sqft"]
        if snap.get("delivery_year") is not None:
            self._parcel_last_delivery_year[:] = snap["delivery_year"]

        # Restore commute times
        if snap.get("commute_times") is not None:
            self._commute_times[:] = snap["commute_times"]

        # Reset per-corridor dynamic state
        for cid in self._corridor_meta:
            self._bus_headways[cid] = BASE_BUS_HEADWAY
            self._feeder_headways[cid] = BASE_BUS_HEADWAY
            self._apm_headways[cid] = APM_HEADWAY_MIN
            self._bus_speeds[cid] = BUS_SPEED_KPH
            self._feeder_coverage[cid] = 0.15
            self._restructure_pressure[cid] = 0.0
            self._feeder_budget_utilization[cid] = 0.0
            self._transfer_walk_min[cid] = 0.0
            self._sector_coverage[cid] = None
            self._tsp_speed_factors[cid] = 1.0
            self._bus_service_profiles[cid] = None
            self._feeder_service_profiles[cid] = None
            self._directional_split[cid] = 0.60

        # Reset base catchment tracking so each corridor gets fresh baselines
        self._base_catchment_pop.clear()
        self._base_catchment_jobs.clear()
        self._base_feeder_pop_catch.clear()
        self._congestion_factor = 1.0
        self._last_restructure_ridership.clear()

        # Reset cumulative development tracking
        self._cumulative_corridor_pop.clear()
        self._cumulative_corridor_jobs.clear()

        # Reset pending deliveries and commercial demand accumulator
        self._pending_deliveries.clear()
        self._pending_comm_demand.clear()

        # Clear cached bus network state
        if hasattr(self, '_cached_bus_network'):
            self._cached_bus_network = None

        # Reset DemandDrivenDevelopmentModel per-corridor state
        self.demand_dev_model._rent_multiplier = 1.0
        self.demand_dev_model._corridor_res_units.clear()
        self.demand_dev_model._corridor_comm_sqft.clear()
        self.demand_dev_model._corridor_households.clear()
        self.demand_dev_model._corridor_jobs.clear()
        self.demand_dev_model._cumulative_corridor_pop.clear()
        self.demand_dev_model._segment_supply.clear()
        self.demand_dev_model._segment_rents.clear()
        self.demand_dev_model._baseline_catchment.clear()
        # Restore metro base for compounding (corridor independence)
        if snap.get("metro_population") is not None:
            self.demand_dev_model.growth.metro_population = snap["metro_population"]
        if snap.get("metro_jobs") is not None:
            self.demand_dev_model.growth.metro_jobs = snap["metro_jobs"]

        # Reset warning flags
        self._d1_warned = False
        self._d4_warned = False

    # ------------------------------------------------------------------
    # Single corridor evaluation (the main per-corridor loop)
    # ------------------------------------------------------------------

    def _run_single_corridor(
        self,
        corridor_id: str,
        baseline: dict,
    ) -> Tuple[List[dict], List[dict]]:
        """Run the full time-step loop for one corridor.

        Returns (results_rows, diagnostics_rows).
        """
        self._restore_baseline(baseline)

        results_rows: List[dict] = []
        diagnostics_rows: List[dict] = []

        prev_metrics: Optional[Dict[str, float]] = None
        converged_streak = 0
        divergent_streak = 0

        for ti, year in enumerate(self.time_steps):
            step_years = (
                self.time_steps[ti] - self.time_steps[ti - 1]
                if ti > 0 else 1
            )

            # A. Ridership
            ridership = self._compute_ridership(year, active_corridor_id=corridor_id)
            rdata = ridership.get(corridor_id, {})
            daily_riders = rdata.get("daily_riders", 0.0)

            # B. Accessibility
            accessibility = self._compute_accessibility(ridership)
            acc_arr = accessibility.get(corridor_id, np.zeros(len(self.parcels)))

            # C. Development
            dev_result = self._run_development_model(
                year, corridor_id, acc_arr, daily_riders, step_years)

            # D. Update population/jobs
            self._update_pop_jobs(corridor_id, dev_result, step_years)

            # E. Bus restructuring (with within-year iteration)
            # Restructuring changes headways → which changes ridership →
            # which changes optimal restructuring.  2-3 passes catches
            # the main feedback without full convergence.
            _BUS_INNER_ITERS = 3
            _BUS_CONVERGENCE_TOL = 0.02  # 2% ridership change = converged
            if self.bus_restructure and year > 0:
                _prev_riders = daily_riders
                for _bi in range(_BUS_INNER_ITERS):
                    self._restructure_bus(corridor_id, ridership, year)
                    if _bi < _BUS_INNER_ITERS - 1:
                        # Re-compute ridership with updated bus headways
                        ridership = self._compute_ridership(year, active_corridor_id=corridor_id)
                        _new_riders = ridership.get(corridor_id, {}).get("daily_riders", 0.0)
                        if _prev_riders > 0:
                            _delta = abs(_new_riders - _prev_riders) / max(_prev_riders, 1.0)
                            if _delta < _BUS_CONVERGENCE_TOL:
                                break
                        _prev_riders = _new_riders
                # Update daily_riders for the result row from final iteration
                rdata = ridership.get(corridor_id, {})
                daily_riders = rdata.get("daily_riders", daily_riders)

            # F. Deliver pending units
            self._deliver_pending(year)

            # Build result row
            new_pop = dev_result.get("new_pop", 0.0)
            new_jobs = dev_result.get("new_jobs", 0.0)
            new_units = dev_result.get("new_units", 0.0)

            row = {
                "year": year,
                "corridor_id": corridor_id,
                "daily_riders": daily_riders,
                "daily_riders_annual_avg": rdata.get("daily_riders_annual_avg", daily_riders),
                "base_riders": rdata.get("base_riders", 0.0),
                "awareness": rdata.get("awareness", 0.0),
                "apm_mode_share": rdata.get("apm_share", 0.0),
                "pop_walk_catch": rdata.get("pop_catch", 0.0),
                "jobs_walk_catch": rdata.get("jobs_catch", 0.0),
                "feeder_pop_catch": rdata.get("feeder_pop_catch", 0.0),
                "campus_pop_catch": rdata.get("campus_pop_catch", 0.0),
                "campus_alignment": rdata.get("campus_alignment", 0.0),
                "student_apm_share": rdata.get("student_apm_share", 0.0),
                "student_apm_daily": rdata.get("student_apm_daily", 0.0),
                "work_commute_daily": rdata.get("work_commute_daily", 0.0),
                "local_nonwork_daily": rdata.get("local_nonwork_daily", 0.0),
                "campus_daily": rdata.get("campus_daily", 0.0),
                "destination_daily": rdata.get("destination_daily", 0.0),
                "generator_daily": rdata.get("generator_daily", 0.0),
                "induced_daily": rdata.get("induced_daily", 0.0),
                "latent_daily": rdata.get("latent_daily", 0.0),
                "non_campus_daily": rdata.get("non_campus_daily", 0.0),
                "bus_headway": self._bus_headways.get(corridor_id, BASE_BUS_HEADWAY),
                "feeder_headway": self._feeder_headways.get(corridor_id, BASE_BUS_HEADWAY),
                "apm_headway": self._apm_headways.get(corridor_id, APM_HEADWAY_MIN),
                "feeder_coverage": self._feeder_coverage.get(corridor_id, 0.15),
                "bus_restructure_pressure": self._restructure_pressure.get(corridor_id, 0.0),
                "feeder_budget_utilization": self._feeder_budget_utilization.get(corridor_id, 0.0),
                "new_units": new_units,
                "new_comm_sqft": dev_result.get("new_comm_sqft", 0.0),
                "new_homestead_sqft": dev_result.get("new_homestead_sqft", 0.0),
                "new_rental_sqft": dev_result.get("new_rental_sqft", 0.0),
                "new_pop": new_pop,
                "new_jobs": new_jobs,
                "riders_SE01": rdata.get("riders_SE01", 0.0),
                "riders_SE02": rdata.get("riders_SE02", 0.0),
                "riders_SE03": rdata.get("riders_SE03", 0.0),
                "low_income_access_ratio": rdata.get("low_income_access_ratio", 0.0),
            }
            results_rows.append(row)

            # G. Convergence
            current_metrics = {
                "daily_riders": daily_riders,
                "new_pop": new_pop,
                "new_jobs": new_jobs,
            }
            conv = evaluate_convergence(
                current_metrics, prev_metrics,
                self._ridership_tol, self._development_tol,
                self._convergence_floor,
            )
            prev_metrics = current_metrics

            diag = {
                "year": year,
                "corridor_id": corridor_id,
                "daily_riders": daily_riders,
                "new_pop": new_pop,
                "new_jobs": new_jobs,
            }
            diag.update(conv)
            diagnostics_rows.append(diag)

            # Check convergence-based stop (only in adaptive mode)
            if self.adaptive_stop and ti > 0:
                summary = summarize_year_convergence(
                    {corridor_id: conv}, self._divergence_threshold)
                stop_result = evaluate_stop_conditions(
                    year_all_converged=summary["all_converged"],
                    year_divergent=summary["divergence_flag"],
                    converged_streak=converged_streak,
                    divergent_streak=divergent_streak,
                    adaptive_stop=self.adaptive_stop,
                    consecutive_converged_steps=self._consecutive_converged_steps,
                    stop_on_divergence=self._stop_on_divergence,
                    consecutive_divergent_steps=self._consecutive_divergent_steps,
                )
                converged_streak = stop_result["converged_streak"]
                divergent_streak = stop_result["divergent_streak"]
                if stop_result["stop_triggered"]:
                    logger.info(
                        f"  {corridor_id}: early stop at year {year} "
                        f"({stop_result['stop_reason']})")
                    break

        return results_rows, diagnostics_rows

    # ------------------------------------------------------------------
    # Public run() entry point
    # ------------------------------------------------------------------

    def run(
        self,
        parallel: bool = False,
        time_steps: Optional[Tuple[int, ...]] = None,
    ) -> pd.DataFrame:
        """Run the full equilibrium loop across all corridors.

        Parameters
        ----------
        parallel : use multiprocessing for corridor evaluation
        time_steps : override time steps (None = use self.time_steps)
        """
        if time_steps is not None:
            self.time_steps = time_steps

        if len(self.time_steps) > self._max_time_steps:
            raise ValueError(
                f"time_steps has {len(self.time_steps)} entries, "
                f"exceeds max_time_steps={self._max_time_steps}"
            )

        baseline = self._snapshot_baseline()

        if parallel:
            self._run_parallel(baseline)
        else:
            self._run_serial(baseline)

        return self._compile_results()

    def _run_serial(self, baseline: dict):
        """Run corridors sequentially."""
        cid_list = list(self._corridor_meta.keys())
        for ci, cid in enumerate(cid_list):
            logger.info(f"\n--- Corridor {cid} ({ci+1}/{len(cid_list)}) ---")
            results, diagnostics = self._run_single_corridor(cid, baseline)
            self._results.extend(results)
            self._diagnostics.extend(diagnostics)

    def _run_parallel(self, baseline: dict):
        """Run corridors in parallel using ProcessPoolExecutor."""
        cid_list = list(self._corridor_meta.keys())
        n_workers = min(os.cpu_count() or 1, len(cid_list))

        # Memory check: limit workers to avoid OOM
        try:
            import psutil
            avail_mb = psutil.virtual_memory().available / (1024 * 1024)
            per_worker_mb = 500  # estimate
            max_from_mem = max(1, int(avail_mb / per_worker_mb))
            n_workers = min(n_workers, max_from_mem)
        except ImportError:
            n_workers = min(n_workers, 4)

        # Batch corridors across workers
        batch_size = max(1, len(cid_list) // n_workers)
        batches = []
        for i in range(0, len(cid_list), batch_size):
            batches.append(cid_list[i:i + batch_size])

        logger.info(f"Running {len(cid_list)} corridors across {n_workers} workers")

        model_bytes = pickle.dumps(self)
        baseline_bytes = pickle.dumps(baseline)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_corridor_worker,
            initargs=(model_bytes,),
        ) as executor:
            futures = []
            for batch in batches:
                futures.append(executor.submit(_run_corridor_batch, (batch, baseline_bytes)))

            for future in futures:
                try:
                    results, diagnostics = future.result()
                    self._results.extend(results)
                    self._diagnostics.extend(diagnostics)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
                    traceback.print_exc()

    # ------------------------------------------------------------------
    # Capacity accounting
    # ------------------------------------------------------------------

    def _consume_parcel_capacity(
        self,
        pos: int,
        res_sqft: float,
        comm_sqft: float,
    ) -> dict:
        """Consume development capacity on a single parcel.

        Returns dict with delivered_sqft and remaining capacity.
        """
        zone = ""
        if self.zone_col and self.zone_col in self.parcels.columns:
            zone = str(self.parcels.iloc[pos].get(self.zone_col, ""))
        max_far = self._get_parcel_max_far(zone)
        parcel_sqft = self.parcels.iloc[pos].get(
            "parcel_sqft", self.parcels.iloc[pos].get("Shape_Area", 5000))
        theoretical = max_far * float(parcel_sqft)

        total_delivered = float(self._total_res_units[pos]) * AVG_UNIT_SQFT + float(self._total_comm_sqft[pos])
        remaining = max(theoretical - total_delivered, 0.0)

        actual_res = min(res_sqft, remaining)
        remaining -= actual_res
        actual_comm = min(comm_sqft, remaining)

        return {
            "res_sqft": actual_res,
            "comm_sqft": actual_comm,
            "remaining": max(remaining - actual_comm, 0.0),
        }

    def _summarize_capacity_ledger(self) -> dict:
        """Summarize total and remaining development capacity."""
        total_cap = 0.0
        consumed = 0.0
        n = len(self.parcels)
        for i in range(n):
            zone = ""
            if self.zone_col and self.zone_col in self.parcels.columns:
                zone = str(self.parcels.iloc[i].get(self.zone_col, ""))
            max_far = self._get_parcel_max_far(zone)
            parcel_sqft = self.parcels.iloc[i].get(
                "parcel_sqft", self.parcels.iloc[i].get("Shape_Area", 5000))
            cap = max_far * float(parcel_sqft)
            total_cap += cap
            used = (float(self._total_res_units[i]) * AVG_UNIT_SQFT +
                    float(self._total_comm_sqft[i]))
            consumed += min(used, cap)

        return {
            "total_capacity_sqft": total_cap,
            "consumed_sqft": consumed,
            "remaining_sqft": total_cap - consumed,
            "utilization_pct": consumed / max(total_cap, 1.0) * 100,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> pd.DataFrame:
        """Return per-corridor-year convergence diagnostics."""
        if not self._diagnostics:
            return pd.DataFrame()
        df = pd.DataFrame(self._diagnostics)
        df = df.sort_values(["corridor_id", "year"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Development model (the big one)
    # ------------------------------------------------------------------

    def _run_development_model(
        self,
        year: int,
        corridor_id: str,
        accessibility: np.ndarray,
        daily_riders: float,
        step_years: int = 1,
    ) -> dict:
        """Run Tier 2 development model for one corridor-year.

        Combines:
          A. Market-rate residential (relocation MNL -> SqFtProForma)
          B. Student housing (formula-driven from enrollment)
          C. Commercial (proforma-based)
          D. Vacancy-rent feedback
          E. Cumulative tracking

        Returns dict with new_units, new_comm_sqft, new_pop, new_jobs.
        """
        import time as _time
        _dev_t0 = _time.perf_counter()

        if year == 0:
            return {"new_units": 0.0, "new_comm_sqft": 0.0, "new_pop": 0.0, "new_jobs": 0.0}

        spatial = self._corridor_spatial_cache.get(corridor_id, {})
        w_walk_idx = spatial.get("w_walk_idx")
        if w_walk_idx is None or len(w_walk_idx) == 0:
            return {"new_units": 0.0, "new_comm_sqft": 0.0, "new_pop": 0.0, "new_jobs": 0.0}

        # Corridor parcels (within walk zone)
        corridor_positions = w_walk_idx

        # Stock depreciation: remove 0.4% of existing units per year.
        # US housing stock loss rate ~0.3-0.5%/year (Urban Institute).
        # Lost units create replacement demand that sustains construction
        # even at zero net population growth.  Near transit, depreciated
        # low-intensity stock is replaced by higher-intensity TOD.
        _ANNUAL_DEPRECIATION_RATE = 0.004
        _depreciated = self._total_res_units[corridor_positions] * _ANNUAL_DEPRECIATION_RATE * step_years
        self._total_res_units[corridor_positions] -= _depreciated
        self._occupied_res_units[corridor_positions] -= _depreciated
        # Floor at zero
        np.maximum(self._total_res_units[corridor_positions], 0.0, out=self._total_res_units[corridor_positions])
        np.maximum(self._occupied_res_units[corridor_positions], 0.0, out=self._occupied_res_units[corridor_positions])

        # ---- A. Market-rate residential via MNL + SqFtProForma ----
        # Relocation MNL determines demand (how many HH relocate to corridor)
        from src.relocation_model import HouseholdRelocationModel, TARGET_VAC_RES

        reloc_model = HouseholdRelocationModel()

        n_parcels = len(self.parcels)
        # Developable mask: parcels with valid zone and minimum size
        developable = np.zeros(n_parcels, dtype=bool)
        zone_arr = (
            self.parcels[self.zone_col].values if self.zone_col and
            self.zone_col in self.parcels.columns else np.full(n_parcels, "R3")
        )
        for i in corridor_positions:
            zone = str(zone_arr[i])
            entry = ZONING_MATRIX.get(zone)
            if entry is not None:
                _, use_type, _ = entry
                if use_type != "undevelopable":
                    developable[i] = True

        corridor_hh = reloc_model.allocate_movers_to_corridor(
            near_800_idx=corridor_positions,
            developable_mask=developable,
            n_parcels=n_parcels,
            rents=self._current_rents,
            commute_times=self._commute_times,
            accessibility=accessibility,
            year=year,
        )

        # Scale demand for step size
        corridor_hh *= step_years

        # Scenario capture multiplier: upzoning attracts more households to
        # the corridor (agglomeration / amenity effect).  current_zoning=1.0,
        # no_zoning=1.30.
        _scenario_mult = self.demand_dev_model.capture.get_scenario_multiplier(
            self.development_scenario)
        corridor_hh *= _scenario_mult

        # Rent-responsive migration: corridors with below-average rents
        # attract slightly more movers (cheaper housing pulls in-migration).
        # Stawarz 2021: destination rent coefficient -0.42 for gross migration;
        # attenuated to -0.20 for net effect in a small metro.
        _MIGRATION_RENT_ELASTICITY = -0.20
        if len(corridor_positions) > 0 and self._mean_metro_rent > 0:
            _mean_corridor_rent = float(
                np.mean(self._current_rents[corridor_positions][
                    self._current_rents[corridor_positions] > 0
                ]) if np.any(self._current_rents[corridor_positions] > 0)
                else self._mean_metro_rent
            )
            _rent_ratio = _mean_corridor_rent / self._mean_metro_rent
            _rent_adj = 1.0 + _MIGRATION_RENT_ELASTICITY * (_rent_ratio - 1.0)
            corridor_hh *= max(_rent_adj, 0.80)  # cap at 20% reduction

        # Convert HH demand to target units.
        # corridor_hh includes movers who will occupy existing vacant units,
        # not just those needing new construction.  Subtract the corridor's
        # existing vacant capacity so only the residual drives new starts.
        _corridor_total = float(self._total_res_units[corridor_positions].sum())
        _corridor_occupied = float(self._occupied_res_units[corridor_positions].sum())
        _existing_vacant = max(_corridor_total - _corridor_occupied, 0.0)
        _new_build_hh = max(corridor_hh - _existing_vacant, 0.0)
        target_res_units = max(int(_new_build_hh / (1.0 - TARGET_VAC_RES)), 0)

        # Developer confidence ramp: sigmoid from AbsorptionParams, modulated
        # by ridership so low-ridership corridors develop slower
        abs_params = self.demand_dev_model.absorption
        confidence = abs_params.confidence_factor(year, daily_riders)

        # Vacancy-responsive dampener: developers pull back as corridor
        # vacancy rises above the construction-phase tolerance threshold.
        # Onset at 10% (above structural + normal delivery-phase buffer),
        # linear decline to zero at 16% (distressed market, financing
        # unavailable).  Floor at 10% so replacement/student demand persists.
        # Sources: NMHC/NAHB multifamily market indicators; SILO model
        # vacancy-gap approach (Moeckel et al. 2018).
        _VACANCY_DAMPENER_ONSET = 0.10
        _VACANCY_DAMPENER_SHUTOFF = 0.16
        _VACANCY_DAMPENER_FLOOR = 0.10
        _corridor_vacancy = (
            1.0 - _corridor_occupied / max(_corridor_total, 1.0)
            if _corridor_total > 0 else 0.0
        )
        _vac_excess = max(0.0, _corridor_vacancy - _VACANCY_DAMPENER_ONSET)
        _vac_range = _VACANCY_DAMPENER_SHUTOFF - _VACANCY_DAMPENER_ONSET
        _vacancy_dampener = max(
            _VACANCY_DAMPENER_FLOOR,
            1.0 - _vac_excess / _vac_range,
        )
        confidence *= _vacancy_dampener

        target_res_units = int(target_res_units * confidence)

        # Supply-side cap: county-level absorption constraint
        max_corridor_units = int(
            abs_params.county_annual_res_permits
            * abs_params.corridor_capacity_share
            * step_years
        )
        target_res_units = min(target_res_units, max_corridor_units)

        # Cost escalation from county capacity utilization
        cum_pop = self._cumulative_corridor_pop.get(corridor_id, 0.0)
        _metro_pop = float(self._metro_growth_params.get("metro_population", 230_000))
        res_cost_esc = abs_params.cost_escalation(
            cum_pop / max(AVG_HOUSEHOLD_SIZE, 1.0), step_years)

        # Zoning cost multiplier: scenario-dependent regulatory cost reduction
        _zoning_cost = self.demand_dev_model.zoning_costs.get_multiplier(
            self.development_scenario)

        # Run proforma
        proforma_df = self._prepare_proforma_df(
            corridor_positions, accessibility, daily_riders, year)

        new_res_sqft = 0.0
        new_comm_sqft = 0.0
        developed_positions = np.array([], dtype=int)

        if proforma_df is not None and target_res_units > 0:
            res_sqft, _, dev_pos = self._run_proforma_developer(
                proforma_df, target_res_units, use_form="residential",
                cost_escalation_factor=res_cost_esc,
                zoning_rent_factor=_zoning_cost)
            new_res_sqft += res_sqft
            if len(dev_pos) > 0:
                developed_positions = dev_pos

        # ---- B. Student housing (formula-driven) ----
        # Not proforma-based — enrollment-driven with constant growth
        student_units = 0.0
        if self._institutional_weights is not None:
            campus_mask = self._institutional_weights[corridor_positions] >= 2.0
            campus_parcels = corridor_positions[campus_mask]
            if len(campus_parcels) > 0:
                # Purpose-built student housing: FAR-responsive delivery rate.
                # Base 100 units/year (calibrated to observed Lafayette near-campus
                # delivery of 175-250 units/year total, of which ~60% is purpose-
                # built student housing).  Scales with available FAR so no_zoning
                # allows taller student towers.  Lifetime cap 500 units/corridor.
                _BASE_STUDENT_UNITS_YR = 100.0
                _STUDENT_LIFETIME_CAP = 500.0
                _campus_zone = str(self.parcels.iloc[campus_parcels[0]].get(
                    self.zone_col, "R3")) if self.zone_col else "R3"
                _campus_far = self._get_parcel_max_far(_campus_zone)
                _far_ratio = min(_campus_far / 1.8, 2.5)  # 1.8 = R3U-equivalent baseline
                _student_rate = _BASE_STUDENT_UNITS_YR * max(_far_ratio, 1.0)
                student_units = min(
                    _student_rate * step_years * confidence,
                    _STUDENT_LIFETIME_CAP,
                )
                student_sqft = student_units * AVG_UNIT_SQFT
                new_res_sqft += student_sqft

        # ---- Tenure classification (homestead vs rental) ----
        #
        # SIMPLIFICATION: we classify new construction as homestead or
        # rental based on the zoning of the developed parcel, not by
        # modeling individual unit tenure.  This is justified because:
        #
        #   1. Indiana IC 6-1.1-12-37 defines "homestead" as owner-
        #      occupied principal residence.  Zone codes are a strong
        #      proxy: SF zones (R1/R1A/R1B/R1T/R1U/PDRS) produce
        #      detached houses that are overwhelmingly owner-occupied;
        #      R2+ and MU zones produce apartments and townhomes that
        #      are predominantly renter-occupied, especially near a
        #      university campus.
        #
        #   2. The proforma selects high-FAR parcels first, so even
        #      though ~37% of station-area parcels are in SF zones,
        #      only ~2-5% of built sqft lands on them (current_zoning).
        #      This matches national TOD patterns where new station-
        #      area construction is 85-95% multifamily (TCRP Report 128).
        #
        #   3. Under no_zoning (FAR caps removed), the proforma builds
        #      dense multifamily on former SF lots.  A FAR ceiling of
        #      1.5 catches this: any parcel developed above FAR 1.5 is
        #      classified as rental regardless of its zone code, since
        #      the physical product is an apartment building, not a
        #      single-family home.
        #
        #   4. R2/R2U (duplex/townhome) zones are classified as rental.
        #      Some R2 units are owner-occupied (Indiana allows homestead
        #      on one half of a duplex), but near Purdue the vast
        #      majority are investor-owned student rentals.  This is
        #      conservative for TIF — it slightly overstates capturable
        #      increment in EDA areas.
        #
        #   5. Student housing (formula-driven, not proforma) is always
        #      rental — university housing is never owner-occupied.
        #
        # The resulting homestead share on new construction is:
        #   current_zoning: ~1-3% (proforma avoids low-FAR SF parcels)
        #   no_zoning:       ~0%  (all parcels built at FAR >> 1.5)
        #
        # County-wide existing stock is 45.6% owner-occupied (ACS 2022),
        # but station-area new construction is overwhelmingly rental.
        # The evaluator at apm_corridor_evaluation_integrated.py reads
        # new_homestead_sqft/new_rental_sqft directly; the fallback
        # TIF_HOMESTEAD_SHARE (financial_params.py) is legacy-only.
        #
        market_res_sqft = new_res_sqft - (student_units * AVG_UNIT_SQFT if student_units > 0 else 0.0)
        _SF_ZONES = {"R1", "R1A", "R1B", "R1T", "R1U", "PDRS"}
        _SF_FAR_CEILING = 1.5  # above this, output is MF regardless of zone
        if len(developed_positions) > 0 and market_res_sqft > 0:
            _zone_arr = (
                self.parcels[self.zone_col].values if self.zone_col and
                self.zone_col in self.parcels.columns else np.full(n_parcels, "R3")
            )
            _sf_count = 0
            for p in developed_positions:
                if str(_zone_arr[p]) not in _SF_ZONES:
                    continue
                # Under no_zoning, check effective FAR — dense builds
                # on SF-zoned parcels are multifamily, not homestead
                if self.development_scenario == "no_zoning":
                    _eff_far = self._get_parcel_max_far(str(_zone_arr[p]))
                    if _eff_far > _SF_FAR_CEILING:
                        continue
                _sf_count += 1
            _hs_frac = _sf_count / len(developed_positions)
            new_homestead_sqft = market_res_sqft * _hs_frac
        else:
            new_homestead_sqft = 0.0
        new_rental_sqft = new_res_sqft - new_homestead_sqft

        # ---- C. Commercial development ----
        # Independent commercial demand from metro job growth, not a
        # fixed ratio of residential.  Mirrors the residential chain:
        # metro jobs × growth rate → corridor capture → vacancy absorption
        # → net new demand.
        _metro_jobs = float(self._metro_growth_params.get("metro_jobs", 95_000))
        _job_growth_rate = float(self._metro_growth_params.get(
            "annual_job_growth_rate", 0.018))
        _annual_new_jobs = _metro_jobs * _job_growth_rate
        _comm_capture = self.demand_dev_model.capture
        _comm_capture_rate = min(
            _comm_capture.base_corridor_capture_rate
            + (_comm_capture.max_corridor_capture_rate
               - _comm_capture.base_corridor_capture_rate)
            * min(year / max(_comm_capture.capture_ramp_years, 1.0), 1.0),
            _comm_capture.max_corridor_capture_rate,
        )
        _corridor_new_jobs = (
            _annual_new_jobs * _comm_capture_rate * _scenario_mult
            * confidence * _vacancy_dampener * step_years)
        _comm_demand_sqft = _corridor_new_jobs * SQFT_PER_EMPLOYEE

        # Subtract existing vacant commercial capacity
        _existing_vacant_comm = max(
            float(self._total_comm_sqft[corridor_positions].sum())
            - float(self._occupied_comm_sqft[corridor_positions].sum()),
            0.0)
        _new_comm_demand_sqft = max(_comm_demand_sqft - _existing_vacant_comm, 0.0)

        # Accumulate fractional demand to avoid integer-truncation spikes
        _pending = self._pending_comm_demand.get(corridor_id, 0.0)
        _pending += _new_comm_demand_sqft / max(SQFT_PER_EMPLOYEE, 1.0)
        target_comm_units = max(int(_pending), 0)
        self._pending_comm_demand[corridor_id] = _pending - target_comm_units

        comm_cost_esc = abs_params.cost_escalation_commercial(
            new_comm_sqft, step_years) if new_comm_sqft > 0 else 1.0
        if proforma_df is not None and target_comm_units > 0:
            _, comm_sqft, _ = self._run_proforma_developer(
                proforma_df, target_comm_units, use_form="office",
                cost_escalation_factor=comm_cost_esc,
                zoning_rent_factor=_zoning_cost)
            new_comm_sqft += comm_sqft

        # ---- D. Schedule deliveries via height-dependent occupancy ----
        new_units = new_res_sqft / AVG_UNIT_SQFT

        # Estimate stories from developed sqft and parcel area to select
        # the appropriate lease-up schedule.  parcel_coverage = 0.8 matches
        # SqFtProForma default.
        _est_stories = 3  # default: low-rise
        if len(developed_positions) > 0 and new_res_sqft > 0:
            _avg_lot = float(self._parcel_sqft_arr[developed_positions].mean())
            _est_stories = int(np.ceil(
                new_res_sqft / (max(_avg_lot, 2000.0) * 0.8 * max(len(developed_positions), 1))
            ))
            _est_stories = max(_est_stories, 1)
        _res_schedule = get_occupancy_schedule(_est_stories, "residential")
        # Commercial is typically lower-rise than residential near stations
        # (1-2 story retail/office under multifamily podium).
        _comm_stories = min(_est_stories, 4)
        _comm_schedule = get_occupancy_schedule(_comm_stories, "commercial")
        _max_offsets = max(len(_res_schedule), len(_comm_schedule))

        for offset in range(_max_offsets):
            # Residential incremental fraction for this offset
            if offset < len(_res_schedule):
                res_cur = _res_schedule[offset]
                res_prev = _res_schedule[offset - 1] if offset > 0 else 0.0
            else:
                res_cur = _res_schedule[-1]
                res_prev = _res_schedule[-1]
            res_inc = max(res_cur - res_prev, 0.0)

            # Commercial incremental fraction for this offset
            if offset < len(_comm_schedule):
                comm_cur = _comm_schedule[offset]
                comm_prev = _comm_schedule[offset - 1] if offset > 0 else 0.0
            else:
                comm_cur = _comm_schedule[-1]
                comm_prev = _comm_schedule[-1]
            comm_inc = max(comm_cur - comm_prev, 0.0)

            if res_inc <= 0 and comm_inc <= 0:
                continue
            delivery_year = year + offset
            # first_delivery: the first non-zero tranche has reduced
            # occupancy (lease-up absorption lag — tenants don't fill
            # a building instantly on delivery).
            _is_first = (res_prev == 0.0 and res_inc > 0)
            self._pending_deliveries.setdefault(delivery_year, []).append({
                "corridor_id": corridor_id,
                "pos": corridor_positions,
                "res_sqft": new_res_sqft * res_inc,
                "comm_sqft": new_comm_sqft * comm_inc,
                "res_units": new_units * res_inc,
                "first_delivery": _is_first,
            })

        # ---- E. Vacancy-rent feedback ----
        if len(corridor_positions) > 0 and year > 0:
            self._update_rents_from_vacancy(corridor_positions, step_years, current_year=year)

        # ---- F. Cumulative tracking ----
        # Population from terminal schedule occupancy, not raw approved sqft.
        # High-rise schedule caps at 92% (permanent structural vacancy);
        # low-rise at 100%.  Student housing is formula-driven, always 100%.
        _delivered_res_frac = _res_schedule[-1]
        _delivered_comm_frac = _comm_schedule[-1]
        # new_units already includes student_units (student_sqft was added
        # to new_res_sqft at line 2102).  Don't add student_units again.
        new_pop = new_units * _delivered_res_frac * AVG_HOUSEHOLD_SIZE
        new_jobs = new_comm_sqft * _delivered_comm_frac / SQFT_PER_EMPLOYEE

        self._cumulative_corridor_pop[corridor_id] = (
            self._cumulative_corridor_pop.get(corridor_id, 0.0) + new_pop
        )
        self._cumulative_corridor_jobs[corridor_id] = (
            self._cumulative_corridor_jobs.get(corridor_id, 0.0) + new_jobs
        )

        _dev_elapsed = _time.perf_counter() - _dev_t0
        logger.debug(
            f"  {corridor_id} yr{year}: +{new_units:.0f} units, "
            f"+{new_comm_sqft:,.0f} comm sqft, "
            f"+{new_pop:.0f} pop, +{new_jobs:.0f} jobs "
            f"({_dev_elapsed:.1f}s)"
        )

        return {
            "new_units": new_units,
            "new_comm_sqft": new_comm_sqft,
            "new_homestead_sqft": new_homestead_sqft,
            "new_rental_sqft": new_rental_sqft,
            "new_pop": new_pop,
            "new_jobs": new_jobs,
            "student_units": student_units,
        }

    # ------------------------------------------------------------------
    # Pending delivery tracking
    # ------------------------------------------------------------------

    def _track_unit_delivery(
        self,
        positions: np.ndarray,
        res_units: float,
        comm_sqft: float,
        year: int = 0,
        first_delivery: bool = False,
    ):
        """Track delivered units on parcels (update totals and occupied).

        first_delivery: if True, this is the first tranche of a new
        development approval.  Occupancy is reduced to 50% of target
        to model lease-up absorption lag (tenants don't fill a building
        instantly).  Subsequent tranches use full target occupancy.
        """
        if len(positions) == 0 or (res_units <= 0 and comm_sqft <= 0):
            return
        n = len(positions)
        per_parcel_units = res_units / max(n, 1)
        per_parcel_comm = comm_sqft / max(n, 1)
        from src.relocation_model import TARGET_VAC_RES
        # First delivery: 50% of target occupancy (absorption lag)
        _occ_frac = (1.0 - TARGET_VAC_RES) * (0.5 if first_delivery else 1.0)
        _comm_occ = 0.90 * (0.5 if first_delivery else 1.0)
        for p in positions:
            self._total_res_units[p] += per_parcel_units
            self._occupied_res_units[p] += per_parcel_units * _occ_frac
            self._total_comm_sqft[p] += per_parcel_comm
            self._occupied_comm_sqft[p] += per_parcel_comm * _comm_occ
            self._parcel_last_delivery_year[p] = year

    def _track_unit_delivery_fallback(
        self,
        corridor_id: str,
        res_units: float,
        comm_sqft: float,
    ):
        """Fallback: distribute units across walk-zone parcels."""
        spatial = self._corridor_spatial_cache.get(corridor_id, {})
        w_walk_idx = spatial.get("w_walk_idx")
        if w_walk_idx is not None and len(w_walk_idx) > 0:
            self._track_unit_delivery(w_walk_idx, res_units, comm_sqft)

    def _deliver_pending(self, year: int):
        """Deliver any pending development scheduled for this year."""
        deliveries = self._pending_deliveries.pop(year, [])
        for d in deliveries:
            pos = d.get("pos", np.array([], dtype=int))
            res_units = d.get("res_units", 0.0)
            comm_sqft = d.get("comm_sqft", 0.0)
            first_delivery = d.get("first_delivery", False)
            if len(pos) > 0:
                self._track_unit_delivery(
                    pos, res_units, comm_sqft, year=year,
                    first_delivery=first_delivery)

    # ------------------------------------------------------------------
    # Population / jobs update
    # ------------------------------------------------------------------

    def _update_pop_jobs(
        self,
        corridor_id: str,
        dev_result: dict,
        step_years: int = 1,
    ):
        """Distribute new pop/jobs to parcels near corridor stops.

        Uses distance-decay weighting to concentrate growth near stations.
        Targets developed parcels (within walk zone).
        """
        new_pop = dev_result.get("new_pop", 0)
        new_jobs = dev_result.get("new_jobs", 0)

        if new_pop <= 0 and new_jobs <= 0:
            return

        spatial = self._corridor_spatial_cache.get(corridor_id, {})
        w_walk_idx = spatial.get("w_walk_idx")
        w_walk_val = spatial.get("w_walk_val")
        if w_walk_idx is None or len(w_walk_idx) == 0:
            return

        # Weight by walk-zone decay values
        weights = w_walk_val.copy()
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return

        frac = weights / weight_sum

        # POP_PER_PARCEL_CAP: mega-parcel fix
        POP_PER_PARCEL_CAP = 2000

        # Distribute new pop
        if new_pop > 0:
            pop_add = frac * new_pop
            # Cap per-parcel addition
            current_pop = self.parcels[self.pop_col].values[w_walk_idx]
            headroom = np.maximum(POP_PER_PARCEL_CAP - current_pop, 0)
            pop_add = np.minimum(pop_add, headroom)
            self.parcels.iloc[w_walk_idx, self.parcels.columns.get_loc(self.pop_col)] += pop_add

        # Distribute new jobs
        if new_jobs > 0 and self.jobs_col:
            jobs_add = frac * new_jobs
            jobs_loc = self.parcels.columns.get_loc(self.jobs_col)
            for i, idx in enumerate(w_walk_idx):
                self.parcels.iat[idx, jobs_loc] += jobs_add[i]

        # Regional growth reallocation: small spillover to other areas.
        # Elasticity of 0.05 means at mature ridership (~2500 daily),
        # corridor captures ~3% more development than baseline.
        # Handled implicitly through DemandDrivenDevelopmentModel's metro growth rate.

    # ------------------------------------------------------------------
    # OD demand weight for feeder routing
    # ------------------------------------------------------------------

    def _compute_od_demand_weight(
        self,
        feeder_idx: np.ndarray,
        walk_idx: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute per-feeder-parcel OD demand relevance [0, 1].

        For each parcel in the feeder zone, computes what fraction of its
        LODES commute flows have destinations within the APM walk zone.
        A high value means the parcel's workers commute to APM-served jobs,
        so a feeder bus connecting this parcel to APM is valuable.

        Returns array of length len(feeder_idx), or None on failure.
        Performance: O(n_od_flows) scan — fast for typical LODES (~50K rows).
        """
        if not hasattr(self, 'od_flows') or self.od_flows is None:
            return None
        try:
            _walk_set = set(int(i) for i in walk_idx)
            _feeder_list = feeder_idx.astype(int)
            _n_feeder = len(_feeder_list)
            if _n_feeder == 0:
                return None

            # Build parcel-position → feeder-array-index map
            _feeder_pos_to_idx = {}
            for _fi, _fp in enumerate(_feeder_list):
                _feeder_pos_to_idx.setdefault(int(_fp), _fi)

            # Scan OD flows: count total and walk-dest trips per feeder parcel
            _total = np.zeros(_n_feeder, dtype=np.float64)
            _walk_dest = np.zeros(_n_feeder, dtype=np.float64)

            _pid_arr = self.parcels.index.astype(str).values
            _pid_to_pos = {str(pid): i for i, pid in enumerate(_pid_arr)}

            for _, row in self.od_flows.iterrows():
                _o_pid = str(row.get("origin_parcel", "")).lstrip("ST")
                _d_pid = str(row.get("dest_parcel", "")).lstrip("ST")
                _trips = float(row.get("trips", 0))
                if _trips <= 0:
                    continue
                _o_pos = _pid_to_pos.get(_o_pid)
                if _o_pos is None:
                    continue
                _fi = _feeder_pos_to_idx.get(_o_pos)
                if _fi is None:
                    continue
                _total[_fi] += _trips
                _d_pos = _pid_to_pos.get(_d_pid)
                if _d_pos is not None and _d_pos in _walk_set:
                    _walk_dest[_fi] += _trips

            _has_trips = _total > 0
            _weight = np.where(_has_trips, _walk_dest / _total, 0.5)
            return _weight
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Bus restructuring
    # ------------------------------------------------------------------

    def _restructure_bus(
        self,
        corridor_id: str,
        ridership: Dict[str, dict],
        year: int,
    ):
        """Adjust bus headways based on APM ridership maturity.

        Uses threshold-based restructuring: trigger when ridership changes
        >= RESTRUCTURE_RIDERSHIP_THRESHOLD since last event.  Year 0 always
        triggers.
        """
        rdata = ridership.get(corridor_id, {})
        daily_riders = rdata.get("daily_riders", 0.0)

        # Check if restructuring should trigger
        last_riders = self._last_restructure_ridership.get(corridor_id, 0.0)
        if last_riders > 0:
            change = abs(daily_riders - last_riders) / max(last_riders, 1.0)
            if change < RESTRUCTURE_RIDERSHIP_THRESHOLD and year > 0:
                return  # No significant change, skip restructuring

        self._last_restructure_ridership[corridor_id] = daily_riders

        # Compute restructuring pressure
        comp = DEFAULT_BUS_COMPETITIVENESS
        prod = DEFAULT_BUS_PRODUCTIVITY
        if self._gtfs_competitiveness is not None:
            try:
                cdf = self._gtfs_competitiveness
                if "corridor_id" in cdf.columns:
                    row = cdf[cdf["corridor_id"] == corridor_id]
                    if len(row) > 0:
                        comp = float(row.iloc[0].get("competitiveness_score", comp))
                        prod = float(row.iloc[0].get("productivity_score", prod))
            except Exception:
                pass

        pressure = _restructure_pressure(
            daily_riders, MATURE_RIDERSHIP_TARGET, comp, prod)
        self._restructure_pressure[corridor_id] = pressure

        # Restructure parallel bus: headway increases as APM matures
        _max_par = self._bus_max_parallel_headway
        parallel_headway = BASE_BUS_HEADWAY * (1.0 + 2.0 * pressure)
        parallel_headway = min(parallel_headway, _max_par)
        self._bus_headways[corridor_id] = parallel_headway

        # Feeder bus: demand-proportional frequency allocation.
        # Instead of a flat headway, compute an aggregate feeder headway
        # from restructuring pressure, then adjust per-sector based on
        # population weight.  The corridor-level headway stored here is
        # the effective (population-weighted) average; individual sector
        # headways feed into sector coverage quality scores.
        _min_feed = self._bus_min_feeder_headway
        _max_feed = self._bus_max_feeder_headway
        _base_feeder_hw = max(
            _min_feed,
            BASE_BUS_HEADWAY * (1.0 - 0.5 * pressure),
        )
        _base_feeder_hw = min(_base_feeder_hw, _max_feed)

        # Demand-proportional adjustment: if sector coverage data exists,
        # allocate shorter headways to high-population sectors and longer
        # to low-population sectors, keeping total vehicle-hours constant.
        _sc = self._sector_coverage.get(corridor_id)
        if _sc is not None and hasattr(_sc, 'pop_weight') and _sc.pop_weight.sum() > 0:
            _pw = _sc.pop_weight
            _mean_pw = _pw.mean()
            if _mean_pw > 0:
                # Sector-level headway: inverse-proportional to population share.
                # High-pop sector → shorter headway; low-pop → longer.
                # Clamp ratio to [0.5, 2.0] to prevent extreme allocations.
                _ratios = np.clip(_pw / _mean_pw, 0.5, 2.0)
                _sector_hws = _base_feeder_hw / _ratios  # shorter hw = more freq
                _sector_hws = np.clip(_sector_hws, _min_feed, _max_feed)
                # Update sector coverage quality with demand-proportional headways
                _new_quality = np.clip(15.0 / np.maximum(_sector_hws, 1.0), 0.1, 1.0)
                _sc.coverage = _new_quality * np.where(_sc.coverage > 0, 1.0, 0.0)
                self._sector_coverage[corridor_id] = _sc
                # Effective headway = population-weighted harmonic mean
                _inv_hw = 1.0 / np.maximum(_sector_hws, 1.0)
                _eff_hw = float(_pw.sum() / np.maximum((_pw * _inv_hw).sum(), 1e-9))
                feeder_headway = float(np.clip(_eff_hw, _min_feed, _max_feed))
            else:
                feeder_headway = _base_feeder_hw
        else:
            feeder_headway = _base_feeder_hw

        # Budget-feasibility ceiling: the pressure-driven headway cannot be
        # shorter than what the bus budget can actually fund.
        #
        # Budget available for feeders = total bus budget minus parallel +
        # independent route costs.  Parallel cost ≈ vehicle-hours at the
        # parallel headway; independent routes are ~40% of pre-APM budget.
        # Feeder vehicle-hours = budget remainder / cost per veh-hr.
        # Feeder headway = (service span in min × n_feeder_routes) / feeder_vhr_per_day.
        #
        # n_feeder_routes approximated as active sectors (coverage > 0).
        _bus_budget = CITYBUS_ANNUAL_BUDGET_USD
        _cost_per_vhr = DEFAULT_COST_PER_VEH_HOUR
        _total_vhr = _bus_budget / max(_cost_per_vhr, 1.0)
        # Parallel route consumption: one route per direction at the parallel headway
        _par_trips_per_day = DEFAULT_SERVICE_SPAN_HOURS * 60.0 / max(parallel_headway, 1.0)
        _par_daily_vhr = _par_trips_per_day * 2.0  # round-trip
        _par_annual_vhr = _par_daily_vhr * SERVICE_DAYS_PER_YEAR
        # Independent routes: ~30% of total budget (equity-protected baseline).
        # CityBus runs ~19 routes; with APM, 4-6 parallel routes degrade and
        # 2-3 independent routes consolidate, leaving ~10 at reduced headways.
        _independent_vhr = _total_vhr * 0.30
        _feeder_vhr = max(_total_vhr - _par_annual_vhr - _independent_vhr, 0.0)
        # Active feeder routes: scales with ridership.  A small-city APM
        # starts with 2 feeders and grows to 4 as ridership justifies
        # service expansion.  8 sectors exist spatially but not all get
        # dedicated bus routes — low-demand sectors share routes or go unserved.
        # Capped at 4: CityBus $13.5M budget can fund ~4 feeders at 20-min
        # headway after parallel route degradation and independent route costs.
        _n_feeder_routes = max(2, min(4, int(2 + 2 * pressure)))
        if _sc is not None and hasattr(_sc, 'coverage'):
            _active = int((np.asarray(_sc.coverage) > 0).sum())
            _n_feeder_routes = max(2, min(_n_feeder_routes, _active))
        # Budget-feasible minimum headway
        _feeder_daily_vhr = _feeder_vhr / max(SERVICE_DAYS_PER_YEAR, 1)
        if _feeder_daily_vhr > 0 and _n_feeder_routes > 0:
            # Each route needs service_span_hours of one vehicle per headway cycle
            _budget_feasible_hw = (
                DEFAULT_SERVICE_SPAN_HOURS * 60.0 * _n_feeder_routes
                / max(_feeder_daily_vhr, 1.0)
            )
        else:
            _budget_feasible_hw = _max_feed

        # Clamp: can't promise service the budget can't fund
        feeder_headway = max(feeder_headway, _budget_feasible_hw)
        feeder_headway = float(np.clip(feeder_headway, _min_feed, _max_feed))

        # Diagnostic: how close is pressure-driven allocation to budget-optimal?
        # Ratio < 1.0 means we're using less feeder capacity than budget allows.
        _budget_optimal_hw = max(_budget_feasible_hw, _min_feed)
        _capacity_utilization = _budget_optimal_hw / max(feeder_headway, 1.0)
        # (values near 1.0 = at budget limit; <0.5 = pressure not using available budget)
        self._feeder_budget_utilization[corridor_id] = float(
            np.clip(_capacity_utilization, 0.0, 1.0))

        self._feeder_headways[corridor_id] = feeder_headway

        # APM headway: demand-responsive
        try:
            apm_hw = compute_apm_headway(daily_riders)
            self._apm_headways[corridor_id] = apm_hw
        except Exception:
            pass

        # Feeder coverage: increases with restructuring pressure
        base_cov = 0.15
        self._feeder_coverage[corridor_id] = min(
            base_cov + 0.60 * pressure, 0.75)

        # Dynamic bus network (GTFS-based) if available
        if self._bus_routes is not None:
            try:
                spatial = self._corridor_spatial_cache.get(corridor_id, {})
                stops_proj = spatial.get("stops_proj")
                corridor_row = self._corridor_rows[corridor_id]

                if stops_proj is not None:
                    classifications = classify_routes_for_corridor(
                        self._bus_routes, corridor_row, stops_proj)

                    # Decide restructuring per route
                    decisions = decide_route_restructuring(
                        classifications, pressure, year)

                    # Apply decisions
                    if decisions:
                        applied = apply_restructuring_decisions(
                            self._bus_routes, decisions, pressure)

                        # Update service profiles
                        if applied is not None:
                            if hasattr(applied, "parallel_profile"):
                                self._bus_service_profiles[corridor_id] = applied.parallel_profile
                            if hasattr(applied, "feeder_profile"):
                                self._feeder_service_profiles[corridor_id] = applied.feeder_profile
                            if hasattr(applied, "feeder_coverage"):
                                self._feeder_coverage[corridor_id] = applied.feeder_coverage

                    # Sector coverage from bus network analysis
                    try:
                        sector_cov = check_coverage_equity(
                            self._bus_routes, classifications, corridor_row)
                        if sector_cov is not None:
                            # Demand-weighted adjustment: scale sector pop_weight
                            # by fraction of OD flows with APM-served destinations.
                            _w5000_idx = spatial.get("w5000_idx")
                            _w_walk_idx = spatial.get("w_walk_idx")
                            if (_w5000_idx is not None and _w_walk_idx is not None
                                    and len(_w5000_idx) > 0 and len(_w_walk_idx) > 0
                                    and hasattr(self, 'od_flows')):
                                try:
                                    _od_weight = self._compute_od_demand_weight(
                                        _w5000_idx, _w_walk_idx)
                                    if _od_weight is not None and len(_od_weight) > 0:
                                        # Map per-feeder-parcel weight to sectors
                                        _fpop = sector_cov.pop_weight
                                        # Aggregate OD weight per sector from
                                        # feeder-parcel-level data
                                        _fsect = spatial.get("feeder_parcel_sector")
                                        _fpop_arr = spatial.get("feeder_parcel_pop")
                                        if _fsect is not None and _fpop_arr is not None:
                                            _n_fp = min(len(_od_weight), len(_fsect), len(_fpop_arr))
                                            for _s in range(8):
                                                _sm = _fsect[:_n_fp] == _s
                                                _sp = _fpop_arr[:_n_fp][_sm]
                                                if _sp.sum() > 0:
                                                    _sw = float(
                                                        (_od_weight[:_n_fp][_sm] * _sp).sum()
                                                        / _sp.sum())
                                                    # Floor at 0.2 so no sector is zeroed
                                                    _sw = max(_sw, 0.2)
                                                    sector_cov.pop_weight[_s] *= _sw
                                except Exception:
                                    pass
                            self._sector_coverage[corridor_id] = sector_cov
                            self._feeder_coverage[corridor_id] = sector_cov.effective_coverage
                    except Exception:
                        pass

                    # Bus speed from GTFS
                    try:
                        bus_spd = self._bus_speeds.get(corridor_id, BUS_SPEED_KPH)
                        if hasattr(applied, "avg_speed_kph"):
                            self._bus_speeds[corridor_id] = applied.avg_speed_kph
                    except Exception:
                        pass

                    # Dynamic transfer walk time: average distance from
                    # feeder bus stops to nearest APM station (meters → minutes).
                    try:
                        from scipy.spatial import cKDTree
                        _feeder_routes = [r for r in classifications
                                          if getattr(r, 'classification', '') == 'feeder']
                        if _feeder_routes and stops_proj is not None and len(stops_proj) >= 2:
                            _bus_stops_xy = spatial.get("bus_stops_by_route", {})
                            _all_fstops = []
                            for _fr in _feeder_routes:
                                _fstops = _bus_stops_xy.get(str(_fr.route_id))
                                if _fstops is not None and len(_fstops) > 0:
                                    _all_fstops.append(_fstops)
                            if _all_fstops:
                                _all_fstops = np.vstack(_all_fstops)
                                _stn_tree = cKDTree(stops_proj)
                                _dists, _ = _stn_tree.query(_all_fstops, k=1)
                                # Only count stops within 400m of a station (actual transfer points)
                                _transfer_stops = _dists[_dists <= 400.0]
                                if len(_transfer_stops) > 0:
                                    _avg_dist_m = float(_transfer_stops.mean())
                                    # Walk time: 80m/min (4.8 km/h walk speed)
                                    self._transfer_walk_min[corridor_id] = _avg_dist_m / 80.0
                    except Exception:
                        pass

            except Exception as e:
                logger.debug(f"  Dynamic bus restructuring failed for {corridor_id}: {e}")

        if pressure > 0.1:
            logger.debug(
                f"  {corridor_id} bus: parallel={parallel_headway:.0f}min, "
                f"feeder={feeder_headway:.0f}min "
                f"(pressure={pressure:.0%}, APM hw={self._apm_headways[corridor_id]:.1f}min)"
            )

    # ------------------------------------------------------------------
    # Compile results
    # ------------------------------------------------------------------

    def _compile_results(self) -> pd.DataFrame:
        """Build final results DataFrame."""
        df = pd.DataFrame(self._results)
        df = df.sort_values(["corridor_id", "year"]).reset_index(drop=True)
        return df

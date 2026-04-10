"""
Ridership computation mixin for LandUseTransportModel.
Extracted for file size management — no behavioral changes.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Dict, Optional

from scripts.generate_improved_ridership import (
    compute_lodes_ridership,
    CAR_SPEED_KPH,
    BUS_SPEED_KPH,
)
from src.spatial_constants import (
    PROJECT_CRS, WALK_CATCHMENT_M, US_SURVEY_FT_TO_M,
)
from src.model_constants import (
    RAMP_MIDPOINT, RAMP_STEEPNESS,
    STUDENT_AWARENESS_ADVANCE_YR, GENERATOR_AWARENESS_ADVANCE_YR,
    MAX_CATCHMENT_SCALE, WORK_OFFPEAK_EXPANSION,
    STUDENT_CAMPUS_TRIP_RATE, STUDENT_APM_SHARE_FLOOR,
    STUDENT_APM_SHARE_CAP, STUDENT_CAMPUS_TRIP_DIST_KM,
    STUDENT_TRIP_PURPOSES, STUDENT_OFFCAMPUS_DESTINATIONS,
    NONWORK_TRIP_PURPOSES, NONWORK_TRIP_CAPTURE_FRACTION,
    MEDICAL_DAILY_TRIPS, RETAIL_TRIP_RATE, RETAIL_JOB_FRACTION,
    EVENT_DAILY_EQUIVALENT,
    NON_COMMUTE_GENERATOR_APM_SHARE_FALLBACK,
    GENERATOR_PROXIMITY_DECAY_BETA,
    INDUCED_TRIP_ELASTICITY, INDUCED_DEMAND_THRESHOLD_YEAR,
    ZERO_CAR_HH_SHARE, ZERO_CAR_BY_INCOME,
    ZERO_CAR_SUPPRESSED_TRIPS, LATENT_TRIP_RELEASE_RATE,
    LATENT_FEEDER_RELEASE_DISCOUNT,
    LATENT_MATURITY_TARGET, LATENT_QUALITY_FLOOR,
    STUDENT_PRESENCE_FACTOR, FACULTY_PRESENCE_FACTOR,
    STUDENT_ANNUAL_FACTOR, COMMUTE_ANNUAL_FACTOR,
    NONWORK_ANNUAL_FACTOR, GENERATOR_ANNUAL_FACTOR,
    INDUCED_ANNUAL_FACTOR, LATENT_ANNUAL_FACTOR,
    FALLBACK_INCOME_SHARES,
    CONGESTION_ELASTICITY, BUS_CONGESTION_SHARE,
    PERIOD_CONGESTION_FACTORS,
    COMPONENT_PERIOD_WEIGHTS,
    MATURE_RIDERSHIP_TARGET,
    _PERIODS,
    _phasing_gate,
    _resolve_congestion_profile,
)
from src.model_helpers import (
    _sparse_dot,
    _sparse_accumulate,
)
import logging

logger = logging.getLogger(__name__)


class RidershipMixin:
    """Ridership computation methods.

    Mixed into LandUseTransportModel — all ``self`` references
    resolve against the full model instance at runtime.
    """

    def _compute_destination_coverage(
        self,
        stops_proj: np.ndarray,
        max_walk_m: float = WALK_CATCHMENT_M,
    ) -> float:
        """Fraction of student off-campus destinations within walk distance.

        Returns a value in [0, 1].  Uses exponential decay (half-credit at
        800m) matching the campus_alignment decay constant.
        """
        if stops_proj is None or len(stops_proj) == 0:
            return 0.0

        # Use cached projected destination coordinates (meters).
        # stops_proj is already in meters (EPSG:2965 feet × US_SURVEY_FT_TO_M).
        dest_arr = getattr(self, "_dest_proj_m", None)
        if dest_arr is None:
            # Fallback: compute on the fly (e.g. tests using __new__)
            try:
                from pyproj import Transformer
                _to_proj = Transformer.from_crs("EPSG:4326", "EPSG:2965", always_xy=True)
                _pts = []
                for _lon, _lat in STUDENT_OFFCAMPUS_DESTINATIONS.values():
                    px, py = _to_proj.transform(_lon, _lat)
                    _pts.append((px * US_SURVEY_FT_TO_M, py * US_SURVEY_FT_TO_M))
                dest_arr = np.array(_pts) if _pts else None
                self._dest_proj_m = dest_arr
            except Exception:
                return 0.0
        if dest_arr is None or len(dest_arr) == 0:
            return 0.0

        # Distance from each destination to its nearest station (both in meters)
        scores = []
        for d in dest_arr:
            dists = np.sqrt(((stops_proj - d) ** 2).sum(axis=1))
            min_dist_m = float(dists.min())
            # Exponential decay: exp(-0.000866 * d), half-credit at 800m
            scores.append(np.exp(-0.000866 * min_dist_m))

        return float(np.mean(scores))

    def _compute_student_apm_share(
        self,
        campus_to_station_m: float,
        apm_headway_min: float,
        bus_headway_min: float,
        corridor_length_km: float,
        n_stops: int,
        car_speed_kph: float = None,
        trip_dist_km: float = None,
        curve_delay_s: float = 0.0,
    ) -> float:
        """Compute corridor-specific student APM mode share via MNL logit.

        Instead of a static STUDENT_APM_MODE_SHARE, this evaluates APM vs
        bus/car/walk for a representative trip using the same utility
        parameters as the commute logit, plus student ASC adjustments.

        Parameters
        ----------
        campus_to_station_m : distance from campus center to nearest station
        apm_headway_min : current APM headway for this corridor
        bus_headway_min : current bus headway for this corridor
        corridor_length_km : corridor length (for effective speed calc)
        n_stops : number of stops on corridor
        trip_dist_km : purpose-specific trip distance (default: legacy 1.5 km)
        """
        from scripts.generate_improved_ridership import (
            BETA_IVT as _RAW_BETA_IVT, BETA_WAIT, BETA_ACCESS, BETA_COST,
            ASC_BUS, APM_FARE, BUS_FARE,
            WALK_SPEED_KPH, CAR_SPEED_KPH, CAR_COST_PER_MILE,
            CAR_CIRCUITY, WALK_CIRCUITY, BUS_CIRCUITY,
            CAMPUS_PARKING_COST_PER_TRIP,
            BIKE_SPEED_KPH, BIKE_ACCESS_PENALTY_MIN,
            BIKE_COST_PER_TRIP, ASC_BIKE_APM,
            compute_effective_apm_speed, compute_effective_brt_speed,
        )
        # Apply beta_distance_mult to student MNL (same scaling as commute MNL)
        _beta_dist_mult = float(self._model_options.get("beta_distance_mult", 1.0))
        BETA_IVT = _RAW_BETA_IVT * _beta_dist_mult
        # Use transit-mode-appropriate ASC (BRT=0.05 vs APM=0.12)
        transit_asc = self._transit_asc
        from src.data.purdue_transit_demand import (
            STUDENT_ASC_ADJUSTMENTS, SP_PLUS_ASC_PENALTY,
        )

        # Additive ASC shift for student APM preference (from model_options)
        _mode_bias = float(self._model_options.get("student_mode_bias", 0.0))

        # Use sampled MNL coefficients if available, otherwise fixed defaults
        _mnl = self._model_options.get("mnl_coefficients", None)
        if _mnl is not None:
            _b_ivt, _b_wait, _b_access, _b_cost = _mnl
        else:
            _b_ivt, _b_wait, _b_access, _b_cost = BETA_IVT, BETA_WAIT, BETA_ACCESS, BETA_COST

        trip_km = trip_dist_km if trip_dist_km is not None else STUDENT_CAMPUS_TRIP_DIST_KM
        access_km = campus_to_station_m / 1000.0

        # Transit utility: walk to station + ride + wait
        if self._transit_mode_name == "brt":
            apm_speed = compute_effective_brt_speed(corridor_length_km, n_stops)
        else:
            apm_speed = compute_effective_apm_speed(
                corridor_length_km, n_stops, curve_delay_s=curve_delay_s)
        apm_ivtt = (trip_km / apm_speed) * 60.0
        apm_wait = apm_headway_min / 2.0
        apm_access = (access_km * WALK_CIRCUITY / WALK_SPEED_KPH) * 60.0
        # SP+ campus shuttle penalty for intracampus trips (< 1.0 km)
        _sp_plus = SP_PLUS_ASC_PENALTY if trip_km < 1.0 else 0.0
        u_apm = (_b_ivt * apm_ivtt + _b_wait * apm_wait +
                 _b_access * apm_access + _b_cost * APM_FARE +
                 transit_asc + STUDENT_ASC_ADJUSTMENTS["apm"] + _mode_bias + _sp_plus)

        # Bus utility
        bus_ivtt = (trip_km * BUS_CIRCUITY / BUS_SPEED_KPH) * 60.0
        bus_wait = min(bus_headway_min / 2.0, 10.0)  # TCRP 165 cap
        bus_access = (access_km * 0.5 * WALK_CIRCUITY / WALK_SPEED_KPH) * 60.0
        u_bus = (_b_ivt * bus_ivtt + _b_wait * bus_wait +
                 _b_access * bus_access + _b_cost * BUS_FARE +
                 ASC_BUS + STUDENT_ASC_ADJUSTMENTS["bus"])

        # Car utility (students have low car ownership → strong ASC penalty)
        _cskph = car_speed_kph if car_speed_kph is not None else CAR_SPEED_KPH
        car_ivtt = (trip_km * CAR_CIRCUITY / _cskph) * 60.0
        car_miles = trip_km * CAR_CIRCUITY * 0.621371
        car_cost = car_miles * CAR_COST_PER_MILE + CAMPUS_PARKING_COST_PER_TRIP
        u_car = (_b_ivt * car_ivtt + _b_access * 2.0 +
                 _b_cost * car_cost + (-0.05) + STUDENT_ASC_ADJUSTMENTS["car"])

        # Walk utility (students walk a lot on campus)
        walk_ivtt = (trip_km * WALK_CIRCUITY / WALK_SPEED_KPH) * 60.0
        u_walk = (_b_ivt * walk_ivtt + 0.05 +
                  STUDENT_ASC_ADJUSTMENTS["walk"])

        # Bike-to-APM utility (5th mode): cycle to station, ride APM.
        # Purdue has high cycling rates — flat campus, extensive bike infra.
        # Competes with walking at longer access distances and with car for
        # students who lack vehicle access.
        bike_access_ivt = (access_km / BIKE_SPEED_KPH) * 60.0
        # SP+ penalty excluded: bike-to-APM implies longer access distance,
        # not the short on-campus loops that SP+ shuttles compete with.
        u_bike_apm = (_b_ivt * (apm_ivtt + bike_access_ivt) +
                      _b_wait * apm_wait +
                      _b_access * BIKE_ACCESS_PENALTY_MIN +
                      _b_cost * (APM_FARE + BIKE_COST_PER_TRIP) +
                      ASC_BIKE_APM + STUDENT_ASC_ADJUSTMENTS["apm"] + _mode_bias)

        exp_apm = np.exp(np.clip(u_apm, -20, 20))
        exp_bus = np.exp(np.clip(u_bus, -20, 20))
        exp_car = np.exp(np.clip(u_car, -20, 20))
        # Walk only viable for short distances
        exp_walk = np.exp(np.clip(u_walk, -20, 20)) if trip_km <= 2.0 else 0.0
        # Suppress bike-to-APM for short trips — nobody bikes to a station
        # to ride APM for < 1.5 km, then walks from the destination station.
        exp_bike = np.exp(np.clip(u_bike_apm, -20, 20)) if trip_km >= 1.5 else 0.0

        denom = exp_apm + exp_bike + exp_bus + exp_car + exp_walk
        if denom <= 0:
            return STUDENT_APM_SHARE_FLOOR

        # APM share includes both walk-to-APM and bike-to-APM
        share = float((exp_apm + exp_bike) / denom)
        bus_share = float(exp_bus / denom)
        return (
            float(np.clip(share, STUDENT_APM_SHARE_FLOOR, STUDENT_APM_SHARE_CAP)),
            bus_share,
        )

    def _compute_campus_alignment(
        self,
        parcel_dist: np.ndarray,
    ) -> float:
        """Distance-weighted campus alignment score for a corridor.

        Returns a score in [0, 1] representing how well this corridor serves
        the campus.  Uses continuous exponential decay rather than a binary
        1200m cutoff, so corridors get proportionally less credit for campus
        parcels that are farther from stations.  A corridor with stops at
        the campus core scores ~1.0; one 3+ km away scores near 0.
        """
        if self._institutional_weights is None:
            return 0.0
        campus_mask = self._institutional_weights >= 2.0
        if not campus_mask.any():
            return 0.0

        campus_weights = self._institutional_weights[campus_mask]
        campus_dists = parcel_dist[campus_mask]
        total_weight = float(campus_weights.sum())
        if total_weight <= 0:
            return 0.0

        # Exponential decay: half-credit at 800m, near-zero beyond 2500m
        # β = ln(2)/800 ≈ 0.000866
        ALIGNMENT_DECAY = 0.000866
        proximity_score = np.exp(-ALIGNMENT_DECAY * campus_dists)
        served_weight = float((campus_weights * proximity_score).sum())
        return float(np.clip(served_weight / total_weight, 0.0, 1.0))

    def _compute_nonwork_apm_share(
        self,
        purpose_spec: dict,
        apm_headway_min: float,
        bus_headway_min: float,
        apm_effective_speed: float,
        car_speed_kph: float = None,
    ) -> float:
        """4-mode MNL for non-work trip purposes.

        Uses purpose-specific parameters (trip distance, parking, VOT
        multiplier, transit ASC multiplier) instead of borrowing the
        commute APM share.  Based on USDOT Revised Departmental Guidance
        Table 4 (value-of-time by trip purpose) and TCRP 95 Ch 9
        (non-work transit elasticities).

        Returns transit mode share for this purpose.
        """
        from scripts.generate_improved_ridership import (
            BETA_IVT as _RAW_IVT, BETA_WAIT as _RAW_WAIT,
            BETA_ACCESS as _RAW_ACC, BETA_COST as _RAW_COST,
            ASC_BUS as _ASC_BUS,
            APM_FARE as _APM_FARE, BUS_FARE as _BUS_FARE,
            WALK_SPEED_KPH as _WALK_SPD, CAR_SPEED_KPH as _CAR_SPD_DEFAULT,
            CAR_COST_PER_MILE as _CAR_CPM, CAR_CIRCUITY as _CAR_CIRC,
            WALK_CIRCUITY as _WALK_CIRC, BUS_CIRCUITY as _BUS_CIRC,
            BUS_SPEED_KPH as _BUS_SPD,
            effective_wait_time as _eff_wait,
        )
        # Use sampled MNL coefficients if available
        _mnl = self._model_options.get("mnl_coefficients", None)
        if _mnl is not None:
            _b_ivt, _b_wait, _b_acc, _b_cost = _mnl
        else:
            _b_ivt, _b_wait, _b_acc, _b_cost = _RAW_IVT, _RAW_WAIT, _RAW_ACC, _RAW_COST

        # Purpose-specific VOT scaling (USDOT guidance: non-work VOT is
        # 40-60% of commute; NCHRP 716 Table 7-3 by purpose).
        # VOT = |BETA_IVT| * 60 / |BETA_COST|.  Lower non-work VOT means
        # time matters *less* for discretionary trips, not that cost matters
        # less.  Scale IVT/WAIT/ACCESS coefficients, keep cost unchanged.
        _vot = purpose_spec.get("vot_mult", 0.50)
        _b_ivt_nw = _b_ivt * _vot
        _b_wait_nw = _b_wait * _vot
        _b_acc_nw = _b_acc * _vot
        _b_cost_nw = _b_cost  # cost sensitivity unchanged
        # Transit ASC multiplier (non-habitual → lower propensity)
        _asc_mult = purpose_spec.get("asc_transit_mult", 0.50)
        _transit_asc_nw = self._transit_asc * _asc_mult

        dist_km = purpose_spec["dist_km"]
        parking = purpose_spec.get("parking", 0.0)
        _car_spd = car_speed_kph if car_speed_kph is not None else _CAR_SPD_DEFAULT
        access_min = 5.0  # typical walk access (400m at 5 kph)

        # Transit (APM/BRT)
        apm_ivtt = (dist_km / max(apm_effective_speed, 1.0)) * 60.0
        apm_wait = apm_headway_min / 2.0
        u_transit = (
            _b_ivt_nw * apm_ivtt + _b_wait_nw * apm_wait
            + _b_acc_nw * access_min + _b_cost_nw * _APM_FARE
            + _transit_asc_nw
        )
        # Bus
        bus_ivtt = (dist_km * _BUS_CIRC / _BUS_SPD) * 60.0
        bus_wait = _eff_wait(bus_headway_min)
        u_bus = (
            _b_ivt_nw * bus_ivtt + _b_wait_nw * bus_wait
            + _b_acc_nw * access_min * 1.3 + _b_cost_nw * _BUS_FARE
            + _ASC_BUS * _asc_mult
        )
        # Car (free parking at suburban retail, some parking downtown)
        car_ivtt = (dist_km * _CAR_CIRC / _car_spd) * 60.0
        car_miles = dist_km * _CAR_CIRC * 0.621371
        car_cost = car_miles * _CAR_CPM + parking
        u_car = _b_ivt_nw * car_ivtt + _b_acc_nw * 2.0 + _b_cost_nw * car_cost + (-0.05)
        # Walk (only viable for short trips)
        walk_ivtt = (dist_km * _WALK_CIRC / _WALK_SPD) * 60.0
        u_walk = _b_ivt_nw * walk_ivtt + 0.05

        e_t = np.exp(np.clip(u_transit, -20, 20))
        e_bus = np.exp(np.clip(u_bus, -20, 20))
        e_car = np.exp(np.clip(u_car, -20, 20))
        e_walk = np.exp(np.clip(u_walk, -20, 20)) if dist_km <= 2.0 else 0.0
        denom = e_t + e_bus + e_car + e_walk
        return float(e_t / denom) if denom > 0 else 0.0

    def _compute_ridership(
        self, year: int, active_corridor_id: Optional[str] = None,
    ) -> Dict[str, dict]:
        """Compute corridor ridership with temporal ramp.

        Parameters
        ----------
        year : simulation year
        active_corridor_id : if set, only compute ridership for this corridor
            (used in per-corridor independent evaluation mode)
        """
        # --- Alternative / phasing gate ---
        _alternative = str(self._model_options.get("alternative", "build"))
        _opening_delay = int(self._model_options.get("opening_delay_years", 0))
        _apm_active = (
            _alternative == "build"
            and _phasing_gate(year, _opening_delay)
        )

        # Logistic awareness curve
        if _apm_active:
            # Adjust awareness for delayed opening: awareness ramp starts at opening
            _effective_year = year - _opening_delay
            awareness = 1.0 / (1.0 + np.exp(-RAMP_STEEPNESS * (_effective_year - RAMP_MIDPOINT)))
        else:
            awareness = 0.0

        pop_arr = self.parcels[self.pop_col].to_numpy(dtype=np.float64, copy=False)
        jobs_arr = self.parcels[self.jobs_col].to_numpy(dtype=np.float64, copy=False)

        ridership = {}
        _cid_list = [active_corridor_id] if active_corridor_id else list(self._corridor_meta.keys())
        for cid in _cid_list:
            corridor = self._corridor_rows[cid]
            bus_hw = self._bus_headways[cid]
            feeder_hw = self._feeder_headways[cid]
            # Treat feeders > 30 min as nonexistent (TCRP 165)
            if feeder_hw > 30.0:
                feeder_cov = 0.0
            else:
                feeder_cov = self._feeder_coverage[cid]
            sector_cov = self._sector_coverage.get(cid)
            apm_hw = self._apm_headways[cid]
            bus_spd = self._bus_speeds[cid]
            spatial = self._corridor_spatial_cache.get(cid, {})
            parcel_dist = spatial.get("parcel_dist")
            w_walk_idx = spatial.get("w_walk_idx")
            w_walk_val = spatial.get("w_walk_val")
            w5000_idx = spatial.get("w5000_idx")
            w5000_val = spatial.get("w5000_val")

            if w_walk_idx is None or parcel_dist is None:
                pop_catch = 0.0
                jobs_catch = 0.0
                feeder_pop_catch = 0.0
                feeder_jobs_catch = 0.0
                parcels_served = 0
                apm_trips = 0.0
                od_trips = 0.0
                origin_trips = 0.0
                flows = 0
                income_dict = {"SE01": 0.0, "SE02": 0.0, "SE03": 0.0}
                effective_car_speed = CAR_SPEED_KPH
                _apm_trips_by_period = {p: 0.0 for p in _PERIODS}
                _od_trips_by_period = {p: 0.0 for p in _PERIODS}
                _origin_trips_by_period = {p: 0.0 for p in _PERIODS}
                _bus_profile = None
                _feeder_profile = None
            else:
                pop_catch = _sparse_dot(pop_arr, w_walk_idx, w_walk_val)
                jobs_catch = _sparse_dot(jobs_arr, w_walk_idx, w_walk_val)
                if w5000_idx is not None and len(w5000_idx) > 0:
                    feeder_pop_catch = _sparse_dot(pop_arr, w5000_idx, w5000_val)
                    feeder_jobs_catch = _sparse_dot(jobs_arr, w5000_idx, w5000_val)
                else:
                    feeder_pop_catch = 0.0
                    feeder_jobs_catch = 0.0
                parcels_served = len(w_walk_idx)

                # LODES mode choice with current bus headway and feeder zone
                # Congestion feedback: car speed degrades as catchment grows.
                # On first period (no baseline yet), use default speed.
                if cid in self._base_catchment_pop:
                    _base_total = (
                        self._base_catchment_pop[cid]
                        + self._base_feeder_pop_catch.get(cid, 0.0)
                    )
                    _curr_total = pop_catch + feeder_pop_catch
                    if _base_total > 0:
                        _pop_growth = max(
                            (_curr_total - _base_total) / _base_total, 0.0
                        )
                        _cong_mult = 1.0 + CONGESTION_ELASTICITY * _pop_growth
                        effective_car_speed = CAR_SPEED_KPH / _cong_mult
                        # Bus speed also degrades with congestion, but less
                        # (fixed routes, some signal priority → 60% of car effect)
                        _bus_cong_mult = 1.0 + CONGESTION_ELASTICITY * BUS_CONGESTION_SHARE * _pop_growth
                        effective_bus_speed = bus_spd / _bus_cong_mult
                        # Persist for dynamic commute times in relocation MNL
                        self._congestion_factor = _cong_mult
                    else:
                        effective_car_speed = CAR_SPEED_KPH
                        effective_bus_speed = bus_spd
                else:
                    effective_car_speed = CAR_SPEED_KPH
                    effective_bus_speed = bus_spd

                # --- Per-period MNL: run compute_lodes_ridership 3× ---
                # Each period gets its own car speed (congestion×period factor)
                # and bus headway (from ServiceProfile).  This replaces the
                # post-hoc apply_time_of_day() approximation with the full MNL.
                _bus_profile = self._bus_service_profiles.get(cid)
                _feeder_profile = self._feeder_service_profiles.get(cid)
                _cong_profile = self._congestion_profiles.get(
                    cid, PERIOD_CONGESTION_FACTORS
                )

                _apm_trips_by_period = {}
                _od_trips_by_period = {}
                _origin_trips_by_period = {}
                _income_by_period = {}
                _common_lodes_kwargs = dict(
                    return_by_income=True,
                    _parcel_cache=self.parcel_cache,
                    _od_cache=self.od_cache,
                    _parcel_stop_dist=parcel_dist,
                    feeder_coverage_fraction=feeder_cov,
                    sector_coverage=sector_cov,
                    parking_costs=self._parking_costs,
                    institutional_weights=self._institutional_weights,
                    apm_headway_min=apm_hw,
                    transfer_walk_min=self._transfer_walk_min.get(cid, 0.0),
                    bus_speed_kph=effective_bus_speed,
                    stop_distance_matrix=spatial.get("stop_distance_matrix"),
                    _parcel_stop_idx=spatial.get("parcel_stop_idx"),
                    _corridor_length_km=spatial.get("corridor_length_km"),
                    _apm_effective_speed=spatial.get("apm_effective_speed"),
                    _apm_circuity=spatial.get("apm_circuity"),
                    tsp_speed_factor=self._tsp_speed_factors.get(cid, 1.0),
                    pop_active=self._pop_active,
                    mnl_coefficients=self._model_options.get("mnl_coefficients"),
                    transit_asc=self._transit_asc,
                )

                for _period in _PERIODS:
                    _period_car_speed = (
                        effective_car_speed * _cong_profile[_period]
                    )
                    # Bus headway varies by period via ServiceProfile
                    _period_bus_hw = (
                        _bus_profile.headway_for_period(_period)
                        if _bus_profile is not None else bus_hw
                    )
                    _period_feeder_hw = (
                        _feeder_profile.headway_for_period(_period)
                        if _feeder_profile is not None else feeder_hw
                    )
                    # Peak headway for commute MNL (AM peak uses am_peak_headway,
                    # other periods use period-appropriate headway)
                    _period_peak_hw = (
                        _bus_profile.am_peak_headway_min
                        if _bus_profile is not None and _period == "am_peak"
                        else None
                    )

                    (
                        _p_apm, _p_od, _p_origin, _p_flows,
                        _p_income, _p_dir_split,
                    ) = compute_lodes_ridership(
                        corridor,
                        self.parcels,
                        self.od_flows,
                        bus_headway_min=_period_bus_hw,
                        feeder_headway_min=_period_feeder_hw,
                        car_speed_kph=_period_car_speed,
                        bus_peak_headway_min=_period_peak_hw,
                        **_common_lodes_kwargs,
                    )
                    _apm_trips_by_period[_period] = _p_apm
                    _od_trips_by_period[_period] = _p_od
                    _origin_trips_by_period[_period] = _p_origin
                    _income_by_period[_period] = _p_income

                # Use last period's dir_split (stable across periods),
                # clipped to configured directional bounds
                _ds = _p_dir_split
                if hasattr(self, '_commute_direction_min') and self._commute_direction_min is not None:
                    _ds = max(_ds, self._commute_direction_min)
                if hasattr(self, '_commute_direction_max') and self._commute_direction_max is not None:
                    _ds = min(_ds, self._commute_direction_max)
                self._directional_split[cid] = _ds
                flows = _p_flows  # flows count is same across periods

                # Aggregate income dict: weight by work_commute period weights
                # (income segmentation comes from LODES commute flows)
                _work_pw = COMPONENT_PERIOD_WEIGHTS["work_commute"]
                income_dict = {"SE01": 0.0, "SE02": 0.0, "SE03": 0.0}
                for _period in _PERIODS:
                    _pw = _work_pw[_period]
                    for _seg in ("SE01", "SE02", "SE03"):
                        income_dict[_seg] += (
                            _income_by_period[_period].get(_seg, 0.0) * _pw
                        )

                # Aggregate apm_trips/od_trips/origin_trips for diagnostics
                # (weighted by work_commute periods for backward compatibility)
                apm_trips = sum(
                    _apm_trips_by_period[p] * _work_pw[p] for p in _PERIODS
                )
                od_trips = sum(
                    _od_trips_by_period[p] * _work_pw[p] for p in _PERIODS
                )
                origin_trips = sum(
                    _origin_trips_by_period[p] * _work_pw[p] for p in _PERIODS
                )

            # Snapshot base catchment on first encounter (year 0)
            if cid not in self._base_catchment_pop:
                self._base_catchment_pop[cid] = pop_catch
                self._base_catchment_jobs[cid] = jobs_catch
                self._base_feeder_pop_catch[cid] = feeder_pop_catch

            # ==============================================================
            # Three-component ridership model
            # ==============================================================
            #
            # Component 1 -- COMMUTE (LODES x all-day factor)
            # apm_trips from compute_lodes_ridership() counts one-way
            # commuters choosing APM in the logit.  x2 for round trip.
            # WORK_OFFPEAK_EXPANSION (1.15) covers reverse commute and
            # off-peak work trips only.  Non-work handled by Component 1b.
            # Scaled by catchment growth so TOD development feeds back.
            #
            # Component 2 -- STUDENT (campus catchment x trip rate)
            # Campus-affiliated trips not in LODES: class-to-class,
            # dorm-to-library, dining, recreation.  Uses institutional
            # weights to identify campus population near corridor.
            #
            # Component 3 -- SELF-SELECTION (on commute only)
            # TCRP 128: early TOD movers self-select for transit; bonus
            # decays as development scales (Michaelis-Menten kinetics).
            # Applied to commute component only -- students are already
            # inherently transit-favorable.
            # ==============================================================

            # LODES APM share (diagnostic)
            # Only trust when both values are substantial; otherwise use
            # conservative fallback to avoid penalizing data-sparse corridors
            used_lodes_share_fallback = False
            if od_trips > 5 and apm_trips > 0.1:
                avg_apm_share = np.clip(apm_trips / od_trips, 0.08, 0.60)
            else:
                avg_apm_share = 0.15
                used_lodes_share_fallback = True

            # Self-selection (TCRP 128) removed: the logit already computes
            # mode share at each location, and catchment_scale already grows
            # trip opportunities with TOD development.  Adding a separate
            # self-selection multiplier double-counted residential sorting.
            self_selection_mult = 1.0
            base_pop_c = self._base_catchment_pop[cid]

            # --- Component 1a: Both-ends work trips (LODES x off-peak expansion) ---
            # Period-weighted: each period's apm_trips reflects the MNL run
            # with that period's car speed and bus headway.  Employment
            # corridors gain from peak congestion pushing car utility down.
            base_jobs_c = self._base_catchment_jobs[cid]
            pop_growth_frac = (
                (pop_catch - base_pop_c) / base_pop_c
                if base_pop_c > 10 else 0.0
            )
            jobs_growth_frac = (
                (jobs_catch - base_jobs_c) / base_jobs_c
                if base_jobs_c > 10 else 0.0
            )
            avg_growth = 0.5 * (max(pop_growth_frac, 0.0) + max(jobs_growth_frac, 0.0))
            catchment_scale = min(1.0 + avg_growth, MAX_CATCHMENT_SCALE)
            _work_pw = COMPONENT_PERIOD_WEIGHTS["work_commute"]
            both_ends_daily = sum(
                _work_pw[p] * _apm_trips_by_period.get(p, 0.0)
                * 2.0 * WORK_OFFPEAK_EXPANSION * catchment_scale
                for p in _PERIODS
            )

            # --- Component 1b: Purpose-specific non-work trips (NHTS) ---
            # Each purpose has its own 4-mode MNL with purpose-specific
            # parameters (trip distance, parking cost, VOT multiplier,
            # transit ASC multiplier).  Period-weighted: off-peak dominates
            # (70%), where car speeds are high and APM advantage is smaller.
            # Campus pop excluded (handled by Component 2).
            origin_only_daily = 0.0
            campus_pop_catch_1b = 0.0  # will be set in Component 2 block
            if (
                self._institutional_weights is not None
                and w_walk_idx is not None and len(w_walk_idx) > 0
            ):
                campus_frac_arr_1b = np.clip(
                    (self._institutional_weights - 1.0) / 4.0, 0.0, 1.0
                )
                campus_pop_catch_1b = _sparse_dot(
                    pop_arr * campus_frac_arr_1b, w_walk_idx, w_walk_val
                )
            general_pop_catch = max(0.0, pop_catch - campus_pop_catch_1b)

            # Per-purpose MNL, period-weighted
            _nonwork_pw = COMPONENT_PERIOD_WEIGHTS["local_nonwork"]
            _apm_eff_spd_nw = max(spatial.get("apm_effective_speed", 30.0), 1.0)
            for _purpose_name, _purpose_spec in NONWORK_TRIP_PURPOSES.items():
                _purpose_daily = 0.0
                for _period in _PERIODS:
                    _period_car_spd_nw = (
                        effective_car_speed * _cong_profile[_period]
                    )
                    _period_bus_hw_nw = (
                        _bus_profile.headway_for_period(_period)
                        if _bus_profile is not None else bus_hw
                    )
                    # Purpose-specific MNL with own VOT, distance, parking
                    _nw_share = self._compute_nonwork_apm_share(
                        purpose_spec=_purpose_spec,
                        apm_headway_min=apm_hw,
                        bus_headway_min=_period_bus_hw_nw,
                        apm_effective_speed=_apm_eff_spd_nw,
                        car_speed_kph=_period_car_spd_nw,
                    )
                    _purpose_daily += (
                        general_pop_catch * _purpose_spec["rate"]
                        * _nw_share
                        * NONWORK_TRIP_CAPTURE_FRACTION
                        * _nonwork_pw[_period]
                    )
                origin_only_daily += _purpose_daily

            commute_apm_daily = both_ends_daily + origin_only_daily

            # --- Component 2: Student (campus catchment via logit) ---
            # Uses enrollment-based campus presence instead of residential
            # population.  University buildings have near-zero pop_arr but
            # thousands of daily users.  PURDUE_TOTAL is distributed across
            # campus-weighted parcels and distance-weighted by weights_1200.
            student_apm_daily = 0.0
            student_apm_share = 0.0
            campus_alignment = 0.0
            _student_purpose_daily = {}
            dest_coverage = 0.0
            if (
                self._institutional_weights is not None
                and w_walk_idx is not None and len(w_walk_idx) > 0
                and parcel_dist is not None
            ):
                from src.data.purdue_transit_demand import (
                    PURDUE_ENROLLMENT,
                    PURDUE_FACULTY_STAFF,
                )
                # Transit-eligible campus presence: not all 67K campus
                # people generate APM-relevant trips.  Factors:
                #   Daily attendance: ~65% (35% absent on a given day)
                #   Transit-eligible trip length: ~40% of campus trips
                #     are long enough (>400m) for APM vs walking
                #   Faculty/staff: mostly drive+park, low transit share
                # Net: 25% of students, 10% of faculty/staff
                _spf = float(self._model_options.get(
                    "student_presence_factor", STUDENT_PRESENCE_FACTOR))
                CAMPUS_TOTAL = (
                    PURDUE_ENROLLMENT * _spf
                    + PURDUE_FACULTY_STAFF * FACULTY_PRESENCE_FACTOR
                )  # ~14,560 at default 0.25

                # Weight by (institutional_weight - 1.0): university bldgs=4.0,
                # student housing=2-3, campus-adjacent=0.5, non-campus=0
                campus_weight_arr = np.maximum(
                    self._institutional_weights - 1.0, 0.0
                )
                total_campus_weight = float(campus_weight_arr.sum())
                if total_campus_weight > 0:
                    # Fraction of campus presence within walk catchment
                    weighted_catch = _sparse_dot(
                        campus_weight_arr, w_walk_idx, w_walk_val
                    )
                    campus_pop_catch = CAMPUS_TOTAL * (
                        weighted_catch / total_campus_weight
                    )
                else:
                    campus_pop_catch = 0.0

                # Corridor-campus alignment: fraction of campus served
                campus_alignment = self._compute_campus_alignment(parcel_dist)

                # Dynamic directional split: campus corridors have stronger
                # peak-direction peaking (students commuting to campus).
                # LODES OD split already set; bias upward for high alignment.
                if campus_alignment > 0.6:
                    self._directional_split[cid] = max(
                        self._directional_split[cid], 0.72)
                elif campus_alignment > 0.3:
                    self._directional_split[cid] = max(
                        self._directional_split[cid], 0.65)
                # else: keep LODES-derived split (already clipped 0.50-0.80)
                # Re-clip to configured directional bounds
                if hasattr(self, '_commute_direction_max') and self._commute_direction_max is not None:
                    self._directional_split[cid] = min(
                        self._directional_split[cid], self._commute_direction_max)

                # Campus-to-nearest-station distance (meters)
                campus_parcel_mask = self._institutional_weights >= 3.0
                if campus_parcel_mask.any() and parcel_dist is not None:
                    campus_dists = parcel_dist[campus_parcel_mask]
                    campus_to_station_m = float(np.percentile(campus_dists, 25))
                else:
                    # Fallback: use proximity from campus center to stops
                    from pyproj import Transformer
                    _tf = Transformer.from_crs("EPSG:4326", PROJECT_CRS, always_xy=True)
                    # Convert campus center from EPSG:2965 feet to meters
                    cx, cy = _tf.transform(self._CAMPUS_LON, self._CAMPUS_LAT)
                    cx *= US_SURVEY_FT_TO_M
                    cy *= US_SURVEY_FT_TO_M
                    stops_proj = spatial.get("stops_proj")  # already in meters
                    if stops_proj is not None and len(stops_proj) > 0:
                        d = np.sqrt(((stops_proj - [cx, cy])**2).sum(axis=1))
                        campus_to_station_m = float(d.min())
                    else:
                        campus_to_station_m = 5000.0

                # Corridor-specific student APM share via logit
                _cmeta = self._corridor_meta[cid]
                corr_len_km = _cmeta["length_km"]
                corr_n_stops = _cmeta["n_stops"]

                # Student TOD feedback: as TOD brings more housing near
                # stations, some is student housing → more students in catchment
                student_tod_scale = 1.0
                if base_pop_c > 0 and pop_catch > base_pop_c:
                    # Student pop grows at 30% of general growth rate,
                    # capped at 50% increase (student housing is a small
                    # fraction of TOD, and Purdue enrollment is capacity-limited)
                    growth_ratio = min(pop_catch / base_pop_c - 1.0, 1.67)
                    student_tod_scale = 1.0 + 0.30 * growth_ratio  # max 1.50

                # campus_alignment is retained as a diagnostic metric but NOT
                # multiplied into ridership.  Distance is already penalized
                # twice: (1) weights_1200 decay in campus_pop_catch, and
                # (2) access time in the student_apm_share logit.  Adding
                # campus_alignment tripled the distance penalty.

                # Purpose-split student ridership: each sub-purpose has its
                # own trip distance (affecting MNL walk competitiveness) and
                # coverage requirement (off-campus destinations).
                # Period-weighted: student MNL runs per-period with period car
                # speeds and bus headways.  Students are midday-dominant (55%
                # off-peak) so corridors gain less from peak congestion here.
                student_apm_daily = 0.0
                _student_purpose_daily = {}
                _share_weighted_sum = 0.0
                _bus_share_weighted_sum = 0.0
                _rate_sum = 0.0
                dest_coverage = spatial.get("dest_coverage", 0.0)

                _str_mult = float(self._model_options.get(
                    "student_trip_rate_multiplier", 1.0))
                _student_pw = COMPONENT_PERIOD_WEIGHTS["student"]

                for _purpose, _spec in STUDENT_TRIP_PURPOSES.items():
                    _purpose_daily_total = 0.0
                    _purpose_share_accum = 0.0
                    _bus_share_accum = 0.0
                    for _period in _PERIODS:
                        _period_car_spd = (
                            effective_car_speed * _cong_profile[_period]
                        )
                        _period_bus_hw_s = (
                            _bus_profile.headway_for_period(_period)
                            if _bus_profile is not None else bus_hw
                        )
                        _purpose_share, _purpose_bus_share = self._compute_student_apm_share(
                            campus_to_station_m=campus_to_station_m,
                            apm_headway_min=apm_hw,
                            bus_headway_min=_period_bus_hw_s,
                            corridor_length_km=corr_len_km,
                            n_stops=corr_n_stops,
                            car_speed_kph=_period_car_spd,
                            trip_dist_km=_spec["dist_km"],
                            curve_delay_s=float(_cmeta.get("curve_delay_s", 0.0)),
                        )
                        _p_daily = (
                            campus_pop_catch * _spec["rate"] * _str_mult
                            * _purpose_share * student_tod_scale
                            * _student_pw[_period]
                        )
                        if _spec["needs_dest_check"]:
                            _p_daily *= dest_coverage
                        _purpose_daily_total += _p_daily
                        _purpose_share_accum += _purpose_share * _student_pw[_period]
                        _bus_share_accum += _purpose_bus_share * _student_pw[_period]

                    _student_purpose_daily[_purpose] = _purpose_daily_total
                    student_apm_daily += _purpose_daily_total
                    _share_weighted_sum += _spec["rate"] * _purpose_share_accum
                    _bus_share_weighted_sum += _spec["rate"] * _bus_share_accum
                    _rate_sum += _spec["rate"]

                # Weighted-average share across purposes and periods (diagnostic)
                student_apm_share = _share_weighted_sum / _rate_sum if _rate_sum > 0 else 0.0
                student_bus_share = _bus_share_weighted_sum / _rate_sum if _rate_sum > 0 else 0.0

                # --- Validation cross-checks (D1-D4) ---
                # Emit each warning category once per model run, not per corridor×year.
                if campus_pop_catch > 0:
                    # D1: CityBus floor — implied bus demand vs NTD observed
                    # BRT draws from existing bus riders more than grade-separated
                    # APM, so BRT floor is lower (NTD: BRT mode share ~60% of rail).
                    _CITYBUS_FLOOR = 0.12 if self._transit_mode_name == "brt" else 0.20
                    _implied_bus_per_person = (
                        _rate_sum * student_bus_share * _str_mult
                    )
                    if _implied_bus_per_person < _CITYBUS_FLOOR:
                        if not getattr(self, "_d1_warned", False):
                            self._d1_warned = True
                            import warnings
                            warnings.warn(
                                f"Student transit demand ({_implied_bus_per_person:.2f} "
                                f"trips/person) below CityBus floor ({_CITYBUS_FLOOR})",
                                stacklevel=2,
                            )

                    # D4: Intracampus walk dominance — walk should dominate at 0.8km
                    _intra_share = _student_purpose_daily.get("intracampus", 0.0)
                    _intra_rate = STUDENT_TRIP_PURPOSES["intracampus"]["rate"]
                    if _intra_rate > 0 and campus_pop_catch > 0:
                        _intra_apm_frac = _intra_share / (
                            campus_pop_catch * _intra_rate * _str_mult * student_tod_scale
                        ) if (campus_pop_catch * _intra_rate * _str_mult * student_tod_scale) > 0 else 0
                        if _intra_apm_frac > 0.40:
                            if not getattr(self, "_d4_warned", False):
                                self._d4_warned = True
                                import warnings
                                warnings.warn(
                                    f"Intracampus APM share {_intra_apm_frac:.0%} exceeds 40% "
                                    f"-- walk should dominate at 0.8km",
                                    stacklevel=2,
                                )

            # --- Component 4: Non-commute trip generators ---
            # Institutional attractors not captured by residential generation:
            # medical (IU Health Arnett), retail near stations, events.
            # No overlap deduction needed: Component 1a is work-only, and
            # Component 1b is residential generation (different population
            # base from attraction-based generator trips).
            generator_daily = 0.0
            if w_walk_idx is not None and len(w_walk_idx) > 0 and parcel_dist is not None:
                # Retail/service jobs near stations attract shopping trips.
                # Only customer-facing jobs generate walk-in trips; filter
                # total jobs by RETAIL_JOB_FRACTION (~20%).
                total_jobs_catch = _sparse_dot(jobs_arr, w_walk_idx, w_walk_val)
                retail_jobs_catch = total_jobs_catch * RETAIL_JOB_FRACTION
                retail_trips = retail_jobs_catch * RETAIL_TRIP_RATE

                # Medical + event trips scaled by corridor proximity to
                # downtown/campus where these generators concentrate
                generator_proximity = 0.0
                if parcel_dist is not None:
                    # Use median distance of served parcels as proxy
                    served_mask = parcel_dist <= WALK_CATCHMENT_M
                    if served_mask.any():
                        med_dist = float(np.median(parcel_dist[served_mask]))
                        generator_proximity = np.exp(-GENERATOR_PROXIMITY_DECAY_BETA * med_dist)

                fixed_trips = (MEDICAL_DAILY_TRIPS + EVENT_DAILY_EQUIVALENT) * generator_proximity
                # Generator APM share via MNL (mode-sensitive) instead of
                # fixed constant.  Uses "personal" purpose spec as proxy
                # (medical/errand trips, 2.5 km, $1 parking).
                _gen_share = self._compute_nonwork_apm_share(
                    purpose_spec=NONWORK_TRIP_PURPOSES["personal"],
                    apm_headway_min=apm_hw,
                    bus_headway_min=bus_hw,
                    apm_effective_speed=max(
                        spatial.get("apm_effective_speed", 30.0), 1.0),
                    car_speed_kph=effective_car_speed,
                )
                generator_daily = (retail_trips + fixed_trips) * _gen_share

            # --- Component 5: Induced demand (trip rate elasticity) ---
            # Better transit service induces new non-commute trips that
            # would not have occurred without the APM (TCRP 95).
            # Applied to origin-only + generator components only --
            # LODES commute OD pairs and student rates are fixed.
            induced_daily = 0.0
            if year >= INDUCED_DEMAND_THRESHOLD_YEAR:
                non_commute_base = origin_only_daily + generator_daily
                if non_commute_base > 0:
                    # Service quality signal: how close to mature ridership
                    pre_induced = (
                        commute_apm_daily * self_selection_mult
                        + student_apm_daily
                        + non_commute_base
                    )
                    service_quality = np.clip(
                        pre_induced / MATURE_RIDERSHIP_TARGET, 0.0, 1.0
                    )
                    induced_daily = (
                        non_commute_base
                        * INDUCED_TRIP_ELASTICITY
                        * np.log1p(service_quality)
                    )

            # --- Component 6: Latent demand (zero-car households) ---
            # Zero-car HH suppress discretionary trips due to poor
            # mobility.  New APM service releases some suppressed trips.
            # Income-differentiated: low-wage HH have much higher zero-car
            # rates (ACS B08201).  Weight by income segment shares from LODES.
            #
            # Zero-car MNL: car removed from choice set (IIA).  These HH
            # can't drive, so APM share is higher than general-population
            # avg_apm_share.  Walk zone uses 3-mode MNL (APM, bus, walk);
            # feeder zone uses APM+feeder vs bus (walk not viable at those
            # distances), scaled by feeder_cov and w5000_val decay.
            from scripts.generate_improved_ridership import (
                BETA_IVT as _B_IVT_default, BETA_WAIT as _B_WAIT_default,
                BETA_ACCESS as _B_ACC_default, BETA_COST as _B_COST_default,
                ASC_BUS as _ASC_BUS,
                APM_FARE as _APM_FARE, BUS_FARE as _BUS_FARE,
                WALK_SPEED_KPH as _WALK_SPD, WALK_CIRCUITY as _WALK_CIRC,
                BUS_CIRCUITY as _BUS_CIRC,
                FEEDER_BUS_SPEED_KPH as _FEED_SPD,
                FEEDER_TRANSFER_DISUTILITY as _FEED_DISUTIL,
                TRANSFER_FARE as _XFER_FARE,
                effective_wait_time as _eff_wait,
                compute_transfer_penalty as _xfer_pen,
            )
            from src.mode_choice import (
                DEFAULT_INCOME_SEGMENT_COEFFICIENTS as _ISC,
            )
            # Use sampled MNL coefficients if available, otherwise defaults
            _mnl_zc = self._model_options.get("mnl_coefficients", None)
            if _mnl_zc is not None:
                _B_IVT, _B_WAIT, _B_ACC, _B_COST = _mnl_zc
            else:
                _B_IVT, _B_WAIT, _B_ACC, _B_COST = (
                    _B_IVT_default, _B_WAIT_default, _B_ACC_default, _B_COST_default
                )
            # Use transit-mode-appropriate ASC (BRT=0.05 vs APM=0.12)
            _transit_asc = self._transit_asc

            # -- Walk-zone zero-car MNL (representative 400m trip) --
            # Period-weighted: bus headway varies by period (peak is more
            # frequent → APM competes harder with bus → lower zero-car APM
            # share in peaks).  APM headway constant across periods.
            _rep_km_w = 0.4  # weighted median walk-zone trip
            _rep_access_min = (0.4 * _WALK_CIRC / _WALK_SPD) * 60.0
            _apm_eff_spd = max(spatial.get("apm_effective_speed", 30.0), 1.0)
            _latent_pw = COMPONENT_PERIOD_WEIGHTS["latent"]

            # Income shares for feeder-zone MNL weighting.
            # Derive from LODES income dict; fall back to national average.
            _inc_total_lat = sum(income_dict.values())
            if _inc_total_lat > 0:
                _lat_inc_shares = {
                    s: income_dict[s] / _inc_total_lat for s in ("SE01", "SE02", "SE03")
                }
            else:
                _lat_inc_shares = FALLBACK_INCOME_SHARES

            _zc_walk_share = 0.0
            _zc_feeder_share = 0.0
            for _period in _PERIODS:
                _lp_bus_hw = (
                    _bus_profile.headway_for_period(_period)
                    if _bus_profile is not None else bus_hw
                )
                _lp_feeder_hw = (
                    _feeder_profile.headway_for_period(_period)
                    if _feeder_profile is not None else feeder_hw
                )
                # Walk-zone 3-mode MNL (no car)
                _u_apm_zc = (
                    _B_IVT * (_rep_km_w / _apm_eff_spd) * 60.0
                    + _B_WAIT * (apm_hw / 2.0)
                    + _B_ACC * _rep_access_min
                    + _B_COST * _APM_FARE + _transit_asc
                )
                _u_bus_zc = (
                    _B_IVT * (_rep_km_w * _BUS_CIRC / bus_spd) * 60.0
                    + _B_WAIT * _eff_wait(_lp_bus_hw)
                    + _B_ACC * _rep_access_min * 1.3
                    + _B_COST * _BUS_FARE + _ASC_BUS
                )
                _u_walk_zc = _B_IVT * (_rep_km_w * _WALK_CIRC / _WALK_SPD) * 60.0 + 0.05
                _e_a_zc = np.exp(np.clip(_u_apm_zc, -20, 20))
                _e_b_zc = np.exp(np.clip(_u_bus_zc, -20, 20))
                _e_w_zc = np.exp(np.clip(_u_walk_zc, -20, 20))
                _zc_walk_share += float(_e_a_zc / (_e_a_zc + _e_b_zc + _e_w_zc)) * _latent_pw[_period]

                # Feeder-zone 2-mode MNL (APM+feeder vs bus)
                # Income-segmented: loop over SE01/SE02/SE03 with segment-
                # specific cost sensitivity and transit ASC adjustments.
                # Population-weighted mean distance of feeder-zone parcels
                # to their nearest station (replaces hardcoded 2.5 km).
                _f_dists = parcel_dist[w5000_idx]
                _f_pops = pop_arr[w5000_idx]
                if len(_f_dists) > 0 and _f_pops.sum() > 0:
                    _rep_km_f = float(np.clip(
                        np.average(_f_dists, weights=_f_pops) / 1000.0,
                        1.0, 5.0,
                    ))
                else:
                    _rep_km_f = 2.5  # fallback
                _feed_bus_ivt = (_rep_km_f * _WALK_CIRC / _FEED_SPD) * 60.0
                _apm_ivt_f = (_rep_km_f / _apm_eff_spd) * 60.0
                _tsp_f = self._tsp_speed_factors.get(cid, 1.0)
                _xpen = _xfer_pen(
                    apm_hw, _lp_feeder_hw,
                    tsp_active=_tsp_f > 1.0,
                    pop_active=self._pop_active,
                )
                _bus_access_f = 2.0 * (0.4 * _WALK_CIRC / _WALK_SPD) * 60.0 * 1.3
                _period_feeder_share = 0.0
                for _seg in ("SE01", "SE02", "SE03"):
                    _sc = _ISC[_seg]
                    _s_B_IVT = _B_IVT * _sc["ivt_mult"]
                    _s_B_WAIT = _B_WAIT * _sc["wait_mult"]
                    _s_B_ACC = _B_ACC * _sc["access_mult"]
                    _s_B_COST = _B_COST * _sc["cost_mult"]
                    _u_apmf_zc = (
                        _s_B_IVT * (_feed_bus_ivt + _apm_ivt_f)
                        + _s_B_WAIT * (apm_hw / 2.0 + _eff_wait(_lp_feeder_hw))
                        + _s_B_ACC * _xpen
                        + _s_B_COST * (_APM_FARE + _XFER_FARE)
                        + _transit_asc + _sc.get("asc_apm", 0.0) + _FEED_DISUTIL
                    )
                    _u_bus_f_zc = (
                        _s_B_IVT * (_rep_km_f * _BUS_CIRC / bus_spd) * 60.0
                        + _s_B_WAIT * _eff_wait(_lp_bus_hw)
                        + _s_B_ACC * _bus_access_f
                        + _s_B_COST * _BUS_FARE + _ASC_BUS + _sc.get("asc_bus", 0.0)
                    )
                    _e_af_zc = np.exp(np.clip(_u_apmf_zc, -20, 20))
                    _e_bf_zc = np.exp(np.clip(_u_bus_f_zc, -20, 20))
                    _seg_share = float(_e_af_zc / (_e_af_zc + _e_bf_zc))
                    _period_feeder_share += _seg_share * _lat_inc_shares[_seg]
                _zc_feeder_share += _period_feeder_share * _latent_pw[_period]

            latent_daily = 0.0
            latent_by_income = {"SE01": 0.0, "SE02": 0.0, "SE03": 0.0}
            _inc_total = sum(income_dict.values())
            # Per-sector feeder population: partition feeder-zone pop by
            # bearing sector so each sector's coverage multiplies only
            # the population in that direction.  A corridor with great
            # coverage toward dense residential and none toward farmland
            # scores higher than uniform-average coverage would suggest.
            _w5000_sector = spatial.get("w5000_sector")
            _has_sector_detail = (
                sector_cov is not None
                and hasattr(sector_cov, 'coverage')
                and _w5000_sector is not None
                and len(_w5000_sector) > 0
                and w5000_idx is not None
                and len(w5000_idx) > 0
            )
            if _has_sector_detail:
                # Compute coverage-weighted feeder pop: sum over sectors
                # of (sector_pop × sector_coverage), using the same
                # w5000_val decay weights as feeder_pop_catch.
                _f_pop = pop_arr[w5000_idx] * w5000_val
                _sector_pop = np.bincount(
                    _w5000_sector, weights=_f_pop, minlength=8
                )
                _sector_weighted_feeder_pop = float(
                    np.dot(_sector_pop[:8], sector_cov.coverage[:8])
                )
            else:
                # Fallback: scalar coverage × total feeder pop
                _sector_weighted_feeder_pop = feeder_pop_catch * feeder_cov

            if (pop_catch > 0 or feeder_pop_catch > 0) and (
                _zc_walk_share > 0 or _zc_feeder_share > 0
            ):
                # Use population-based income shares for latent demand, not
                # LODES APM trip shares.  income_dict reflects commute APM
                # trips per segment — low-income workers are underrepresented
                # in LODES (work outside catchment), making SE01 share tiny
                # and latent demand negligible (~0.6 trips/day vs ~329).
                _inc_shares = FALLBACK_INCOME_SHARES

                # Weighted zero-car rate by income segment
                for seg, _zc_rate_raw in ZERO_CAR_BY_INCOME.items():
                    zc_rate = min(_zc_rate_raw * self._zero_car_mult, 1.0)
                    seg_share = _inc_shares.get(seg, 0.33)
                    # Walk zone: zero-car MNL APM share (no car in choice set)
                    walk_latent = (
                        pop_catch * seg_share * zc_rate
                        * ZERO_CAR_SUPPRESSED_TRIPS
                        * LATENT_TRIP_RELEASE_RATE
                        * _zc_walk_share
                    )
                    # Feeder zone: per-sector coverage × per-sector pop.
                    # _sector_weighted_feeder_pop already incorporates
                    # directional coverage and w5000_val decay.
                    # Discounted release rate: transfer penalty reduces the
                    # effective mobility gain for feeder-zone zero-car HH.
                    feeder_latent = (
                        _sector_weighted_feeder_pop * seg_share * zc_rate
                        * ZERO_CAR_SUPPRESSED_TRIPS
                        * LATENT_TRIP_RELEASE_RATE
                        * LATENT_FEEDER_RELEASE_DISCOUNT
                        * _zc_feeder_share
                    )
                    seg_latent = walk_latent + feeder_latent
                    latent_by_income[seg] = seg_latent
                    latent_daily += seg_latent

            # Service-quality gate: scale latent demand by corridor
            # effectiveness.  Weak corridors with few pre-latent riders
            # don't provide enough destination coverage to release
            # suppressed trips from zero-car households.
            pre_latent = (
                commute_apm_daily * self_selection_mult
                + student_apm_daily + generator_daily + induced_daily
            )
            latent_quality = float(np.clip(
                pre_latent / LATENT_MATURITY_TARGET,
                LATENT_QUALITY_FLOOR, 1.0,
            ))
            if latent_quality < 1.0:
                latent_daily *= latent_quality
                for seg in latent_by_income:
                    latent_by_income[seg] *= latent_quality

            # --- Combined daily ridership ---
            base_riders = (
                commute_apm_daily * self_selection_mult
                + student_apm_daily
                + generator_daily
                + induced_daily
                + latent_daily
            )

            # Period-weighted MNL replaces post-hoc apply_time_of_day().
            # Each component already incorporates period-specific car speeds
            # and bus headways via COMPONENT_PERIOD_WEIGHTS, so no further
            # time-of-day adjustment is needed.

            # Apply component-specific awareness ramp
            # Students see APM on campus daily → faster awareness buildup
            # Generators (medical, retail) adopt quickly via signage
            # Commuters/latent follow the general logistic ramp
            student_awareness = 1.0 / (
                1.0 + np.exp(-RAMP_STEEPNESS * (year - RAMP_MIDPOINT + STUDENT_AWARENESS_ADVANCE_YR))
            )
            generator_awareness = 1.0 / (
                1.0 + np.exp(-RAMP_STEEPNESS * (year - RAMP_MIDPOINT + GENERATOR_AWARENESS_ADVANCE_YR))
            )
            # Weight awareness by component contribution
            if base_riders > 0:
                aware_riders = (
                    commute_apm_daily * self_selection_mult * awareness
                    + student_apm_daily * student_awareness
                    + generator_daily * generator_awareness
                    + induced_daily * awareness
                    + latent_daily * awareness
                )
                daily_riders = aware_riders * self.ridership_scale_multiplier
            else:
                daily_riders = 0.0

            # Component-specific seasonal adjustment (constants at module level).
            # Work (1a) and non-work (1b) get different annual factors.
            work_portion = both_ends_daily * self_selection_mult
            nonwork_portion = origin_only_daily * self_selection_mult
            student_portion = student_apm_daily
            generator_portion = generator_daily
            induced_portion = induced_daily
            latent_portion = latent_daily
            if base_riders > 0:
                work_annual = work_portion * COMMUTE_ANNUAL_FACTOR
                nonwork_annual = nonwork_portion * NONWORK_ANNUAL_FACTOR
                student_annual = student_portion * STUDENT_ANNUAL_FACTOR
                generator_annual = generator_portion * GENERATOR_ANNUAL_FACTOR
                induced_annual = induced_portion * INDUCED_ANNUAL_FACTOR
                latent_annual = latent_portion * LATENT_ANNUAL_FACTOR
                blended_factor = (
                    (work_annual + nonwork_annual + student_annual
                     + generator_annual + induced_annual + latent_annual)
                    / base_riders
                )
            else:
                blended_factor = COMMUTE_ANNUAL_FACTOR
            daily_riders_annual_avg = daily_riders * blended_factor

            # --- Awareness-adjusted component values ---
            # Scale each raw component by its awareness ramp so stored values
            # sum to daily_riders.  Period weighting is already embedded in
            # each component (no post-hoc TOD adjustment needed).
            _scale = self.ridership_scale_multiplier
            _work_aware = both_ends_daily * self_selection_mult * awareness * _scale
            _nonwork_aware = origin_only_daily * self_selection_mult * awareness * _scale
            _student_aware = student_apm_daily * student_awareness * _scale
            _generator_aware = generator_daily * generator_awareness * _scale
            _induced_aware = induced_daily * awareness * _scale
            _latent_aware = latent_daily * awareness * _scale

            # Non-campus ridership: total minus student component.
            # Shows which corridors serve the broader community vs. just Purdue.
            _non_campus_daily = daily_riders - _student_aware

            # --- Income segment allocation ---
            # Commute component: use LODES income distribution
            # Student component: Purdue skews low income (ACS 2022 Tippecanoe
            #   18-24yr: ~70% below poverty).  Allocate 65/25/10.
            # Generator/induced/latent: use population-based fallback shares.
            _STUDENT_INCOME = {"SE01": 0.65, "SE02": 0.25, "SE03": 0.10}
            _FALLBACK_INCOME = {"SE01": 0.35, "SE02": 0.35, "SE03": 0.30}
            _riders_by_income = {"SE01": 0.0, "SE02": 0.0, "SE03": 0.0}
            # Commute (work + non-work): LODES distribution
            _commute_aware = _work_aware + _nonwork_aware
            if _inc_total > 0:
                for _seg in ("SE01", "SE02", "SE03"):
                    _riders_by_income[_seg] += (
                        income_dict.get(_seg, 0.0) / _inc_total * _commute_aware
                    )
            else:
                for _seg in ("SE01", "SE02", "SE03"):
                    _riders_by_income[_seg] += _FALLBACK_INCOME[_seg] * _commute_aware
            # Student: campus income distribution
            for _seg in ("SE01", "SE02", "SE03"):
                _riders_by_income[_seg] += _STUDENT_INCOME[_seg] * _student_aware
            # Generator + induced + latent: population fallback
            _other_aware = _generator_aware + _induced_aware + _latent_aware
            for _seg in ("SE01", "SE02", "SE03"):
                _riders_by_income[_seg] += _FALLBACK_INCOME[_seg] * _other_aware

            ridership[cid] = {
                "daily_riders": daily_riders,
                "daily_riders_annual_avg": daily_riders_annual_avg,
                "base_riders": base_riders,
                # Awareness-adjusted display components (sum to daily_riders)
                "work_commute_daily": _work_aware,
                "local_nonwork_daily": _nonwork_aware,
                "campus_daily": _student_aware,
                "destination_daily": _generator_aware,
                "equity_daily": _induced_aware,
                "latent_aware_daily": _latent_aware,
                "non_campus_daily": _non_campus_daily,
                # Raw (pre-awareness) components for diagnostics
                "commute_apm_daily": commute_apm_daily,
                "both_ends_daily": both_ends_daily,
                "origin_only_daily": origin_only_daily,
                "student_apm_daily": student_apm_daily,
                "student_home_campus_daily": _student_purpose_daily.get("home_to_campus", 0.0),
                "student_offcampus_daily": _student_purpose_daily.get("campus_to_offcampus", 0.0),
                "student_intracampus_daily": _student_purpose_daily.get("intracampus", 0.0),
                "student_dest_coverage": dest_coverage,
                "student_apm_share": student_apm_share,
                "student_non_apm_share": (1.0 - student_apm_share) if student_apm_share > 0 else 0.0,
                "campus_alignment": campus_alignment,
                "generator_daily": generator_daily,
                "induced_daily": induced_daily,
                "latent_daily": latent_daily,
                "awareness": awareness,
                "student_awareness": student_awareness,
                "generator_awareness": generator_awareness,
                "apm_share": avg_apm_share,
                "pop_catch": pop_catch,
                "campus_pop_catch": campus_pop_catch,
                "jobs_catch": jobs_catch,
                "feeder_pop_catch": feeder_pop_catch,
                "feeder_jobs_catch": feeder_jobs_catch,
                "feeder_headway": feeder_hw,
                "feeder_coverage": feeder_cov,
                "feeder_coverage_sector_min": float(
                    sector_cov.coverage.min() if sector_cov is not None else feeder_cov
                ),
                "feeder_coverage_sector_max": float(
                    sector_cov.coverage.max() if sector_cov is not None else feeder_cov
                ),
                "feeder_coverage_sector_std": float(
                    sector_cov.coverage.std() if sector_cov is not None else 0.0
                ),
                "parcels_served": parcels_served,
                "lodes_apm_trips": float(apm_trips),
                "lodes_od_trips": float(od_trips),
                "lodes_origin_trips": float(origin_trips),
                "lodes_flows_captured": int(flows),
                "income_apm_trips": income_dict,
                # Equity metrics: component-aware income allocation
                "riders_SE01": _riders_by_income["SE01"],
                "riders_SE02": _riders_by_income["SE02"],
                "riders_SE03": _riders_by_income["SE03"],
                "latent_SE01": latent_by_income.get("SE01", 0.0),
                "latent_SE02": latent_by_income.get("SE02", 0.0),
                "latent_SE03": latent_by_income.get("SE03", 0.0),
                "low_income_access_ratio": (
                    _riders_by_income["SE01"]
                    / max(_riders_by_income["SE03"], 1e-6)
                ),
                "used_lodes_share_fallback": bool(used_lodes_share_fallback),
                "used_lodes_fallback": bool(used_lodes_share_fallback),
            }
            fallback_tag = " [fallback]" if ridership[cid]["used_lodes_fallback"] else ""
            logger.info(
                f"  {cid}: {daily_riders:,.0f} riders/day "
                f"(awareness={awareness:.0%}){fallback_tag}"
            )

        return ridership

    # ------------------------------------------------------------------
    # Accessibility scoring
    # ------------------------------------------------------------------

    def _compute_accessibility(self, ridership: dict) -> dict:
        """Compute parcel-level accessibility score from corridor ridership.

        accessibility_i = sum_s(ridership_s * exp(-beta * dist_is)) / reference
        where s indexes stops and dist_is is parcel-to-stop distance.

        Uses a reference of 10,000 daily riders (same as the rent premium
        quality factor reference) so that corridors in the 4K-11K range
        have meaningful spread instead of all saturating at 1.0.
        The score is NOT clipped at 1.0 \u2014 corridors exceeding the
        reference get proportionally higher accessibility, which feeds
        into the location choice logit via log1p(acc) where diminishing
        returns are built into the log transform.

        **End-to-end diminishing returns (ridership \u2192 development):**

        This first link (ridership \u2192 accessibility) is intentionally linear.
        Concavity is introduced downstream at two points:

        1. Rent premium: ``quality = sqrt(riders / 20,000)``
           (_prepare_proforma_df) \u2014 sqrt is concave, so the marginal
           rent premium from each additional rider decreases.

        2. Relocation MNL: ``beta_transit * log1p(accessibility)``
           (relocation_model.py) \u2014 log1p is concave, so the marginal
           utility of accessibility for household location choice
           decreases as accessibility grows.

        Applying concavity at *both* the accessibility computation *and*
        its downstream consumers would double-compress the signal and
        underweight the meaningful difference between, say, 2K and 8K
        daily riders.  The current architecture preserves linear
        proportionality at this step so that the downstream sqrt and
        log1p transforms operate on an uncompressed input.
        """
        raw_access = np.zeros(len(self.parcels), dtype=np.float64)

        for cid, spatial in self._corridor_spatial_cache.items():
            riders = ridership.get(cid, {}).get("daily_riders", 0)
            if riders <= 0:
                continue
            w_walk_idx = spatial.get("w_walk_idx")
            w_walk_val = spatial.get("w_walk_val")
            if w_walk_idx is None or len(w_walk_idx) == 0:
                continue

            _ACC_REFERENCE_RIDERS = 10000.0
            raw_access[w_walk_idx] += riders * w_walk_val / _ACC_REFERENCE_RIDERS

        access_scores = {cid: raw_access for cid in ridership}
        return access_scores
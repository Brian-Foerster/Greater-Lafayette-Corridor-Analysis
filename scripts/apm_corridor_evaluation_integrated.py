"""APM corridor evaluation with integrated static + dynamic financial analysis."""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scripts.financial_corridor_ranking import FinancialCorridorRanker, integrate_dynamic_ridership_data
from scripts.parcel_development_allocation import load_corridor_data
from scripts.station_area_delineation import TIFDistrictGenerator

from src.finance import annual_ridership_to_revenue_musd, annualize_daily_ridership_series, npv_irr
from src.financial_params import (
    compute_capital_cost, BOND_RATE, DEBT_TERM_YEARS, OPERATING_DAYS_PER_YEAR,
    TIF_AREA_TYPE_DEFAULT,
    APM_MODE as _APM_MODE,
    BRT_MODE as _BRT_MODE,
    SERVICE_DAYS_PER_YEAR as _SERVICE_DAYS,
    compute_apm_annual_vehicle_hours,
    compute_brt_annual_vehicle_hours,
    PROPERTY_TAX_RATE,
    TIF_CAPTURE_RATE,
    TIF_CAPTURE_RATE_CONSERVATIVE,
    BACKGROUND_APPRECIATION_RATE,
    FARE_PER_TRIP_USD,
)
from src.bus_network import compute_apm_headway, compute_brt_headway

logger = logging.getLogger(__name__)


VALID_SCENARIOS = ("current_zoning", "no_zoning")

# Value per sqft for new construction by use type (Tippecanoe County AV-derived
# defaults; overridden at runtime when AV data is available).
DEFAULT_NEW_CONSTRUCTION_VALUE_PSF: Dict[str, float] = {
    "residential": 130.0,
    "commercial": 110.0,
    "mixed_use": 120.0,
    "industrial": 75.0,
}

# Accessibility premium on existing property values near stations.
STATION_PROXIMITY_PREMIUM = 0.05


# ---------------------------------------------------------------------------
# Campus parking payment
# ---------------------------------------------------------------------------

def compute_campus_payment_series(
    campus_daily_series: np.ndarray,
    years: int,
) -> np.ndarray:
    """Annual campus payment series (USD), scaling with student ridership.

    Students diverted from car -> freed parking spaces -> avoided construction
    or surface-lot land liberation.  Purdue pays an annual fee proportional
    to the parking value displaced.
    """
    from src.financial_params import (
        CAR_DIVERSION_STUDENT,
        STRUCTURED_PARKING_COST_PER_SPACE,
        SURFACE_SPACES_PER_ACRE,
        CAMPUS_LAND_VALUE_PER_ACRE,
        PARKING_OM_PER_SPACE_ANNUAL,
        PURDUE_BORROWING_RATE,
        CAMPUS_PAYMENT_TERM_YEARS,
        RIDERS_PER_DISPLACED_SPACE,
    )

    _arr = np.asarray(campus_daily_series, dtype=float)
    if _arr.size == 0 or _arr[-1] <= 0:
        return np.zeros(years, dtype=float)

    campus_daily_mature = float(_arr[-1])
    diverted_mature = campus_daily_mature * CAR_DIVERSION_STUDENT
    displaced_mature = diverted_mature / RIDERS_PER_DISPLACED_SPACE

    # Channel A: avoided garage construction
    construction_avoided = displaced_mature * STRUCTURED_PARKING_COST_PER_SPACE
    # Channel B: surface lot land liberation
    acres = displaced_mature / SURFACE_SPACES_PER_ACRE
    land_value = acres * CAMPUS_LAND_VALUE_PER_ACRE
    one_time = max(construction_avoided, land_value)

    r = PURDUE_BORROWING_RATE
    n = CAMPUS_PAYMENT_TERM_YEARS
    annuity = (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    om_avoided = displaced_mature * PARKING_OM_PER_SPACE_ANNUAL
    payment_mature = one_time * annuity + om_avoided

    series = np.zeros(years, dtype=float)
    for yr in range(min(years, len(_arr))):
        ratio = _arr[yr] / campus_daily_mature
        series[yr] = payment_mature * ratio
    return series


# ---------------------------------------------------------------------------
# AV value per sqft cache
# ---------------------------------------------------------------------------

_av_value_psf_cache: Optional[Dict[str, float]] = None

def _load_av_value_psf(
    av_path: Path = Path("data/raw/total_av.geojson"),
) -> Dict[str, float]:
    """Load assessed-value-based $/sqft by use type from raw parcel data."""
    global _av_value_psf_cache
    if _av_value_psf_cache is not None:
        return _av_value_psf_cache

    MIN_PLAUSIBLE_PSF = 20.0
    try:
        import geopandas as _gpd
        av = _gpd.read_file(av_path)
        val_col = next((c for c in ["CurTotAV", "total_av", "assessed_value"] if c in av.columns), None)
        fla_col = next((c for c in ["FloorArea", "floor_area", "sqft"] if c in av.columns), None)
        if val_col is None or fla_col is None:
            return dict(DEFAULT_NEW_CONSTRUCTION_VALUE_PSF)

        av[val_col] = pd.to_numeric(av[val_col], errors="coerce").fillna(0)
        av[fla_col] = pd.to_numeric(av[fla_col], errors="coerce").fillna(0)

        def _classify(cc):
            cc = str(cc)
            if cc.startswith("1") or cc.startswith("2"):
                return "residential"
            if cc.startswith("4") or cc.startswith("5"):
                return "commercial"
            return "mixed_use"

        if "ClassCode" in av.columns:
            av["_use_type"] = av["ClassCode"].apply(_classify)
        else:
            av["_use_type"] = "mixed_use"

        result = {}
        for ut, grp in av.groupby("_use_type"):
            total_val = grp[val_col].sum()
            total_fla = grp[fla_col].sum()
            if total_fla > 0:
                psf = total_val / total_fla
                if psf >= MIN_PLAUSIBLE_PSF:
                    result[ut] = psf
        for ut, default_val in DEFAULT_NEW_CONSTRUCTION_VALUE_PSF.items():
            if ut not in result:
                result[ut] = default_val
        _av_value_psf_cache = result
        return result
    except Exception:
        return dict(DEFAULT_NEW_CONSTRUCTION_VALUE_PSF)


# ---------------------------------------------------------------------------
# Endogenous TIF computation
# ---------------------------------------------------------------------------

def _compute_endogenous_tif(
    corridor_id: str,
    base_value: float,
    feedback_df: Optional[pd.DataFrame],
    value_psf: Dict[str, float],
    years: int,
    property_tax_rate: float = PROPERTY_TAX_RATE,
    tif_capture_rate: float = TIF_CAPTURE_RATE_CONSERVATIVE,
    background_rate: float = BACKGROUND_APPRECIATION_RATE,
    proximity_premium: float = STATION_PROXIMITY_PREMIUM,
    assessment_lag_years: int = 1,
    area_type: str = TIF_AREA_TYPE_DEFAULT,
    base_res_share: float = 0.5,
    apply_sb1_erosion: bool = True,
    transit_mode: str = "apm",
) -> Tuple[float, float, float, np.ndarray]:
    """Compute TIF from endogenous development outputs + background appreciation.

    Uses three endogenous sqft streams from the feedback loop:
      new_homestead_sqft -- owner-occupied (R1/R2 zones), 1% cap, excluded from EDA
      new_rental_sqft -- multifamily rental + student, 2% cap, capturable in EDA
      new_comm_sqft -- office/retail, 3% cap, capturable

    Returns (future_value, cumulative_tif, annual_tif_mean, annual_tif_series).
    """
    from src.finance import interpolate_sb1_erosion
    from src.financial_params import (
        CIRCUIT_BREAKER_CAP_HOMESTEAD,
        CIRCUIT_BREAKER_CAP_RENTAL,
        CIRCUIT_BREAKER_CAP_COMMERCIAL,
        TIF_RESIDENTIAL_MAX_YEARS,
        TIF_HOMESTEAD_SHARE,
    )

    proximity_uplift = base_value * proximity_premium
    cumulative_tif = 0.0
    new_value_by_vintage: Dict[int, float] = {}
    annual_tif_series = np.zeros(years, dtype=float)

    # Extract per-year development from feedback loop results.
    dev_by_year: Dict[int, Dict[str, float]] = {}
    _has_tenure_split = False
    if feedback_df is not None and not feedback_df.empty:
        cdf = feedback_df[feedback_df["corridor_id"] == corridor_id].copy()
        _has_tenure_split = (
            "new_homestead_sqft" in cdf.columns
            and "new_rental_sqft" in cdf.columns
        )
        for _, row in cdf.iterrows():
            yr = int(row.get("year", 0))
            entry: Dict[str, float] = {
                "new_comm_sqft": float(row.get("new_comm_sqft", 0)),
            }
            if _has_tenure_split:
                entry["new_homestead_sqft"] = float(row.get("new_homestead_sqft", 0))
                entry["new_rental_sqft"] = float(row.get("new_rental_sqft", 0))
            else:
                res = float(row.get("new_res_sqft", 0))
                entry["new_homestead_sqft"] = res * TIF_HOMESTEAD_SHARE
                entry["new_rental_sqft"] = res * (1.0 - TIF_HOMESTEAD_SHARE)
            dev_by_year[yr] = entry

    res_psf = value_psf.get("residential", DEFAULT_NEW_CONSTRUCTION_VALUE_PSF["residential"])
    comm_psf = value_psf.get("commercial", DEFAULT_NEW_CONSTRUCTION_VALUE_PSF["commercial"])

    # Mode-specific assessed value multiplier
    from src.model_constants import FIXED_GUIDEWAY_RENT_MULT, BRT_RENT_MULT
    _mode_av_mult = BRT_RENT_MULT if transit_mode == "brt" else FIXED_GUIDEWAY_RENT_MULT
    res_psf = res_psf * _mode_av_mult
    comm_psf = comm_psf * _mode_av_mult

    # Detect step size from feedback data
    _available_years = sorted(dev_by_year.keys())
    if len(_available_years) >= 2:
        _gaps = [_available_years[i + 1] - _available_years[i]
                 for i in range(len(_available_years) - 1)]
        _step_years = float(max(1, int(np.median(_gaps))))
    else:
        _step_years = 1.0

    _EMPTY_DEV = {"new_homestead_sqft": 0.0, "new_rental_sqft": 0.0, "new_comm_sqft": 0.0}

    for yr in range(1, years + 1):
        dev = dev_by_year.get(yr, None)
        if dev is None:
            if _available_years:
                closest = min(_available_years, key=lambda y: abs(y - yr))
                dev = dev_by_year[closest].copy()
                if _step_years > 1:
                    dev = {k: v / _step_years for k, v in dev.items()}
            else:
                dev = _EMPTY_DEV.copy()
        else:
            if _step_years > 1:
                dev = {k: v / _step_years for k, v in dev.items()}

        new_homestead_value = dev["new_homestead_sqft"] * res_psf
        new_rental_value = dev["new_rental_sqft"] * res_psf
        new_comm_value = dev["new_comm_sqft"] * comm_psf
        new_construction_value = new_homestead_value + new_rental_value + new_comm_value
        new_value_by_vintage[yr] = new_construction_value

        base_appreciated = (base_value + proximity_uplift) * (1 + background_rate) ** yr

        appreciated_homestead = 0.0
        appreciated_rental = 0.0
        appreciated_comm = 0.0
        for build_yr, nominal_val in new_value_by_vintage.items():
            if yr - build_yr < assessment_lag_years:
                continue
            age = yr - build_yr
            appreciated = nominal_val * (1 + background_rate) ** age
            bdev = dev_by_year.get(build_yr)
            if bdev is None and _available_years:
                _closest_by = min(_available_years, key=lambda y: abs(y - build_yr))
                bdev = dev_by_year[_closest_by]
            elif bdev is None:
                bdev = _EMPTY_DEV
            total_sqft = (bdev["new_homestead_sqft"] + bdev["new_rental_sqft"]
                          + bdev["new_comm_sqft"])
            if total_sqft > 0:
                hfrac = bdev["new_homestead_sqft"] / total_sqft
                rfrac = bdev["new_rental_sqft"] / total_sqft
                cfrac = bdev["new_comm_sqft"] / total_sqft
            else:
                hfrac = base_res_share * TIF_HOMESTEAD_SHARE
                rfrac = base_res_share * (1.0 - TIF_HOMESTEAD_SHARE)
                cfrac = 1.0 - base_res_share
            appreciated_homestead += appreciated * hfrac
            appreciated_rental += appreciated * rfrac
            appreciated_comm += appreciated * cfrac

        base_increment = max(base_appreciated - base_value, 0.0)
        homestead_increment = (appreciated_homestead
                               + base_increment * base_res_share * TIF_HOMESTEAD_SHARE)
        rental_increment = (appreciated_rental
                            + base_increment * base_res_share * (1.0 - TIF_HOMESTEAD_SHARE))
        comm_increment = appreciated_comm + base_increment * (1.0 - base_res_share)

        adj_capture = tif_capture_rate
        if apply_sb1_erosion:
            adj_capture *= interpolate_sb1_erosion(yr)

        homestead_rate = min(property_tax_rate, CIRCUIT_BREAKER_CAP_HOMESTEAD)
        rental_rate = min(property_tax_rate, CIRCUIT_BREAKER_CAP_RENTAL)
        comm_rate = min(property_tax_rate, CIRCUIT_BREAKER_CAP_COMMERCIAL)

        if area_type == "eda":
            annual_tif = (
                rental_increment * rental_rate
                + comm_increment * comm_rate
            ) * adj_capture
        else:
            annual_tif = (
                homestead_increment * homestead_rate
                + rental_increment * rental_rate
                + comm_increment * comm_rate
            ) * adj_capture
            if yr > TIF_RESIDENTIAL_MAX_YEARS:
                annual_tif = (
                    rental_increment * rental_rate
                    + comm_increment * comm_rate
                ) * adj_capture

        annual_tif_series[yr - 1] = annual_tif
        cumulative_tif += annual_tif

    # Future value at horizon
    appreciated_new_final = 0.0
    for build_yr, nominal_val in new_value_by_vintage.items():
        age = years - build_yr
        appreciated_new_final += nominal_val * (1 + background_rate) ** max(age, 0)

    future_value = (
        (base_value + proximity_uplift) * (1 + background_rate) ** years
        + appreciated_new_final
    )
    annual_tif_mean = cumulative_tif / years if years > 0 else 0.0
    return future_value, cumulative_tif, annual_tif_mean, annual_tif_series


# ---------------------------------------------------------------------------
# Macro TIF fallback
# ---------------------------------------------------------------------------

def _load_macro_tif_profile(
    scenario: str = "current_zoning",
    years: int = 25,
    data_dir: Path = Path("data"),
    baseline_growth_rate: float = BACKGROUND_APPRECIATION_RATE,
) -> Optional[Dict[str, Any]]:
    """Load macro TIF profile from property tax uplift CSVs."""
    processed = data_dir / "processed"
    csv_path = processed / f"property_tax_uplift_{scenario}.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        base_total = float(df["baseline_value_dollars"].iloc[0])
        if "projected_value_dollars" in df.columns:
            future_total = float(df["projected_value_dollars"].iloc[-1])
        else:
            future_total = base_total * (1 + baseline_growth_rate) ** years
        increment = future_total - base_total
        cumulative_tif = max(0.0, increment * PROPERTY_TAX_RATE * TIF_CAPTURE_RATE_CONSERVATIVE)
        return {
            "base_total": base_total,
            "future_total": future_total,
            "cumulative_tif_total": cumulative_tif,
            "source": f"property_tax_uplift_{scenario}.csv",
            "scenario": scenario,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Annual debt service
# ---------------------------------------------------------------------------

def _annual_debt_service(
    capital_cost: float,
    interest_rate: float = BOND_RATE,
    term_years: int = DEBT_TERM_YEARS,
) -> float:
    """Amortized annual debt service payment."""
    if capital_cost <= 0 or term_years <= 0:
        return 0.0
    r = max(float(interest_rate), 1e-9)
    n = int(term_years)
    return capital_cost * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


# ---------------------------------------------------------------------------
# Dynamic ridership parsing
# ---------------------------------------------------------------------------

def _parse_daily_series_value(value: Any, years: int) -> Optional[np.ndarray]:
    """Parse dynamic daily ridership series from a cell value."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("[") or text.startswith("{"):
            parsed = json.loads(text)
        else:
            return None
    return annualize_daily_ridership_series(parsed, years=years)


def _build_annual_ridership_series(
    daily_ridership: float,
    daily_ridership_series: Any,
    years: int,
) -> tuple[np.ndarray, str]:
    """Return annual ridership series plus trace source label."""
    parsed_dynamic = _parse_daily_series_value(daily_ridership_series, years)
    if parsed_dynamic is not None:
        return parsed_dynamic, "dynamic_daily_series"
    return annualize_daily_ridership_series(float(daily_ridership), years=years), "static_daily_ridership"


# ---------------------------------------------------------------------------
# Demand-responsive O&M
# ---------------------------------------------------------------------------

def _compute_demand_responsive_apm_om(
    annual_ridership_series: np.ndarray,
    length_km: float,
    n_stops: int = 8,
    operating_days: float = 312.0,
    om_escalation_rate: float = 0.03,
) -> np.ndarray:
    """Compute per-year APM O&M using three-tier cost structure."""
    years = len(annual_ridership_series)
    om_series = np.zeros(years, dtype=float)
    for yr in range(years):
        daily_riders = float(annual_ridership_series[yr]) / operating_days
        headway = compute_apm_headway(
            daily_riders, corridor_length_km=length_km, n_stops=n_stops,
        )
        annual_veh_hours = compute_apm_annual_vehicle_hours(length_km, n_stops, headway)
        base_cost = _APM_MODE.compute_annual_om(length_km, n_stops, annual_veh_hours)
        om_series[yr] = base_cost * (1.0 + om_escalation_rate) ** yr
    return om_series


# ---------------------------------------------------------------------------
# BRT mode-compare helpers
# ---------------------------------------------------------------------------

BRT_FEDERAL_SHARE_DEFAULT = 0.0


def _brt_ridership_discount(length_km: float) -> float:
    """Length-sensitive BRT ridership discount relative to APM.

    Short corridors where APM's speed advantage barely matters see BRT
    closer to APM ridership; long corridors with many stops see a larger
    gap.  Calibrated so that ~5 km (typical corridor) returns ~0.70,
    matching the previous flat assumption.

    Returns a multiplier in [0.55, 0.90].
    """
    return float(np.clip(0.90 - 0.04 * length_km, 0.55, 0.90))


def _compute_demand_responsive_brt_om(
    annual_ridership_series: np.ndarray,
    length_km: float,
    n_stops: int = 8,
    operating_days: float = 312.0,
    om_escalation_rate: float = 0.03,
) -> np.ndarray:
    """Compute per-year BRT O&M using three-tier cost structure."""
    years = len(annual_ridership_series)
    om_series = np.zeros(years, dtype=float)
    for yr in range(years):
        daily_riders = float(annual_ridership_series[yr]) / operating_days
        raw_hw = compute_apm_headway(
            daily_riders, corridor_length_km=length_km, n_stops=n_stops,
        )
        headway = max(_BRT_MODE.min_headway_min, min(raw_hw, _BRT_MODE.max_headway_min))
        annual_veh_hours = compute_brt_annual_vehicle_hours(length_km, n_stops, headway)
        base_cost = _BRT_MODE.compute_annual_om(length_km, n_stops, annual_veh_hours)
        om_series[yr] = base_cost * (1.0 + om_escalation_rate) ** yr
    return om_series


def _evaluate_corridor_brt(
    corridor_id: str,
    apm_result: Dict[str, Any],
    *,
    feedback_df: Optional[pd.DataFrame] = None,
    brt_feedback_df: Optional[pd.DataFrame] = None,
    macro_tif_profile: Optional[Dict[str, float]] = None,
    value_psf: Optional[Dict[str, float]] = None,
    cashflow_years: int = 25,
    interest_rate: float = 0.05,
    fare_per_trip_usd: float = 2.00,
    farebox_capture_rate: float = 1.0,
    federal_share: float = BRT_FEDERAL_SHARE_DEFAULT,
    state_share: float = 0.0,
) -> Dict[str, Any]:
    """Re-evaluate a corridor using BRT cost/performance parameters."""
    from src.financial_params import compute_brt_capital_cost

    length_km = float(apm_result["length_km"])
    n_stops = int(apm_result["n_stops"])
    years = max(int(cashflow_years), 1)

    # --- Ridership ---
    _brt_cdf = None
    if brt_feedback_df is not None and not brt_feedback_df.empty:
        _brt_cdf = brt_feedback_df[
            brt_feedback_df["corridor_id"] == corridor_id
        ].sort_values("year")
    if _brt_cdf is not None and not _brt_cdf.empty and "daily_ridership" in _brt_cdf.columns:
        _brt_series_raw = _brt_cdf["daily_ridership"].values.astype(float)
        if len(_brt_series_raw) >= years:
            brt_annual_series = _brt_series_raw[:years] * OPERATING_DAYS_PER_YEAR
        else:
            brt_annual_series = np.zeros(years, dtype=float)
            brt_annual_series[:len(_brt_series_raw)] = _brt_series_raw * OPERATING_DAYS_PER_YEAR
            brt_annual_series[len(_brt_series_raw):] = brt_annual_series[len(_brt_series_raw) - 1]
        brt_daily = float(_brt_series_raw[-1])
        demand_trace_source = "brt_feedback_loop"
    else:
        apm_daily = float(apm_result["daily_ridership"])
        _discount = _brt_ridership_discount(length_km)
        brt_daily = apm_daily * _discount
        _apm_series = apm_result.get("annual_ridership_series")
        if _apm_series is not None and hasattr(_apm_series, '__len__') and len(_apm_series) > 0:
            brt_annual_series = np.asarray(_apm_series, dtype=float) * _discount
            demand_trace_source = f"apm_discount_{_discount:.2f}"
        else:
            brt_annual_series, demand_trace_source = _build_annual_ridership_series(
                daily_ridership=brt_daily, daily_ridership_series=None, years=years,
            )

    # --- Capital cost ---
    gross_capital = compute_brt_capital_cost(length_km, n_stops)
    local_share = max(0.0, 1.0 - float(federal_share) - float(state_share))
    capital_cost = gross_capital * local_share
    annual_debt = _annual_debt_service(capital_cost, interest_rate, DEBT_TERM_YEARS)

    # --- O&M ---
    brt_om = _compute_demand_responsive_brt_om(
        brt_annual_series, length_km=length_km, n_stops=n_stops,
    )
    annual_operating_cost = float(np.mean(brt_om))
    om_mean = float(np.mean(brt_om))

    # --- TIF ---
    cumulative_tif = float(apm_result.get("tif_revenue_cumulative", 0.0))
    _annual_tif_series = None

    if _brt_cdf is not None and not _brt_cdf.empty and "new_res_sqft" in _brt_cdf.columns:
        _base_val = float(apm_result.get("base_property_value", 0.0))
        _vpsf = value_psf if value_psf else DEFAULT_NEW_CONSTRUCTION_VALUE_PSF
        try:
            _fv, _cum_tif, _ann_tif_mean, _ann_tif_arr = _compute_endogenous_tif(
                corridor_id=corridor_id, base_value=_base_val,
                feedback_df=brt_feedback_df, value_psf=_vpsf,
                years=years, transit_mode="brt",
            )
            _annual_tif_series = _ann_tif_arr
            cumulative_tif = _cum_tif
        except Exception:
            pass

    if _annual_tif_series is None:
        _flat_tif = cumulative_tif / years if years > 0 else 0.0
        _annual_tif_series = np.full(years, _flat_tif, dtype=float)

    # --- Campus payment ---
    campus_series = np.zeros(years, dtype=float)
    if _brt_cdf is not None and "campus_daily" in _brt_cdf.columns:
        _campus_daily = _brt_cdf["campus_daily"].values.astype(float)
        _campus_annual = compute_campus_payment_series(_campus_daily, years=years)
        for yr in range(min(years, len(_campus_annual))):
            campus_series[yr] = _campus_annual[yr]
    else:
        apm_campus_mean = float(apm_result.get("campus_payment_annual_musd", 0.0))
        if apm_campus_mean > 0.001:
            campus_series[:] = apm_campus_mean * _discount * 1_000_000.0

    # --- Dynamic finance metrics ---
    dynamic_finance = _compute_dynamic_finance_metrics(
        brt_annual_series,
        years=years,
        annual_tif_series_usd=_annual_tif_series,
        capital_cost_usd=capital_cost,
        annual_debt_service_usd=annual_debt,
        annual_operating_cost_usd=annual_operating_cost,
        discount_rate=interest_rate,
        fare_per_trip_usd=fare_per_trip_usd,
        farebox_capture_rate=farebox_capture_rate,
        demand_responsive_om_series=brt_om,
        campus_payment_series_usd=campus_series,
    )

    dscr_year5 = float(dynamic_finance["dscr_year5"])
    dscr_year25 = float(dynamic_finance["dscr_year25"])
    dscr_min = float(dynamic_finance["dscr_min"])

    return {
        "corridor_id": corridor_id,
        "transit_mode": "BRT",
        "scenario": apm_result.get("scenario", "current_zoning"),
        "length_km": length_km,
        "n_stops": n_stops,
        "daily_ridership": dynamic_finance["annual_ridership_mean"] / OPERATING_DAYS_PER_YEAR,
        "daily_ridership_static_input": brt_daily,
        "n_parcels_tif": int(apm_result.get("n_parcels_tif", 0)),
        "base_property_value": float(apm_result.get("base_property_value", 0.0)),
        "future_property_value": float(apm_result.get("future_property_value", 0.0)),
        "property_value_increase": float(apm_result.get("property_value_increase", 0.0)),
        "tif_revenue_cumulative": cumulative_tif,
        "gross_capital_cost": float(gross_capital),
        "federal_share": float(federal_share),
        "state_share": float(state_share),
        "capital_cost_local": float(capital_cost),
        "capital_cost": float(capital_cost),
        "annual_operating_cost_static": float(annual_operating_cost),
        "annual_operating_cost_yr0": float(brt_om[0]) if len(brt_om) > 0 else 0.0,
        "annual_operating_cost_final": float(brt_om[-1]) if len(brt_om) > 0 else 0.0,
        "annual_om_mean_usd": float(om_mean),
        "annual_debt_service": float(annual_debt),
        "annual_tif_revenue": cumulative_tif / years if years > 0 else 0.0,
        "debt_coverage_ratio_tif_only": (
            (cumulative_tif / years) / annual_debt if annual_debt > 0 else 0.0
        ),
        "dscr_year5": float(dscr_year5),
        "dscr_year25": float(dscr_year25),
        "dscr_min": float(dscr_min),
        "debt_coverage_ratio": float(dscr_min),
        "project_npv_dynamic_musd": float(dynamic_finance["project_npv_dynamic_musd"]),
        "project_irr_dynamic": float(dynamic_finance.get("project_irr_dynamic", float("nan"))),
        "demand_trace_source": demand_trace_source,
        "demand_trace_years": years,
        "financially_viable": bool(dscr_min >= 1.0),
        "self_sufficiency": float(dynamic_finance.get("system_self_sufficiency", 0.0)),
        "annual_ridership_effective": float(dynamic_finance["annual_ridership_mean"]),
        "annual_ridership_total_dynamic": float(dynamic_finance["annual_ridership_total"]),
        "campus_payment_annual_musd": float(dynamic_finance.get("campus_payment_annual_mean_musd", 0.0)),
        "cost_per_rider": 0.0,  # computed downstream
    }


# ---------------------------------------------------------------------------
# Dynamic finance metrics
# ---------------------------------------------------------------------------

def _compute_dynamic_finance_metrics(
    annual_ridership_series: np.ndarray,
    *,
    years: int,
    annual_tif_series_usd: np.ndarray,
    capital_cost_usd: float,
    annual_debt_service_usd: float,
    annual_operating_cost_usd: float,
    discount_rate: float,
    fare_per_trip_usd: float,
    farebox_capture_rate: float,
    om_escalation_rate: float = 0.03,
    demand_responsive_om_series: Optional[np.ndarray] = None,
    campus_payment_series_usd: Optional[np.ndarray] = None,
    net_bus_cost_delta_usd: float = 0.0,
) -> Dict[str, float]:
    """Compute dynamic project finance metrics for one corridor."""
    operating_days = OPERATING_DAYS_PER_YEAR

    _tif_arr = np.asarray(annual_tif_series_usd, dtype=float)
    if _tif_arr.size < years:
        _tif_arr = np.pad(_tif_arr, (0, years - _tif_arr.size), constant_values=0.0)
    annual_tif_series_musd = _tif_arr[:years] / 1_000_000.0

    farebox_series_musd = annual_ridership_to_revenue_musd(
        annual_ridership_series,
        fare_per_trip_usd=fare_per_trip_usd,
        capture_rate=farebox_capture_rate,
        years=years,
    )

    campus_series_musd = np.zeros(years, dtype=float)
    if campus_payment_series_usd is not None:
        _cp = np.asarray(campus_payment_series_usd, dtype=float)
        if _cp.size >= years:
            campus_series_musd = _cp[:years] / 1_000_000.0
        elif _cp.size > 0:
            campus_series_musd[:_cp.size] = _cp / 1_000_000.0

    annual_total_revenue_musd = (
        annual_tif_series_musd + farebox_series_musd + campus_series_musd
    )

    # O&M: demand-responsive if available, otherwise static + escalation.
    if demand_responsive_om_series is not None and len(demand_responsive_om_series) >= years:
        om_escalated = demand_responsive_om_series[:years].astype(float)
    else:
        om_base = float(annual_operating_cost_usd)
        om_escalated = np.array(
            [om_base * (1.0 + om_escalation_rate) ** yr for yr in range(years)],
            dtype=float,
        )
    annual_total_cost_musd = (
        float(annual_debt_service_usd) + om_escalated
    ) / 1_000_000.0

    annual_om_only_musd = om_escalated / 1_000_000.0
    project_finance = npv_irr(
        annual_revenue_musd=annual_total_revenue_musd,
        capex_musd_val=float(capital_cost_usd) / 1_000_000.0,
        years=years,
        discount_rate=discount_rate,
        annual_cost_musd=annual_om_only_musd,
    )
    farebox_finance = npv_irr(
        annual_revenue_musd=farebox_series_musd,
        capex_musd_val=0.0,
        years=years,
        discount_rate=discount_rate,
    )

    annual_revenue_usd = annual_total_revenue_musd * 1_000_000.0
    annual_total_cost_usd = annual_total_cost_musd * 1_000_000.0

    if annual_debt_service_usd > 0:
        _dscr_series = annual_revenue_usd / float(annual_debt_service_usd)
        dscr_year5 = float(_dscr_series[min(4, years - 1)])
        dscr_year25 = float(_dscr_series[min(24, years - 1)])
        dscr_min = float(np.min(_dscr_series))
    else:
        dscr_year5 = dscr_year25 = dscr_min = float("inf")

    total_coverage_ratio = float(np.mean(
        np.where(annual_total_cost_usd > 0,
                 annual_revenue_usd / annual_total_cost_usd,
                 np.nan)
    ))

    system_annual_cost_usd = annual_total_cost_usd + float(net_bus_cost_delta_usd)
    system_self_sufficiency = (
        float(np.mean(annual_revenue_usd / system_annual_cost_usd))
        if np.all(system_annual_cost_usd > 0)
        else float("nan")
    )

    return {
        "annual_ridership_mean": float(np.mean(annual_ridership_series)),
        "annual_ridership_total": float(np.sum(annual_ridership_series)),
        "farebox_revenue_annual_mean_musd": float(np.mean(farebox_series_musd)),
        "farebox_revenue_npv_musd": float(farebox_finance["npv_musd"]),
        "project_npv_dynamic_musd": float(project_finance["npv_musd"]),
        "project_irr_dynamic": float(project_finance["irr"]),
        "dscr_year5": dscr_year5,
        "dscr_year25": dscr_year25,
        "dscr_min": dscr_min,
        "total_coverage_ratio_dynamic": total_coverage_ratio,
        "system_self_sufficiency": system_self_sufficiency,
        "net_bus_cost_delta_usd": float(net_bus_cost_delta_usd),
        "annual_tif_effective_usd": float(np.mean(_tif_arr[:years])),
        "annual_om_mean_usd": float(np.mean(om_escalated)),
        "campus_payment_annual_mean_musd": float(np.mean(campus_series_musd)),
        "campus_payment_npv_musd": float(np.sum(
            campus_series_musd / np.power(1 + discount_rate, np.arange(1, years + 1))
        )),
    }


# ---------------------------------------------------------------------------
# Uncertainty framework
# ---------------------------------------------------------------------------

def _default_uncertainty_framework() -> Dict[str, Any]:
    """Default Week 21 uncertainty assumptions."""
    return {
        "runner_defaults": {
            "n_draws": 2000,
            "random_seed": 42,
            "percentiles": [10, 50, 90],
            "viability_threshold_dcr": 1.0,
        },
        "parameter_ranges": {
            "ridership_multiplier": {"distribution": "triangular", "low": 0.80, "mode": 1.00, "high": 1.25},
            "tif_multiplier": {"distribution": "triangular", "low": 0.60, "mode": 1.00, "high": 1.40},
            "capital_cost_multiplier": {"distribution": "triangular", "low": 0.75, "mode": 1.00, "high": 1.50},
            "operating_cost_multiplier": {"distribution": "triangular", "low": 0.85, "mode": 1.00, "high": 1.25},
            "fare_multiplier": {"distribution": "triangular", "low": 0.85, "mode": 1.00, "high": 1.15},
            "discount_rate_delta": {"distribution": "triangular", "low": -0.015, "mode": 0.00, "high": 0.015},
            "student_demand_multiplier": {"distribution": "triangular", "low": 0.65, "mode": 1.00, "high": 1.40},
        },
    }


def load_uncertainty_framework(
    config_path: Path | str = Path("scenarios_config.json"),
) -> Dict[str, Any]:
    """Load Week 21 uncertainty ranges from scenario config with safe defaults."""
    framework = _default_uncertainty_framework()
    path = Path(config_path)
    if not path.exists():
        return framework
    try:
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return framework

    metadata = config.get("metadata", {})
    payload = metadata.get("uncertainty_framework", {})
    if not isinstance(payload, dict):
        return framework

    runner_defaults = payload.get("runner_defaults", {})
    if isinstance(runner_defaults, dict):
        framework["runner_defaults"].update(runner_defaults)

    parameter_ranges = payload.get("parameter_ranges", {})
    if isinstance(parameter_ranges, dict):
        for name, spec in parameter_ranges.items():
            if not isinstance(spec, dict):
                continue
            base = framework["parameter_ranges"].get(name, {})
            merged = dict(base)
            merged.update(spec)
            framework["parameter_ranges"][name] = merged

    corr_block = payload.get("correlation_matrix")
    if isinstance(corr_block, dict) and "matrix" in corr_block:
        framework["correlation_matrix"] = corr_block["matrix"]
        framework["correlation_param_order"] = corr_block.get("param_order", [])
    elif isinstance(corr_block, list):
        framework["correlation_matrix"] = corr_block

    return framework


def _sample_uncertainty_parameter(
    name: str, spec: Dict[str, Any], n_draws: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sample one uncertainty parameter vector (independent — no correlation)."""
    dist = str(spec.get("distribution", "triangular")).strip().lower()
    low = float(spec.get("low", 1.0))
    high = float(spec.get("high", 1.0))
    if high < low:
        high = low

    if dist == "uniform":
        return rng.uniform(low, high, size=n_draws)
    if dist == "triangular":
        mode = float(spec.get("mode", 0.5 * (low + high)))
        mode = min(max(mode, low), high)
        return rng.triangular(left=low, mode=mode, right=high, size=n_draws)
    if dist == "lognormal":
        mu = float(spec.get("mu", 0.0))
        sigma = float(spec.get("sigma", 0.1))
        return np.clip(rng.lognormal(mean=mu, sigma=max(sigma, 1e-9), size=n_draws), low, high)
    if dist == "normal":
        mean = float(spec.get("mean", 1.0))
        std = float(spec.get("std", 0.05))
        return np.clip(rng.normal(loc=mean, scale=max(std, 1e-9), size=n_draws), low, high)
    return rng.triangular(left=low, mode=0.5*(low+high), right=high, size=n_draws)


# ---------------------------------------------------------------------------
# Correlated parameter sampling via Gaussian copula (Component 3A)
# ---------------------------------------------------------------------------

# Order: ridership, tif, capital, operating, fare, discount_rate_delta,
#         student_demand, beta_distance, employment_growth, zero_car_share
_PARAM_NAMES_ORDERED = [
    "ridership_multiplier",
    "tif_multiplier",
    "capital_cost_multiplier",
    "operating_cost_multiplier",
    "fare_multiplier",
    "discount_rate_delta",
    "student_demand_multiplier",
    "beta_distance_mult",
    "employment_growth_rate",
    "zero_car_share_mult",
]

# Per-scenario correlation matrices (Gaussian copula)
# Sources & rationale:
#   ridership <-> TIF: 0.20 (current) / 0.40 (no_zoning)
#     Higher ridership drives more TOD → larger TIF base.
#   capital <-> operating: 0.50 (current) / 0.60 (no_zoning)
#     NTD data: per-km capital and O&M correlated for small-urban FG systems.
#   capital <-> ridership: -0.10 (current) / 0.00 (no_zoning)
#     Constrained zoning: longer corridors add cost but diminishing demand.
#   ridership <-> fare: 0.10 (all) — weak, assumed.
#   beta_distance_mult <-> ridership: +0.40 (distance sensitivity scales MNL)
#   employment_growth <-> tif: +0.30 (job growth → commercial AV)
#   beta_distance <-> zero_car_share: +0.20
#   employment_growth <-> ridership: +0.20
_DEFAULT_CORRELATION_MATRICES: Dict[str, np.ndarray] = {
    "current_zoning": np.array([
        # rider  tif    cap    om    fare  disc   stud   beta   empl   zcar
        [1.00,  0.20, -0.10,  0.00,  0.10,  0.00,  0.00,  0.40,  0.20,  0.00],  # ridership
        [0.20,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.30,  0.00],  # tif
        [-0.10, 0.00,  1.00,  0.50,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # capital
        [0.00,  0.00,  0.50,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # operating
        [0.10,  0.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # fare
        [0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00],  # discount
        [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00],  # student
        [0.40,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  0.20],  # beta_dist
        [0.20,  0.30,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00],  # empl_growth
        [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.20,  0.00,  1.00],  # zero_car
    ]),
    "no_zoning": np.array([
        [1.00,  0.40,  0.00,  0.00,  0.10,  0.00,  0.00,  0.40,  0.20,  0.00],
        [0.40,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.30,  0.00],
        [0.00,  0.00,  1.00,  0.60,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
        [0.00,  0.00,  0.60,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
        [0.10,  0.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00],
        [0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00],
        [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00],
        [0.40,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  0.20],
        [0.20,  0.30,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00],
        [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.20,  0.00,  1.00],
    ]),
}
_DEFAULT_CORRELATION_MATRIX = _DEFAULT_CORRELATION_MATRICES["current_zoning"]


def _quantile_from_spec(
    name: str,
    spec: Dict[str, Any],
    u: np.ndarray,
) -> np.ndarray:
    """Inverse CDF (quantile function) for a parameter spec at uniform values u.

    Uses scipy.special functions directly to avoid the slow scipy.stats import.
    """
    from scipy.special import ndtri  # norm.ppf

    dist = str(spec.get("distribution", "triangular")).strip().lower()
    low = float(spec.get("low", 1.0))
    high = float(spec.get("high", 1.0))
    if high < low:
        raise ValueError(
            f"Parameter '{name}': low ({low}) > high ({high}). "
            f"Fix the uncertainty_framework config in scenarios_config.json."
        )

    if dist == "uniform":
        scale = max(high - low, 1e-12)
        return low + u * scale

    if dist == "triangular":
        mode = float(spec.get("mode", 0.5 * (low + high)))
        mode = min(max(mode, low), high)
        scale = max(high - low, 1e-12)
        c = (mode - low) / scale
        out = np.empty_like(u)
        mask_lo = u <= c
        mask_hi = ~mask_lo
        out[mask_lo] = low + np.sqrt(u[mask_lo] * scale * (mode - low))
        out[mask_hi] = high - np.sqrt((1.0 - u[mask_hi]) * scale * (high - mode))
        return out

    if dist == "normal":
        mean = float(spec.get("mean", 1.0))
        std = float(spec.get("std", 0.05))
        return np.clip(ndtri(u) * max(std, 1e-9) + mean, low, high)

    if dist == "lognormal":
        mu = float(spec.get("mu", 0.0))
        sigma = float(spec.get("sigma", 0.1))
        return np.clip(np.exp(ndtri(u) * max(sigma, 1e-9) + mu), low, high)

    raise ValueError(f"Unsupported distribution for {name}: {dist}")


def _correlated_sample(
    parameter_ranges: Dict[str, Dict[str, Any]],
    n_draws: int,
    rng: np.random.Generator,
    correlation_matrix: Optional[np.ndarray] = None,
    scenario: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Sample uncertainty parameters with rank correlation via Gaussian copula.

    1. Sample from multivariate normal with the correlation matrix.
    2. Convert each marginal to uniform via Phi (standard normal CDF).
    3. Invert through each parameter's marginal quantile function.

    Parameters not in the correlation matrix are sampled independently.
    If *scenario* is given and no explicit *correlation_matrix* is provided,
    selects the scenario-specific default correlation matrix.
    """
    from scipy.special import ndtr  # norm.cdf

    if correlation_matrix is None:
        correlation_matrix = _DEFAULT_CORRELATION_MATRICES.get(
            scenario, _DEFAULT_CORRELATION_MATRIX
        )

    n_corr = len(_PARAM_NAMES_ORDERED)
    n_given = correlation_matrix.shape[0]
    if n_given < n_corr:
        padded = np.eye(n_corr)
        padded[:n_given, :n_given] = correlation_matrix
        correlation_matrix = padded
    assert correlation_matrix.shape == (n_corr, n_corr), (
        f"Correlation matrix shape {correlation_matrix.shape} != expected ({n_corr}, {n_corr})"
    )
    assert np.allclose(np.diag(correlation_matrix), 1.0), (
        f"Correlation matrix diagonal must be all 1.0, got {np.diag(correlation_matrix)}"
    )

    # Step 1: Multivariate normal with correlation structure
    mvn_samples = rng.multivariate_normal(
        mean=np.zeros(n_corr), cov=correlation_matrix, size=n_draws,
    )  # shape (n_draws, n_corr)

    # Step 2: Phi to uniform [0, 1]
    uniform_samples = ndtr(mvn_samples)

    # Step 3: Invert through each marginal's quantile function
    result: Dict[str, np.ndarray] = {}
    for i, pname in enumerate(_PARAM_NAMES_ORDERED):
        spec = parameter_ranges.get(pname)
        if spec is None:
            continue
        u = uniform_samples[:, i]
        result[pname] = _quantile_from_spec(pname, spec, u)

    # Sample remaining parameters independently (not in the corr matrix)
    for pname, spec in parameter_ranges.items():
        if pname not in result:
            result[pname] = _sample_uncertainty_parameter(pname, spec, n_draws, rng)

    return result


def compute_uncertainty_bands(
    results_df: pd.DataFrame,
    *,
    config_path: Path | str = Path("scenarios_config.json"),
    cashflow_years: int = 25,
    fare_per_trip_usd: float = 2.00,
    farebox_capture_rate: float = 1.0,
    discount_rate: float = 0.05,
    n_draws: Optional[int] = None,
    random_seed: Optional[int] = None,
    percentiles: Optional[Sequence[float]] = None,
    viability_threshold: Optional[float] = None,
    behavioral_surrogates: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Week 21 Monte Carlo uncertainty engine for corridor finance outputs."""
    if results_df is None or results_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, {}

    framework = load_uncertainty_framework(config_path)
    defaults = framework.get("runner_defaults", {})
    draw_count = int(n_draws) if n_draws is not None else int(defaults.get("n_draws", 500))
    seed = int(random_seed) if random_seed is not None else int(defaults.get("random_seed", 42))
    pct_values = sorted({float(p) for p in (percentiles or defaults.get("percentiles", [10, 50, 90]))})
    viability_cutoff = float(viability_threshold) if viability_threshold is not None else float(defaults.get("viability_threshold_dcr", 1.0))

    rng = np.random.default_rng(seed)
    param_ranges = {k: v for k, v in framework.get("parameter_ranges", {}).items() if isinstance(v, dict)}

    # External correlation matrix from config (if any)
    corr_matrix = framework.get("correlation_matrix")
    if corr_matrix is not None:
        corr_matrix = np.asarray(corr_matrix, dtype=float)

    # Per-scenario correlated sampling: each scenario gets its own copula draws
    # using the scenario-specific correlation matrix.
    scenarios_in_data = set()
    if "scenario" in results_df.columns:
        scenarios_in_data = {str(s) for s in results_df["scenario"].unique() if s}

    sampled_by_scenario: Dict[str, Dict[str, np.ndarray]] = {}
    if param_ranges:
        if scenarios_in_data:
            for scen in sorted(scenarios_in_data):
                scen_rng = np.random.default_rng(seed + hash(scen) % (2**31))
                sampled_by_scenario[scen] = _correlated_sample(
                    param_ranges, draw_count, scen_rng,
                    correlation_matrix=corr_matrix, scenario=scen,
                )
        else:
            sampled_by_scenario[""] = _correlated_sample(
                param_ranges, draw_count, rng, correlation_matrix=corr_matrix,
            )
    if not sampled_by_scenario:
        sampled_by_scenario[""] = {
            pname: np.ones(draw_count) for pname in _PARAM_NAMES_ORDERED
        }

    metrics = ["daily_ridership", "tif_revenue_cumulative", "debt_coverage_ratio", "project_npv_dynamic_musd"]
    draw_frames = []
    wide_rows = []
    long_rows = []

    for _, row in results_df.iterrows():
        cid = str(row.get("corridor_id", ""))
        scenario = str(row.get("scenario", ""))
        years_f = float(max(cashflow_years, 1))

        # Select per-scenario correlated samples
        sampled = sampled_by_scenario.get(
            scenario, next(iter(sampled_by_scenario.values()))
        )

        base_daily = max(float(row.get("daily_ridership", 0)), 0)
        base_tif = max(float(row.get("tif_revenue_cumulative", 0)), 0)
        base_capital = max(float(row.get("capital_cost", 0)), 0)
        base_opex = max(float(row.get("annual_operating_cost", row.get("annual_om_mean_usd", 0))), 0)
        base_debt = max(float(row.get("annual_debt_service", 0)), 0)

        rm = sampled.get("ridership_multiplier", np.ones(draw_count))
        tm = sampled.get("tif_multiplier", np.ones(draw_count))
        cm = sampled.get("capital_cost_multiplier", np.ones(draw_count))
        om = sampled.get("operating_cost_multiplier", np.ones(draw_count))
        fm = sampled.get("fare_multiplier", np.ones(draw_count))
        dd = sampled.get("discount_rate_delta", np.zeros(draw_count))

        daily_riders = base_daily * rm
        annual_riders = daily_riders * OPERATING_DAYS_PER_YEAR
        tif_cum = base_tif * tm
        annual_tif = tif_cum / years_f
        annual_farebox = annual_riders * fare_per_trip_usd * fm * farebox_capture_rate
        _capex_adj = base_capital * cm
        _bond_rate = np.clip(BOND_RATE + dd, 1e-6, None)
        _amort = (_bond_rate * (1 + _bond_rate) ** DEBT_TERM_YEARS) / ((1 + _bond_rate) ** DEBT_TERM_YEARS - 1)
        annual_debt = _capex_adj * _amort
        annual_opex = base_opex * om
        annual_revenue = annual_tif + annual_farebox
        annual_cost = annual_debt + annual_opex

        dcr = np.where(annual_debt > 0, annual_revenue / annual_debt, np.inf)
        disc_rates = np.clip(discount_rate + dd, 0.001, None)
        pv_factor = (1.0 - np.power(1.0 + disc_rates, -years_f)) / disc_rates
        npv_musd = (-_capex_adj + (annual_revenue - annual_opex) * pv_factor) / 1e6

        draws = pd.DataFrame({
            "corridor_id": cid,
            "scenario": scenario,
            "draw": np.arange(draw_count),
            "daily_ridership": daily_riders,
            "tif_revenue_cumulative": tif_cum,
            "debt_coverage_ratio": dcr,
            "project_npv_dynamic_musd": npv_musd,
            "financially_viable": dcr >= viability_cutoff,
            "capital_cost": _capex_adj,
            "annual_operating_cost": annual_opex,
            "annual_debt_service": annual_debt,
        })
        draw_frames.append(draws)

        # Percentile bands
        band_row = {"corridor_id": cid, "scenario": scenario}
        for metric in metrics:
            vals = draws[metric].to_numpy()
            for p in pct_values:
                p_label = str(int(p)) if float(p).is_integer() else str(p)
                val = float(np.percentile(vals, p))
                band_row[f"{metric}_p{p_label}"] = val
                long_rows.append({"corridor_id": cid, "scenario": scenario,
                                  "metric": metric, "percentile": float(p), "value": val})
        band_row["financial_viability_probability"] = float(draws["financially_viable"].mean())
        wide_rows.append(band_row)

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(long_rows)
    draws_df = pd.concat(draw_frames, ignore_index=True) if draw_frames else pd.DataFrame()

    metadata = {
        "n_draws": int(draw_count),
        "random_seed": int(seed),
        "percentiles": [float(p) for p in pct_values],
        "viability_threshold_dcr": float(viability_cutoff),
        "behavioral_surrogates_used": behavioral_surrogates is not None and len(behavioral_surrogates) > 0,
        "behavioral_surrogate_metrics": sorted(behavioral_surrogates.keys()) if behavioral_surrogates else [],
    }
    return wide_df, long_df, draws_df, metadata


# ---------------------------------------------------------------------------
# Robust corridor ranking
# ---------------------------------------------------------------------------

def robust_corridor_ranking(
    draws_df: pd.DataFrame,
    top_k: int = 5,
    shortfall_pct: float = 10.0,
    robust_metric: str = "cvar",
) -> pd.DataFrame:
    """Rank corridors by CVaR/expected shortfall from Monte Carlo draws."""
    if draws_df is None or draws_df.empty:
        return pd.DataFrame()

    _draw_col = "draw" if "draw" in draws_df.columns else "sample_id"
    required = {"corridor_id", _draw_col, "debt_coverage_ratio"}
    if not required.issubset(draws_df.columns):
        return pd.DataFrame()

    corridor_ids = sorted(draws_df["corridor_id"].unique())
    draws = sorted(draws_df[_draw_col].unique())
    n_draws = len(draws)
    n_tail = max(1, int(n_draws * shortfall_pct / 100.0))

    # Build DCR matrix
    dcr_matrix: Dict[Any, Dict[str, float]] = {}
    for d in draws:
        d_slice = draws_df[draws_df[_draw_col] == d]
        dcr_matrix[d] = dict(zip(d_slice["corridor_id"], d_slice["debt_coverage_ratio"]))

    # P(top-k)
    top_k_counts = {cid: 0 for cid in corridor_ids}
    for d in draws:
        draw_dcrs = dcr_matrix.get(d, {})
        if not draw_dcrs:
            continue
        ranked = sorted(draw_dcrs.items(), key=lambda x: x[1], reverse=True)
        for cid, _ in ranked[:top_k]:
            top_k_counts[cid] += 1

    # Per-corridor stats
    expected_shortfall = {}
    p10_dcr = {}
    median_dcr = {}
    mean_dcr = {}
    for cid in corridor_ids:
        cid_dcrs = [dcr_matrix.get(d, {}).get(cid, 0.0) for d in draws]
        cid_dcrs_sorted = sorted(cid_dcrs)
        expected_shortfall[cid] = float(np.mean(cid_dcrs_sorted[:n_tail]))
        p10_dcr[cid] = float(np.percentile(cid_dcrs, 10))
        median_dcr[cid] = float(np.median(cid_dcrs))
        mean_dcr[cid] = float(np.mean(cid_dcrs))

    # Max regret
    max_regret = {cid: 0.0 for cid in corridor_ids}
    for d in draws:
        draw_dcrs = dcr_matrix.get(d, {})
        if not draw_dcrs:
            continue
        best_dcr = max(draw_dcrs.values())
        for cid in corridor_ids:
            gap = best_dcr - draw_dcrs.get(cid, 0.0)
            if gap > max_regret[cid]:
                max_regret[cid] = gap

    def _norm(vals):
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        return [0.5] * len(vals) if rng < 1e-12 else [(v - lo) / rng for v in vals]

    p_top = [top_k_counts[cid] / max(n_draws, 1) for cid in corridor_ids]
    es_vals = [expected_shortfall[cid] for cid in corridor_ids]
    regret_vals = [max_regret[cid] for cid in corridor_ids]

    p_norm = _norm(p_top)
    tail_vals = [p10_dcr[cid] for cid in corridor_ids] if robust_metric == "p10" else es_vals
    tail_norm = _norm(tail_vals)
    regret_norm = _norm([-r for r in regret_vals])

    rows = []
    for i, cid in enumerate(corridor_ids):
        composite = 0.40 * p_norm[i] + 0.35 * tail_norm[i] + 0.25 * regret_norm[i]
        rows.append({
            "corridor_id": cid,
            "p_top_k": round(p_top[i], 4),
            "expected_shortfall_dcr": round(es_vals[i], 4),
            "p10_dcr": round(p10_dcr[cid], 4),
            "max_regret_dcr": round(regret_vals[i], 4),
            "median_dcr": round(median_dcr[cid], 4),
            "mean_dcr": round(mean_dcr[cid], 4),
            "robust_score": round(composite, 4),
        })

    result = pd.DataFrame(rows).sort_values("robust_score", ascending=False).reset_index(drop=True)
    result["robust_rank"] = range(1, len(result) + 1)
    return result


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

def _load_stress_scenarios(
    config_path: Path | str = Path("scenarios_config.json"),
) -> Dict[str, Dict[str, float]]:
    """Load stress scenario definitions from config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("stress_scenarios", {})
    except Exception:
        return {}


def _run_stress_scenario(
    corridor_result: Dict[str, Any],
    scenario_params: Dict[str, float],
    cashflow_years: int = 25,
    base_interest_rate: float = 0.05,
    base_fare: float = 2.00,
    farebox_capture_rate: float = 1.0,
) -> Dict[str, float]:
    """Run a single stress scenario for one corridor."""
    daily_ridership = float(corridor_result.get("daily_ridership", 0))
    capital_cost = float(corridor_result.get("capital_cost", 0))
    annual_om = float(corridor_result.get("annual_om_mean_usd", 0))
    annual_debt_svc = float(corridor_result.get("annual_debt_service", 0))
    annual_tif = float(corridor_result.get("annual_tif_revenue", 0))

    ridership_mult = scenario_params.get("ridership_multiplier", 1.0)
    student_mult = scenario_params.get("student_demand_multiplier", 1.0)
    capital_mult = scenario_params.get("capital_cost_multiplier", 1.0)
    om_mult = scenario_params.get("om_cost_multiplier", 1.0)
    fare_mult = scenario_params.get("fare_multiplier", 1.0)
    tif_mult = scenario_params.get("tif_capture_multiplier", 1.0)
    discount_delta = scenario_params.get("discount_rate_delta", 0.0)

    effective_ridership = daily_ridership * ridership_mult * student_mult
    annual_riders = effective_ridership * OPERATING_DAYS_PER_YEAR
    effective_capital = capital_cost * capital_mult
    effective_om = annual_om * om_mult
    effective_fare = base_fare * fare_mult
    effective_tif = annual_tif * tif_mult
    eff_rate = max(base_interest_rate + discount_delta, 0.01)

    annual_debt = _annual_debt_service(effective_capital, eff_rate, DEBT_TERM_YEARS)
    farebox = annual_riders * effective_fare * farebox_capture_rate
    revenue = farebox + effective_tif
    dcr = revenue / annual_debt if annual_debt > 0 else 0.0
    self_suff = revenue / (annual_debt + effective_om) if (annual_debt + effective_om) > 0 else 0.0

    disc_factors = np.power(1 + eff_rate, -np.arange(1, cashflow_years + 1))
    annual_net = np.full(cashflow_years, revenue - effective_om)
    npv = float(np.sum(annual_net * disc_factors)) - effective_capital

    return {
        "dcr": round(dcr, 4),
        "self_sufficiency": round(self_suff, 4),
        "npv_musd": round(npv / 1e6, 2),
        "daily_ridership": round(effective_ridership, 0),
        "capital_cost_musd": round(effective_capital / 1e6, 1),
    }


def run_stress_tests(
    results_df: pd.DataFrame,
    config_path: Path | str = Path("scenarios_config.json"),
    cashflow_years: int = 25,
    interest_rate: float = 0.05,
    fare_per_trip_usd: float = 2.00,
    farebox_capture_rate: float = 1.0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run all stress scenarios for all corridors."""
    stress_scenarios = _load_stress_scenarios(config_path)
    if not stress_scenarios:
        return {}

    all_stress: Dict[str, Dict[str, Dict[str, float]]] = {}
    for _, row in results_df.iterrows():
        cid = str(row.get("corridor_id", ""))
        corridor_stress = {}
        for sname, sparams in stress_scenarios.items():
            corridor_stress[sname] = _run_stress_scenario(
                row.to_dict(), sparams,
                cashflow_years=cashflow_years,
                base_interest_rate=interest_rate,
                base_fare=fare_per_trip_usd,
                farebox_capture_rate=farebox_capture_rate,
            )
        all_stress[cid] = corridor_stress
    return all_stress


# ---------------------------------------------------------------------------
# Single corridor evaluation
# ---------------------------------------------------------------------------

def evaluate_corridor_with_financial_analysis(
    corridor_id: str,
    daily_ridership: float,
    length_km: float,
    n_stops: int = 0,
    buffer_distance_m: float = 400,
    capital_cost_per_km: float = 100_000_000,
    scenario: str = "current_zoning",
    data_dir: Path = Path("data"),
    use_property_tax_model: bool = False,
    daily_ridership_series: Any = None,
    cashflow_years: int = 25,
    interest_rate: float = 0.05,
    debt_term_years: int = 25,
    annual_operating_cost_fixed_usd: float = 2_500_000.0,
    annual_operating_cost_per_km_usd: float = 350_000.0,
    fare_per_trip_usd: float = 2.00,
    farebox_capture_rate: float = 1.0,
    financial_viability_threshold: float = 1.0,
    macro_tif_profile: Optional[Dict[str, float]] = None,
    corridor_ridership_share: float = 0.0,
    attempt_spatial: bool = True,
    feedback_df: Optional[pd.DataFrame] = None,
    value_psf: Optional[Dict[str, float]] = None,
    federal_share: float = 0.0,
    state_share: float = 0.0,
) -> Dict[str, Any]:
    """Comprehensive evaluation of one corridor with dynamic-demand finance outputs."""
    years = max(int(cashflow_years), 1)
    if n_stops <= 0:
        n_stops = max(3, int(length_km + 1))

    # Step 1: TIF district delineation
    parcels_in_tif = pd.DataFrame()
    n_parcels_tif = 0
    tif_estimation_source = "spatial"
    spatial_ok = True
    _annual_tif_series = None

    if attempt_spatial:
        try:
            parcels, stations = load_corridor_data(corridor_id, data_dir)
            tif_generator = TIFDistrictGenerator(buffer_distance_m=buffer_distance_m)
            tif_district = tif_generator.create_station_buffers(stations)
            parcels_in_tif = tif_generator.select_parcels_in_tif(parcels, tif_district)
            n_parcels_tif = len(parcels_in_tif)
        except Exception:
            spatial_ok = False
    else:
        spatial_ok = False

    # Step 2/3: Property values and TIF
    base_value = 0.0
    future_value = 0.0
    cumulative_tif = 0.0

    if spatial_ok and n_parcels_tif > 0:
        # Get base value from parcels
        for col in ["total_av", "CurTotAV", "value", "assessed_value"]:
            if col in parcels_in_tif.columns:
                base_value = float(pd.to_numeric(parcels_in_tif[col], errors="coerce").fillna(0).sum())
                break
        if base_value <= 0:
            base_value = float(n_parcels_tif * 175_000)

        _vpsf = value_psf if value_psf else DEFAULT_NEW_CONSTRUCTION_VALUE_PSF
        future_value, cumulative_tif, _annual_tif_mean, _annual_tif_series = _compute_endogenous_tif(
            corridor_id=corridor_id,
            base_value=base_value,
            feedback_df=feedback_df,
            value_psf=_vpsf,
            years=years,
        )
        tif_estimation_source = "endogenous"
    elif macro_tif_profile is not None:
        base_value = macro_tif_profile["base_total"] * corridor_ridership_share
        future_value = macro_tif_profile["future_total"] * corridor_ridership_share
        cumulative_tif = macro_tif_profile["cumulative_tif_total"] * corridor_ridership_share
        tif_estimation_source = "macro_fallback"
    else:
        tif_estimation_source = "none"

    if _annual_tif_series is None:
        _flat_tif = cumulative_tif / years if years > 0 else 0.0
        _annual_tif_series = np.full(years, _flat_tif, dtype=float)

    # Step 4: Financial viability
    _n_stn = max(int(n_stops), 3)
    gross_capital_cost = compute_capital_cost(float(length_km), _n_stn)
    local_share = max(0.0, 1.0 - float(federal_share) - float(state_share))
    capital_cost = gross_capital_cost * local_share
    annual_debt_service = _annual_debt_service(capital_cost, interest_rate, debt_term_years)
    annual_operating_cost = (
        float(annual_operating_cost_fixed_usd) + float(length_km) * float(annual_operating_cost_per_km_usd)
    )

    annual_ridership_series, demand_trace_source = _build_annual_ridership_series(
        daily_ridership=daily_ridership,
        daily_ridership_series=daily_ridership_series,
        years=years,
    )

    demand_responsive_om = _compute_demand_responsive_apm_om(
        annual_ridership_series, length_km=float(length_km), n_stops=n_stops,
    )

    _campus_daily_series = np.zeros(years, dtype=float)
    _campus_payment_series = np.zeros(years, dtype=float)
    if feedback_df is not None and "campus_daily" in feedback_df.columns:
        _cdf = feedback_df[feedback_df["corridor_id"] == corridor_id].sort_values("year")
        if not _cdf.empty:
            _campus_daily_arr = _cdf["campus_daily"].values.astype(float)
            _n = min(len(_campus_daily_arr), years)
            _campus_daily_series[:_n] = _campus_daily_arr[:_n]
            if _n < years:
                _campus_daily_series[_n:] = _campus_daily_series[_n - 1]
            _campus_payment_series = compute_campus_payment_series(_campus_daily_series, years=years)

    net_bus_cost_delta = 0.0
    if feedback_df is not None and "net_bus_cost_delta" in feedback_df.columns:
        _cdf = feedback_df[feedback_df["corridor_id"] == corridor_id]
        if not _cdf.empty:
            net_bus_cost_delta = float(_cdf["net_bus_cost_delta"].iloc[-1])

    dynamic_finance = _compute_dynamic_finance_metrics(
        annual_ridership_series,
        years=years,
        annual_tif_series_usd=_annual_tif_series,
        capital_cost_usd=capital_cost,
        annual_debt_service_usd=annual_debt_service,
        annual_operating_cost_usd=annual_operating_cost,
        discount_rate=interest_rate,
        fare_per_trip_usd=fare_per_trip_usd,
        farebox_capture_rate=farebox_capture_rate,
        demand_responsive_om_series=demand_responsive_om,
        campus_payment_series_usd=_campus_payment_series,
        net_bus_cost_delta_usd=net_bus_cost_delta,
    )

    dscr_year5 = dynamic_finance["dscr_year5"]
    dscr_year25 = dynamic_finance["dscr_year25"]
    dscr_min = dynamic_finance["dscr_min"]
    solvency_ok = bool(dscr_min >= 1.0)
    bond_issuance_ok = bool(dscr_year5 >= 1.25)
    financially_viable = solvency_ok

    om_yr0 = float(demand_responsive_om[0]) if len(demand_responsive_om) > 0 else 0
    om_final = float(demand_responsive_om[-1]) if len(demand_responsive_om) > 0 else 0
    om_mean = dynamic_finance["annual_om_mean_usd"]

    _annual_riders = dynamic_finance["annual_ridership_mean"]
    _annual_subsidy = (
        annual_debt_service + om_mean
        - dynamic_finance["annual_tif_effective_usd"]
        - dynamic_finance["farebox_revenue_annual_mean_musd"] * 1_000_000.0
        - dynamic_finance.get("campus_payment_annual_mean_musd", 0.0) * 1_000_000.0
    )
    cost_per_rider = _annual_subsidy / _annual_riders if _annual_riders > 0 else np.inf

    return {
        "corridor_id": corridor_id,
        "transit_mode": "APM",
        "scenario": scenario,
        "length_km": float(length_km),
        "n_stops": int(n_stops),
        "daily_ridership": dynamic_finance["annual_ridership_mean"] / OPERATING_DAYS_PER_YEAR,
        "daily_ridership_static_input": float(daily_ridership),
        "annual_ridership_series": annual_ridership_series,
        "n_parcels_tif": int(n_parcels_tif),
        "base_property_value": float(base_value),
        "future_property_value": float(future_value),
        "property_value_increase": float(future_value - base_value),
        "tif_revenue_cumulative": float(cumulative_tif),
        "tif_estimation_source": tif_estimation_source,
        "spatial_data_available": spatial_ok,
        "gross_capital_cost": float(gross_capital_cost),
        "federal_share": float(federal_share),
        "state_share": float(state_share),
        "capital_cost_local": float(capital_cost),
        "capital_cost": float(capital_cost),
        "annual_operating_cost_static": float(annual_operating_cost),
        "annual_operating_cost_yr0": float(om_yr0),
        "annual_operating_cost_final": float(om_final),
        "annual_om_mean_usd": float(om_mean),
        "annual_debt_service": float(annual_debt_service),
        "annual_tif_revenue": float(cumulative_tif) / float(years),
        "debt_coverage_ratio_tif_only": (
            float(cumulative_tif) / float(years) / annual_debt_service if annual_debt_service > 0 else 0.0
        ),
        "dscr_year5": float(dscr_year5),
        "dscr_year25": float(dscr_year25),
        "dscr_min": float(dscr_min),
        "debt_coverage_ratio": float(dscr_min),
        "total_coverage_ratio_dynamic": float(dynamic_finance["total_coverage_ratio_dynamic"]),
        "system_self_sufficiency": float(dynamic_finance["system_self_sufficiency"]),
        "net_bus_cost_delta": float(dynamic_finance["net_bus_cost_delta_usd"]),
        "solvency_ok": bool(solvency_ok),
        "bond_issuance_ok": bool(bond_issuance_ok),
        "annual_ridership_effective": float(dynamic_finance["annual_ridership_mean"]),
        "annual_ridership_total_dynamic": float(dynamic_finance["annual_ridership_total"]),
        "project_npv_dynamic_musd": float(dynamic_finance["project_npv_dynamic_musd"]),
        "project_irr_dynamic": float(dynamic_finance["project_irr_dynamic"]),
        "farebox_revenue_annual_mean_musd": float(dynamic_finance["farebox_revenue_annual_mean_musd"]),
        "campus_payment_annual_musd": float(dynamic_finance.get("campus_payment_annual_mean_musd", 0.0)),
        "demand_trace_source": demand_trace_source,
        "demand_trace_years": years,
        "financially_viable": financially_viable,
        "self_sufficiency": float(dynamic_finance.get("system_self_sufficiency", 0.0)),
        "cost_per_rider": float(cost_per_rider),
        "composite_score": float(dynamic_finance.get("system_self_sufficiency", 0.0)),
        "campus_daily_series": _campus_daily_series,
        "campus_ridership_share": float(corridor_ridership_share),
    }


# ---------------------------------------------------------------------------
# Evaluate all corridors
# ---------------------------------------------------------------------------

def evaluate_all_corridors(
    demand_results_path: Path = Path("data/processed/phase2b_corridor_results.csv"),
    buffer_distance_m: float = 400,
    capital_cost_per_km: float = 100_000_000,
    scenario: str = "current_zoning",
    output_path: Path = None,
    dynamic_ridership_path: Path = None,
    feedback_loop_path: Optional[Path] = None,
    cashflow_years: int = 25,
    interest_rate: float = 0.05,
    fare_per_trip_usd: float = 2.00,
    farebox_capture_rate: float = 1.0,
    run_uncertainty: bool = True,
    uncertainty_config_path: Path | str = Path("scenarios_config.json"),
    uncertainty_draws: Optional[int] = None,
    uncertainty_seed: Optional[int] = None,
    uncertainty_percentiles: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Evaluate all corridors with integrated static + dynamic financial analysis."""
    print(f"\n{'=' * 70}")
    print("APM CORRIDOR EVALUATION - INTEGRATED FINANCIAL ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Scenario: {scenario.upper()}")
    print(f"{'=' * 70}")

    if not demand_results_path.exists():
        raise FileNotFoundError(f"Demand results not found: {demand_results_path}")

    demand_results = pd.read_csv(demand_results_path)
    print(f"\nLoaded {len(demand_results)} corridor demand results")

    # Fill length from phase2a metadata when missing
    if "length_km" not in demand_results.columns:
        phase2a_path = Path("data/processed/apm_phase2a_results.csv")
        if phase2a_path.exists():
            phase2a = pd.read_csv(phase2a_path, usecols=["corridor_id", "length_km"])
            phase2a["length_km"] = pd.to_numeric(phase2a["length_km"], errors="coerce")
            phase2a = phase2a.dropna(subset=["corridor_id", "length_km"]).groupby("corridor_id", as_index=False)["length_km"].median()
            demand_results = demand_results.merge(phase2a, on="corridor_id", how="left")
            default_length = float(phase2a["length_km"].median()) if not phase2a.empty else 7.0
            demand_results["length_km"] = pd.to_numeric(demand_results["length_km"], errors="coerce").fillna(default_length)

    if dynamic_ridership_path and Path(dynamic_ridership_path).exists():
        demand_results = integrate_dynamic_ridership_data(
            demand_results,
            dynamic_ridership_path=Path(dynamic_ridership_path),
            cashflow_years=cashflow_years,
        )

    ridership_cols = [c for c in demand_results.columns
                      if ("rider" in str(c).lower()) and ("series" not in str(c).lower())]
    if not ridership_cols:
        raise ValueError("No ridership column found in demand results")
    preferred = ["phase2b_daily_riders", "daily_ridership", "daily_riders"]
    ridership_col = next((c for c in preferred if c in demand_results.columns), ridership_cols[0])

    ridership_values = pd.to_numeric(demand_results[ridership_col], errors="coerce").fillna(0).clip(lower=0)
    ridership_sum = float(ridership_values.sum())
    if ridership_sum > 0:
        demand_results["_corridor_ridership_share"] = (ridership_values / ridership_sum).to_numpy()
    else:
        demand_results["_corridor_ridership_share"] = 1.0 / max(len(demand_results), 1)

    macro_tif_profile = _load_macro_tif_profile(scenario=scenario, years=cashflow_years)

    # Load feedback loop development data
    feedback_df = None
    _fb_path = feedback_loop_path or dynamic_ridership_path
    if _fb_path and Path(_fb_path).exists():
        feedback_df = pd.read_csv(_fb_path)
        if "new_res_sqft" not in feedback_df.columns:
            feedback_df = None

    value_psf = _load_av_value_psf()

    all_results = []
    for _, row in demand_results.iterrows():
        corridor_id = row["corridor_id"]
        try:
            result = evaluate_corridor_with_financial_analysis(
                corridor_id=corridor_id,
                daily_ridership=row[ridership_col],
                daily_ridership_series=row.get("daily_ridership_series", None),
                length_km=row.get("length_km", row.get("length_m", 7000) / 1000),
                n_stops=int(row.get("n_stops", 0)),
                buffer_distance_m=buffer_distance_m,
                capital_cost_per_km=capital_cost_per_km,
                scenario=scenario,
                cashflow_years=cashflow_years,
                interest_rate=interest_rate,
                fare_per_trip_usd=fare_per_trip_usd,
                farebox_capture_rate=farebox_capture_rate,
                macro_tif_profile=macro_tif_profile,
                corridor_ridership_share=float(row.get("_corridor_ridership_share", 0)),
                attempt_spatial=True,
                feedback_df=feedback_df,
                value_psf=value_psf,
            )
            all_results.append(result)

            # BRT mode-compare
            try:
                brt_result = _evaluate_corridor_brt(
                    corridor_id, result,
                    feedback_df=feedback_df,
                    macro_tif_profile=macro_tif_profile,
                    value_psf=value_psf,
                    cashflow_years=cashflow_years,
                    interest_rate=interest_rate,
                    fare_per_trip_usd=fare_per_trip_usd,
                    farebox_capture_rate=farebox_capture_rate,
                )
                all_results.append(brt_result)
            except Exception as brt_exc:
                print(f"\n  BRT evaluation failed for {corridor_id}: {brt_exc}")

        except Exception as exc:
            print(f"\n  ERROR evaluating {corridor_id}: {exc}")
            continue

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        return pd.DataFrame()

    # Rank
    results_df["composite_score"] = results_df.get("self_sufficiency", 0.0)
    if "transit_mode" in results_df.columns:
        results_df["rank"] = results_df.groupby("transit_mode")["self_sufficiency"].rank(
            ascending=False, method="dense",
        ).astype(int)
    else:
        results_df["rank"] = results_df["self_sufficiency"].rank(ascending=False, method="dense").astype(int)
    results_df = results_df.sort_values(
        ["transit_mode", "rank"] if "transit_mode" in results_df.columns else ["rank"]
    )

    # Uncertainty
    if run_uncertainty:
        _apm_df = results_df[results_df.get("transit_mode", "APM") == "APM"] if "transit_mode" in results_df.columns else results_df
        uncertainty_wide_df, uncertainty_long_df, uncertainty_draws_df, uncertainty_meta = compute_uncertainty_bands(
            _apm_df,
            config_path=uncertainty_config_path,
            cashflow_years=cashflow_years,
            fare_per_trip_usd=fare_per_trip_usd,
            farebox_capture_rate=farebox_capture_rate,
            discount_rate=interest_rate,
            n_draws=uncertainty_draws,
            random_seed=uncertainty_seed,
            percentiles=uncertainty_percentiles,
        )
        if not uncertainty_wide_df.empty:
            results_df = results_df.merge(uncertainty_wide_df, on=["corridor_id", "scenario"], how="left")

    # Stress testing
    stress_results = run_stress_tests(results_df, cashflow_years=cashflow_years,
                                       interest_rate=interest_rate, fare_per_trip_usd=fare_per_trip_usd)
    if stress_results:
        stress_rows = []
        for _, row in results_df.iterrows():
            cid = str(row.get("corridor_id", ""))
            row_data = {"corridor_id": cid}
            if cid in stress_results:
                for sname, metrics in stress_results[cid].items():
                    for mkey, mval in metrics.items():
                        row_data[f"stress_{sname}_{mkey}"] = mval
            stress_rows.append(row_data)
        stress_df = pd.DataFrame(stress_rows)
        results_df = results_df.merge(stress_df, on="corridor_id", how="left")

    # Save
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out, index=False)
        print(f"\nResults saved to: {out}")

    return results_df


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    """Run integrated evaluation over available corridor demand outputs.

    Evaluates both scenarios:
    - current_zoning: existing FAR limits, status quo
    - no_zoning: all FAR caps removed within TIF boundary
    """
    _candidates = [
        Path("data/processed/apm_optimized_search_results.csv"),
        Path("data/processed/phase2b_corridor_results.csv"),
        Path("data/processed/apm_phase2a_results.csv"),
    ]
    _existing = [p for p in _candidates if p.exists()]
    if _existing:
        phase2b_path = max(_existing, key=lambda p: p.stat().st_mtime)
    else:
        phase2b_path = _candidates[0]

    if not phase2b_path.exists():
        print("ERROR: No corridor results found")
        return
    print(f"Using corridor results: {phase2b_path}")

    scenarios = [
        ("current_zoning", "corridors_integrated_financial_current_zoning.csv"),
        ("no_zoning", "corridors_integrated_financial_no_zoning.csv"),
    ]

    _search_df = pd.read_csv(phase2b_path)
    _search_ids = set(_search_df["corridor_id"].astype(str).unique())
    print(f"Search results: {len(_search_ids)} corridors ({phase2b_path.name})")

    scenario_results: Dict[str, pd.DataFrame] = {}
    for scenario_name, output_file in scenarios:
        feedback_path = Path(f"data/processed/feedback_loop_results_{scenario_name}.csv")
        if not feedback_path.exists():
            print(f"\nWARNING: Skipping scenario '{scenario_name}' -- "
                  f"feedback file not found: {feedback_path}")
            print(f"  Run: python scripts/run_feedback_loop.py --scenario {scenario_name}")
            continue

        # Corridor-set consistency check
        _fb_df = pd.read_csv(feedback_path, usecols=["corridor_id"])
        _fb_ids = set(_fb_df["corridor_id"].astype(str).unique())
        _only_in_search = _search_ids - _fb_ids
        _only_in_feedback = _fb_ids - _search_ids
        if _only_in_search or _only_in_feedback:
            print(f"\nERROR: Corridor ID mismatch for '{scenario_name}'!")
            if _only_in_search:
                print(f"  In search but not feedback: {sorted(_only_in_search)}")
            if _only_in_feedback:
                print(f"  In feedback but not search: {sorted(_only_in_feedback)}")
            print(f"  Skipping this scenario.")
            continue

        print(f"\n\n{'=' * 70}")
        print(f"SCENARIO: {scenario_name.upper()}")
        print(f"Feedback data: {feedback_path}")
        print(f"{'=' * 70}")

        result_df = evaluate_all_corridors(
            demand_results_path=phase2b_path,
            buffer_distance_m=400,
            capital_cost_per_km=100_000_000,
            scenario=scenario_name,
            output_path=Path("data/processed") / output_file,
            dynamic_ridership_path=feedback_path,
            feedback_loop_path=feedback_path,
        )
        scenario_results[scenario_name] = result_df

    # Cross-scenario comparison
    print(f"\n\n{'=' * 70}")
    print("CROSS-SCENARIO COMPARISON")
    print(f"{'=' * 70}")
    for name, df in scenario_results.items():
        if df.empty:
            print(f"\n  {name.upper()}: no results")
            continue
        n_viable = int(df["financially_viable"].sum())
        mean_dcr = df["debt_coverage_ratio"].mean()
        mean_tif = df["tif_revenue_cumulative"].mean()
        best = df.iloc[0]
        print(f"\n  {name.upper()}:")
        print(f"    Viable corridors: {n_viable}/{len(df)}")
        print(f"    Mean DCR: {mean_dcr:.2f}x")
        print(f"    Mean TIF (25-yr): ${mean_tif / 1e9:.2f}B")
        print(f"    Best corridor: {best['corridor_id']} (DCR {best['debt_coverage_ratio']:.2f}x)")

    # Regenerate viewer data
    if scenario_results:
        try:
            from scripts.run_feedback_loop import _generate_viewer_data

            feedback_paths = {
                sc: f"data/processed/feedback_loop_results_{sc}.csv"
                for sc in scenario_results
                if Path(f"data/processed/feedback_loop_results_{sc}.csv").exists()
            }
            # Discover BRT feedback CSVs so they aren't dropped during regeneration
            brt_paths = {
                sc: f"data/processed/feedback_loop_results_{sc}_brt.csv"
                for sc in feedback_paths
                if Path(f"data/processed/feedback_loop_results_{sc}_brt.csv").exists()
            }
            if feedback_paths:
                print(f"\nRegenerating viewer with evaluation financials...")
                _generate_viewer_data(
                    feedback_paths,
                    evaluation_dfs=scenario_results,
                    brt_paths=brt_paths or None,
                )
        except ImportError:
            pass
        except Exception as e:
            print(f"Viewer regeneration failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

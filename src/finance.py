"""Financial utilities for corridor evaluation (capex, NPV, IRR)."""
from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd

from src.financial_params import (
    PROPERTY_TAX_RATE,
    TIF_CAPTURE_RATE_CONSERVATIVE,
    compute_capital_cost,
    compute_brt_capital_cost,
    CAPITAL_COST_GUIDEWAY_PER_KM,
    CAPITAL_COST_PER_STATION,
    O_AND_M_FIXED_USD,
    O_AND_M_PER_KM_USD,
    O_AND_M_PER_STATION_USD,
    BRT_O_AND_M_FIXED_USD,
    BRT_O_AND_M_PER_KM_USD,
    BRT_O_AND_M_PER_STATION_USD,
    APM_MODE,
    BRT_MODE,
    compute_apm_annual_vehicle_hours,
    compute_brt_annual_vehicle_hours,
)


def capex_musd(
    length_km: float,
    n_stations: int,
    transit_mode: str = "apm",
    **kwargs,
) -> float:
    """Capital cost in millions USD via MECE decomposition.

    Dispatches to APM or BRT cost model based on *transit_mode*.
    """
    if transit_mode == "brt":
        return compute_brt_capital_cost(length_km, n_stations, **kwargs) / 1_000_000.0
    return compute_capital_cost(length_km, n_stations, **kwargs) / 1_000_000.0


def annual_om_musd(
    length_km: float,
    n_stations: int,
    transit_mode: str = "apm",
    headway_min: float | None = None,
) -> float:
    """Annual O&M cost in millions USD using three-tier model.

    When *headway_min* is provided, uses frequency-sensitive Tier 3
    (vehicle operations).  Otherwise falls back to a default headway
    of 5 min (APM) or 7.5 min (BRT) for backward compatibility.
    """
    if transit_mode == "brt":
        hw = headway_min if headway_min is not None else 7.5
        avh = compute_brt_annual_vehicle_hours(length_km, n_stations, hw)
        return BRT_MODE.compute_annual_om(length_km, n_stations, avh) / 1_000_000.0
    hw = headway_min if headway_min is not None else 5.0
    avh = compute_apm_annual_vehicle_hours(length_km, n_stations, hw)
    return APM_MODE.compute_annual_om(length_km, n_stations, avh) / 1_000_000.0


def _coerce_annual_series(
    values: float | Sequence[float] | Mapping[object, float],
    years: int,
    name: str,
    *,
    allow_negative: bool = False,
) -> np.ndarray:
    """Coerce scalar/list/dict values into a validated annual series."""
    n_years = int(years)
    if n_years <= 0:
        raise ValueError("years must be >= 1")

    if isinstance(values, numbers.Number):
        arr = np.full(n_years, float(values), dtype=float)
    elif isinstance(values, Mapping):
        key_to_val: dict[int, float] = {}
        for raw_key, raw_val in values.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} mapping keys must be year integers, got {raw_key!r}") from exc
            key_to_val[key] = float(raw_val)

        keys = set(key_to_val.keys())
        zero_based = set(range(n_years))
        one_based = set(range(1, n_years + 1))
        if keys == zero_based:
            arr = np.array([key_to_val[idx] for idx in range(n_years)], dtype=float)
        elif keys == one_based:
            arr = np.array([key_to_val[idx] for idx in range(1, n_years + 1)], dtype=float)
        else:
            raise ValueError(
                f"{name} mapping keys must match 0..{n_years-1} or 1..{n_years}, got {sorted(keys)}"
            )
    elif isinstance(values, (np.ndarray, list, tuple, pd.Series)):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != n_years:
            raise ValueError(f"{name} length {arr.size} does not match years={n_years}")
    else:
        raise TypeError(
            f"{name} must be a number, sequence, or mapping, got {type(values).__name__}"
        )

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    if not allow_negative and (arr < 0).any():
        raise ValueError(f"{name} cannot contain negative values")
    return arr


def annualize_daily_ridership_series(
    daily_ridership: float | Sequence[float] | Mapping[object, float],
    *,
    years: int = 25,
    operating_days_per_year: float = 312.0,
) -> np.ndarray:
    """Convert daily ridership assumptions to annual rider counts.

    Default 312 equivalent operating days: academic calendar (240 weekdays)
    + summer/break service (72 equivalent days), weighted by Lafayette's
    academic/summer ridership ratio (2.28x).  Aligns with
    OPERATING_DAYS_PER_YEAR in src/financial_params.py.
    """
    daily = _coerce_annual_series(daily_ridership, years, "daily_ridership")
    days = float(operating_days_per_year)
    if not np.isfinite(days) or days <= 0:
        raise ValueError("operating_days_per_year must be positive and finite")
    return daily * days


def annual_ridership_to_revenue_musd(
    annual_ridership: float | Sequence[float] | Mapping[object, float],
    *,
    fare_per_trip_usd: float,
    capture_rate: float = 1.0,
    years: int = 25,
) -> np.ndarray:
    """Convert annual rider series to annual fare revenue (million USD)."""
    riders = _coerce_annual_series(annual_ridership, years, "annual_ridership")
    fare = float(fare_per_trip_usd)
    capture = float(capture_rate)
    if not np.isfinite(fare) or fare < 0:
        raise ValueError("fare_per_trip_usd must be non-negative and finite")
    if not np.isfinite(capture) or capture < 0:
        raise ValueError("capture_rate must be non-negative and finite")
    return (riders * fare * capture) / 1_000_000.0


def npv_irr(
    annual_revenue_musd: float | Sequence[float] | Mapping[object, float],
    capex_musd_val: float,
    years: int = 25,
    discount_rate: float = 0.05,
    annual_cost_musd: float | Sequence[float] | Mapping[object, float] = 0.0,
) -> dict:
    """Compute NPV/IRR from annual revenue/cost series (or scalar assumptions)."""
    capex = float(capex_musd_val)
    if not np.isfinite(capex) or capex < 0:
        raise ValueError("capex_musd_val must be non-negative and finite")

    revenue_series = _coerce_annual_series(annual_revenue_musd, years, "annual_revenue_musd")
    cost_series = _coerce_annual_series(annual_cost_musd, years, "annual_cost_musd")
    net_series = revenue_series - cost_series

    cashflows = [-capex] + net_series.tolist()
    years_arr = np.arange(0, years + 1)
    npv_val = float(np.sum(np.array(cashflows) / np.power(1 + discount_rate, years_arr)))
    try:
        import numpy_financial as npf
        irr_val = float(npf.irr(cashflows))
    except ImportError:
        import warnings
        warnings.warn(
            "numpy_financial not installed — IRR will be NaN. "
            "Install with: pip install numpy-financial",
            stacklevel=2,
        )
        irr_val = float("nan")
    except Exception:
        irr_val = float("nan")
    return {
        "npv_musd": npv_val,
        "irr": irr_val,
        "annual_revenue_series_musd": revenue_series,
        "annual_cost_series_musd": cost_series,
        "annual_net_cashflow_musd": net_series,
        "cashflows_musd": np.array(cashflows, dtype=float),
    }


def tif_revenue_from_uplift(
    uplift_total_usd: float,
    capture_rate: float = TIF_CAPTURE_RATE_CONSERVATIVE,
    tax_rate: float = PROPERTY_TAX_RATE,
) -> float:
    """Compute annual TIF revenue (USD) from uplift and capture rate.

    Defaults to the conservative 85% capture rate (admin overhead, assessment
    lag, appeals).  Pass ``capture_rate=1.0`` for the Indiana IC 36-7-14
    statutory default.
    """
    return uplift_total_usd * capture_rate * tax_rate


# ===================================================================
# TIF annual/cumulative revenue with Indiana-specific adjustments
# ===================================================================

from src.financial_params import (
    TIF_CAPTURE_RATE_CONSERVATIVE as _DEFAULT_CAPTURE,
    PROPERTY_TAX_RATE as _DEFAULT_TAX_RATE,
    effective_tif_tax_rate as _effective_tif_tax_rate,
    TIF_RESIDENTIAL_MAX_YEARS,
    TIF_AREA_TYPE_DEFAULT,
    SB1_CAPTURE_EROSION,
)
from typing import Callable, Tuple, List


def _default_phasing(year: int) -> float:
    """Fallback phasing when no external phasing function is provided.

    Mirrors the TCRP 128 S-curve used in tif_financing_model.py.
    """
    if year <= 0:
        return 0.0
    if year <= 2:
        return 0.05 * year / 2.0
    if year <= 5:
        return 0.05 + 0.28 * (year - 2) / 3.0
    if year <= 15:
        return 0.33 + 0.52 * (year - 5) / 10.0
    return 1.0


def interpolate_sb1_erosion(year: int) -> float:
    """Linearly interpolate SB 1 capture-rate erosion for a given year."""
    knots = sorted(SB1_CAPTURE_EROSION.items())
    if year <= knots[0][0]:
        return knots[0][1]
    if year >= knots[-1][0]:
        return knots[-1][1]
    for i in range(len(knots) - 1):
        y0, v0 = knots[i]
        y1, v1 = knots[i + 1]
        if y0 <= year <= y1:
            frac = (year - y0) / (y1 - y0)
            return v0 + frac * (v1 - v0)
    return knots[-1][1]  # pragma: no cover


def tif_annual_revenue(
    increment_value: float,
    year: int,
    capture_rate: float = _DEFAULT_CAPTURE,
    property_tax_rate: float = _DEFAULT_TAX_RATE,
    res_share: float = 0.5,
    phasing_fn: Callable[[int], float] = _default_phasing,
    area_type: str = TIF_AREA_TYPE_DEFAULT,
    apply_sb1_erosion: bool = True,
    assessment_lag: bool = True,
) -> float:
    """Single-year TIF revenue with all Indiana-specific adjustments.

    Applies: EDA/Redevelopment area type rules, circuit breaker effective
    rate, TCRP 128 phasing, HEA 1120 residential 20-year cap, 1-year
    assessment lag (Indiana taxes paid in arrears), and SB 1 (2025)
    capture-rate erosion.

    Parameters
    ----------
    increment_value : total assessed value increment (future - base)
    year : TIF year (1-indexed)
    capture_rate : statutory/effective capture rate (before SB 1 erosion)
    property_tax_rate : gross levy rate
    res_share : residential share of TIF district for circuit breaker
    phasing_fn : callable(year) -> float in [0, 1] for development phasing
    area_type : "eda" or "redevelopment"
        - EDA (IC 36-7-14-39(a)): residential increment generates zero TIF.
          Only commercial/industrial share produces revenue.
        - Redevelopment (IC 36-7-14-15): all increment captured, subject to
          HEA 1120 20-year residential cap.
    apply_sb1_erosion : if True, apply SB 1 (2025) time-varying erosion to
        the capture rate (new exemptions reduce effective capture over time)
    assessment_lag : if True, shift revenue by 1 year (Indiana taxes paid
        in arrears — assessment in year N drives taxes in year N+1)
    """
    # Assessment lag: revenue for year N is based on assessment in year N-1
    effective_year = (year - 1) if assessment_lag else year
    if effective_year <= 0:
        return 0.0

    # SB 1 capture-rate erosion
    adj_capture = capture_rate
    if apply_sb1_erosion:
        adj_capture *= interpolate_sb1_erosion(year)

    phasing = phasing_fn(effective_year)

    # Area type determines which increment generates TIF
    if area_type == "eda":
        # EDA (IC 36-7-14-39(a)): residential AV is continuously added back
        # to the base.  Only the commercial/industrial SHARE of the total
        # increment generates TIF, taxed at the commercial effective rate.
        comm_share = 1.0 - res_share
        comm_rate = min(property_tax_rate, 0.03)  # circuit breaker cap
        annual = increment_value * comm_share * phasing * comm_rate * adj_capture
    else:
        # Redevelopment area: all increment captured at blended rate
        eff_rate = _effective_tif_tax_rate(property_tax_rate, res_share)
        annual = increment_value * phasing * eff_rate * adj_capture
        # HEA 1120: residential share expires after 20 years
        if year > TIF_RESIDENTIAL_MAX_YEARS:
            annual *= (1.0 - res_share)

    return annual


def tif_cumulative_revenue(
    increment_value: float,
    years: int,
    **kwargs,
) -> Tuple[float, List[float]]:
    """Cumulative TIF over N years, returning (total, year_by_year_list)."""
    annual = []
    for yr in range(1, years + 1):
        annual.append(tif_annual_revenue(increment_value, yr, **kwargs))
    return sum(annual), annual

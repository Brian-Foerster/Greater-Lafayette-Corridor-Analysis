"""Vacancy-driven rent feedback functions.

Extracted as standalone functions so tests can import them directly
without pulling in the heavy LandUseTransportModel (geopandas/osmnx).
The model's _update_rents_from_vacancy() method delegates to these.
"""
from __future__ import annotations

import numpy as np

from src.relocation_model import TARGET_VAC_RES

# Per percentage-point deviation, per year.
# CoStar submarket data for secondary Midwest metros: when vacancy
# exceeds natural rate by 5+ ppt, effective rents decline 10-20%/yr.
# Asymmetry: rents rise faster than they fall (Genesove & Mayer 2001).
RENT_UP_SPEED = 0.04     # undersupply -> rents rise
RENT_DOWN_SPEED = 0.03   # oversupply -> rents fall (sticky downward)
MAX_ANNUAL_CHANGE = 0.15  # +/-15% per year cap (submarket-level)
TARGET_VAC_COMM = 0.10
MIN_RENT_PSF = 5.0
MAX_RENT_PSF = 150.0  # 10× Lafayette mean (~$15/sqft); prevents runaway rents


# Transit proximity rent floor: empirical evidence (Singer 2025, Jiang et al.
# 2020) shows rents within 800m of rail stations never fall below ~82% of
# initial levels even during building booms.  Captive demand (reduced car
# dependency, walkability amenity) creates a floor that prevents the full
# vacancy-driven spiral.
TRANSIT_RENT_FLOOR = 0.82


def update_rents_from_vacancy(
    current_rents: np.ndarray,
    total_units: np.ndarray,
    occupied_units: np.ndarray,
    corridor_positions: np.ndarray,
    step_years: int = 1,
    total_comm_sqft: np.ndarray | None = None,
    occupied_comm_sqft: np.ndarray | None = None,
    recently_built: np.ndarray | None = None,
    initial_rents: np.ndarray | None = None,
    station_area_mask: np.ndarray | None = None,
) -> dict:
    """Adjust per-parcel rents based on local vacancy.

    Asymmetric: rents rise faster than they fall (real-world stickiness).
    Multi-year steps use compound scaling: ``annual_adj ** step_years``.

    Parcels with zero total_units (no housing stock) get no rent adjustment
    since there is no vacancy signal.  This is correct for truly empty parcels,
    but parcels with existing housing that are not flagged as developable will
    also have frozen rents if total_units was never initialized from baseline
    housing stock.  The caller should seed total_units from ACS/parcel data.

    Parameters
    ----------
    current_rents : array, mutated in-place
    total_units, occupied_units : residential unit arrays
    corridor_positions : indices of parcels to adjust
    step_years : duration of this time step in years
    total_comm_sqft, occupied_comm_sqft : optional commercial arrays.
        When provided, commercial vacancy is blended into the adjustment.

    Returns dict with diagnostic info: mean_vacancy, mean_rent_change, etc.
    """
    pos = corridor_positions.astype(int)
    if len(pos) == 0:
        return {"mean_vacancy": 0.0, "mean_rent_change": 1.0}

    # Residential vacancy
    total_u = total_units[pos]
    occ_u = occupied_units[pos]
    has_units = total_u > 0

    # Detect occupied > total (data inconsistency from unsynchronized updates)
    overcount = has_units & (occ_u > total_u)
    n_overcount = int(overcount.sum())
    if n_overcount > 0:
        import warnings
        warnings.warn(
            f"vacancy_rent_feedback: {n_overcount} parcels have occupied_units > "
            f"total_units (max excess: {float((occ_u - total_u)[overcount].max()):.1f}). "
            f"Clamping occupied to total.",
            stacklevel=2,
        )
        occ_u = np.minimum(occ_u, total_u)

    _occ_ratio = np.divide(occ_u, total_u, out=np.zeros_like(occ_u), where=has_units)
    vacancy = np.where(
        has_units,
        np.maximum(0.0, 1.0 - _occ_ratio),
        TARGET_VAC_RES,
    )

    deviation = vacancy - TARGET_VAC_RES
    # Per-year adjustment from deviation (percentage points × speed)
    annual_adj = np.where(
        deviation > 0,
        1.0 - RENT_DOWN_SPEED * deviation * 100.0,
        1.0 + RENT_UP_SPEED * (-deviation) * 100.0,
    )
    # Clamp per-year, then compound over step_years
    annual_adj = np.clip(annual_adj, 1.0 - MAX_ANNUAL_CHANGE, 1.0 + MAX_ANNUAL_CHANGE)
    adjustment = np.power(annual_adj, step_years)

    # Commercial vacancy blending (when arrays provided)
    has_comm = None
    comm_vacancy = None
    if total_comm_sqft is not None and occupied_comm_sqft is not None:
        total_cs = total_comm_sqft[pos]
        occ_cs = occupied_comm_sqft[pos]
        has_comm = total_cs > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            comm_vacancy = np.where(
                has_comm,
                np.maximum(0.0, 1.0 - occ_cs / total_cs),
                TARGET_VAC_COMM,
            )
        comm_deviation = comm_vacancy - TARGET_VAC_COMM
        comm_annual = np.where(
            comm_deviation > 0,
            1.0 - RENT_DOWN_SPEED * comm_deviation * 100.0,
            1.0 + RENT_UP_SPEED * (-comm_deviation) * 100.0,
        )
        comm_annual = np.clip(comm_annual, 1.0 - MAX_ANNUAL_CHANGE, 1.0 + MAX_ANNUAL_CHANGE)
        comm_adjustment = np.power(comm_annual, step_years)

        # Blend: 70% residential, 30% commercial where both exist.
        # Reflects Lafayette's housing-dominated market; would need recalibration
        # for office/retail-heavy districts.  Ideally derived from parcel-level
        # use mix, but requires building-level data not yet available.
        both = has_units & has_comm
        blended = adjustment.copy()
        blended[both] = 0.70 * adjustment[both] + 0.30 * comm_adjustment[both]
        comm_only = has_comm & ~has_units
        blended[comm_only] = comm_adjustment[comm_only]
    else:
        blended = adjustment

    # New-stock rent stickiness: recently built parcels resist downward
    # adjustment (developer holds asking rents for debt service / loan covenants).
    # Upward adjustment is unchanged — new stock captures market gains.
    if recently_built is not None:
        _new_mask = recently_built[pos]
        _downward = blended < 1.0
        _sticky = _new_mask & _downward
        if _sticky.any():
            blended[_sticky] = 1.0 - 0.5 * (1.0 - blended[_sticky])

    current_rents[pos] *= blended

    # Transit rent floor: station-area parcels cannot fall below 82% of
    # initial rents.  Prevents vacancy death spiral while preserving the
    # feedback signal for moderate oversupply.
    if initial_rents is not None and station_area_mask is not None:
        _transit = station_area_mask[pos]
        if _transit.any():
            _floor = initial_rents[pos] * TRANSIT_RENT_FLOOR
            current_rents[pos] = np.where(
                _transit,
                np.maximum(current_rents[pos], _floor),
                current_rents[pos],
            )

    current_rents[pos] = np.clip(current_rents[pos], MIN_RENT_PSF, MAX_RENT_PSF)

    mean_vac = float(vacancy[has_units].mean()) if has_units.any() else TARGET_VAC_RES
    mean_adj = float(blended.mean())

    result = {
        "mean_vacancy": mean_vac,
        "mean_rent_change": mean_adj,
        "parcels_with_units": int(has_units.sum()),
        "parcels_total": len(pos),
    }
    if has_comm is not None:
        mean_comm_vac = float(comm_vacancy[has_comm].mean()) if has_comm.any() else TARGET_VAC_COMM
        result["mean_comm_vacancy"] = mean_comm_vac
        result["parcels_with_comm"] = int(has_comm.sum())

    return result


def track_unit_delivery(
    total_units: np.ndarray,
    occupied_units: np.ndarray,
    total_comm_sqft: np.ndarray,
    occupied_comm_sqft: np.ndarray,
    pos,
    res_sqft: float = 0.0,
    comm_sqft: float = 0.0,
    avg_unit_sqft: float = 900.0,
) -> None:
    """Update per-parcel unit/occupancy trackers when sqft is delivered.

    New residential units are delivered **vacant** — absorption happens via
    :func:`absorb_vacant_units` at the start of the next time step.  This
    creates a lease-up lag so vacancy rises when supply floods in and rents
    can respond before the next round of proforma feasibility checks.

    Commercial space still fills at 90% (tenant pre-leasing is standard).
    """
    _COMM_FILL_RATE = 0.90
    if res_sqft > 0:
        new_units = res_sqft / avg_unit_sqft
        total_units[pos] += new_units
        # Units start vacant — absorbed next year via absorb_vacant_units()
    if comm_sqft > 0:
        total_comm_sqft[pos] += comm_sqft
        occupied_comm_sqft[pos] += comm_sqft * _COMM_FILL_RATE


def track_unit_delivery_fallback(
    total_units: np.ndarray,
    occupied_units: np.ndarray,
    near_idx: np.ndarray,
    frac: np.ndarray,
    pop: float = 0.0,
    avg_household_size: float = 2.56,
) -> None:
    """Update unit trackers for fallback deliveries (aggregate pop)."""
    if pop > 0:
        units = pop / avg_household_size
        per_parcel_units = frac * units
        total_units[near_idx] += per_parcel_units
        # Units start vacant — absorbed next year via absorb_vacant_units()


# NMHC 2023: average lease-up = 15-18 months for Class A apartments
# in secondary markets.  ~60-70% of vacant units fill per year.
ANNUAL_FILL_RATE = 0.65


def absorb_vacant_units(
    total_units: np.ndarray,
    occupied_units: np.ndarray,
    positions: np.ndarray,
    step_years: int = 1,
) -> float:
    """Fill vacant residential units at empirical absorption rate.

    Called once per corridor per time step, *before* rent feedback, so that
    vacancy reflects both new (empty) deliveries and gradual lease-up of
    prior deliveries.

    Returns total units absorbed this step.
    """
    vacant = total_units[positions] - occupied_units[positions]
    # Only absorb where there are actual vacancies
    vacant = np.maximum(vacant, 0.0)
    absorbed = vacant * (1.0 - (1.0 - ANNUAL_FILL_RATE) ** step_years)
    occupied_units[positions] += absorbed
    return float(absorbed.sum())

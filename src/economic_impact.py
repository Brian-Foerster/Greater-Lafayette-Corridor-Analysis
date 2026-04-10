"""Economic & fiscal impact analysis for APM corridors.

Implements USDOT BCA framework (2024), broader fiscal impact, economic
activity multipliers, and FTA Small Starts metrics.  Follows the 7
accounting rules in docs/ECONOMIC_FISCAL_IMPACT_PLAN.md to prevent
double-counting.

Key accounting rules enforced:
  Rule 1 — Financial viability and societal BCA are separate analyses.
  Rule 2 — Property uplift in BCR = spillover beyond TIF boundary only.
  Rule 3 — Farebox is a cost offset, not a benefit.
  Rule 4 — Multiplier analysis is presented separately, never in BCR.
  Rule 5 — Regional reallocation is net-zero at metro level.
  Rule 6 — Agglomeration uses lower-bound elasticity (deferred to Phase 5).
  Rule 7 — TOD construction jobs: no second multiplier.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.financial_params import (
    # Existing
    CAPITAL_COST_PER_KM,
    compute_capital_cost,
    BOND_RATE,
    DEBT_TERM_YEARS,
    O_AND_M_FIXED_USD,
    O_AND_M_PER_KM_USD,
    O_AND_M_PER_STATION_USD,
    O_AND_M_ESCALATION_RATE,
    FARE_PER_TRIP_USD,
    OPERATING_DAYS_PER_YEAR,
    PROPERTY_TAX_RATE,
    TIF_CAPTURE_RATE_CONSERVATIVE,
    TIF_YEARS,
    # BCA
    BCR_DISCOUNT_RATE_LOW,
    BCR_DISCOUNT_RATE_HIGH,
    VTTS_PERSONAL_PER_HOUR,
    VTTS_BUSINESS_PER_HOUR,
    VTTS_REAL_GROWTH_RATE,
    VOC_MARGINAL_PER_MILE,
    VOC_FULL_PER_MILE,
    CRASH_COST_PER_VMT,
    CO2_TONS_PER_VMT,
    SOCIAL_COST_CO2_PER_TON,
    SC_CO2_REAL_ESCALATION,
    CRITERIA_POLLUTANT_PER_VMT,
    TOTAL_EMISSION_COST_PER_VMT,
    WALK_HEALTH_VALUE_PER_MIN,
    AVG_WALK_ACCESS_MIN_PER_TRIP,
    NEW_WALKING_SHARE,
    STRUCTURED_PARKING_COST_PER_SPACE,
    PARKING_STRUCTURE_LIFE_YEARS,
    PEAK_HOUR_PARKING_FACTOR,
    PER_CAPITA_MUNICIPAL_SERVICE_COST,
    AVG_HOME_VALUE_NEW_DEVELOPMENT,
    RESIDUAL_VALUE_SHARE,
    # Employment
    CONSTRUCTION_JOBS_PER_MILLION,
    CONSTRUCTION_LABOR_SHARE,
    CONSTRUCTION_TYPE_II_MULTIPLIER,
    AVG_CONSTRUCTION_WAGE,
    APM_OPS_JOBS_PER_KM,
    RESIDENTIAL_CONSTRUCTION_COST_PSF,
    COMMERCIAL_CONSTRUCTION_COST_PSF,
    # Fiscal
    TIPPECANOE_LIT_RATE,
    INDIANA_STATE_INCOME_TAX_RATE,
    INDIANA_SALES_TAX_RATE,
    RETAIL_SHARE_OF_COMMERCIAL,
    RETAIL_SALES_PER_SQFT,
    AVG_WAGE_WEIGHTED,
    ROAD_MAINTENANCE_COST_PER_VMT,
    TIF_RESIDENTIAL_MAX_YEARS,
    # Car diversion
    CAR_DIVERSION_COMMUTE,
    CAR_DIVERSION_STUDENT,
    CAR_DIVERSION_GENERATOR,
    CAR_DIVERSION_INDUCED,
    CAR_DIVERSION_LATENT,
    # Monte Carlo
    MC_VTTS_MULT,
    MC_CRASH_COST_MULT,
    MC_SC_CO2_MULT,
    MC_WALK_HEALTH_VALUE,
    MC_AGGLOMERATION_ELASTICITY,
)


# ───────────────────────────────────────────────────────────────────
# Data structure
# ───────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class CorridorCBAResult:
    """Full CBA result for one corridor under one scenario.

    Fields are labeled by accounting frame:
      [SOCIETAL]  — Benefit-cost analysis (welfare gains vs. resource costs)
      [ECONOMIC]  — Gross economic activity (NOT additive to BCR)
      [FISCAL]    — Government revenue impact
      [EQUITY]    — Distributional analysis
      [FTA]       — FTA Small Starts metrics
    """
    corridor_id: str
    scenario: str

    # --- [SOCIETAL] intermediate quantities ---
    daily_diverted_car_trips: float = 0.0
    annual_vmt_avoided_miles: float = 0.0
    annual_person_hours_saved: float = 0.0

    # Benefits (NPV, million USD)
    benefit_travel_time_3pct: float = 0.0
    benefit_travel_time_7pct: float = 0.0
    benefit_voc_3pct: float = 0.0
    benefit_voc_7pct: float = 0.0
    benefit_safety_3pct: float = 0.0
    benefit_safety_7pct: float = 0.0
    benefit_emissions_3pct: float = 0.0
    benefit_emissions_7pct: float = 0.0
    benefit_health_3pct: float = 0.0
    benefit_health_7pct: float = 0.0
    benefit_parking_3pct: float = 0.0
    benefit_parking_7pct: float = 0.0
    benefit_agglomeration_3pct: float = 0.0  # Phase 5 stub
    benefit_agglomeration_7pct: float = 0.0
    residual_value_3pct: float = 0.0
    residual_value_7pct: float = 0.0
    total_benefits_3pct: float = 0.0
    total_benefits_7pct: float = 0.0

    # Costs (NPV, million USD)
    cost_capital_3pct: float = 0.0
    cost_capital_7pct: float = 0.0
    cost_om_3pct: float = 0.0
    cost_om_7pct: float = 0.0
    cost_public_services_3pct: float = 0.0
    cost_public_services_7pct: float = 0.0
    total_costs_3pct: float = 0.0
    total_costs_7pct: float = 0.0

    # BCR
    bcr_3pct: float = 0.0
    bcr_7pct: float = 0.0
    bcr_3pct_p10: float = 0.0
    bcr_3pct_p50: float = 0.0
    bcr_3pct_p90: float = 0.0
    bcr_7pct_p10: float = 0.0
    bcr_7pct_p50: float = 0.0
    bcr_7pct_p90: float = 0.0

    # --- [ECONOMIC] gross activity (NOT additive to BCR) ---
    construction_jobs_apm: float = 0.0
    construction_jobs_tod: float = 0.0
    construction_earnings_musd: float = 0.0
    permanent_ops_jobs: float = 0.0
    permanent_tod_jobs: float = 0.0
    private_investment_musd: float = 0.0
    leverage_ratio_gross: float = 0.0
    leverage_ratio_net: float = 0.0

    # --- [FISCAL] government revenue ---
    tif_cumulative_25yr_musd: float = 0.0
    tif_residential_truncated_musd: float = 0.0
    lit_new_jobs_annual: float = 0.0
    lit_construction_annual: float = 0.0
    state_income_tax_annual: float = 0.0
    sales_tax_retail_annual: float = 0.0
    road_maintenance_savings_annual: float = 0.0
    parking_avoided_onetime_musd: float = 0.0
    net_public_service_cost_annual: float = 0.0
    net_fiscal_benefit_annual_yr10: float = 0.0
    net_fiscal_benefit_cumulative_25yr: float = 0.0

    # --- [EQUITY] ---
    transport_savings_se01_annual: float = 0.0
    zero_car_hh_served: int = 0

    # --- [FTA] ---
    fta_cost_per_trip: float = 0.0
    fta_annual_trips_per_musd: float = 0.0

    def to_dict(self) -> dict:
        """Flatten to dict for CSV export."""
        return dataclasses.asdict(self)


# ───────────────────────────────────────────────────────────────────
# NPV helper
# ───────────────────────────────────────────────────────────────────

def _npv(annual_series: np.ndarray, discount_rate: float) -> float:
    """Compute NPV of a series where index 0 = year 1, etc.

    Capital costs at year 0 should be handled separately (no discounting).
    """
    years = np.arange(1, len(annual_series) + 1)
    return float(np.sum(annual_series / np.power(1 + discount_rate, years)))


# ───────────────────────────────────────────────────────────────────
# Step 1: Diverted car trips
# ───────────────────────────────────────────────────────────────────

def compute_diverted_car_trips(results_df: pd.DataFrame) -> float:
    """Compute daily diverted car trips from ridership components.

    Uses component-specific car diversion shares:
      - Commute (LODES): 60% were driving
      - Student: 30% (lower car ownership)
      - Generator (medical/retail/event): 55%
      - Induced/latent: 0% (these trips didn't exist or had no car)

    Args:
        results_df: Feedback loop results for one corridor (all years).
            Accepts awareness-adjusted columns (work_commute_daily, etc.)
            or raw columns (both_ends_daily_raw, etc.) or legacy names.

    Returns:
        Average daily diverted car trips over mature years (year >= 10).
    """
    mature = results_df[results_df["year"] >= 10]
    if mature.empty:
        mature = results_df

    def _col(df, *names):
        """Return the first matching column as a Series, or zeros."""
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series([0.0] * len(df), index=df.index)

    # Use awareness-adjusted display columns if available, else raw/legacy
    commute = (
        _col(mature, "work_commute_daily", "both_ends_daily_raw", "both_ends_daily").mean()
        + _col(mature, "local_nonwork_daily", "origin_only_daily_raw", "origin_only_daily").mean()
    )
    student = _col(mature, "campus_daily", "student_apm_daily_raw", "student_apm_daily").mean()
    generator = _col(mature, "destination_daily", "generator_daily_raw", "generator_daily").mean()
    induced = _col(mature, "induced_daily_raw", "induced_daily").mean()
    latent = _col(mature, "latent_daily_raw", "latent_daily").mean()

    return (
        commute * CAR_DIVERSION_COMMUTE
        + student * CAR_DIVERSION_STUDENT
        + generator * CAR_DIVERSION_GENERATOR
        + induced * CAR_DIVERSION_INDUCED
        + latent * CAR_DIVERSION_LATENT
    )


# ───────────────────────────────────────────────────────────────────
# Step 2: VMT avoided
# ───────────────────────────────────────────────────────────────────

def compute_annual_vmt_avoided(
    daily_diverted_car_trips: float,
    avg_car_trip_miles: float,
) -> float:
    """Annual vehicle-miles of travel avoided by car-to-APM diversion.

    Args:
        daily_diverted_car_trips: From compute_diverted_car_trips().
        avg_car_trip_miles: Average one-way car trip distance in miles.
            Typically: corridor_length_km × CAR_CIRCUITY × 0.621371 × 0.5
            (half corridor length as average trip).

    Returns:
        Annual VMT avoided (round-trip × operating days).
    """
    return daily_diverted_car_trips * avg_car_trip_miles * 2.0 * OPERATING_DAYS_PER_YEAR


# ───────────────────────────────────────────────────────────────────
# Step 3: Travel time savings
# ───────────────────────────────────────────────────────────────────

def compute_travel_time_savings(
    daily_diverted_car_trips: float,
    car_travel_time_min: float,
    apm_travel_time_min: float,
    *,
    business_share: float = 0.20,
    years: int = DEBT_TERM_YEARS,
) -> Tuple[float, np.ndarray]:
    """Annual person-hours saved and monetized 25-year series.

    Args:
        daily_diverted_car_trips: Daily car-to-APM diversions.
        car_travel_time_min: Average car trip time (minutes).
        apm_travel_time_min: Average APM trip time including walk + wait (minutes).
        business_share: Share of trips at business VTTS (default 20% for commute).
        years: Analysis horizon.

    Returns:
        (annual_person_hours_saved, monetized_annual_series) where the series
        is a 25-element array in USD with 1.2%/yr real VTTS growth.
    """
    time_saved_per_trip_min = car_travel_time_min - apm_travel_time_min
    if time_saved_per_trip_min <= 0:
        return 0.0, np.zeros(years)

    annual_person_hours = (
        daily_diverted_car_trips
        * time_saved_per_trip_min
        / 60.0
        * OPERATING_DAYS_PER_YEAR
    )

    # Income-weighted VTTS
    vtts_base = (
        (1.0 - business_share) * VTTS_PERSONAL_PER_HOUR
        + business_share * VTTS_BUSINESS_PER_HOUR
    )

    # 1.2%/yr real growth in VTTS (USDOT guidance)
    growth = np.power(1.0 + VTTS_REAL_GROWTH_RATE, np.arange(1, years + 1))
    monetized = annual_person_hours * vtts_base * growth

    return annual_person_hours, monetized


# ───────────────────────────────────────────────────────────────────
# Step 4: BCR benefit series
# ───────────────────────────────────────────────────────────────────

def compute_bcr_benefits(
    annual_vmt_avoided: float,
    travel_time_series: np.ndarray,
    daily_riders: float,
    capital_cost_usd: float,
    *,
    years: int = DEBT_TERM_YEARS,
) -> Dict[str, float]:
    """Compute all BCR benefit NPVs at 3% and 7%.

    Returns dict with keys like 'travel_time_3pct', 'voc_3pct', etc.
    All values in million USD.
    """
    results = {}

    for label, rate in [("3pct", BCR_DISCOUNT_RATE_LOW), ("7pct", BCR_DISCOUNT_RATE_HIGH)]:
        # A. Travel time savings (already monetized with VTTS growth)
        results[f"travel_time_{label}"] = _npv(travel_time_series, rate) / 1e6

        # B. VOC savings ($0.22/mile)
        voc_series = np.full(years, annual_vmt_avoided * VOC_MARGINAL_PER_MILE)
        results[f"voc_{label}"] = _npv(voc_series, rate) / 1e6

        # C. Safety ($0.12/VMT)
        safety_series = np.full(years, annual_vmt_avoided * CRASH_COST_PER_VMT)
        results[f"safety_{label}"] = _npv(safety_series, rate) / 1e6

        # D. Emissions ($0.086/VMT base, with 2.5%/yr SC-CO2 escalation)
        base_emission_usd = annual_vmt_avoided * TOTAL_EMISSION_COST_PER_VMT
        co2_escalation = np.power(1.0 + SC_CO2_REAL_ESCALATION, np.arange(1, years + 1))
        # Only CO2 component escalates; criteria pollutants stay flat
        co2_base = annual_vmt_avoided * CO2_TONS_PER_VMT * SOCIAL_COST_CO2_PER_TON
        criteria_base = annual_vmt_avoided * CRITERIA_POLLUTANT_PER_VMT
        emission_series = co2_base * co2_escalation + criteria_base
        results[f"emissions_{label}"] = _npv(emission_series, rate) / 1e6

        # E. Health benefits from walking
        annual_walk_min = (
            daily_riders
            * AVG_WALK_ACCESS_MIN_PER_TRIP
            * OPERATING_DAYS_PER_YEAR
            * NEW_WALKING_SHARE
        )
        health_series = np.full(years, annual_walk_min * WALK_HEALTH_VALUE_PER_MIN)
        results[f"health_{label}"] = _npv(health_series, rate) / 1e6

        # F. Parking avoided (avoided structured parking construction)
        avoided_spaces = daily_riders * CAR_DIVERSION_COMMUTE * PEAK_HOUR_PARKING_FACTOR * 0.3
        annual_parking_savings = (
            avoided_spaces
            * STRUCTURED_PARKING_COST_PER_SPACE
            / PARKING_STRUCTURE_LIFE_YEARS
        )
        parking_series = np.full(years, annual_parking_savings)
        results[f"parking_{label}"] = _npv(parking_series, rate) / 1e6

        # G. Agglomeration (Phase 5 stub — returns 0)
        results[f"agglomeration_{label}"] = 0.0

        # H. Residual value (50% of capital at year 25)
        residual_usd = capital_cost_usd * RESIDUAL_VALUE_SHARE
        results[f"residual_{label}"] = residual_usd / (1 + rate) ** years / 1e6

        # Total benefits
        results[f"total_benefits_{label}"] = sum(
            results[f"{cat}_{label}"]
            for cat in [
                "travel_time", "voc", "safety", "emissions",
                "health", "parking", "agglomeration", "residual",
            ]
        )

    return results


# ───────────────────────────────────────────────────────────────────
# Step 5: BCR cost series
# ───────────────────────────────────────────────────────────────────

def compute_bcr_costs(
    capital_cost_usd: float,
    length_km: float,
    n_stations: int,
    cumulative_new_pop: float,
    *,
    years: int = DEBT_TERM_YEARS,
) -> Dict[str, float]:
    """Compute all BCR cost NPVs at 3% and 7%.

    All values in million USD.  Farebox is NOT included (Rule 3: cost offset,
    not a benefit or cost in societal accounting).
    """
    results = {}

    for label, rate in [("3pct", BCR_DISCOUNT_RATE_LOW), ("7pct", BCR_DISCOUNT_RATE_HIGH)]:
        # Capital cost at year 0 (no discounting)
        results[f"capital_{label}"] = capital_cost_usd / 1e6

        # O&M with 3%/yr escalation
        base_om = (
            O_AND_M_FIXED_USD
            + O_AND_M_PER_KM_USD * length_km
            + O_AND_M_PER_STATION_USD * n_stations
        )
        escalation = np.power(1.0 + O_AND_M_ESCALATION_RATE, np.arange(1, years + 1))
        om_series = base_om * escalation
        results[f"om_{label}"] = _npv(om_series, rate) / 1e6

        # Net public service cost for new residents
        # Property tax from new homes partially offsets municipal cost
        avg_hh_size = 2.56
        property_tax_offset = (
            AVG_HOME_VALUE_NEW_DEVELOPMENT * PROPERTY_TAX_RATE / avg_hh_size
        )
        net_cost_per_person = max(
            0.0, PER_CAPITA_MUNICIPAL_SERVICE_COST - property_tax_offset
        )
        # Phase in: new pop accumulates over the 25 years
        # Use cumulative_new_pop as mature value, linearly ramped
        pop_series = np.linspace(0, cumulative_new_pop, years)
        public_service_series = pop_series * net_cost_per_person
        results[f"public_services_{label}"] = _npv(public_service_series, rate) / 1e6

        # Total costs
        results[f"total_costs_{label}"] = (
            results[f"capital_{label}"]
            + results[f"om_{label}"]
            + results[f"public_services_{label}"]
        )

    return results


# ───────────────────────────────────────────────────────────────────
# Step 6: BCR computation
# ───────────────────────────────────────────────────────────────────

def compute_bcr(
    benefits: Dict[str, float],
    costs: Dict[str, float],
) -> Dict[str, float]:
    """BCR = total benefits / total costs at each discount rate."""
    results = {}
    for label in ("3pct", "7pct"):
        b = benefits[f"total_benefits_{label}"]
        c = costs[f"total_costs_{label}"]
        results[f"bcr_{label}"] = b / c if c > 0 else 0.0
    return results


# ───────────────────────────────────────────────────────────────────
# Step 7: Monte Carlo BCR
# ───────────────────────────────────────────────────────────────────

def compute_bcr_monte_carlo(
    daily_diverted_car_trips: float,
    annual_vmt_avoided: float,
    annual_person_hours_saved: float,
    daily_riders: float,
    capital_cost_usd: float,
    length_km: float,
    n_stations: int,
    cumulative_new_pop: float,
    car_travel_time_min: float,
    apm_travel_time_min: float,
    *,
    n_draws: int = 500,
    years: int = DEBT_TERM_YEARS,
    rng: Optional[np.random.Generator] = None,
    ridership_multipliers: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Monte Carlo propagation of CBA uncertainty.

    Varies: VTTS, crash cost, SC-CO2, walk health value, ridership (±20%).
    Returns p10/p50/p90 for BCR at 3% and 7%.

    Parameters
    ----------
    ridership_multipliers : optional (n_draws,) array of pre-sampled ridership
        multipliers from the Gaussian copula (Component 3A).  When provided,
        these are used instead of independent triangular draws, ensuring
        methodological consistency with the financial Monte Carlo.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Use correlated ridership multipliers if provided
    if ridership_multipliers is not None:
        n_draws = len(ridership_multipliers)

    bcr_3 = np.empty(n_draws)
    bcr_7 = np.empty(n_draws)

    for i in range(n_draws):
        # Draw multipliers from triangular distributions
        vtts_mult = rng.triangular(*MC_VTTS_MULT)
        crash_mult = rng.triangular(*MC_CRASH_COST_MULT)
        co2_mult = rng.triangular(*MC_SC_CO2_MULT)
        health_val = rng.triangular(*MC_WALK_HEALTH_VALUE)
        if ridership_multipliers is not None:
            rider_mult = float(ridership_multipliers[i])
        else:
            rider_mult = rng.triangular(0.80, 1.00, 1.20)

        # Apply ridership multiplier
        dvt = daily_diverted_car_trips * rider_mult
        vmt = annual_vmt_avoided * rider_mult
        riders = daily_riders * rider_mult
        hours = annual_person_hours_saved * rider_mult

        for j, (label, rate) in enumerate([
            ("3pct", BCR_DISCOUNT_RATE_LOW),
            ("7pct", BCR_DISCOUNT_RATE_HIGH),
        ]):
            # Benefits
            vtts_base = (0.80 * VTTS_PERSONAL_PER_HOUR + 0.20 * VTTS_BUSINESS_PER_HOUR) * vtts_mult
            growth = np.power(1.0 + VTTS_REAL_GROWTH_RATE, np.arange(1, years + 1))
            tt_series = hours * vtts_base * growth
            b_tt = _npv(tt_series, rate)

            b_voc = _npv(np.full(years, vmt * VOC_MARGINAL_PER_MILE), rate)
            b_safety = _npv(np.full(years, vmt * CRASH_COST_PER_VMT * crash_mult), rate)

            co2_esc = np.power(1.0 + SC_CO2_REAL_ESCALATION, np.arange(1, years + 1))
            co2_base = vmt * CO2_TONS_PER_VMT * SOCIAL_COST_CO2_PER_TON * co2_mult
            crit_base = vmt * CRITERIA_POLLUTANT_PER_VMT
            b_emissions = _npv(co2_base * co2_esc + crit_base, rate)

            walk_min = riders * AVG_WALK_ACCESS_MIN_PER_TRIP * OPERATING_DAYS_PER_YEAR * NEW_WALKING_SHARE
            b_health = _npv(np.full(years, walk_min * health_val), rate)

            avoided_spaces = riders * CAR_DIVERSION_COMMUTE * PEAK_HOUR_PARKING_FACTOR * 0.3
            ann_park = avoided_spaces * STRUCTURED_PARKING_COST_PER_SPACE / PARKING_STRUCTURE_LIFE_YEARS
            b_parking = _npv(np.full(years, ann_park), rate)

            residual = capital_cost_usd * RESIDUAL_VALUE_SHARE / (1 + rate) ** years

            total_b = b_tt + b_voc + b_safety + b_emissions + b_health + b_parking + residual

            # Costs (not varied in MC — costs are known with less uncertainty)
            base_om = O_AND_M_FIXED_USD + O_AND_M_PER_KM_USD * length_km + O_AND_M_PER_STATION_USD * n_stations
            om_esc = np.power(1.0 + O_AND_M_ESCALATION_RATE, np.arange(1, years + 1))
            c_om = _npv(base_om * om_esc, rate)

            avg_hh = 2.56
            tax_offset = AVG_HOME_VALUE_NEW_DEVELOPMENT * PROPERTY_TAX_RATE / avg_hh
            net_svc = max(0.0, PER_CAPITA_MUNICIPAL_SERVICE_COST - tax_offset)
            pop_s = np.linspace(0, cumulative_new_pop, years)
            c_svc = _npv(pop_s * net_svc, rate)

            total_c = capital_cost_usd + c_om + c_svc

            bcr_val = total_b / total_c if total_c > 0 else 0.0

            if j == 0:
                bcr_3[i] = bcr_val
            else:
                bcr_7[i] = bcr_val

    return {
        "bcr_3pct_p10": float(np.percentile(bcr_3, 10)),
        "bcr_3pct_p50": float(np.percentile(bcr_3, 50)),
        "bcr_3pct_p90": float(np.percentile(bcr_3, 90)),
        "bcr_7pct_p10": float(np.percentile(bcr_7, 10)),
        "bcr_7pct_p50": float(np.percentile(bcr_7, 50)),
        "bcr_7pct_p90": float(np.percentile(bcr_7, 90)),
    }


# ───────────────────────────────────────────────────────────────────
# Step 8: Fiscal impact
# ───────────────────────────────────────────────────────────────────

def compute_fiscal_impact(
    cumulative_new_jobs: float,
    cumulative_new_pop: float,
    cumulative_new_res_sqft: float,
    cumulative_new_comm_sqft: float,
    capital_cost_usd: float,
    annual_vmt_avoided: float,
    daily_diverted_car_trips: float,
    tif_cumulative_usd: float,
    *,
    years: int = DEBT_TERM_YEARS,
    res_share: float = 0.5,
) -> Dict[str, float]:
    """Compute broader fiscal impact streams.

    Returns annual (year 10 steady-state) and 25-year cumulative values.
    All dollar values in USD (not millions).
    """
    # LIT from permanent jobs (steady state = mature jobs × wage × rate)
    lit_new_jobs = cumulative_new_jobs * AVG_WAGE_WEIGHTED * TIPPECANOE_LIT_RATE

    # LIT from construction phase (APM + TOD)
    private_inv = (
        cumulative_new_res_sqft * RESIDENTIAL_CONSTRUCTION_COST_PSF
        + cumulative_new_comm_sqft * COMMERCIAL_CONSTRUCTION_COST_PSF
    )
    total_construction = capital_cost_usd + private_inv
    construction_wages = total_construction * CONSTRUCTION_LABOR_SHARE
    # Spread APM construction over 4 years, TOD over 15 years
    apm_annual_wages = capital_cost_usd * CONSTRUCTION_LABOR_SHARE / 4.0
    tod_annual_wages = private_inv * CONSTRUCTION_LABOR_SHARE / 15.0
    lit_construction = (apm_annual_wages + tod_annual_wages) * TIPPECANOE_LIT_RATE

    # State income tax from permanent jobs
    state_income_tax = cumulative_new_jobs * AVG_WAGE_WEIGHTED * INDIANA_STATE_INCOME_TAX_RATE

    # Sales tax from new retail
    new_retail_sqft = cumulative_new_comm_sqft * RETAIL_SHARE_OF_COMMERCIAL
    annual_retail_sales = new_retail_sqft * RETAIL_SALES_PER_SQFT
    sales_tax = annual_retail_sales * INDIANA_SALES_TAX_RATE

    # Road maintenance savings
    road_savings = annual_vmt_avoided * ROAD_MAINTENANCE_COST_PER_VMT

    # Parking avoided (one-time capital savings)
    avoided_spaces = daily_diverted_car_trips * PEAK_HOUR_PARKING_FACTOR
    parking_avoided = avoided_spaces * STRUCTURED_PARKING_COST_PER_SPACE

    # Net public service cost
    avg_hh = 2.56
    tax_offset = AVG_HOME_VALUE_NEW_DEVELOPMENT * PROPERTY_TAX_RATE / avg_hh
    net_svc_per_person = max(0.0, PER_CAPITA_MUNICIPAL_SERVICE_COST - tax_offset)
    net_public_service_cost = cumulative_new_pop * net_svc_per_person

    # TIF with residential cap: use year-by-year tif_annual_revenue()
    # from src/finance.py which implements HEA 1120's 20-year residential cap,
    # SB 1 erosion, circuit breaker, and assessment lag correctly.
    from src.finance import tif_annual_revenue
    # Estimate increment value from cumulative TIF (invert the simple formula)
    # tif_cumulative ≈ increment_value × tax_rate × capture × Σ phasing(yr)
    # We use the year-by-year function with "redevelopment" area type which
    # correctly applies the 20-year residential cap.
    _avg_annual_tif = tif_cumulative_usd / max(years, 1)
    _est_increment = _avg_annual_tif / max(PROPERTY_TAX_RATE * TIF_CAPTURE_RATE_CONSERVATIVE, 1e-9)
    tif_truncated = sum(
        tif_annual_revenue(
            _est_increment, yr,
            capture_rate=TIF_CAPTURE_RATE_CONSERVATIVE,
            property_tax_rate=PROPERTY_TAX_RATE,
            res_share=res_share,
            area_type="redevelopment",
        )
        for yr in range(1, years + 1)
    )

    # Year 10 net fiscal benefit (steady state estimate)
    yr10_benefit = (
        lit_new_jobs
        + lit_construction  # construction still ongoing for TOD
        + road_savings
        - net_public_service_cost
    )

    # 25-year cumulative net fiscal
    cumulative_fiscal = (
        tif_truncated
        + lit_new_jobs * years
        + lit_construction * 15  # construction duration
        + state_income_tax * years
        + sales_tax * years
        + road_savings * years
        + parking_avoided  # one-time
        - net_public_service_cost * years
    )

    return {
        "tif_cumulative_25yr_musd": tif_cumulative_usd / 1e6,
        "tif_residential_truncated_musd": tif_truncated / 1e6,
        "lit_new_jobs_annual": lit_new_jobs,
        "lit_construction_annual": lit_construction,
        "state_income_tax_annual": state_income_tax,
        "sales_tax_retail_annual": sales_tax,
        "road_maintenance_savings_annual": road_savings,
        "parking_avoided_onetime_musd": parking_avoided / 1e6,
        "net_public_service_cost_annual": net_public_service_cost,
        "net_fiscal_benefit_annual_yr10": yr10_benefit,
        "net_fiscal_benefit_cumulative_25yr": cumulative_fiscal / 1e6,
    }


# ───────────────────────────────────────────────────────────────────
# Step 9: Economic activity (multipliers — NOT in BCR)
# ───────────────────────────────────────────────────────────────────

def compute_economic_activity(
    capital_cost_usd: float,
    cumulative_new_res_sqft: float,
    cumulative_new_comm_sqft: float,
    length_km: float,
) -> Dict[str, float]:
    """Compute gross economic activity metrics.

    These are labeled [ECONOMIC] and must NEVER be added to the BCR
    (Rule 4 — multiplier analysis is separate).
    """
    capital_m = capital_cost_usd / 1e6

    # APM construction jobs (with Type II multiplier)
    direct_apm_jobs = capital_m * CONSTRUCTION_JOBS_PER_MILLION
    total_apm_jobs = direct_apm_jobs * CONSTRUCTION_TYPE_II_MULTIPLIER

    # TOD construction jobs (NO second multiplier — Rule 7)
    private_inv = (
        cumulative_new_res_sqft * RESIDENTIAL_CONSTRUCTION_COST_PSF
        + cumulative_new_comm_sqft * COMMERCIAL_CONSTRUCTION_COST_PSF
    )
    private_inv_m = private_inv / 1e6
    tod_jobs = private_inv_m * CONSTRUCTION_JOBS_PER_MILLION  # direct only

    # Construction earnings
    total_construction_jobs = total_apm_jobs + tod_jobs
    construction_earnings = total_construction_jobs * AVG_CONSTRUCTION_WAGE / 1e6

    # Permanent operations jobs
    ops_jobs = length_km * APM_OPS_JOBS_PER_KM

    # Permanent TOD jobs
    tod_permanent_jobs = cumulative_new_comm_sqft / 200.0

    # Leverage ratios
    leverage_gross = private_inv_m / capital_m if capital_m > 0 else 0.0
    # Net of baseline: assume 2% background growth → ~5 years worth at baseline
    baseline_private = private_inv_m * 0.20  # rough 20% would have happened anyway
    leverage_net = (private_inv_m - baseline_private) / capital_m if capital_m > 0 else 0.0

    return {
        "construction_jobs_apm": total_apm_jobs,
        "construction_jobs_tod": tod_jobs,
        "construction_earnings_musd": construction_earnings,
        "permanent_ops_jobs": ops_jobs,
        "permanent_tod_jobs": tod_permanent_jobs,
        "private_investment_musd": private_inv_m,
        "leverage_ratio_gross": leverage_gross,
        "leverage_ratio_net": leverage_net,
    }


# ───────────────────────────────────────────────────────────────────
# Step 10: FTA Small Starts metrics
# ───────────────────────────────────────────────────────────────────

def compute_fta_metrics(
    annual_ridership: float,
    capital_cost_usd: float,
    length_km: float,
    n_stations: int,
) -> Dict[str, float]:
    """FTA Small Starts cost-effectiveness metrics.

    Uses FTA 30-year capital recovery factor at 5%.
    """
    # Capital Recovery Factor: r(1+r)^n / ((1+r)^n - 1)
    r = 0.05
    n = 30  # FTA standard
    crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
    annualized_capital = capital_cost_usd * crf

    annual_om = (
        O_AND_M_FIXED_USD
        + O_AND_M_PER_KM_USD * length_km
        + O_AND_M_PER_STATION_USD * n_stations
    )
    annualized_cost = annualized_capital + annual_om
    annualized_cost_musd = annualized_cost / 1e6

    cost_per_trip = annualized_cost / annual_ridership if annual_ridership > 0 else float("inf")
    trips_per_musd = annual_ridership / annualized_cost_musd if annualized_cost_musd > 0 else 0.0

    return {
        "fta_cost_per_trip": cost_per_trip,
        "fta_annual_trips_per_musd": trips_per_musd,
    }


# ───────────────────────────────────────────────────────────────────
# Step 11: Equity (transport savings for SE01)
# ───────────────────────────────────────────────────────────────────

def compute_equity_metrics(
    results_df: pd.DataFrame,
    avg_car_trip_miles: float,
) -> Dict[str, float]:
    """Compute equity metrics for low-income households.

    Returns annual transport savings for SE01 households who shift to APM.
    """
    mature = results_df[results_df["year"] >= 10]
    if mature.empty:
        mature = results_df

    riders_se01 = mature.get("riders_SE01", pd.Series([0.0])).mean()
    latent_se01 = mature.get("latent_SE01", pd.Series([0.0])).mean()

    # SE01 riders who shifted from car save full AAA cost per mile
    car_riders_se01 = riders_se01 * CAR_DIVERSION_COMMUTE
    annual_car_savings = (
        car_riders_se01
        * avg_car_trip_miles * 2.0  # round trip
        * VOC_FULL_PER_MILE  # Full ownership cost for equity analysis
        * OPERATING_DAYS_PER_YEAR
    )
    # Subtract fare cost
    annual_fare_cost = riders_se01 * FARE_PER_TRIP_USD * OPERATING_DAYS_PER_YEAR
    transport_savings = annual_car_savings - annual_fare_cost

    return {
        "transport_savings_se01_annual": max(0.0, transport_savings),
        "zero_car_hh_served": int(latent_se01 * OPERATING_DAYS_PER_YEAR),
    }


# ───────────────────────────────────────────────────────────────────
# Step 11b: Spatial Equity Mapping (Component 5, Addition 1)
# ───────────────────────────────────────────────────────────────────

def compute_spatial_equity(
    accessibility_with_apm: np.ndarray,
    accessibility_baseline: np.ndarray,
    parcel_pop: np.ndarray,
    parcel_income_segment: np.ndarray,
) -> Dict[str, float]:
    """Compute per-income-segment accessibility change.

    Parameters
    ----------
    accessibility_with_apm : (M,) accessibility scores with APM
    accessibility_baseline : (M,) accessibility scores without APM
    parcel_pop : (M,) population per parcel
    parcel_income_segment : (M,) 1=SE01, 2=SE02, 3=SE03

    Returns
    -------
    dict with mean_delta by segment and equity_ratio (SE01/SE03).
    """
    delta = accessibility_with_apm - accessibility_baseline
    results: Dict[str, float] = {}

    for seg, label in [(1, "SE01"), (2, "SE02"), (3, "SE03")]:
        mask = parcel_income_segment == seg
        pop = parcel_pop[mask]
        total_pop = pop.sum()
        if total_pop > 0:
            mean_delta = float((delta[mask] * pop).sum() / total_pop)
        else:
            mean_delta = 0.0
        results[f"mean_delta_{label}"] = mean_delta

    # Equity ratio: >1.0 means low-income benefits proportionally more (pro-equity).
    # Use signed division so both-positive and both-negative cases are handled:
    #   SE01>0, SE03>0 → ratio = improvement / improvement (normal case)
    #   SE01>0, SE03<0 → SE01 gains while SE03 loses → highly pro-equity (capped)
    #   SE01<0, SE03<0 → both lose → ratio < 1 if SE01 loses more (anti-equity)
    se03_delta = results.get("mean_delta_SE03", 0.0)
    se01_delta = results.get("mean_delta_SE01", 0.0)
    if abs(se03_delta) > 1e-6:
        raw_ratio = se01_delta / se03_delta
        # Cap extreme values (e.g., opposite signs produce very large ratios)
        results["equity_ratio"] = float(np.clip(raw_ratio, -10.0, 10.0))
    else:
        results["equity_ratio"] = 1.0 if se01_delta >= 0 else 0.0

    return results


# ───────────────────────────────────────────────────────────────────
# Step 11c: Displacement Risk Index (Component 5, Addition 2)
# ───────────────────────────────────────────────────────────────────

# HUD standard affordability threshold
RENT_BURDEN_THRESHOLD = 0.30
MEDIAN_INCOME_SE01 = 15_000.0  # LODES definition, lowest-earning third


def compute_displacement_risk(
    current_rents: np.ndarray,
    parcel_pop_se01: np.ndarray,
    prev_parcel_pop_se01: Optional[np.ndarray] = None,
    prev_rents: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Track rent burden and SE01 displacement.

    Parameters
    ----------
    current_rents : (M,) annual rent per parcel ($/yr)
    parcel_pop_se01 : (M,) SE01 population on each parcel (current year)
    prev_parcel_pop_se01 : (M,) SE01 population previous year (for displacement count)
    prev_rents : (M,) annual rent per parcel previous year.  If None, uses
        current_rents as proxy (conservative: overstates prev at-risk pop).

    Returns
    -------
    dict with at_risk_parcels, at_risk_se01_pop, displaced_se01 (if prev provided).
    """
    rent_burden = current_rents / MEDIAN_INCOME_SE01
    at_risk = rent_burden > RENT_BURDEN_THRESHOLD

    at_risk_se01 = float(parcel_pop_se01[at_risk].sum()) if at_risk.any() else 0.0

    result: Dict[str, float] = {
        "at_risk_parcels": int(at_risk.sum()),
        "at_risk_se01_pop": at_risk_se01,
        "mean_rent_burden_se01": float(
            (rent_burden * parcel_pop_se01).sum() / max(parcel_pop_se01.sum(), 1.0)
        ),
    }

    if prev_parcel_pop_se01 is not None:
        # Compute prev-year at-risk from prev-year rents (not current rents)
        if prev_rents is not None:
            prev_burden = prev_rents / MEDIAN_INCOME_SE01
            prev_at_risk = prev_burden > RENT_BURDEN_THRESHOLD
        else:
            # Fallback: use current at-risk mask (conservative estimate)
            prev_at_risk = at_risk
        prev_at_risk_pop = float(prev_parcel_pop_se01[prev_at_risk].sum()) if prev_at_risk.any() else 0.0
        curr_at_risk_pop = float(parcel_pop_se01[at_risk].sum()) if at_risk.any() else 0.0
        result["displaced_se01"] = max(0.0, prev_at_risk_pop - curr_at_risk_pop)

    return result


# ───────────────────────────────────────────────────────────────────
# Step 11d: CEJST / Justice40 Screening (Component 5, Addition 3)
# ───────────────────────────────────────────────────────────────────

def screen_ej_communities(
    station_coords: np.ndarray,
    tract_centroids: np.ndarray,
    tract_pop: np.ndarray,
    tract_is_disadvantaged: np.ndarray,
    catchment_m: float = 1200.0,
    ridership_benefit: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Screen corridor against CEJST disadvantaged community designations.

    Parameters
    ----------
    station_coords : (S, 2) station coordinates in meters
    tract_centroids : (T, 2) census tract centroids in meters
    tract_pop : (T,) population per tract
    tract_is_disadvantaged : (T,) boolean, True = CEJST disadvantaged
    catchment_m : walk catchment radius (meters)
    ridership_benefit : (T,) optional ridership benefit accruing to each tract

    Returns
    -------
    dict with dac_tract_count, dac_pop_share, dac_benefit_share,
    justice40_compliant.
    """
    from scipy.spatial import cKDTree

    if len(station_coords) == 0 or len(tract_centroids) == 0:
        return {
            "dac_tract_count": 0,
            "dac_pop_share": 0.0,
            "dac_benefit_share": 0.0,
            "justice40_compliant": False,
        }

    station_tree = cKDTree(station_coords)
    # Use query_ball_point to find all tracts within catchment radius.
    # query(k=1) only returns the nearest station distance, which misses
    # tracts equidistant from multiple stations and is semantically wrong
    # for a radius-based catchment check.
    neighbors = station_tree.query_ball_point(tract_centroids, r=catchment_m)
    in_catchment = np.array([len(n) > 0 for n in neighbors], dtype=bool)

    catchment_pop = tract_pop[in_catchment]
    catchment_dac = tract_is_disadvantaged[in_catchment]
    total_pop = catchment_pop.sum()

    dac_mask = in_catchment & tract_is_disadvantaged.astype(bool)
    dac_pop = tract_pop[dac_mask].sum()
    dac_pop_share = float(dac_pop / max(total_pop, 1.0))
    dac_tract_count = int(dac_mask.sum())

    # Benefit share: fraction of ridership benefit going to DAC tracts
    dac_benefit_share = 0.0
    if ridership_benefit is not None and in_catchment.any():
        total_benefit = ridership_benefit[in_catchment].sum()
        dac_benefit = ridership_benefit[dac_mask].sum() if dac_mask.any() else 0.0
        dac_benefit_share = float(dac_benefit / max(total_benefit, 1e-6))

    return {
        "dac_tract_count": dac_tract_count,
        "dac_pop_share": round(dac_pop_share, 4),
        "dac_benefit_share": round(dac_benefit_share, 4),
        "justice40_compliant": dac_benefit_share >= 0.40,
    }


# ───────────────────────────────────────────────────────────────────
# Step 11e: FTA Cost-Effectiveness Table (Component 6, Enhancement 2)
# ───────────────────────────────────────────────────────────────────

def build_fta_metrics_table(
    corridor_df: pd.DataFrame,
    tsm_annual_ridership: float = 0.0,
    tsm_annual_cost: float = 0.0,
) -> pd.DataFrame:
    """Build FTA-standard cost-effectiveness table for all corridors.

    Parameters
    ----------
    corridor_df : DataFrame with corridor_id, daily_ridership, capital_cost,
        length_km, n_stops columns.
    tsm_annual_ridership : annual ridership for TSM (bus-only) alternative.
    tsm_annual_cost : total annualized cost for TSM alternative.
    """
    rows = []
    for _, row in corridor_df.iterrows():
        cid = row.get("corridor_id", "")
        daily = float(row.get("daily_ridership", 0))
        annual_trips = daily * OPERATING_DAYS_PER_YEAR
        length = float(row.get("length_km", 0))
        n_stn = int(row.get("n_stops", row.get("stations", 0)))
        capex = float(row.get("capital_cost", compute_capital_cost(length, n_stn)))

        # FTA cost-effectiveness uses 30-year useful life for fixed guideway
        # (FTA Capital Investment Grants guidance, Section 5309).
        # This differs from the project finance model which uses
        # BOND_RATE=0.05, DEBT_TERM_YEARS=25 from financial_params.py.
        # The longer amortization period is intentional: FTA evaluates
        # infrastructure over its engineering useful life, not the debt term.
        r, n = 0.05, 30  # FTA standard, NOT project finance parameters
        crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
        annualized_cap = capex * crf
        # Use mid-horizon (year 15) escalated O&M as representative annual cost.
        # Base-year O&M understates by ~34% at year 10 (3% compounded).
        # FTA guidance uses average annual cost over the analysis period.
        base_om = O_AND_M_FIXED_USD + O_AND_M_PER_KM_USD * length + O_AND_M_PER_STATION_USD * n_stn
        _fta_horizon = 30
        _om_escalated = base_om * np.power(1.0 + O_AND_M_ESCALATION_RATE, np.arange(1, _fta_horizon + 1))
        annual_om = float(np.mean(_om_escalated))
        total_annual = annualized_cap + annual_om

        cost_per_trip = total_annual / annual_trips if annual_trips > 0 else float("inf")
        incr_trips = annual_trips - tsm_annual_ridership
        incr_cost = total_annual - tsm_annual_cost
        incr_cpt = incr_cost / incr_trips if incr_trips > 0 else float("inf")

        # FTA rating
        if incr_cpt < 4.00:
            rating = "High"
        elif incr_cpt < 8.00:
            rating = "Medium-High"
        elif incr_cpt < 12.00:
            rating = "Medium"
        else:
            rating = "Low"

        rows.append({
            "corridor_id": cid,
            "annual_operating_cost_musd": round(annual_om / 1e6, 2),
            "annualized_capital_musd": round(annualized_cap / 1e6, 2),
            "total_annualized_cost_musd": round(total_annual / 1e6, 2),
            "annual_trips": int(annual_trips),
            "incremental_trips_vs_tsm": int(incr_trips),
            "cost_per_trip": round(cost_per_trip, 2),
            "incremental_cost_per_trip": round(incr_cpt, 2) if incr_cpt < 1e9 else None,
            "fta_rating": rating,
        })

    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────────────
# Step 12: Top-level orchestrator
# ───────────────────────────────────────────────────────────────────

def compute_corridor_cba(
    results_df: pd.DataFrame,
    corridor_id: str,
    scenario: str,
    length_km: float,
    n_stations: int,
    *,
    tif_cumulative_usd: float = 0.0,
    run_monte_carlo: bool = True,
    n_mc_draws: int = 500,
) -> CorridorCBAResult:
    """Compute full CBA for one corridor under one scenario.

    Args:
        results_df: Feedback loop results for this corridor (all years).
        corridor_id: e.g. "C1".
        scenario: e.g. "current_zoning".
        length_km: Corridor length in km.
        n_stations: Number of stops.
        tif_cumulative_usd: 25-year cumulative TIF revenue (USD).
        run_monte_carlo: Whether to run MC uncertainty analysis.
        n_mc_draws: Number of MC draws.

    Returns:
        CorridorCBAResult with all fields populated.
    """
    result = CorridorCBAResult(corridor_id=corridor_id, scenario=scenario)

    # --- Corridor parameters ---
    capital_cost_usd = compute_capital_cost(length_km, n_stations)
    car_circuity = 1.20  # from memory: CAR_CIRCUITY
    avg_car_trip_miles = length_km * car_circuity * 0.621371 * 0.5  # half corridor

    # APM travel time: walk (5 min) + wait (4 min) + IVT + walk (5 min)
    apm_speed_kph = 40.0
    apm_ivt_min = (length_km * 0.5) / apm_speed_kph * 60  # half corridor avg
    apm_travel_time_min = 5.0 + 4.0 + apm_ivt_min + 5.0

    # Car travel time
    car_speed_kph = 35.0  # urban arterial with congestion
    car_travel_time_min = (avg_car_trip_miles / 0.621371) / car_speed_kph * 60

    # --- Ridership summaries ---
    mature = results_df[results_df["year"] >= 10]
    if mature.empty:
        mature = results_df
    daily_riders = mature["daily_riders"].mean()
    annual_ridership = daily_riders * OPERATING_DAYS_PER_YEAR

    # Development summaries (cumulative over 25 years)
    cumulative_new_pop = results_df["new_pop"].sum()
    cumulative_new_jobs = results_df["new_jobs"].sum()
    cumulative_new_res_sqft = results_df["new_res_sqft"].sum()
    cumulative_new_comm_sqft = results_df["new_comm_sqft"].sum()
    res_share = (
        cumulative_new_res_sqft / (cumulative_new_res_sqft + cumulative_new_comm_sqft)
        if (cumulative_new_res_sqft + cumulative_new_comm_sqft) > 0
        else 0.5
    )

    # --- Step 1: Diverted car trips ---
    daily_diverted = compute_diverted_car_trips(results_df)
    result.daily_diverted_car_trips = daily_diverted

    # --- Step 2: VMT avoided ---
    annual_vmt = compute_annual_vmt_avoided(daily_diverted, avg_car_trip_miles)
    result.annual_vmt_avoided_miles = annual_vmt

    # --- Step 3: Travel time savings ---
    annual_hours, tt_series = compute_travel_time_savings(
        daily_diverted, car_travel_time_min, apm_travel_time_min,
    )
    result.annual_person_hours_saved = annual_hours

    # --- Step 4: BCR benefits ---
    benefits = compute_bcr_benefits(
        annual_vmt, tt_series, daily_riders, capital_cost_usd,
    )
    for key in [
        "travel_time", "voc", "safety", "emissions", "health",
        "parking", "agglomeration",
    ]:
        for label in ("3pct", "7pct"):
            setattr(result, f"benefit_{key}_{label}", benefits[f"{key}_{label}"])
    # Residual value uses a different field name prefix
    result.residual_value_3pct = benefits["residual_3pct"]
    result.residual_value_7pct = benefits["residual_7pct"]
    result.total_benefits_3pct = benefits["total_benefits_3pct"]
    result.total_benefits_7pct = benefits["total_benefits_7pct"]

    # --- Step 5: BCR costs ---
    costs = compute_bcr_costs(
        capital_cost_usd, length_km, n_stations, cumulative_new_pop,
    )
    for key in ["capital", "om", "public_services"]:
        for label in ("3pct", "7pct"):
            setattr(result, f"cost_{key}_{label}", costs[f"{key}_{label}"])
    result.total_costs_3pct = costs["total_costs_3pct"]
    result.total_costs_7pct = costs["total_costs_7pct"]

    # --- Step 6: BCR ---
    bcr = compute_bcr(benefits, costs)
    result.bcr_3pct = bcr["bcr_3pct"]
    result.bcr_7pct = bcr["bcr_7pct"]

    # --- Step 7: Monte Carlo ---
    if run_monte_carlo:
        mc = compute_bcr_monte_carlo(
            daily_diverted, annual_vmt, annual_hours, daily_riders,
            capital_cost_usd, length_km, n_stations, cumulative_new_pop,
            car_travel_time_min, apm_travel_time_min,
            n_draws=n_mc_draws,
        )
        result.bcr_3pct_p10 = mc["bcr_3pct_p10"]
        result.bcr_3pct_p50 = mc["bcr_3pct_p50"]
        result.bcr_3pct_p90 = mc["bcr_3pct_p90"]
        result.bcr_7pct_p10 = mc["bcr_7pct_p10"]
        result.bcr_7pct_p50 = mc["bcr_7pct_p50"]
        result.bcr_7pct_p90 = mc["bcr_7pct_p90"]

    # --- Step 8: Fiscal impact ---
    fiscal = compute_fiscal_impact(
        cumulative_new_jobs, cumulative_new_pop,
        cumulative_new_res_sqft, cumulative_new_comm_sqft,
        capital_cost_usd, annual_vmt, daily_diverted,
        tif_cumulative_usd, res_share=res_share,
    )
    for key, val in fiscal.items():
        setattr(result, key, val)

    # --- Step 9: Economic activity ---
    econ = compute_economic_activity(
        capital_cost_usd, cumulative_new_res_sqft,
        cumulative_new_comm_sqft, length_km,
    )
    for key, val in econ.items():
        setattr(result, key, val)

    # --- Step 10: FTA metrics ---
    fta = compute_fta_metrics(
        annual_ridership, capital_cost_usd, length_km, n_stations,
    )
    result.fta_cost_per_trip = fta["fta_cost_per_trip"]
    result.fta_annual_trips_per_musd = fta["fta_annual_trips_per_musd"]

    # --- Step 11: Equity ---
    equity = compute_equity_metrics(results_df, avg_car_trip_miles)
    result.transport_savings_se01_annual = equity["transport_savings_se01_annual"]
    result.zero_car_hh_served = equity["zero_car_hh_served"]

    return result


# ───────────────────────────────────────────────────────────────────
# Step 13: Taxpayer summary format
# ───────────────────────────────────────────────────────────────────

def format_taxpayer_summary(r: CorridorCBAResult) -> str:
    """One-page formatted summary for public consumption."""
    daily_riders = r.annual_vmt_avoided_miles / (
        r.daily_diverted_car_trips * 2.0 * OPERATING_DAYS_PER_YEAR
    ) if r.daily_diverted_car_trips > 0 else 0
    # Re-derive daily riders from diverted (approximate)
    # Better: pass daily_riders explicitly, but we don't store it on result
    cars_removed = r.daily_diverted_car_trips
    annual_vmt_m = r.annual_vmt_avoided_miles / 1e6
    co2_tons = r.annual_vmt_avoided_miles * CO2_TONS_PER_VMT

    total_jobs = (
        r.construction_jobs_apm + r.construction_jobs_tod
        + r.permanent_ops_jobs + r.permanent_tod_jobs
    )

    lines = [
        "=" * 55,
        f"  CORRIDOR {r.corridor_id}: Economic Impact Summary",
        "=" * 55,
        "",
        "  FOR EVERY $1 OF PUBLIC INVESTMENT:",
        f"    ${r.bcr_3pct:.2f} in community benefits (BCR at 3%)",
        f"    ${r.leverage_ratio_gross:.2f} in private development activity",
        "",
        "  JOBS:",
        f"    {r.construction_jobs_apm + r.construction_jobs_tod:,.0f} construction jobs",
        f"    {r.permanent_ops_jobs:,.0f} permanent operations jobs",
        f"    {r.permanent_tod_jobs:,.0f} permanent jobs from new development",
        "",
        "  TAX REVENUE:",
        f"    ${r.lit_new_jobs_annual / 1e6:.2f}M in annual income tax from new jobs",
        f"    ${r.tif_residential_truncated_musd:.1f}M cumulative TIF revenue (25 years)",
        "",
        "  TRANSPORTATION:",
        f"    {cars_removed:,.0f} equivalent cars removed from road daily",
        f"    {annual_vmt_m:.1f} million fewer vehicle miles per year",
        f"    {co2_tons:,.0f} tons CO2 reduced annually",
        "",
        "  EQUITY:",
        f"    ${r.transport_savings_se01_annual:,.0f} annual transport savings (low-income)",
        f"    {r.zero_car_hh_served:,} zero-car household trips enabled per year",
        "",
        "  UNCERTAINTY RANGE (p10-p90):",
        f"    BCR at 3%: {r.bcr_3pct_p10:.2f} - {r.bcr_3pct_p90:.2f}",
        f"    BCR at 7%: {r.bcr_7pct_p10:.2f} - {r.bcr_7pct_p90:.2f}",
        "",
        "=" * 55,
    ]
    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────
# Agglomeration stub (Phase 5)
# ───────────────────────────────────────────────────────────────────

def compute_agglomeration_benefits() -> Dict[str, float]:
    """Phase 5 stub: agglomeration benefits (Graham 2007 framework).

    Requires zone-to-zone generalized cost matrix (before/after APM),
    sector-specific employment, and GDP per worker by sector.
    See docs/ECONOMIC_FISCAL_IMPACT_PLAN.md Section 6 for methodology.
    """
    return {
        "agglomeration_3pct": 0.0,
        "agglomeration_7pct": 0.0,
    }

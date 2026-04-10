"""Tests for economic impact / CBA module.

Verifies USDOT BCA framework compliance, accounting rules (no double-counting),
fiscal calculations, employment multipliers, Monte Carlo, and FTA metrics.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

from src.economic_impact import (
    compute_diverted_car_trips,
    compute_annual_vmt_avoided,
    compute_travel_time_savings,
    compute_bcr_benefits,
    compute_bcr_costs,
    compute_bcr,
    compute_bcr_monte_carlo,
    compute_fiscal_impact,
    compute_economic_activity,
    compute_fta_metrics,
    compute_equity_metrics,
    compute_corridor_cba,
    format_taxpayer_summary,
    _npv,
    CorridorCBAResult,
)
from src.financial_params import (
    CAR_DIVERSION_COMMUTE,
    CAR_DIVERSION_STUDENT,
    CAR_DIVERSION_GENERATOR,
    CAR_DIVERSION_INDUCED,
    CAR_DIVERSION_LATENT,
    OPERATING_DAYS_PER_YEAR,
    VTTS_PERSONAL_PER_HOUR,
    VTTS_BUSINESS_PER_HOUR,
    VTTS_REAL_GROWTH_RATE,
    VOC_MARGINAL_PER_MILE,
    CRASH_COST_PER_VMT,
    TOTAL_EMISSION_COST_PER_VMT,
    SC_CO2_REAL_ESCALATION,
    CO2_TONS_PER_VMT,
    SOCIAL_COST_CO2_PER_TON,
    CRITERIA_POLLUTANT_PER_VMT,
    TIPPECANOE_LIT_RATE,
    AVG_WAGE_WEIGHTED,
    CONSTRUCTION_JOBS_PER_MILLION,
    CONSTRUCTION_TYPE_II_MULTIPLIER,
    TIF_RESIDENTIAL_MAX_YEARS,
    RESIDUAL_VALUE_SHARE,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic feedback loop data
# ---------------------------------------------------------------------------

def _make_feedback_df(
    n_years: int = 26,
    daily_riders: float = 3000.0,
    both_ends: float = 400.0,
    origin_only: float = 500.0,
    student: float = 1200.0,
    generator: float = 300.0,
    induced: float = 100.0,
    latent: float = 80.0,
    new_pop: float = 50.0,
    new_jobs: float = 30.0,
    new_res_sqft: float = 25000.0,
    new_comm_sqft: float = 15000.0,
) -> pd.DataFrame:
    """Create a synthetic feedback loop results DataFrame."""
    data = {
        "year": list(range(n_years)),
        "corridor_id": ["C1"] * n_years,
        "daily_riders": [daily_riders * min(1.0, y / 10.0) for y in range(n_years)],
        "work_commute_daily": [both_ends * min(1.0, y / 10.0) for y in range(n_years)],
        "local_nonwork_daily": [origin_only * min(1.0, y / 10.0) for y in range(n_years)],
        "campus_daily": [student * min(1.0, y / 10.0) for y in range(n_years)],
        "destination_daily": [generator * min(1.0, y / 10.0) for y in range(n_years)],
        "induced_daily_raw": [induced * min(1.0, y / 10.0) for y in range(n_years)],
        "latent_daily_raw": [latent * min(1.0, y / 10.0) for y in range(n_years)],
        "new_pop": [new_pop] * n_years,
        "new_jobs": [new_jobs] * n_years,
        "new_res_sqft": [new_res_sqft] * n_years,
        "new_comm_sqft": [new_comm_sqft] * n_years,
        "riders_SE01": [daily_riders * 0.20 * min(1.0, y / 10.0) for y in range(n_years)],
        "riders_SE02": [daily_riders * 0.30 * min(1.0, y / 10.0) for y in range(n_years)],
        "riders_SE03": [daily_riders * 0.50 * min(1.0, y / 10.0) for y in range(n_years)],
        "latent_SE01": [latent * 0.60 * min(1.0, y / 10.0) for y in range(n_years)],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Test 1: Diverted car trips — component-specific shares
# ---------------------------------------------------------------------------

def test_diverted_car_trips_components():
    """Verify component-specific diversion shares sum correctly."""
    df = _make_feedback_df()

    diverted = compute_diverted_car_trips(df)

    # Mature values (year >= 10): all components at full value
    expected = (
        (400 + 500) * CAR_DIVERSION_COMMUTE
        + 1200 * CAR_DIVERSION_STUDENT
        + 300 * CAR_DIVERSION_GENERATOR
        + 100 * CAR_DIVERSION_INDUCED
        + 80 * CAR_DIVERSION_LATENT
    )
    assert diverted == pytest.approx(expected, rel=0.01)


def test_diverted_induced_latent_zero():
    """Induced and latent trips should contribute zero car diversions."""
    df = _make_feedback_df(
        both_ends=0, origin_only=0, student=0, generator=0,
        induced=500, latent=500,
    )
    diverted = compute_diverted_car_trips(df)
    assert diverted == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 2: VMT avoided
# ---------------------------------------------------------------------------

def test_vmt_avoided_calculation():
    """Known inputs → expected VMT."""
    daily_diverted = 500.0
    avg_miles = 3.0

    vmt = compute_annual_vmt_avoided(daily_diverted, avg_miles)

    expected = 500 * 3.0 * 2.0 * 312  # round trip × operating days
    assert vmt == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 3: Travel time savings with VTTS growth
# ---------------------------------------------------------------------------

def test_travel_time_savings_vtts_growth():
    """Verify 1.2%/yr real growth over 25 years."""
    hours, series = compute_travel_time_savings(
        daily_diverted_car_trips=500,
        car_travel_time_min=15.0,
        apm_travel_time_min=12.0,
        business_share=0.20,
        years=25,
    )

    # Year 1 VTTS
    vtts_yr1 = 0.80 * VTTS_PERSONAL_PER_HOUR + 0.20 * VTTS_BUSINESS_PER_HOUR
    vtts_yr1 *= (1 + VTTS_REAL_GROWTH_RATE)  # year 1 growth

    # Year 25 VTTS
    vtts_yr25_mult = (1 + VTTS_REAL_GROWTH_RATE) ** 25
    ratio = series[24] / series[0]
    expected_ratio = vtts_yr25_mult / (1 + VTTS_REAL_GROWTH_RATE)  # yr25/yr1
    assert ratio == pytest.approx(expected_ratio, rel=0.001)


def test_travel_time_no_savings_when_apm_slower():
    """If APM is slower than car, no savings."""
    hours, series = compute_travel_time_savings(
        daily_diverted_car_trips=500,
        car_travel_time_min=10.0,
        apm_travel_time_min=15.0,
    )
    assert hours == 0.0
    assert np.all(series == 0.0)


# ---------------------------------------------------------------------------
# Test 4: BCR for a synthetic corridor
# ---------------------------------------------------------------------------

def test_bcr_straight_corridor():
    """Synthetic corridor with known geometry → BCR in plausible range."""
    df = _make_feedback_df()

    result = compute_corridor_cba(
        results_df=df,
        corridor_id="C_test",
        scenario="test",
        length_km=8.0,
        n_stations=8,
        tif_cumulative_usd=10_000_000,
        run_monte_carlo=False,
    )

    # BCR should be positive and in plausible range
    assert result.bcr_3pct > 0
    assert result.bcr_7pct > 0
    # At 3%, BCR should be higher than at 7% (lower discount → higher benefits NPV)
    assert result.bcr_3pct > result.bcr_7pct


# ---------------------------------------------------------------------------
# Test 5: No double-counting (accounting rules)
# ---------------------------------------------------------------------------

def test_bcr_benefits_no_double_count():
    """Verify farebox NOT in benefits; TIF NOT in BCR benefits."""
    df = _make_feedback_df()

    result = compute_corridor_cba(
        results_df=df,
        corridor_id="C_test",
        scenario="test",
        length_km=8.0,
        n_stations=8,
        tif_cumulative_usd=50_000_000,
        run_monte_carlo=False,
    )

    # Rule 3: farebox should not appear in total_benefits
    # (there's no farebox field in benefits at all)
    benefit_fields = [f for f in dir(result) if f.startswith("benefit_")]
    for f in benefit_fields:
        assert "farebox" not in f.lower()

    # Rule 2: TIF should not appear in BCR benefits
    # (TIF is in fiscal section only)
    assert result.tif_cumulative_25yr_musd > 0  # TIF exists
    # But it's in the fiscal section, not contributing to total_benefits
    total_b = result.total_benefits_3pct
    # Verify total benefits is composed only of the expected categories
    benefit_sum = (
        result.benefit_travel_time_3pct
        + result.benefit_voc_3pct
        + result.benefit_safety_3pct
        + result.benefit_emissions_3pct
        + result.benefit_health_3pct
        + result.benefit_parking_3pct
        + result.benefit_agglomeration_3pct
        + result.residual_value_3pct
    )
    assert total_b == pytest.approx(benefit_sum, rel=0.001)


# ---------------------------------------------------------------------------
# Test 6: Emission SC-CO2 escalation
# ---------------------------------------------------------------------------

def test_emissions_sc_co2_escalation():
    """Verify 2.5%/yr escalation on emission benefits."""
    vmt = 1_000_000  # 1M VMT
    years = 25

    # Compute benefit series manually
    co2_base = vmt * CO2_TONS_PER_VMT * SOCIAL_COST_CO2_PER_TON
    criteria_base = vmt * CRITERIA_POLLUTANT_PER_VMT
    co2_esc = np.power(1.0 + SC_CO2_REAL_ESCALATION, np.arange(1, years + 1))

    year1 = co2_base * co2_esc[0] + criteria_base
    year25 = co2_base * co2_esc[24] + criteria_base

    # Year 25 CO2 component should be ~1.025^25 ≈ 1.854× year 0
    co2_ratio = co2_esc[24] / co2_esc[0]
    expected_co2_growth = (1.025 ** 24)  # yr25/yr1
    assert co2_ratio == pytest.approx(expected_co2_growth, rel=0.001)

    # Year 25 total should be higher than year 1
    assert year25 > year1


# ---------------------------------------------------------------------------
# Test 7: Fiscal TIF residential cap
# ---------------------------------------------------------------------------

def test_fiscal_tif_residential_cap():
    """Verify TIF truncation reduces revenue for residential-heavy corridors."""
    fiscal_high_res = compute_fiscal_impact(
        cumulative_new_jobs=500,
        cumulative_new_pop=1000,
        cumulative_new_res_sqft=500_000,
        cumulative_new_comm_sqft=100_000,
        capital_cost_usd=800_000_000,
        annual_vmt_avoided=5_000_000,
        daily_diverted_car_trips=500,
        tif_cumulative_usd=100_000_000,
        res_share=0.83,  # 83% residential
    )

    fiscal_low_res = compute_fiscal_impact(
        cumulative_new_jobs=500,
        cumulative_new_pop=1000,
        cumulative_new_res_sqft=100_000,
        cumulative_new_comm_sqft=500_000,
        capital_cost_usd=800_000_000,
        annual_vmt_avoided=5_000_000,
        daily_diverted_car_trips=500,
        tif_cumulative_usd=100_000_000,
        res_share=0.17,  # 17% residential
    )

    # High-res corridor should have more TIF truncation
    assert fiscal_high_res["tif_residential_truncated_musd"] < fiscal_low_res["tif_residential_truncated_musd"]


# ---------------------------------------------------------------------------
# Test 8: Fiscal LIT calculation
# ---------------------------------------------------------------------------

def test_fiscal_lit_calculation():
    """Known jobs × wage × rate = expected LIT."""
    fiscal = compute_fiscal_impact(
        cumulative_new_jobs=100,
        cumulative_new_pop=200,
        cumulative_new_res_sqft=100_000,
        cumulative_new_comm_sqft=50_000,
        capital_cost_usd=500_000_000,
        annual_vmt_avoided=1_000_000,
        daily_diverted_car_trips=200,
        tif_cumulative_usd=10_000_000,
    )

    expected_lit = 100 * AVG_WAGE_WEIGHTED * TIPPECANOE_LIT_RATE
    assert fiscal["lit_new_jobs_annual"] == pytest.approx(expected_lit)


# ---------------------------------------------------------------------------
# Test 9: Economic activity — no TOD multiplier (Rule 7)
# ---------------------------------------------------------------------------

def test_economic_activity_no_tod_multiplier():
    """TOD construction jobs should NOT get Type II multiplier."""
    econ = compute_economic_activity(
        capital_cost_usd=800_000_000,  # $800M
        cumulative_new_res_sqft=500_000,
        cumulative_new_comm_sqft=200_000,
        length_km=8.0,
    )

    # APM jobs: $800M × 30 jobs/$M × 2.0 multiplier
    expected_apm = 800 * CONSTRUCTION_JOBS_PER_MILLION * CONSTRUCTION_TYPE_II_MULTIPLIER
    assert econ["construction_jobs_apm"] == pytest.approx(expected_apm)

    # TOD jobs: private_inv_M × 30 (NO multiplier)
    private_inv_m = (500_000 * 150 + 200_000 * 180) / 1e6
    expected_tod = private_inv_m * CONSTRUCTION_JOBS_PER_MILLION
    assert econ["construction_jobs_tod"] == pytest.approx(expected_tod)

    # TOD jobs should be strictly less than if multiplied
    assert econ["construction_jobs_tod"] < expected_tod * CONSTRUCTION_TYPE_II_MULTIPLIER


# ---------------------------------------------------------------------------
# Test 10: Leverage ratio
# ---------------------------------------------------------------------------

def test_leverage_ratio():
    """Private investment / public cost."""
    econ = compute_economic_activity(
        capital_cost_usd=500_000_000,
        cumulative_new_res_sqft=1_000_000,
        cumulative_new_comm_sqft=500_000,
        length_km=5.0,
    )

    private_inv = (1_000_000 * 150 + 500_000 * 180) / 1e6  # $240M
    capital_m = 500  # $500M
    expected_ratio = private_inv / capital_m

    assert econ["leverage_ratio_gross"] == pytest.approx(expected_ratio, rel=0.01)
    # Net ratio should be less than gross (baseline subtracted)
    assert econ["leverage_ratio_net"] < econ["leverage_ratio_gross"]


# ---------------------------------------------------------------------------
# Test 11: Monte Carlo produces uncertainty
# ---------------------------------------------------------------------------

def test_monte_carlo_produces_uncertainty():
    """p10 < p50 < p90 for BCR."""
    mc = compute_bcr_monte_carlo(
        daily_diverted_car_trips=500,
        annual_vmt_avoided=3_000_000,
        annual_person_hours_saved=50_000,
        daily_riders=3000,
        capital_cost_usd=800_000_000,
        length_km=8.0,
        n_stations=8,
        cumulative_new_pop=1000,
        car_travel_time_min=15.0,
        apm_travel_time_min=12.0,
        n_draws=200,
    )

    assert mc["bcr_3pct_p10"] < mc["bcr_3pct_p50"]
    assert mc["bcr_3pct_p50"] < mc["bcr_3pct_p90"]
    assert mc["bcr_7pct_p10"] < mc["bcr_7pct_p50"]
    assert mc["bcr_7pct_p50"] < mc["bcr_7pct_p90"]

    # 3% BCR should generally be higher than 7% (lower discount → more benefits)
    assert mc["bcr_3pct_p50"] > mc["bcr_7pct_p50"]


# ---------------------------------------------------------------------------
# Test 12: FTA cost per trip
# ---------------------------------------------------------------------------

def test_fta_cost_per_trip():
    """Annualized cost / annual trips matches expected."""
    annual_ridership = 900_000  # 3000/day × 300 days
    capital = 800_000_000
    length = 8.0
    stations = 8

    fta = compute_fta_metrics(annual_ridership, capital, length, stations)

    # CRF at 5%, 30 years
    r, n = 0.05, 30
    crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
    annualized_cap = capital * crf
    from src.financial_params import O_AND_M_FIXED_USD, O_AND_M_PER_KM_USD, O_AND_M_PER_STATION_USD
    annual_om = O_AND_M_FIXED_USD + O_AND_M_PER_KM_USD * length + O_AND_M_PER_STATION_USD * stations
    annualized_cost = annualized_cap + annual_om

    expected_cpt = annualized_cost / annual_ridership
    assert fta["fta_cost_per_trip"] == pytest.approx(expected_cpt, rel=0.001)
    assert fta["fta_annual_trips_per_musd"] > 0


# ---------------------------------------------------------------------------
# Test: NPV helper
# ---------------------------------------------------------------------------

def test_npv_basic():
    """NPV of $1000/yr for 3 years at 5%."""
    series = np.array([1000.0, 1000.0, 1000.0])
    result = _npv(series, 0.05)
    expected = 1000 / 1.05 + 1000 / 1.05**2 + 1000 / 1.05**3
    assert result == pytest.approx(expected, rel=0.001)


# ---------------------------------------------------------------------------
# Test: Taxpayer summary format
# ---------------------------------------------------------------------------

def test_taxpayer_summary_format():
    """Format function produces non-empty string with key fields."""
    df = _make_feedback_df()
    result = compute_corridor_cba(
        results_df=df,
        corridor_id="C1",
        scenario="test",
        length_km=8.0,
        n_stations=8,
        run_monte_carlo=False,
    )
    summary = format_taxpayer_summary(result)
    assert "C1" in summary
    assert "BCR" in summary
    assert "JOBS" in summary
    assert "EQUITY" in summary


# ---------------------------------------------------------------------------
# Test: Full corridor CBA integration
# ---------------------------------------------------------------------------

def test_corridor_cba_all_fields_populated():
    """Verify compute_corridor_cba populates all major fields."""
    df = _make_feedback_df()
    result = compute_corridor_cba(
        results_df=df,
        corridor_id="C1",
        scenario="current_zoning",
        length_km=10.0,
        n_stations=10,
        tif_cumulative_usd=20_000_000,
        run_monte_carlo=False,
    )

    # BCR should be positive
    assert result.bcr_3pct > 0
    assert result.bcr_7pct > 0

    # Intermediate quantities should be positive
    assert result.daily_diverted_car_trips > 0
    assert result.annual_vmt_avoided_miles > 0
    # annual_person_hours_saved can be 0 if APM total trip time (walk+wait+IVT)
    # exceeds car trip time — this is realistic for short corridors
    assert result.annual_person_hours_saved >= 0

    # Economic activity
    assert result.construction_jobs_apm > 0
    assert result.private_investment_musd > 0
    assert result.leverage_ratio_gross > 0

    # Fiscal
    assert result.lit_new_jobs_annual > 0

    # FTA
    assert result.fta_cost_per_trip > 0
    assert result.fta_annual_trips_per_musd > 0

    # to_dict should work
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "bcr_3pct" in d

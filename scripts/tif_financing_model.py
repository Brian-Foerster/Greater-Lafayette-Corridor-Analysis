"""
TIF (Tax Increment Financing) Modeling: Infrastructure Investment from Tax Growth
=================================================================================

Models how property tax increments from APM corridor can fund infrastructure and
operations over 25 years. Compares TIF revenue under zoning vs. no-zoning scenarios.

Key concepts:
- Base year: Year 0 property tax revenue (baseline)
- Increment: Difference between actual and baseline (no-APM) growth
- Capture: Percentage of increment dedicated to public infrastructure
- Uses: APM operations subsidy, station improvements, area improvements

Outputs:
- tif_annual_revenue.csv (year-by-year TIF revenue available)
- tif_allocation_scenarios.csv (different infrastructure spending allocations)
- tif_breakeven_analysis.csv (when does TIF revenue equal APM costs?)
- property_value_summary_tif.csv (total assessed value by year)

No subsidies assumption:
- APM operations must be covered by fares or TIF revenue
- All TIF revenue goes to public benefit (infrastructure, operations)
- Property owners benefit from value increases not captured (15%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)

# TIF Configuration
TIF_CONFIG = {
    "property_tax_rate": 0.01,  # 1% annual property tax
    "capture_rate": 0.85,  # 85% of increment to TIF district, 15% to property owners
    "baseline_growth_rate": 0.02,  # 2% annual baseline (no APM)
    "apm_opening_year": 1,
    "time_horizon": 25,
}

# Development phasing configuration
# Models realistic timing of property value increases following transit investment
# Based on transit-oriented development research (TCRP Report 128)
DEVELOPMENT_PHASING = {
    "planning_phase": {
        "years": (0, 2),       # Years 0-2: Planning and approvals
        "value_capture": 0.05, # Only 5% of ultimate value increase captured
        "description": "Zoning changes, planning approvals, initial market response"
    },
    "early_development": {
        "years": (2, 5),       # Years 2-5: First developments complete
        "value_capture": 0.33, # 33% of ultimate value increase realized
        "description": "First wave of TOD projects, property speculation"
    },
    "buildout": {
        "years": (5, 15),      # Years 5-15: Main buildout period
        "value_capture": 0.85, # 85% of ultimate value increase realized
        "description": "Major development activity, market maturation"
    },
    "maturation": {
        "years": (15, 25),     # Years 15-25: Full maturation
        "value_capture": 1.00, # 100% of ultimate value increase realized
        "description": "Fully mature transit corridor, stable values"
    }
}

# APM Operating Cost Scenarios (scales with corridor length)
APM_COSTS = {
    "operating_cost_fixed": 1_500_000,   # $1.5M/year fixed overhead
    "operating_cost_per_km": 200_000,    # $200K/km/year variable
    "capital_cost_per_km": 50_000_000,   # $50M per km capital cost
    "debt_service_period": 20,           # 20-year revenue bonds
    "default_length_km": 8.0,            # Default corridor length for breakeven
}

# Location-based phasing speed modifiers
# Corridors near university/downtown develop faster; suburban corridors slower
PHASING_SPEED = {
    "urban_core": 1.3,    # 30% faster development (near Purdue/downtown)
    "suburban": 0.7,      # 30% slower development
    "default": 1.0,       # Standard phasing
}

# TIF allocation strategies
ALLOCATION_STRATEGIES = {
    "operations_focused": {
        "description": "Prioritize APM operations subsidy",
        "allocation": {
            "apm_operations_subsidy": 0.6,
            "station_area_improvements": 0.2,
            "service_expansion": 0.1,
            "reserve": 0.1,
        }
    },
    "infrastructure_focused": {
        "description": "Prioritize area infrastructure and development incentives",
        "allocation": {
            "apm_operations_subsidy": 0.3,
            "station_area_improvements": 0.4,
            "development_incentives": 0.2,
            "service_expansion": 0.1,
        }
    },
    "balanced": {
        "description": "Balance operations and area improvement",
        "allocation": {
            "apm_operations_subsidy": 0.4,
            "station_area_improvements": 0.3,
            "development_incentives": 0.15,
            "service_expansion": 0.1,
            "reserve": 0.05,
        }
    },
}


def get_phasing_multiplier(year: int, phasing: dict = DEVELOPMENT_PHASING,
                           speed_factor: float = 1.0) -> float:
    """Get the development phasing multiplier for a given year.

    Returns a value between 0 and 1 indicating what fraction of the
    ultimate property value increase has been realized by that year.

    Args:
        year: Project year (0-25)
        phasing: Development phasing configuration
        speed_factor: Location-based speed modifier (>1 = faster, <1 = slower).
            Urban core corridors near university/downtown develop faster (1.3x).
            Suburban corridors develop slower (0.7x).

    This models the reality that transit-oriented development takes time:
    - Years 0-2: Planning, approvals, speculation (5% of value)
    - Years 2-5: First developments (33% of value)
    - Years 5-15: Main buildout (85% of value)
    - Years 15+: Full maturation (100% of value)
    """
    # Apply speed factor: effective_year progresses faster/slower
    effective_year = year * speed_factor

    for phase_name, phase_config in phasing.items():
        start_year, end_year = phase_config["years"]
        if start_year <= effective_year < end_year:
            # Linear interpolation within phase
            if phase_name == "planning_phase":
                prev_capture = 0.0
            else:
                # Find previous phase capture
                prev_capture = 0.0
                for prev_name, prev_config in phasing.items():
                    if prev_config["years"][1] == start_year:
                        prev_capture = prev_config["value_capture"]
                        break

            current_capture = phase_config["value_capture"]
            phase_duration = end_year - start_year
            year_in_phase = effective_year - start_year

            # Linear interpolation
            return prev_capture + (current_capture - prev_capture) * (year_in_phase / phase_duration)

    # Beyond last phase: full capture
    return 1.0


def apply_development_phasing(values_df: pd.DataFrame) -> pd.DataFrame:
    """Apply realistic development phasing to property value projections.

    Adjusts the 'projected_value_dollars' to reflect the reality that
    property value increases following transit investment occur gradually,
    not instantaneously.

    The original model assumes full value capture immediately, which is
    unrealistic. This function phases in the value increase over 15 years.
    """
    df = values_df.copy()

    # Calculate the APM-induced value increase for each row
    df['apm_value_increase'] = df['projected_value_dollars'] - df['baseline_value_dollars']

    # Apply phasing multiplier to the increase
    df['phasing_multiplier'] = df['year'].apply(get_phasing_multiplier)
    df['phased_increase'] = df['apm_value_increase'] * df['phasing_multiplier']

    # Calculate phased projected value
    df['projected_value_phased'] = df['baseline_value_dollars'] + df['phased_increase']

    # Replace original projected value
    df['projected_value_original'] = df['projected_value_dollars']
    df['projected_value_dollars'] = df['projected_value_phased']

    return df


def load_property_values(zoning_case="zoning", apply_phasing=True):
    """Load property value projections from uplift simulation.

    Parameters:
    - zoning_case: "zoning" or "no_zoning"
    - apply_phasing: If True, applies realistic development timing to value increases
    """
    if zoning_case == "zoning":
        path = Path("data/processed/property_tax_uplift_zoning.csv")
    else:
        path = Path("data/processed/property_tax_uplift_no_zoning.csv")

    if not path.exists():
        logger.debug(f"  ⚠ Property tax uplift file not found: {path}")
        logger.debug(f"    Run property_tax_uplift_simulation.py first")
        return None

    df = pd.read_csv(path)

    if apply_phasing:
        logger.debug(f"  Applying development phasing to {zoning_case} scenario...")
        df = apply_development_phasing(df)

    return df


def calculate_annual_tax_revenue(values_df, tax_rate=0.01):
    """Calculate annual property tax revenue from projected values."""
    # Group by year
    annual_revenue = values_df.groupby("year").agg({
        "projected_value_dollars": "sum"
    }).reset_index()
    
    annual_revenue["annual_tax_revenue"] = annual_revenue["projected_value_dollars"] * tax_rate
    
    return annual_revenue[["year", "projected_value_dollars", "annual_tax_revenue"]]


def calculate_baseline_tax_revenue(values_df, baseline_growth=0.02, tax_rate=0.01):
    """Calculate baseline (no-APM) tax revenue for increment calculation."""
    # Use Year 0 baseline value as starting point
    year_0_value = values_df[values_df["year"] == 0]["baseline_value_dollars"].sum()
    
    years = sorted(values_df["year"].unique())
    baseline_revenues = []
    
    for year in years:
        # Baseline value grows at constant rate (no APM effect)
        baseline_value = year_0_value * (1 + baseline_growth) ** year
        baseline_tax = baseline_value * tax_rate
        baseline_revenues.append({
            "year": year,
            "baseline_value_dollars": baseline_value,
            "baseline_tax_revenue": baseline_tax,
        })
    
    return pd.DataFrame(baseline_revenues)


def calculate_tif_revenue(values_df, scenario_name="zoning", config=TIF_CONFIG):
    """Calculate TIF revenue: increment x capture rate."""
    logger.info(f"Calculating TIF revenue ({scenario_name})...")
    
    # Get actual tax revenue
    actual_revenue = calculate_annual_tax_revenue(values_df, config["property_tax_rate"])
    
    # Get baseline (no-APM) tax revenue
    baseline_revenue = calculate_baseline_tax_revenue(
        values_df, 
        config["baseline_growth_rate"], 
        config["property_tax_rate"]
    )
    
    # Merge
    merged = actual_revenue.merge(baseline_revenue, on="year")
    
    # Calculate increment and captured portion (clip negatives to zero)
    merged["tax_increment"] = (merged["annual_tax_revenue"] - merged["baseline_tax_revenue"]).clip(lower=0.0)
    merged["tif_revenue_captured"] = merged["tax_increment"] * config["capture_rate"]
    merged["property_owner_benefit"] = merged["tax_increment"] * (1 - config["capture_rate"])
    merged["scenario"] = scenario_name
    
    # Cumulative TIF
    merged["cumulative_tif_revenue"] = merged["tif_revenue_captured"].cumsum()
    
    return merged


def analyze_tif_allocation(tif_revenue, scenario_name="zoning", allocation_strategy="balanced"):
    """Allocate TIF revenue to different uses."""
    logger.info(f"Allocating TIF revenue ({allocation_strategy})...")
    
    allocation = ALLOCATION_STRATEGIES[allocation_strategy]["allocation"]
    
    results = tif_revenue[["year", "tif_revenue_captured"]].copy()
    results["scenario"] = scenario_name
    results["strategy"] = allocation_strategy
    
    for category, pct in allocation.items():
        results[f"{category}_allocation"] = results["tif_revenue_captured"] * pct
    
    return results


def calculate_breakeven_analysis(tif_revenue, apm_costs=APM_COSTS, corridor_length_km=None):
    """Analyze when TIF revenue covers APM costs (scaled by corridor length)."""
    logger.info("Analyzing TIF breakeven scenarios...")

    length_km = corridor_length_km or apm_costs.get("default_length_km", 8.0)

    results = tif_revenue[["year", "cumulative_tif_revenue", "scenario"]].copy()

    # Operating cost scales with corridor length
    annual_ops = (apm_costs["operating_cost_fixed"] +
                  apm_costs["operating_cost_per_km"] * length_km)

    # Debt service from capital cost
    capital_cost = apm_costs["capital_cost_per_km"] * length_km
    debt_period = apm_costs["debt_service_period"]
    interest_rate = 0.05
    if interest_rate > 0:
        r = interest_rate
        n = debt_period
        debt_service = capital_cost * (r * (1 + r)**n) / ((1 + r)**n - 1)
    else:
        debt_service = capital_cost / debt_period

    # Exclude year 0 from breakeven search (0 >= 0 is trivially true)
    nonzero = results[results["year"] > 0]

    # Scenario 1: TIF covers annual operations only
    results["cumulative_ops_subsidy_needed"] = annual_ops * results["year"]
    results["ops_subsidy_covered_by_tif"] = results["cumulative_tif_revenue"] >= results["cumulative_ops_subsidy_needed"]

    ops_breakeven = nonzero.loc[
        nonzero["cumulative_tif_revenue"] >= annual_ops * nonzero["year"], "year"
    ].min()

    # Scenario 2: TIF covers debt service
    results["cumulative_debt_service_needed"] = debt_service * results["year"]
    results["debt_service_covered_by_tif"] = results["cumulative_tif_revenue"] >= results["cumulative_debt_service_needed"]

    debt_breakeven = nonzero.loc[
        nonzero["cumulative_tif_revenue"] >= debt_service * nonzero["year"], "year"
    ].min()

    # Scenario 3: Full cost recovery (ops + debt)
    full_cost = annual_ops + debt_service
    results["cumulative_full_cost_needed"] = full_cost * results["year"]
    results["full_cost_covered_by_tif"] = results["cumulative_tif_revenue"] >= results["cumulative_full_cost_needed"]

    full_breakeven = nonzero.loc[
        nonzero["cumulative_tif_revenue"] >= full_cost * nonzero["year"], "year"
    ].min()

    return results, {
        "operations_breakeven_year": ops_breakeven if pd.notna(ops_breakeven) else 26,
        "debt_service_breakeven_year": debt_breakeven if pd.notna(debt_breakeven) else 26,
        "full_cost_breakeven_year": full_breakeven if pd.notna(full_breakeven) else 26,
        "annual_ops_cost": annual_ops,
        "annual_debt_service": debt_service,
        "corridor_length_km": length_km,
    }


def compare_tif_scenarios(zoning_tif, no_zoning_tif):
    """Compare TIF revenue between zoning and no-zoning scenarios."""
    logger.info("Comparing TIF scenarios...")
    
    merged = zoning_tif[["year", "cumulative_tif_revenue"]].merge(
        no_zoning_tif[["year", "cumulative_tif_revenue"]],
        on="year",
        suffixes=("_zoning", "_no_zoning")
    )
    
    merged["tif_difference"] = merged["cumulative_tif_revenue_zoning"] - merged["cumulative_tif_revenue_no_zoning"]
    merged["tif_difference_pct"] = (merged["tif_difference"] / merged["cumulative_tif_revenue_no_zoning"] * 100)
    
    return merged


def main():
    logger.info("=" * 100)
    logger.info("TIF (TAX INCREMENT FINANCING) MODELING: 25-YEAR ANALYSIS")
    logger.info("WITH REALISTIC DEVELOPMENT PHASING")
    logger.info("=" * 100)

    # Print phasing configuration
    logger.info("\nDevelopment Phasing Configuration:")
    for phase_name, config in DEVELOPMENT_PHASING.items():
        years = config["years"]
        capture = config["value_capture"]
        desc = config["description"]
        logger.debug(f"  Years {years[0]}-{years[1]}: {capture*100:.0f}% value capture ({desc})")

    # Load property values with phasing applied
    logger.info("\nLoading property value projections (with phasing)...")
    zoning_values = load_property_values("zoning", apply_phasing=True)
    no_zoning_values = load_property_values("no_zoning", apply_phasing=True)
    
    if zoning_values is None or no_zoning_values is None:
        logger.info("\n✗ Property value data not found. Run property_tax_uplift_simulation.py first.")
        return
    
    # Calculate TIF revenue for both scenarios
    zoning_tif = calculate_tif_revenue(zoning_values, "zoning", TIF_CONFIG)
    no_zoning_tif = calculate_tif_revenue(no_zoning_values, "no_zoning", TIF_CONFIG)
    
    # Analyze allocations
    zoning_allocation = analyze_tif_allocation(zoning_tif, "zoning", "balanced")
    no_zoning_allocation = analyze_tif_allocation(no_zoning_tif, "no_zoning", "balanced")
    
    # Breakeven analysis
    zoning_breakeven, zoning_breakeven_years = calculate_breakeven_analysis(zoning_tif, APM_COSTS)
    no_zoning_breakeven, no_zoning_breakeven_years = calculate_breakeven_analysis(no_zoning_tif, APM_COSTS)
    
    # Scenario comparison
    comparison = compare_tif_scenarios(zoning_tif, no_zoning_tif)
    
    # Save outputs
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("\n" + "=" * 100)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 100)
    
    # TIF revenue
    zoning_tif.to_csv(output_dir / "tif_revenue_zoning.csv", index=False)
    no_zoning_tif.to_csv(output_dir / "tif_revenue_no_zoning.csv", index=False)
    logger.info(f"[DONE] Saved TIF revenue projections")
    
    # Allocations
    zoning_allocation.to_csv(output_dir / "tif_allocation_zoning.csv", index=False)
    no_zoning_allocation.to_csv(output_dir / "tif_allocation_no_zoning.csv", index=False)
    logger.info(f"[DONE] Saved TIF allocation scenarios")
    
    # Breakeven
    zoning_breakeven.to_csv(output_dir / "tif_breakeven_zoning.csv", index=False)
    no_zoning_breakeven.to_csv(output_dir / "tif_breakeven_no_zoning.csv", index=False)
    logger.info(f"[DONE] Saved breakeven analysis")
    
    # Comparison
    comparison.to_csv(output_dir / "tif_scenario_comparison.csv", index=False)
    logger.info(f"[DONE] Saved scenario comparison")
    
    # Print summary
    logger.info("\n" + "=" * 100)
    logger.info("KEY FINDINGS")
    logger.info("=" * 100)
    
    z25 = zoning_tif[zoning_tif["year"] == 25].iloc[0]
    nz25 = no_zoning_tif[no_zoning_tif["year"] == 25].iloc[0]
    
    logger.info(f"\n25-Year TIF Revenue Summary:")
    logger.debug(f"  Zoning case:")
    logger.debug(f"    Cumulative TIF revenue: ${z25['cumulative_tif_revenue']:,.0f}")
    logger.debug(f"    Average annual: ${z25['tif_revenue_captured']:,.0f}")
    logger.info(f"\n  No-zoning case:")
    logger.debug(f"    Cumulative TIF revenue: ${nz25['cumulative_tif_revenue']:,.0f}")
    logger.debug(f"    Average annual: ${nz25['tif_revenue_captured']:,.0f}")
    logger.info(f"\n  Difference:")
    logger.debug(f"    Additional TIF with zoning: ${z25['cumulative_tif_revenue'] - nz25['cumulative_tif_revenue']:,.0f}")
    
    logger.info(f"\nBreakeven Analysis (when TIF revenue covers costs):")
    logger.debug(f"  Zoning case:")
    logger.debug(f"    Annual operations ($2.5M/yr): Year {zoning_breakeven_years['operations_breakeven_year']}")
    logger.debug(f"    Debt service ($3.5M/yr): Year {zoning_breakeven_years['debt_service_breakeven_year']}")
    logger.debug(f"    Full cost (ops + debt): Year {zoning_breakeven_years['full_cost_breakeven_year']}")
    
    logger.info(f"\n  No-zoning case:")
    logger.debug(f"    Annual operations: Year {no_zoning_breakeven_years['operations_breakeven_year']}")
    logger.debug(f"    Debt service: Year {no_zoning_breakeven_years['debt_service_breakeven_year']}")
    logger.debug(f"    Full cost: Year {no_zoning_breakeven_years['full_cost_breakeven_year']}")
    
    logger.info(f"\nTIF Allocation (Balanced Strategy):")
    z_alloc = zoning_allocation[zoning_allocation["year"] == 25].iloc[0]
    logger.debug(f"  Zoning case (Year 25):")
    logger.debug(f"    APM operations subsidy: ${z_alloc['apm_operations_subsidy_allocation']:,.0f}")
    logger.debug(f"    Station area improvements: ${z_alloc['station_area_improvements_allocation']:,.0f}")
    logger.debug(f"    Service expansion: ${z_alloc['service_expansion_allocation']:,.0f}")

    # Show phasing impact
    logger.info(f"\nDevelopment Phasing Impact:")
    logger.debug(f"  Value capture by year (zoning scenario):")
    for year in [2, 5, 10, 15, 25]:
        multiplier = get_phasing_multiplier(year)
        z_year = zoning_tif[zoning_tif["year"] == year]
        if len(z_year) > 0:
            tif_rev = z_year.iloc[0]["cumulative_tif_revenue"]
            logger.debug(f"    Year {year:2d}: {multiplier*100:5.1f}% of ultimate value, "
                  f"${tif_rev:,.0f} cumulative TIF")

    logger.info("\n  Key insight: Without phasing, early-year TIF revenue would be")
    logger.debug("  overestimated by 67-95%. Phasing reflects realistic development timing.")

    logger.info("\n" + "=" * 100)
    logger.info("[DONE] TIF MODELING COMPLETE")
    logger.info("=" * 100 + "\n")


def build_dynamic_finance_scenario_output(
    tif_df: pd.DataFrame,
    scenario_name: str,
    *,
    dynamic_ridership_df: pd.DataFrame = None,
    fare_per_trip_usd: float = 2.0,
    farebox_capture_rate: float = 1.0,
    apm_costs: dict = None,
    discount_rate: float = 0.05,
) -> pd.DataFrame:
    """Overlay dynamic ridership data onto TIF cashflows for scenario output.

    Interpolates ridership between observed years, computes per-year financials,
    and adds traceability columns.

    Parameters
    ----------
    tif_df : DataFrame with year, tif_revenue_captured, cumulative_tif_revenue, scenario
    dynamic_ridership_df : DataFrame with corridor_id, year, daily_riders
    fare_per_trip_usd : fare per trip
    farebox_capture_rate : fraction of fare revenue captured
    apm_costs : dict with operating_cost_fixed, operating_cost_per_km,
                capital_cost_per_km, debt_service_period, default_length_km
    discount_rate : for NPV computation
    """
    if apm_costs is None:
        apm_costs = APM_COSTS

    years = sorted(tif_df["year"].unique())
    n_years = len(years)

    # Aggregate ridership across corridors per year, interpolate gaps
    if dynamic_ridership_df is not None and not dynamic_ridership_df.empty:
        agg = dynamic_ridership_df.groupby("year")["daily_riders"].sum()
        obs_years = sorted(agg.index)
        obs_values = [float(agg.loc[y]) for y in obs_years]
        # Interpolate to all years in tif_df
        daily_riders_interp = np.interp(years, obs_years, obs_values)
        demand_source = "feedback_loop_dynamic"
    else:
        daily_riders_interp = np.zeros(n_years)
        demand_source = "no_ridership_data"

    length_km = apm_costs.get("default_length_km", 8.0)
    capital_cost = length_km * apm_costs.get("capital_cost_per_km", 50_000_000)
    debt_period = apm_costs.get("debt_service_period", 20)
    annual_debt = capital_cost / debt_period if debt_period > 0 else 0.0
    annual_op = (apm_costs.get("operating_cost_fixed", 1_500_000)
                 + apm_costs.get("operating_cost_per_km", 200_000) * length_km)

    # Build output rows
    out_rows = []
    cumul_net = 0.0
    cashflows = [-capital_cost]

    for i, yr in enumerate(years):
        tif_row = tif_df[tif_df["year"] == yr].iloc[0]
        tif_rev = float(tif_row["tif_revenue_captured"])
        daily = daily_riders_interp[i]
        annual_riders = daily * 312  # OPERATING_DAYS_PER_YEAR
        farebox = annual_riders * fare_per_trip_usd * farebox_capture_rate
        total_revenue = farebox + tif_rev
        net = total_revenue - annual_op - annual_debt
        cumul_net += net
        dcr = total_revenue / annual_debt if annual_debt > 0 else 0.0
        cashflows.append(net)

        out_rows.append({
            "year": yr,
            "scenario": scenario_name,
            "tif_revenue_captured": tif_rev,
            "daily_riders_modeled_total": daily,
            "annual_ridership": annual_riders,
            "farebox_revenue": farebox,
            "total_revenue": total_revenue,
            "annual_operating_cost": annual_op,
            "annual_debt_service": annual_debt,
            "net_cashflow": net,
            "cumulative_net": cumul_net,
            "debt_coverage_ratio_dynamic": dcr,
            "demand_trace_source": demand_source,
        })

    out_df = pd.DataFrame(out_rows)

    # Compute project NPV
    years_arr = np.arange(len(cashflows))
    npv_val = float(np.sum(np.array(cashflows) / np.power(1 + discount_rate, years_arr)))
    out_df["project_npv_dynamic_musd"] = npv_val / 1e6

    # IRR
    try:
        from src.finance import npv_irr as _npv_irr
        result = _npv_irr(
            annual_revenue_musd=[r["total_revenue"] / 1e6 for r in out_rows],
            capex_musd_val=capital_cost / 1e6,
            years=n_years,
            discount_rate=discount_rate,
            annual_cost_musd=[(annual_op + annual_debt) / 1e6] * n_years,
        )
        out_df["project_irr"] = result["irr"]
    except Exception:
        out_df["project_irr"] = float("nan")

    return out_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

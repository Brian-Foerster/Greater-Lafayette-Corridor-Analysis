"""
Financial Corridor Ranking - Multi-Objective Optimization

Ranks APM corridors by combining:
1. Ridership (demand model)
2. TIF revenue potential (property tax uplift)
3. Cost efficiency (cost per rider)
4. Financial viability (debt service coverage)

Key Improvement over Phase 1/2a:
- Phase 1/2a: Rank by ridership only
- This: Multi-objective with financial feasibility filter
- Impact: Identifies corridors that are BOTH high-ridership AND financially viable

Author: UrbanSim APM Analysis
Date: January 13, 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import logging
logger = logging.getLogger(__name__)


class FinancialCorridorRanker:
    """Rank corridors using multi-objective optimization."""
    
    def __init__(self,
                 ridership_weight: float = 0.50,
                 tif_weight: float = 0.30,
                 efficiency_weight: float = 0.20,
                 min_debt_coverage: float = 1.25):
        """
        Initialize ranker with objective weights.

        Args:
            ridership_weight: Weight for ridership metric (0-1)
            tif_weight: Weight for TIF revenue metric (0-1)
            efficiency_weight: Weight for cost/rider efficiency (0-1)
            min_debt_coverage: Minimum debt service coverage ratio (1.25x per bond underwriting standards)

        Note: Weights should sum to 1.0
        """
        assert abs((ridership_weight + tif_weight + efficiency_weight) - 1.0) < 0.01, \
            "Weights must sum to 1.0"
        
        self.ridership_weight = ridership_weight
        self.tif_weight = tif_weight
        self.efficiency_weight = efficiency_weight
        self.min_debt_coverage = min_debt_coverage
    
    def calculate_debt_service(self,
                              capital_cost: float,
                              interest_rate: float = 0.05,
                              term_years: int = 30) -> float:
        """
        Calculate annual debt service using standard amortization formula.
        
        Formula: A = P * [r(1+r)^n] / [(1+r)^n - 1]
        where A = annual payment, P = principal, r = rate, n = periods
        """
        if interest_rate == 0:
            return capital_cost / term_years
        
        r = interest_rate
        n = term_years
        
        annual_payment = capital_cost * (r * (1 + r)**n) / ((1 + r)**n - 1)
        return annual_payment
    
    def calculate_tif_revenue(self,
                             base_property_value: float,
                             future_property_value: float,
                             property_tax_rate: float = 0.012,
                             tif_capture_rate: float = 0.75,
                             years: int = 25) -> float:
        """
        Calculate cumulative TIF revenue over project lifetime.
        
        Args:
            base_property_value: Current property value in TIF district
            future_property_value: Projected property value at year 25
            property_tax_rate: Effective property tax rate (1.2% typical)
            tif_capture_rate: % of increment captured by TIF (75% typical)
            years: TIF district duration
        
        Returns:
            Cumulative TIF revenue over years
        """
        # Annual increment (simplified linear growth assumption)
        annual_increment = (future_property_value - base_property_value) / years
        
        # TIF revenue grows each year as increment compounds
        cumulative_tif = 0
        current_value = base_property_value
        
        for year in range(years):
            current_value += annual_increment
            annual_tax_increment = (current_value - base_property_value) * property_tax_rate
            tif_revenue = annual_tax_increment * tif_capture_rate
            cumulative_tif += tif_revenue
        
        return cumulative_tif
    
    def calculate_debt_coverage_ratio(self,
                                      tif_revenue_cumulative: float,
                                      annual_debt_service: float,
                                      years: int = 25) -> float:
        """
        Calculate debt service coverage ratio using phased TIF revenue.

        Uses year-5 TIF revenue (early development phase) rather than
        25-year average, since bond underwriters evaluate early-year coverage.
        Development phasing: 5% (yr 0-2), 33% (yr 2-5), 85% (yr 5-15), 100% (yr 15+).

        Typical thresholds:
        - < 1.0: Cannot cover debt (NOT viable)
        - 1.0-1.25: Barely covers debt (HIGH RISK)
        - 1.25-1.5: Adequate coverage (MODERATE RISK)
        - > 1.5: Strong coverage (LOW RISK)
        """
        if annual_debt_service == 0:
            return float('inf')

        # Compute year-specific TIF using phased development
        # Average annual ultimate TIF (if no phasing)
        avg_annual_tif = tif_revenue_cumulative / years

        # Year-5 TIF is ~33% of ultimate (early development phase)
        # This is more conservative than the 25-yr average
        year5_tif = avg_annual_tif * 0.33

        # Mature-phase TIF (years 5-15 average = ~85% of ultimate)
        mature_tif = avg_annual_tif * 0.85

        # Use weighted average: 20% weight on early years, 80% on mature
        # This reflects that most bond life is in the mature phase
        effective_annual_tif = 0.2 * year5_tif + 0.8 * mature_tif

        return effective_annual_tif / annual_debt_service
    
    def normalize_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize metrics to 0-1 scale for multi-objective comparison.
        
        Uses min-max normalization: (x - min) / (max - min)
        """
        df = df.copy()
        
        # Ridership: higher is better
        if 'daily_ridership' in df.columns and df['daily_ridership'].max() > 0:
            df['ridership_normalized'] = (
                (df['daily_ridership'] - df['daily_ridership'].min()) /
                (df['daily_ridership'].max() - df['daily_ridership'].min())
            )
        else:
            df['ridership_normalized'] = 0.0
        
        # TIF revenue: higher is better
        if 'tif_revenue_cumulative' in df.columns and df['tif_revenue_cumulative'].max() > 0:
            df['tif_normalized'] = (
                (df['tif_revenue_cumulative'] - df['tif_revenue_cumulative'].min()) /
                (df['tif_revenue_cumulative'].max() - df['tif_revenue_cumulative'].min())
            )
        else:
            df['tif_normalized'] = 0.0
        
        # Cost per rider: LOWER is better, so invert
        if 'cost_per_rider' in df.columns and df['cost_per_rider'].max() > 0:
            df['efficiency_normalized'] = (
                (df['cost_per_rider'].max() - df['cost_per_rider']) /
                (df['cost_per_rider'].max() - df['cost_per_rider'].min())
            )
        else:
            df['efficiency_normalized'] = 0.0
        
        return df
    
    def calculate_composite_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted composite score from normalized metrics.
        
        Score = w1*ridership + w2*tif + w3*efficiency
        where all metrics are normalized 0-1
        """
        score = (
            self.ridership_weight * df['ridership_normalized'] +
            self.tif_weight * df['tif_normalized'] +
            self.efficiency_weight * df['efficiency_normalized']
        )
        
        return score
    
    def rank_corridors(self,
                      corridor_results: pd.DataFrame,
                      capital_cost_per_km: float = 50_000_000,
                      interest_rate: float = 0.05,
                      term_years: int = 30,
                      property_tax_rate: float = 0.012,
                      cashflow_years: int = 25,
                      fare_per_trip_usd: float = 2.0,
                      farebox_capture_rate: float = 1.0) -> pd.DataFrame:
        """
        Rank corridors using multi-objective optimization.

        Args:
            corridor_results: DataFrame with corridor metrics
                Required columns: corridor_id, length_km, daily_ridership
                Optional: base_property_value, future_property_value,
                          daily_ridership_series (list of daily riders per year)
            capital_cost_per_km: APM capital cost ($/km)
            interest_rate: Municipal bond rate
            term_years: Bond term
            property_tax_rate: Effective property tax rate
            cashflow_years: Number of years for NPV / dynamic ridership
            fare_per_trip_usd: Fare per boarding
            farebox_capture_rate: Fraction of riders paying fare

        Returns:
            Ranked DataFrame with financial metrics added
        """
        df = corridor_results.copy()

        # --- Dynamic ridership support ---
        has_dyn = "daily_ridership_series" in df.columns
        df["has_dynamic_ridership"] = False
        if has_dyn:
            # Validate series lengths match cashflow_years
            for idx, row in df.iterrows():
                series = row["daily_ridership_series"]
                if isinstance(series, (list, np.ndarray)) and len(series) > 0:
                    if len(series) != cashflow_years:
                        raise ValueError(
                            f"Corridor {row.get('corridor_id', idx)}: "
                            f"daily_ridership_series length {len(series)} != "
                            f"cashflow_years {cashflow_years}"
                        )
                    df.at[idx, "has_dynamic_ridership"] = True

        # Effective annual ridership (mean of dynamic series or static)
        def _effective_annual(row):
            if row["has_dynamic_ridership"]:
                return float(np.mean(row["daily_ridership_series"])) * 365
            return float(row.get("daily_ridership", 0)) * 365

        df["annual_ridership_effective"] = df.apply(_effective_annual, axis=1)

        # Farebox revenue NPV
        def _farebox_npv(row):
            if row["has_dynamic_ridership"]:
                series = np.array(row["daily_ridership_series"], dtype=float)
                annual_rev = series * 365 * fare_per_trip_usd * farebox_capture_rate
            else:
                annual_rev = np.full(
                    cashflow_years,
                    row.get("daily_ridership", 0) * 365 * fare_per_trip_usd * farebox_capture_rate,
                )
            # NPV at interest_rate
            discounts = np.array([(1 + interest_rate) ** (-y) for y in range(cashflow_years)])
            return float(np.sum(annual_rev * discounts)) / 1_000_000.0

        df["farebox_revenue_npv_musd"] = df.apply(_farebox_npv, axis=1)

        # Calculate capital cost
        df['capital_cost'] = df['length_km'] * capital_cost_per_km

        # Calculate annual operating cost (scales with corridor length)
        # Base: $1.5M/yr fixed + $200K/km/yr variable
        df['annual_operating_cost'] = 1_500_000 + df['length_km'] * 200_000

        # Calculate annual debt service
        df['annual_debt_service'] = df['capital_cost'].apply(
            lambda c: self.calculate_debt_service(c, interest_rate, term_years)
        )

        # Total annual cost (debt + operations)
        df['total_annual_cost'] = df['annual_debt_service'] + df['annual_operating_cost']

        # Calculate TIF revenue (if data available)
        if 'base_property_value' in df.columns and 'future_property_value' in df.columns:
            df['tif_revenue_cumulative'] = df.apply(
                lambda row: self.calculate_tif_revenue(
                    row['base_property_value'],
                    row['future_property_value'],
                    property_tax_rate
                ),
                axis=1
            )

            # Calculate debt coverage ratio
            df['debt_coverage_ratio'] = df.apply(
                lambda row: self.calculate_debt_coverage_ratio(
                    row['tif_revenue_cumulative'],
                    row['annual_debt_service']
                ),
                axis=1
            )

            # Financial viability flag
            df['financially_viable'] = df['debt_coverage_ratio'] >= self.min_debt_coverage
        else:
            # No TIF data available - mark all as unknown
            df['tif_revenue_cumulative'] = 0.0
            df['debt_coverage_ratio'] = 0.0
            df['financially_viable'] = False

        # Calculate cost per rider using effective annual ridership
        df['cost_per_rider'] = df['total_annual_cost'] / df['annual_ridership_effective']
        df['cost_per_rider'] = df['cost_per_rider'].replace([np.inf, -np.inf], np.nan)

        # Also set daily_ridership from effective if needed for normalization
        if 'daily_ridership' not in df.columns:
            df['daily_ridership'] = df['annual_ridership_effective'] / 365

        # Normalize metrics
        df = self.normalize_metrics(df)

        # Calculate composite score
        df['composite_score'] = self.calculate_composite_score(df)

        # Rank by composite score (descending); fill NaN to avoid int cast error
        df['composite_score'] = df['composite_score'].fillna(0.0)
        df['rank'] = df['composite_score'].rank(ascending=False, method='dense').astype(int)

        # Sort by rank
        df = df.sort_values('rank')

        return df


def integrate_dynamic_ridership_data(
    demand_df: pd.DataFrame,
    dynamic_ridership_path: Path = None,
    cashflow_years: int = 25,
) -> pd.DataFrame:
    """Enrich corridor results with per-corridor daily ridership trajectories.

    Reads a feedback-loop CSV with columns (corridor_id, year, daily_riders)
    and interpolates to produce a ``daily_ridership_series`` column containing
    a list of length *cashflow_years* for each corridor.

    Parameters
    ----------
    demand_df : DataFrame
        Corridor results with at least ``corridor_id`` and ``daily_ridership``.
    dynamic_ridership_path : Path, optional
        CSV with (corridor_id, year, daily_riders).
    cashflow_years : int
        Number of years for the output series.

    Returns
    -------
    DataFrame with ``daily_ridership_series`` column added.
    """
    df = demand_df.copy()
    if dynamic_ridership_path is None or not Path(dynamic_ridership_path).exists():
        # No dynamic data — flat series from static ridership
        df["daily_ridership_series"] = df["daily_ridership"].apply(
            lambda r: [float(r)] * cashflow_years
        )
        return df

    dyn = pd.read_csv(dynamic_ridership_path)
    series_map: dict = {}
    for cid, grp in dyn.groupby("corridor_id"):
        grp = grp.sort_values("year")
        years = grp["year"].values
        riders = grp["daily_riders"].values
        # Interpolate to annual series of length cashflow_years
        target_years = np.arange(cashflow_years)
        interp = np.interp(target_years, years, riders)
        series_map[cid] = interp.tolist()

    def _lookup(row):
        cid = row["corridor_id"]
        if cid in series_map:
            return series_map[cid]
        return [float(row.get("daily_ridership", 0.0))] * cashflow_years

    df["daily_ridership_series"] = df.apply(_lookup, axis=1)
    return df


def load_phase2a_results(data_dir: Path = Path("data")) -> pd.DataFrame:
    """Load Phase 2a corridor evaluation results."""
    
    # Try multiple possible file locations
    possible_paths = [
        data_dir / "processed" / "apm_phase2a_results.csv",
        data_dir / "processed" / "corridor_results_phase2a.csv",
        data_dir / "processed" / "phase2a_corridor_results.csv",
        data_dir / "processed" / "apm_iterative_improved_results.csv"
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Loading results from: {path.name}")
            return pd.read_csv(path)
    
    raise FileNotFoundError(f"No Phase 2a results found in {data_dir / 'processed'}")


def integrate_property_tax_data(corridor_results: pd.DataFrame,
                                property_tax_dir: Path = Path("data/processed")) -> pd.DataFrame:
    """
    Integrate property tax uplift data with corridor results.
    
    Looks for files like: property_tax_C23_zoning.csv, property_tax_C23_no_zoning.csv
    """
    df = corridor_results.copy()
    
    # Initialize columns
    df['base_property_value'] = 0.0
    df['future_property_value'] = 0.0
    
    for idx, row in df.iterrows():
        corridor_id = row['corridor_id']
        
        # Try to load property tax data for this corridor
        tax_file_zoning = property_tax_dir / f"property_tax_{corridor_id}_zoning.csv"
        
        if tax_file_zoning.exists():
            tax_data = pd.read_csv(tax_file_zoning)
            
            # Extract Year 0 and Year 25 values
            if 'year' in tax_data.columns and 'total_property_value' in tax_data.columns:
                base_value = tax_data[tax_data['year'] == 0]['total_property_value'].iloc[0]
                future_value = tax_data[tax_data['year'] == 25]['total_property_value'].iloc[0]
                
                df.at[idx, 'base_property_value'] = base_value
                df.at[idx, 'future_property_value'] = future_value
    
    return df


def generate_financial_ranking(corridor_results_path: Path = None,
                               weights: Dict[str, float] = None,
                               output_path: Path = None) -> pd.DataFrame:
    """
    Generate financial ranking for all corridors.
    
    Args:
        corridor_results_path: Path to corridor results CSV (optional)
        weights: Custom weights for objectives (optional)
        output_path: Where to save ranked results (optional)
    
    Returns:
        Ranked DataFrame with financial metrics
    """
    logger.info(f"\n{'='*70}")
    logger.info("FINANCIAL CORRIDOR RANKING")
    logger.info(f"{'='*70}")
    
    # Load corridor results
    if corridor_results_path:
        corridor_results = pd.read_csv(corridor_results_path)
    else:
        corridor_results = load_phase2a_results()
    
    logger.info(f"Loaded {len(corridor_results)} corridors")
    
    # Integrate property tax data
    logger.info("Integrating property tax uplift data...")
    corridor_results = integrate_property_tax_data(corridor_results)
    
    # Create ranker
    if weights:
        ranker = FinancialCorridorRanker(**weights)
    else:
        ranker = FinancialCorridorRanker(
            ridership_weight=0.50,
            tif_weight=0.30,
            efficiency_weight=0.20,
            min_debt_coverage=1.25
        )
    
    logger.info(f"\nRanking criteria:")
    logger.debug(f"  Ridership:   {ranker.ridership_weight*100:.0f}%")
    logger.debug(f"  TIF Revenue: {ranker.tif_weight*100:.0f}%")
    logger.debug(f"  Efficiency:  {ranker.efficiency_weight*100:.0f}%")
    logger.debug(f"  Min debt coverage: {ranker.min_debt_coverage:.2f}x")
    
    # Rank corridors
    logger.info("\nCalculating financial metrics and ranking...")
    ranked = ranker.rank_corridors(
        corridor_results,
        capital_cost_per_km=50_000_000,  # $50M per km
        interest_rate=0.05,              # 5% municipal bonds
        term_years=30,                   # 30-year bonds
        property_tax_rate=0.012          # 1.2% property tax
    )
    
    # Display top 10
    logger.info(f"\n{'='*70}")
    logger.info("TOP 10 CORRIDORS (Multi-Objective Ranking)")
    logger.info(f"{'='*70}")
    
    display_cols = [
        'rank', 'corridor_id', 'length_km', 'daily_ridership',
        'tif_revenue_cumulative', 'debt_coverage_ratio', 
        'cost_per_rider', 'composite_score', 'financially_viable'
    ]
    
    # Format for display
    top_10 = ranked.head(10)[display_cols].copy()
    top_10['tif_revenue_b'] = top_10['tif_revenue_cumulative'] / 1e9
    top_10['cost_per_rider'] = top_10['cost_per_rider'].round(2)
    top_10['composite_score'] = top_10['composite_score'].round(3)
    
    logger.info(top_10[[
        'rank', 'corridor_id', 'daily_ridership', 'tif_revenue_b',
        'debt_coverage_ratio', 'cost_per_rider', 'composite_score', 'financially_viable'
    ]].to_string(index=False))
    
    # Summary statistics
    n_viable = ranked['financially_viable'].sum()
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Financially viable corridors: {n_viable} / {len(ranked)} ({100*n_viable/len(ranked):.1f}%)")
    logger.info(f"Mean debt coverage (viable): {ranked[ranked['financially_viable']]['debt_coverage_ratio'].mean():.2f}x")
    logger.info(f"Best corridor: {ranked.iloc[0]['corridor_id']}")
    logger.debug(f"  Daily ridership: {ranked.iloc[0]['daily_ridership']:,.0f}")
    logger.debug(f"  TIF revenue: ${ranked.iloc[0]['tif_revenue_cumulative']/1e9:.2f}B")
    logger.debug(f"  Debt coverage: {ranked.iloc[0]['debt_coverage_ratio']:.2f}x")
    logger.debug(f"  Composite score: {ranked.iloc[0]['composite_score']:.3f}")
    
    # Save if path provided
    if output_path:
        ranked.to_csv(output_path, index=False)
        logger.info(f"\nSaved ranked results to: {output_path}")
    
    return ranked


def main():
    """Test financial ranking with Phase 2a results."""
    
    ranked = generate_financial_ranking(
        output_path=Path("data/processed/corridors_ranked_financial.csv")
    )
    
    # Sensitivity analysis: different weight combinations
    logger.info(f"\n{'='*70}")
    logger.info("SENSITIVITY: Different Objective Weights")
    logger.info(f"{'='*70}")
    
    weight_scenarios = [
        {"name": "Ridership Focus", "ridership_weight": 0.70, "tif_weight": 0.20, "efficiency_weight": 0.10},
        {"name": "Financial Focus", "ridership_weight": 0.30, "tif_weight": 0.50, "efficiency_weight": 0.20},
        {"name": "Balanced", "ridership_weight": 0.50, "tif_weight": 0.30, "efficiency_weight": 0.20},
    ]
    
    for scenario in weight_scenarios:
        name = scenario.pop('name')
        ranker = FinancialCorridorRanker(**scenario, min_debt_coverage=1.0)
        
        corridor_results = load_phase2a_results()
        corridor_results = integrate_property_tax_data(corridor_results)
        ranked = ranker.rank_corridors(corridor_results)
        
        top_3 = ranked.head(3)['corridor_id'].tolist()
        logger.info(f"\n{name}: {scenario}")
        logger.debug(f"  Top 3: {', '.join(top_3)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

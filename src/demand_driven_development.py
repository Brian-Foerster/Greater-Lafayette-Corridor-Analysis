"""Demand-Driven Development Model
===================================

Replaces the proportional-absorption approach with a demand-gap model
grounded in DiPasquale-Wheaton (1992) and stock-flow housing theory.

Key principles:
  1. Development volume is driven by exogenous regional growth, not zoning capacity.
  2. Zoning determines *where* growth can go and *maximum building size*,
     not *how much* growth occurs.
  3. A vacancy-rent feedback loop prevents oversupply: when vacancy rises,
     effective rents fall, reducing the number of feasible parcels.
  4. Transit accessibility shifts the *location* of demand (more households
     choose station-adjacent parcels) but does not create demand.

References:
  - DiPasquale & Wheaton (1992), "The Markets for Real Estate Assets and
    Space: A Conceptual Framework", JAREA.
  - Saiz (2010), "The Geographic Determinants of Housing Supply", QJE.
  - UrbanSim developer model: Developer.compute_units_to_build().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.financial_params import CONSTRUCTION_LOAN_RATE as _CONSTRUCTION_LOAN_RATE


# ============================================================================
# Default parameters (Lafayette MSA calibration)
# ============================================================================

@dataclass
class MetroGrowthParams:
    """Exogenous regional growth assumptions.

    Sources:
      - IBRC Lafayette MSA forecast (2024): ~1.5% annual pop growth
      - FRED LWLPOP series: MSA population ~232,000 (2022)
      - Census ACS: avg household size 2.4
      - BLS/IBRC: ~3,000-5,000 new jobs/year in MSA
    """
    metro_population: float = 230_000.0  # Census 2024 est. 229,701 (FRED LWLPOP); rounded
    metro_jobs: float = 96_546.0      # LODES 2023 WAC C000, Tippecanoe County
    annual_pop_growth_rate: float = 0.015
    annual_job_growth_rate: float = 0.018
    # ACS 2020-2024 B25010 = 2.28 county-wide, but includes large student pop in
    # 1-person units. 2.56 (ACS B25010 weighted by new housing mix) better reflects
    # NEW housing units near transit (family/couple-oriented TOD, not dorms).
    avg_household_size: float = 2.56
    sqft_per_employee: float = 200.0  # CoreNet Global 2023; aligned with SQFT_PER_EMPLOYEE in land_use_transport_model
    avg_unit_sqft: float = 900.0


@dataclass
class MarketParams:
    """Market equilibrium parameters.

    Natural vacancy rates calibrated to Lafayette MSA:
      - Residential: ACS 2022 5-year estimates (B25002/B25004):
        homeowner vacancy 1.4%, rental vacancy 5.8%.
        Weighted by ~60% rental (university town) = 4.0%.
      - Commercial: CBRE Midwest Q4 2023 = ~8% office vacancy
        for secondary Midwest markets.

    Rent adjustment speed controls how quickly rents respond to
    vacancy deviations.  A value of 0.5 is the reference rate for
    5-year steps.  For shorter steps, the effective speed is scaled
    by (step_years / 5.0) to maintain the same cumulative effect.
    """
    target_vacancy_residential: float = 0.06  # structural equilibrium (frictional + seasonal)
    # Commercial vacancy: 8% is weighted avg of retail (5%) and office (12%)
    # for Lafayette/secondary Midwest markets (CBRE Q4 2023). National post-COVID
    # office vacancy (~20%+) is driven by Class A/B in gateway cities; Lafayette's
    # smaller, service-oriented office stock has lower remote-work exposure.
    # Retail vacancy in college towns remains tight (~4-6%). If office-heavy
    # corridors are added, consider raising to 0.12 for office-only parcels.
    target_vacancy_commercial: float = 0.08
    rent_adjustment_speed: float = 0.50
    min_rent_multiplier: float = 0.60
    max_rent_multiplier: float = 1.40


@dataclass
class CorridorCaptureParams:
    """Controls what share of metro-wide growth is captured by
    transit corridors (within 800m of stations).

    Literature (TCRP, Nelson 2013):
      - Mature rail systems capture 10-25% of metro growth near stations.
      - New systems in small metros: 5-15% initially, rising with maturity.

    The capture rate is modulated by:
      1. Time (ramp from base to max as system matures)
      2. Scenario (relaxed zoning enables TOD, boosting capture)
      3. Development intensity (density feedback: more building near
         stations signals successful TOD, attracts more demand)

    Scenario multipliers reflect that upzoning enables denser, more
    walkable station areas that attract a larger share of metro growth.
    Evidence: Auckland upzoning captured 5% stock increase (Greenaway-
    McGrevy 2023); Portland MAX corridors captured 15-25% of metro
    growth post-upzoning (Nelson 2013).
    """
    # Empirical per-corridor capture: LODES OD flows show 24-30% of
    # metro workers live/work within 800m of each corridor's stations.
    # In independent evaluation mode, each corridor gets its own share.
    base_corridor_capture_rate: float = 0.15   # initial (pre-maturity)
    max_corridor_capture_rate: float = 0.27    # mature (empirical mean ~26.5%)
    capture_ramp_years: float = 15.0

    # Scenario-dependent multipliers on capture rate.
    # current_zoning: baseline (zoning limits density near stations)
    # no_zoning: full density freedom near stations → +30% capture
    scenario_capture_multiplier_current_zoning: float = 1.00
    scenario_capture_multiplier_no_zoning: float = 1.30

    # Density-responsive feedback: when development intensity (units built
    # per catchment household) is high, capture rate increases next period.
    enable_density_feedback: bool = True
    density_feedback_sensitivity: float = 0.3  # log-scale sensitivity
    max_density_multiplier: float = 1.50
    min_density_multiplier: float = 0.80

    def get_scenario_multiplier(self, scenario: str) -> float:
        s = str(scenario).strip().lower()
        if s == "no_zoning":
            return self.scenario_capture_multiplier_no_zoning
        return self.scenario_capture_multiplier_current_zoning




@dataclass
class ZoningCostParams:
    """Scenario-dependent cost and rent adjustments from zoning relaxation.

    **Cost side** (NAHB/NMHC 2022, adjusted for Indiana WRLURI):
      National regulatory cost = 40.6% of total development cost.
      Indiana estimated = ~28.4% (no IZ mandates, lower NIMBY, faster approvals).
      Zoning-sensitive share in Indiana = ~8.8%.

      Scenarios:
        current_zoning: full regulatory burden (baseline)
        no_zoning:      -6.4% costs (all zoning-sensitive costs eliminated)

    **Revenue side** — supply elasticity rent discount:
      Zoning restricts supply, creating a scarcity premium that inflates
      rents above competitive equilibrium.  Relaxing zoning increases
      supply elasticity, compressing this premium.  The discount applies
      to *property value* in the proforma, reducing the number of
      parcels where development pencils out.

      Under current zoning, restricted FAR limits the stock of developable
      parcels near stations → rents stay above competitive level.  Under
      no_zoning, FAR 8.0 makes virtually every parcel developable →
      developers compete on price → rents converge toward marginal cost.

      Evidence:
        - Glaeser & Gyourko (2003): zoning restricts supply and raises
          prices.  Removing restrictions shifts the supply curve right:
          more housing built, prices moderate toward construction cost.
        - Saiz (2010): supply elasticity determines whether demand
          shocks produce quantity (units) or price (rent) increases.
          Removing zoning makes supply more elastic → more construction.
        - Auckland 2016 Unitary Plan: upzoning 75% of land produced
          +50% more dwellings and rents 26-33% below counterfactual
          (Greenaway-McGrevy & Phillips 2023).
        - Houston: no traditional zoning; 80K+ units on small lots;
          housing prices lower than comparable TX metros.
        - Rent moderation is an OUTCOME of increased supply, not an
          upfront input.  The model handles this via endogenous vacancy
          feedback (update_vacancy_feedback), not a static discount.

    Sources:
      NAHB/NMHC (2022) "Regulation: 40.6% of Multifamily Development Costs"
      Gyourko et al. (2021) Wharton Residential Land Use Regulatory Index
      Glaeser & Gyourko (2003) QJE
      Saiz (2010) QJE
      2025 Tippecanoe County Rental Report
    """
    # Cost multipliers (lower = cheaper to build)
    cost_multiplier_current_zoning: float = 1.000
    cost_multiplier_no_zoning: float = 0.936

    # Rent/value discount: DISABLED (all 1.0).
    # The original design applied an upfront rent discount for scenarios
    # with relaxed zoning, citing Glaeser & Gyourko (2003) and Saiz
    # (2010).  However, those papers argue that removing restrictions
    # INCREASES supply while moderating rents -- they do not argue that
    # deregulation reduces development.  Empirical evidence uniformly
    # confirms this: Auckland (2016 upzoning: +50% dwellings, rents
    # -26-33% vs counterfactual), Houston (no zoning: 80K+ units on
    # small lots), Tokyo (145K units/yr under permissive national
    # zoning).  An upfront rent discount front-loads a price effect
    # that should emerge gradually as new supply enters the market.
    # The model's endogenous vacancy feedback (update_vacancy_feedback)
    # already handles rent moderation when supply outpaces demand,
    # which is the correct mechanism.
    rent_discount_current_zoning: float = 1.00
    rent_discount_no_zoning: float = 1.00

    def get_multiplier(self, scenario: str) -> float:
        s = str(scenario).strip().lower()
        if s == "no_zoning":
            return self.cost_multiplier_no_zoning
        return self.cost_multiplier_current_zoning

    def get_rent_discount(self, scenario: str) -> float:
        """Return supply-elasticity rent discount for scenario."""
        s = str(scenario).strip().lower()
        if s == "no_zoning":
            return self.rent_discount_no_zoning
        return self.rent_discount_current_zoning


@dataclass
class AbsorptionParams:
    """County-capacity construction constraint and developer confidence.

    Replaces the former hard per-corridor cap (120 units/yr) with two
    empirically grounded mechanisms:

    1. **County-capacity cost escalation** (supply-side):
       When corridor development in one step exceeds the construction
       capacity available near the corridor, marginal costs rise due
       to labor competition, overtime, and imported subcontractors.
       This is an upward-sloping supply curve for construction services.

       Each corridor is evaluated independently as the sole APM
       investment.  The question is: if this one line is built, how
       much of the county's construction activity concentrates near
       its stations?

       Census BPS (Tippecanoe County, 2015-2024):
         Total permits: 430-2,004/yr, mean 1,232, median 1,263
         SF: stable ~442/yr (CV 8%)
         MF 5+: lumpy, 0-1,585/yr, mean ~777/yr
         Peak years: 1,751 (2018), 2,004 (2023)

       As the sole major infrastructure investment, one APM corridor
       could attract 30-50% of county construction activity near its
       stations.  Portland MAX lines each attracted ~20-25% of metro
       construction, but Portland has multiple competing lines.  A
       single-line system in a smaller metro has a larger draw.

       Threshold is set at the normal-year capacity.  Below threshold:
       no cost escalation.  Above: costs rise by capacity_cost_elasticity
       per 100% of excess.  Marginal projects become infeasible.

    2. **Developer confidence ramp** (demand-side):
       Applied as a multiplier on the corridor capture rate, not as a
       supply ceiling.  Reflects the behavioral lag between transit
       opening and TOD investment commitment.
         IndyGo Red Line: major projects 2-3 years after launch.
         Minneapolis Green Line: 5-year lag to peak development.
         Portland MAX: 3-5 years to first TOD (Cervero & Landis, 1997).
       Ramp: logistic from 30% at Year 0 to ~97% by Year 15.
    """
    # County-wide annual residential permit capacity (Census BPS mean)
    county_annual_res_permits: float = 1_300.0
    # Share of county construction that concentrates near the corridor
    # when it is the sole APM investment.  Higher than multi-line systems
    # because there is no competing transit corridor.
    # Portland MAX (multi-line): ~20-25% each.  Single-line: ~40%.
    corridor_capacity_share: float = 0.40
    # Cost elasticity: fractional cost increase per 100% above capacity
    # 0.30 = 30% cost increase when demand is 2x corridor capacity
    capacity_cost_elasticity: float = 0.30

    # County-wide annual commercial sqft capacity
    county_annual_comm_sqft: float = 300_000.0
    corridor_comm_capacity_share: float = 0.40

    # Developer confidence ramp (applied to capture rate, not supply)
    enable_confidence_ramp: bool = True
    min_developer_confidence: float = 0.05
    confidence_ramp_midpoint_years: float = 5.0
    confidence_ramp_steepness: float = 0.60
    # Ridership-based confidence: developers respond to observed transit use.
    # FTA before-and-after studies (Denver W Line, Dallas Orange) show
    # development accelerates 2-5 years after ridership demonstrates viability.
    # Floor = minimum ridership multiplier (even at 0 riders, time-based ramp
    # still applies).  Saturation = daily riders at which ridership signal
    # reaches full confidence (1.0).
    # Note: with zero riders, confidence asymptotes at ridership_floor × ~1.0
    # (time sigmoid saturates) ≈ 0.15.  Corridors with persistently low
    # ridership are nearly frozen — correct behavior for speculative TOD
    # in a 230K metro with no fixed-guideway transit precedent.
    confidence_ridership_saturation: float = 8_000.0
    confidence_ridership_floor: float = 0.15
    # Per-year project start cap (shared across residential, student,
    # commercial calls within a single corridor-year).
    # Tippecanoe County: 3-4 multifamily permits/yr avg, 6 peak (Census C-40).
    max_project_starts_per_year: int = 8
    # Speculative rent premium discount: fraction of transit rent premium
    # realized before the system proves itself.  Developers discount
    # projections; premiums mature over premium_maturation_years.
    # FTA before-after: realized premiums avg 60-80% of projected in years 1-5.
    speculative_premium_floor: float = 0.50
    premium_maturation_years: float = 8.0

    @property
    def corridor_annual_res_capacity(self) -> float:
        """Normal annual residential capacity near the corridor."""
        return self.county_annual_res_permits * self.corridor_capacity_share

    @property
    def corridor_annual_comm_capacity(self) -> float:
        """Normal annual commercial sqft capacity near the corridor."""
        return self.county_annual_comm_sqft * self.corridor_comm_capacity_share

    def confidence_factor(self, year: int, daily_riders: float = 0.0) -> float:
        """Developer confidence multiplier for a given year and ridership.

        Blends a time-based sigmoid ramp (developers plan ahead based on
        construction timelines) with a ridership signal (developers respond
        to observed transit use).  The ridership factor scales from
        ``confidence_ridership_floor`` at 0 riders to 1.0 at
        ``confidence_ridership_saturation`` daily riders.

        The final confidence is: time_factor × ridership_factor, ensuring
        that low-ridership corridors develop more slowly even in later years.
        """
        if not self.enable_confidence_ramp:
            return 1.0  # disabled ramp = full confidence
        if year <= 0:
            return self.min_developer_confidence
        # Time-based component (unchanged sigmoid)
        time_factor = self.min_developer_confidence + (
            1.0 - self.min_developer_confidence
        ) / (1.0 + np.exp(
            -self.confidence_ramp_steepness
            * (year - self.confidence_ramp_midpoint_years)
        ))
        # Ridership-based component
        if daily_riders <= 0 or self.confidence_ridership_saturation <= 0:
            ridership_factor = self.confidence_ridership_floor
        else:
            raw = daily_riders / self.confidence_ridership_saturation
            ridership_factor = self.confidence_ridership_floor + (
                1.0 - self.confidence_ridership_floor
            ) * min(raw, 1.0)
        return max(time_factor * ridership_factor, self.min_developer_confidence)

    def cost_escalation(
        self, delivered_units: float, step_years: int,
    ) -> float:
        """Residential construction cost premium from exceeding county capacity.

        Returns a multiplier >= 1.0 applied to construction costs.
        """
        capacity = self.corridor_annual_res_capacity * step_years
        if capacity <= 0 or delivered_units <= capacity:
            return 1.0
        excess_ratio = delivered_units / capacity - 1.0
        return 1.0 + self.capacity_cost_elasticity * excess_ratio

    def cost_escalation_commercial(
        self, delivered_comm_sqft: float, step_years: int,
    ) -> float:
        """Commercial construction cost premium from exceeding county capacity.

        Uses the commercial capacity track (county_annual_comm_sqft)
        independently from residential delivery.
        """
        capacity = self.corridor_annual_comm_capacity * step_years
        if capacity <= 0 or delivered_comm_sqft <= capacity:
            return 1.0
        excess_ratio = delivered_comm_sqft / capacity - 1.0
        return 1.0 + self.capacity_cost_elasticity * excess_ratio


# ============================================================================
# Segment-based development parameters
# ============================================================================

@dataclass
class DeveloperSegment:
    """Single development segment with its own demand, vacancy, rent, and cost parameters.

    Each segment is evaluated independently per corridor: demand from model
    state → compare to segment supply → vacancy → rent → pro forma feasibility.

    References:
      - DiPasquale & Wheaton (1992): stock-flow housing equilibrium
      - NHTS 2017: housing consumption rates
      - Purdue Housing Survey: student housing preferences
      - ITE/ULI: commercial sqft per capita for neighborhood-serving retail
      - Indiana IHCDA: QAP transit proximity scoring (LIHTC)
    """
    name: str                          # "market_rate", "student", "commercial"
    target_vacancy: float              # equilibrium vacancy rate
    rent_adjustment_speed: float       # how fast segment rents respond to vacancy
    min_rent_multiplier: float
    max_rent_multiplier: float
    parcel_filter: str                 # "residential", "campus_residential", "any", "commercial"
    housing_consumption_rate: float    # units per person of segment demand (0.39, 0.30, etc.)
    unit_sqft: float                   # avg unit size for this segment
    cost_subsidy: float                # fractional cost reduction (0.0 for market, ~0.15 for LIHTC)
    confidence_half_years: float       # years to 50% developer confidence


# Default segments (Lafayette MSA calibration)
DEFAULT_SEGMENTS: List[DeveloperSegment] = [
    DeveloperSegment(
        name="market_rate",
        target_vacancy=0.04,           # ACS 2022 Lafayette MSA
        rent_adjustment_speed=0.50,
        min_rent_multiplier=0.60,
        max_rent_multiplier=1.40,
        parcel_filter="residential",
        housing_consumption_rate=0.39,  # 1/2.56 (avg HH size)
        unit_sqft=900,
        cost_subsidy=0.0,
        confidence_half_years=7.0,      # Minneapolis Green Line: 5-year lag
    ),
    DeveloperSegment(
        name="student",
        target_vacancy=0.03,           # student housing runs tighter
        rent_adjustment_speed=0.50,
        min_rent_multiplier=0.60,
        max_rent_multiplier=1.40,
        parcel_filter="campus_residential",
        housing_consumption_rate=0.30,  # fraction seeking transit-accessible housing
        unit_sqft=500,                 # smaller student units
        cost_subsidy=0.0,
        confidence_half_years=2.0,     # student housing responds quickly
    ),
    DeveloperSegment(
        name="commercial",
        target_vacancy=0.08,           # CBRE Midwest
        rent_adjustment_speed=0.50,
        min_rent_multiplier=0.60,
        max_rent_multiplier=1.40,
        parcel_filter="commercial",
        housing_consumption_rate=0.0,   # not housing
        unit_sqft=200,                 # sqft per employee (CoreNet Global 2023)
        cost_subsidy=0.0,
        confidence_half_years=9.0,     # commercial follows residential
    ),
]

# Commercial sqft per capita (ITE/ULI standard for neighborhood-serving retail)
_COMMERCIAL_SQFT_PER_CAPITA = 40.0


# ============================================================================
# Core model
# ============================================================================

class DemandDrivenDevelopmentModel:
    """Allocates development based on demand gaps, not zoning capacity.

    Architecture: endogenous property-market equilibrium per segment.

    For each corridor, each year, each segment:
      1. Demand from model state (pop_catch, campus_pop_catch, cumulative_pop)
      2. Compare to segment supply → segment vacancy rate
      3. Vacancy drives segment rent (tight market = high rent)
      4. Segment rent feeds into pro forma feasibility
      5. Build if property_value > cost → new supply feeds back to step 2

    Segments:
      - market_rate: catchment pop → housing units (housing_consumption_rate)
      - student: campus-affiliated pop → student housing (student_housing_rate)
      - commercial: cumulative corridor pop → neighborhood retail (sqft_per_capita)
    """

    def __init__(
        self,
        growth_params: Optional[MetroGrowthParams] = None,
        market_params: Optional[MarketParams] = None,
        capture_params: Optional[CorridorCaptureParams] = None,
        zoning_cost_params: Optional[ZoningCostParams] = None,
        absorption_params: Optional[AbsorptionParams] = None,
        segments: Optional[List[DeveloperSegment]] = None,
    ):
        self.growth = growth_params or MetroGrowthParams()
        self.market = market_params or MarketParams()
        self.capture = capture_params or CorridorCaptureParams()
        self.zoning_costs = zoning_cost_params or ZoningCostParams()
        self.absorption = absorption_params or AbsorptionParams()
        self.segments_list: List[DeveloperSegment] = segments or list(DEFAULT_SEGMENTS)
        self._segments_by_name: Dict[str, DeveloperSegment] = {
            s.name: s for s in self.segments_list
        }

        # Track cumulative corridor-level supply for vacancy calculation
        self._corridor_res_units: Dict[str, float] = {}
        self._corridor_comm_sqft: Dict[str, float] = {}
        self._corridor_households: Dict[str, float] = {}
        self._corridor_jobs: Dict[str, float] = {}

        # Metro-wide rent multiplier (evolves via vacancy feedback)
        # Kept for backward compat; segment rents are the primary mechanism.
        self._rent_multiplier: float = 1.0

        # Per-corridor, per-segment supply and rents
        # Keys: (corridor_id) → {segment_name: value}
        self._segment_supply: Dict[str, Dict[str, float]] = {}
        self._segment_rents: Dict[str, Dict[str, float]] = {}
        # Cumulative corridor population (drives commercial demand lag)
        self._cumulative_corridor_pop: Dict[str, float] = {}
        # Baseline catchment signals at year 0 — only growth above baseline
        # generates new construction (existing stock already serves baseline pop)
        self._baseline_catchment: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # NOTE: allocate_to_parcels(), update_vacancy_feedback(), and
    # compute_segment_demands() removed — they used the dead
    # DeveloperProForma path.  Parcel allocation now goes through
    # SqFtProForma in land_use_transport_model._run_proforma_developer().
    # ------------------------------------------------------------------

    @property
    def rent_multiplier(self) -> float:
        return self._rent_multiplier

    # ------------------------------------------------------------------
    # Segment-based demand and rent (endogenous development)
    # ------------------------------------------------------------------

    def _ensure_segment_state(self, cid: str) -> None:
        """Initialize per-corridor segment tracking if not present."""
        seg_names = [s.name for s in self.segments_list]
        if cid not in self._segment_supply:
            self._segment_supply[cid] = {n: 0.0 for n in seg_names}
        if cid not in self._segment_rents:
            self._segment_rents[cid] = {n: 1.0 for n in seg_names}
        if cid not in self._cumulative_corridor_pop:
            self._cumulative_corridor_pop[cid] = 0.0

    def _segment_confidence(self, seg: DeveloperSegment, year: int) -> float:
        """Per-segment developer confidence (logistic ramp)."""
        if year <= 0:
            return 0.30
        return 0.30 + 0.70 / (
            1.0 + np.exp(-0.40 * (year - seg.confidence_half_years))
        )

    def compute_student_demand(
        self,
        corridor_id: str,
        campus_pop_catch: float,
        year: int,
        step_years: int = 1,
    ) -> Dict[str, float]:
        """Formula-driven student housing demand (Option C).

        Students remain outside the relocation MNL because their location
        choice is dominated by campus proximity and price, with ~25% annual
        turnover (vs 5% for regular households).
        """
        self._ensure_segment_state(corridor_id)
        seg = self._segments_by_name.get("student")
        if seg is None:
            return {"demand_sqft": 0.0, "segment_rent": 1.0, "confidence": 0.0}

        confidence = self._segment_confidence(seg, year)
        supply = self._segment_supply.get(corridor_id, {}).get("student", 0.0)

        # Only GROWTH above year-0 baseline generates new housing demand.
        # Existing students already have housing — only enrollment growth
        # or transit-induced redistribution creates incremental demand.
        baseline = self._baseline_catchment.get(corridor_id, {})
        if "student" not in baseline:
            baseline["student"] = campus_pop_catch
            self._baseline_catchment[corridor_id] = baseline
        growth = max(campus_pop_catch - baseline["student"], 0.0)
        raw_demand_units = growth * seg.housing_consumption_rate
        unmet = max(raw_demand_units - supply, 0.0)

        # Cap to absorption capacity (scaled by step duration)
        annual_cap = self.absorption.corridor_annual_res_capacity * max(step_years, 1)
        unmet = min(unmet, annual_cap)

        demand_sqft = unmet * seg.unit_sqft * confidence

        # Student rent tracking (simple — less responsive than market)
        rents = self._segment_rents.get(corridor_id, {})
        current_rent = rents.get("student", 1.0)

        return {
            "demand_sqft": demand_sqft,
            "segment_rent": current_rent,
            "confidence": confidence,
        }

    def update_segment_supply(
        self,
        corridor_id: str,
        segment_name: str,
        corridor_dev: Dict[str, float],
    ) -> None:
        """Update per-corridor, per-segment supply after allocation.

        Called once per segment after allocate_to_parcels().
        """
        self._ensure_segment_state(corridor_id)
        supply = self._segment_supply[corridor_id]

        if segment_name == "commercial":
            supply[segment_name] = supply.get(segment_name, 0.0) + corridor_dev.get("new_comm_sqft", 0.0)
        else:
            supply[segment_name] = supply.get(segment_name, 0.0) + corridor_dev.get("new_units", 0.0)

        # Note: _cumulative_corridor_pop is updated in
        # LandUseTransportModel._run_development_model() section E, which
        # is the single source of truth for cumulative pop across all
        # channels (residential + student + commercial jobs).

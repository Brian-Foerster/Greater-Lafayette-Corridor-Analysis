"""Shared financial parameters for APM corridor evaluation.

Single source of truth for all financial constants used across the search,
ranking, evaluation, and feedback loop scripts.  Any script that needs
capital cost, tax rates, TIF parameters, or O&M assumptions should import
from this module rather than defining its own constants.

Sources:
    - Capital cost: user-specified $100M/km (grade-separated AGT)
    - Property tax: WL-TSC district, DLGF 2024 certified ($2.32/$100 AV)
    - TIF capture: Indiana IC 36-7-14 statutory default (100%)
    - O&M: FTA National Transit Database, AGT peers (2023)
    - Bond rate: Bloomberg Municipal Bond Index (2024)
    - Debt term: Indiana TIF max life (25 years)
    - Fare: CityBus 2026 integrated fare ($2.00)
    - Operating days: FTA STOPS methodology (300 equivalent days)
    - BCA: USDOT BCA Guidance (2024), EPA SC-CO2 (2024), AAA (2024)
    - Fiscal: Indiana IC 6-3.6, BLS OES, Census ARTS
    - Employment: APTA (2020), BLS Lafayette MSA, RSMeans Midwest (2024)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ===================================================================
# Capital Cost — MECE Decomposition (2025$)
# ===================================================================
# Sources: Detroit People Mover, Miami Metromover, Jacksonville Skyway
# (inflation-adjusted to 2025$); TRB APM cost breakdown guidance;
# Alstom Innovia pricing.  Excludes airport APMs (LAX, PHX) which have
# 3-10x cost premiums from terminal integration and security.

# 1. Guideway & civil works (40-55% of total)
#    Elevated dual-track: columns, precast beams, running surface, guide/power
#    rails, barriers, drainage, utility relocation.
#    Detroit ~$55M/km (guideway-only), Miami ext ~$60M/km.
#    Midwest greenfield discount ~10% vs. coastal applied.
CAPITAL_COST_GUIDEWAY_PER_KM = 55_000_000  # $/route-km

# 2. Stations (20-30% of total)
#    Platform, canopy/enclosure, vertical circulation (elevators, stairs, ADA),
#    HVAC, lighting, fare collection, passenger information displays.
#    Peer range: $13-17M/station (Jacksonville low, Miami high).
CAPITAL_COST_PER_STATION = 15_000_000  # $/station

# 3. Vehicles
#    Alstom Innovia APM 100/200/300 series, ~$2.5-3.5M per car (2024).
#    Two-car consists (50 pax/car = 100 pax/train).
CAPITAL_COST_PER_VEHICLE = 3_000_000  # $/vehicle (car)
DEFAULT_FLEET_VEHICLES = 40  # 20 trains × 2 cars (covers 90s headway on ~8km)

# 4. Fixed systems (signaling, control center, power distribution, comms)
#    Relatively constant for small systems (5-15 km).
CAPITAL_COST_SYSTEMS_FIXED = 15_000_000  # $ lump sum

# 5. Professional services & contingency
#    Design, construction management, environmental, permitting.
#    FTA Standard Cost Categories: 8-12% typical.
PROFESSIONAL_SERVICES_RATE = 0.10  # 10% of direct costs

# APM operational parameters for fleet sizing
# Average speed includes acceleration, deceleration, and dwell — NOT cruise speed.
# LAX APM: 76 km/h top, ~22 km/h avg; Morgantown PRT: ~17 km/h avg.
# For urban corridor with stops every ~1 km, 27 km/h is mid-range.
APM_AVG_SPEED_KPH = 27.0         # Average operating speed (incl accel/decel/dwell)
APM_DWELL_TIME_S = 25.0          # Station dwell time (platform doors, level boarding)
APM_DESIGN_HEADWAY_MIN = 1.5     # 90-second design headway (physical minimum for AGT)
CARS_PER_TRAIN = 2               # Two-car consists (50 pax/car = 100 pax/train)

# BRT operational parameters
BRT_AVG_SPEED_KPH = 22.5         # TCRP Report 90 Vol II: 12-17 mph avg w/ dedicated lanes
BRT_DWELL_TIME_S = 35.0          # Off-board fare collection + level boarding (BRT standard)
BRT_VEHICLES_PER_CONSIST = 1     # Single articulated bus

# Service span parameters (parameterized, not hardcoded)
SERVICE_SPAN_HOURS = 18.0        # 5am-11pm typical APM/BRT weekday
SERVICE_DAYS_PER_YEAR = 312      # Weekdays + Saturdays (260 + 52)


def compute_fleet_vehicles(
    length_km: float,
    n_stations: int,
    headway_min: float = APM_DESIGN_HEADWAY_MIN,
    cars_per_train: int | None = None,
) -> int:
    """Compute fleet size (vehicles/cars) needed for a corridor and headway.

    Parameters
    ----------
    cars_per_train : override for CARS_PER_TRAIN (default 2).
        Set to 4 for 4-car consists (200 pax/train, halves frequency
        requirement at same capacity but doubles per-train rolling stock).

    Formula: round_trip_min / headway = trains needed, × cars_per_train.

    Cross-check (8 km, 6 stations, 1.5 min headway, 27 km/h avg, 2-car):
        Travel: 2 × 8/27 × 60 = 35.6 min
        Dwell:  2 × 6 × 25/60 = 5.0 min
        Round trip: 40.6 min → 28 trains × 2 = 56 vehicles
    """
    _cpt = cars_per_train if cars_per_train is not None else CARS_PER_TRAIN
    travel_min = 2.0 * length_km / APM_AVG_SPEED_KPH * 60.0
    dwell_min = 2.0 * n_stations * APM_DWELL_TIME_S / 60.0
    round_trip_min = travel_min + dwell_min
    n_trains = max(2, math.ceil(round_trip_min / max(headway_min, 0.5)))
    return n_trains * _cpt


def compute_apm_annual_vehicle_hours(
    length_km: float,
    n_stations: int,
    headway_min: float,
    cars_per_train: int | None = None,
) -> float:
    """Annual vehicle-hours for APM at a given headway.

    Returns total vehicle-hours (individual cars, not trains) per year.
    ``cars_per_train`` overrides CARS_PER_TRAIN (default 2).
    """
    _cpt = cars_per_train if cars_per_train is not None else CARS_PER_TRAIN
    travel_min = 2.0 * length_km / APM_AVG_SPEED_KPH * 60.0
    dwell_min = 2.0 * n_stations * APM_DWELL_TIME_S / 60.0
    round_trip_min = travel_min + dwell_min
    n_trains = max(1, math.ceil(round_trip_min / max(headway_min, 0.5)))
    daily_veh_hours = n_trains * _cpt * SERVICE_SPAN_HOURS
    return daily_veh_hours * SERVICE_DAYS_PER_YEAR


def compute_brt_annual_vehicle_hours(
    length_km: float,
    n_stations: int,
    headway_min: float,
) -> float:
    """Annual vehicle-hours for BRT at a given headway."""
    travel_min = 2.0 * length_km / BRT_AVG_SPEED_KPH * 60.0
    dwell_min = 2.0 * n_stations * BRT_DWELL_TIME_S / 60.0
    round_trip_min = travel_min + dwell_min
    n_vehicles = max(1, math.ceil(round_trip_min / max(headway_min, 1.0)))
    daily_veh_hours = n_vehicles * BRT_VEHICLES_PER_CONSIST * SERVICE_SPAN_HOURS
    return daily_veh_hours * SERVICE_DAYS_PER_YEAR


def compute_capital_cost(
    length_km: float,
    n_stations: int,
    n_vehicles: int | None = None,
    curve_cost_mult: float = 1.0,
    escalation: bool = True,
    cars_per_train: int | None = None,
) -> float:
    """MECE capital cost: guideway + stations + vehicles + systems + services.

    Parameters
    ----------
    length_km : corridor route length in km
    n_stations : number of stations
    n_vehicles : fleet size (cars, not trains).  When *None*, automatically
        computed from corridor dimensions via :func:`compute_fleet_vehicles`.
    curve_cost_mult : multiplier for guideway curvature cost (from physics model).
                      Applied only to the guideway component.
    escalation : if True (default), apply mid-construction cost escalation.
        Set to False for constant-dollar comparisons (e.g., corridor search
        scoring where escalation is a uniform multiplier that doesn't affect
        relative ranking).
    cars_per_train : override for CARS_PER_TRAIN (default 2) when
        auto-computing fleet size.  Ignored when *n_vehicles* is provided
        explicitly.

    Returns
    -------
    Total capital cost in USD.

    Cross-check (8 km, 6 stations, 40 vehicles, escalation=True):
        Guideway: 8 × $55M = $440M
        Stations: 6 × $15M = $90M
        Vehicles: 40 × $3M = $120M
        Systems: $15M
        Professional: 10% × $665M = $66.5M
        Subtotal: ~$731M
        Escalation: ×1.092 (4.5%/yr over 4 yr, midpoint)
        Total: ~$798M → ~$100M/km
    """
    if n_vehicles is None:
        n_vehicles = compute_fleet_vehicles(
            length_km, n_stations, cars_per_train=cars_per_train,
        )
    direct = (
        CAPITAL_COST_GUIDEWAY_PER_KM * length_km * curve_cost_mult
        + CAPITAL_COST_PER_STATION * n_stations
        + CAPITAL_COST_PER_VEHICLE * n_vehicles
        + CAPITAL_COST_SYSTEMS_FIXED
    )
    total = direct * (1 + PROFESSIONAL_SERVICES_RATE)

    # Mid-construction escalation: costs drawn down progressively over
    # the construction period.  Average outstanding cost sees ~half the
    # total escalation period.  Equivalent lump-sum: (1 + rate)^(period/2).
    # FTA Standard Cost Category worksheet methodology.
    if escalation:
        esc_factor = (1 + CONSTRUCTION_COST_ESCALATION_RATE) ** (CONSTRUCTION_PERIOD_YEARS / 2.0)
        total *= esc_factor

    return total


# Legacy blended rate (for reference and backward compatibility)
CAPITAL_COST_PER_KM = 100_000_000  # Approximate blended $/km

# Debt / bond parameters
BOND_RATE = 0.05          # 5% municipal bond rate
DEBT_TERM_YEARS = 25      # Indiana TIF max life

# Property tax & TIF
PROPERTY_TAX_RATE = 0.0232        # WL-TSC district (DLGF 2024)
TIF_CAPTURE_RATE = 1.00           # Indiana IC 36-7-14 statutory default
TIF_CAPTURE_RATE_CONSERVATIVE = 0.85  # With admin overhead discount
TIF_YEARS = 25                    # TIF district lifetime
BACKGROUND_APPRECIATION_RATE = 0.02  # Non-TOD general property value inflation

# Indiana circuit breaker caps (IC 6-1.1-20.6)
# These cap the EFFECTIVE property tax rate collected, regardless of gross levy.
# When assessed values rise from TOD development, more parcels hit caps and
# the effective rate collected drops below the gross levy rate.
CIRCUIT_BREAKER_CAP_HOMESTEAD = 0.01    # 1% of AV (owner-occupied)
CIRCUIT_BREAKER_CAP_RENTAL = 0.02       # 2% of AV (non-homestead residential)
# Note: Multi-family apartments (4+ units) are DLGF property class code
# 400-series ("Commercial"), but IC 6-1.1-20.6-4 defines "residential property"
# for circuit breaker purposes as buildings with 2+ dwelling units.  Apartments
# therefore receive the 2% cap, not 3%.  Agricultural land (IC 6-1.1-4-13)
# also receives the 2% cap.
CIRCUIT_BREAKER_CAP_COMMERCIAL = 0.03   # 3% of AV (commercial/industrial)


def effective_tif_tax_rate(
    gross_rate: float = PROPERTY_TAX_RATE,
    res_share: float = 0.5,
) -> float:
    """Compute effective TIF tax rate after Indiana circuit breaker caps.

    TOD development is primarily rental residential (capped at 2%) and
    commercial (capped at 3%).  The gross levy rate of 2.32% exceeds the
    2% residential cap, so residential TIF captures are reduced.

    Returns the blended effective rate for the given residential share.
    """
    res_effective = min(gross_rate, CIRCUIT_BREAKER_CAP_RENTAL)
    comm_effective = min(gross_rate, CIRCUIT_BREAKER_CAP_COMMERCIAL)
    return res_share * res_effective + (1.0 - res_share) * comm_effective

# Operating costs — three-tier structure
# Tier 1: Fixed overhead (headway-independent) — control center, admin, insurance
# Tier 2: Infrastructure maintenance (per-km, headway-independent) — guideway
#          inspection, cleaning, power system, winter de-icing, switch maintenance
# Tier 3: Vehicle operations (per-vehicle-hour, scales with frequency) — energy,
#          vehicle maintenance, parts, NTD-reported AGT cost per revenue-veh-hr
#
# Sources: Detroit People Mover, Morgantown PRT, Jacksonville Skyway (NTD 2023);
#          Morgantown ~$5M/yr for 14km decomposed into fixed/infra/ops.
O_AND_M_FIXED_USD = 1_500_000       # Tier 1: fixed overhead ($/yr)
O_AND_M_INFRA_PER_KM_USD = 200_000  # Tier 2: guideway/infrastructure ($/km/yr)
                                     # Morgantown $180K/km, Phoenix $220K/km; 200K midpoint
O_AND_M_PER_STATION_USD = 150_000   # Tier 2: per-station ($/station/yr)
                                     # Elevator/escalator maint, HVAC, security, cleaning
O_AND_M_VEH_HOUR_USD = 65.0         # Tier 3: per-vehicle-hour ($/veh-hr)
                                     # Energy + vehicle maintenance + parts
                                     # NTD AGT peers: $50-80/veh-hr (excl guideway)
O_AND_M_ESCALATION_RATE = 0.03      # BLS CPI transport services

# Legacy aliases for backward compatibility (sum of tier 1+2 at typical frequency)
# These approximate the old formula for code that hasn't been updated yet.
O_AND_M_PER_KM_USD = O_AND_M_INFRA_PER_KM_USD  # alias for tier 2 infra component

# Construction timing & escalation
# Capital costs are in 2025$ but construction takes 3-5 years, during which
# costs escalate (ENR Building Cost Index: 4-6%/yr for transit, 2020-2024).
CONSTRUCTION_COST_ESCALATION_RATE = 0.045  # 4.5%/yr (ENR BCI avg)
CONSTRUCTION_PERIOD_YEARS = 4              # Typical APM construction period
CONSTRUCTION_LOAN_RATE = 0.08              # Construction financing rate

# Revenue parameters
FARE_PER_TRIP_USD = 2.00          # CityBus 2026 integrated fare
OPERATING_DAYS_PER_YEAR = 312     # Academic calendar (240) + summer/break (72) weighted average

# Funding shares (default: 100% local)
FEDERAL_SHARE = 0.0
STATE_SHARE = 0.0
LOCAL_SHARE = 1.0

# ===================================================================
# Benefit-Cost Analysis (USDOT BCA Guidance 2024)
# ===================================================================

# Discount rates (USDOT requires both)
BCR_DISCOUNT_RATE_LOW = 0.03       # Social rate of time preference
BCR_DISCOUNT_RATE_HIGH = 0.07      # Opportunity cost of capital

# Value of Travel Time Savings (USDOT 2024, 2022$)
VTTS_PERSONAL_PER_HOUR = 18.80
VTTS_BUSINESS_PER_HOUR = 31.00
VTTS_REAL_GROWTH_RATE = 0.012      # 1.2%/yr real growth (USDOT guidance)

# Vehicle Operating Costs (AAA 2024)
VOC_MARGINAL_PER_MILE = 0.22      # Fuel + tires + maintenance only (for BCR)
VOC_FULL_PER_MILE = 0.655         # Full ownership cost (for equity analysis)

# Safety (FHWA, external crash cost — Parry & Small 2005)
CRASH_COST_PER_VMT = 0.12         # External share: fatal + injury + PDO

# Emissions (EPA 2024)
CO2_TONS_PER_VMT = 0.000348       # EPA avg light-duty vehicle 2024
SOCIAL_COST_CO2_PER_TON = 190.0   # EPA/OMB 2024, 2020$, 3% near-term
SC_CO2_REAL_ESCALATION = 0.025    # 2.5%/yr real (EPA schedule)
CRITERIA_POLLUTANT_PER_VMT = 0.020  # NOx/PM2.5/SO2, urban fringe (USDOT BCA)
TOTAL_EMISSION_COST_PER_VMT = CO2_TONS_PER_VMT * SOCIAL_COST_CO2_PER_TON + CRITERIA_POLLUTANT_PER_VMT  # ~$0.086

# Health benefits (WHO HEAT methodology, US adaptation)
WALK_HEALTH_VALUE_PER_MIN = 0.15  # $/min walking, lower bound
AVG_WALK_ACCESS_MIN_PER_TRIP = 10.0  # 800m round trip at 4.8 kph
NEW_WALKING_SHARE = 0.60          # Share of riders generating net new walking

# Parking (ITE 2024, Purdue context)
STRUCTURED_PARKING_COST_PER_SPACE = 35_000
PARKING_STRUCTURE_LIFE_YEARS = 30
PEAK_HOUR_PARKING_FACTOR = 0.85   # Occupancy adjustment

# Public service cost (Tippecanoe County budget / population)
PER_CAPITA_MUNICIPAL_SERVICE_COST = 2_800  # $/person/yr
AVG_HOME_VALUE_NEW_DEVELOPMENT = 200_000   # Typical new TOD unit value

# Residual value (USDOT guidance, 40-yr infrastructure)
RESIDUAL_VALUE_SHARE = 0.50       # Credit 50% of capital at year 25

# ===================================================================
# Employment & Construction (APTA 2020, BLS, RSMeans)
# ===================================================================

CONSTRUCTION_JOBS_PER_MILLION = 30       # APTA 2020 benchmark
CONSTRUCTION_LABOR_SHARE = 0.40          # Labor share of construction cost
CONSTRUCTION_TYPE_II_MULTIPLIER = 2.0    # RIMS II midpoint for transit construction
AVG_CONSTRUCTION_WAGE = 55_000           # BLS, Lafayette MSA

APM_OPS_JOBS_PER_KM = 7                 # 50-100 FTEs for 5-15km system

# TOD construction costs (RSMeans Midwest 2024)
RESIDENTIAL_CONSTRUCTION_COST_PSF = 150  # Wood-frame residential
COMMERCIAL_CONSTRUCTION_COST_PSF = 180   # Steel-frame commercial

# ===================================================================
# Broader Fiscal Impact (Indiana statutes, Census)
# ===================================================================

TIPPECANOE_LIT_RATE = 0.011              # Local Income Tax (IC 6-3.6)
INDIANA_STATE_INCOME_TAX_RATE = 0.0305   # Flat state rate
INDIANA_SALES_TAX_RATE = 0.07            # State sales tax
RETAIL_SHARE_OF_COMMERCIAL = 0.35        # ULI TOD mixed-use benchmark
RETAIL_SALES_PER_SQFT = 350              # Census Annual Retail Trade Survey
AVG_WAGE_WEIGHTED = 42_000               # LODES Lafayette MSA weighted avg
ROAD_MAINTENANCE_COST_PER_VMT = 0.05     # FHWA local road share

# TIF residential cap (Indiana HEA 1120, 2023)
TIF_RESIDENTIAL_MAX_YEARS = 20           # Was 25 for all uses pre-2023

# TIF allocation area types (Indiana IC 36-7-14)
#   "eda"            — Economic Development Area (IC 36-7-14-39(a), post-1995).
#                      No blight finding required.  Residential AV is continuously
#                      added back to the base; only commercial/industrial increment
#                      generates TIF.  This is the likely designation for a new APM
#                      corridor through campus/suburban areas.
#   "redevelopment"  — Redevelopment Area (IC 36-7-14-15).  Requires a finding of
#                      blight/deterioration.  ALL increment (residential + commercial)
#                      is captured, subject to HEA 1120 20-year residential cap.
TIF_AREA_TYPE_DEFAULT = "eda"

# Share of NEW residential construction that is homestead vs rental.
# This is the FALLBACK used only when the feedback loop CSV lacks per-parcel
# tenure columns (new_homestead_sqft / new_rental_sqft).  When those columns
# are present, the evaluator uses them directly and this constant is ignored.
#
# The development model classifies by ZONING of the developed parcel:
#   R1/R1A/R1B/R1T → homestead (SF, FAR 1.0-1.2)
#   R2+/MU/CB/etc  → rental (MF, FAR 1.2-6.0+)
# The proforma strongly selects high-FAR parcels near stations, so the
# built output is overwhelmingly multifamily even though 48.7% of corridor-
# zone parcels are in SF zones by count.
#
# Empirical basis (Tippecanoe County enriched parcels):
#   County-wide: 64% homestead by AV (corrected for 5xx misclassification)
#   Corridor zone parcels: 49% in SF zones by count
#   But proforma-selected parcels are almost entirely R3/R3U/MU (high FAR)
#   Recent new construction near stations: <1% SF by AV
# Fallback set at 0.02 (2%) — reflects proforma's strong MF selection bias.
TIF_HOMESTEAD_SHARE = 0.02  # 2% owner-occupied, 98% multifamily rental

# SB 1 (2025) capture-rate erosion schedule.
# New exemptions/deductions will reduce effective TIF capture over time.
# Year → multiplier applied to the base capture rate.
SB1_CAPTURE_EROSION = {
    0: 1.00,   # no erosion at designation
    5: 0.97,   # early erosion from new homestead deductions
    10: 0.92,  # circuit breaker losses compound
    15: 0.87,  # projected mid-life erosion
    20: 0.82,  # late-life — analyst projections suggest up to 1/3 loss
    25: 0.78,  # end of TIF life
}

# ===================================================================
# Car Diversion Shares by Ridership Component
# ===================================================================

CAR_DIVERSION_COMMUTE = 0.60      # LODES commute: ~60% diverted from car
CAR_DIVERSION_STUDENT = 0.30      # Students: lower car ownership
CAR_DIVERSION_GENERATOR = 0.55    # Medical/retail/event trips
CAR_DIVERSION_INDUCED = 0.00      # New trips — no prior mode to divert from
CAR_DIVERSION_LATENT = 0.00       # Zero-car HH — had no car to divert

# ===================================================================
# Campus Parking Payment (Purdue connection)
# ===================================================================
# Students diverted from car → freed parking spaces → avoided construction
# or surface-lot land liberation.  Purdue pays an annual fee to the APM
# operator proportional to the parking value displaced.

RIDERS_PER_DISPLACED_SPACE = 3.0          # CU Boulder study: 2.75, conservative
# Note: STRUCTURED_PARKING_COST_PER_SPACE = $35,000 already exists (line 247)

# Land liberation (Channel B — alternative to construction avoidance)
SURFACE_SPACES_PER_ACRE = 130             # ITE surface lot standard
CAMPUS_LAND_VALUE_PER_ACRE = 3_000_000    # Discovery Park / peripheral campus

# Annual O&M avoidance (additive to one-time value)
PARKING_OM_PER_SPACE_ANNUAL = 600.0       # $/space/year, Midwest surface lot

# Purdue payment terms
PURDUE_BORROWING_RATE = 0.04              # Moody's Aa1 university bond rate
CAMPUS_PAYMENT_TERM_YEARS = 25            # Matches TIF/debt term

# ===================================================================
# Monte Carlo Distributions for CBA
# Format: (low, mode, high) for triangular distribution
# ===================================================================

MC_VTTS_MULT = (0.80, 1.00, 1.20)
MC_CRASH_COST_MULT = (0.70, 1.00, 1.30)
MC_SC_CO2_MULT = (0.50, 1.00, 2.00)
MC_WALK_HEALTH_VALUE = (0.10, 0.15, 0.30)
MC_AGGLOMERATION_ELASTICITY = (0.03, 0.05, 0.10)


# ===================================================================
# Transit Mode Abstraction (Component 4 — Alternatives Analysis)
# ===================================================================

# BRT O&M — three-tier structure (NTD 2023 peer average, 200-500K metros)
# Defined here (before TransitMode) so BRT_MODE can reference them.
_BRT_O_AND_M_FIXED_USD = 800_000          # Tier 1: fixed overhead ($/yr)
_BRT_O_AND_M_INFRA_PER_KM_USD = 120_000   # Tier 2: lane/signal infrastructure ($/km/yr)
_BRT_O_AND_M_PER_STATION_USD = 100_000    # Tier 2: per-station ($/station/yr)
_BRT_O_AND_M_VEH_HOUR_USD = 140.0         # Tier 3: per-vehicle-hour ($/veh-hr)
                                           # Midwestern NTD peers: $130-150/veh-hr


@dataclass(frozen=True)
class TransitMode:
    """Technology-specific parameters for APM or BRT alternatives.

    Used by the feedback loop and evaluation pipeline so the same model
    runs with different transit technology assumptions.
    """
    name: str
    capital_cost_per_km: float        # $M/km (guideway + stations amortized)
    operating_cost_per_vhr: float     # $/vehicle-hour (Tier 3 vehicle ops)
    om_fixed_usd: float               # Tier 1: annual fixed overhead ($)
    om_infra_per_km_usd: float        # Tier 2: infrastructure maint ($/km/yr)
    om_per_station_usd: float         # Tier 2: per-station maint ($/station/yr)
    speed_kph: float                  # average operating speed (incl accel/decel)
    dwell_time_s: float               # station dwell time (seconds)
    capacity_per_vehicle: int         # passengers per vehicle unit
    min_headway_min: float            # physical minimum headway
    max_headway_min: float            # policy maximum
    grade_separated: bool             # affects reliability, mode choice
    vehicles_per_consist: int         # for fleet sizing (APM: 2 cars/train)
    service_span_hours: float = SERVICE_SPAN_HOURS   # daily service hours
    service_days_per_year: int = SERVICE_DAYS_PER_YEAR  # annual revenue days

    def compute_annual_om(
        self,
        length_km: float,
        n_stations: int,
        annual_vehicle_hours: float,
    ) -> float:
        """Three-tier O&M: fixed + infrastructure + vehicle operations."""
        tier1 = self.om_fixed_usd
        tier2 = self.om_infra_per_km_usd * length_km + self.om_per_station_usd * n_stations
        tier3 = self.operating_cost_per_vhr * annual_vehicle_hours
        return tier1 + tier2 + tier3


APM_MODE = TransitMode(
    name="APM",
    capital_cost_per_km=100_000_000,
    operating_cost_per_vhr=O_AND_M_VEH_HOUR_USD,     # $65/veh-hr
    om_fixed_usd=O_AND_M_FIXED_USD,                   # $1.5M/yr
    om_infra_per_km_usd=O_AND_M_INFRA_PER_KM_USD,     # $200K/km/yr
    om_per_station_usd=O_AND_M_PER_STATION_USD,        # $150K/stn/yr
    speed_kph=APM_AVG_SPEED_KPH,                       # 27 km/h avg
    dwell_time_s=APM_DWELL_TIME_S,                     # 25s
    capacity_per_vehicle=50,
    min_headway_min=1.5,
    max_headway_min=10.0,
    grade_separated=True,
    vehicles_per_consist=2,
)

BRT_MODE = TransitMode(
    name="BRT",
    capital_cost_per_km=25_000_000,
    operating_cost_per_vhr=_BRT_O_AND_M_VEH_HOUR_USD,  # $140/veh-hr
    om_fixed_usd=_BRT_O_AND_M_FIXED_USD,                # $800K/yr
    om_infra_per_km_usd=_BRT_O_AND_M_INFRA_PER_KM_USD,  # $120K/km/yr
    om_per_station_usd=_BRT_O_AND_M_PER_STATION_USD,     # $100K/stn/yr
    speed_kph=BRT_AVG_SPEED_KPH,                        # 22.5 km/h avg
    dwell_time_s=BRT_DWELL_TIME_S,                      # 35s
    capacity_per_vehicle=60,                             # 60-ft articulated bus
    min_headway_min=5.0,                                 # Driver-operated minimum
    max_headway_min=10.0,                                # FTA high-frequency standard
    grade_separated=False,
    vehicles_per_consist=1,
)

# BRT-specific cost parameters
BRT_CAPITAL_COST_PER_STATION = 4_000_000    # At-grade station: shelter, real-time info,
                                             # level boarding, ITS. IndyGo Red Line: ~$3-5M/stn
BRT_CAPITAL_COST_VEHICLES = 1_200_000        # 60-ft articulated electric bus (NTD 2023)
BRT_SYSTEMS_FIXED = 5_000_000               # ITS, signal priority, fare collection, depot mods

# Public aliases for BRT O&M constants (canonical values defined above TransitMode)
BRT_O_AND_M_FIXED_USD = _BRT_O_AND_M_FIXED_USD
BRT_O_AND_M_INFRA_PER_KM_USD = _BRT_O_AND_M_INFRA_PER_KM_USD
BRT_O_AND_M_PER_STATION_USD = _BRT_O_AND_M_PER_STATION_USD
BRT_O_AND_M_VEH_HOUR_USD = _BRT_O_AND_M_VEH_HOUR_USD
BRT_O_AND_M_PER_KM_USD = BRT_O_AND_M_INFRA_PER_KM_USD  # legacy alias


def compute_brt_fleet_vehicles(
    length_km: float,
    n_stations: int,
    headway_min: float = 5.0,
) -> int:
    """Fleet size for BRT corridor (single vehicles, not consists)."""
    travel_min = 2.0 * length_km / BRT_AVG_SPEED_KPH * 60.0
    dwell_min = 2.0 * n_stations * BRT_DWELL_TIME_S / 60.0
    round_trip_min = travel_min + dwell_min
    return max(2, math.ceil(round_trip_min / max(headway_min, 1.0)))


def compute_brt_capital_cost(
    length_km: float,
    n_stations: int,
    n_vehicles: int | None = None,
    escalation: bool = True,
) -> float:
    """BRT capital cost: guideway + stations + vehicles + systems + services."""
    if n_vehicles is None:
        n_vehicles = compute_brt_fleet_vehicles(length_km, n_stations)
    direct = (
        BRT_MODE.capital_cost_per_km * length_km
        + BRT_CAPITAL_COST_PER_STATION * n_stations
        + BRT_CAPITAL_COST_VEHICLES * n_vehicles
        + BRT_SYSTEMS_FIXED
    )
    total = direct * (1 + PROFESSIONAL_SERVICES_RATE)
    if escalation:
        # BRT builds faster (2-3 yr vs 4 for APM)
        esc_factor = (1 + CONSTRUCTION_COST_ESCALATION_RATE) ** (3.0 / 2.0)
        total *= esc_factor
    return total


def incremental_metrics(
    build_annual_cost: float,
    build_annual_trips: float,
    tsm_annual_cost: float,
    tsm_annual_trips: float,
) -> dict:
    """Compute FTA-standard incremental metrics (Build vs. TSM baseline).

    Parameters
    ----------
    build_annual_cost : total annualized cost of Build alternative ($)
    build_annual_trips : annual unlinked trips for Build
    tsm_annual_cost : total annualized cost of TSM alternative ($)
    tsm_annual_trips : annual unlinked trips for TSM

    Returns
    -------
    dict with incremental ridership, cost per trip, and FTA rating.
    """
    incr_trips = build_annual_trips - tsm_annual_trips
    incr_cost = build_annual_cost - tsm_annual_cost

    if incr_trips > 0:
        incr_cost_per_trip = incr_cost / incr_trips
    else:
        incr_cost_per_trip = float("inf")

    # FTA Small Starts cost-effectiveness ratings (2024 thresholds)
    if incr_cost_per_trip < 4.00:
        fta_rating = "High"
    elif incr_cost_per_trip < 8.00:
        fta_rating = "Medium-High"
    elif incr_cost_per_trip < 12.00:
        fta_rating = "Medium"
    else:
        fta_rating = "Low"

    return {
        "incremental_trips": incr_trips,
        "incremental_cost": incr_cost,
        "incremental_cost_per_trip": round(incr_cost_per_trip, 2),
        "fta_cost_effectiveness_rating": fta_rating,
    }

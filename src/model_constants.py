"""
Model Constants for the Land-Use-Transport Feedback Loop
=========================================================

All top-level constants extracted from ``land_use_transport_model.py``.
Import from here instead of the monolith module.
"""
from __future__ import annotations

import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

# Temporal ridership ramp (logistic S-curve)
# FTA New Starts data: opening ridership 40-60% of mature levels.
# Small metro (Greater Lafayette) = fast awareness diffusion.
RAMP_MIDPOINT = 0       # 50% awareness at opening day
RAMP_STEEPNESS = 0.8    # yr0=50%, yr2=83%, yr5=98%, yr10=100%

# Component-specific awareness ramp offsets (years ahead of general ramp)
# Students see APM daily on campus → near-immediate awareness (~95% at year 0)
# Generators (medical, retail) adopt quickly via signage/marketing (~80% at year 0)
# Commuters and latent demand follow the general logistic ramp
STUDENT_AWARENESS_ADVANCE_YR = 3.0    # student ramp 3 years ahead → ~92% at yr0
GENERATOR_AWARENESS_ADVANCE_YR = 1.5  # generators 1.5 years ahead → ~77% at yr0

# Development conversion factors
AVG_HOUSEHOLD_SIZE = 2.56      # persons per residential unit (ACS B25010 Tippecanoe County)
SQFT_PER_EMPLOYEE = 200        # CoreNet Global 2023: ~150-180 office, ~200 blended TOD mix
AVG_UNIT_SQFT = 900            # average residential unit size (sqft)

# Low-density residential zones that produce owner-occupied homestead (1% cap).
# All other zones with residential development produce multifamily rental
# (non-homestead, 2% cap, capturable in post-2025 EDA TIF).
_LOW_DENSITY_ZONES = frozenset({"R1", "R1A", "R1B", "R1T", "R1U", "R2", "R2U"})

# Mixed-use FAR capacity splitting: fraction reserved for residential.
# CB/CBW typicals: ground-floor retail (15-25% of FAR) + upper-floor
# residential (75-85%).  70/30 is a conservative mid-range split.
MIXED_USE_RES_FAR_SHARE = 0.70

# Construction/occupancy lag: multi-year delivery schedule.
# Real-world: entitlement 1-2yr + construction 2-3yr + lease-up 0.5-1yr = 3-5yr.
# Each entry is the fraction of development occupied at that year offset:
#   Year 0 (same year):  0% (entitlement/design begins)
#   Year +1:             0% (under construction)
#   Year +2:            33% (first units lease up)
#   Year +3:            67% (most units occupied)
#   Year +4:           100% (fully occupied)
# Incremental delivery per year = schedule[i] - schedule[i-1].
OCCUPANCY_SCHEDULE = (0.0, 0.0, 0.33, 0.67, 1.0)

# Height-dependent lease-up: small buildings fill faster, tall buildings fill
# slower and stabilize below 100% (NMHC: Class A MF in secondary markets
# stabilizes at 93-95%).
# Key: max stories for that tier.
OCCUPANCY_SCHEDULES = {
    4:   (0.0, 0.0, 0.50, 0.85, 1.00),              # low-rise (wood frame)
    12:  (0.0, 0.0, 0.25, 0.55, 0.85, 0.95),         # mid-rise (podium/concrete)
    999: (0.0, 0.0, 0.15, 0.40, 0.65, 0.85, 0.92),   # high-rise (steel)
}

# Commercial lease-up is slower than residential: office/retail in secondary
# markets like Greater Lafayette requires tenant buildout, lease negotiation,
# and certificate of occupancy per suite.  BOMA/CBRE secondary-market data
# shows 1-2 year lag relative to residential at the same height tier.
OCCUPANCY_SCHEDULES_COMMERCIAL = {
    4:   (0.0, 0.0, 0.30, 0.60, 0.85, 0.95),         # low-rise retail/office
    12:  (0.0, 0.0, 0.15, 0.35, 0.55, 0.75, 0.90),   # mid-rise office
    999: (0.0, 0.0, 0.10, 0.25, 0.45, 0.65, 0.80, 0.90),  # high-rise office tower
}


def get_occupancy_schedule(n_stories: int, use_type: str = "residential") -> tuple:
    """Return occupancy schedule for a building of the given height and use.

    Parameters
    ----------
    n_stories : int
        Estimated building height in stories.
    use_type : str
        ``"residential"`` (default) or ``"commercial"``.
    """
    schedules = (OCCUPANCY_SCHEDULES_COMMERCIAL
                 if use_type == "commercial"
                 else OCCUPANCY_SCHEDULES)
    for threshold in sorted(schedules.keys()):
        if n_stories <= threshold:
            return schedules[threshold]
    return schedules[999]


# Legacy constant for backward compatibility with 5-year steps.
# When step_years >= 5, the multi-year pipeline collapses to this single value
# (period-average of the schedule above = 40%).
CONSTRUCTION_OCCUPANCY_FRACTION = 0.40

# Self-selection (TCRP 128) removed.  The logit already computes mode
# share at each location, and catchment_scale grows trip opportunities
# with TOD development.  A separate self-selection multiplier would
# double-count the residential sorting that the development model and
# logit already capture.

# Maximum catchment growth multiplier.  Additive growth capped here to
# prevent extreme extrapolation when base catchment pop/jobs is small.
# 3.0 = catchment can triple over 25 years (consistent with aggressive
# TOD corridors like Portland MAX Yellow Line).
MAX_CATCHMENT_SCALE = 3.0

# Work-trip off-peak expansion (STOPS guidance, NHTS 2017 200-500K UZA).
# After x2 round trip, covers reverse commute, mid-day business, flex schedules.
# NHTS workers average 1.1 work trips/day (WFH, sick, vacation days).
# Round trip = 2.2; expansion over base round-trip = 2.2/2.0 = 1.10.
# Add 5% for mid-day meetings and off-site business -> 1.15.
# Non-work trips are handled separately by Component 1b (purpose-specific
# NHTS rates), eliminating double-counting with the old 3.0 factor.
WORK_OFFPEAK_EXPANSION = 1.15

# Student campus trips -- captures trips not in LODES (class, library,
# dining, recreation).  2.0 = total daily person-trips per student
# (before mode split).  Morgantown PRT validation: 12-16K daily riders
# with ~28K students → 0.4-0.5 APM trips/student/day, consistent with
# a 2.0 total trip rate and ~20-25% APM mode share.
STUDENT_CAMPUS_TRIP_RATE = 2.0    # total daily trips per campus-affiliated person
# STUDENT_APM_MODE_SHARE is now computed per-corridor via logit (see
# _compute_student_apm_share).  The representative campus trip distance
# and the student ASC adjustments drive the share; corridors far from
# campus or with poor headway get lower shares.
STUDENT_APM_SHARE_FLOOR = 0.05    # minimum *student* APM share (even far corridors get some campus trips)
# Note: commute mode share (Component 1a) has no floor — it is determined
# entirely by the 5-mode MNL in _compute_mode_shares().
STUDENT_APM_SHARE_CAP = 0.55      # Morgantown PRT ceiling — real-world max for campus APM
STUDENT_CAMPUS_TRIP_DIST_KM = 1.5  # representative intra-campus trip length

# Purpose-specific student trip rates (sum = 2.0, preserves calibration).
# Replaces single STUDENT_CAMPUS_TRIP_RATE with three sub-purposes that
# have different distances and coverage requirements.
STUDENT_TRIP_PURPOSES = {
    "home_to_campus": {
        "rate": 1.0,       # 1 round-trip/day (class commute)
        "dist_km": 2.5,    # housing clusters → campus core
        "needs_dest_check": False,  # campus_pop_catch already filters by station proximity
        # Note: this checks destination (campus) proximity, not origin (housing).
        # A fuller model would verify student housing clusters are near stations too.
    },
    "campus_to_offcampus": {
        "rate": 0.7,        # shopping, dining, social trips
        "dist_km": 1.5,     # campus → State St, Chauncey, etc.
        "needs_dest_check": True,   # destination must be near a station
    },
    "intracampus": {
        "rate": 0.3,        # library, cross-campus class changes
        "dist_km": 0.8,     # short campus trips
        "needs_dest_check": False,  # both ends on campus — checked by campus_pop_catch
    },
}

# Key student off-campus destinations (WGS84 lon/lat).
# Commercial clusters students frequent for shopping/dining/social.
STUDENT_OFFCAMPUS_DESTINATIONS = {
    "state_street":      (-86.9081, 40.4237),
    "chauncey_village":  (-86.9060, 40.4220),
    "wabash_landing":    (-86.8990, 40.4260),
    "sagamore_pkwy":     (-86.8850, 40.4170),
    "levee_plaza":       (-86.8930, 40.4210),
}

# Purpose-specific non-work trip generation rates (Component 1b).
# NHTS 2017 Table 15, urbanized areas 200K-500K population.
# ~93% of catchment commuters live near a station but work elsewhere.
# They can't commute via APM, but make local non-work trips that APM serves.
#
# Each purpose has its own 4-mode MNL with purpose-specific parameters:
#   rate: daily person-trips per person (home-based, workers)
#   dist_km: representative one-way trip distance (NHTS 2017 Table 26)
#   parking: destination parking cost (suburban retail=free, downtown=$2-4)
#   vot_mult: value-of-time multiplier vs commute (USDOT Revised
#     Departmental Guidance Table 4: local personal=0.50, all purposes=0.83;
#     NCHRP 716 Table 7-3: shopping 0.50, social 0.60, escort 0.40)
#   asc_transit_mult: transit ASC multiplier (non-habitual trips have lower
#     transit propensity; TCRP 95 Ch 9: non-work transit elasticity ~50-70%
#     of commute)
NONWORK_TRIP_PURPOSES = {
    "shopping":   {"rate": 0.65, "dist_km": 2.0, "parking": 0.0,
                   "vot_mult": 0.50, "asc_transit_mult": 0.70},
    "social_rec": {"rate": 0.45, "dist_km": 3.0, "parking": 0.0,
                   "vot_mult": 0.60, "asc_transit_mult": 0.55},
    "personal":   {"rate": 0.40, "dist_km": 2.5, "parking": 1.00,
                   "vot_mult": 0.50, "asc_transit_mult": 0.65},
    "escort":     {"rate": 0.30, "dist_km": 4.0, "parking": 0.0,
                   "vot_mult": 0.40, "asc_transit_mult": 0.30},
}
# Non-work trip capture fraction: what share of discretionary trips
# have both origin AND destination compatible with corridor service.
# FTA STOPS model: 30-50% of walk-zone population is the effective
# non-work transit market (accounting for: destinations outside corridor,
# trip chaining making transit infeasible, schedule constraints, mode
# inertia).  TCRP 95 Ch 9 Table 9-3: non-work transit trip rate =
# ~35% of total non-work trip rate for station-area residents.
NONWORK_TRIP_CAPTURE_FRACTION = 0.35

# Non-commute trip generators (Component 4).
# Major attractors not in LODES: medical, retail, event, visitor.
# Modeled as fixed daily trips attracted to institutional/commercial parcels
# within corridor catchment.
MEDICAL_DAILY_TRIPS = 350            # IU Health Arnett: ~350 visits/day (120K/yr)
RETAIL_TRIP_RATE = 0.5               # retail trips/job/day near stations
# Not all catchment jobs attract customer visits.  In a university town,
# ~25% education, ~12% manufacturing, ~8% government generate few walk-in
# trips.  Customer-facing sectors (retail trade NAICS 44-45, food services
# NAICS 72, personal services) are ~20% of Tippecanoe County employment
# (LODES WAC by NAICS sector).  Medical already has its own component.
RETAIL_JOB_FRACTION = 0.20           # fraction of catchment jobs that are retail/service
EVENT_ANNUAL_TRIPS = 200_000         # Purdue football/basketball, Loeb Playhouse, etc.
EVENT_DAILY_EQUIVALENT = 550         # 200K / 365 days
# Generator APM share is now computed via a 4-mode MNL (see
# _compute_generator_apm_share) instead of a fixed constant.
# Retained as fallback when spatial cache is missing.
NON_COMMUTE_GENERATOR_APM_SHARE_FALLBACK = 0.12
GENERATOR_PROXIMITY_DECAY_BETA = 0.0005  # exp decay for generator distance (half at ~1400m)

# Induced demand -- trip rate elasticity (TCRP Report 95, Ch. 15)
# Transit improvements induce new trips (not just mode shift).
# Literature: trip generation elasticity to accessibility 0.05-0.15.
# Much weaker than highway induced demand (~1.0, Duranton-Turner 2011).
# Applied to non-commute trips only (LODES commute OD pairs are fixed).
# Raised from 0.10 to 0.12: at 0.10 induced demand was ~2% of total
# ridership — low relative to FTA New Starts before/after studies
# showing 3-5% induced for new fixed-guideway.  0.12 is still within
# TCRP 95 range and produces ~2.5-3% induced share.
INDUCED_TRIP_ELASTICITY = 0.12      # TCRP 95 upper-mid range
INDUCED_DEMAND_THRESHOLD_YEAR = 5   # don't apply until system is established

# Latent demand -- zero-car households (ACS B08201)
# Tippecanoe County: 8.6% of HH have zero vehicles (ACS 2022).
# These HH are "transit captive" -- they suppress discretionary trips
# because alternatives are poor.  New high-quality transit (APM) releases
# some of these suppressed trips.
# Literature: NHTS 2009 via TCRP 161 mobility gap: zero-car HH make
# ~2.4 trips/HH/day vs ~4.5 for one-vehicle HH (per-person gap ~1.0).
# Blumenberg (2017): 79% of zero-car HH are "car-less" (constrained),
# 21% are "car-free" (by choice, no suppressed demand).
# Release rate 0.40: conservative -- zero-car HH distribute across
# transit, walking, biking; not all released trips go to APM.
ZERO_CAR_HH_SHARE = 0.076          # 9.67% ACS 2020-2024 B08201 × 0.79 car-less (Blumenberg 2017)
# Income-differentiated zero-car rates (ACS B08201 by earnings):
ZERO_CAR_BY_INCOME = {
    "SE01": 0.25,   # low-wage (<$1,250/mo): 25% zero-car
    "SE02": 0.06,   # mid-wage ($1,250-$3,333/mo): 6% zero-car
    "SE03": 0.02,   # high-wage (>$3,333/mo): 2% zero-car
}
ZERO_CAR_SUPPRESSED_TRIPS = 1.0    # fewer trips/person/day vs car-owning HH
LATENT_TRIP_RELEASE_RATE = 0.40    # fraction of suppressed trips that materialize
# Feeder-zone discount: zero-car HH in the feeder zone (1.2-7 km) must
# transfer to reach the APM.  Transfer penalty reduces effective mobility
# gain from APM, so fewer suppressed trips are released.  TCRP 95 Ch. 9:
# required transfer reduces ridership 30-50%.  Conservative 0.35 discount.
LATENT_FEEDER_RELEASE_DISCOUNT = 0.35

# Service-quality gate: latent demand materializes only when the corridor
# demonstrates destination-rich service (proxied by pre-latent ridership).
# Uses a higher target than MATURE_RIDERSHIP_TARGET (2,500, calibrated for
# bus restructuring) because latent trip release requires useful coverage
# of shopping, medical, and social destinations — not just any ridership.
LATENT_MATURITY_TARGET = 8_000    # riders/day for full latent release
LATENT_QUALITY_FLOOR = 0.10       # minimum quality: even weak corridors help

# Transit-eligible campus presence factors
# Not all 67K campus people generate APM-relevant trips:
#   Daily attendance: ~65% (35% absent on a given day)
#   Transit-eligible trip length: ~40% of campus trips > 400m
#   Faculty/staff: mostly drive+park, low transit share
STUDENT_PRESENCE_FACTOR = 0.25     # 25% of enrolled students
FACULTY_PRESENCE_FACTOR = 0.10     # 10% of faculty/staff

# Component-specific seasonal (annual-average) factors.
# Peak ridership occurs during academic year; annualization adjusts for
# summer/break periods where ridership is lower.
STUDENT_ANNUAL_FACTOR = (9.0 + 3.0 / 2.28) / 12.0  # 0.860 — academic calendar
COMMUTE_ANNUAL_FACTOR = 0.95        # slight summer dip (vacation, WFH)
NONWORK_ANNUAL_FACTOR = 0.92        # non-work travel dips more in summer/holidays
GENERATOR_ANNUAL_FACTOR = 0.90      # events seasonal, medical year-round
INDUCED_ANNUAL_FACTOR = 0.92        # follows generator/origin-only pattern
LATENT_ANNUAL_FACTOR = 0.95         # zero-car HH need transit year-round

# Fallback LODES income segment shares (ACS national average)
# Used when corridor-specific LODES data has zero trips
FALLBACK_INCOME_SHARES = {"SE01": 0.29, "SE02": 0.42, "SE03": 0.29}

# Congestion feedback on car travel time
# As TOD development adds population to a corridor catchment, local VMT
# increases and car travel times rise, making transit relatively more
# attractive.  This positive feedback loop is well-documented:
#   - TTI Urban Mobility Report (2023): small metros (200-500K pop),
#     travel time elasticity to VMT growth ≈ 0.2-0.5.
#   - Duranton & Turner (2011 AER): VMT grows ~1:1 with population.
# Conservative 0.30: 10% pop growth → 3% slower car travel.
# At 50% catchment pop growth (typical top corridor Year 25): car is 13% slower.
CONGESTION_ELASTICITY = 0.30

# Fraction of car congestion elasticity that applies to buses.
# Buses share road space but use fixed routes and some signal priority.
# TCRP 165: bus speeds degrade at ~50-70% of car-speed degradation rate.
BUS_CONGESTION_SHARE = 0.60

# Parking scarcity feedback (TCRP 128, Cervero 2004, Litman 2023,
# Lehner & Peer 2019 meta-analysis, VTPI Transport Elasticities 2025).
# As TOD densifies, surface parking redevelops and remaining supply tightens.
# Effective parking cost rises by ELASTICITY × developed_fraction.
# Spatially varying: suburban land values (~$200-800K/acre) are below the
# ~$3M/acre structured-parking threshold → slower conversion pressure.
# Campus has observed scarcity (Purdue garage permits 2.5× surface).
# Range in literature: 1.5-3.0.
PARKING_SCARCITY_ELASTICITY_SUBURBAN = 1.5   # abundant land, slow conversion
PARKING_SCARCITY_ELASTICITY_DOWNTOWN = 2.0   # mixed surface/structured
PARKING_SCARCITY_ELASTICITY_CAMPUS = 2.5     # constrained, permits sell out

# Period-specific congestion factors — calibrated via Google Routes API
# (scripts/calibrate_congestion_factors.py, 80 road segments, 8 departure times).
# Raw calibrated ratios (period_speed / off_peak_speed): AM=0.943, PM=0.939,
# OP=1.000.  Renormalized so the work-commute weighted average (0.45/0.35/0.20)
# equals 1.0, preserving CAR_SPEED_KPH = 30 as the effective all-day average.
# Raw norm factor = 0.9531; each factor = raw / 0.9531.
PEAK_AM_CONGESTION_FACTOR = 0.99   # AM peak: 0.9431/0.9531 = 0.9895
PEAK_PM_CONGESTION_FACTOR = 0.99   # PM peak: 0.9391/0.9531 = 0.9853
OFFPEAK_CONGESTION_FACTOR = 1.05   # Off-peak: 1.0000/0.9531 = 1.0492

# Global fallback — used when no per-corridor profile is available.
PERIOD_CONGESTION_FACTORS = {
    "am_peak": PEAK_AM_CONGESTION_FACTOR,
    "pm_peak": PEAK_PM_CONGESTION_FACTOR,
    "off_peak": OFFPEAK_CONGESTION_FACTOR,
}

# Per-corridor-group congestion profiles from calibration.  Each corridor's
# car-alternative route experiences different congestion levels depending on
# which roads it parallels.  Keyed by corridor group prefix (e.g. "C1").
# Corridors not matching any prefix use the global fallback above.
# Renormalized per-group so each group's work-commute weighted average = 1.0.
CORRIDOR_CONGESTION_PROFILES: dict[str, dict[str, float]] = {
    "C1": {"am_peak": 0.995, "pm_peak": 0.965, "off_peak": 1.072},
    "C2": {"am_peak": 1.007, "pm_peak": 0.972, "off_peak": 1.033},
    "C3": {"am_peak": 0.987, "pm_peak": 0.978, "off_peak": 1.067},
    "C4": {"am_peak": 1.003, "pm_peak": 0.969, "off_peak": 1.047},
    "C5": {"am_peak": 0.989, "pm_peak": 0.987, "off_peak": 1.046},
    "C6": {"am_peak": 0.987, "pm_peak": 0.986, "off_peak": 1.054},
}

# Component-to-period temporal distribution (NHTS 2017, metro 200-500K).
# Each component has a different time-of-day profile that determines how
# much of its ridership falls in each period.  Employment corridors benefit
# from peak congestion (higher APM mode share); campus corridors are
# midday-dominant (free-flow car → smaller APM advantage).
COMPONENT_PERIOD_WEIGHTS = {
    "work_commute":  {"am_peak": 0.45, "pm_peak": 0.35, "off_peak": 0.20},
    "local_nonwork": {"am_peak": 0.10, "pm_peak": 0.20, "off_peak": 0.70},
    "student":       {"am_peak": 0.30, "pm_peak": 0.15, "off_peak": 0.55},
    "generator":     {"am_peak": 0.15, "pm_peak": 0.20, "off_peak": 0.65},
    "induced":       {"am_peak": 0.10, "pm_peak": 0.20, "off_peak": 0.70},
    "latent":        {"am_peak": 0.15, "pm_peak": 0.20, "off_peak": 0.65},
}

_PERIODS = ("am_peak", "pm_peak", "off_peak")

# Bus restructuring
MATURE_RIDERSHIP_TARGET = 2500  # peer small-city APM; CityBus avg ~400/route/day
# Threshold-based restructuring: trigger a bus network redesign when
# ridership changes ≥5% since the last event.  Year 0 always triggers.
# At 5%, restructuring fires every 1-2 years during steady growth,
# producing smooth headway trajectories instead of long flat plateaus.
RESTRUCTURE_RIDERSHIP_THRESHOLD = 0.05
# NOTE: BASE_BUS_HEADWAY is set in land_use_transport_model.py because it
# depends on the runtime import of BUS_HEADWAY_MIN from generate_improved_ridership.
MAX_PARALLEL_BUS_HEADWAY = 60.0  # No agency runs 90-min parallel service alongside frequent rail
MIN_FEEDER_BUS_HEADWAY = 15.0
DEFAULT_BUS_COMPETITIVENESS = 0.50
DEFAULT_BUS_PRODUCTIVITY = 0.50
DEFAULT_BUS_MAX_FEEDER_HEADWAY = 30.0  # TCRP 165: >30min loses 70%+ of riders
DEFAULT_BUS_SERVICE_SPAN_HOURS = 18.0
DEFAULT_BUS_PARALLEL_ROUTE_EQUIV = 1.0
DEFAULT_BUS_FEEDER_ROUTE_EQUIV = 0.6
DEFAULT_BUS_SERVICE_HOUR_BUDGET_MULTIPLIER = 1.10

# Accessibility → rent premium (TCRP Report 128: 25-40% near transit)
# Empirical: parcel AV regression shows 93% premium at 0m, but rent
# premiums are smaller than AV premiums.  Use 40% (TCRP upper bound).
MAX_RENT_PREMIUM = 0.40  # 40% max rent uplift from accessibility
ANNUAL_STATION_RENT_GROWTH = 0.015  # ACS 2018-2022 Lafayette: ~3.6% nominal - 2.5% inflation ≈ 1.1% real + ~0.4% TOD premium

# Mode-specific rent premium ("permanence premium"): fixed-guideway
# transit (APM/rail) commands higher rents than BRT because permanent
# infrastructure signals long-term commitment, reducing developer risk
# and anchoring land-use expectations.
#
# Literature basis:
#   Cervero & Duncan 2002 (Santa Clara County):
#     - Commercial parcels within 0.25mi of LRT: ~23% land value premium
#     - Large apartments within 0.25mi of LRT: up to 45% land premium
#   Debrezion, Pels & Rietveld 2007 (meta-analysis, 73 studies):
#     - Rail residential premium: ~4.2% avg, commercial: ~16.4% avg
#   Mohammad, Graham & Melo 2013 (meta-analysis, 102 observations):
#     - Largest rail premiums at 500-800m from stations
#   Zhang & Yen 2020 (meta-analysis, 23 BRT studies):
#     - BRT residential premium: ~5% central estimate (range 2-8%)
#   FTA Report No. 0022 (developer interviews):
#     - Permanence of fixed infrastructure cited as key factor in
#       developer willingness to invest near transit
#
# Derivation of 12%:
#   Rail land-value premium over BRT: ~10-25% (Cervero & Duncan 2002)
#   Land value premiums capitalize future growth, so rent premiums are
#   ~60% of land-value premiums → 6-15% rent differential.
#   12% is the central estimate for a blended residential + commercial
#   context; conservative would be 8-10%, aggressive 15%.
#
# Applied as a multiplier on total adjusted rent (not just the transit
# premium portion, which was too weak at 2-3% effective differential).
FIXED_GUIDEWAY_RENT_MULT = 1.12   # APM/rail: 12% above BRT baseline
BRT_RENT_MULT = 1.05              # BRT: 5% premium (Cleveland HealthLine-class;
                                   # meta-analysis median ~4.3%, high-quality 7%)

# Convergence and run-control defaults (calibrated for annual steps)
# Annual changes are smaller than 5-year averages, so tolerances are tighter.
DEFAULT_RIDERSHIP_CONVERGENCE_TOL = 0.03   # 3% sufficient for corridor ranking
DEFAULT_DEVELOPMENT_CONVERGENCE_TOL = 0.05  # 5% sufficient for corridor ranking
DEFAULT_CONVERGENCE_FLOOR = 25.0
DEFAULT_ADAPTIVE_STOP = False
DEFAULT_MAX_TIME_STEPS = 100
DEFAULT_CONSECUTIVE_CONVERGED_STEPS = 3    # was 2 for 5-year steps
DEFAULT_STOP_ON_DIVERGENCE = False
DEFAULT_DIVERGENCE_THRESHOLD = 1.0
DEFAULT_CONSECUTIVE_DIVERGENT_STEPS = 2
CAPACITY_EPS = 1e-6

# Ridership scale multiplier: 1.0 means trust the LODES mode choice model
# directly.  Cross-validation against CityBus (Nov 2025, 5,982 daily
# boardings) shows scale=1.0 produces a 2.8% boarding rate = 4x bus,
# within the TCRP 3-8x range for fixed guideway in small metros.
# No arbitrary fudge factor needed — every factor in the chain has a
# traceable empirical source (LODES OD, trip rates, time-of-day).
RIDERSHIP_SCALE_MULTIPLIER_DEFAULT = 1.0

# CityBus validation baseline (Nov 2025 data)
# NOTE: 5,982 is the weekday average from CityBus internal ridership reports
# (15 fixed routes, excludes Purdue Inner Loop and demand-response).
# NTD system-wide figure (~9,000+) includes all modes, weekend service,
# and uses unlinked passenger trips (UPT) which double-counts transfers.
# The 5,982 figure is the correct benchmark for fixed-route weekday boardings.
CITYBUS_SYSTEM_DAILY_BOARDINGS = 5982   # 15 fixed routes, weekday average
CITYBUS_TOP_ROUTES = {                  # per-route daily boardings
    "4B_Purdue_West": 2997,
    "1B_Purdue_WL": 984,
    "6_Purdue_Salisbury": 634,
}
# Expected ratio: APM ridership should be 3-8x highest bus route (TCRP)
VALIDATION_APM_TO_BUS_RATIO_RANGE = (3.0, 8.0)
COMMUTE_DIRECTION_MIN_DEFAULT = 0.10
COMMUTE_DIRECTION_MAX_DEFAULT = 0.80


# ============================================================================
# Small pure helper functions (no class/self dependency)
# ============================================================================

def _resolve_congestion_profile(corridor_id: str) -> dict[str, float]:
    """Return the per-period congestion factors for a corridor.

    Matches the corridor_id against CORRIDOR_CONGESTION_PROFILES keys
    (prefix match, e.g. corridor "C1_campus_downtown" matches "C1").
    Falls back to global PERIOD_CONGESTION_FACTORS if no match.
    """
    for prefix, profile in CORRIDOR_CONGESTION_PROFILES.items():
        if corridor_id.startswith(prefix):
            return profile
    return PERIOD_CONGESTION_FACTORS


def infer_zone_code_from_assessor_class(class_code: object) -> str:
    """Infer a coarse zoning bucket from assessor-style class codes.

    This fallback is used when parcel zoning fields are unavailable in the
    input parcel file (e.g., sales-linked parcel extracts).
    """
    text = "".join(ch for ch in str(class_code) if ch.isdigit())
    if not text:
        return ""

    if text.startswith(("53", "54", "55")):
        return "R3"
    if text.startswith("52"):
        return "R2"
    if text.startswith(("50", "51")):
        return "R1"

    head = text[0]
    if head == "4":
        return "GB"    # DLGF 4xx = Commercial property → General Business zone
    if head == "6":
        return "I1"
    if head == "1":
        return "A"     # DLGF 1xx = Agricultural property → Agricultural zone
    return "R1"


def _phasing_gate(year: int, opening_delay_years: int = 0) -> bool:
    """Return True if the APM is operational at the given year.

    Parameters
    ----------
    year : current simulation year (0-based)
    opening_delay_years : years of construction delay before APM opens
    """
    return year >= opening_delay_years


def _clip_param(raw_value: object, default: float, lower: float, upper: float) -> float:
    """Cast and clamp a numeric config parameter to guardrail bounds."""
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = float(default)
    return float(np.clip(value, lower, upper))

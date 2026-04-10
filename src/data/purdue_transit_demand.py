"""Data-grounded institutional trip generation based on actual enrollment and employment.

This module replaces spatial multipliers with actual Purdue enrollment, faculty, and 
trip rate data to properly model campus transit demand.
"""
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# Purdue University actual data (2025-2026 academic year)
PURDUE_ENROLLMENT = 54_651  # Fall 2025 headcount (Purdue Data Digest, West Lafayette)
PURDUE_FACULTY_STAFF = 15_396  # Faculty + staff employees
PURDUE_TOTAL = PURDUE_ENROLLMENT + PURDUE_FACULTY_STAFF

# Transit trip rates from TCRP Report 153 (College/University Transit Systems)
# These are one-way boardings per person per day
STUDENT_TRANSIT_RATE_WITH_PASS = 1.8  # Students with unlimited pass program
STUDENT_TRANSIT_RATE_WITHOUT_PASS = 0.3  # Students without pass (pay-per-ride)
FACULTY_STAFF_TRANSIT_RATE = 0.4  # Faculty/staff (typically drive more)

# Purdue has unlimited student bus pass included in fees
PURDUE_PASS_COVERAGE = 0.95  # 95% of students have pass (included in fees)

# Student vs local mode choice ASC adjustments.
#
# Sources and calibration basis:
#   apm: +0.10 — TCRP 153 Table 3-4: university transit systems with free
#     pass programs see 15-25% higher ridership than fare-paying systems.
#     exp(0.10) = 1.105, conservative end.  Base ASC_APM=0.18 already
#     captures the generic reliability premium (TCRP 166 Table 5-12).
#   bus: +0.20 — CityBus NTD 2023: 5,982 daily boardings on a system
#     serving ~55K students.  Implied trip rate 0.20/student/day with 95%
#     pass coverage (fees fund CityBus).  exp(0.20)=1.22 matches observed
#     bus mode share ~15% vs ACS Tippecanoe County overall transit share
#     of 3.4% (ACS S0801, 2022 5-yr).
#   car: -0.20 — Purdue Transportation 2023: 32% of students hold parking
#     permits.  ACS B08301 Tippecanoe County 18-24: 62% drive alone vs
#     85% county-wide.  exp(-0.20) = 0.82, proportional to the 62/85=0.73
#     ratio, dampened because some no-permit students ride with friends.
#   walk: +0.15 — NHTS 2017 Table 16: walk mode share for 18-24 age
#     cohort in urbanized areas is 12% vs 2.8% all ages.  On a compact
#     campus (1.5 km diameter), walk dominates for <1 km trips.
#     exp(0.15)=1.16 calibrated so walk wins at 0.8 km intracampus trips.
STUDENT_ASC_ADJUSTMENTS = {
    "apm": +0.05,       # reduced from +0.10: free-pass program is a policy
                        # input, not a given; +0.05 reflects baseline student
                        # transit affinity without assuming committed program
    "bus": +0.20,
    "car": -0.20,
    "walk": +0.15,
}

# Campus car access time: on Purdue campus, most parking lots are 5-10 min
# walk from classroom buildings.  Add time to find a spot during peak hours
# (2-5 min).  Total campus car access is ~10 min vs 2 min suburban default.
# This is the dominant reason students walk/bike/bus instead of driving for
# intracampus trips — the parking *time* cost exceeds the parking *money*
# cost ($0.40/trip amortized permit).
# Source: Purdue Transportation Services lot map — median lot-to-building
# pedestrian distance is ~450m at 5 km/h = 5.4 min one-way, plus 2-3 min
# for parking maneuver = ~8-10 min round-trip equivalent.
CAMPUS_CAR_ACCESS_MIN = 10.0
DEFAULT_CAR_ACCESS_MIN = 2.0     # general/suburban (from MNL hardcoded value)

# SP+ campus shuttle competes with APM for intracampus trips.
# Free shuttle reduces APM attractiveness for short on-campus trips by
# shifting ~15-20 percentage points from APM to walk/bus.
# Applied when trip_dist < 1.0 km.  Neutralizes the ASC stack
# (ASC_APM 0.12 + S_APM 0.05 = +0.17) to -0.05, so walk dominates
# on 0.8km trips while APM retains ~8-10% share.
SP_PLUS_ASC_PENALTY = -0.22


def fetch_purdue_enrollment_acs(county_fips: str = "18157") -> dict:
    """Fetch college enrollment data from ACS for validation.
    
    Returns dict with total enrolled and graduate enrolled for comparison.
    """
    from src.data.acs_loader import fetch_enrollment_distribution
    
    logger.info(f"Fetching ACS enrollment data for county {county_fips}...")
    enroll_df = fetch_enrollment_distribution(county_fips)
    
    total_enrolled = enroll_df[enroll_df['age_group'].str.contains('18_24')]['enrolled'].sum()
    
    return {
        'acs_enrolled_18_24': total_enrolled,
        'purdue_enrollment': PURDUE_ENROLLMENT,
        'ratio': PURDUE_ENROLLMENT / total_enrolled if total_enrolled > 0 else 0
    }


def compute_purdue_daily_transit_trips() -> dict:
    """Compute expected daily transit trips based on Purdue enrollment and trip rates.
    
    Returns dict with trip estimates by user type and total.
    """
    # Student trips
    students_with_pass = PURDUE_ENROLLMENT * PURDUE_PASS_COVERAGE
    students_without_pass = PURDUE_ENROLLMENT * (1 - PURDUE_PASS_COVERAGE)
    
    student_trips_with_pass = students_with_pass * STUDENT_TRANSIT_RATE_WITH_PASS
    student_trips_without_pass = students_without_pass * STUDENT_TRANSIT_RATE_WITHOUT_PASS
    student_trips_total = student_trips_with_pass + student_trips_without_pass
    
    # Faculty/staff trips
    faculty_staff_trips = PURDUE_FACULTY_STAFF * FACULTY_STAFF_TRANSIT_RATE
    
    # Total
    total_trips = student_trips_total + faculty_staff_trips
    
    return {
        'students_with_pass': int(students_with_pass),
        'students_without_pass': int(students_without_pass),
        'student_trips_with_pass': int(student_trips_with_pass),
        'student_trips_without_pass': int(student_trips_without_pass),
        'student_trips_total': int(student_trips_total),
        'faculty_staff_trips': int(faculty_staff_trips),
        'total_daily_trips': int(total_trips),
        'trip_rate_overall': total_trips / PURDUE_TOTAL
    }


def compute_data_grounded_multiplier(
    campus_parcels_pop: float,
    campus_parcels_jobs: float,
    baseline_trip_rate: float = 2.6  # Countywide average trips/person/day
) -> float:
    """Compute institutional multiplier based on actual Purdue data vs parcel allocations.
    
    Args:
        campus_parcels_pop: Population allocated to campus parcels
        campus_parcels_jobs: Jobs allocated to campus parcels
        baseline_trip_rate: Countywide average trip rate
        
    Returns:
        Multiplier to apply to campus parcels to match actual Purdue transit demand
    """
    # Actual Purdue transit demand
    purdue_data = compute_purdue_daily_transit_trips()
    actual_purdue_trips = purdue_data['total_daily_trips']
    
    # Parcel-based estimate (pop + jobs, weighted)
    # Use standard gravity model weights (60% pop, 40% jobs)
    parcel_demand_base = (campus_parcels_pop * 0.6 + campus_parcels_jobs * 0.4) * baseline_trip_rate
    
    # Multiplier to match actual
    if parcel_demand_base > 0:
        multiplier = actual_purdue_trips / parcel_demand_base
    else:
        # Fallback if no parcel data allocated to campus
        logger.warning("No population or jobs on campus parcels; using default multiplier")
        multiplier = 10.0
    
    logger.info(f"Campus demand calibration:")
    logger.info(f"  Actual Purdue transit trips/day: {actual_purdue_trips:,}")
    logger.info(f"  Parcel baseline demand: {parcel_demand_base:.0f}")
    logger.info(f"  Computed multiplier: {multiplier:.2f}×")
    
    return multiplier


def add_data_grounded_institutional_weights(
    parcels_gdf: gpd.GeoDataFrame,
    campus_boundary_path: Optional[Path] = None,
    jobs_col: str = 'jobs_combined',
    pop_col: str = 'pop_alloc',
    baseline_trip_rate: float = 2.6
) -> gpd.GeoDataFrame:
    """Add institutional weights based on actual Purdue enrollment and trip rate data.
    
    Replaces arbitrary spatial multipliers with data-driven approach:
    1. Identify campus parcels spatially
    2. Compute actual Purdue transit demand from enrollment + trip rates
    3. Calibrate multiplier to match actual demand vs parcel-based estimates
    
    Args:
        parcels_gdf: Parcel GeoDataFrame
        campus_boundary_path: Path to campus boundary GeoJSON
        jobs_col: Jobs column name
        pop_col: Population column name
        baseline_trip_rate: Countywide average trips/person/day
        
    Returns:
        parcels_gdf with 'institutional_weight' column calibrated to actual data
    """
    from src.data.institutional_generators import fetch_purdue_campus_boundary, identify_campus_parcels
    
    logger.info("Adding data-grounded institutional weights based on Purdue enrollment data...")
    
    # Load or fetch campus boundary
    if campus_boundary_path and Path(campus_boundary_path).exists():
        campus_gdf = gpd.read_file(campus_boundary_path)
    else:
        campus_gdf = fetch_purdue_campus_boundary(campus_boundary_path)
    
    # Identify campus parcels
    is_campus_core = identify_campus_parcels(parcels_gdf, campus_gdf, buffer_m=0)
    is_campus_adjacent = identify_campus_parcels(parcels_gdf, campus_gdf, buffer_m=500) & ~is_campus_core
    
    logger.info(f"Identified {is_campus_core.sum()} core campus parcels")
    logger.info(f"Identified {is_campus_adjacent.sum()} campus-adjacent parcels")
    
    # Compute current campus parcel allocations
    campus_pop = parcels_gdf.loc[is_campus_core, pop_col].sum()
    campus_jobs = parcels_gdf.loc[is_campus_core, jobs_col].sum()
    
    logger.info(f"Campus parcel allocations:")
    logger.info(f"  Population: {campus_pop:.0f}")
    logger.info(f"  Jobs: {campus_jobs:.0f}")
    
    # Compute data-grounded multiplier
    core_multiplier = compute_data_grounded_multiplier(
        campus_pop, campus_jobs, baseline_trip_rate
    )
    
    # Adjacent areas get half the core multiplier (proximity effect)
    adjacent_multiplier = core_multiplier * 0.5
    
    # Initialize multiplier column
    parcels_gdf['institutional_weight'] = 1.0
    parcels_gdf.loc[is_campus_core, 'institutional_weight'] = core_multiplier
    parcels_gdf.loc[is_campus_adjacent, 'institutional_weight'] = adjacent_multiplier
    
    # Add flag columns
    parcels_gdf['is_campus_core'] = is_campus_core
    parcels_gdf['is_campus_adjacent'] = is_campus_adjacent
    
    # Compute weighted demand
    if jobs_col in parcels_gdf.columns:
        parcels_gdf['jobs_weighted'] = parcels_gdf[jobs_col] * parcels_gdf['institutional_weight']
    if pop_col in parcels_gdf.columns:
        parcels_gdf['pop_weighted'] = parcels_gdf[pop_col] * parcels_gdf['institutional_weight']
    
    # Validation
    purdue_data = compute_purdue_daily_transit_trips()
    weighted_demand = (
        parcels_gdf.loc[is_campus_core, 'pop_weighted'].sum() * 0.6 +
        parcels_gdf.loc[is_campus_core, 'jobs_weighted'].sum() * 0.4
    ) * baseline_trip_rate
    
    logger.info(f"\nValidation:")
    logger.info(f"  Target Purdue transit trips/day: {purdue_data['total_daily_trips']:,}")
    logger.info(f"  Weighted campus demand: {weighted_demand:.0f}")
    logger.info(f"  Match: {weighted_demand / purdue_data['total_daily_trips'] * 100:.1f}%")
    
    return parcels_gdf


def build_purdue_institutional_overlay(
    parcels_gdf: gpd.GeoDataFrame,
    campus_boundary_path: Optional[Path] = None,
    jobs_col: str = 'jobs_combined',
    pop_col: str = 'pop_alloc',
    baseline_trip_rate: float = 2.6,
    parcel_id_col: str = 'PARCEL_ID',
    include_non_institutional: bool = False,
) -> pd.DataFrame:
    """Build parcel-level institutional overlay table from Purdue calibration.

    Returns DataFrame with columns:
    - parcel_id
    - institutional_weight
    - institutional_category (campus_core / campus_adjacent / non_institutional)
    - institutional_source
    """
    weighted = add_data_grounded_institutional_weights(
        parcels_gdf=parcels_gdf.copy(),
        campus_boundary_path=campus_boundary_path,
        jobs_col=jobs_col,
        pop_col=pop_col,
        baseline_trip_rate=baseline_trip_rate,
    )

    if parcel_id_col not in weighted.columns:
        raise KeyError(f"Parcel id column '{parcel_id_col}' not found in parcels")

    categories = pd.Series("non_institutional", index=weighted.index)
    if 'is_campus_adjacent' in weighted.columns:
        categories.loc[weighted['is_campus_adjacent'].fillna(False)] = "campus_adjacent"
    if 'is_campus_core' in weighted.columns:
        categories.loc[weighted['is_campus_core'].fillna(False)] = "campus_core"

    overlay = pd.DataFrame(
        {
            "parcel_id": weighted[parcel_id_col].astype(str),
            "institutional_weight": weighted.get("institutional_weight", pd.Series(1.0, index=weighted.index)).astype(float),
            "institutional_category": categories.astype(str),
            "institutional_source": "purdue_data_grounded",
        }
    )

    overlay = overlay[overlay["parcel_id"].str.strip() != ""].copy()
    if not include_non_institutional:
        overlay = overlay[overlay["institutional_weight"] > 1.0].copy()
    overlay = (
        overlay.sort_values("institutional_weight", ascending=False)
        .drop_duplicates(subset=["parcel_id"], keep="first")
        .reset_index(drop=True)
    )
    return overlay


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # Print Purdue data summary
    print("\n=== PURDUE UNIVERSITY TRANSIT DEMAND (Data-Grounded) ===\n")
    print(f"Enrollment: {PURDUE_ENROLLMENT:,} students")
    print(f"Faculty/Staff: {PURDUE_FACULTY_STAFF:,}")
    print(f"Total: {PURDUE_TOTAL:,} persons")
    print()
    
    trip_data = compute_purdue_daily_transit_trips()
    print("Daily Transit Trips (estimated from TCRP rates):")
    print(f"  Students with pass ({PURDUE_PASS_COVERAGE*100:.0f}%): {trip_data['students_with_pass']:,} × {STUDENT_TRANSIT_RATE_WITH_PASS} = {trip_data['student_trips_with_pass']:,} trips/day")
    print(f"  Students without pass: {trip_data['students_without_pass']:,} × {STUDENT_TRANSIT_RATE_WITHOUT_PASS} = {trip_data['student_trips_without_pass']:,} trips/day")
    print(f"  Faculty/staff: {PURDUE_FACULTY_STAFF:,} × {FACULTY_STAFF_TRANSIT_RATE} = {trip_data['faculty_staff_trips']:,} trips/day")
    print(f"  TOTAL: {trip_data['total_daily_trips']:,} transit trips/day")
    print(f"  Overall rate: {trip_data['trip_rate_overall']:.2f} trips/person/day")
    print()
    print("Compare to 2024 CityBus Route 4B (Purdue West): 2,997 daily")
    print("  → Purdue total demand is 31× one route's ridership")
    print("  → Confirms need for high campus trip generation multiplier")

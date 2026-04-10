"""
Fetch ACS (American Community Survey) tables from the US Census API and
produce simplified distributions used by our synthetic population generator.

Currently implemented:
- Household income distribution (B19001) mapped to low/medium/high brackets

Usage:
  python -m src.data.acs_loader --state 18 --county 157 --year 2022 \
      --out data/processed/acs_overrides.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import requests


def _fetch_census_json(url: str, params: Dict[str, str]) -> List[List[str]]:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_income_distribution(state_fips: str, county_fips: str, year: str = "2022") -> Dict[str, float]:
    """Fetch income distribution from ACS5 B19001 and map to low/med/high.

    low: <$50k (B19001_002E..B19001_010E)
    medium: $50k-$100k (B19001_011E..B19001_013E)
    high: >$100k (B19001_014E..B19001_017E)
    """
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    vars_ = [
        "B19001_001E",  # total households
        *[f"B19001_{i:03d}E" for i in range(2, 18)],
    ]
    params = {
        "get": ",".join(vars_),
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}",
    }
    data = _fetch_census_json(base, params)
    header = data[0]
    values = data[1]
    row = dict(zip(header, values))

    total = int(row["B19001_001E"]) or 1
    # Sums per mapping
    low_sum = sum(int(row[f"B19001_{i:03d}E"]) for i in range(2, 11))
    med_sum = sum(int(row[f"B19001_{i:03d}E"]) for i in range(11, 14))
    high_sum = sum(int(row[f"B19001_{i:03d}E"]) for i in range(14, 18))

    return {
        "low": low_sum / total,
        "medium": med_sum / total,
        "high": high_sum / total,
    }


def fetch_household_size_distribution(state_fips: str, county_fips: str, year: str = "2022") -> Dict[int, float]:
    """Fetch household size distribution from ACS5 B11016 (Household Type by Size)."""
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    # B11016_002E..B11016_008E are size 1..7+
    vars_ = ["B11016_001E", *[f"B11016_{i:03d}E" for i in range(2, 9)]]
    params = {
        "get": ",".join(vars_),
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}",
    }
    data = _fetch_census_json(base, params)
    header = data[0]
    values = data[1]
    row = dict(zip(header, values))
    total = int(row["B11016_001E"]) or 1

    # Map 1..7+ into 1,2,3,4,5+
    size_counts = {
        1: int(row.get("B11016_002E", 0)),
        2: int(row.get("B11016_003E", 0)),
        3: int(row.get("B11016_004E", 0)),
        4: int(row.get("B11016_005E", 0)),
        5: int(row.get("B11016_006E", 0)) + int(row.get("B11016_007E", 0)) + int(row.get("B11016_008E", 0)),
    }
    return {k: v / total for k, v in size_counts.items()}


def fetch_vehicle_availability_distribution(state_fips: str, county_fips: str, year: str = "2022") -> Dict[int, float]:
    """Fetch household vehicle availability from ACS5 B08201.

    Returns a distribution over {0,1,2+} vehicles.
    """
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    # B08201_002E..B08201_010E counts households by vehicles available (0..7+)
    vars_ = ["B08201_001E", *[f"B08201_{i:03d}E" for i in range(2, 11)]]
    params = {
        "get": ",".join(vars_),
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}",
    }
    data = _fetch_census_json(base, params)
    header = data[0]
    values = data[1]
    row = dict(zip(header, values))
    total = int(row["B08201_001E"]) or 1

    zero = int(row.get("B08201_002E", 0))
    one = int(row.get("B08201_003E", 0))
    two_plus = sum(int(row.get(f"B08201_{i:03d}E", 0)) for i in range(4, 11))
    return {
        0: zero / total,
        1: one / total,
        2: two_plus / total,
    }


def fetch_age_distribution(state_fips: str, county_fips: str, year: str = "2022") -> Dict[str, float]:
    """Fetch age distribution from ACS5 B01001 (Sex by Age)."""
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    vars_ = ["B01001_001E"] + [f"B01001_{i:03d}E" for i in range(2, 26)]
    params = {
        "get": ",".join(vars_),
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}",
    }
    data = _fetch_census_json(base, params)
    header = data[0]
    values = data[1]
    row = dict(zip(header, values))
    total = int(row["B01001_001E"]) or 1

    # Simplified age bins - focus on 18-24 student age
    age_0_17 = sum(int(row.get(f"B01001_{i:03d}E", 0)) for i in [3,4,5,6,7,27,28,29,30,31])
    age_18_24 = sum(int(row.get(f"B01001_{i:03d}E", 0)) for i in [8,9,10,32,33,34])
    age_25_64 = sum(int(row.get(f"B01001_{i:03d}E", 0)) for i in range(11, 20)) + sum(int(row.get(f"B01001_{i:03d}E", 0)) for i in range(35, 44))
    age_65_plus = sum(int(row.get(f"B01001_{i:03d}E", 0)) for i in range(20, 26)) + sum(int(row.get(f"B01001_{i:03d}E", 0)) for i in range(44, 50))

    return {
        "age_0_17": age_0_17 / total,
        "age_18_24": age_18_24 / total,
        "age_25_64": age_25_64 / total,
        "age_65_plus": age_65_plus / total,
    }


def fetch_enrollment_distribution(state_fips: str, county_fips: str, year: str = "2022") -> Dict[str, float]:
    """Fetch school enrollment from ACS5 B14001."""
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    vars_ = ["B14001_001E", "B14001_002E"]
    params = {
        "get": ",".join(vars_),
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}",
    }
    data = _fetch_census_json(base, params)
    header = data[0]
    values = data[1]
    row = dict(zip(header, values))
    total = int(row["B14001_001E"]) or 1
    enrolled = int(row.get("B14001_002E", 0))
    return {"enrollment_rate": enrolled / total}


def write_overrides_json(out_path: Path, overrides: Dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fetch ACS distributions and write overrides JSON")
    p.add_argument("--state", required=True, help="State FIPS (e.g., 18)")
    p.add_argument("--county", required=True, help="County FIPS (e.g., 157)")
    p.add_argument("--year", default="2022", help="ACS year (default 2022)")
    p.add_argument("--out", default="data/processed/acs_overrides.json", help="Output path for overrides JSON")
    args = p.parse_args()

    # Fetch and build overrides
    income = fetch_income_distribution(args.state, args.county, args.year)
    hh_size = fetch_household_size_distribution(args.state, args.county, args.year)
    vehicles = fetch_vehicle_availability_distribution(args.state, args.county, args.year)
    age_dist = fetch_age_distribution(args.state, args.county, args.year)
    enrollment = fetch_enrollment_distribution(args.state, args.county, args.year)

    overrides = {
        "income_bracket": income,
        "household_size": hh_size,
        "vehicle_availability": vehicles,
        "age_distribution": age_dist,
        "enrollment": enrollment,
    }
    
    print(f"County population estimate: {age_dist}")
    print(f"Enrollment rate: {enrollment}")

    write_overrides_json(Path(args.out), overrides)
    print(f"Wrote ACS overrides to {args.out}")

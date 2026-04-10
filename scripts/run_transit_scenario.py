#!/usr/bin/env python
"""Run stop-level transit ridership scenario from parcel and stop files."""
import argparse

import geopandas as gpd

from src.transit_model import run_transit_scenario


def main() -> int:
    parser = argparse.ArgumentParser(description="Run transit ridership scenario.")
    parser.add_argument("--parcels", required=True, help="Input parcels GeoJSON path")
    parser.add_argument("--stops", required=True, help="Input stops GeoJSON path")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--jobs-col", default="jobs", help="Parcel jobs column name")
    parser.add_argument("--beta-walk", type=float, default=0.0024, help="Walk decay parameter")
    parser.add_argument("--max-walk-m", type=float, default=1000.0, help="Maximum walk distance")
    parser.add_argument("--walk-speed-kph", type=float, default=5.0, help="Walk speed")
    args = parser.parse_args()

    parcels = gpd.read_file(args.parcels)
    stops = gpd.read_file(args.stops)
    run_transit_scenario(
        parcels_gdf=parcels,
        stops_gdf=stops,
        out_path=args.out,
        jobs_col=args.jobs_col,
        beta_walk=args.beta_walk,
        max_walk_m=args.max_walk_m,
        walk_speed_kph=args.walk_speed_kph,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

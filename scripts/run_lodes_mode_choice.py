#!/usr/bin/env python
"""
LODES-Based Mode Choice Integration Script
==========================================

Integrates Census LODES commute flows with mode choice model to estimate
APM corridor ridership using real home-to-work patterns.

Usage:
    python scripts/run_lodes_mode_choice.py [--corridor CORRIDOR_ID]
    python scripts/run_lodes_mode_choice.py --corridor C1 --car-parking-cost 5.0
    python scripts/run_lodes_mode_choice.py --corridor C1 --parking-sensitivity --parking-costs 0,2.5,5,7.5
    python scripts/run_lodes_mode_choice.py --use-institutional-overlay
    python scripts/run_lodes_mode_choice.py --institutional-overlay data/processed/institutional_overlay.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from src.transit_model import load_lodes_for_mode_choice, DEFAULT_LODES_OD_PATH
from src.mode_choice import mode_split_for_od
from src.source_manifest import validate_source_manifest_file

import logging
logger = logging.getLogger(__name__)



# Time-of-day factors from improved_demand_model_v2.py
TIME_OF_DAY_FACTORS = {
    'am_peak': {'base': 1.40, 'SE01': 1.20, 'SE02': 1.45, 'SE03': 1.55},
    'pm_peak': {'base': 1.35, 'SE01': 1.15, 'SE02': 1.40, 'SE03': 1.50},
    'off_peak': {'base': 0.80, 'SE01': 0.95, 'SE02': 0.75, 'SE03': 0.65},
}

# Period distribution (fraction of daily trips in each period)
PERIOD_DISTRIBUTION = {
    'am_peak': 0.30,   # 30% of trips in AM peak (7-10am)
    'pm_peak': 0.25,   # 25% of trips in PM peak (4-7pm)
    'off_peak': 0.45,  # 45% of trips off-peak
}


def apply_time_of_day_factors(od_df: pd.DataFrame) -> pd.DataFrame:
    """Apply time-of-day demand factors based on LODES income segments.

    SE01 (low wage <$1250/mo): More off-peak (retail/service workers)
    SE02 (mid wage $1250-$3333/mo): Standard 9-5 peak pattern
    SE03 (high wage >$3333/mo): Stronger peak concentration

    Returns DataFrame with time_period column and adjusted trips.
    """
    if not all(col in od_df.columns for col in ['SE01', 'SE02', 'SE03']):
        # If no income data, use base factors
        logger.debug("  Warning: No income segment data. Using base time-of-day factors.")
        records = []
        for period, fraction in PERIOD_DISTRIBUTION.items():
            period_df = od_df.copy()
            period_df['time_period'] = period
            period_df['period_trips'] = period_df['trips'] * fraction * TIME_OF_DAY_FACTORS[period]['base']
            records.append(period_df)
        return pd.concat(records, ignore_index=True)

    # Weight factors by income segment
    total_jobs = od_df['SE01'] + od_df['SE02'] + od_df['SE03'] + 0.001

    records = []
    for period, fraction in PERIOD_DISTRIBUTION.items():
        period_df = od_df.copy()
        period_df['time_period'] = period

        # Income-weighted factor
        factors = TIME_OF_DAY_FACTORS[period]
        weighted_factor = (
            od_df['SE01'] * factors['SE01'] +
            od_df['SE02'] * factors['SE02'] +
            od_df['SE03'] * factors['SE03']
        ) / total_jobs

        period_df['period_factor'] = weighted_factor
        period_df['period_trips'] = period_df['trips'] * fraction * weighted_factor
        records.append(period_df)

    return pd.concat(records, ignore_index=True)


def load_corridor_stops(corridor_id: str, corridors_path: Path = None) -> gpd.GeoDataFrame:
    """Load stop geometries for a specific corridor.

    If the corridor layer stores line geometries, derive evenly spaced stop points.
    """
    if corridors_path is None:
        corridors_path = Path("data/processed/apm_phase2a_corridors.geojson")

    if not corridors_path.exists():
        raise FileNotFoundError(f"Corridors file not found: {corridors_path}")

    corridors = gpd.read_file(corridors_path)
    corridor = corridors[corridors['corridor_id'] == corridor_id].copy()

    if corridor.empty:
        available = corridors['corridor_id'].tolist()
        raise ValueError(f"Corridor {corridor_id} not found. Available: {available[:5]}...")

    geom = corridor.iloc[0].geometry
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "Point":
        out = corridor.copy()
        out["stop_id"] = [f"{corridor_id}_S1"]
        return out[["stop_id", "geometry"]]

    if geom_type in {"LineString", "MultiLineString"}:
        n_stops_raw = corridor.iloc[0].get("n_stops", 0)
        try:
            n_stops = max(int(n_stops_raw), 2)
        except Exception:
            n_stops = 8

        # For multilines, use a merged representative line via boundary-to-boundary interpolation.
        line = geom
        if geom_type == "MultiLineString":
            # Use longest component for stable stop spacing.
            line = max(list(geom.geoms), key=lambda g: g.length)

        distances = np.linspace(0.0, 1.0, n_stops)
        points = [line.interpolate(float(d), normalized=True) for d in distances]
        return gpd.GeoDataFrame(
            {
                "stop_id": [f"{corridor_id}_S{i+1}" for i in range(len(points))],
                "sequence": list(range(1, len(points) + 1)),
            },
            geometry=points,
            crs=corridor.crs,
        )

    raise ValueError(f"Unsupported corridor geometry type for stop extraction: {geom_type}")


def parse_parking_costs(parking_costs: str) -> list[float]:
    """Parse comma-separated parking costs into a validated float list."""
    values = []
    for token in parking_costs.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value < 0:
            raise ValueError(f"Parking cost cannot be negative: {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one parking cost is required")
    return values


def run_mode_choice_for_corridor(
    corridor_id: str,
    parcels_gdf: gpd.GeoDataFrame,
    lodes_od_path: Path = DEFAULT_LODES_OD_PATH,
    apm_headway_min: float = 5.0,
    bus_headway_min: float = 30.0,
    apm_fare: float = 2.0,   # CityBus 2026 integrated fare
    bus_fare: float = 2.0,   # CityBus 2026 fare
    car_parking_cost: float = 0.0,
    apply_tod: bool = True,
    use_institutional_overlay: bool = False,
    institutional_overlay_path: Path = None,
    institutional_strength: float = 1.0,
    institutional_max_multiplier: float = 5.0,
) -> dict:
    """Run mode choice model for a corridor using LODES OD data.

    Returns dict with ridership metrics.
    """
    logger.info(f"\nProcessing corridor: {corridor_id}")

    # Load corridor stops
    corridor_stops = load_corridor_stops(corridor_id)
    logger.debug(f"  Loaded corridor geometry")

    # Load LODES data
    od_df = load_lodes_for_mode_choice(
        lodes_od_path,
        min_trips=0.01,
        apply_institutional_overlay=use_institutional_overlay,
        parcels_gdf=parcels_gdf,
        institutional_overlay_path=institutional_overlay_path,
        institutional_strength=institutional_strength,
        institutional_max_multiplier=institutional_max_multiplier,
    )
    logger.debug(f"  Loaded {len(od_df):,} OD flows from LODES")

    # Apply time-of-day factors if requested
    if apply_tod:
        od_df = apply_time_of_day_factors(od_df)
        trips_col = 'period_trips'
        logger.debug(f"  Applied time-of-day factors: {len(od_df):,} period-OD pairs")
    else:
        trips_col = 'trips'

    # Ensure trips column exists
    if trips_col not in od_df.columns:
        od_df[trips_col] = od_df['trips']

    # Ensure a single canonical trips column for mode_split_for_od.
    od_for_mode = od_df.copy()
    if trips_col != "trips":
        od_for_mode["trips"] = pd.to_numeric(od_for_mode[trips_col], errors="coerce").fillna(0.0)
    else:
        od_for_mode["trips"] = pd.to_numeric(od_for_mode["trips"], errors="coerce").fillna(0.0)

    # Run mode split
    # Note: For full implementation, need bus stops GeoDataFrame
    result_df = mode_split_for_od(
        od_df=od_for_mode,
        parcels_gdf=parcels_gdf,
        bus_stops_gdf=None,  # TODO: Load bus stops
        apm_stops_gdf=corridor_stops,
        apm_headway_min=apm_headway_min,
        bus_headway_min=bus_headway_min,
        apm_fare=apm_fare,
        bus_fare=bus_fare,
        car_parking_cost=car_parking_cost,
    )

    # Aggregate results
    total_trips = result_df['trips'].sum()
    apm_trips = result_df['apm_trips'].sum()
    apm_share = apm_trips / total_trips if total_trips > 0 else 0

    metrics = {
        'corridor_id': corridor_id,
        'total_od_trips': total_trips,
        'apm_trips': apm_trips,
        'apm_share': apm_share,
        'bus_trips': result_df['bus_trips'].sum(),
        'car_trips': result_df['car_trips'].sum(),
        'walk_trips': result_df['walk_trips'].sum(),
        'n_od_pairs': len(od_df),
        'car_parking_cost': car_parking_cost,
    }

    if "trips_base" in result_df.columns:
        base_total = float(result_df["trips_base"].sum())
        delta_total = float(result_df.get("trips_institutional_delta", pd.Series(dtype=float)).sum())
        pair_count = int(result_df.get("institutional_pair_flag", pd.Series(dtype=bool)).sum())
        metrics["institutional_overlay_applied"] = bool(
            result_df.get("institutional_overlay_applied", pd.Series(dtype=bool)).any()
        )
        metrics["institutional_overlay_source"] = (
            result_df.get("institutional_overlay_source", pd.Series(["none"])).iloc[0]
            if len(result_df) > 0 else "none"
        )
        metrics["base_total_od_trips"] = base_total
        metrics["institutional_extra_trips"] = delta_total
        metrics["institutional_extra_trip_pct"] = (delta_total / base_total) if base_total > 0 else 0.0
        metrics["institutional_od_pairs"] = pair_count
        logger.debug(
            f"  Institutional overlay: +{delta_total:,.0f} trips "
            f"({metrics['institutional_extra_trip_pct']*100:.1f}%), pairs={pair_count:,}"
        )
    else:
        metrics["institutional_overlay_applied"] = False
        metrics["institutional_overlay_source"] = "none"
        metrics["base_total_od_trips"] = total_trips
        metrics["institutional_extra_trips"] = 0.0
        metrics["institutional_extra_trip_pct"] = 0.0
        metrics["institutional_od_pairs"] = 0

    logger.debug(
        f"  Results: {apm_trips:,.0f} APM trips ({apm_share*100:.1f}% share), "
        f"parking=${car_parking_cost:.2f}"
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run LODES-based mode choice analysis")
    parser.add_argument("--corridor", type=str, help="Corridor ID to analyze (default: all)")
    parser.add_argument("--output", type=str, default="data/processed/lodes_mode_choice_results.csv",
                        help="Output CSV path")
    parser.add_argument("--no-tod", action="store_true", help="Disable time-of-day factors")
    parser.add_argument("--car-parking-cost", type=float, default=0.0,
                        help="Per-trip parking cost added to car mode utility")
    parser.add_argument("--parking-sensitivity", action="store_true",
                        help="Run sensitivity sweep across parking costs")
    parser.add_argument("--parking-costs", type=str, default="0,2.5,5.0,7.5",
                        help="Comma-separated parking costs for sensitivity sweep")
    parser.add_argument("--use-institutional-overlay", action="store_true",
                        help="Apply institutional demand overlay during LODES loading")
    parser.add_argument("--institutional-overlay", type=str, default=None,
                        help="Optional overlay CSV/GeoJSON path for parcel-level institutional weights")
    parser.add_argument("--institutional-strength", type=float, default=1.0,
                        help="Overlay blend strength (0=no effect, 1=full effect)")
    parser.add_argument("--institutional-max-multiplier", type=float, default=5.0,
                        help="Hard cap on institutional OD trip multiplier")
    parser.add_argument("--source-manifest", type=str, default="data/processed/source_manifest.csv",
                        help="Source manifest CSV path")
    parser.add_argument("--max-source-age-days", type=int, default=3650,
                        help="Upper bound on allowed source age in days")
    parser.add_argument("--skip-source-manifest-validation", action="store_true",
                        help="Skip source manifest validation before running")
    args = parser.parse_args()

    if not args.skip_source_manifest_validation:
        validate_source_manifest_file(
            args.source_manifest,
            max_source_age_days=args.max_source_age_days,
        )
        logger.info(f"Source manifest validated: {args.source_manifest}")

    logger.info("=" * 70)
    logger.info(f"Car parking cost: ${args.car_parking_cost:.2f}")
    if args.parking_sensitivity:
        logger.info(f"Parking sensitivity costs: {args.parking_costs}")
    logger.info(f"Institutional overlay: {'ON' if args.use_institutional_overlay else 'OFF'}")
    if args.institutional_overlay:
        logger.info(f"Institutional overlay path: {args.institutional_overlay}")
    logger.info("=" * 70)
    logger.info("LODES-Based Mode Choice Analysis")
    logger.info("=" * 70)

    # Load parcels (support multiple known artifact names)
    parcel_candidates = [
        Path("data/processed/parcels_enriched.geojson"),
        Path("data/processed/parcels_enriched_final.geojson"),
        Path("data/processed/parcels_enriched_with_access_test2.geojson"),
        Path("data/processed/parcels_clean.geojson"),
    ]
    parcels_path = next((p for p in parcel_candidates if p.exists()), None)
    if parcels_path is None:
        logger.info(
            "Error: Parcels file not found. Expected one of: "
            + ", ".join(str(p) for p in parcel_candidates)
        )
        sys.exit(1)

    logger.info(f"\nLoading parcels from {parcels_path}...")
    parcels_gdf = gpd.read_file(parcels_path)
    logger.debug(f"  Loaded {len(parcels_gdf):,} parcels")

    # Load corridors
    corridors_path = Path("data/processed/apm_phase2a_corridors.geojson")
    if not corridors_path.exists():
        logger.info(f"Error: Corridors file not found at {corridors_path}")
        sys.exit(1)

    corridors = gpd.read_file(corridors_path)
    corridor_ids = corridors['corridor_id'].tolist()

    # Filter to specific corridor if requested
    if args.corridor:
        if args.corridor not in corridor_ids:
            logger.info(f"Error: Corridor {args.corridor} not found")
            logger.info(f"Available: {corridor_ids[:10]}...")
            sys.exit(1)
        corridor_ids = [args.corridor]

    logger.info(f"\nProcessing {len(corridor_ids)} corridor(s)...")

    # Run analysis for each corridor
    results = []
    if args.parking_sensitivity:
        try:
            parking_costs = parse_parking_costs(args.parking_costs)
        except ValueError as exc:
            logger.info(f"Error: {exc}")
            sys.exit(1)

        for corridor_id in corridor_ids:
            for parking_cost in parking_costs:
                try:
                    metrics = run_mode_choice_for_corridor(
                        corridor_id=corridor_id,
                        parcels_gdf=parcels_gdf,
                        apply_tod=not args.no_tod,
                        car_parking_cost=parking_cost,
                        use_institutional_overlay=args.use_institutional_overlay,
                        institutional_overlay_path=Path(args.institutional_overlay) if args.institutional_overlay else None,
                        institutional_strength=args.institutional_strength,
                        institutional_max_multiplier=args.institutional_max_multiplier,
                    )
                    results.append(metrics)
                except Exception as e:
                    logger.debug(f"  Error processing {corridor_id} (parking={parking_cost}): {e}")
                    continue
    else:
        for corridor_id in corridor_ids:
            try:
                metrics = run_mode_choice_for_corridor(
                    corridor_id=corridor_id,
                    parcels_gdf=parcels_gdf,
                    apply_tod=not args.no_tod,
                    car_parking_cost=args.car_parking_cost,
                    use_institutional_overlay=args.use_institutional_overlay,
                    institutional_overlay_path=Path(args.institutional_overlay) if args.institutional_overlay else None,
                    institutional_strength=args.institutional_strength,
                    institutional_max_multiplier=args.institutional_max_multiplier,
                )
                results.append(metrics)
            except Exception as e:
                logger.debug(f"  Error processing {corridor_id}: {e}")
                continue

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"{'=' * 70}")

        # Summary
        logger.info(f"\nSummary:")
        logger.debug(f"  Corridors analyzed: {len(results)}")
        logger.debug(f"  Total APM trips: {results_df['apm_trips'].sum():,.0f}")
        logger.debug(f"  Avg APM share: {results_df['apm_share'].mean()*100:.1f}%")
        logger.debug(f"  Best corridor: {results_df.loc[results_df['apm_trips'].idxmax(), 'corridor_id']}")
        if "institutional_extra_trips" in results_df.columns:
            extra = float(results_df["institutional_extra_trips"].sum())
            base = float(results_df.get("base_total_od_trips", pd.Series([0.0])).sum())
            pct = (extra / base * 100) if base > 0 else 0.0
            logger.debug(f"  Institutional overlay contribution: +{extra:,.0f} trips ({pct:.2f}% of base)")
        if args.parking_sensitivity and "car_parking_cost" in results_df.columns:
            logger.info("\nParking sensitivity summary:")
            sweep = (
                results_df.groupby("car_parking_cost")
                .agg(apm_trips=("apm_trips", "sum"), apm_share=("apm_share", "mean"))
                .reset_index()
                .sort_values("car_parking_cost")
            )
            logger.info(sweep.to_string(index=False))
    else:
        logger.info("\nNo results to save.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

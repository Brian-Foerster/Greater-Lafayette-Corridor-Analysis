"""Shared helper to regenerate parcels_enriched_final.geojson when missing.

Both optimized_corridor_search.py and run_feedback_loop.py need the enriched
parcels file with zone assignment, population, and employment allocation.
This module centralises that logic so changes only need to happen in one place.
"""
import logging
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_enriched_parcels(
    target: Path,
    raw_dir: Path = Path("data/raw"),
    proc_dir: Path = Path("data/processed"),
) -> Path:
    """Return *target* if it exists; otherwise regenerate from raw data.

    Regeneration includes:
    1. Zone spatial join (parcels × zones)
    2. Census-based population allocation (blocks × building floor area)
    3. LEHD WAC employment allocation + combination

    Raises ``FileNotFoundError`` with a diagnostic message if required
    input files are missing.
    """
    if target.exists():
        return target

    raw_parcels = raw_dir / "parcels.geojson"
    raw_zones = raw_dir / "zones.geojson"
    if not raw_parcels.exists() or not raw_zones.exists():
        raise FileNotFoundError(
            f"Cannot regenerate {target.name}: need {raw_parcels} and {raw_zones}"
        )

    logger.info(f"  {target.name} not found — regenerating from raw data...")
    parcels = gpd.read_file(raw_parcels)

    def _norm(x):
        if pd.isna(x):
            return None
        digits = re.sub(r"[^0-9]", "", str(x).strip())
        return digits if digits else None

    parcels["PARCEL_ID"] = parcels["STKEY"].apply(_norm)
    zones = gpd.read_file(raw_zones)
    if parcels.crs != zones.crs:
        zones = zones.to_crs(parcels.crs)

    centroids = parcels.copy()
    centroids["geometry"] = centroids.geometry.centroid
    joined = gpd.sjoin(
        centroids, zones[["RefName", "geometry"]], how="left", predicate="within"
    )
    joined = joined.drop_duplicates(subset=["PIN"], keep="first")
    joined["geometry"] = parcels.loc[joined.index, "geometry"].values
    joined = joined.rename(columns={"RefName": "zone_code"})
    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined = joined[joined.geometry.notna() & ~joined.geometry.is_empty].copy()

    # --- Population and employment allocation ---
    blocks_shp = raw_dir / "tl_2020_18_tabblock20" / "tl_2020_18_tabblock20.shp"
    density_csv = raw_dir / "zoning_density_estimates.csv"
    wac_geojson = proc_dir / "tippecanoe_jobs_wac.geojson"
    improvements = raw_dir / "improvements_av.geojson"

    if not blocks_shp.exists() or not density_csv.exists() or not wac_geojson.exists():
        raise FileNotFoundError(
            f"Cannot allocate population/jobs for {target.name}. Need:\n"
            f"  Census blocks: {blocks_shp} (exists={blocks_shp.exists()})\n"
            f"  Density table: {density_csv} (exists={density_csv.exists()})\n"
            f"  WAC blocks:    {wac_geojson} (exists={wac_geojson.exists()})\n"
            "Run: python scripts/allocate_pop_and_employment.py with required args"
        )

    from src.zoning_population import (
        load_census_blocks,
        load_density_table,
        load_building_area,
        allocate_population_and_jobs,
    )
    from src.allocate_employment import (
        allocate_wac_jobs_to_parcels,
        combine_employment_sources,
    )

    blocks = load_census_blocks(blocks_shp)
    density_df = load_density_table(density_csv)
    wac_blocks = gpd.read_file(wac_geojson)
    building_area_df = load_building_area(improvements) if improvements.exists() else None

    # Drop zone_code before allocation — allocate_population_and_jobs does its
    # own sjoin with zones and renames the result to zone_code.
    joined = joined.drop(columns=["zone_code"], errors="ignore")

    logger.info("  Allocating population (census blocks + building floor area)...")
    joined = allocate_population_and_jobs(
        joined, zones, blocks, density_df,
        zoning_field="RefName", building_area_df=building_area_df,
    )
    logger.info(f"    pop_alloc total: {joined['pop_alloc'].sum():.0f}")

    logger.info("  Allocating LEHD WAC employment...")
    joined = allocate_wac_jobs_to_parcels(joined, wac_blocks, building_area_df=building_area_df)
    joined = combine_employment_sources(joined, lehd_weight=0.7, pop_weight=0.3)
    logger.info(f"    jobs_combined total: {joined['jobs_combined'].sum():.0f}")

    joined = joined.drop_duplicates(subset=["PARCEL_ID"], keep="first")
    target.parent.mkdir(parents=True, exist_ok=True)
    joined.to_file(str(target), driver="GeoJSON")
    logger.info(f"  Wrote {len(joined)} enriched parcels to {target.name}")
    return target

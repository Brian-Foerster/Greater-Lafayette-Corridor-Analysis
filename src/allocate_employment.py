"""Allocate LEHD WAC employment data to parcels using building floor area.

Uses FinishLivingArea as the primary weight for distributing block-level
employment to parcels. This ensures employment concentrates at parcels with
actual commercial/institutional buildings rather than empty land.
"""
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from typing import Optional
from src.spatial_constants import PROJECT_CRS


def allocate_wac_jobs_to_parcels(
    parcels_gdf: gpd.GeoDataFrame,
    wac_blocks_gdf: gpd.GeoDataFrame,
    building_area_df: Optional[pd.DataFrame] = None,
) -> gpd.GeoDataFrame:
    """Allocate WAC jobs from census blocks to parcels.

    Uses building floor area as primary weight when available, falling back
    to intersection area (capped) for parcels without building data.

    Args:
        parcels_gdf: GeoDataFrame with parcel geometries and STKEY_norm
        wac_blocks_gdf: GeoDataFrame with census blocks and 'jobs_wac' column
        building_area_df: output of load_building_area() with stkey_raw, total_fla

    Returns:
        parcels_gdf with new column 'jobs_lehd_wac'
    """
    parcels = parcels_gdf.to_crs(PROJECT_CRS).copy()
    wac = wac_blocks_gdf.to_crs(PROJECT_CRS).copy()

    # Join building FLA to parcels
    parcels["_fla"] = 0.0
    if building_area_df is not None and len(building_area_df) > 0:
        # Handle both raw parcels (STKEY) and enriched parcels (STKEY_norm)
        if "STKEY_norm" in parcels.columns:
            stkey_col = parcels["STKEY_norm"].astype(str).str.lstrip("ST")
        elif "STKEY" in parcels.columns:
            stkey_col = parcels["STKEY"].astype(str)
        else:
            stkey_col = pd.Series("", index=parcels.index)
        fla_map = building_area_df.set_index("stkey_raw")["total_fla"]
        parcels["_fla"] = stkey_col.map(fla_map).fillna(0.0)

    # Overlay: only keep intersection geometries
    inter = gpd.overlay(
        parcels[["PARCEL_ID", "_fla", "geometry"]],
        wac[["GEOID20", "jobs_wac", "geometry"]],
        how="intersection",
    )

    if inter.empty:
        result = parcels_gdf.copy()
        result["jobs_lehd_wac"] = 0.0
        return result

    # Intersection area
    inter["inter_area"] = inter.geometry.area

    # Weight: FLA if available, else capped intersection area
    # Cap fallback at median FLA to prevent large empty parcels from dominating
    median_fla = float(parcels.loc[parcels["_fla"] > 0, "_fla"].median()) if (parcels["_fla"] > 0).any() else 2000.0
    inter["weight"] = np.where(
        inter["_fla"] > 0,
        inter["_fla"],
        np.minimum(inter["inter_area"], median_fla),
    )

    # Allocate each block's jobs across parcels by weight share
    inter["block_weight_sum"] = inter.groupby("GEOID20")["weight"].transform("sum")
    inter["allocated"] = inter["jobs_wac"] * (
        inter["weight"] / inter["block_weight_sum"].clip(lower=1.0)
    )

    # Sum by parcel
    jobs_by_parcel = inter.groupby("PARCEL_ID")["allocated"].sum().reset_index()
    jobs_by_parcel.columns = ["PARCEL_ID", "jobs_lehd_wac"]

    # Merge back to original parcels
    result = parcels_gdf.copy()
    if "jobs_lehd_wac" in result.columns:
        result = result.drop(columns=["jobs_lehd_wac"])
    if not jobs_by_parcel.empty:
        result = result.merge(jobs_by_parcel, on="PARCEL_ID", how="left")
    if "jobs_lehd_wac" not in result.columns:
        result["jobs_lehd_wac"] = 0.0
    result["jobs_lehd_wac"] = result["jobs_lehd_wac"].fillna(0.0)

    return result


def combine_employment_sources(
    parcels_gdf: gpd.GeoDataFrame,
    lehd_weight: float = 0.7,
    pop_weight: float = 0.3,
) -> gpd.GeoDataFrame:
    """Combine LEHD WAC jobs with population-estimated jobs.

    Args:
        parcels_gdf: GeoDataFrame with 'jobs_lehd_wac' and 'pop_alloc' columns
        lehd_weight: Weight for LEHD WAC employment (0.7 = 70% trust LEHD)
        pop_weight: Weight for population-estimated jobs (0.3 = 30% fallback)

    Returns:
        parcels_gdf with 'jobs_combined' column
    """
    parcels = parcels_gdf.copy()

    if "jobs_lehd_wac" not in parcels.columns:
        parcels["jobs_lehd_wac"] = 0.0
    if "pop_alloc" not in parcels.columns:
        parcels["pop_alloc"] = 0.0

    # Estimate jobs from population (assume ~0.5 jobs per person in labor force)
    parcels["jobs_from_pop"] = parcels["pop_alloc"] * 0.5

    # Combine
    parcels["jobs_combined"] = (
        lehd_weight * parcels["jobs_lehd_wac"]
        + pop_weight * parcels["jobs_from_pop"]
    )

    return parcels


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Allocate LEHD employment to parcels")
    p.add_argument("--parcels", required=True, help="Parcels GeoJSON")
    p.add_argument("--wac-blocks", required=True, help="WAC blocks GeoJSON with jobs_wac column")
    p.add_argument("--out", default="data/processed/parcels_with_employment.geojson")
    args = p.parse_args()

    parcels = gpd.read_file(args.parcels)
    wac_blocks = gpd.read_file(args.wac_blocks)

    print(f"Allocating {wac_blocks['jobs_wac'].sum():.0f} WAC jobs to {len(parcels)} parcels...")
    parcels = allocate_wac_jobs_to_parcels(parcels, wac_blocks)

    print(f"Combining employment sources...")
    parcels = combine_employment_sources(parcels)

    parcels.to_file(args.out, driver="GeoJSON")
    print(f"Wrote {len(parcels)} parcels to {args.out}")
    print(f"  Total LEHD WAC jobs: {parcels['jobs_lehd_wac'].sum():.0f}")
    print(f"  Total combined jobs: {parcels['jobs_combined'].sum():.0f}")

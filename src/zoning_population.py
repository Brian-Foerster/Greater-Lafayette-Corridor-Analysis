"""Allocate census population to parcels using building floor area.

Primary weight is FinishLivingArea from building improvements data, which
directly measures building capacity rather than theoretical zoning density.
This prevents large empty parcels (e.g., Purdue campus fields zoned A with
$0 AV) from absorbing thousands of people.

Group quarters population (dorms, institutions) is separated from household
population using HOUSING20 from Census blocks.

Inputs:
- Census blocks (2020) with POP20 and HOUSING20.
- Building improvements with FinishLivingArea (sqft) and STKEY.
- Parcel geometries with PARCEL_ID, STKEY_norm.
- Zoning polygons + density table as fallback for parcels without buildings.

Method:
1. Join improvements to parcels via STKEY (strip ST prefix) -> total_fla per parcel.
2. For each census block:
   a. Separate household pop (min(POP20, HOUSING20 * 2.39)) from GQ pop.
   b. Allocate household pop to parcels weighted by total_fla with iterative
      cap-and-redistribute (excess from capped parcels goes to uncapped ones).
   c. Allocate GQ pop to parcels weighted by total_fla. For no-building
      blocks (e.g., Purdue state property), use GQ-aware caps.
3. Physical cap: max pop per parcel = total_fla / 200 sqft (building),
   or GQ-scaled cap for no-building parcels in group-quarters blocks.
4. Unmatched blocks (no parcel centroids) assigned to nearest parcels.

Outputs:
- Parcels GeoDataFrame with pop_alloc, jobs_alloc, zone_code columns.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from src.spatial_constants import PROJECT_CRS

logger = logging.getLogger(__name__)

VACANCY_FACTOR = 1.0  # Census POP20 = actual residents; no vacancy discount needed
AVG_HOUSEHOLD_SIZE = 2.39
MIN_SQFT_PER_PERSON = 200  # Physical cap (accommodates dorms)
MAX_REDISTRIBUTE_ITERS = 10  # Cap-and-redistribute convergence limit


def load_census_blocks(blocks_path: Path, countyfp: str = "157") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(blocks_path)
    if "COUNTYFP20" in gdf.columns:
        gdf = gdf[gdf["COUNTYFP20"] == countyfp]
    if "POP20" in gdf.columns:
        gdf = gdf.rename(columns={"POP20": "population"})
    return gdf


def load_density_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={
            "Zone": "zone",
            "Population_density_people_per_sqkm": "pop_density",
            "Job_density_jobs_per_sqkm": "job_density",
        }
    )
    return df


def load_building_area(improvements_path: Path) -> pd.DataFrame:
    """Load building improvements and aggregate FinishLivingArea per parcel STKEY.

    Returns DataFrame with columns: stkey_raw, total_fla, building_count.
    The stkey_raw column matches parcel STKEY_norm with the 'ST' prefix stripped.
    """
    logger.info(f"  Loading building improvements from {improvements_path}...")
    imp = gpd.read_file(improvements_path)
    logger.info(f"    {len(imp)} improvement records")

    # Aggregate floor area by STKEY
    imp["stkey_raw"] = imp["STKEY"].astype(str)
    imp["fla"] = pd.to_numeric(imp["FinishLivingArea"], errors="coerce").fillna(0)

    agg = (
        imp.groupby("stkey_raw")
        .agg(total_fla=("fla", "sum"), building_count=("fla", "count"))
        .reset_index()
    )
    agg = agg[agg["total_fla"] > 0]
    logger.info(f"    {len(agg)} parcels with building data (total FLA: {agg['total_fla'].sum():,.0f} sqft)")

    return agg


def _choose_density(row_zone: str, density_df: pd.DataFrame) -> tuple[float, float]:
    """Return (pop_density, job_density) for a zone with PD fallback."""
    if row_zone in density_df["zone"].values:
        d = density_df.set_index("zone").loc[row_zone]
        return float(d.get("pop_density", 0) or 0), float(d.get("job_density", 0) or 0)
    if str(row_zone).startswith("PD"):
        if "PDMX" in density_df["zone"].values:
            d = density_df.set_index("zone").loc["PDMX"]
            return float(d.get("pop_density", 0) or 0), float(d.get("job_density", 0) or 0)
        pd_rows = density_df[density_df["zone"].str.startswith("PD")]
        if not pd_rows.empty:
            return float(pd_rows["pop_density"].median()), float(pd_rows["job_density"].median())
    return float(density_df["pop_density"].median()), float(density_df["job_density"].median())


def _cap_and_redistribute(pop_to_alloc: float, weights: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Allocate population with iterative cap-and-redistribute.

    When initial allocation exceeds a parcel's cap, the excess is
    redistributed proportionally to uncapped parcels. Repeats until
    all allocations are within caps or no capacity remains.
    """
    alloc = np.zeros_like(weights, dtype=float)
    remaining = pop_to_alloc
    available = np.ones(len(weights), dtype=bool)
    remaining_caps = caps.copy()

    for _ in range(MAX_REDISTRIBUTE_ITERS):
        if remaining < 0.5 or not available.any():
            break

        w = weights * available
        total_w = w.sum()
        if total_w <= 0:
            break

        shares = w / total_w
        proposed = remaining * shares

        # Find parcels that exceed their remaining cap
        over = proposed > remaining_caps
        if not over.any():
            # All fit within caps
            alloc += proposed
            remaining = 0
            break

        # Allocate capped parcels at their cap, redistribute the rest
        alloc[over] += remaining_caps[over]
        alloc[~over] += proposed[~over]
        excess = (proposed[over] - remaining_caps[over]).sum()

        # Remove capped parcels from future rounds
        available[over] = False
        remaining_caps[over] = 0
        remaining_caps[~over] -= proposed[~over]
        remaining_caps = np.maximum(remaining_caps, 0)
        remaining = excess

    return alloc


def allocate_population_and_jobs(
    parcels_gdf: gpd.GeoDataFrame,
    zoning_gdf: gpd.GeoDataFrame,
    blocks_gdf: gpd.GeoDataFrame,
    density_df: pd.DataFrame,
    zoning_field: str = "ZONE",
    av_col: Optional[str] = None,
    building_area_df: Optional[pd.DataFrame] = None,
) -> gpd.GeoDataFrame:
    """Allocate census population to parcels using building floor area weights.

    Uses FinishLivingArea as the primary weight. Falls back to capped
    area*density for parcels without building data. Separates group quarters
    from household population using HOUSING20.

    Args:
        parcels_gdf: parcels with geometry, PARCEL_ID, STKEY_norm.
        zoning_gdf: zoning polygons with zoning_field code.
        blocks_gdf: census blocks with 'population' and optionally 'HOUSING20'.
        density_df: density lookup table (fallback for parcels without buildings).
        zoning_field: column in zoning_gdf with zone code.
        av_col: optional column in parcels for fallback weighting.
        building_area_df: output of load_building_area() with stkey_raw, total_fla.
    Returns:
        parcels GeoDataFrame with pop_alloc, jobs_alloc, zone_code columns.
    """
    parcels = parcels_gdf.copy()

    # --- Ensure PARCEL_ID and STKEY columns exist ---
    if "PARCEL_ID" not in parcels.columns:
        if "STKEY" in parcels.columns:
            parcels["PARCEL_ID"] = "ST" + parcels["STKEY"].astype(str)
        else:
            parcels["PARCEL_ID"] = parcels.index.astype(str)

    # Normalize STKEY for building join: try STKEY_norm, then STKEY
    if "STKEY_norm" in parcels.columns:
        parcels["_stkey_raw"] = parcels["STKEY_norm"].astype(str).str.lstrip("ST")
    elif "STKEY" in parcels.columns:
        parcels["_stkey_raw"] = parcels["STKEY"].astype(str)
    else:
        parcels["_stkey_raw"] = ""

    # --- Attach zoning ---
    zone_cols = [zoning_field]
    zoning = zoning_gdf[zone_cols + ["geometry"]]
    parcels = gpd.sjoin(parcels, zoning, how="left", predicate="intersects")
    parcels = parcels.rename(columns={zoning_field: "zone_code"})

    # Drop spatial join duplicates (parcel touching multiple zones: keep first)
    n_before = len(parcels)
    parcels = parcels.drop_duplicates(subset=["PARCEL_ID"], keep="first")
    if "index_right" in parcels.columns:
        parcels = parcels.drop(columns=["index_right"])
    n_after = len(parcels)
    if n_before != n_after:
        logger.info(f"  Deduplicated zoning sjoin: {n_before} -> {n_after} parcels")

    # --- Compute area ---
    metric = parcels.to_crs(PROJECT_CRS)
    parcels["area_km2"] = metric.geometry.area / 1e6

    # --- Zoning-based density (fallback) ---
    pop_density_list = []
    job_density_list = []
    for z in parcels["zone_code"].fillna(""):
        pop_d, job_d = _choose_density(str(z), density_df)
        pop_density_list.append(pop_d)
        job_density_list.append(job_d)
    parcels["pop_density"] = pop_density_list
    parcels["job_density"] = job_density_list

    # --- Join building floor area ---
    parcels["total_fla"] = 0.0
    parcels["has_building"] = False

    if building_area_df is not None and len(building_area_df) > 0:
        fla_map = building_area_df.set_index("stkey_raw")["total_fla"]
        parcels["total_fla"] = parcels["_stkey_raw"].map(fla_map).fillna(0.0)
        parcels["has_building"] = parcels["total_fla"] > 0

        n_with = parcels["has_building"].sum()
        n_total = len(parcels)
        logger.info(f"  Building data joined: {n_with}/{n_total} parcels ({n_with/n_total*100:.1f}%)")
    else:
        logger.info("  WARNING: No building data provided; using area*density fallback for all parcels")

    # --- Compute weights ---
    # Fallback: parcels without buildings get a small constant weight (100 sqft)
    # so they only receive population when no buildings exist in the block.
    FALLBACK_FLA = 100.0  # Minimal weight for no-building parcels

    parcels["weight_pop"] = np.where(
        parcels["has_building"],
        parcels["total_fla"],
        FALLBACK_FLA,
    )
    parcels["weight_job"] = np.where(
        parcels["has_building"],
        parcels["total_fla"],
        FALLBACK_FLA,
    )

    # --- Physical capacity cap ---
    # Building parcels: capped by floor area / 200 sqft per person
    # No-building parcels: base cap of 50 (raised per-block for GQ-heavy blocks)
    parcels["max_pop_cap"] = np.where(
        parcels["has_building"],
        parcels["total_fla"] / MIN_SQFT_PER_PERSON,
        50.0,  # Base cap; raised dynamically for GQ blocks in allocation loop
    )

    # --- Prepare outputs ---
    parcels["pop_alloc"] = 0.0
    parcels["jobs_alloc"] = 0.0

    # --- Check if HOUSING20 available for GQ separation ---
    has_housing = "HOUSING20" in blocks_gdf.columns

    # --- Single bulk spatial join (much faster than per-block overlay) ---
    logger.info("  Performing spatial join of parcels to census blocks...")
    # Use parcel centroids for the join (faster and avoids split-parcel issues)
    parcels_metric = parcels.to_crs(PROJECT_CRS).copy()
    parcels["_centroid_x"] = parcels_metric.geometry.centroid.x
    parcels["_centroid_y"] = parcels_metric.geometry.centroid.y
    centroids = gpd.GeoDataFrame(
        parcels[["PARCEL_ID"]],
        geometry=gpd.points_from_xy(parcels["_centroid_x"], parcels["_centroid_y"], crs=PROJECT_CRS),
    )
    blocks_metric = blocks_gdf.to_crs(PROJECT_CRS)

    # Spatial join: assign each parcel centroid to the block it falls in
    joined = gpd.sjoin(centroids, blocks_metric, how="left", predicate="within")
    # Merge block data back to parcels
    block_cols = ["population"]
    if has_housing:
        block_cols.append("HOUSING20")
    if "GEOID20" in blocks_metric.columns:
        block_cols.append("GEOID20")

    parcels["_block_pop"] = 0.0
    parcels["_block_housing"] = 0.0
    parcels["_block_id"] = ""

    for col in block_cols:
        if col in joined.columns:
            parcels[f"_block_{col}"] = joined[col].values

    # Handle column mapping
    parcels["_block_pop"] = joined["population"].fillna(0).values if "population" in joined.columns else 0.0
    if has_housing and "HOUSING20" in joined.columns:
        parcels["_block_housing"] = joined["HOUSING20"].fillna(0).values
    if "GEOID20" in joined.columns:
        parcels["_block_id"] = joined["GEOID20"].fillna("").values

    logger.info(f"  Parcels matched to blocks: {(parcels['_block_pop'] > 0).sum()}")

    # --- Identify unmatched blocks (handled after main allocation) ---
    blocks_metric_with_pop = blocks_metric[blocks_metric["population"] > 0].copy()
    matched_block_ids = set(parcels[parcels["_block_id"] != ""]["_block_id"].unique())
    unmatched = blocks_metric_with_pop[~blocks_metric_with_pop["GEOID20"].isin(matched_block_ids)]
    if len(unmatched) > 0:
        logger.info(f"  Unmatched blocks: {len(unmatched)} ({unmatched['population'].sum():,.0f} pop) - will assign post-allocation")

    # --- Allocate per block group with cap-and-redistribute ---
    populated_blocks = parcels[parcels["_block_pop"] > 0]["_block_id"].unique()
    logger.info(f"  Processing {len(populated_blocks)} populated blocks...")

    pop_alloc = pd.Series(0.0, index=parcels.index)
    jobs_alloc = pd.Series(0.0, index=parcels.index)

    for block_id in populated_blocks:
        mask = parcels["_block_id"] == block_id
        block_parcels = parcels.loc[mask]

        if len(block_parcels) == 0:
            continue

        pop = block_parcels.iloc[0]["_block_pop"]
        housing = block_parcels.iloc[0]["_block_housing"] if has_housing else pop / AVG_HOUSEHOLD_SIZE

        # Separate GQ from household
        household_pop = min(pop, housing * AVG_HOUSEHOLD_SIZE)
        gq_pop = pop - household_pop

        # --- Block-level capacity calibration ---
        # If total FLA-based capacity in the block is less than the census
        # population, scale up caps so the block can absorb its census total.
        # This handles dense student housing where actual sqft/person < 200.
        n_parcels = len(block_parcels)
        n_no_bldg = (~block_parcels["has_building"]).sum()
        bldg_idx = block_parcels.index[block_parcels["has_building"]]
        no_bldg_idx = block_parcels.index[~block_parcels["has_building"]]

        total_block_fla = block_parcels["total_fla"].sum()
        base_bldg_cap = total_block_fla / MIN_SQFT_PER_PERSON
        base_nobldg_cap = n_no_bldg * 50.0
        base_capacity = base_bldg_cap + base_nobldg_cap
        needed = pop * VACANCY_FACTOR

        if base_capacity > 0 and needed > base_capacity:
            # Scale building caps first
            if len(bldg_idx) > 0 and base_bldg_cap > 0:
                bldg_scale = needed / base_capacity
                parcels.loc[bldg_idx, "max_pop_cap"] *= bldg_scale
                # Recheck: is scaled building cap + base no-bldg cap enough?
                new_bldg_cap = base_bldg_cap * bldg_scale
                if new_bldg_cap + base_nobldg_cap < needed and n_no_bldg > 0:
                    # Also scale no-building caps
                    shortfall = needed - new_bldg_cap
                    nobldg_per = max(50.0, shortfall / n_no_bldg * 1.05)
                    parcels.loc[no_bldg_idx, "max_pop_cap"] = np.maximum(
                        parcels.loc[no_bldg_idx, "max_pop_cap"], nobldg_per
                    )
            elif n_no_bldg > 0:
                # All no-building: scale to fit census pop
                per_parcel = max(50.0, needed / n_no_bldg * 1.05)
                parcels.loc[no_bldg_idx, "max_pop_cap"] = np.maximum(
                    parcels.loc[no_bldg_idx, "max_pop_cap"], per_parcel
                )

        # GQ-aware cap: ensure no-building parcels can absorb GQ population
        if gq_pop > 50 and n_no_bldg > 0:
            gq_cap_per_parcel = max(50.0, (gq_pop * VACANCY_FACTOR / n_no_bldg) * 1.05)
            parcels.loc[no_bldg_idx, "max_pop_cap"] = np.maximum(
                parcels.loc[no_bldg_idx, "max_pop_cap"], gq_cap_per_parcel
            )

        # Household allocation with cap-and-redistribute
        weights = block_parcels["weight_pop"].values
        caps = parcels.loc[mask, "max_pop_cap"].values
        if weights.sum() > 0 and household_pop > 0:
            hh_alloc = _cap_and_redistribute(
                household_pop * VACANCY_FACTOR, weights, caps
            )
            pop_alloc.loc[mask] += hh_alloc
            # Reduce remaining cap for GQ allocation
            caps = caps - hh_alloc

        # GQ allocation with cap-and-redistribute
        if gq_pop > 0:
            bldg_mask = mask & parcels["has_building"]
            if bldg_mask.any():
                gq_weights = parcels.loc[bldg_mask, "weight_pop"].values
                gq_caps = caps[block_parcels["has_building"].values]
                gq_alloc = _cap_and_redistribute(
                    gq_pop * VACANCY_FACTOR, gq_weights, gq_caps
                )
                pop_alloc.loc[bldg_mask] += gq_alloc
            else:
                # No buildings: distribute evenly with GQ-aware caps
                n = mask.sum()
                if n > 0:
                    even_weights = np.ones(n)
                    gq_caps_all = parcels.loc[mask, "max_pop_cap"].values - pop_alloc.loc[mask].values
                    gq_caps_all = np.maximum(gq_caps_all, 0)
                    gq_alloc = _cap_and_redistribute(
                        gq_pop * VACANCY_FACTOR, even_weights, gq_caps_all
                    )
                    pop_alloc.loc[mask] += gq_alloc

        # Jobs allocation
        job_weights = block_parcels["weight_job"].values
        job_total = job_weights.sum()
        if job_total > 0:
            job_shares = job_weights / job_total
            jobs_alloc.loc[mask] += pop * job_shares

    parcels["pop_alloc"] = pop_alloc
    parcels["jobs_alloc"] = jobs_alloc

    # --- Post-allocation: assign unmatched block population to nearest parcels ---
    if len(unmatched) > 0:
        parcel_centroids_metric = parcels_metric.geometry.centroid
        unmatched_pop_added = 0.0
        for _, block_row in unmatched.iterrows():
            bc = block_row.geometry.centroid
            dists = parcel_centroids_metric.distance(bc)
            # Find 5 nearest parcels and distribute proportionally by FLA
            nearest_k = dists.nsmallest(5).index
            k_weights = parcels.loc[nearest_k, "weight_pop"].values
            k_total = k_weights.sum()
            if k_total > 0:
                shares = k_weights / k_total
                add_pop = block_row["population"] * VACANCY_FACTOR * shares
                parcels.loc[nearest_k, "pop_alloc"] += add_pop
                unmatched_pop_added += add_pop.sum()
        logger.info(f"  Unmatched blocks: added {unmatched_pop_added:,.0f} pop to nearest parcels")

    # Clean up temp columns
    parcels = parcels.drop(columns=[c for c in parcels.columns if c.startswith("_")])

    logger.info(f"  Total pop allocated: {parcels['pop_alloc'].sum():,.0f}")
    logger.info(f"  Max pop on single parcel: {parcels['pop_alloc'].max():,.1f}")

    return parcels

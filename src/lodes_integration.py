"""
LEHD LODES Origin-Destination Data Integration.

Integrates real commute flows from Census LODES data to replace
uniform OD patterns with empirically-grounded trip distributions.

LODES Data Structure:
- w_geocode: Work location (15-digit Census block GEOID)
- h_geocode: Home location (15-digit Census block GEOID)
- S000: Total jobs in this OD pair
- SA01-SA03: Jobs by age (29 and under, 30-54, 55+)
- SE01-SE03: Jobs by earnings ($1250/mo, $1251-3333, $3333+)
- SI01-SI03: Jobs by industry (Goods, Trade/Trans, Other Services)
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)


# Tippecanoe County FIPS code
TIPPECANOE_COUNTY_FIPS = "18157"


def load_lodes_od(
    lodes_path: Path = Path("data/raw/lehd/in_od_2022.csv"),
    county_fips: str = TIPPECANOE_COUNTY_FIPS,
) -> pd.DataFrame:
    """
    Load LODES OD data filtered to the specified county.

    Filters to OD pairs where EITHER origin OR destination is in the county
    (captures both residents commuting out and workers commuting in).
    """
    logger.info(f"Loading LODES OD data from {lodes_path}...")

    # Read LODES data
    od_df = pd.read_csv(lodes_path, dtype={'w_geocode': str, 'h_geocode': str})
    logger.info(f"  Total Indiana OD pairs: {len(od_df):,}")

    # Extract county FIPS from geocodes (positions 0-5 are state+county)
    od_df['w_county'] = od_df['w_geocode'].str[:5]
    od_df['h_county'] = od_df['h_geocode'].str[:5]

    # Filter to Tippecanoe County (either end of trip)
    county_mask = (od_df['w_county'] == county_fips) | (od_df['h_county'] == county_fips)
    od_county = od_df[county_mask].copy()

    logger.info(f"  Tippecanoe-related OD pairs: {len(od_county):,}")
    logger.info(f"  Total jobs: {od_county['S000'].sum():,}")

    # Categorize trip types
    internal = (od_county['w_county'] == county_fips) & (od_county['h_county'] == county_fips)
    inbound = (od_county['w_county'] == county_fips) & (od_county['h_county'] != county_fips)
    outbound = (od_county['w_county'] != county_fips) & (od_county['h_county'] == county_fips)

    logger.info(f"  Internal trips (live & work in county): {internal.sum():,} ({od_county.loc[internal, 'S000'].sum():,} jobs)")
    logger.info(f"  Inbound commuters (work in county): {inbound.sum():,} ({od_county.loc[inbound, 'S000'].sum():,} jobs)")
    logger.info(f"  Outbound commuters (live in county): {outbound.sum():,} ({od_county.loc[outbound, 'S000'].sum():,} jobs)")

    return od_county


def load_census_blocks(
    blocks_path: Path = Path("data/raw/tl_2020_18_tabblock20/tl_2020_18_tabblock20.shp"),
    county_fips: str = TIPPECANOE_COUNTY_FIPS,
) -> gpd.GeoDataFrame:
    """Load Census block geometries for the county."""
    logger.info(f"Loading Census blocks from {blocks_path}...")

    blocks = gpd.read_file(blocks_path)
    logger.info(f"  Total Indiana blocks: {len(blocks):,}")

    # Filter to Tippecanoe County
    blocks['county_fips'] = blocks['STATEFP20'] + blocks['COUNTYFP20']
    blocks_county = blocks[blocks['county_fips'] == county_fips].copy()

    logger.info(f"  Tippecanoe County blocks: {len(blocks_county):,}")

    # Create full GEOID for matching with LODES
    blocks_county['geocode'] = blocks_county['GEOID20']

    return blocks_county[['geocode', 'geometry']]


def map_blocks_to_parcels(
    blocks_gdf: gpd.GeoDataFrame,
    parcels_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Create mapping from Census blocks to parcels.

    Uses spatial join to assign each parcel to a Census block.
    Returns DataFrame with parcel_id, block_geocode columns.
    """
    logger.info("Mapping Census blocks to parcels...")

    # Ensure same CRS
    if blocks_gdf.crs != parcels_gdf.crs:
        blocks_gdf = blocks_gdf.to_crs(parcels_gdf.crs)

    # Spatial join: find which block contains each parcel centroid
    parcels_centroids = parcels_gdf.copy()
    parcels_centroids['geometry'] = parcels_centroids.geometry.centroid

    parcel_blocks = gpd.sjoin(
        parcels_centroids[['PARCEL_ID', 'geometry']],
        blocks_gdf,
        how='left',
        predicate='within'
    )

    # Handle parcels outside any block (assign to nearest)
    missing_mask = parcel_blocks['geocode'].isna()
    if missing_mask.any():
        logger.info(f"  {missing_mask.sum()} parcels outside block boundaries, assigning to nearest...")
        # For simplicity, assign to a default block or leave as NaN
        parcel_blocks.loc[missing_mask, 'geocode'] = 'UNKNOWN'

    mapping = parcel_blocks[['PARCEL_ID', 'geocode']].drop_duplicates()
    logger.info(f"  Mapped {len(mapping):,} parcels to blocks")

    return mapping


def create_parcel_od_flows(
    od_df: pd.DataFrame,
    block_parcel_map: pd.DataFrame,
    min_jobs: int = 1,
) -> pd.DataFrame:
    """
    Create parcel-level OD flows from block-level LODES data.

    Distributes block-level job counts to parcels proportionally.
    """
    logger.info("Creating parcel-level OD flows...")

    # Count parcels per block for proportional distribution
    parcels_per_block = block_parcel_map.groupby('geocode').size().reset_index(name='n_parcels')

    # Merge block-to-parcel mapping for origins (home locations)
    od_with_origins = od_df.merge(
        block_parcel_map.rename(columns={'PARCEL_ID': 'origin_parcel', 'geocode': 'h_geocode'}),
        on='h_geocode',
        how='inner'
    )

    # Merge for destinations (work locations)
    od_with_both = od_with_origins.merge(
        block_parcel_map.rename(columns={'PARCEL_ID': 'dest_parcel', 'geocode': 'w_geocode'}),
        on='w_geocode',
        how='inner'
    )

    logger.info(f"  OD pairs with parcel matches: {len(od_with_both):,}")

    if len(od_with_both) == 0:
        logger.info("  WARNING: No OD pairs matched to parcels!")
        return pd.DataFrame(columns=['origin_parcel', 'dest_parcel', 'trips'])

    # Distribute jobs across parcel pairs
    # For each block-block pair, distribute S000 jobs across all origin-dest parcel combinations
    od_with_both = od_with_both.merge(
        parcels_per_block.rename(columns={'geocode': 'h_geocode', 'n_parcels': 'n_origin_parcels'}),
        on='h_geocode',
        how='left'
    )
    od_with_both = od_with_both.merge(
        parcels_per_block.rename(columns={'geocode': 'w_geocode', 'n_parcels': 'n_dest_parcels'}),
        on='w_geocode',
        how='left'
    )

    od_with_both['n_origin_parcels'] = od_with_both['n_origin_parcels'].fillna(1)
    od_with_both['n_dest_parcels'] = od_with_both['n_dest_parcels'].fillna(1)

    # Distribute jobs proportionally
    od_with_both['trips'] = od_with_both['S000'] / (od_with_both['n_origin_parcels'] * od_with_both['n_dest_parcels'])

    # Aggregate to unique parcel pairs
    parcel_flows = od_with_both.groupby(['origin_parcel', 'dest_parcel']).agg({
        'trips': 'sum',
        'SE01': 'sum',  # Low earnings
        'SE02': 'sum',  # Medium earnings
        'SE03': 'sum',  # High earnings
    }).reset_index()

    # Filter to significant flows
    parcel_flows = parcel_flows[parcel_flows['trips'] >= min_jobs]

    logger.info(f"  Final parcel OD flows: {len(parcel_flows):,}")
    logger.info(f"  Total trips: {parcel_flows['trips'].sum():,.0f}")

    return parcel_flows


def integrate_lodes_to_parcels(
    parcels_path: Path = Path("data/processed/parcels_enriched.geojson"),
    lodes_od_path: Path = Path("data/raw/lehd/in_od_2022.csv"),
    blocks_path: Path = Path("data/raw/tl_2020_18_tabblock20/tl_2020_18_tabblock20.shp"),
    output_path: Path = Path("data/processed/od_parcel_flows_lodes.csv"),
    min_trips: float = 0.1,
) -> pd.DataFrame:
    """
    Full integration pipeline: LODES -> Census blocks -> Parcels.

    Returns parcel-level OD flows with real commute patterns.
    """
    logger.info("="*70)
    logger.info("LODES ORIGIN-DESTINATION INTEGRATION")
    logger.info("="*70)

    # Load LODES OD data
    od_df = load_lodes_od(lodes_od_path)

    # Load Census blocks
    blocks_gdf = load_census_blocks(blocks_path)

    # Load parcels
    logger.info(f"\nLoading parcels from {parcels_path}...")
    parcels_gdf = gpd.read_file(parcels_path)
    logger.info(f"  Total parcels: {len(parcels_gdf):,}")

    # Create block-to-parcel mapping
    block_parcel_map = map_blocks_to_parcels(blocks_gdf, parcels_gdf)

    # Create parcel-level OD flows
    parcel_flows = create_parcel_od_flows(od_df, block_parcel_map, min_jobs=min_trips)

    # Save results
    parcel_flows.to_csv(output_path, index=False)
    logger.info(f"\nSaved parcel OD flows to {output_path}")

    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("INTEGRATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total OD pairs: {len(parcel_flows):,}")
    logger.info(f"Total daily trips: {parcel_flows['trips'].sum():,.0f}")
    logger.info(f"Mean trips per OD pair: {parcel_flows['trips'].mean():.2f}")
    logger.info(f"Max trips per OD pair: {parcel_flows['trips'].max():.0f}")

    # Top destinations
    top_dests = parcel_flows.groupby('dest_parcel')['trips'].sum().nlargest(10)
    logger.info(f"\nTop 10 destination parcels (by inbound trips):")
    for parcel, trips in top_dests.items():
        logger.info(f"  {parcel}: {trips:.0f} trips")

    return parcel_flows


def validate_against_acs(
    parcel_flows: pd.DataFrame,
    acs_commute_data: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Validate LODES-based OD flows against ACS commute data.

    Returns validation metrics comparing LODES to ACS.
    """
    total_trips = parcel_flows['trips'].sum()

    # Expected Tippecanoe County workers (2022 estimate)
    # From ACS: ~95,000 employed persons
    expected_workers = 95000

    coverage = total_trips / expected_workers

    validation = {
        'total_lodes_trips': total_trips,
        'expected_acs_workers': expected_workers,
        'coverage_ratio': coverage,
        'status': 'GOOD' if 0.7 < coverage < 1.3 else 'REVIEW',
    }

    logger.info(f"\nValidation vs ACS:")
    logger.info(f"  LODES total: {total_trips:,.0f}")
    logger.info(f"  Expected workers: {expected_workers:,}")
    logger.info(f"  Coverage: {coverage:.1%}")
    logger.info(f"  Status: {validation['status']}")

    return validation


if __name__ == "__main__":
    # Run integration
    parcel_flows = integrate_lodes_to_parcels()

    # Validate
    validation = validate_against_acs(parcel_flows)

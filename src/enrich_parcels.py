"""Enrich parcels with AV (assessed value) and sales data.

This script:
1. Loads cleaned parcels (parcels_clean.geojson or parcels_with_condo_flag.geojson)
2. Joins land_av, improvements_av, total_av layers by PARCEL_ID using explicit merge
3. Attempts to link sales records (2021) to parcels by PARCEL_ID or spatial join
4. Outputs enriched GeoJSON and CSV files ready for orca ingestion

Handles column overlaps by using suffixes and explicit indexing.
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

RAW_DIR = Path('data/raw')
OUT_DIR = Path('data/processed')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_av_layer(av_path, id_prop='PARCEL_ID'):
    """Load AV layer, normalize id column, extract value column."""
    gdf = gpd.read_file(str(av_path))
    # Detect id field
    candidates = ['PARCEL_ID', 'ParcelID', 'PARCELID', 'parcel_id', 'PIN', 'STKEY', 'StKeyFull', 'ParcelNo']
    id_field = None
    for c in candidates:
        if c in gdf.columns:
            id_field = c
            break
    if id_field is None:
        raise KeyError(f"Could not detect parcel id in {av_path}")

    # Normalize into canonical PARCEL_ID string keys compatible with normalized parcels
    def normalize_stkey_val(x):
        import re
        if pd.isna(x):
            return pd.NA
        s = str(x).upper()
        s = re.sub(r'[^A-Z0-9]', '', s)
        if s.isdigit():
            s = f"ST{s}"
        return s if s else pd.NA

    if id_field in ('STKEY', 'StKeyFull', 'ParcelNo'):
        gdf['PARCEL_ID'] = gdf[id_field].apply(normalize_stkey_val)
    else:
        # numeric id (e.g., PIN or ParcelID) -> prefix with PIN to ensure string keys
        gdf[id_field] = pd.to_numeric(gdf[id_field], errors='coerce')
        def pin_key(x):
            if pd.isna(x):
                return pd.NA
            return f"PIN{int(x)}"
        gdf['PARCEL_ID'] = gdf[id_field].apply(pin_key)
    
    # Find the numeric value column (not geometry, not parcel id)
    val_cols = [c for c in gdf.columns if c not in ('PARCEL_ID', 'geometry', 'PIN', 'OBJECTID', 'GlobalID', 'STKEY', 'Shape_Length', 'Shape_Area')]
    if not val_cols:
        raise KeyError(f"No value column found in {av_path}")
    
    val_col = val_cols[0]
    
    # Return just id and value (drop geometry)
    result = gdf[['PARCEL_ID', val_col]].drop_duplicates(subset=['PARCEL_ID']).copy()
    return result, val_col


def enrich_parcels_with_av(parcels_path, av_files_dict):
    """Load parcels and join AV layers.
    
    Args:
        parcels_path: path to parcels GeoJSON
        av_files_dict: dict of {av_name: av_path}, e.g., {'land_av': 'data/raw/land_av.geojson', ...}
    
    Returns:
        GeoDataFrame with enriched attributes
    """
    logger.info(f"Loading parcels from {parcels_path}")
    parcels = gpd.read_file(str(parcels_path))
    logger.info(f"  Loaded {len(parcels)} parcels")
    
    # Convert to DataFrame for attribute-only join (preserve geometry separately)
    parcels_df = parcels.drop(columns='geometry', errors='ignore')
    
    for av_name, av_path in av_files_dict.items():
        av_path = Path(av_path)
        if not av_path.exists():
            logger.info(f"  Skipping {av_name}: file not found")
            continue
        
        logger.info(f"Loading {av_name} from {av_path}")
        try:
            av_df, val_col = load_and_prepare_av_layer(av_path)
            logger.info(f"  Loaded {len(av_df)} AV records")
            
            # Rename value column to avoid overlap
            av_df = av_df.rename(columns={val_col: f'{av_name}_{val_col}'})
            
            # Merge on PARCEL_ID
            parcels_df = parcels_df.merge(av_df, on='PARCEL_ID', how='left')
            logger.info(f"  Joined {av_name}; new columns: {list(parcels_df.columns[-1:])}")
        except Exception as e:
            logger.info(f"  Error processing {av_name}: {e}")
            continue
    
    # Restore geometry
    # Add normalized STKEY column for provenance and mapping
    def normalize_stkey_val(x):
        import re
        if pd.isna(x):
            return pd.NA
        s = str(x).upper()
        s = re.sub(r'[^A-Z0-9]', '', s)
        if s.isdigit():
            s = f"ST{s}"
        return s if s else pd.NA

    base_stkey_col = None
    for candidate in ('STKEY', 'StKeyFull', 'ParcelNo'):
        if candidate in parcels_df.columns:
            base_stkey_col = candidate
            break

    if base_stkey_col:
        parcels_df['STKEY_norm'] = parcels_df[base_stkey_col].apply(normalize_stkey_val)
    else:
        parcels_df['STKEY_norm'] = pd.NA

    enriched = gpd.GeoDataFrame(parcels_df, geometry=parcels.geometry, crs=parcels.crs)
    return enriched


def link_sales_to_parcels(sales_path, parcels_gdf):
    """Load sales records and link to parcels by PARCEL_ID."""
    # Load sales (accept either a GeoDataFrame or a path)
    if isinstance(sales_path, gpd.GeoDataFrame):
        sales = sales_path.copy()
        logger.info(f"Loading sales from GeoDataFrame: {len(sales)} records")
    else:
        logger.info(f"Loading sales from {sales_path}")
        sales = gpd.read_file(str(sales_path))
        logger.info(f"  Loaded {len(sales)} sales records")

    # Try to detect id field in sales
    id_candidates = ['STKEY', 'StKeyFull', 'ParcelNo', 'ParcelID', 'PARCEL_ID', 'PARCELID', 'parcel_id', 'PIN']
    id_field = None
    for c in id_candidates:
        if c in sales.columns:
            id_field = c
            break

    def normalize_stkey_val(x):
        import re
        if pd.isna(x):
            return pd.NA
        s = str(x).upper()
        s = re.sub(r'[^A-Z0-9]', '', s)
        return s if s else pd.NA

    result = sales.copy()
    linked_total = 0

    if id_field in ('STKEY', 'StKeyFull', 'ParcelNo'):
        result['PARCEL_ID'] = result[id_field].apply(normalize_stkey_val)
        merged = result.merge(parcels_gdf[['PARCEL_ID']], on='PARCEL_ID', how='left', indicator=True)
        linked_total = merged['_merge'].eq('both').sum()
        logger.info(f"  Attribute merge via STKEY: {linked_total} records linked by STKEY")
        result = merged
    elif id_field in ('ParcelID', 'PARCEL_ID', 'PARCELID', 'parcel_id'):
        # numeric id — try attribute match to parcels.PIN
        result[id_field] = pd.to_numeric(result[id_field], errors='coerce')
        if 'PIN' in parcels_gdf.columns:
            merged = result.merge(parcels_gdf[['PARCEL_ID', 'PIN']], left_on=id_field, right_on='PIN', how='left')
            linked_total = merged['PARCEL_ID'].notna().sum()
            logger.info(f"  Attribute merge via numeric id->PIN: {linked_total} records linked")
            result = merged
        else:
            logger.info("  No PIN column in parcels to match numeric ids; falling back to spatial join")
    elif id_field == 'PIN':
        result['PIN'] = pd.to_numeric(result['PIN'], errors='coerce')
        merged = result.merge(parcels_gdf[['PARCEL_ID', 'PIN']], on='PIN', how='left')
        linked_total = merged['PARCEL_ID'].notna().sum()
        logger.info(f"  Attribute merge via PIN: {linked_total} records linked by PIN")
        result = merged
    else:
        logger.info(f"  No id field detected; will attempt spatial join only")

    # For any records still unlinked, attempt spatial join
    if 'PARCEL_ID' in result.columns:
        still = result[result['PARCEL_ID'].isna()].copy()
    else:
        still = result.copy()

    if len(still) > 0 and not still.geometry.is_empty.all() and not parcels_gdf.geometry.is_empty.all():
        if still.crs != parcels_gdf.crs:
            still = still.to_crs(parcels_gdf.crs)

        parcels_id_geom = parcels_gdf[['PARCEL_ID', 'geometry']].copy()
        sales_spatial = gpd.sjoin(still, parcels_id_geom, how='left', predicate='within')
        sales_spatial = sales_spatial.rename(columns={'PARCEL_ID_right': 'PARCEL_ID_spatial'})
        logger.info(f"  Spatial join: {sales_spatial['PARCEL_ID_spatial'].notna().sum()} sales linked by geometry")

        # If PARCEL_ID column exists, fill from spatial join where missing
        if 'PARCEL_ID' in result.columns:
            sales_spatial = sales_spatial.rename(columns={'PARCEL_ID_spatial': 'PARCEL_ID'})
            combined = pd.concat([result[result['PARCEL_ID'].notna()], sales_spatial], ignore_index=True)
        else:
            # no PARCEL_ID column existed; use spatial result's index
            combined = sales_spatial
    else:
        combined = result

    # final cleanup
    return combined


def main(parcels_path='data/processed/parcels_clean.geojson',
         av_files=None,
         sales_path='data/raw/sales_2021.geojson'):
    if av_files is None:
        av_files = {
            'land_av': RAW_DIR / 'land_av.geojson',
            'improvements_av': RAW_DIR / 'improvements_av.geojson',
            'total_av': RAW_DIR / 'total_av.geojson'
        }
    
    # Enrich parcels with AV
    enriched = enrich_parcels_with_av(parcels_path, av_files)
    
    # Write enriched parcels
    enriched.to_file(OUT_DIR / 'parcels_enriched.geojson', driver='GeoJSON')
    enriched.drop(columns='geometry', errors='ignore').to_csv(OUT_DIR / 'parcels_enriched.csv', index=False)
    logger.info(f"\nWrote enriched parcels: {OUT_DIR / 'parcels_enriched.geojson'}")
    logger.info(f"Columns: {list(enriched.columns)}")
    
    # Link sales
    if Path(sales_path).exists():
        sales_linked = link_sales_to_parcels(sales_path, enriched)
        sales_linked.to_file(OUT_DIR / 'sales_2021_linked.geojson', driver='GeoJSON')
        logger.info(f"\nWrote linked sales: {OUT_DIR / 'sales_2021_linked.geojson'}")
    
    return enriched


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Enrich parcels with AV and sales data')
    parser.add_argument('--parcels', default='data/processed/parcels_clean.geojson')
    parser.add_argument('--sales', default='data/raw/sales_2021.geojson')
    args = parser.parse_args()
    
    main(parcels_path=args.parcels, sales_path=args.sales)

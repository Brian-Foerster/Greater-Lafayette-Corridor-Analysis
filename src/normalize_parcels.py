"""Normalize parcels and related AV/sales layers for processing.

Operations performed:
- Detect parcel id column and coerce to integer (drops fractional .0)
- Extract duplicate parcel ids to `data/processed/parcels_duplicates.geojson`
- Extract parcels with missing geometry to `data/processed/parcels_missing_geometry.geojson`
- Keep cleaned unique parcels to `data/processed/parcels_clean.geojson` and attributes to CSV
- Attempt to join AV layers (`land_av`, `improvements_av`, `total_av`) to parcels by parcel id
- Attempt to spatially link sales to parcels when PARCEL_ID is missing and write `data/processed/sales_linked.geojson`

The script is defensive and will skip heavy operations if `geopandas` is not available.
"""
from pathlib import Path
import json
import sys
import logging

logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
    import pandas as pd
except Exception:
    gpd = None
    pd = None


DEFAULT_PARCELS = Path('data/raw/parcels.geojson')
OUT_DIR = Path('data/processed')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _detect_id_field(gdf, candidates=None):
    if candidates is None:
        # Prefer STKEY (tax key) when available because it has the highest coverage
        # Fallback to PIN then ParcelID
        candidates = ['STKEY', 'StKeyFull', 'ParcelNo', 'PIN', 'PARCEL_ID', 'ParcelID', 'PARCELID', 'parcel_id', 'parcelid', 'PID', 'PARCELID_NUM']
    for c in candidates:
        if c in gdf.columns:
            return c
    # fallback: try any column that looks like an id by name
    for c in gdf.columns:
        if 'parcel' in c.lower() and 'id' in c.lower():
            return c
    return None


def coerce_parcel_id(gdf, id_col):
    import numpy as np
    import re
    # convert to numeric and ensure integrality
    # If id_col looks like a string tax key (STKEY, StKeyFull, ParcelNo), normalize as string
    key_lower = id_col.lower()
    if 'stkey' in key_lower or 'parcelno' in key_lower:
        def normalize_key(x):
            if pd.isna(x):
                return pd.NA
            s = str(x).upper()
            # Strip "ST" prefix from synthetic person IDs before cleaning
            if s.startswith('ST'):
                s = s[2:]
            # remove non-alphanumeric characters
            s = re.sub(r'[^A-Z0-9]', '', s)
            return s if s else pd.NA

        gdf[id_col] = gdf[id_col].apply(normalize_key)
        missing = gdf[id_col].isna().sum()
        if missing > 0:
            logger.info(f"Warning: {missing} missing/invalid values in {id_col} after normalization; they will be set to NA")
        # normalize column name to PARCEL_ID (string key)
        gdf = gdf.rename(columns={id_col: 'PARCEL_ID'})
        return gdf

    # Otherwise attempt numeric coercion
    ser = pd.to_numeric(gdf[id_col], errors='coerce')
    # report non-numeric
    nonnum = ser[ser.isna()].shape[0]
    if nonnum > 0:
        logger.info(f"Warning: {nonnum} non-numeric values in {id_col}; they will be set to NaN")
    # check integrality safely (handle index alignment and NA)
    nonint_series = ser.dropna().apply(lambda x: not float(x).is_integer())
    nonint_count = int(nonint_series.sum()) if not nonint_series.empty else 0
    if nonint_count > 0:
        logger.info(f"Warning: {nonint_count} non-integral parcel ids found; they will be coerced by truncation")
    # For numeric ids (e.g., PIN), convert to a canonical string key to avoid mixed-type ids
    # e.g., PIN 111 -> 'PIN111'
    def num_to_key(x):
        if pd.isna(x):
            return pd.NA
        try:
            xi = int(x)
            return f"PIN{xi}"
        except Exception:
            return pd.NA

    gdf['PARCEL_ID'] = ser.apply(num_to_key)
    return gdf


def split_duplicates_and_missing(gdf):
    # missing geometries
    missing_geom = gdf[gdf.geometry.isnull()].copy()
    if not missing_geom.empty:
        missing_geom.to_file(OUT_DIR / 'parcels_missing_geometry.geojson', driver='GeoJSON')
        logger.info(f"Wrote {len(missing_geom)} features with missing geometry to parcels_missing_geometry.geojson")
    else:
        logger.info("No missing geometries found in parcels")

    # duplicates
    dup_mask = gdf.duplicated(subset=['PARCEL_ID'], keep=False)
    dupes = gdf[dup_mask].copy()
    if not dupes.empty:
        dupes.to_file(OUT_DIR / 'parcels_duplicates.geojson', driver='GeoJSON')
        logger.info(f"Wrote {len(dupes)} duplicate parcel features to parcels_duplicates.geojson")
    else:
        logger.info("No duplicate parcel ids found")

    # cleaned: drop duplicates keeping first, drop missing geometries
    cleaned = gdf[~dup_mask].copy()
    cleaned = cleaned[~cleaned.geometry.isnull()].copy()
    cleaned.to_file(OUT_DIR / 'parcels_clean.geojson', driver='GeoJSON')
    # write attributes CSV
    cleaned.drop(columns='geometry', errors='ignore').to_csv(OUT_DIR / 'parcels_clean.csv', index=False)
    logger.info(f"Wrote cleaned parcels: {len(cleaned)} features to parcels_clean.geojson and parcels_clean.csv")
    return cleaned


def join_av_layers(parcels_gdf, av_paths):
    # For each av layer, detect id and join on parcel id
    for av_path in av_paths:
        p = Path(av_path)
        if not p.exists():
            logger.info(f"AV file not found, skipping: {p}")
            continue
        ag = gpd.read_file(str(p))
        id_field = _detect_id_field(ag)
        if id_field is None:
            logger.info(f"Could not detect parcel id field in {p}; skipping join")
            continue
        ag[id_field] = pd.to_numeric(ag[id_field], errors='coerce').astype('Int64')
        ag = ag.rename(columns={id_field: 'PARCEL_ID'})
        # choose first numeric val column if multiple
        val_cols = [c for c in ag.columns if c.lower() not in ('parcel_id', 'geometry')]
        suffix = p.stem
        # pick a primary value column if present (e.g., LAND_AV)
        if len(val_cols) == 0:
            logger.info(f"No value columns found in {p}; skipping")
            continue
        val_col = val_cols[0]
        # reduce to id + val
        ag2 = ag[['PARCEL_ID', val_col]].drop_duplicates(subset=['PARCEL_ID'])
        ag2 = ag2.set_index('PARCEL_ID')
        # join (handle geometry carefully to avoid overlap)
        parcels_gdf = parcels_gdf.set_index('PARCEL_ID')
        # temporarily drop geometry if present to avoid column overlap
        geom_col = parcels_gdf.get('geometry', None)
        if geom_col is not None:
            parcels_gdf_work = parcels_gdf.drop(columns='geometry')
        else:
            parcels_gdf_work = parcels_gdf
        parcels_gdf_work = parcels_gdf_work.join(ag2, how='left')
        # restore geometry
        if geom_col is not None:
            parcels_gdf_work['geometry'] = geom_col
        parcels_gdf = parcels_gdf_work
        parcels_gdf = parcels_gdf.reset_index()
        # rename joined column to make sense
        newname = f"{suffix}_{val_col}"
        parcels_gdf = parcels_gdf.rename(columns={val_col: newname})
        logger.info(f"Joined {p.name} on PARCEL_ID as {newname}")
    # write enriched
    parcels_gdf.to_file(OUT_DIR / 'parcels_enriched.geojson', driver='GeoJSON')
    parcels_gdf.drop(columns='geometry', errors='ignore').to_csv(OUT_DIR / 'parcels_enriched.csv', index=False)
    logger.info(f"Wrote enriched parcels to parcels_enriched.geojson and parcels_enriched.csv")
    return parcels_gdf


def link_sales_to_parcels(sales_path, parcels_gdf):
    p = Path(sales_path)
    if not p.exists():
        logger.info(f"Sales file not found, skipping: {p}")
        return
    sg = gpd.read_file(str(p))
    # try to detect parcel id in sales
    id_field = _detect_id_field(sg)
    if id_field is not None:
        sg[id_field] = pd.to_numeric(sg[id_field], errors='coerce').astype('Int64')
        sg = sg.rename(columns={id_field: 'PARCEL_ID'})
        # count nulls
        nulls = sg['PARCEL_ID'].isna().sum()
        logger.info(f"Sales file {p.name} has {nulls} records with missing PARCEL_ID")
    else:
        logger.info("No PARCEL_ID field detected in sales; will spatially join by geometry if possible")
    # spatial join for missing parcel ids
    if 'PARCEL_ID' not in sg.columns or sg['PARCEL_ID'].isna().any():
        if sg.geometry.is_empty.all() or parcels_gdf.geometry.is_empty.all():
            logger.info("Skipping spatial join because geometries missing in sales or parcels")
        else:
            # ensure both in same crs
            if sg.crs != parcels_gdf.crs:
                sg = sg.to_crs(parcels_gdf.crs)
            joined = gpd.sjoin(sg, parcels_gdf[['PARCEL_ID', 'geometry']], how='left', predicate='within')
            # sjoin may produce index_right column
            if 'PARCEL_ID' not in joined.columns and 'index_right' in joined.columns:
                joined = joined.rename(columns={'index_right': 'PARCEL_ID'})
            # save
            joined.to_file(OUT_DIR / 'sales_linked.geojson', driver='GeoJSON')
            logger.info(f"Wrote sales linked file to sales_linked.geojson with {len(joined)} records")
    else:
        sg.to_file(OUT_DIR / 'sales_linked.geojson', driver='GeoJSON')
        logger.info(f"Sales had PARCEL_ID; wrote sales_linked.geojson with {len(sg)} records")


def main(parcels_path=DEFAULT_PARCELS, av_files=None, sales_file=None, force=False):
    if gpd is None or pd is None:
        raise RuntimeError("geopandas and pandas are required to run normalization")

    parcels_path = Path(parcels_path)
    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels file not found: {parcels_path}")

    logger.info(f"Reading parcels from {parcels_path}")
    gdf = gpd.read_file(str(parcels_path))

    id_field = _detect_id_field(gdf)
    if id_field is None:
        raise KeyError("Could not detect parcel id field in parcels file")
    logger.info(f"Detected parcel id field: {id_field}")

    gdf = coerce_parcel_id(gdf, id_field)

    # split duplicates and missing
    cleaned = split_duplicates_and_missing(gdf)

    # join AVs (skip for now; will be implemented with explicit suffix handling)
    if av_files:
        logger.info("Note: AV file joining skipped due to column overlap; will be addressed in next iteration")
        # av_list = [a.strip() for a in av_files.split(',') if a.strip()]
        # cleaned = join_av_layers(cleaned, av_list)

    # link sales (skip for now due to same column overlap issue)
    if sales_file:
        logger.info("Note: Sales file linking skipped due to potential column overlap; will be addressed in next iteration")
        # link_sales_to_parcels(sales_file, cleaned)

    # final validation
    from src.validate import validate_geojson
    try:
        validate_geojson(OUT_DIR / 'parcels_clean.geojson', id_prop='PARCEL_ID')
        logger.info("Final validation passed for cleaned parcels")
    except Exception as e:
        logger.info(f"Validation failed for cleaned parcels: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Normalize parcels and join AV/sales')
    parser.add_argument('--parcels', default=str(DEFAULT_PARCELS))
    parser.add_argument('--av-files', default=None, help='Comma-separated AV layer paths')
    parser.add_argument('--sales-file', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    main(parcels_path=args.parcels, av_files=args.av_files, sales_file=args.sales_file, force=args.force)

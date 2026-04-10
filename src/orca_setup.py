#!/usr/bin/env python
"""
Register parcels and sales tables as orca-compatible tables.
Provides setup functions for model estimation pipeline.
"""
import argparse
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

# Access helpers
from src.build_network import compute_nearest_node_distances, compute_travel_time_accessibilities, compute_parcel_travel_time_to_nearest_poi, compute_walk_distance_and_time, compute_job_accessibilities, compute_decay_job_accessibilities
import logging

logger = logging.getLogger(__name__)

try:
    import orca
except ImportError:
    orca = None


def setup_orca_from_parcels(parcels_path, sales_path=None, compute_stats=True):
    """
    Register parcels and optionally sales as orca tables.
    
    Args:
        parcels_path: Path to parcels_enriched.geojson
        sales_path: Path to sales_complete.geojson (optional)
        compute_stats: If True, compute and print summary statistics
    
    Returns:
        Tuple of (parcels_df, sales_df or None, stats)
    """
    if orca is None:
        raise RuntimeError("orca not installed; install with: pip install orca urbansim")
    
    logger.info(f"Loading parcels from {parcels_path}")
    parcels_gdf = gpd.read_file(str(parcels_path))
    parcels_df = pd.DataFrame(parcels_gdf.drop(columns=['geometry']))
    logger.info(f"  Loaded {len(parcels_df)} parcels, {len(parcels_df.columns)} attributes")
    
    # Register as orca table
    orca.add_table('parcels', parcels_df)
    logger.info("  Registered as orca table 'parcels'")
    
    sales_df = None
    if sales_path and Path(sales_path).exists():
        logger.info(f"Loading sales from {sales_path}")
        sales_gdf = gpd.read_file(str(sales_path))
        sales_df = pd.DataFrame(sales_gdf.drop(columns=['geometry']))
        logger.info(f"  Loaded {len(sales_df)} sales records across {sales_df['year'].nunique()} years")
        
        # Register as orca table
        orca.add_table('sales', sales_df)
        logger.info("  Registered as orca table 'sales'")
    
    # Compute statistics
    stats = {}
    if compute_stats:
        stats['parcels'] = {
            'count': len(parcels_df),
            'columns': len(parcels_df.columns),
            'av_columns': [c for c in parcels_df.columns if 'AV' in c.upper()],
        }
        
        # AV statistics (use current assessed values)
        for av_col in ['CurLandAV', 'CurImpAV', 'CurTotAV']:
            if av_col in parcels_df.columns:
                valid = parcels_df[av_col].notna()
                if valid.any():
                    stats[f'{av_col}_stats'] = {
                        'count_valid': valid.sum(),
                        'mean': parcels_df.loc[valid, av_col].mean(),
                        'median': parcels_df.loc[valid, av_col].median(),
                        'min': parcels_df.loc[valid, av_col].min(),
                        'max': parcels_df.loc[valid, av_col].max(),
                    }
        
        # District statistics
        if 'district_number' in parcels_df.columns:
            stats['by_district'] = {
                'count': parcels_df['district_number'].value_counts().to_dict(),
            }
            
            # District-level AV
            if 'CurTotAV' in parcels_df.columns:
                district_av = parcels_df.groupby('district_number')['CurTotAV'].agg(['count', 'mean', 'sum'])
                stats['by_district']['total_av'] = district_av.to_dict('index')
        
        # Sales statistics
        if sales_df is not None:
            stats['sales'] = {
                'count': len(sales_df),
                'year_range': f"{sales_df['year'].min()}-{sales_df['year'].max()}",
                'by_year': sales_df['year'].value_counts().sort_index().to_dict(),
                'linked_to_parcels': sales_df['PARCEL_ID'].notna().sum(),
                'parcel_coverage': sales_df['PARCEL_ID'].notna().sum() / len(sales_df) * 100,
            }
            
            if 'SalesAmount' in sales_df.columns:
                valid_sales = sales_df[sales_df['SalesAmount'].notna()]
                if len(valid_sales) > 0:
                    stats['sales']['amount_stats'] = {
                        'mean': valid_sales['SalesAmount'].mean(),
                        'median': valid_sales['SalesAmount'].median(),
                        'min': valid_sales['SalesAmount'].min(),
                        'max': valid_sales['SalesAmount'].max(),
                    }
    
    return parcels_df, sales_df, stats


def register_accessibility_into_orca(parcels_enriched_path, access_df=None, pois_gdf=None, jobs_gdf=None, nodes_gdf=None, edges_gdf=None, thresholds_s=(300,600,1200), pandana_net=None, speed_kph=50, decay_measures=None, out_path='data/processed/parcels_enriched_with_access.geojson', register_table=True):
    """Merge accessibility variables into parcels enriched file and optionally register with orca.

    You can either pass `access_df` (already computed accessibility per PARCEL_ID), or provide
    `pois_gdf` plus `nodes_gdf` and `edges_gdf` to compute access measures on-the-fly.

    - Writes merged parcels to `out_path` and re-registers the parcels table with orca if `register_table` is True.
    """
    p = Path(parcels_enriched_path)
    if not p.exists():
        raise FileNotFoundError(f"Parcels file not found: {p}")

    parcels_gdf = gpd.read_file(str(p))

    if access_df is None and pois_gdf is not None and nodes_gdf is not None and edges_gdf is not None:
        # Compute parcels' nearest nodes
        parcels_access_df = compute_nearest_node_distances(parcels_gdf, nodes_gdf)
        # compute travel-time-based counts using pandana if available, otherwise use Euclidean fallback
        access_counts = None
        try:
            access_counts = compute_travel_time_accessibilities(nodes_gdf, edges_gdf, pois_gdf, parcels_access_df, thresholds_s=thresholds_s, pandana_net=pandana_net, speed_kph=speed_kph, parcels_gdf=parcels_gdf)
        except Exception as e:
            logger.info(f"Warning: failed to compute travel-time accessibilities via pandana: {e}; falling back to Euclidean approximation")
            access_counts = compute_travel_time_accessibilities(nodes_gdf, edges_gdf, pois_gdf, parcels_access_df, thresholds_s=thresholds_s, pandana_net=None, speed_kph=speed_kph, parcels_gdf=parcels_gdf)

        # compute tt to nearest poi (Euclidean fallback or pandana if available)
        try:
            # compute walk distance and time (stop proximity) and merge
            walk = compute_walk_distance_and_time(parcels_gdf, pois_gdf, walk_speed_kph=speed_kph)
            # merge walk metrics into access_counts
            access_counts = access_counts.merge(walk, on='PARCEL_ID', how='left')
        except Exception as e:
            logger.info(f"Warning: failed to compute walk proximity: {e}")

        # If jobs_gdf provided, compute job accessibility measures and merge
        if jobs_gdf is not None:
            try:
                job_access = compute_job_accessibilities(nodes_gdf, edges_gdf, jobs_gdf, parcels_access_df, thresholds_s=thresholds_s, pandana_net=pandana_net, speed_kph=speed_kph, parcels_gdf=parcels_gdf, weight_col='jobs')
                access_counts = access_counts.merge(job_access, on='PARCEL_ID', how='left')
            except Exception as e:
                logger.info(f"Warning: failed to compute job accessibilities: {e}")

            # Compute decay-weighted job accessibilities if requested
            if decay_measures is not None:
                for decay in decay_measures:
                    try:
                        # decay may be tuple like ('exp', 100.0) or dict-like
                        if isinstance(decay, dict):
                            d_type = decay.get('type')
                            d_param = decay.get('param')
                        else:
                            d_type, d_param = decay
                        decay_df = compute_decay_job_accessibilities(nodes_gdf, edges_gdf, jobs_gdf, parcels_access_df, decay_type=d_type, decay_param=d_param, thresholds_s=None, pandana_net=pandana_net, speed_kph=speed_kph, parcels_gdf=parcels_gdf, weight_col='jobs')
                        access_counts = access_counts.merge(decay_df, on='PARCEL_ID', how='left')
                    except Exception as e:
                        logger.info(f"Warning: failed to compute decay job access ({decay}): {e}")

        access_df = access_counts

    if access_df is None:
        raise ValueError('access_df or pois_gdf+nodes_gdf+edges_gdf must be provided')

    merged = parcels_gdf.merge(access_df, on='PARCEL_ID', how='left')
    merged.to_file(out_path, driver='GeoJSON')
    logger.info(f"Wrote merged parcels with access columns to {out_path}")

    if register_table and orca is not None:
        try:
            # Re-register updated parcels
            parcels_df = merged.drop(columns='geometry')
            orca.add_table('parcels', parcels_df)
            logger.info("Registered updated parcels as orca table 'parcels'")
        except Exception as e:
            logger.info(f"Warning: failed to register updated parcels with orca: {e}")
    elif register_table:
        logger.info("Orca not installed; skipping table registration")

    return merged


def print_stats(stats):
    """Pretty-print statistics dictionary."""
    logger.info("\n=== Summary Statistics ===\n")
    
    if 'parcels' in stats:
        s = stats['parcels']
        logger.info(f"Parcels: {s['count']} records")
        logger.info(f"  Attributes: {s['columns']}")
        logger.info(f"  AV columns: {', '.join(s['av_columns'])}")
    
    # AV stats
    for key in stats:
        if key.endswith('_stats') and key != 'parcels':
            av_col = key.replace('_stats', '')
            s = stats[key]
            logger.info(f"\n{av_col}:")
            logger.info(f"  Valid: {s['count_valid']}")
            logger.info(f"  Mean: ${s['mean']:,.0f}")
            logger.info(f"  Median: ${s['median']:,.0f}")
            logger.info(f"  Range: ${s['min']:,.0f} - ${s['max']:,.0f}")
    
    # District stats
    if 'by_district' in stats:
        s = stats['by_district']
        if 'count' in s:
            logger.info(f"\nParcel Count by District:")
            for district, count in sorted(s['count'].items()):
                logger.info(f"  District {district}: {count} parcels")
    
    # Sales stats
    if 'sales' in stats:
        s = stats['sales']
        logger.info(f"\nSales: {s['count']} records")
        logger.info(f"  Year range: {s['year_range']}")
        logger.info(f"  Linked to parcels: {s['linked_to_parcels']} ({s['parcel_coverage']:.1f}%)")
        logger.info(f"  By year:")
        for year, count in sorted(s['by_year'].items()):
            logger.info(f"    {year}: {count} records")
        if 'amount_stats' in s:
            a = s['amount_stats']
            logger.info(f"  Sales amount (USD):")
            logger.info(f"    Mean: ${a['mean']:,.0f}")
            logger.info(f"    Median: ${a['median']:,.0f}")
            logger.info(f"    Range: ${a['min']:,.0f} - ${a['max']:,.0f}")


def generate_model_inputs(parcels_with_access_path, out_csv_path, variables=None, normalize=False, log_transform=None):
    """Generate a CSV of model-ready variables from a parcels GeoJSON with access columns.

    - parcels_with_access_path: path to GeoJSON with parcels and access columns
    - out_csv_path: output CSV file path
    - variables: list of columns to include; if None, picks defaults:
        ['PARCEL_ID', 'dist_to_node_m', 'nearest_node_id', 'walk_time_to_stop_s', 'tt_to_nearest_poi_s'] + all columns starting with 'job_' or 'opp_'
    - normalize: if True, produce z-score normalized versions for numeric vars (columns suffixed with '_z')
    - log_transform: list of columns to apply log1p to (before normalization)
    """
    p = Path(parcels_with_access_path)
    if not p.exists():
        raise FileNotFoundError(f"Parcels with access file not found: {p}")

    parcels = gpd.read_file(str(p))
    df = parcels.drop(columns=['geometry'], errors='ignore')

    if variables is None:
        # pick sensible defaults
        base = ['PARCEL_ID', 'dist_to_node_m', 'nearest_node_id', 'walk_time_to_stop_s', 'tt_to_nearest_poi_s']
        others = [c for c in df.columns if c.startswith('job_') or c.startswith('opp_')]
        variables = base + others

    # Keep only requested variables (if they exist)
    out = df[[c for c in variables if c in df.columns]].copy()
    # Fill NaNs with zeros for numeric columns
    numeric_cols = out.select_dtypes(include=['number']).columns.tolist()
    out[numeric_cols] = out[numeric_cols].fillna(0)

    # Apply log transform if requested. Use original df so transformations are available
    if log_transform is not None:
        for c in log_transform:
            if c in df.columns:
                out[f"{c}_log1p"] = np.log1p(df[c].fillna(0).astype(float))

    # Normalize if requested (z-score)
    if normalize:
        for c in numeric_cols:
            mu = out[c].mean()
            sd = out[c].std()
            if sd == 0 or np.isnan(sd):
                out[f"{c}_z"] = 0.0
            else:
                out[f"{c}_z"] = (out[c] - mu) / sd

    out.to_csv(out_csv_path, index=False)
    return out


def run_example_model(model_inputs_csv, out_csv_path=None, target_col=None, features=None, register_with_orca=False):
    """Run a tiny example linear model on generated model inputs.

    - model_inputs_csv: CSV produced by `generate_model_inputs`.
    - out_csv_path: optional path to write predictions CSV (if None, returns DataFrame only).
    - target_col: column name to use as dependent variable; if None, will use `CurTotAV` (log1p) when available or synthesize a target from job sums.
    - features: list of feature columns to use; if None, select numeric columns excluding PARCEL_ID and target.
    - register_with_orca: if True and orca is installed, register the predictions as an orca table 'model_inputs'

    Returns: DataFrame with predictions.
    """
    df = pd.read_csv(model_inputs_csv)

    # choose target
    if target_col is None:
        if 'CurTotAV' in df.columns:
            y = np.log1p(df['CurTotAV'].fillna(0).astype(float)).values
            target_col = 'CurTotAV_log1p'
            df[target_col] = y
        else:
            # synthesize target as weighted sum of job features if present
            job_cols = [c for c in df.columns if c.startswith('job_sum') or c.startswith('job_count')]
            if len(job_cols) > 0:
                y = df[job_cols].sum(axis=1).fillna(0).astype(float).values
                target_col = 'synth_target'
                df[target_col] = y
            else:
                # fallback: use zeros
                y = np.zeros(len(df))
                target_col = 'synth_target'
                df[target_col] = y

    # choose features
    if features is None:
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        features = [c for c in numeric if c not in ['PARCEL_ID', target_col]]

    if len(features) == 0:
        raise ValueError('No numeric features available to fit a model')

    # build design matrix with intercept
    X = df[features].fillna(0).astype(float).values
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # compute coefficients via least squares
    try:
        coef, *_ = np.linalg.lstsq(X, df[target_col].astype(float).values, rcond=None)
    except Exception as e:
        raise RuntimeError(f'Failed to fit example model: {e}')

    preds = X.dot(coef)
    df['predicted'] = preds

    if out_csv_path is not None:
        df.to_csv(out_csv_path, index=False)

    # Optionally register with orca
    if register_with_orca and orca is not None:
        try:
            orca.add_table('model_inputs', df.drop(columns=['geometry'], errors='ignore'))
        except Exception:
            pass

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Setup orca tables for model estimation")
    parser.add_argument('--parcels', type=str, default='data/processed/parcels_enriched.geojson',
                        help='Path to enriched parcels GeoJSON')
    parser.add_argument('--sales', type=str, default='data/processed/sales_complete.geojson',
                        help='Path to consolidated sales GeoJSON')
    parser.add_argument('--stats', action='store_true', help='Print summary statistics')
    
    args = parser.parse_args()
    
    try:
        parcels_df, sales_df, stats = setup_orca_from_parcels(args.parcels, args.sales, compute_stats=True)
        
        if args.stats:
            print_stats(stats)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

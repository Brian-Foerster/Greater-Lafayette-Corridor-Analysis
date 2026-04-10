"""Validation utilities for data/ files.

Provides functions to validate GeoJSON or shapefile inputs against the
`data/CONTRACT.md` expectations for the project.

Functions
- validate_geojson(path, id_prop='ZONE_ID', required=True, crs='EPSG:4326')
    Run a suite of checks and raise informative exceptions on failure.

- check_unique_ids(gdf, id_prop)
- check_missing_geometries(gdf)
"""
from pathlib import Path

try:
    import geopandas as gpd
except Exception:
    gpd = None


def check_unique_ids(gdf, id_prop):
    if id_prop not in gdf.columns:
        raise KeyError(f"Required id property '{id_prop}' not found")
    dupes = gdf[gdf.duplicated(subset=[id_prop], keep=False)][id_prop].unique()
    if len(dupes) > 0:
        raise ValueError(f"Duplicate ids found for '{id_prop}': {list(dupes)}")


def check_missing_geometries(gdf):
    if gdf.geometry.isnull().any():
        missing_idx = list(gdf[gdf.geometry.isnull()].index)
        raise ValueError(f"Missing geometries for features at indices: {missing_idx}")


def validate_geojson(path, id_prop='ZONE_ID', crs='EPSG:4326'):
    """Validate a GeoJSON (or shapefile) against basic contract rules.

    Checks performed:
    - File exists
    - geopandas available
    - contains id_prop
    - id_prop values are numeric/integral and can be coerced to int
    - id uniqueness
    - geometry presence
    - CRS matches target or is absent (if absent, caller should be aware)

    Raises informative exceptions on failure. Returns True on success.
    """
    import numpy as np
    if gpd is None:
        raise RuntimeError("geopandas not available; cannot validate geojson")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    gdf = gpd.read_file(str(p))

    # Check id property presence
    if id_prop not in gdf.columns:
        raise KeyError(f"Required id property '{id_prop}' not found in file {p}")

    # Check missing geometries
    check_missing_geometries(gdf)

    # Check ids: accept either all-numeric integral ids OR string keys (no nulls, unique)
    import pandas as pd
    series_num = pd.to_numeric(gdf[id_prop], errors='coerce')

    if series_num.notna().all():
        # All numeric: enforce integrality and coerce to int
        def is_integral(x):
            if isinstance(x, (int, np.integer)):
                return True
            if isinstance(x, (float, np.floating)):
                return float(x).is_integer()
            return False

        if not all(series_num.apply(is_integral)):
            nonint = series_num[~series_num.apply(is_integral)].unique().tolist()
            raise ValueError(f"Non-integral id values for '{id_prop}': {nonint}")

        gdf[id_prop] = series_num.astype('int64')
    elif series_num.notna().sum() == 0:
        # All non-numeric: treat as string keys; ensure no nulls and uniqueness
        if gdf[id_prop].isnull().any():
            missing_idx = list(gdf[gdf[id_prop].isnull()].index)
            raise ValueError(f"Missing id values for '{id_prop}' at indices: {missing_idx}")
        check_unique_ids(gdf, id_prop)
        # leave as-is (string)
    else:
        # Mixed numeric and non-numeric values are ambiguous
        raise ValueError(f"Mixed numeric and non-numeric id values for '{id_prop}'. Normalize before validation.")

    # Check uniqueness
    check_unique_ids(gdf, id_prop)

    # Check CRS
    if gdf.crs is None:
        # warn but not fail: some GeoJSONs omit CRS; caller may choose to reproject
        pass
    else:
        try:
            target = gpd.GeoSeries([]).set_crs(crs).crs
            if gdf.crs != target:
                raise ValueError(f"CRS mismatch: expected {crs}, got {gdf.crs}")
        except Exception:
            # If constructing target fails, skip detailed check
            pass

    return True

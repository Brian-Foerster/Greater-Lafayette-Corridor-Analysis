"""
Parcel Development Allocation — corridor data loading utilities.

Provides ``load_corridor_data()`` (used by apm_corridor_evaluation_integrated.py)
and supporting helpers for reading parcels/stations from project GeoJSON files.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from shapely import wkt as shapely_wkt


_GEODATA_CACHE: Dict[str, gpd.GeoDataFrame] = {}
_TABLE_CACHE: Dict[str, pd.DataFrame] = {}


def _read_geojson_fast(path: Path) -> gpd.GeoDataFrame:
    """Read GeoJSON via json module (bypasses fiona, 10-100x faster)."""
    import json as _json
    from shapely.geometry import shape as _shape
    with open(path, "r") as fh:
        gj = _json.load(fh)
    rows = [feat.get("properties", {}) for feat in gj.get("features", [])]
    geoms = [
        _shape(feat["geometry"]) if feat.get("geometry") else None
        for feat in gj["features"]
    ]
    crs = gj.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs=crs)


def _read_geodata_cached(path: Path) -> gpd.GeoDataFrame:
    """Read geodata once per path and reuse in-process for batch calls.

    For GeoJSON files, uses json-based reader (bypasses slow fiona).
    For large files, transparently caches to GeoParquet.
    """
    key = str(path.resolve())
    if key not in _GEODATA_CACHE:
        # Try GeoParquet cache first (fastest)
        parquet_path = path.with_suffix(".parquet")
        if (
            parquet_path.exists()
            and parquet_path.stat().st_mtime >= path.stat().st_mtime
        ):
            try:
                _GEODATA_CACHE[key] = gpd.read_parquet(parquet_path)
                return _GEODATA_CACHE[key]
            except Exception:
                pass  # fall through to GeoJSON read

        # Use fast json-based reader for GeoJSON (avoids fiona)
        if str(path).endswith(".geojson") or str(path).endswith(".json"):
            _GEODATA_CACHE[key] = _read_geojson_fast(path)
        else:
            _GEODATA_CACHE[key] = gpd.read_file(path)

        # Cache to GeoParquet for future speed (only for large files)
        if path.stat().st_size > 5_000_000:  # > 5 MB
            try:
                _GEODATA_CACHE[key].to_parquet(parquet_path)
            except Exception:
                pass  # non-critical
    return _GEODATA_CACHE[key]


def _read_table_cached(path: Path, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    """Read CSV once per path and reuse in-process for batch calls."""
    key = str(path.resolve())
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = pd.read_csv(path)
    df = _TABLE_CACHE[key]
    if usecols:
        cols = [c for c in usecols if c in df.columns]
        if cols:
            return df[cols].copy()
    return df.copy()


def _load_corridor_line_from_tabular_sources(
    corridor_id: str,
    data_dir: Path,
):
    """
    Load corridor geometry from tabular WKT sources when GeoJSON is incomplete.

    Returns:
        shapely geometry or None
    """
    cid = str(corridor_id)
    csv_candidates = [
        data_dir / "processed" / "phase2b_corridor_results_geometry_adjusted.csv",
        data_dir / "processed" / "apm_phase2a_results.csv",
    ]
    for csv_path in csv_candidates:
        if not csv_path.exists():
            continue
        try:
            df = _read_table_cached(csv_path, usecols=["corridor_id", "geometry"])
        except Exception:
            continue
        if "corridor_id" not in df.columns or "geometry" not in df.columns:
            continue
        matches = df[
            (df["corridor_id"].astype(str) == cid)
            & df["geometry"].notna()
            & (df["geometry"].astype(str).str.len() > 0)
        ]
        if matches.empty:
            continue
        wkt_text = str(matches.iloc[0]["geometry"])
        try:
            return shapely_wkt.loads(wkt_text)
        except Exception:
            continue
    return None


def load_corridor_data(corridor_id: str,
                       data_dir: Path = Path("data")) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load corridor-specific parcels and stations.

    Returns:
        (parcels_gdf, stations_gdf)
    """
    # Load parcels with enriched data (support multiple artifact names)
    parcel_candidates = [
        data_dir / "processed" / "parcels_improved_gravity_v2.geojson",
        data_dir / "processed" / "parcels_enriched.geojson",
        data_dir / "processed" / "parcels_enriched_final.geojson",
        data_dir / "processed" / "sales_complete.geojson",
        data_dir / "processed" / "parcels_enriched_with_access_test2.geojson",
        data_dir / "processed" / "sales_2021_linked.geojson",
        data_dir / "processed" / "parcels_clean.geojson",
    ]
    existing_candidates = [p for p in parcel_candidates if p.exists()]
    if not existing_candidates:
        raise FileNotFoundError(
            "No parcels file found. Expected one of: "
            + ", ".join(str(p) for p in parcel_candidates)
        )
    # Prefer the largest candidate file to avoid selecting tiny smoke-test artifacts.
    parcels_path = max(existing_candidates, key=lambda p: p.stat().st_size)

    parcels = _read_geodata_cached(parcels_path).copy()

    # Load corridor-specific stations
    corridors_path = data_dir / "processed" / "apm_phase2a_corridors.geojson"
    stops_path = data_dir / "processed" / "apm_phase2a_stops.geojson"

    if stops_path.exists():
        all_stops = _read_geodata_cached(stops_path)
        cid = str(corridor_id)
        stations = all_stops[all_stops["corridor_id"].astype(str) == cid].copy()
    else:
        stations = gpd.GeoDataFrame()

    if stations.empty:
        # Fallback: extract stations from corridor geometry when stop artifacts
        # are missing for a given corridor.
        cid = str(corridor_id)
        line_geom = None
        line_crs = "EPSG:4326"
        if corridors_path.exists():
            corridors = _read_geodata_cached(corridors_path)
            corridor_rows = corridors[corridors["corridor_id"].astype(str) == cid]
            if not corridor_rows.empty:
                line_geom = corridor_rows.iloc[0].geometry
                line_crs = corridors.crs

        if line_geom is None:
            line_geom = _load_corridor_line_from_tabular_sources(cid, data_dir)
            line_crs = "EPSG:4326"

        if line_geom is None:
            raise ValueError(
                f"Corridor {corridor_id} not found in corridor geometry sources "
                f"({corridors_path}, phase2b_corridor_results_geometry_adjusted.csv, apm_phase2a_results.csv)"
            )

        corridor_gdf = gpd.GeoDataFrame(
            {"corridor_id": [cid]},
            geometry=[line_geom],
            crs=line_crs,
        )
        corridor_utm = corridor_gdf.to_crs("EPSG:32616")
        line = corridor_utm.geometry.iloc[0]

        spacing_m = 1500.0
        length_m = float(line.length)
        if length_m <= 0:
            distances = np.array([0.0], dtype=float)
        else:
            distances = np.arange(0.0, length_m + 1e-9, spacing_m, dtype=float)
            if distances.size == 0 or distances[-1] < length_m:
                distances = np.append(distances, length_m)
        points_utm = [line.interpolate(float(d)) for d in distances]

        stations = gpd.GeoDataFrame(
            {"corridor_id": cid, "stop_id": range(len(points_utm))},
            geometry=points_utm,
            crs="EPSG:32616",
        ).to_crs(parcels.crs)

    return parcels, stations

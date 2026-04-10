"""Identify and enhance institutional trip generators.

This module identifies major institutional trip generators (universities, hospitals,
large employers) and applies multipliers to their attractiveness for transit modeling.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point, shape
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

OVERLAY_PARCEL_ID_CANDIDATES = ("parcel_id", "PARCEL_ID", "stkey", "STKEY")
OVERLAY_WEIGHT_CANDIDATES = (
    "institutional_weight",
    "demand_multiplier",
    "institutional_multiplier",
    "weight",
    "multiplier",
)


def load_institutional_overlay(
    overlay_source: Union[str, Path],
    parcel_id_col: Optional[str] = None,
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """Load and normalize institutional overlay records.

    Supported sources:
    - Local CSV/GeoJSON/Parquet files
    - HTTP(S) CSV/GeoJSON URLs

    Returns a DataFrame with normalized columns:
    - `parcel_id` (string)
    - `institutional_weight` (float, >=0)
    - `institutional_category` (optional)
    - `institutional_source` (optional)
    """
    src = str(overlay_source)
    is_url = src.startswith("http://") or src.startswith("https://")

    if is_url:
        if src.lower().endswith(".csv"):
            df = pd.read_csv(src)
        else:
            df = gpd.read_file(src)
    else:
        p = Path(src)
        if not p.exists():
            raise FileNotFoundError(f"Institutional overlay not found: {p}")
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        elif p.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
        else:
            df = gpd.read_file(p)

    if isinstance(df, gpd.GeoDataFrame):
        df = pd.DataFrame(df.drop(columns="geometry", errors="ignore"))

    # Resolve id/weight columns with explicit override first, then fallbacks.
    id_candidates = [parcel_id_col] if parcel_id_col else []
    id_candidates.extend(OVERLAY_PARCEL_ID_CANDIDATES)
    wt_candidates = [weight_col] if weight_col else []
    wt_candidates.extend(OVERLAY_WEIGHT_CANDIDATES)

    id_col = next((c for c in id_candidates if c and c in df.columns), None)
    wt_col = next((c for c in wt_candidates if c and c in df.columns), None)

    if id_col is None:
        raise ValueError(
            "Overlay is missing parcel id column. Expected one of "
            f"{OVERLAY_PARCEL_ID_CANDIDATES}."
        )
    if wt_col is None:
        raise ValueError(
            "Overlay is missing weight column. Expected one of "
            f"{OVERLAY_WEIGHT_CANDIDATES}."
        )

    out = df.copy()
    out["parcel_id"] = out[id_col].astype(str).str.strip()
    out["institutional_weight"] = pd.to_numeric(out[wt_col], errors="coerce").fillna(1.0)
    out["institutional_weight"] = out["institutional_weight"].clip(lower=0.0)

    # Optional descriptive columns if present.
    if "institutional_category" not in out.columns:
        if "category" in out.columns:
            out["institutional_category"] = out["category"].astype(str)
        elif "zone_type" in out.columns:
            out["institutional_category"] = out["zone_type"].astype(str)
        else:
            out["institutional_category"] = "institutional"
    if "institutional_source" not in out.columns:
        if "source" in out.columns:
            out["institutional_source"] = out["source"].astype(str)
        else:
            out["institutional_source"] = Path(src).name if not is_url else src

    out = out[out["parcel_id"] != ""].copy()
    out = out[["parcel_id", "institutional_weight", "institutional_category", "institutional_source"]]

    # If duplicates exist, keep highest weight (most conservative overlay).
    out = (
        out.sort_values("institutional_weight", ascending=False)
        .drop_duplicates(subset=["parcel_id"], keep="first")
        .reset_index(drop=True)
    )
    return out


def fetch_purdue_campus_boundary(output_path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Fetch Purdue University campus boundary from OpenStreetMap.
    
    Args:
        output_path: Optional path to save the result as GeoJSON
        
    Returns:
        GeoDataFrame with campus boundary polygon
    """
    # Overpass query for Purdue University campus
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Query for university buildings and landuse in West Lafayette
    query = """
    [out:json][timeout:60];
    area["name"="Tippecanoe County"]->.searchArea;
    (
      // University buildings
      way(area.searchArea)["amenity"="university"]["name"~"Purdue"];
      relation(area.searchArea)["amenity"="university"]["name"~"Purdue"];
      // Academic/institutional landuse
      way(area.searchArea)["landuse"="institutional"]["operator"~"Purdue"];
      way(area.searchArea)["building"="university"]["operator"~"Purdue"];
      // Specific campus areas
      way(area.searchArea)["name"~"Purdue University"];
      relation(area.searchArea)["name"~"Purdue University"];
    );
    out geom;
    """
    
    logger.info("Fetching Purdue campus boundary from OpenStreetMap...")
    response = requests.post(overpass_url, data={'data': query}, timeout=120)
    response.raise_for_status()
    
    data = response.json()
    elements = data.get('elements', [])
    
    if not elements:
        logger.warning("No Purdue campus features found in OSM")
        # Fallback: create bounding box around known Purdue location
        # Purdue campus center: approximately 40.4237° N, 86.9212° W
        center_lon, center_lat = -86.9212, 40.4237
        buffer_deg = 0.015  # ~1.5km radius
        from shapely.geometry import box
        bbox = box(
            center_lon - buffer_deg,
            center_lat - buffer_deg,
            center_lon + buffer_deg,
            center_lat + buffer_deg
        )
        gdf = gpd.GeoDataFrame(
            [{'name': 'Purdue University (fallback)', 'source': 'fallback_bbox'}],
            geometry=[bbox],
            crs="EPSG:4326"
        )
    else:
        # Parse OSM elements into geometries
        geometries = []
        for elem in elements:
            if elem['type'] == 'way' and 'geometry' in elem:
                coords = [(node['lon'], node['lat']) for node in elem['geometry']]
                if len(coords) >= 3:
                    from shapely.geometry import Polygon
                    geometries.append(Polygon(coords))
            elif elem['type'] == 'relation' and 'members' in elem:
                # Handle multipolygon relations
                for member in elem['members']:
                    if 'geometry' in member:
                        coords = [(node['lon'], node['lat']) for node in member['geometry']]
                        if len(coords) >= 3:
                            from shapely.geometry import Polygon
                            geometries.append(Polygon(coords))
        
        if not geometries:
            logger.warning("Could not parse OSM geometries, using fallback")
            center_lon, center_lat = -86.9212, 40.4237
            buffer_deg = 0.015
            from shapely.geometry import box
            bbox = box(
                center_lon - buffer_deg,
                center_lat - buffer_deg,
                center_lon + buffer_deg,
                center_lat + buffer_deg
            )
            gdf = gpd.GeoDataFrame(
                [{'name': 'Purdue University (fallback)', 'source': 'fallback_bbox'}],
                geometry=[bbox],
                crs="EPSG:4326"
            )
        else:
            # Merge all campus geometries into single boundary
            campus_geom = unary_union(geometries).convex_hull
            gdf = gpd.GeoDataFrame(
                [{'name': 'Purdue University', 'source': 'osm'}],
                geometry=[campus_geom],
                crs="EPSG:4326"
            )
    
    if output_path:
        gdf.to_file(output_path, driver='GeoJSON')
        logger.info(f"Saved campus boundary to {output_path}")
    
    return gdf


def identify_campus_parcels(
    parcels_gdf: gpd.GeoDataFrame,
    campus_gdf: gpd.GeoDataFrame,
    buffer_m: float = 100.0
) -> pd.Series:
    """Identify parcels within or near campus boundary.
    
    Args:
        parcels_gdf: Parcel GeoDataFrame (must have same CRS as campus_gdf)
        campus_gdf: Campus boundary GeoDataFrame
        buffer_m: Buffer distance in meters for "near campus" classification
        
    Returns:
        Boolean Series indicating campus parcels
    """
    # Ensure both are in same projected CRS for accurate buffering
    if parcels_gdf.crs != campus_gdf.crs:
        campus_gdf = campus_gdf.to_crs(parcels_gdf.crs)
    
    # If CRS is geographic (lat/lon), convert to projected for buffering
    if parcels_gdf.crs.is_geographic:
        utm_crs = "EPSG:32616"  # UTM Zone 16N for Indiana
        parcels_proj = parcels_gdf.to_crs(utm_crs)
        campus_proj = campus_gdf.to_crs(utm_crs)
        buffer_native = buffer_m  # UTM is in meters
    else:
        parcels_proj = parcels_gdf
        campus_proj = campus_gdf
        # EPSG:2965 uses US survey feet — convert meter buffer to feet
        _US_FT_TO_M = 0.3048006096012192
        buffer_native = buffer_m / _US_FT_TO_M

    # Buffer campus boundary
    campus_buffered = campus_proj.geometry.buffer(buffer_native).unary_union
    
    # Check intersection
    is_campus = parcels_proj.geometry.intersects(campus_buffered)
    
    return is_campus


def add_institutional_multipliers(
    parcels_gdf: gpd.GeoDataFrame,
    campus_boundary_path: Optional[Path] = None,
    academic_multiplier: float = 5.0,
    campus_proximity_multiplier: float = 2.5,
    jobs_col: str = 'jobs_combined',
    pop_col: str = 'pop_alloc'
) -> gpd.GeoDataFrame:
    """Add institutional trip generation multipliers to parcels.
    
    Args:
        parcels_gdf: Parcel GeoDataFrame
        campus_boundary_path: Path to campus boundary GeoJSON (if None, fetches from OSM)
        academic_multiplier: Multiplier for parcels within campus core
        campus_proximity_multiplier: Multiplier for parcels near campus
        jobs_col: Column name for jobs
        pop_col: Column name for population
        
    Returns:
        Updated parcels_gdf with 'institutional_weight' column
    """
    logger.info("Adding institutional trip generation multipliers...")
    
    # Load or fetch campus boundary
    if campus_boundary_path and Path(campus_boundary_path).exists():
        campus_gdf = gpd.read_file(campus_boundary_path)
        logger.info(f"Loaded campus boundary from {campus_boundary_path}")
    else:
        campus_gdf = fetch_purdue_campus_boundary(campus_boundary_path)
    
    # Identify campus parcels
    is_campus_core = identify_campus_parcels(parcels_gdf, campus_gdf, buffer_m=0)
    is_campus_adjacent = identify_campus_parcels(parcels_gdf, campus_gdf, buffer_m=500) & ~is_campus_core
    
    logger.info(f"Identified {is_campus_core.sum()} core campus parcels")
    logger.info(f"Identified {is_campus_adjacent.sum()} campus-adjacent parcels")
    
    # Initialize multiplier column
    parcels_gdf['institutional_weight'] = 1.0
    
    # Apply multipliers
    parcels_gdf.loc[is_campus_core, 'institutional_weight'] = academic_multiplier
    parcels_gdf.loc[is_campus_adjacent, 'institutional_weight'] = campus_proximity_multiplier
    
    # Add flag columns for diagnostics
    parcels_gdf['is_campus_core'] = is_campus_core
    parcels_gdf['is_campus_adjacent'] = is_campus_adjacent
    
    # Compute weighted jobs/pop
    if jobs_col in parcels_gdf.columns:
        parcels_gdf['jobs_weighted'] = parcels_gdf[jobs_col] * parcels_gdf['institutional_weight']
    if pop_col in parcels_gdf.columns:
        parcels_gdf['pop_weighted'] = parcels_gdf[pop_col] * parcels_gdf['institutional_weight']
    
    logger.info(f"Applied institutional multipliers: {academic_multiplier}x core, {campus_proximity_multiplier}x adjacent")
    
    return parcels_gdf


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.data.institutional_generators <parcels.geojson> [output.geojson]")
        print("       python -m src.data.institutional_generators --fetch-boundary <output.geojson>")
        sys.exit(1)
    
    if sys.argv[1] == '--fetch-boundary':
        output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('data/processed/purdue_campus_boundary.geojson')
        fetch_purdue_campus_boundary(output)
        print(f"Saved campus boundary to {output}")
    else:
        parcels_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else parcels_path.parent / f"{parcels_path.stem}_institutional{parcels_path.suffix}"
        
        logger.info(f"Loading parcels from {parcels_path}")
        parcels = gpd.read_file(parcels_path)
        
        parcels = add_institutional_multipliers(parcels)
        
        logger.info(f"Saving enhanced parcels to {output_path}")
        parcels.to_file(output_path, driver='GeoJSON')
        
        print(f"\nSummary:")
        print(f"  Core campus parcels: {parcels['is_campus_core'].sum()}")
        print(f"  Adjacent parcels: {parcels['is_campus_adjacent'].sum()}")
        print(f"  Regular parcels: {(~parcels['is_campus_core'] & ~parcels['is_campus_adjacent']).sum()}")
        print(f"\nWeighted jobs (campus core): {parcels[parcels['is_campus_core']]['jobs_weighted'].sum():.0f}")
        print(f"Original jobs (campus core): {parcels[parcels['is_campus_core']]['jobs_combined'].sum():.0f}")

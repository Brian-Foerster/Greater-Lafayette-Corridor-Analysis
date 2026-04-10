"""
Fetch OSM POIs via Overpass for a study area (derived from parcels extents)
and aggregate counts to parcels to build attractiveness scores.

Categories mapped to destination purposes:
- retail_poi: shop=*, mall
- food_poi: amenity=restaurant|fast_food|cafe|bar|pub
- service_poi: amenity=bank|pharmacy|clinic|doctors|hospital|hairdresser|post_office
- education_poi: amenity=school|university|college|kindergarten|library
- recreation_poi: leisure=park|sports_centre|pitch|swimming_pool|playground

Output CSV columns:
  PARCEL_ID, retail_poi, food_poi, service_poi, education_poi, recreation_poi, attractiveness

Usage:
  python -m src.data.osm_poi_loader --parcels data/processed/parcels_enriched_final.geojson \
      --out data/processed/poi_attractiveness.csv
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import shape
from src.spatial_constants import PROJECT_CRS

logger = logging.getLogger(__name__)


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


CATEGORY_FILTERS = {
    "retail_poi": ["shop"],
    "food_poi": ["amenity=restaurant", "amenity=fast_food", "amenity=cafe", "amenity=bar", "amenity=pub"],
    "service_poi": [
        "amenity=bank",
        "amenity=pharmacy",
        "amenity=clinic",
        "amenity=doctors",
        "amenity=hospital",
        "amenity=hairdresser",
        "amenity=post_office",
    ],
    "education_poi": ["amenity=school", "amenity=university", "amenity=college", "amenity=kindergarten", "amenity=library"],
    "recreation_poi": ["leisure=park", "leisure=sports_centre", "leisure=pitch", "leisure=swimming_pool", "leisure=playground"],
}


def _bbox_from_parcels(parcels: gpd.GeoDataFrame) -> str:
    minx, miny, maxx, maxy = parcels.to_crs(4326).total_bounds
    return f"{miny},{minx},{maxy},{maxx}"  # south,west,north,east


def _build_query(bbox: str) -> str:
    # Build Overpass query covering all relevant tags
    filters = []
    for vals in CATEGORY_FILTERS.values():
        for v in vals:
            if "=" in v:
                key, val = v.split("=", 1)
                filters.append(f"node[{key}={val}]({bbox});")
                filters.append(f"way[{key}={val}]({bbox});")
                filters.append(f"relation[{key}={val}]({bbox});")
            else:
                filters.append(f"node[{v}]({bbox});")
                filters.append(f"way[{v}]({bbox});")
                filters.append(f"relation[{v}]({bbox});")
    union = "".join(filters)
    return f"[out:json][timeout:120];({union});out center;"


def fetch_pois(bbox: str) -> gpd.GeoDataFrame:
    query = _build_query(bbox)
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    elements = data.get("elements", [])
    features = []
    for el in elements:
        if "lat" in el and "lon" in el:
            geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif "center" in el:  # ways/relations with center
            geom = {"type": "Point", "coordinates": [el["center"]["lon"], el["center"]["lat"]]}
        else:
            continue
        tags = el.get("tags", {})
        features.append({"type": "Feature", "geometry": geom, "properties": {"raw_tags": tags}})
    if not features:
        return gpd.GeoDataFrame(columns=["geometry"])
    gdf = gpd.GeoDataFrame.from_features(features, crs=4326)
    return gdf


def classify_category(tags: Dict) -> str:
    # Simple classifier based on presence of keys
    amenity = tags.get("amenity", "")
    shop = tags.get("shop", "")
    leisure = tags.get("leisure", "")
    if amenity in ["restaurant", "fast_food", "cafe", "bar", "pub"]:
        return "food_poi"
    if shop:
        return "retail_poi"
    if amenity in ["bank", "pharmacy", "clinic", "doctors", "hospital", "hairdresser", "post_office"]:
        return "service_poi"
    if amenity in ["school", "university", "college", "kindergarten", "library"]:
        return "education_poi"
    if leisure in ["park", "sports_centre", "pitch", "swimming_pool", "playground"]:
        return "recreation_poi"
    return "other"


def aggregate_to_parcels(parcels: gpd.GeoDataFrame, pois: gpd.GeoDataFrame) -> pd.DataFrame:
    if pois.empty:
        # Return zero counts
        out = parcels[["PARCEL_ID"]].copy()
        for cat in CATEGORY_FILTERS.keys():
            out[cat] = 0
        out["attractiveness"] = 0
        return out

    pois = pois.copy()
    pois["category"] = pois["raw_tags"].apply(classify_category)
    pois = pois[pois["category"] != "other"]
    if pois.empty:
        return aggregate_to_parcels(parcels, gpd.GeoDataFrame())

    # Spatial join
    parcels_proj = parcels.to_crs(PROJECT_CRS)
    pois_proj = pois.to_crs(PROJECT_CRS)
    joined = gpd.sjoin(pois_proj, parcels_proj[["PARCEL_ID", "geometry"]], how="inner", predicate="within")

    # Count per parcel per category
    counts = joined.groupby(["PARCEL_ID", "category"]).size().unstack(fill_value=0)
    out = parcels_proj[["PARCEL_ID"]].copy()
    for cat in CATEGORY_FILTERS.keys():
        series = counts.get(cat, pd.Series(dtype=float))
        out[cat] = out["PARCEL_ID"].map(series).fillna(0)

    # Compute composite attractiveness as weighted sum (heavier weight on food/retail)
    weights = {
        "retail_poi": 1.0,
        "food_poi": 1.0,
        "service_poi": 0.6,
        "education_poi": 0.4,
        "recreation_poi": 0.5,
    }
    attr = sum(out[c] * w for c, w in weights.items())
    max_attr = attr.max() or 1.0
    out["attractiveness"] = attr / max_attr
    return out


def run(parcels_path: Path, out_path: Path) -> None:
    parcels = gpd.read_file(parcels_path)
    bbox = _bbox_from_parcels(parcels)
    logger.info(f"Querying OSM within bbox {bbox}")
    pois = fetch_pois(bbox)
    logger.info(f"Fetched {len(pois):,} raw POIs")
    agg = aggregate_to_parcels(parcels, pois)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    logger.info(f"Wrote POI attractiveness to {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fetch OSM POIs and aggregate to parcels")
    p.add_argument("--parcels", required=True, help="Parcels GeoJSON")
    p.add_argument("--out", default="data/processed/poi_attractiveness.csv", help="Output CSV path")
    args = p.parse_args()

    run(Path(args.parcels), Path(args.out))

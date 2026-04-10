"""Generate APM candidate corridors and stops with spacing constraints.

Heuristics:
- Build demand points from parcels with population + jobs weights.
- Connect top-N weighted centroids into straight-line corridors within a max length.
- Enforce max average stops per km (default 1.5) and minimum stop spacing (>=100m and >= ~0.67km implied by 1.5/km).

Outputs:
- corridors_gdf: lines with length_km, n_stops
- stops_gdf: points with stop_id, corridor_id, sequence
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from src.spatial_constants import PROJECT_CRS, US_SURVEY_FT_TO_M
from shapely.geometry import LineString, Point


def build_demand_points(parcels_gdf: gpd.GeoDataFrame, pop_col: str = "pop_alloc", jobs_col: str = "jobs_alloc", top_n: int = 40) -> gpd.GeoDataFrame:
    df = parcels_gdf.copy()
    df["demand_wt"] = df.get(pop_col, 0).fillna(0) + df.get(jobs_col, 0).fillna(0)
    df = df[df["demand_wt"] > 0]
    df = df.sort_values("demand_wt", ascending=False).head(top_n)
    if df.crs is None:
        df = df.set_crs(4326)
    metric = df.to_crs(PROJECT_CRS)
    metric["geometry"] = metric.geometry.centroid
    return metric.to_crs(4326)


def _generate_stops_along_line(line: LineString, corridor_id: str, max_stops_per_km: float = 1.5, min_stop_spacing_m: float = 700.0) -> Tuple[gpd.GeoDataFrame, int]:
    # line.length is in EPSG:2965 US survey feet — convert to meters for parameter logic
    length_ft = line.length
    length_m = length_ft * US_SURVEY_FT_TO_M
    length_km = length_m / 1000.0
    max_stops = max(2, int(np.floor(length_km * max_stops_per_km)) + 1)
    # spacing in feet (convert meter params to feet for geometry operations)
    min_spacing_ft = min_stop_spacing_m / US_SURVEY_FT_TO_M
    spacing_ft = max(100.0 / US_SURVEY_FT_TO_M, min_spacing_ft, length_ft / max(1, max_stops - 1))
    # recompute stop count from spacing
    n_stops = int(np.floor(length_ft / spacing_ft)) + 1
    distances = np.linspace(0, length_ft, n_stops)  # feet for interpolate()
    # Ensure at least two stops; if spacing yields <2, force endpoints
    points = [line.interpolate(d) for d in distances]
    if len(points) < 2:
        points = [line.coords[0], line.coords[-1]]
        points = [Point(pt) for pt in points]
        n_stops = 2
    stops = gpd.GeoDataFrame(
        {
            "corridor_id": corridor_id,
            "stop_id": [f"{corridor_id}_S{i+1}" for i in range(n_stops)],
            "sequence": list(range(1, n_stops + 1)),
        },
        geometry=points,
        crs=PROJECT_CRS,
    )
    return stops, n_stops


def _select_seed_points(pts_metric: gpd.GeoDataFrame, k: int = 5) -> list[int]:
    # Greedy farthest-point selection seeded by top weight
    idx_sorted = list(pts_metric.sort_values("demand_wt", ascending=False).index)
    if not idx_sorted:
        return []
    seeds = [idx_sorted[0]]
    while len(seeds) < min(k, len(idx_sorted)):
        best_idx = None
        best_score = -1
        for idx in idx_sorted:
            if idx in seeds:
                continue
            dists = pts_metric.geometry.loc[seeds].distance(pts_metric.geometry.loc[idx])
            score = dists.min() * pts_metric.loc[idx, "demand_wt"]
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        seeds.append(best_idx)
    return seeds


def _score_corridor(line: LineString, pts_metric: gpd.GeoDataFrame) -> float:
    # Score by demand captured near the line and demand at endpoints
    buffer = line.buffer(600.0 / US_SURVEY_FT_TO_M)  # 600m catchment (convert to feet)
    near = pts_metric[pts_metric.geometry.within(buffer)]
    demand_sum = near["demand_wt"].sum()
    length_km = max(0.1, line.length * US_SURVEY_FT_TO_M / 1000.0)
    return demand_sum / length_km


def generate_apm_corridors(
    parcels_gdf: gpd.GeoDataFrame,
    pop_col: str = "pop_alloc",
    jobs_col: str = "jobs_alloc",
    top_n: int = 40,
    max_length_km: float = 20.0,
    max_stops_per_km: float = 1.5,
    min_stop_spacing_m: float = 700.0,
    seed_count: int = 5,
    bend_candidates: int = 5,
    max_routes_per_seed: int = 3,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Generate candidate APM corridors with simple heuristics (non-straight allowed).

    Heuristic:
    - Build top-N demand points (pop+jobs).
    - Pick diverse seed points (greedy farthest on weight).
    - From each seed, evaluate best straight and one-bend routes to other demand points within length.
    - Score by demand within 600m buffer per km; keep top routes per seed.
    """
    pts = build_demand_points(parcels_gdf, pop_col=pop_col, jobs_col=jobs_col, top_n=top_n)
    if pts.crs is None:
        pts = pts.set_crs(4326)
    metric = pts.to_crs(PROJECT_CRS)

    seeds = _select_seed_points(metric, k=seed_count)
    corridors = []
    all_stops = []
    cid = 1
    for seed_idx in seeds:
        seed_pt = metric.geometry.loc[seed_idx]
        candidates = []
        for dest_idx, dest_pt in metric.geometry.items():
            if dest_idx == seed_idx:
                continue
            # Straight line candidate
            line_straight = LineString([seed_pt, dest_pt])
            length_km = line_straight.length * US_SURVEY_FT_TO_M / 1000.0
            if 0 < length_km <= max_length_km:
                candidates.append((line_straight, _score_corridor(line_straight, metric), dest_idx))
            # One-bend candidates through top bend_candidates points (skip seed/dest)
            mid_candidates = metric.sort_values("demand_wt", ascending=False).head(bend_candidates).index
            for mid_idx in mid_candidates:
                if mid_idx in (seed_idx, dest_idx):
                    continue
                mid_pt = metric.geometry.loc[mid_idx]
                line_bent = LineString([seed_pt, mid_pt, dest_pt])
                length_km_bent = line_bent.length * US_SURVEY_FT_TO_M / 1000.0
                if 0 < length_km_bent <= max_length_km:
                    candidates.append((line_bent, _score_corridor(line_bent, metric), dest_idx))
        # keep top routes for this seed
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:max_routes_per_seed]
        for line, score, _dest_idx in candidates:
            corridor_id = f"C{cid}"
            stops_gdf, n_stops = _generate_stops_along_line(line, corridor_id, max_stops_per_km=max_stops_per_km, min_stop_spacing_m=min_stop_spacing_m)
            corridors.append({"corridor_id": corridor_id, "length_km": line.length * US_SURVEY_FT_TO_M / 1000.0, "n_stops": n_stops, "score": score, "geometry": line})
            all_stops.append(stops_gdf)
            cid += 1

    corridors_gdf = gpd.GeoDataFrame(corridors, geometry="geometry", crs=PROJECT_CRS)
    if all_stops:
        stops_gdf = gpd.GeoDataFrame(pd.concat(all_stops, ignore_index=True), crs=PROJECT_CRS)
    else:
        stops_gdf = gpd.GeoDataFrame(columns=["corridor_id", "stop_id", "sequence", "geometry"], crs=PROJECT_CRS)

    # Reproject back to WGS84 for compatibility
    corridors_gdf = corridors_gdf.to_crs(4326)
    stops_gdf = stops_gdf.to_crs(4326)
    return corridors_gdf, stops_gdf

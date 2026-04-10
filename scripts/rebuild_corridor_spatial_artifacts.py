#!/usr/bin/env python
"""
Rebuild corridor spatial artifacts for all 40 corridors from local data only.

Outputs:
  - data/processed/apm_phase2a_corridors.geojson
  - data/processed/apm_phase2a_stops.geojson
  - data/processed/phase2b_corridor_results.csv (length/n_stops backfilled)
  - data/processed/corridor_geometry_rebuild_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt

import logging
logger = logging.getLogger(__name__)



PROCESSED = Path("data/processed")
CORRIDORS_OUT = PROCESSED / "apm_phase2a_corridors.geojson"
STOPS_OUT = PROCESSED / "apm_phase2a_stops.geojson"
PHASE2B_PATH = PROCESSED / "phase2b_corridor_results.csv"


def _corridor_sort_key(corridor_id: str) -> Tuple[int, str]:
    cid = str(corridor_id)
    if cid.startswith("C") and cid[1:].isdigit():
        return (int(cid[1:]), cid)
    return (10_000_000, cid)


def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _parse_corridor_num(corridor_id: str) -> int | None:
    cid = str(corridor_id)
    if cid.startswith("C") and cid[1:].isdigit():
        return int(cid[1:])
    return None


def _create_backup_once(path: Path) -> Path:
    backup = path.with_name(f"{path.stem}.pre_geometry_rebuild_backup{path.suffix}")
    if path.exists() and not backup.exists():
        backup.write_bytes(path.read_bytes())
    return backup


def _load_geometry_templates() -> pd.DataFrame:
    candidates = [
        ("apm_phase2a_results", PROCESSED / "apm_phase2a_results.csv"),
        ("phase2b_geometry_adjusted", PROCESSED / "phase2b_corridor_results_geometry_adjusted.csv"),
    ]

    frames: List[pd.DataFrame] = []
    for source_name, path in candidates:
        if not path.exists():
            continue
        df = _safe_read_csv(path)
        required = {"corridor_id", "geometry"}
        if not required.issubset(df.columns):
            continue

        keep_cols = ["corridor_id", "geometry", "length_km", "n_stops", "bus_competition", "mean_apm_share"]
        cols = [c for c in keep_cols if c in df.columns]
        tmp = df[cols].copy()
        tmp["geometry_source"] = source_name
        frames.append(tmp)

    if not frames:
        raise FileNotFoundError(
            "No geometry template source found. Expected apm_phase2a_results.csv "
            "or phase2b_corridor_results_geometry_adjusted.csv."
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["geometry"].notna()].copy()
    merged["corridor_id"] = merged["corridor_id"].astype(str)
    merged["corridor_num"] = merged["corridor_id"].map(_parse_corridor_num)
    merged = merged[merged["corridor_num"].notna()].copy()
    merged = merged[(merged["corridor_num"] >= 1) & (merged["corridor_num"] <= 25)].copy()

    # Keep latest source row per corridor id; phase2b_geometry_adjusted has priority.
    source_order = {"apm_phase2a_results": 0, "phase2b_geometry_adjusted": 1}
    merged["source_rank"] = merged["geometry_source"].map(source_order).fillna(0)
    merged = merged.sort_values(["corridor_id", "source_rank"]).drop_duplicates(
        subset=["corridor_id"], keep="last"
    )

    merged["length_km"] = pd.to_numeric(merged.get("length_km"), errors="coerce")
    merged["n_stops"] = pd.to_numeric(merged.get("n_stops"), errors="coerce")
    merged["bus_competition"] = pd.to_numeric(merged.get("bus_competition"), errors="coerce")
    merged["mean_apm_share"] = pd.to_numeric(merged.get("mean_apm_share"), errors="coerce")

    # Parse WKT geometries.
    merged["geometry_obj"] = merged["geometry"].map(lambda x: shapely_wkt.loads(str(x)))
    merged = merged.sort_values("corridor_id", key=lambda s: s.map(_corridor_sort_key)).reset_index(drop=True)
    return merged


def _assignment_min_cost(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        return linear_sum_assignment(cost)
    except Exception:
        # Fallback: greedy if scipy is unavailable.
        rows, cols = [], []
        used_cols: set[int] = set()
        for i in range(cost.shape[0]):
            col_order = np.argsort(cost[i])
            chosen = next((int(c) for c in col_order if int(c) not in used_cols), None)
            if chosen is None:
                chosen = int(col_order[0])
            rows.append(i)
            cols.append(chosen)
            used_cols.add(chosen)
        return np.array(rows, dtype=int), np.array(cols, dtype=int)


def _build_missing_mapping(
    demand_df: pd.DataFrame,
    template_df: pd.DataFrame,
) -> Dict[str, str]:
    demand = demand_df.copy()
    demand["corridor_id"] = demand["corridor_id"].astype(str)
    demand["phase2b_daily_riders"] = pd.to_numeric(demand["phase2b_daily_riders"], errors="coerce").fillna(0.0)

    template_ids = set(template_df["corridor_id"].astype(str))
    known_ids = sorted(
        [cid for cid in demand["corridor_id"].tolist() if cid in template_ids],
        key=_corridor_sort_key,
    )
    missing_ids = sorted(
        [cid for cid in demand["corridor_id"].tolist() if cid not in template_ids],
        key=_corridor_sort_key,
    )

    if not missing_ids:
        return {}
    if not known_ids:
        raise ValueError("No known template corridor IDs available for surrogate mapping.")
    if len(missing_ids) > len(known_ids):
        raise ValueError("Missing corridors exceed available template corridors for one-to-one assignment.")

    known_vals = demand.set_index("corridor_id").loc[known_ids, "phase2b_daily_riders"].to_numpy(dtype=float)
    missing_vals = demand.set_index("corridor_id").loc[missing_ids, "phase2b_daily_riders"].to_numpy(dtype=float)

    cost = np.abs(missing_vals[:, None] - known_vals[None, :])
    r_idx, c_idx = _assignment_min_cost(cost)
    return {missing_ids[int(r)]: known_ids[int(c)] for r, c in zip(r_idx, c_idx)}


def _build_corridor_layer(
    demand_df: pd.DataFrame,
    template_df: pd.DataFrame,
    surrogate_map: Dict[str, str],
) -> gpd.GeoDataFrame:
    demand = demand_df.copy()
    demand["corridor_id"] = demand["corridor_id"].astype(str)
    demand["phase2b_daily_riders"] = pd.to_numeric(demand["phase2b_daily_riders"], errors="coerce").fillna(0.0)
    demand = demand.sort_values("corridor_id", key=lambda s: s.map(_corridor_sort_key)).reset_index(drop=True)

    template_lookup = template_df.set_index("corridor_id")
    rows: List[Dict[str, object]] = []

    for _, drow in demand.iterrows():
        cid = str(drow["corridor_id"])
        template_cid = cid if cid in template_lookup.index else surrogate_map.get(cid)
        if template_cid is None or template_cid not in template_lookup.index:
            raise KeyError(f"No geometry template found for corridor {cid}")

        trow = template_lookup.loc[template_cid]
        length_km = float(trow["length_km"]) if pd.notna(trow["length_km"]) else np.nan
        n_stops_val = trow["n_stops"] if pd.notna(trow["n_stops"]) else np.nan
        try:
            n_stops = int(max(2, int(round(float(n_stops_val)))))
        except Exception:
            n_stops = 5

        rows.append(
            {
                "corridor_id": cid,
                "length_km": length_km,
                "n_stops": n_stops,
                "source": (
                    trow["geometry_source"]
                    if cid == template_cid
                    else f"surrogate_from_{template_cid}"
                ),
                "ridership_est": float(drow["phase2b_daily_riders"]),
                "tif_potential": np.nan,
                "cost_efficiency": np.nan,
                "bus_competition": float(trow["bus_competition"]) if pd.notna(trow["bus_competition"]) else np.nan,
                "apm_share": float(trow["mean_apm_share"]) if pd.notna(trow["mean_apm_share"]) else np.nan,
                "weighted_demand": np.nan,
                "template_corridor_id": template_cid,
                "geometry_mapping_source": (
                    "direct_template" if cid == template_cid else "ridership_min_cost_template"
                ),
                "geometry": trow["geometry_obj"],
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.sort_values("corridor_id", key=lambda s: s.map(_corridor_sort_key)).reset_index(drop=True)
    return gdf


def _build_stops_layer(corridors_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    stop_rows: List[Dict[str, object]] = []

    for _, row in corridors_gdf.iterrows():
        cid = str(row["corridor_id"])
        line = row.geometry
        if line.geom_type == "MultiLineString":
            line = max(list(line.geoms), key=lambda g: g.length)
        if line.geom_type != "LineString":
            continue

        n_stops = int(max(2, int(row.get("n_stops", 5))))
        line_utm = gpd.GeoSeries([line], crs="EPSG:4326").to_crs("EPSG:32616").iloc[0]
        length_m = float(line_utm.length)
        distances = np.linspace(0.0, length_m, n_stops)
        points_utm = [line_utm.interpolate(float(d)) for d in distances]
        points_wgs84 = gpd.GeoSeries(points_utm, crs="EPSG:32616").to_crs("EPSG:4326")

        for seq, pt in enumerate(points_wgs84.tolist(), start=1):
            stop_rows.append(
                {
                    "corridor_id": cid,
                    "stop_id": f"{cid}_S{seq}",
                    "sequence": seq,
                    "geometry": pt,
                }
            )

    stops = gpd.GeoDataFrame(stop_rows, geometry="geometry", crs="EPSG:4326")
    stops = stops.sort_values(["corridor_id", "sequence"], key=lambda s: s.map(_corridor_sort_key) if s.name == "corridor_id" else s).reset_index(drop=True)
    return stops


def _backfill_phase2b_lengths(
    demand_df: pd.DataFrame,
    corridors_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    out = demand_df.copy()
    out["corridor_id"] = out["corridor_id"].astype(str)

    meta = corridors_gdf[["corridor_id", "length_km", "n_stops", "template_corridor_id", "geometry_mapping_source"]].copy()
    out = out.merge(meta, on="corridor_id", how="left")
    return out


def main() -> None:
    if not PHASE2B_PATH.exists():
        raise FileNotFoundError(f"Missing demand file: {PHASE2B_PATH}")

    demand = pd.read_csv(PHASE2B_PATH)
    if "corridor_id" not in demand.columns or "phase2b_daily_riders" not in demand.columns:
        raise ValueError("phase2b_corridor_results.csv must include corridor_id and phase2b_daily_riders")

    # Backup first so this operation is reversible.
    _create_backup_once(CORRIDORS_OUT)
    _create_backup_once(STOPS_OUT)
    _create_backup_once(PHASE2B_PATH)

    templates = _load_geometry_templates()
    surrogate_map = _build_missing_mapping(demand, templates)

    corridors_gdf = _build_corridor_layer(demand, templates, surrogate_map)
    stops_gdf = _build_stops_layer(corridors_gdf)
    demand_backfilled = _backfill_phase2b_lengths(demand, corridors_gdf)

    CORRIDORS_OUT.parent.mkdir(parents=True, exist_ok=True)
    corridors_gdf.to_file(CORRIDORS_OUT, driver="GeoJSON")
    stops_gdf.to_file(STOPS_OUT, driver="GeoJSON")
    demand_backfilled.to_csv(PHASE2B_PATH, index=False)

    summary = {
        "corridors_written": int(len(corridors_gdf)),
        "stops_written": int(len(stops_gdf)),
        "phase2b_rows_written": int(len(demand_backfilled)),
        "template_corridors_available": int(templates["corridor_id"].nunique()),
        "surrogate_mappings_count": int(len(surrogate_map)),
        "surrogate_mappings": dict(sorted(surrogate_map.items(), key=lambda kv: _corridor_sort_key(kv[0]))),
        "corridors_file": str(CORRIDORS_OUT),
        "stops_file": str(STOPS_OUT),
        "phase2b_file": str(PHASE2B_PATH),
    }

    summary_path = PROCESSED / "corridor_geometry_rebuild_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Rebuild complete.")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()


"""
Consolidated scenario evaluation module.

Includes:
- Basic scenario comparison (add stop and rerun)
- Baseline vs scenario ridership comparison
- Land value uplift calculation
- TIF revenue estimation
- Batch corridor evaluation with multiprocessing
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

from src.spatial_constants import PROJECT_CRS, US_SURVEY_FT_TO_M  # noqa: E402
from src.build_network import (
    build_nodes_edges_from_roads,
    add_travel_time_to_edges,
    compute_nearest_node_distances,
    compute_travel_time_accessibilities,
)
from src.transit_model import (
    run_transit_scenario,
    assign_boardings_to_routes,
    assign_ridership_gravity,
    assign_ridership_enhanced,
    assign_ridership_hybrid,
    assign_ridership_lodes,
    DEFAULT_LODES_OD_PATH,
)
from src.finance import capex_musd, npv_irr
from src.mode_choice import calculate_mode_shares_vectorized, apply_fare_elasticity
from src.land_use_transport_model import LandUseTransportModel
from src.financial_params import TIF_CAPTURE_RATE_CONSERVATIVE, PROPERTY_TAX_RATE


def load_scenario_config(config_path: Path | str = Path("scenarios_config.json")) -> dict:
    """Load scenario configuration JSON."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# BASIC SCENARIO COMPARISON
# ============================================================================

def add_stop_and_rerun(
    parcels_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    new_stop_geom,
    new_stop_id: str = 'NEW',
    jobs_col: str = 'jobs',
    thresholds_s: tuple = (300, 600, 1200),
    beta_walk: float = 0.01,
    speed_kph: float = 5,
):
    """Add a new stop, rebuild network, recompute access and transit ridership.

    Returns a dict: {'baseline': {...}, 'scenario': {...}, 'delta': {...}}
    """
    nodes, edges = build_nodes_edges_from_roads(roads_gdf)
    edges = add_travel_time_to_edges(edges, speed_kph=speed_kph)
    parcels_access = compute_nearest_node_distances(parcels_gdf, nodes)

    baseline_stop_ridership = run_transit_scenario(
        parcels_gdf, stops_gdf, out_path=None, jobs_col=jobs_col, beta_walk=beta_walk
    )

    po_gdf = parcels_gdf.copy()
    po_gdf['jobs'] = parcels_gdf.get(jobs_col, 0)
    access_baseline = compute_travel_time_accessibilities(
        nodes, edges, po_gdf, parcels_access,
        thresholds_s=thresholds_s, pandana_net=None,
        speed_kph=speed_kph, parcels_gdf=parcels_gdf
    )

    scenario_stops = stops_gdf.append(
        {'stop_id': new_stop_id, 'geometry': new_stop_geom}, ignore_index=True
    )
    scenario_stop_ridership = run_transit_scenario(
        parcels_gdf, scenario_stops, out_path=None, jobs_col=jobs_col, beta_walk=beta_walk
    )
    scenario_access = compute_travel_time_accessibilities(
        nodes, edges, po_gdf, parcels_access,
        thresholds_s=thresholds_s, pandana_net=None,
        speed_kph=speed_kph, parcels_gdf=parcels_gdf
    )

    b = baseline_stop_ridership.reset_index().set_index('stop_id')
    s = scenario_stop_ridership.reset_index().set_index('stop_id')
    all_stops = b.index.union(s.index)
    b = b.reindex(all_stops, fill_value=0.0)
    s = s.reindex(all_stops, fill_value=0.0)
    stop_delta = s['potential_trips'] - b['potential_trips']

    access_delta = scenario_access.set_index('PARCEL_ID').subtract(
        access_baseline.set_index('PARCEL_ID'), fill_value=0
    )

    return {
        'baseline': {'access': access_baseline, 'stop_ridership': baseline_stop_ridership},
        'scenario': {'access': scenario_access, 'stop_ridership': scenario_stop_ridership},
        'delta': {'stop_delta': stop_delta, 'access_delta': access_delta}
    }


def write_scenario_report(result: dict, out_dir: str):
    """Write scenario comparison results to CSV files."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sd = result['delta']['stop_delta'].reset_index()
    sd.columns = ['stop_id', 'delta_potential_trips']
    sd.to_csv(Path(out_dir) / 'stop_delta.csv', index=False)
    ad = result['delta']['access_delta'].reset_index()
    ad.to_csv(Path(out_dir) / 'parcel_access_delta.csv', index=False)
    return True


# ============================================================================
# BASELINE VS SCENARIO COMPARISON
# ============================================================================

def run_baseline_scenario(
    parcels_gdf: gpd.GeoDataFrame,
    existing_stops_gdf: gpd.GeoDataFrame,
    jobs_col: str = 'jobs',
    beta_walk: float = 0.01,
    max_walk_m: float = 1000,
    walk_speed_kph: float = 5,
):
    """Run baseline (existing transit) scenario."""
    stop_agg = assign_ridership_gravity(
        parcels_gdf, existing_stops_gdf,
        jobs_col=jobs_col, beta_walk=beta_walk,
        max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph
    )
    total = float(stop_agg['potential_trips'].sum())
    return {'stop_ridership': stop_agg, 'total_ridership': total}


def run_scenario_with_new_transit(
    parcels_gdf: gpd.GeoDataFrame,
    baseline_stops_gdf: gpd.GeoDataFrame,
    new_stops_gdf: gpd.GeoDataFrame,
    jobs_col: str = 'jobs',
    beta_walk: float = 0.01,
    max_walk_m: float = 1000,
    walk_speed_kph: float = 5,
):
    """Run scenario with new transit added to baseline stops."""
    all_stops = pd.concat([baseline_stops_gdf, new_stops_gdf], ignore_index=True)
    stop_agg = assign_ridership_gravity(
        parcels_gdf, all_stops,
        jobs_col=jobs_col, beta_walk=beta_walk,
        max_walk_m=max_walk_m, walk_speed_kph=walk_speed_kph
    )
    total = float(stop_agg['potential_trips'].sum())

    new_stop_ids = new_stops_gdf['stop_id'].tolist() if 'stop_id' in new_stops_gdf.columns else []
    new_stop_agg = stop_agg[stop_agg.index.isin(new_stop_ids)]

    return {
        'stop_ridership': stop_agg,
        'total_ridership': total,
        'new_stop_ridership': new_stop_agg
    }


# ============================================================================
# LAND VALUE UPLIFT
# ============================================================================

def compute_uplift_comparison(
    parcels_gdf: gpd.GeoDataFrame,
    baseline_results: dict,
    scenario_results: dict,
    land_value_col: str = 'CurLandAV',
    tif_gdf: Optional[gpd.GeoDataFrame] = None,
    tif_capture_rate: float = TIF_CAPTURE_RATE_CONSERVATIVE,
):
    """Estimate land value uplift from baseline to scenario."""
    baseline_stop_ridership = baseline_results.get('stop_ridership', pd.DataFrame())
    scenario_stop_ridership = scenario_results.get('stop_ridership', pd.DataFrame())
    new_stop_ridership = scenario_results.get('new_stop_ridership', scenario_stop_ridership)

    parcels = parcels_gdf.copy()
    if land_value_col not in parcels.columns:
        parcels[land_value_col] = 100000

    parcels['baseline_lv'] = parcels[land_value_col].fillna(0).astype(float)

    if not new_stop_ridership.empty and 'geometry' in parcels.columns:
        stops_gdf = new_stop_ridership.copy()
        if 'geometry' not in stops_gdf.columns:
            parcels['uplift_factor'] = 0.05
        else:
            parcels_proj = parcels.to_crs(PROJECT_CRS)
            stops_proj = stops_gdf.to_crs(PROJECT_CRS) if hasattr(stops_gdf, 'crs') else stops_gdf

            p_coords = np.vstack([
                parcels_proj.geometry.centroid.x,
                parcels_proj.geometry.centroid.y
            ]).T

            if hasattr(stops_proj, 'geometry'):
                s_coords = np.vstack([stops_proj.geometry.x, stops_proj.geometry.y]).T
            else:
                parcels['uplift_factor'] = 0.05
                s_coords = None

            if s_coords is not None and len(s_coords) > 0:
                tree = cKDTree(s_coords)
                dists_ft, _ = tree.query(p_coords, k=1)
                # EPSG:2965 distances are in US survey feet — convert to meters
                _US_FT_M = 0.3048006096012192
                dists = dists_ft * _US_FT_M
                parcels['dist_to_stop_m'] = dists
                parcels['uplift_factor'] = 0.20 * np.exp(-dists / 400.0)
                parcels['uplift_factor'] = parcels['uplift_factor'].clip(lower=0.0, upper=0.25)
            else:
                parcels['uplift_factor'] = 0.05
    else:
        baseline_trips = baseline_results['total_ridership']
        scenario_trips = scenario_results['total_ridership']

        if baseline_trips > 0:
            pct_increase = (scenario_trips - baseline_trips) / baseline_trips
        else:
            pct_increase = 1.0 if scenario_trips > 0 else 0.0

        uplift_factor = 0.05 * (pct_increase / 0.10)
        # Do not force positive uplift when ridership is flat or declining.
        uplift_factor = max(0.0, min(uplift_factor, 0.20))
        parcels['uplift_factor'] = uplift_factor

    parcels['scenario_lv'] = parcels['baseline_lv'] * (1.0 + parcels['uplift_factor'])
    parcels['uplift'] = parcels['scenario_lv'] - parcels['baseline_lv']
    total_uplift = float(parcels['uplift'].sum())

    baseline_trips = baseline_results.get('total_ridership', 0.0)
    scenario_trips = scenario_results.get('total_ridership', 0.0)

    result = {
        'uplift_by_parcel': parcels[['PARCEL_ID', 'baseline_lv', 'scenario_lv', 'uplift']],
        'total_uplift': total_uplift,
        'summary': {
            'baseline_ridership': baseline_trips,
            'scenario_ridership': scenario_trips,
            'ridership_increase': scenario_trips - baseline_trips,
            'ridership_pct_increase': ((scenario_trips - baseline_trips) / max(baseline_trips, 1.0)) * 100,
            'total_uplift_usd': total_uplift
        }
    }

    if tif_gdf is not None and len(tif_gdf) > 0:
        tif_id_col = 'tif_id' if 'tif_id' in tif_gdf.columns else (
            'name' if 'name' in tif_gdf.columns else tif_gdf.columns[0]
        )

        parcels_join = parcels.to_crs(PROJECT_CRS)
        tif_prepped = tif_gdf.to_crs(PROJECT_CRS)[[tif_id_col, 'geometry']].rename(columns={tif_id_col: 'tif_id'})

        parcels_tif = gpd.sjoin(
            parcels_join[['PARCEL_ID', 'uplift', 'geometry']],
            tif_prepped,
            how='inner',
            predicate='intersects'
        )

        if len(parcels_tif) > 0:
            tif_agg = parcels_tif.groupby('tif_id')['uplift'].sum().reset_index()
            tif_agg['captured_value'] = tif_agg['uplift'] * tif_capture_rate
            tif_agg['annual_revenue_est'] = tif_agg['captured_value'] * PROPERTY_TAX_RATE
            result['tif_revenue_annual'] = tif_agg
            result['summary']['tif_total_annual_revenue'] = float(tif_agg['annual_revenue_est'].sum())
        else:
            result['summary']['tif_total_annual_revenue'] = 0.0
    else:
        result['summary']['tif_total_annual_revenue'] = 0.0

    return result


def run_full_comparison(
    parcels_gdf: gpd.GeoDataFrame,
    baseline_stops_gdf: gpd.GeoDataFrame,
    new_stops_gdf: gpd.GeoDataFrame,
    stop_to_route_df: Optional[pd.DataFrame] = None,
    tif_gdf: Optional[gpd.GeoDataFrame] = None,
    land_value_col: str = 'CurLandAV',
    jobs_col: str = 'jobs',
    beta_walk: float = 0.01,
    tif_capture_rate: float = TIF_CAPTURE_RATE_CONSERVATIVE,
    out_dir: Optional[str] = None,
):
    """Run full baseline vs scenario comparison and write outputs."""
    baseline = run_baseline_scenario(parcels_gdf, baseline_stops_gdf, jobs_col=jobs_col, beta_walk=beta_walk)
    scenario = run_scenario_with_new_transit(
        parcels_gdf, baseline_stops_gdf, new_stops_gdf, jobs_col=jobs_col, beta_walk=beta_walk
    )

    uplift = compute_uplift_comparison(
        parcels_gdf, baseline, scenario,
        land_value_col=land_value_col, tif_gdf=tif_gdf, tif_capture_rate=tif_capture_rate
    )

    result = {
        'baseline_results': baseline,
        'scenario_results': scenario,
        'uplift_results': uplift,
        'summary': uplift['summary']
    }

    if stop_to_route_df is not None:
        route_agg = assign_boardings_to_routes(scenario['stop_ridership'], stop_to_route_df)
        result['route_results'] = route_agg
        result['summary']['total_route_boardings'] = float(route_agg['allocated_trips'].sum())

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        baseline['stop_ridership'].reset_index().to_csv(out_path / 'baseline_stop_ridership.csv', index=False)
        scenario['stop_ridership'].reset_index().to_csv(out_path / 'scenario_stop_ridership.csv', index=False)
        scenario['new_stop_ridership'].reset_index().to_csv(out_path / 'new_stop_ridership.csv', index=False)
        uplift['uplift_by_parcel'].to_csv(out_path / 'parcel_uplift.csv', index=False)

        if 'tif_revenue_annual' in uplift:
            uplift['tif_revenue_annual'].to_csv(out_path / 'tif_revenue_annual.csv', index=False)

        if 'route_results' in result:
            result['route_results'].reset_index().to_csv(out_path / 'route_boardings.csv', index=False)

        summary_df = pd.DataFrame([result['summary']])
        summary_df.to_csv(out_path / 'summary_metrics.csv', index=False)

    return result


# ============================================================================
# BATCH CORRIDOR EVALUATION
# ============================================================================

def _estimate_corridor_trip_distance(
    corridor_row: dict,
    stops_gdf: gpd.GeoDataFrame,
    od_distances_m: Optional[dict] = None,
) -> float:
    """Estimate average trip distance for a corridor."""
    cid = corridor_row['corridor_id']
    length_km = float(corridor_row['length_km'])

    if od_distances_m and cid in od_distances_m:
        return od_distances_m[cid]

    return length_km * 0.8 * 1000.0


def _stop_to_route_from_corridor(stops_gdf: gpd.GeoDataFrame, corridor_id: str) -> pd.DataFrame:
    df = stops_gdf[stops_gdf["corridor_id"] == corridor_id].copy()
    df["route_id"] = corridor_id
    df["freq"] = 1.0
    return df[["stop_id", "route_id", "freq"]]


def _evaluate_single_corridor(args):
    """Evaluate a single corridor (for multiprocessing)."""
    (
        corridor_row, parcels_gdf, stops_gdf, calibration,
        pop_col, jobs_col, fare_base, fare_apm,
        capture_rate, tif_buffer_m, land_value_col, discount_rate, years,
        apm_headway_min, bus_headway_min, service_reduction,
        od_weights, od_weight_col, od_distances_m, use_lodes_od,
    ) = args

    cid = corridor_row["corridor_id"]
    length_km = float(corridor_row["length_km"])
    n_stops = int(corridor_row["n_stops"])

    stops_subset = stops_gdf[stops_gdf["corridor_id"] == cid]

    if len(stops_subset) < 2:
        ridership = 0.0
        stop_potential = pd.DataFrame({"stop_id": [], "potential_trips": []}).set_index("stop_id")
    elif use_lodes_od:
        # Use hybrid LODES + gravity model for realistic commute patterns
        stop_potential = assign_ridership_hybrid(
            parcels_gdf, stops_subset,
            lodes_od_path=DEFAULT_LODES_OD_PATH,
            pop_col=pop_col, jobs_col=jobs_col,
            beta_walk=calibration.beta, max_walk_m=1000, walk_speed_kph=5,
            alpha_pop=0.6, alpha_jobs=0.4,
            lodes_weight=0.7,  # 70% work trips, 30% non-work
        )
    else:
        stop_potential = assign_ridership_enhanced(
            parcels_gdf, stops_subset,
            pop_col=pop_col, jobs_col=jobs_col,
            beta_walk=calibration.beta, max_walk_m=1000, walk_speed_kph=5,
            alpha_pop=0.6, alpha_jobs=0.4,
            mass_col=(od_weight_col if od_weights else None),
        )

    if not stop_potential.empty:
        corridor_trip_distance_m = _estimate_corridor_trip_distance(corridor_row, stops_gdf, od_distances_m)
        corridor_trip_distance_miles = corridor_trip_distance_m / 1609.34

        if 'distance_m' in stop_potential.columns:
            access_miles = stop_potential['distance_m'].values / 1609.34
            distances_miles = access_miles + corridor_trip_distance_miles
        else:
            distances_miles = np.full(len(stop_potential), corridor_trip_distance_miles)

        apm_headway_adjusted = apm_headway_min / service_reduction
        bus_headway_adjusted = bus_headway_min / service_reduction

        mode_shares = calculate_mode_shares_vectorized(
            distances=distances_miles,
            apm_headway=apm_headway_adjusted,
            bus_headway=bus_headway_adjusted,
            apm_fare=fare_apm,
            bus_fare=fare_base,
        )

        stop_potential["potential_trips"] = (
            stop_potential.get("potential_trips", pd.Series(dtype=float)) *
            calibration.scale *
            mode_shares['apm_share'].values
        )
    else:
        stop_potential["potential_trips"] = pd.Series(dtype=float)

    ridership = float(stop_potential["potential_trips"].sum()) if not stop_potential.empty else 0.0

    if not stop_potential.empty:
        if 'stop_id' not in stop_potential.columns:
            stop_potential = stop_potential.reset_index()
        stop_potential = stop_potential.merge(stops_subset[['stop_id', 'geometry']], on='stop_id', how='left')
        stop_potential = gpd.GeoDataFrame(stop_potential, geometry='geometry', crs=stops_subset.crs)

    stop_to_route = _stop_to_route_from_corridor(stops_subset, cid)
    route_alloc = assign_boardings_to_routes(
        stop_potential.set_index('stop_id') if 'stop_id' in stop_potential.columns else stop_potential,
        stop_to_route,
        route_freq_col="freq"
    )

    tif_new = stops_subset.to_crs(PROJECT_CRS).buffer(tif_buffer_m / US_SURVEY_FT_TO_M)
    tif_gdf_new = gpd.GeoDataFrame(geometry=tif_new, crs=PROJECT_CRS).to_crs(4326)
    tif_gdf_new["name"] = [f"{cid}_TIF"] * len(tif_gdf_new)

    baseline = {"total_ridership": 0.0, "stop_ridership": pd.DataFrame(columns=["stop_id", "potential_trips"])}
    scenario = {"total_ridership": ridership, "stop_ridership": stop_potential, "new_stop_ridership": stop_potential}
    uplift = compute_uplift_comparison(
        parcels_gdf, baseline, scenario,
        land_value_col=land_value_col, tif_gdf=tif_gdf_new, tif_capture_rate=capture_rate
    )

    annual_tif_usd = uplift.get("summary", {}).get("tif_total_annual_revenue", 0.0)
    capex = capex_musd(length_km, n_stops)
    finance = npv_irr(annual_tif_usd / 1e6, capex, years=years, discount_rate=discount_rate)

    return {
        "corridor_id": cid,
        "length_km": length_km,
        "n_stops": n_stops,
        "ridership": scenario["total_ridership"],
        "ridership_units": "monthly_riders",
        "annual_tif_usd": annual_tif_usd,
        "capex_musd": capex,
        "npv_musd": finance["npv_musd"],
        "irr": finance["irr"],
    }


def evaluate_corridors(
    parcels_gdf: gpd.GeoDataFrame,
    corridors_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    calibration,
    tif_gdf: gpd.GeoDataFrame,
    land_value_col: str = "CurLandAV",
    jobs_col: str = "jobs_alloc",
    pop_col: str = "pop_alloc",
    fare_base: float = 2.0,   # CityBus 2026 fare
    fare_apm: float = 2.0,   # CityBus 2026 integrated fare
    tech_params_path: Path = Path("data/raw/System Technology Parameters.csv"),
    technology: str = "Automated People Mover",
    capture_rate: float = TIF_CAPTURE_RATE_CONSERVATIVE,
    discount_rate: float = 0.05,
    years: int = 25,
    tif_buffer_m: float = 804.672,
    use_multiprocessing: bool = True,
    apm_headway_min: float = 5.0,
    bus_headway_min: float = 30.0,
    service_reduction: float = 1.0,
    od_weights_csv: Optional[Path] = None,
    od_weight_col: str = "od_trips_total",
    use_lodes_od: bool = True,
) -> pd.DataFrame:
    """Evaluate each corridor: ridership, uplift, TIF revenue, NPV/IRR."""
    # tech_params removed — cost model uses constants from financial_params.py
    _ = tech_params_path, technology  # kept in signature for backward compat

    od_weights_flag = False
    parcels_aug = parcels_gdf.copy()
    if od_weights_csv is not None:
        try:
            od_df = pd.read_csv(od_weights_csv)
            out = od_df.groupby('origin_parcel')['trips'].sum().rename('od_out')
            inc = od_df.groupby('dest_parcel')['trips'].sum().rename('od_in')
            od_tot = pd.concat([
                out.rename_axis('PARCEL_ID'),
                inc.rename_axis('PARCEL_ID')
            ], axis=1).fillna(0.0)
            od_tot[od_weight_col] = od_tot['od_out'] + od_tot['od_in']
            parcels_aug = parcels_aug.merge(od_tot[[od_weight_col]].reset_index(), on='PARCEL_ID', how='left')
            parcels_aug[od_weight_col] = parcels_aug[od_weight_col].fillna(0.0)

            alpha_pop, alpha_jobs = 0.6, 0.4
            base_mass = float((alpha_pop * parcels_aug.get(pop_col, 0.0) + alpha_jobs * parcels_aug.get(jobs_col, 0.0)).sum())
            od_mass = float(parcels_aug[od_weight_col].sum())
            if od_mass > 0 and base_mass > 0:
                scale_factor = base_mass / od_mass
                parcels_aug[od_weight_col] = parcels_aug[od_weight_col] * scale_factor
            od_weights_flag = True
        except Exception:
            pass

    od_distances_m = {}
    if od_weights_csv is not None:
        try:
            od_df = pd.read_csv(od_weights_csv)
            if 'origin_parcel' in od_df.columns and 'dest_parcel' in od_df.columns:
                od_origins = parcels_aug.set_index('PARCEL_ID').geometry.to_dict()
                od_dests = parcels_aug.set_index('PARCEL_ID').geometry.to_dict()

                distances_list = []
                for _, od_row in od_df.iterrows():
                    oid = od_row.get('origin_parcel')
                    did = od_row.get('dest_parcel')
                    if oid in od_origins and did in od_dests:
                        try:
                            # Approximate meters from WGS84 degree distance
                            # At ~40.4°N: lon degree ≈ 84937m, lat degree ≈ 110540m
                            _dx = (od_origins[oid].centroid.x - od_dests[did].centroid.x) * 84937
                            _dy = (od_origins[oid].centroid.y - od_dests[did].centroid.y) * 110540
                            dist_m = np.sqrt(_dx**2 + _dy**2)
                            distances_list.append(dist_m)
                        except:
                            pass

                if distances_list:
                    avg_od_distance = np.mean(distances_list)
                    for _, row in corridors_gdf.iterrows():
                        od_distances_m[row['corridor_id']] = avg_od_distance
        except Exception:
            pass

    # Check if LODES data is available
    lodes_available = use_lodes_od and DEFAULT_LODES_OD_PATH.exists()

    corridor_args = [
        (
            row, parcels_aug, stops_gdf, calibration,
            pop_col, jobs_col, fare_base, fare_apm,
            capture_rate, tif_buffer_m, land_value_col, discount_rate, years,
            apm_headway_min, bus_headway_min, service_reduction,
            od_weights_flag, od_weight_col, od_distances_m, lodes_available,
        )
        for _, row in corridors_gdf.iterrows()
    ]

    if use_multiprocessing and len(corridor_args) > 1:
        n_workers = max(1, cpu_count() - 1)
        with Pool(n_workers) as pool:
            results = pool.map(_evaluate_single_corridor, corridor_args)
    else:
        results = [_evaluate_single_corridor(args) for args in corridor_args]

    return pd.DataFrame(results)

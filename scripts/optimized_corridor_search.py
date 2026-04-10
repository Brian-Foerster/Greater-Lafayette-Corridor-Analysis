#!/usr/bin/env python
"""
Station-First APM Corridor Search
===================================

Corridors are defined as ordered lists of station IDs (OSM intersection nodes),
not line geometries. Alignment is generated only for final selected corridors.

Pipeline:
1. STATION SITING - MCLP on OSM road graph intersections (degree >= 3)
2. INITIAL GENERATION - Anchor-pair paths, demand-biased walks, radial corridors
3. SCORING - Per-station KDTree catchment with distance decay (beta=0.00173)
4. NSGA-II EVOLUTION - Station-set mutations/crossovers on Pareto front
5. DIVERSITY SELECTION - MMR with Jaccard overlap on station neighborhoods
6. ALIGNMENT - Road-graph shortest path with Chaikin smoothing (final output only)

Usage:
    python scripts/optimized_corridor_search.py [--iterations 15] [--output 17]
"""
from __future__ import annotations

import argparse
import math
import random
import hashlib
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point

warnings.filterwarnings("ignore", category=FutureWarning)

import logging
logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]

from functools import lru_cache
from src.bus_network import DEFAULT_APM_HEADWAY_MIN as _BN_APM_HW
from src.spatial_constants import PROJECT_CRS, WALK_CATCHMENT_M, FEEDER_CATCHMENT_M

# KDTree query cache: keyed by (station_tuple, query_id) where query_id
# distinguishes parcel vs OD-origin vs OD-dest queries.  Avoids rebuilding
# the same 5-10 point tree and re-querying 61K parcels when mutations
# produce the same station subset as a previous generation.
_KDTREE_QUERY_CACHE: Dict[tuple, tuple] = {}
_KDTREE_CACHE_MAX = 64


def _cached_kdtree_query(
    station_key: tuple,
    query_id: str,
    station_coords: np.ndarray,
    query_coords: np.ndarray,
) -> tuple:
    """Cache-aware KDTree query.  Returns (dists, indices)."""
    cache_key = (station_key, query_id)
    result = _KDTREE_QUERY_CACHE.get(cache_key)
    if result is not None:
        return result
    tree = cKDTree(station_coords)
    dists, idxs = tree.query(query_coords, k=1)
    # Evict oldest entries if cache too large
    if len(_KDTREE_QUERY_CACHE) >= _KDTREE_CACHE_MAX:
        # Remove oldest half
        keys = list(_KDTREE_QUERY_CACHE.keys())
        for k in keys[:len(keys) // 2]:
            del _KDTREE_QUERY_CACHE[k]
    _KDTREE_QUERY_CACHE[cache_key] = (dists, idxs)
    return dists, idxs

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

# Distance decay calibrated to TCRP Report 165
DECAY_BETA = 0.00173  # 50% at 400m

# Mode choice coefficients (from mode_choice.py)
BETA_IVT = -0.055
BETA_WAIT = -0.090
BETA_ACCESS = -0.120
BETA_COST = -0.035
ASC_APM = 0.18

# ── APM Vehicle Specifications ───────────────────────────────────────
# Innovia APM 300 class rubber-tire AGT (Alstom/Bombardier).
# ASCE 21.2-2008 "Automated People Movers — Standard for APMs"
APM_SPEED_KPH = 40                                         # urban service speed (80 kph design max)
APM_LINE_SPEED_MS = APM_SPEED_KPH / 3.6                   # 11.11 m/s
APM_SERVICE_ACCEL_MS2 = 1.0                                # service accel (Innovia/Crystal Mover spec)
APM_SERVICE_DECEL_MS2 = 1.0                                # service braking (symmetric)
APM_LATERAL_ACCEL_LIMIT_MS2 = 0.687                        # 0.07g comfort target (ASCE 21 max is 0.10g)
APM_MIN_CURVE_RADIUS_M = 50.0                              # ASCE 21 restricted service min (25m is depot-only)
APM_MAX_GRADE_PCT = 6.0                                    # standard AGT maximum grade
# Derived: radius at which line speed can be maintained
APM_FULL_SPEED_RADIUS_M = APM_LINE_SPEED_MS ** 2 / APM_LATERAL_ACCEL_LIMIT_MS2  # ~180m
# Curved guideway construction premium (industry 1.5-2.5× for tight curves)
CURVE_CONSTRUCTION_PREMIUM = 1.5
# Effective speed floor: reject if curves degrade avg speed below this × line speed
EFFECTIVE_SPEED_FLOOR_FRACTION = 0.50                      # reject if eff_speed < 20 kph

APM_HEADWAY_MIN = _BN_APM_HW  # from bus_network (5 min)
APM_FARE = 2.0   # CityBus 2026 integrated fare (was 0.0 -- unrealistic)
WALK_SPEED_KPH = 5

# EPSG:2965 (Indiana State Plane East) uses US survey feet, not meters.
# All distances from projected coords must be converted.
US_SURVEY_FT_TO_M = 0.3048006096012192

# Corridor constraints (distances in meters)
MIN_LENGTH_KM = 3.0  # APM minimum: 3 km (~6 min ride); shorter is walkable
MAX_LENGTH_KM = 25.0
MIN_STOP_SPACING_M = 500.0   # meters
MAX_STOP_SPACING_M = 1500.0  # meters

# Geometric constraints for buildable APM alignments
# Financial parameters (shared source of truth)
from src.financial_params import (
    CAPITAL_COST_PER_KM, BOND_RATE, DEBT_TERM_YEARS as BOND_TERM,
    PROPERTY_TAX_RATE, TIF_CAPTURE_RATE_CONSERVATIVE, TIF_YEARS,
    compute_capital_cost,
    CAPITAL_COST_GUIDEWAY_PER_KM,
    effective_tif_tax_rate,
    TIF_AREA_TYPE_DEFAULT,
)
from src.bus_network import compute_apm_headway

# Student ridership proxy constants (mirrors land_use_transport_model.py)
STUDENT_CAMPUS_TRIP_RATE = 2.0        # daily trips per campus-affiliated person
STUDENT_PRESENCE_FACTOR = 0.25        # 25% of enrolled students present/transit-eligible
FACULTY_PRESENCE_FACTOR = 0.10        # 10% of faculty/staff
STUDENT_APM_SHARE_ESTIMATE = 0.20     # simplified average (full model uses per-corridor logit)
MIN_TIF_VIABILITY_USD = 50_000.0      # annual TIF floor to survive as a corridor

# Trip-length feasibility: penalise short corridors that claim demand from
# parcels whose trips are longer than the corridor can serve.  NHTS 2017
# small-metro average all-purpose trip distance is ~8 km.  Coverage factor:
#   coverage = min(corridor_length_km / MEAN_TRIP_DISTANCE_KM, 1.0)
# Applied to weighted_demand after corridor length is known.
MEAN_TRIP_DISTANCE_KM = 8.0

# Circuity constraints — loose backstops (real filtering via physics-based
# curve speed model in compute_curve_speed_penalties).
MAX_SEGMENT_CIRCUITY = 1.75   # road_dist / euclidean per consecutive station pair
MAX_CORRIDOR_CIRCUITY = 1.75  # total road_dist / endpoint euclidean

# Road-path bearing reversal: if the actual road path arriving at a station
# differs from the departure bearing by more than this, the APM would need
# to make a near-U-turn on the road network — physically unrealistic for
# fixed guideway even if station-to-station geometry looks fine.
# 55° allows standard intersection turns but rejects any reversal that
# would require a switchback or tight S-curve.  Tightened from 100° which
# was too permissive — allowed near-backtracking that the curve speed
# model would later penalize anyway.
PATH_BEARING_REVERSAL_DEG = 75

# Road-class cost surface (pathfinding edge-weight multipliers)
ROAD_CLASS_COST_SURFACE = {
    "motorway": 5.0, "motorway_link": 5.0,
    "trunk": 2.0, "trunk_link": 2.0,
    "primary": 1.0, "primary_link": 1.0,
    "secondary": 1.0, "secondary_link": 1.0,
    "tertiary": 1.3, "tertiary_link": 1.3,
    "residential": 2.5,
    "unclassified": 2.0,
    "service": 3.0, "living_street": 3.0,
}

# Transit propensity of employment by zone character (3-tier).
# Jobs in each zone are weighted by their tier's transit propensity
# in the demand scoring.  Population is unweighted.
#
# High (1.0):  office, institutional, walkable neighborhood commercial
#              (Cervero 2006: ~15-20% transit mode share near rail)
# Medium (0.5): residential service jobs, mixed commercial
#              (~5-8% mode share)
# Low (0.20):  auto-oriented retail, industrial, warehouse, agricultural
#              (PSRC: "do not support high ridership"; ~1-3% mode share)
_ZONE_PROPENSITY_TIER = {
    # High: office, institutional, walkable commercial
    "CB": "high", "CBW": "high", "OR": "high",
    "NB": "high", "NBU": "high",
    "PDMX": "high", "PDCC": "high", "MRU": "high",
    # Medium: residential zones (service jobs), duplex/townhome
    "R1": "med", "R1A": "med", "R1B": "med", "R1T": "med", "R1U": "med",
    "R2": "med", "R2U": "med",
    "R3": "med", "R3U": "med", "R3W": "med", "R4W": "med",
    "PDRS": "med", "PDNR": "med",
    # Low: auto-oriented commercial, industrial, agricultural
    "GB": "low", "SHADELAND": "low",
    "I2": "low", "I3": "low",
    "A": "low", "AA": "low", "AW": "low",
    "FP": "low", "RE": "low",
}
_TIER_WEIGHT = {"high": 1.0, "med": 0.50, "low": 0.20}
_DEFAULT_TIER = "med"

# Search parameters
# Bidirectional overlap fraction diversity (replaces Jaccard on station sets)
OVERLAP_BUFFER_M = 400.0              # Two polyline points within 400m are "same area"
OVERLAP_THRESHOLD = 0.75              # >75% bidirectional overlap = near-duplicate
_OVERLAP_BUFFER_FT = OVERLAP_BUFFER_M / US_SURVEY_FT_TO_M  # pre-convert to CRS units
NETWORK_TRANSFER_RADIUS_M = 1200.0
NETWORK_SYNERGY_WEIGHT_DEFAULT = 0.20

# Station-first search parameters
MIN_INTERSECTION_DEGREE = 3
STATION_ROAD_CLASSES = frozenset({
    "primary", "secondary", "tertiary",
    "primary_link", "secondary_link", "tertiary_link",
})
MIN_STATIONS_PER_CORRIDOR = 4
MAX_STATIONS_PER_CORRIDOR = 12
MAX_WALK_CANDIDATES = 25  # Walk visits up to this many candidates; DP picks the best 4-12
ROAD_CIRCUITY_FACTOR = 1.30     # empirical road-path / Euclidean ratio
STATION_PROXIMITY_M = 400.0     # two stations within this are "same" for diversity
ROAD_CLASS_WEIGHTS = {
    "primary": 1.0, "secondary": 1.0, "tertiary": 1.5,
    "residential": 3.0, "unclassified": 2.0,
}

# ---------------------------------------------------------------------------
# Geographic barriers (approximate geometries in EPSG:4326)
# ---------------------------------------------------------------------------
# Wabash River centerline through the Greater Lafayette urban core.
# Crossing requires a bridge — modest cost premium.
# Coordinates from OSM bridge midpoints (bridge=yes edges spanning the river).
_WABASH_RIVER_COORDS_4326 = [
    (-86.8920, 40.4530),  # Sagamore Pkwy bridge midpoint
    (-86.8970, 40.4250),  # SR-26 / Old US-231 bridge midpoint
    (-86.8980, 40.4190),  # Columbia St / South St bridge midpoint
    (-86.9000, 40.4050),  # south of Brown St (river bends west)
    (-86.9020, 40.3940),  # Old US-231 south bridge midpoint
]
# I-65 runs N-S on the east edge of Lafayette
_I65_COORDS_4326 = [
    (-86.8450, 40.5100),
    (-86.8480, 40.4600),
    (-86.8500, 40.4100),
    (-86.8520, 40.3600),
    (-86.8540, 40.3400),
]
# Norfolk Southern mainline — active freight railroad through downtown Lafayette
_RAILROAD_COORDS_4326 = [
    (-86.8950, 40.4500),
    (-86.8920, 40.4350),
    (-86.8900, 40.4200),
    (-86.8880, 40.4000),
]

# Fixed cost per barrier crossing (USD, added to capital cost).
# These are INCREMENTAL costs beyond standard elevated guideway (already in
# per-km rate).  An elevated APM runs at 6-8m AGL — already above railroad
# clearance (7.1m AREMA/FRA) — so rail crossings only need a longer span
# across the right-of-way, not a new grade separation structure.
#   River: deep foundations in water, environmental permitting, different
#     structural system.  $35M is appropriate for Wabash-scale crossing.
#   Highway: I-65 is 60-80m wide; longer span with special structural
#     design needed, but guideway is already elevated.  $15-25M realistic.
#   Railroad: 30-50m span across rail ROW + FRA/railroad permitting and
#     construction flagging.  $3-7M incremental on an elevated guideway.
BARRIER_RIVER_COST_USD = 35_000_000    # $35M per river crossing (Wabash-scale)
BARRIER_HIGHWAY_COST_USD = 20_000_000  # $20M per highway crossing (I-65)
BARRIER_RAILROAD_COST_USD = 5_000_000  # $5M per railroad crossing (CSX/NS)

# ---------------------------------------------------------------------------
# Key destination anchors (EPSG:4326) — ensures geographic diversity
# ---------------------------------------------------------------------------
ANCHOR_DESTINATIONS = [
    # (lon, lat, label)
    (-86.9212, 40.4237, "purdue_campus_center"),       # Purdue campus core
    (-86.8930, 40.4170, "downtown_lafayette"),         # Lafayette CBD — Columbia St / Main St
    (-86.9070, 40.4260, "west_lafayette_downtown"),    # State St / Chauncey Hill — highest-value commercial
    (-86.9250, 40.4650, "purdue_research_park"),       # 725 acres, ~4,000 jobs
    (-86.8037, 40.4012, "iu_health_arnett"),           # IU Health Arnett — 5165 McCarty Ln (parcel 790831100010000038)
    (-86.8498, 40.3933, "tippecanoe_mall"),            # Tippecanoe Mall — 2415 Sagamore Pkwy S (parcel-verified)
    (-86.9100, 40.4600, "sagamore_north"),             # North corridor
    (-86.9350, 40.4400, "west_lafayette_student_housing"),  # Student/staff housing west of campus
    (-86.8345, 40.3940, "franciscan_health"),          # Franciscan Health — 1701 S Creasy Ln (parcel-verified, ~2,200 employees)
    # Outlying demand generators for geographic diversity (parcel-verified 2026-03-21)
    (-86.8278, 40.4179, "east_lafayette_employment"),    # 100 Prelude Ln office park — $119M cell AV, 54 comm/inst parcels
    (-86.8925, 40.3823, "south_lafayette_teal_rd"),      # Teal Rd school ($18.5M) + mixed commercial/industrial — $147M cell AV
    (-86.9417, 40.3930, "lilly_rd_industrial"),           # Caterpillar/industrial campus — $112M cell AV, multiple $28M parcels
    # Additional outlying anchors for corridor diversity (2026-03-21)
    (-86.7950, 40.4300, "east_sagamore_pkwy"),           # East Sagamore Pkwy commercial corridor near I-65
    (-86.8700, 40.3700, "south_sagamore_sr26"),          # South Sagamore / SR-26 South intersection area
    (-86.9150, 40.4800, "wabash_north_us52"),            # Wabash Ave / US-52 north interchange area
    (-86.9380, 40.4115, "purdue_airport"),               # Purdue University Airport / aerospace district
]

# Coordinate transformers
_to_proj = Transformer.from_crs("EPSG:4326", PROJECT_CRS, always_xy=True)
_to_4326 = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)

# Cache projected coordinates by OSM node ID to avoid repeated _to_proj.transform
# calls for the same nodes across 12K+ score_station_set evaluations.
_NODE_PROJ_CACHE: Dict[int, Tuple[float, float]] = {}


def _ensure_enriched_parcels() -> Path:
    """Auto-generate parcels_enriched_final.geojson if missing."""
    from src.ensure_enriched import ensure_enriched_parcels
    target = PROC_DIR / "parcels_enriched_final.geojson"
    return ensure_enriched_parcels(target, raw_dir=RAW_DIR, proc_dir=PROC_DIR)


def load_all_data() -> dict:
    """Load and prepare all input datasets."""
    logger.info("Loading data...")

    # Ensure enriched parcels exist (auto-generate from raw if missing)
    _ensure_enriched_parcels()

    parcel_candidates = [
        PROC_DIR / "parcels_enriched_final.geojson",
        PROC_DIR / "parcels_improved_gravity_v2.geojson",
        PROC_DIR / "parcels_enriched.geojson",
        PROC_DIR / "sales_complete.geojson",
        PROC_DIR / "parcels_clean.geojson",
        PROC_DIR / "parcels_enriched_with_access_test2.geojson",
    ]
    existing_candidates = [p for p in parcel_candidates if p.exists()]
    if not existing_candidates:
        candidate_str = ", ".join(str(p) for p in parcel_candidates)
        raise FileNotFoundError(
            "Missing parcels input. Expected one of: "
            f"{candidate_str}"
        )
    # Prefer the largest artifact to avoid selecting tiny smoke-test fixtures.
    parcel_path = max(existing_candidates, key=lambda p: p.stat().st_size)
    # Fast GeoJSON load: json.load + from_features bypasses fiona (10x faster)
    import json as _json_mod
    with open(parcel_path) as _pf:
        _gj = _json_mod.load(_pf)
    parcels = gpd.GeoDataFrame.from_features(_gj["features"], crs="EPSG:4326")
    del _gj
    logger.debug(f"  Parcels: {len(parcels):,} ({parcel_path.name})")

    # OD flows (LODES-based)
    od_flows = pd.read_csv(
        PROC_DIR / "od_parcel_flows_lodes.csv",
        dtype={"origin_parcel": str, "dest_parcel": str},
    )
    logger.debug(f"  OD flows: {len(od_flows):,}")

    # Bus stops
    bus_stops_path = RAW_DIR / "CityBus2025" / "stops.txt"
    if bus_stops_path.exists():
        bus_df = pd.read_csv(bus_stops_path)
        bus_stops = gpd.GeoDataFrame(
            bus_df,
            geometry=gpd.points_from_xy(bus_df.stop_lon, bus_df.stop_lat),
            crs="EPSG:4326",
        )
    else:
        bus_stops = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    logger.debug(f"  Bus stops: {len(bus_stops):,}")

    # Property values for TIF estimation
    prop_path = PROC_DIR / "property_values_25yr_all_scenarios_fixed.csv"
    prop_values = pd.read_csv(prop_path) if prop_path.exists() else None
    if prop_values is not None:
        logger.debug(f"  Property values: {len(prop_values):,} rows")

    # Population and jobs from enriched parcels (census-based)
    for col, default_col, label in [
        ("pop_alloc", "population", "population"),
        ("jobs_combined", "jobs_lehd_wac", "jobs"),
    ]:
        if col not in parcels.columns:
            if default_col in parcels.columns:
                parcels[col] = pd.to_numeric(parcels[default_col], errors="coerce").fillna(0)
            else:
                raise ValueError(
                    f"parcels file has no {label} column ({col} or {default_col}). "
                    "Run: python scripts/allocate_pop_and_employment.py"
                )
    # Weight jobs by transit propensity of the parcel's zone.
    # Population is unweighted — a resident is a potential rider regardless
    # of surrounding land use.  Jobs are weighted because employment type
    # determines whether workers realistically arrive by transit.
    zone_col_for_prop = next((c for c in ["zone_code", "RefName", "ZONE"] if c in parcels.columns), None)
    if zone_col_for_prop is not None:
        _tier = parcels[zone_col_for_prop].astype(str).map(_ZONE_PROPENSITY_TIER).fillna(_DEFAULT_TIER)
        _job_propensity = _tier.map(_TIER_WEIGHT).astype(float)
        _n_low = int((_tier == "low").sum())
        _n_high = int((_tier == "high").sum())
        logger.debug(f"  Job propensity tiers: {_n_high} high, "
                     f"{int((_tier == 'med').sum())} med, {_n_low} low")
    else:
        _job_propensity = pd.Series(_TIER_WEIGHT[_DEFAULT_TIER], index=parcels.index)
    existing_demand = parcels["pop_alloc"] + parcels["jobs_combined"] * _job_propensity

    # Campus demand proxy: exempt parcels (PropClass 600-699) get zero dev_potential
    # for TIF (correct — they don't generate tax revenue), but should still contribute
    # to demand_wt for station siting.  Use institutional_weight as proxy.
    CAMPUS_DEMAND_PROXY = 500.0  # Calibrated so inst_wt=3-5 ≈ top-10% residential parcel
    _ppc = pd.to_numeric(parcels.get("PriorPropClass", pd.Series(dtype=float)), errors="coerce").fillna(0)
    _is_exempt = (_ppc >= 600) & (_ppc < 700)
    _inst_col = pd.to_numeric(parcels["institutional_weight"] if "institutional_weight" in parcels.columns else pd.Series(1.0, index=parcels.index), errors="coerce").fillna(1.0)
    # Campus demand component: only for exempt parcels with institutional_weight > 1
    _campus_demand = np.where(
        _is_exempt & (_inst_col > 1.0),
        _inst_col * CAMPUS_DEMAND_PROXY,
        0.0,
    )
    n_campus_proxy = int((_campus_demand > 0).sum())
    if n_campus_proxy > 0:
        logger.debug(f"  Campus demand proxy: {n_campus_proxy} parcels, "
              f"total proxy demand: {_campus_demand.sum():.0f}")

    # existing_demand_with_campus: used for station qualification (Change 3)
    # Includes campus proxy but NOT speculative dev_potential
    parcels["existing_demand"] = existing_demand + _campus_demand

    # Development potential: max buildable sqft from zoning × discount
    from src.developer_proforma import ZONING_MATRIX
    zone_col = next((c for c in ["zone_code", "RefName", "ZONE"] if c in parcels.columns), None)
    if zone_col is not None:
        # Compute parcel area in sqft from geometry
        _area_sqft = parcels.to_crs(PROJECT_CRS).geometry.area * 10.7639
        _far = parcels[zone_col].astype(str).map(
            {k: v[0] for k, v in ZONING_MATRIX.items()}
        ).fillna(0.0)
        dev_capacity_sqft = _area_sqft * _far
        parcels["dev_capacity_sqft"] = dev_capacity_sqft
        # Improvement ratio: how built-out a parcel already is (0=vacant, 1=fully improved)
        if "CurImpAV" in parcels.columns and "CurTotAV" in parcels.columns:
            _imp_av = pd.to_numeric(parcels["CurImpAV"], errors="coerce").fillna(0)
            _tot_av_safe = np.maximum(pd.to_numeric(parcels["CurTotAV"], errors="coerce").fillna(0), 1.0)
            parcels["improvement_ratio"] = _imp_av / _tot_av_safe
        else:
            parcels["improvement_ratio"] = 0.0
        # TIF-eligible: not exempt and not already fully built out
        parcels["tif_eligible"] = ~_is_exempt & (parcels["improvement_ratio"] < 0.8)
        _POTENTIAL_SQFT_TO_POP = 2.5 / 900.0
        dev_potential_pop = dev_capacity_sqft * _POTENTIAL_SQFT_TO_POP
        remaining_potential = np.maximum(dev_potential_pop - parcels["pop_alloc"], 0.0)
        DEV_POTENTIAL_DISCOUNT = 0.15
        # demand_wt: full blended metric for scoring (campus proxy + dev_potential)
        parcels["demand_wt"] = existing_demand + _campus_demand + DEV_POTENTIAL_DISCOUNT * remaining_potential
        logger.debug(f"  Demand (existing): {existing_demand.sum():.0f}, "
              f"campus proxy: {_campus_demand.sum():.0f}, "
              f"dev potential (raw): {dev_potential_pop.sum():.0f}, "
              f"blended: {parcels['demand_wt'].sum():.0f}")
    else:
        parcels["dev_capacity_sqft"] = 0.0
        parcels["improvement_ratio"] = 0.0
        parcels["tif_eligible"] = False
        parcels["demand_wt"] = parcels["existing_demand"]
        logger.debug(f"  Demand: pop={parcels['pop_alloc'].sum():.0f}, jobs={parcels['jobs_combined'].sum():.0f}")

    # Institutional weights for student ridership proxy
    inst_weights = None
    if "institutional_weight" in parcels.columns:
        inst_weights = pd.to_numeric(
            parcels["institutional_weight"], errors="coerce"
        ).fillna(1.0).values.astype(np.float64)
        n_campus = int((inst_weights > 1.0).sum())
        logger.debug(f"  Institutional weights: {n_campus} campus parcels (from parcel column)")
    else:
        cache_path = PROC_DIR / "institutional_weights_cache.npy"
        if cache_path.exists():
            try:
                cached = np.load(cache_path)
                if len(cached) == len(parcels):
                    inst_weights = cached.astype(np.float64)
                    n_campus = int((inst_weights > 1.0).sum())
                    logger.debug(f"  Institutional weights: {n_campus} campus parcels (from cache)")
            except Exception:
                pass
    if inst_weights is None:
        inst_weights = np.ones(len(parcels), dtype=np.float64)
        logger.debug("  Institutional weights: not available (student proxy disabled)")

    return {
        "parcels": parcels,
        "od_flows": od_flows,
        "bus_stops": bus_stops,
        "prop_values": prop_values,
        "institutional_weights": inst_weights,
    }


# ============================================================================
# STATION-FIRST CORRIDOR SEARCH  (new pipeline)
# ============================================================================
#
# Corridors are defined as ordered lists of station IDs (OSM intersection
# nodes).  Alignment geometry is generated only for final selected corridors
# via road-graph shortest paths.

def build_candidate_stations(
    G,
    parcels_gdf: gpd.GeoDataFrame,
    bus_stops_proj: Optional[np.ndarray] = None,
    bus_tree: Optional[cKDTree] = None,
) -> dict:
    """Extract candidate APM station sites from the OSM road graph.

    Candidate stations are intersections (degree >= 3) adjacent to at least
    one primary/secondary/tertiary road.  Each candidate is pre-scored for
    demand coverage, TIF potential, and bus competition.

    Returns a dict with arrays indexed by *local* position (0..N-1) and a
    mapping from local index to OSM node ID.
    """
    import networkx as nx

    nodes = list(G.nodes(data=True))
    node_ids_all = np.array([n[0] for n in nodes])

    # --- Filter to major intersections ---
    # degree in the undirected sense (osmnx stores a MultiDiGraph)
    G_undir = G.to_undirected()
    degree_map = dict(G_undir.degree())

    keep_mask = np.zeros(len(nodes), dtype=bool)
    for i, (nid, _data) in enumerate(nodes):
        if degree_map.get(nid, 0) < MIN_INTERSECTION_DEGREE:
            continue
        # Check if at least one incident edge is a major road class
        has_major = False
        for _u, _v, edata in G.edges(nid, data=True):
            hw = edata.get("highway", "")
            if isinstance(hw, list):
                hw = hw[0] if hw else ""
            if hw in STATION_ROAD_CLASSES:
                has_major = True
                break
        if not has_major:
            # Also check incoming edges (DiGraph)
            for _u, _v, edata in G.in_edges(nid, data=True):
                hw = edata.get("highway", "")
                if isinstance(hw, list):
                    hw = hw[0] if hw else ""
                if hw in STATION_ROAD_CLASSES:
                    has_major = True
                    break
        if has_major:
            keep_mask[i] = True

    kept_indices = np.flatnonzero(keep_mask)
    if len(kept_indices) < 20:
        # Fallback: relax to degree >= 2 on any road class
        logger.debug(f"  Warning: only {len(kept_indices)} stations with degree >= {MIN_INTERSECTION_DEGREE}")
        logger.debug("  Relaxing to degree >= 2...")
        for i, (nid, _data) in enumerate(nodes):
            if degree_map.get(nid, 0) >= 2 and not keep_mask[i]:
                keep_mask[i] = True
        kept_indices = np.flatnonzero(keep_mask)

    n_intersection = int(keep_mask.sum())

    # --- Demand-qualified stations on minor roads ---
    # APM stations can be built anywhere; include nodes near top-demand
    # parcels even on minor roads (campus interior, medical campus, etc.)
    DEMAND_QUALIFY_RADIUS_FT = 500.0 / US_SURVEY_FT_TO_M  # ~152m, one short block
    _p_coords_tmp = np.array([
        _to_proj.transform(G.nodes[n[0]]["x"], G.nodes[n[0]]["y"])
        for n in nodes
    ])
    # Station qualification uses existing demand only (pop + jobs + campus proxy),
    # NOT speculative dev_potential.  The full demand_wt (with dev_potential) is
    # still used for scoring (demand_coverage) downstream.
    _p_existing_demand = parcels_gdf["existing_demand"].values.astype(np.float64) \
        if "existing_demand" in parcels_gdf.columns \
        else parcels_gdf["demand_wt"].values.astype(np.float64)
    _parcels_proj = parcels_gdf.to_crs(PROJECT_CRS)
    _p_parcel_coords = np.column_stack([
        _parcels_proj.geometry.centroid.x.values,
        _parcels_proj.geometry.centroid.y.values,
    ])
    _positive = _p_existing_demand > 0
    if _positive.any():
        _threshold = np.percentile(_p_existing_demand[_positive], 95)
        _high_mask = _p_existing_demand >= _threshold
        _high_coords = _p_parcel_coords[_high_mask]
        if len(_high_coords) > 0:
            _hd_tree = cKDTree(_high_coords)
            n_demand_added = 0
            for i, (nid, _data) in enumerate(nodes):
                if keep_mask[i]:
                    continue
                if degree_map.get(nid, 0) < 1:
                    continue
                nearby = _hd_tree.query_ball_point(_p_coords_tmp[i], DEMAND_QUALIFY_RADIUS_FT)
                if len(nearby) > 0:
                    keep_mask[i] = True
                    n_demand_added += 1
            logger.debug(f"  Added {n_demand_added} demand-qualified stations on minor roads")

    # --- Explicit POI stations ---
    EXPLICIT_STATIONS = [
        (-86.9212, 40.4237, "purdue_memorial_union"),
        (-86.9446, 40.4445, "discovery_park"),
        (-86.9176, 40.4316, "ross_ade_stadium"),
        (-86.8037, 40.4012, "iu_health_arnett"),         # 5165 McCarty Ln (parcel-verified)
        (-86.8498, 40.3933, "tippecanoe_mall"),           # 2415 Sagamore Pkwy S (parcel-verified)
        (-86.8950, 40.4210, "citybus_transfer_center"),   # 316 N 3rd St
        (-86.9250, 40.4650, "purdue_research_park"),
        (-86.9070, 40.4260, "chauncey_village"),
        (-86.8345, 40.3940, "franciscan_health"),         # 1701 S Creasy Ln (parcel-verified)
    ]
    _all_node_tree = cKDTree(_p_coords_tmp)
    n_explicit = 0
    _poi_raw_indices: set = set()  # raw indices into `nodes` for POI/bridge stations
    for lon, lat, label in EXPLICIT_STATIONS:
        proj_pt = np.array(_to_proj.transform(lon, lat))
        dist, idx = _all_node_tree.query(proj_pt)
        dist_m = dist * US_SURVEY_FT_TO_M
        if dist_m < 500.0:
            if not keep_mask[idx]:
                n_explicit += 1
                logger.debug(f"  Added explicit station: {label} ({dist_m:.0f}m)")
            keep_mask[idx] = True
            _poi_raw_indices.add(int(idx))
    if n_explicit > 0:
        logger.debug(f"  Added {n_explicit} explicit POI stations")

    # --- Ensure bridge hub nodes are candidates ---
    n_bridge_added = 0
    _nid_to_idx = {int(n[0]): i for i, n in enumerate(nodes)}
    for u, v, edata in G.edges(data=True):
        if edata.get("highway") == "bridge":
            for bnode in (u, v):
                idx = _nid_to_idx.get(int(bnode))
                if idx is not None:
                    if not keep_mask[idx]:
                        n_bridge_added += 1
                    keep_mask[idx] = True
                    _poi_raw_indices.add(int(idx))
    if n_bridge_added > 0:
        logger.debug(f"  Added {n_bridge_added} bridge hub stations")

    kept_indices = np.flatnonzero(keep_mask)
    candidate_node_ids = node_ids_all[kept_indices]
    # Map POI/bridge raw indices to local indices for terminal whitelist
    _raw_to_local = {int(raw): local for local, raw in enumerate(kept_indices)}
    _poi_terminal_locals: set = {
        _raw_to_local[r] for r in _poi_raw_indices if r in _raw_to_local
    }
    logger.debug(f"  Total candidate stations: {len(candidate_node_ids)} "
          f"(intersections={n_intersection}, demand={n_demand_added if _positive.any() else 0}, "
          f"POI={n_explicit})")

    # Extract coordinates (graph stores lon/lat as x/y attributes)
    coords_4326 = np.array([
        (G.nodes[nid]["x"], G.nodes[nid]["y"]) for nid in candidate_node_ids
    ])
    coords_proj = np.array([
        _to_proj.transform(lon, lat) for lon, lat in coords_4326
    ])

    # --- Pre-compute per-station demand coverage ---
    parcels_proj = parcels_gdf.to_crs(PROJECT_CRS)
    p_coords = np.column_stack([
        parcels_proj.geometry.centroid.x.values,
        parcels_proj.geometry.centroid.y.values,
    ])
    p_demand = parcels_gdf["demand_wt"].values.astype(np.float64)
    p_tree = cKDTree(p_coords)

    # --- Demand-weighted offset: shift station coordinates toward nearby
    # demand centroids so catchments center on demand, not on street corners.
    # The station keeps its graph node ID for routing; only the physical
    # location (used for catchment queries and inter-station distance) moves.
    # Cap offset at 150m to keep stations near buildable road ROW.
    _OFFSET_RADIUS_FT = 400.0 / US_SURVEY_FT_TO_M
    _OFFSET_MIN_M = 50.0   # don't bother for tiny shifts
    _OFFSET_MAX_M = 150.0
    n_offset = 0
    for si in range(len(candidate_node_ids)):
        nearby = p_tree.query_ball_point(coords_proj[si], r=_OFFSET_RADIUS_FT)
        if len(nearby) < 3:
            continue
        nearby_idx = np.array(nearby, dtype=int)
        _w = p_demand[nearby_idx]
        _w_sum = _w.sum()
        if _w_sum <= 0:
            continue
        # Demand-weighted centroid
        cx = float(np.dot(_w, p_coords[nearby_idx, 0]) / _w_sum)
        cy = float(np.dot(_w, p_coords[nearby_idx, 1]) / _w_sum)
        dx = cx - coords_proj[si, 0]
        dy = cy - coords_proj[si, 1]
        offset_m = float(np.hypot(dx, dy)) * US_SURVEY_FT_TO_M
        if offset_m < _OFFSET_MIN_M:
            continue
        # Cap at _OFFSET_MAX_M
        frac = min(_OFFSET_MAX_M / offset_m, 1.0)
        coords_proj[si, 0] += dx * frac
        coords_proj[si, 1] += dy * frac
        n_offset += 1
    if n_offset > 0:
        logger.debug(f"  Offset {n_offset} stations toward demand centroids (max {_OFFSET_MAX_M}m)")

    # Build spatial index (after offsets applied)
    station_tree = cKDTree(coords_proj)

    # For each station, sum distance-weighted demand within 1200m
    demand_coverage = np.zeros(len(candidate_node_ids), dtype=np.float64)
    tif_coverage = np.zeros(len(candidate_node_ids), dtype=np.float64)
    parking_cost_avg = np.zeros(len(candidate_node_ids), dtype=np.float64)
    bus_competition_arr = np.zeros(len(candidate_node_ids), dtype=np.float64)

    # Parcel AV for TIF
    if "CurTotAV" in parcels_gdf.columns:
        p_av = pd.to_numeric(parcels_gdf["CurTotAV"], errors="coerce").fillna(0).values
    else:
        p_av = np.zeros(len(parcels_gdf), dtype=np.float64)

    # Parking cost per parcel
    p_parking = np.zeros(len(parcels_gdf), dtype=np.float64)
    if "parking_cost" in parcels_gdf.columns:
        p_parking = pd.to_numeric(parcels_gdf["parking_cost"], errors="coerce").fillna(0).values

    # Exempt mask
    p_exempt = np.zeros(len(parcels_gdf), dtype=bool)
    if "PriorPropClass" in parcels_gdf.columns:
        ppc = pd.to_numeric(parcels_gdf["PriorPropClass"], errors="coerce").fillna(0)
        p_exempt = ((ppc >= 600) & (ppc < 700)).values

    # Institutional weights for student ridership proxy
    if "institutional_weight" in parcels_gdf.columns:
        p_inst_wt = pd.to_numeric(
            parcels_gdf["institutional_weight"], errors="coerce"
        ).fillna(1.0).values.astype(np.float64)
    else:
        p_inst_wt = np.ones(len(parcels_gdf), dtype=np.float64)

    # Development potential: land-value ratio (high = underbuilt parcel).
    # Parcels where land is worth much more than improvements have the most
    # upside for transit-oriented redevelopment.  Exempt parcels (campus,
    # government) get zero since they can't be developed.
    if "CurLandAV" in parcels_gdf.columns:
        p_land_av = pd.to_numeric(parcels_gdf["CurLandAV"], errors="coerce").fillna(0).values
    else:
        p_land_av = np.zeros(len(parcels_gdf), dtype=np.float64)
    p_tot_av_safe = np.maximum(p_av, 1.0)
    # Ratio > 0.5 means land > improvements -> strong redevelopment signal
    p_dev_potential = np.where(
        p_exempt, 0.0, np.clip(p_land_av / p_tot_av_safe, 0.0, 1.0) * p_av
    )

    max_walk_ft = WALK_CATCHMENT_M / US_SURVEY_FT_TO_M
    for si in range(len(candidate_node_ids)):
        sx, sy = coords_proj[si]
        nearby_parcels = p_tree.query_ball_point([sx, sy], r=max_walk_ft)
        if not nearby_parcels:
            continue
        nearby_idx = np.array(nearby_parcels, dtype=int)
        dists_ft = np.sqrt(
            (p_coords[nearby_idx, 0] - sx) ** 2 +
            (p_coords[nearby_idx, 1] - sy) ** 2
        )
        dists = dists_ft * US_SURVEY_FT_TO_M  # convert to meters for decay
        weights = np.exp(-DECAY_BETA * dists)
        demand_coverage[si] = float(np.dot(p_demand[nearby_idx], weights))

        # TIF: taxable AV only (exclude exempt)
        taxable = p_av[nearby_idx].copy()
        taxable[p_exempt[nearby_idx]] = 0.0
        tif_coverage[si] = float(np.dot(taxable, weights))

        # Average parking cost (demand-weighted)
        wt_sum = float(np.sum(weights * p_demand[nearby_idx]))
        if wt_sum > 0:
            parking_cost_avg[si] = float(
                np.dot(p_parking[nearby_idx], weights * p_demand[nearby_idx])
            ) / wt_sum

    # Feeder-zone demand per station (1200m-7000m ring).
    # Stations near bus-transfer points (intersections with existing bus routes)
    # get higher effective feeder demand, reflecting the two-layer catchment
    # model in the feedback loop.
    feeder_coverage = np.zeros(len(candidate_node_ids), dtype=np.float64)
    max_feeder_ft = FEEDER_CATCHMENT_M / US_SURVEY_FT_TO_M
    FEEDER_DECAY_BETA = 0.0005  # matches scorer
    for si in range(len(candidate_node_ids)):
        sx, sy = coords_proj[si]
        nearby_feeder = p_tree.query_ball_point([sx, sy], r=max_feeder_ft)
        if not nearby_feeder:
            continue
        nearby_idx = np.array(nearby_feeder, dtype=int)
        dists_ft = np.sqrt(
            (p_coords[nearby_idx, 0] - sx) ** 2 +
            (p_coords[nearby_idx, 1] - sy) ** 2
        )
        dists_m = dists_ft * US_SURVEY_FT_TO_M
        feeder_ring = (dists_m > WALK_CATCHMENT_M) & (dists_m <= FEEDER_CATCHMENT_M)
        if feeder_ring.any():
            f_weights = np.exp(-FEEDER_DECAY_BETA * dists_m[feeder_ring])
            feeder_coverage[si] = float(np.dot(p_demand[nearby_idx[feeder_ring]], f_weights))

    # Bus competition per station
    if bus_tree is not None and bus_stops_proj is not None:
        for si in range(len(candidate_node_ids)):
            nearby_bus = bus_tree.query_ball_point(coords_proj[si], r=800.0 / US_SURVEY_FT_TO_M)
            bus_competition_arr[si] = len(nearby_bus)

    # Bus-transfer quality: stations near more bus stops get higher effective
    # feeder coverage (riders can actually reach those stations by bus).
    bus_transfer_quality = np.clip(bus_competition_arr / 5.0, 0.1, 1.0)
    feeder_coverage *= bus_transfer_quality

    # Transit Propensity Index (TPI): additive-weighted combination of
    # parking quality, campus captive riders, and bus connectivity.
    # Replaces the old multiplicative mode-shift multiplier.
    #
    # Parking penalty: free parking suppresses transit mode share by ~60%
    # (Cervero 2006).  Range: 0.40 (free) to 1.0 (expensive).
    _park_mult = 0.40 + 0.60 / (1.0 + np.exp(-3.0 * (parking_cost_avg - 1.0)))

    # Campus multiplier (unchanged): captive student riders
    _inst_avg = np.zeros(len(candidate_node_ids))
    for si in range(len(candidate_node_ids)):
        nearby_idx = p_tree.query_ball_point(coords_proj[si], max_walk_ft)
        if len(nearby_idx) > 0:
            _inst_avg[si] = float(np.mean(p_inst_wt[np.array(nearby_idx, dtype=int)]))
    _campus_mult = 1.0 + 0.5 * np.maximum(_inst_avg - 1.0, 0.0)

    # Normalize each factor to [0, 1] for additive weighting.
    _park_norm = (_park_mult - 0.40) / 0.60           # [0, 1]
    _campus_norm = np.clip((_campus_mult - 1.0) / 0.5, 0.0, 1.0)  # [0, 1]
    _bus_norm = bus_transfer_quality                    # [0.1, 1.0]

    # Additive weights: parking dominates (strongest empirical predictor),
    # campus second (captive riders), bus third (feeder access is already
    # captured separately in feeder_coverage).
    _W_PARKING = 0.50
    _W_CAMPUS = 0.35
    _W_BUS = 0.15
    tpi = _W_PARKING * _park_norm + _W_CAMPUS * _campus_norm + _W_BUS * _bus_norm

    # Scale TPI to [0.4, 1.0] — never zero out a station entirely.
    tpi_scaled = 0.40 + 0.60 * tpi
    demand_coverage_adjusted = demand_coverage * tpi_scaled
    logger.debug(f"  TPI range: {tpi_scaled.min():.2f} - {tpi_scaled.max():.2f}")

    # Map ANCHOR_DESTINATIONS to nearest candidate station
    anchor_station_ids = []
    anchor_station_local = []
    for lon, lat, label in ANCHOR_DESTINATIONS:
        ax, ay = _to_proj.transform(lon, lat)
        d, nearest_local = station_tree.query([ax, ay], k=1)
        anchor_station_ids.append(int(candidate_node_ids[nearest_local]))
        anchor_station_local.append(int(nearest_local))

    # Node-ID-to-local-index mapping for O(1) lookup
    node_to_local = {int(nid): i for i, nid in enumerate(candidate_node_ids)}

    # Build graph-adjacency lookup: for each candidate station, its neighbor
    # candidate stations (1-hop on road graph that are also candidates)
    adjacency: Dict[int, List[int]] = {}  # local_idx -> [local_idx, ...]
    for li, nid in enumerate(candidate_node_ids):
        neighbors = []
        for _u, nbr in G.edges(nid):
            nbr_li = node_to_local.get(int(nbr))
            if nbr_li is not None and nbr_li != li:
                neighbors.append(nbr_li)
        # Also check incoming
        for pred, _v in G.in_edges(nid):
            pred_li = node_to_local.get(int(pred))
            if pred_li is not None and pred_li != li and pred_li not in neighbors:
                neighbors.append(pred_li)
        adjacency[li] = neighbors

    n_stations = len(candidate_node_ids)
    logger.debug(f"  Candidate stations: {n_stations} (from {len(nodes)} graph nodes)")
    logger.debug(f"  Demand coverage range: {demand_coverage.min():.0f} - {demand_coverage.max():.0f}")
    logger.debug(f"  Anchor stations: {len(anchor_station_local)}")

    _sd = {
        "node_ids": candidate_node_ids,
        "coords_proj": coords_proj,
        "coords_4326": coords_4326,
        "demand_coverage": demand_coverage_adjusted,
        "demand_coverage_raw": demand_coverage,
        "feeder_coverage": feeder_coverage,
        "tif_coverage": tif_coverage,
        "parking_cost": parking_cost_avg,
        "bus_competition": bus_competition_arr,
        "tree": station_tree,
        "graph": G,
        "adjacency": adjacency,
        "node_to_local": node_to_local,
        "anchor_station_local": anchor_station_local,
        "anchor_station_ids": anchor_station_ids,
        # Parcel-level data needed by score_station_set
        "parcel_coords_proj": p_coords,
        "parcel_demand": p_demand,
        "parcel_av": p_av,
        "parcel_exempt": p_exempt,
        "parcel_parking": p_parking,
        "parcel_inst_wt": p_inst_wt,
        "parcel_dev_potential": p_dev_potential,
        "parcel_max_sqft": pd.to_numeric(
            parcels_gdf.get("dev_capacity_sqft", pd.Series(0, index=parcels_gdf.index)),
            errors="coerce",
        ).fillna(0).values.astype(np.float64),
        "parcel_tif_eligible": parcels_gdf["tif_eligible"].values.astype(bool)
            if "tif_eligible" in parcels_gdf.columns
            else np.zeros(len(parcels_gdf), dtype=bool),
        "parcel_improvement_ratio": pd.to_numeric(
            parcels_gdf.get("improvement_ratio", pd.Series(0, index=parcels_gdf.index)),
            errors="coerce",
        ).fillna(0).values.astype(np.float64),
        "parcel_tree": p_tree,
        "road_dist_cache": {},
        "poi_terminal_locals": _poi_terminal_locals,
    }

    # Pre-partition parcels into a spatial grid for fast spatial filtering.
    # Each cell is GRID_CELL_M wide; score_station_set only queries parcels
    # in cells overlapping the corridor bbox + catchment buffer instead of
    # all 61K parcels.
    _GRID_CELL_M = 2000.0  # ~2km cells (in EPSG:2965 feet: 2000/0.3048)
    _grid_cell_ft = _GRID_CELL_M / US_SURVEY_FT_TO_M
    _px = p_coords[:, 0]
    _py = p_coords[:, 1]
    _gx = ((_px - _px.min()) / _grid_cell_ft).astype(np.int32)
    _gy = ((_py - _py.min()) / _grid_cell_ft).astype(np.int32)
    _grid_origin = np.array([_px.min(), _py.min()])
    _grid_index: Dict[tuple, np.ndarray] = {}
    for _ci in range(int(_gx.max()) + 1):
        for _cj in range(int(_gy.max()) + 1):
            _cell_mask = (_gx == _ci) & (_gy == _cj)
            if _cell_mask.any():
                _grid_index[(_ci, _cj)] = np.flatnonzero(_cell_mask)
    _sd["parcel_grid_index"] = _grid_index
    _sd["parcel_grid_origin"] = _grid_origin
    _sd["parcel_grid_cell_ft"] = _grid_cell_ft
    return _sd


STATION_PAIR_CACHE_PATH = PROC_DIR / "_station_pair_cache.pkl"


def _compute_cache_key(station_data: dict, G) -> str:
    """Deterministic key from graph topology + candidate station set."""
    node_ids = station_data["node_ids"]
    key_data = (
        len(G.nodes),
        len(G.edges),
        len(node_ids),
        tuple(sorted(int(n) for n in node_ids)),
    )
    return hashlib.sha256(str(key_data).encode()).hexdigest()[:16]


def _load_or_compute_station_distances(station_data: dict) -> None:
    """Load cached station-pair distances if valid, else compute and save."""
    G = station_data.get("graph")
    if G is None:
        return

    cache_key = _compute_cache_key(station_data, G)

    if STATION_PAIR_CACHE_PATH.exists():
        try:
            with open(STATION_PAIR_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                station_data["road_dist_cache"] = cached["road_dist_cache"]
                if "path_node_cache" in cached:
                    station_data["path_node_cache"] = cached["path_node_cache"]
                station_data["path_bearing_cache"] = cached["path_bearing_cache"]
                n_pairs = len(cached["road_dist_cache"])
                logger.debug(f"  Loaded station-pair cache ({n_pairs:,} pairs, key={cache_key})")
                return
            else:
                logger.debug(f"  Cache stale (key mismatch) — recomputing")
        except Exception as exc:
            logger.debug(f"  Cache load failed ({exc}) — recomputing")

    # Cache miss — compute from scratch
    _precompute_station_distances(station_data)

    # Save cache (distances + bearings only; paths are large and can be
    # recomputed on demand for the few corridors that pass validation)
    try:
        payload = {
            "cache_key": cache_key,
            "road_dist_cache": station_data["road_dist_cache"],
            "path_bearing_cache": station_data["path_bearing_cache"],
        }
        with open(STATION_PAIR_CACHE_PATH, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        sz_mb = STATION_PAIR_CACHE_PATH.stat().st_size / (1024 * 1024)
        logger.debug(f"  Saved station-pair cache to {STATION_PAIR_CACHE_PATH} ({sz_mb:.1f} MB)")
    except Exception as exc:
        logger.debug(f"  Warning: could not save station-pair cache: {exc}")


def _build_scipy_graph(G):
    """Convert a NetworkX (Multi)DiGraph to scipy CSR matrix for fast Dijkstra.

    Returns (csr_matrix, node_list, node_to_idx) where:
    - csr_matrix has ``apm_cost`` edge weights
    - node_list[i] is the OSM node ID for scipy index *i*
    - node_to_idx maps OSM node ID -> scipy index

    For MultiDiGraphs with parallel edges, keeps only the minimum-weight
    edge per (u, v) pair (COO duplicate entries would be summed, which
    would give incorrect shortest-path distances).
    """
    from scipy.sparse import csr_matrix as _csr_matrix

    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    # Collect minimum-weight edge per (u, v) pair
    best_edge: dict = {}  # (ui, vi) -> min weight
    for u, v, edata in G.edges(data=True):
        w = edata.get("apm_cost")
        if w is None:
            w = edata.get("length", 0)
        if w <= 0:
            w = 1e-6  # avoid zero-weight edges
        key = (node_to_idx[u], node_to_idx[v])
        if key not in best_edge or w < best_edge[key]:
            best_edge[key] = w

    rows = np.empty(len(best_edge), dtype=np.intp)
    cols = np.empty(len(best_edge), dtype=np.intp)
    data = np.empty(len(best_edge), dtype=np.float64)
    for i, ((ui, vi), w) in enumerate(best_edge.items()):
        rows[i] = ui
        cols[i] = vi
        data[i] = w

    mat = _csr_matrix((data, (rows, cols)), shape=(n, n))
    return mat, node_list, node_to_idx


def _reconstruct_path_from_predecessors(predecessors, node_list, src_scipy, tgt_scipy):
    """Reconstruct the shortest-path node sequence from a predecessor array.

    Returns a list of OSM node IDs [src_nid, ..., tgt_nid], or None if
    unreachable (predecessor == -9999).
    """
    if predecessors[tgt_scipy] < 0:
        return None
    path_idx = []
    cur = tgt_scipy
    while cur != src_scipy:
        path_idx.append(cur)
        cur = predecessors[cur]
        if cur < 0:
            return None  # unreachable
    path_idx.append(src_scipy)
    path_idx.reverse()
    return [node_list[i] for i in path_idx]


def _precompute_station_distances(station_data: dict) -> None:
    """Pre-compute road-graph distances and paths between nearby station pairs.

    Uses scipy.sparse.csgraph.dijkstra (C-implemented) instead of NetworkX
    (pure Python) for ~10-30x speedup.  Runs single-source Dijkstra from
    each candidate station and caches distances, paths, and bearings for
    all reachable candidate-station targets within Euclidean threshold.

    Modifies ``station_data`` in place (populates ``road_dist_cache``,
    ``path_node_cache``, ``path_bearing_cache``).
    """
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra
    import time as _time

    G = station_data.get("graph")
    if G is None:
        return

    node_ids = station_data["node_ids"]
    coords_proj = station_data["coords_proj"]
    n_stations = len(node_ids)
    if n_stations < 2:
        return

    # Only precompute pairs within Euclidean threshold (feet in EPSG:2965).
    # Consecutive station spacing is at most MAX_STOP_SPACING_M (1500m);
    # multiply by circuity headroom to cover all reachable *consecutive* pairs.
    _threshold_m = MAX_STOP_SPACING_M * MAX_SEGMENT_CIRCUITY
    _threshold_ft = _threshold_m / US_SURVEY_FT_TO_M

    station_tree = cKDTree(coords_proj)
    # For each station, find nearby candidates.  With ~4,000 stations and
    # a 6km Euclidean radius, expect ~1,000-1,200 neighbors per station.
    neighbor_lists = station_tree.query_ball_tree(station_tree, r=_threshold_ft)

    # Build reverse lookup: OSM node ID -> local station index
    osm_to_local = {int(nid): li for li, nid in enumerate(node_ids)}

    cache = station_data["road_dist_cache"]
    path_cache = station_data.setdefault("path_node_cache", {})
    bearing_cache = station_data.setdefault("path_bearing_cache", {})

    # --- Build scipy sparse graph (one-time) ---
    t0 = _time.perf_counter()
    logger.debug(f"  Building scipy sparse graph from {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges...")
    graph_sparse, sp_node_list, sp_node_to_idx = _build_scipy_graph(G)
    logger.debug(f"  Sparse graph built in {_time.perf_counter()-t0:.1f}s")

    # Map station OSM node IDs to scipy matrix indices
    station_scipy_idx = []
    for nid in node_ids:
        nid_int = int(nid)
        if nid_int in sp_node_to_idx:
            station_scipy_idx.append(sp_node_to_idx[nid_int])
        else:
            station_scipy_idx.append(-1)  # node not in graph
    station_scipy_idx = np.array(station_scipy_idx, dtype=np.intp)

    # apm_cost can be up to 5x physical length
    _cutoff_apm_cost = _threshold_m * 5.0

    n_computed = 0
    n_cached = 0

    _report_every = max(1, n_stations // 20)
    logger.debug(f"  Computing distances for {n_stations} stations "
          f"(scipy Dijkstra, cutoff={_cutoff_apm_cost:.0f})...")
    for src_li in range(n_stations):
        if src_li % _report_every == 0:
            logger.debug(f"    [{src_li}/{n_stations}] "
                  f"{_time.perf_counter()-t0:.0f}s elapsed, "
                  f"{n_computed} pairs computed")

        src_sp = station_scipy_idx[src_li]
        if src_sp < 0:
            continue  # station node not in graph

        neighbors = neighbor_lists[src_li]
        # Skip self and pairs already fully cached
        targets_needed = []
        for tgt_li in neighbors:
            if tgt_li <= src_li:
                continue  # symmetric: only compute (min, max)
            sym_key = (src_li, tgt_li)
            fwd_key = (src_li, tgt_li)
            rev_key = (tgt_li, src_li)
            if sym_key in cache and fwd_key in bearing_cache and rev_key in bearing_cache:
                n_cached += 1
                continue
            if station_scipy_idx[tgt_li] < 0:
                continue  # target node not in graph
            targets_needed.append(tgt_li)

        if not targets_needed:
            continue

        # Single-source Dijkstra from this station (scipy, C-implemented)
        dist_row, pred_row = sp_dijkstra(
            graph_sparse,
            directed=True,
            indices=src_sp,
            limit=_cutoff_apm_cost,
            return_predecessors=True,
        )

        for tgt_li in targets_needed:
            tgt_sp = station_scipy_idx[tgt_li]
            sym_key = (src_li, tgt_li)

            if np.isinf(dist_row[tgt_sp]) or pred_row[tgt_sp] < 0:
                # Unreachable within cutoff
                cache[sym_key] = float("inf")
                continue

            # Reconstruct path (OSM node IDs) from predecessor array
            path = _reconstruct_path_from_predecessors(
                pred_row, sp_node_list, src_sp, tgt_sp)
            if path is None:
                cache[sym_key] = float("inf")
                continue

            # Physical distance (sum of 'length' edges, not apm_cost)
            if sym_key not in cache:
                dist_m = 0.0
                for k in range(len(path) - 1):
                    edata = G.get_edge_data(path[k], path[k + 1])
                    if edata:
                        if isinstance(edata, dict) and 0 in edata:
                            edata = edata[0]
                        dist_m += edata.get("length", 0)
                cache[sym_key] = dist_m

            # Bearings only (paths are recomputed on demand to save memory)
            fwd_key = (src_li, tgt_li)
            if fwd_key not in bearing_cache and len(path) >= 2:
                bearing_cache[fwd_key] = (
                    _path_bearing_at(G, path, from_end=False),
                    _path_bearing_at(G, path, from_end=True),
                )

            rev_key = (tgt_li, src_li)
            rev_path = list(reversed(path))
            if rev_key not in bearing_cache and len(rev_path) >= 2:
                bearing_cache[rev_key] = (
                    _path_bearing_at(G, rev_path, from_end=False),
                    _path_bearing_at(G, rev_path, from_end=True),
                )

            n_computed += 1

    elapsed = _time.perf_counter() - t0
    logger.debug(
        f"  Pre-computed {n_computed} station-pair distances "
        f"({n_cached} already cached) in {elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# DP station selection (replaces greedy spacing)
# ---------------------------------------------------------------------------

def _pairwise_distances(station_locals, coords_proj):
    """Compute pairwise Euclidean distance matrix in meters for candidate stations.

    Parameters
    ----------
    station_locals : list[int]
        Local indices of stations.
    coords_proj : np.ndarray
        Station coordinates in EPSG:2965 (US survey feet).

    Returns
    -------
    np.ndarray : N×N symmetric distance matrix in meters.
    """
    pts = coords_proj[station_locals] * US_SURVEY_FT_TO_M  # (N, 2) in meters
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]   # (N, N, 2)
    return np.sqrt((diff ** 2).sum(axis=2))                 # (N, N)


def dp_select_stations(
    candidates_along_path: List[int],
    demand: np.ndarray,
    dist_m: np.ndarray,
    min_spacing: float = MIN_STOP_SPACING_M,
    max_spacing: float = MAX_STOP_SPACING_M * 2.5,
    min_k: int = MIN_STATIONS_PER_CORRIDOR,
    max_k: int = MAX_STATIONS_PER_CORRIDOR,
) -> Optional[List[int]]:
    """Select stations from ordered candidates to maximize demand coverage.

    Uses dynamic programming over (station_index, count) pairs.
    Anchors (first and last candidate) are always included.

    Parameters
    ----------
    candidates_along_path : list[int]
        Local indices of candidate stations, ordered along the path.
    demand : np.ndarray
        demand_coverage array — indexed by local station index.
    dist_m : np.ndarray
        Pairwise Euclidean distance matrix (N×N, meters) between candidates.
        ``dist_m[i, j]`` is the direct distance between candidate *i* and *j*.
    min_spacing : float
        Minimum distance between consecutive selected stations (meters).
    max_spacing : float
        Maximum distance between consecutive selected stations (meters).
    min_k, max_k : int
        Bounds on the number of selected stations.

    Returns
    -------
    list[int] or None
        Selected local indices (ordered along path), or None if no valid
        selection exists.
    """
    N = len(candidates_along_path)
    if N < min_k:
        return None

    d = np.array([demand[li] for li in candidates_along_path])

    INF = -1e18
    # dp[j][k] = best total demand selecting exactly k stations,
    # where station j is the k-th (last) selected station.
    dp = np.full((N, max_k + 1), INF)
    parent = np.full((N, max_k + 1), -1, dtype=int)

    # Flexible start: allow the DP to begin at any of the first M
    # candidates above a demand floor (25th percentile).  This lets the
    # algorithm skip low-demand origins instead of anchoring at index 0.
    _demand_floor = float(np.percentile(d, 25)) if N >= 4 else 0.0
    _START_WINDOW = min(5, N)
    for j in range(_START_WINDOW):
        if d[j] >= _demand_floor:
            dp[j][1] = d[j]

    for j in range(1, N):
        for k in range(2, min(j + 1, max_k) + 1):
            best_val = INF
            best_i = -1
            for i in range(j - 1, -1, -1):
                gap = dist_m[i, j]
                if gap < min_spacing:
                    continue
                if gap > max_spacing:
                    continue  # can't break — direct distances aren't monotonic
                if dp[i][k - 1] > best_val:
                    best_val = dp[i][k - 1]
                    best_i = i
            if best_i >= 0:
                dp[j][k] = best_val + d[j]
                parent[j][k] = best_i

    # Flexible terminal: consider any of the last M candidates above the
    # demand floor, not just index N-1.  Lets the DP trim low-demand tails
    # (e.g., guideway extending into nature preserves or across highways
    # into agricultural land).
    best_score = INF
    best_k = -1
    best_terminal = -1
    _END_WINDOW = min(5, N)
    for j in range(N - 1, N - 1 - _END_WINDOW, -1):
        if d[j] < _demand_floor:
            continue
        for k in range(min_k, max_k + 1):
            if dp[j][k] > best_score:
                best_score = dp[j][k]
                best_k = k
                best_terminal = j

    if best_k < 0:
        return None  # no valid selection found

    # Backtrack from best terminal to recover selected indices
    selected = []
    j = best_terminal
    k = best_k
    while k > 0:
        selected.append(candidates_along_path[j])
        j = parent[j][k]
        k -= 1

    selected.reverse()
    return selected


def _apply_dp_selection(
    station_list: List[int],
    coords_proj: np.ndarray,
    demand: np.ndarray,
    feeder_coverage: Optional[np.ndarray] = None,
) -> Optional[List[int]]:
    """Apply DP station selection to any ordered station list.

    Computes pairwise Euclidean distances and runs ``dp_select_stations``.
    When ``feeder_coverage`` is provided, blends feeder-zone demand into
    the DP objective so stations with good bus-transfer access score higher.
    Returns the DP-selected subset, or None if no valid selection exists
    (caller should fall back to the original list or skip).
    """
    if len(station_list) < MIN_STATIONS_PER_CORRIDOR:
        return None
    # Blend walk-zone and feeder-zone demand for the DP objective.
    # FEEDER_WEIGHT = FEEDER_TRANSFER_DISCOUNT (0.30) × FEEDER_COVERAGE_ESTIMATE (0.40)
    if feeder_coverage is not None:
        FEEDER_WEIGHT = 0.12
        blended = demand + FEEDER_WEIGHT * feeder_coverage
    else:
        blended = demand
    dist_m = _pairwise_distances(station_list, coords_proj)
    return dp_select_stations(station_list, blended, dist_m)


def validate_station_set(
    station_locals: List[int],
    station_data: dict,
) -> Tuple[bool, str]:
    """Check that a station set forms a valid APM corridor.

    Parameters
    ----------
    station_locals : list of int
        Local indices into station_data arrays (not OSM node IDs).
    """
    n = len(station_locals)
    if n < MIN_STATIONS_PER_CORRIDOR:
        return False, f"too few stations ({n} < {MIN_STATIONS_PER_CORRIDOR})"
    if n > MAX_STATIONS_PER_CORRIDOR:
        return False, f"too many stations ({n} > {MAX_STATIONS_PER_CORRIDOR})"
    if len(set(station_locals)) != n:
        return False, "duplicate stations"

    coords = station_data["coords_proj"]
    pts = coords[station_locals]

    # Vectorized consecutive spacing + total length (single diff computation)
    diffs = np.diff(pts, axis=0)  # (n-1, 2)
    seg_ft = np.linalg.norm(diffs, axis=1)  # (n-1,)
    seg_m = seg_ft * US_SURVEY_FT_TO_M

    too_close = np.where(seg_m < MIN_STOP_SPACING_M)[0]
    if len(too_close) > 0:
        i = int(too_close[0])
        return False, f"spacing {seg_m[i]:.0f}m < {MIN_STOP_SPACING_M}m between stations {i} and {i+1}"
    too_far = np.where(seg_m > MAX_STOP_SPACING_M * 2.5)[0]
    if len(too_far) > 0:
        i = int(too_far[0])
        return False, f"spacing {seg_m[i]:.0f}m > {MAX_STOP_SPACING_M * 2.5:.0f}m between stations {i} and {i+1}"

    total_euclid_ft = float(seg_ft.sum())
    length_km = total_euclid_ft * US_SURVEY_FT_TO_M * ROAD_CIRCUITY_FACTOR / 1000
    if length_km < MIN_LENGTH_KM:
        return False, f"too short ({length_km:.1f}km < {MIN_LENGTH_KM}km)"
    if length_km > MAX_LENGTH_KM:
        return False, f"too long ({length_km:.1f}km > {MAX_LENGTH_KM}km)"

    # Vectorized minimum curve radius check (physics-based, ASCE 21.2-2008)
    if n >= 3:
        v1 = diffs[:-1]  # (n-2, 2) — vectors into each interior vertex
        v2 = diffs[1:]   # (n-2, 2) — vectors out of each interior vertex
        m1 = seg_ft[:-1]  # reuse precomputed segment lengths
        m2 = seg_ft[1:]
        valid_mask = (m1 > 1e-6) & (m2 > 1e-6)
        if valid_mask.any():
            dot = np.sum(v1[valid_mask] * v2[valid_mask], axis=1)
            cos_a = np.clip(dot / (m1[valid_mask] * m2[valid_mask]), -1.0, 1.0)
            theta = np.arccos(cos_a)
            # Filter to non-trivial deflections (> 5°)
            curved = theta > np.radians(5)
            if curved.any():
                m1_curved = m1[valid_mask][curved] * US_SURVEY_FT_TO_M
                m2_curved = m2[valid_mask][curved] * US_SURVEY_FT_TO_M
                T = np.minimum(m1_curved, m2_curved) / 2.0
                half_theta = theta[curved] / 2.0
                R = np.where(half_theta > 1e-6, T / np.tan(half_theta), np.inf)
                bad_curve = np.where(R < APM_MIN_CURVE_RADIUS_M)[0]
                if len(bad_curve) > 0:
                    bi = int(bad_curve[0])
                    # Map back to original station index
                    orig_indices = np.where(valid_mask)[0][np.where(curved)[0]]
                    si = int(orig_indices[bi]) + 1
                    return False, (
                        f"curve radius {R[bi]:.0f}m < {APM_MIN_CURVE_RADIUS_M:.0f}m "
                        f"at station {si}"
                    )

    # Circuity constraints — reuse seg_m from above
    if station_data.get("graph") is not None:
        total_road_m = 0.0
        for i in range(n - 1):
            road_m = _road_graph_distance(station_locals[i], station_locals[i + 1], station_data)
            if road_m == float("inf"):
                return False, f"no road path between stations {i} and {i+1}"
            seg_circuity = road_m / max(seg_m[i], 1.0)
            if seg_circuity > MAX_SEGMENT_CIRCUITY:
                return False, f"segment {i}-{i+1} circuity {seg_circuity:.2f} > {MAX_SEGMENT_CIRCUITY}"
            total_road_m += road_m

        endpoint_euclid_m = float(np.linalg.norm(pts[-1] - pts[0])) * US_SURVEY_FT_TO_M
        if endpoint_euclid_m > 100:
            corridor_circuity = total_road_m / endpoint_euclid_m
            if corridor_circuity > MAX_CORRIDOR_CIRCUITY:
                return False, f"corridor circuity {corridor_circuity:.2f} > {MAX_CORRIDOR_CIRCUITY}"

    # Road-path bearing checks (require bearing_cache from _road_graph_distance)
    bearing_cache = station_data.get("path_bearing_cache", {})
    if bearing_cache:
        # Check 1: inter-segment reversal at interior stations — arrival from
        # previous segment vs departure to next segment.
        if n >= 3:
            for i in range(1, n - 1):
                prev_key = (station_locals[i - 1], station_locals[i])
                next_key = (station_locals[i], station_locals[i + 1])
                if prev_key in bearing_cache and next_key in bearing_cache:
                    _, arrive_brg = bearing_cache[prev_key]
                    depart_brg, _ = bearing_cache[next_key]
                    diff = abs(arrive_brg - depart_brg)
                    if diff > 180:
                        diff = 360 - diff
                    if diff > PATH_BEARING_REVERSAL_DEG:
                        return False, (
                            f"road-path reversal {diff:.0f}° at station {i} "
                            f"(>{PATH_BEARING_REVERSAL_DEG}°)"
                        )

        # Check 2: mid-segment detour — if the road path initially heads AWAY
        # from the next station, it must U-turn within the segment.  Compare
        # the routed departure bearing to the straight-line station bearing.
        for i in range(n - 1):
            seg_key = (station_locals[i], station_locals[i + 1])
            if seg_key not in bearing_cache:
                continue
            depart_brg, _ = bearing_cache[seg_key]

            # Straight-line bearing from station i to station i+1 (in ground coords)
            dx = (pts[i + 1, 0] - pts[i, 0])  # meters (EPSG:2965)
            dy = (pts[i + 1, 1] - pts[i, 1])
            direct_brg = math.degrees(math.atan2(dx, dy)) % 360

            dev = abs(depart_brg - direct_brg)
            if dev > 180:
                dev = 360 - dev
            if dev > PATH_BEARING_REVERSAL_DEG:
                return False, (
                    f"segment {i}-{i+1} departs {dev:.0f}° from direct bearing "
                    f"(route={depart_brg:.0f}°, direct={direct_brg:.0f}°)"
                )

    # Monotonicity: stations must progress along the main axis (first->last).
    # Catches gradual U-turns spread over 2-3 segments that pass the hairpin
    # check individually but cumulatively reverse direction.
    main_vec = pts[-1] - pts[0]
    main_len_sq = float(np.dot(main_vec, main_vec))
    if main_len_sq > 100**2:  # skip for very short corridors
        projections = [
            float(np.dot(pts[i] - pts[0], main_vec) / main_len_sq)
            for i in range(n)
        ]
        max_proj = projections[0]
        for i in range(1, n):
            if projections[i] < max_proj - 0.10:  # 10% backtrack tolerance
                return False, (
                    f"station {i} backtracks along main axis "
                    f"(proj={projections[i]:.2f}, prev_max={max_proj:.2f})"
                )
            max_proj = max(max_proj, projections[i])

    # Terminal road-class constraint: both endpoints must be adjacent to a
    # major road (primary/secondary/tertiary).  Prevents NSGA-II from evolving
    # corridors that terminate on residential/service streets.
    # Explicit POI stations (campus landmarks, hospitals, transit centers) and
    # bridge hubs are exempt — these are legitimate destinations even when OSM
    # classifies the adjacent road as "service" or "unclassified".
    G = station_data.get("graph")
    if G is not None:
        node_ids = station_data["node_ids"]
        poi_locals = station_data.get("poi_terminal_locals", set())
        for terminal_idx in (0, -1):
            local_idx = station_locals[terminal_idx]
            if local_idx in poi_locals:
                continue  # POI/bridge terminals are exempt
            nid = node_ids[local_idx]
            if not _has_major_road_edge(G, nid):
                which = "first" if terminal_idx == 0 else "last"
                return False, f"{which} terminal not on major road"

    return True, "ok"


def score_station_set(
    station_locals: List[int],
    station_data: dict,
    od_flows: Optional[pd.DataFrame] = None,
    parcel_lookup: Optional[dict] = None,
    max_walk_m: float = 1200.0,
) -> dict:
    """Score a corridor defined as a list of station local indices.

    Uses KDTree from station coordinates directly — no line interpolation.
    """
    coords_proj = station_data["coords_proj"]
    pts = coords_proj[station_locals]  # (K, 2)
    station_key = tuple(station_locals)

    # Parcel-level demand capture — spatial grid pre-filter
    p_coords_full = station_data["parcel_coords_proj"]
    p_demand_full = station_data["parcel_demand"]
    p_av_full = station_data["parcel_av"]
    p_exempt_full = station_data["parcel_exempt"]
    p_parking_full = station_data["parcel_parking"]
    p_inst_wt_full = station_data["parcel_inst_wt"]
    n_parcels = len(p_coords_full)

    # Use spatial grid to limit parcels to corridor bbox + feeder buffer.
    # Falls back to all parcels if grid is unavailable.
    FEEDER_MAX_M = FEEDER_CATCHMENT_M  # 7000m from spatial_constants
    _grid_index = station_data.get("parcel_grid_index")
    if _grid_index is not None:
        _grid_origin = station_data["parcel_grid_origin"]
        _grid_cell_ft = station_data["parcel_grid_cell_ft"]
        _buffer_ft = FEEDER_MAX_M / US_SURVEY_FT_TO_M
        _bbox_min = pts.min(axis=0) - _buffer_ft
        _bbox_max = pts.max(axis=0) + _buffer_ft
        _ci_lo = max(int((_bbox_min[0] - _grid_origin[0]) / _grid_cell_ft), 0)
        _cj_lo = max(int((_bbox_min[1] - _grid_origin[1]) / _grid_cell_ft), 0)
        _ci_hi = int((_bbox_max[0] - _grid_origin[0]) / _grid_cell_ft)
        _cj_hi = int((_bbox_max[1] - _grid_origin[1]) / _grid_cell_ft)
        _cell_arrays = []
        for _ci in range(_ci_lo, _ci_hi + 1):
            for _cj in range(_cj_lo, _cj_hi + 1):
                _ca = _grid_index.get((_ci, _cj))
                if _ca is not None:
                    _cell_arrays.append(_ca)
        if _cell_arrays:
            _subset_idx = np.concatenate(_cell_arrays)
        else:
            _subset_idx = np.arange(n_parcels)
    else:
        _subset_idx = np.arange(n_parcels)

    # Query only the subset against the station KDTree
    p_coords = p_coords_full[_subset_idx]
    tree = cKDTree(pts)
    _sub_dists_ft, _sub_nearest = tree.query(p_coords, k=1)
    # Scatter results back to full-length arrays (default: infinite dist)
    p_dists_ft = np.full(n_parcels, np.inf)
    p_nearest_stop = np.zeros(n_parcels, dtype=np.intp)
    p_dists_ft[_subset_idx] = _sub_dists_ft
    p_nearest_stop[_subset_idx] = _sub_nearest
    # Alias full arrays for downstream code
    p_demand = p_demand_full
    p_av = p_av_full
    p_exempt = p_exempt_full
    p_parking = p_parking_full
    p_inst_wt = p_inst_wt_full

    p_dists = p_dists_ft * US_SURVEY_FT_TO_M  # feet -> meters
    dp_weights = np.where(
        p_dists <= max_walk_m,
        np.exp(-DECAY_BETA * p_dists),
        0.0,
    )
    weighted_demand = float(np.sum(p_demand * dp_weights))

    # Bus competition from pre-computed per-station values
    bus_comp_vals = station_data["bus_competition"][station_locals]
    bus_stops_per_km_avg = float(np.mean(bus_comp_vals))
    # More stops/km -> better service -> lower headway.
    # Asymptote: 10 min at high density, 30 min baseline.
    bus_headway_eff = max(10.0, 30.0 / (1.0 + bus_stops_per_km_avg / 5.0))
    bus_quality = 1.0 / (1.0 + (bus_headway_eff / 15.0) ** 1.5)
    bus_competition = float(np.clip(bus_quality, 0.1, 1.0))

    # Length estimate: use road-graph distances if available (cached from
    # validate_station_set), else fall back to Euclidean × circuity factor.
    # Note: coords_proj is in US survey feet; road_graph distances are in meters.
    n_stops = len(station_locals)
    seg_dist_m = np.zeros(max(n_stops - 1, 0))
    total_euclid_m = 0.0
    has_road_graph = station_data.get("graph") is not None
    for i in range(n_stops - 1):
        seg_euclid_ft = np.hypot(
            pts[i + 1, 0] - pts[i, 0], pts[i + 1, 1] - pts[i, 1]
        )
        seg_euclid_m = seg_euclid_ft * US_SURVEY_FT_TO_M
        total_euclid_m += seg_euclid_m
        if has_road_graph:
            rd = _road_graph_distance(station_locals[i], station_locals[i + 1], station_data)
            if rd < float("inf"):
                seg_dist_m[i] = rd
            else:
                seg_dist_m[i] = seg_euclid_m * ROAD_CIRCUITY_FACTOR
        else:
            seg_dist_m[i] = seg_euclid_m * ROAD_CIRCUITY_FACTOR
    length_km = float(seg_dist_m.sum()) / 1000.0

    # Trip-length feasibility: a corridor shorter than the regional average
    # trip cannot serve the full trip for most riders.  Scale weighted_demand
    # by the fraction of a typical trip the corridor covers.
    trip_coverage = min(length_km / MEAN_TRIP_DISTANCE_KM, 1.0)
    weighted_demand *= trip_coverage

    # Stop distance matrix (SDM): cumulative along-route distance between stops.
    # sdm[i, j] gives the route distance (meters) a rider travels from stop i
    # to stop j along the APM guideway.
    cum_dist = np.zeros(n_stops)
    for i in range(n_stops - 1):
        cum_dist[i + 1] = cum_dist[i] + seg_dist_m[i]
    sdm = np.abs(cum_dist[:, None] - cum_dist[None, :])  # (K, K)

    # Per-stop demand: parcel demand weighted by distance decay, assigned to
    # nearest stop.  Used by Option 3b weighted-mean distance computation.
    in_catchment = dp_weights > 0
    weighted_pd = p_demand * dp_weights
    weighted_pd[~in_catchment] = 0.0
    stop_demand = np.bincount(
        p_nearest_stop[in_catchment],
        weights=weighted_pd[in_catchment],
        minlength=n_stops,
    )

    # Corridor circuity from actual length vs endpoint Euclidean
    endpoint_dist_ft = np.hypot(
        pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]
    )
    endpoint_dist_m = endpoint_dist_ft * US_SURVEY_FT_TO_M
    corridor_circuity = (length_km * 1000.0) / max(endpoint_dist_m, 1.0)

    # Physics-based curve speed penalties (ASCE 21.2-2008 lateral accel model)
    # pts is in EPSG:2965 feet; convert to meters for physics model
    curve_info = compute_curve_speed_penalties(pts * US_SURVEY_FT_TO_M)
    curve_cost_mult = curve_info["curve_cost_mult"]
    curve_delay_s = curve_info["total_curve_delay_s"]

    # Reject corridors with physically infeasible curves (radius < APM_MIN_CURVE_RADIUS_M)
    if curve_info["has_infeasible_curve"]:
        return None

    # --- Level A: Road-path bearing reversal penalty ---
    # Discount ridership for station pairs whose road path requires reversals
    # that will likely fail alignment.  Soft penalty (0.7x per reversal)
    # preserves NSGA-II population diversity while steering away from
    # station combinations that validate_station_set will reject.
    bearing_cache = station_data.get("path_bearing_cache", {})
    n_reversals = 0
    if n_stops >= 3:
        for i in range(1, n_stops - 1):
            prev_key = (station_locals[i - 1], station_locals[i])
            next_key = (station_locals[i], station_locals[i + 1])
            if bearing_cache and prev_key in bearing_cache and next_key in bearing_cache:
                _, arrive_brg = bearing_cache[prev_key]
                depart_brg, _ = bearing_cache[next_key]
                diff = abs(arrive_brg - depart_brg)
                if diff > 180:
                    diff = 360 - diff
                if diff > PATH_BEARING_REVERSAL_DEG:
                    n_reversals += 1
            else:
                # Euclidean-bearing fallback when road-path cache misses
                p_prev = pts[i - 1] * US_SURVEY_FT_TO_M
                p_curr = pts[i] * US_SURVEY_FT_TO_M
                p_next = pts[i + 1] * US_SURVEY_FT_TO_M
                arrive_brg_e = math.degrees(
                    math.atan2(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
                ) % 360
                depart_brg_e = math.degrees(
                    math.atan2(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
                ) % 360
                diff_e = abs(arrive_brg_e - depart_brg_e)
                if diff_e > 180:
                    diff_e = 360 - diff_e
                # Looser threshold for Euclidean (roads curve more)
                if diff_e > PATH_BEARING_REVERSAL_DEG + 15:
                    n_reversals += 1

    # Early bail-out: 3+ reversals -> 0.7^3 = 0.34x penalty, uncompetitive
    if n_reversals >= 3:
        return None

    reversal_penalty = 0.7 ** n_reversals if n_reversals > 0 else 1.0

    # --- Level A.5: Dogleg detection (deviation-and-return) ---
    # A dogleg is a pair of consecutive opposite-sign turns where the second
    # turn recovers a large fraction of the first — the corridor deviates
    # sideways to reach a demand pocket then comes back.  This is distinct
    # from a legitimate directional change (e.g., turning south toward a
    # destination), which has no matching recovery turn.
    #
    # Detection: signed turn angle at each interior stop.  If consecutive
    # turns exceed 10° each and have opposite signs, compute the recovery
    # ratio = min(|t1|, |t2|) / max(|t1|, |t2|).  A ratio > 0.40 means
    # the second turn undoes >40% of the first — that's a dogleg.
    _DOGLEG_MIN_ANGLE_DEG = 10.0
    _DOGLEG_RECOVERY_THRESHOLD = 0.40
    _DOGLEG_PENALTY = 0.85  # per dogleg (milder than 0.70 reversal penalty)
    n_doglegs = 0
    if n_stops >= 4:
        pts_m = pts * US_SURVEY_FT_TO_M
        signed_turns = []
        for i in range(1, n_stops - 1):
            v1x = pts_m[i, 0] - pts_m[i - 1, 0]
            v1y = pts_m[i, 1] - pts_m[i - 1, 1]
            v2x = pts_m[i + 1, 0] - pts_m[i, 0]
            v2y = pts_m[i + 1, 1] - pts_m[i, 1]
            cross = v1x * v2y - v1y * v2x
            dot = v1x * v2x + v1y * v2y
            angle = math.degrees(math.atan2(cross, dot))
            signed_turns.append(angle)
        for i in range(len(signed_turns) - 1):
            t1, t2 = signed_turns[i], signed_turns[i + 1]
            if (abs(t1) > _DOGLEG_MIN_ANGLE_DEG
                    and abs(t2) > _DOGLEG_MIN_ANGLE_DEG
                    and t1 * t2 < 0):
                recovery = min(abs(t1), abs(t2)) / max(abs(t1), abs(t2))
                if recovery > _DOGLEG_RECOVERY_THRESHOLD:
                    n_doglegs += 1
    dogleg_penalty = _DOGLEG_PENALTY ** n_doglegs if n_doglegs > 0 else 1.0

    # --- Level B: Approximate road-curve radius check ---
    # For segments with cached road paths, estimate the curve radius at each
    # interior station from the last road-graph edge arriving and first edge
    # departing.  Heavy penalty if any curve is below APM_MIN_CURVE_RADIUS_M.
    path_cache = station_data.get("path_node_cache", {})
    worst_road_radius = float("inf")
    if path_cache and n_stops >= 3:
        G = station_data["graph"]
        for i in range(1, n_stops - 1):
            in_key = (station_locals[i - 1], station_locals[i])
            out_key = (station_locals[i], station_locals[i + 1])
            in_path = path_cache.get(in_key)
            out_path = path_cache.get(out_key)
            if in_path and out_path and len(in_path) >= 2 and len(out_path) >= 2:
                p1 = _node_to_proj(in_path[-2], G)
                p2 = _node_to_proj(in_path[-1], G)
                p3 = _node_to_proj(out_path[1], G)
                if p1 and p2 and p3:
                    p1m = np.array(p1) * US_SURVEY_FT_TO_M
                    p2m = np.array(p2) * US_SURVEY_FT_TO_M
                    p3m = np.array(p3) * US_SURVEY_FT_TO_M
                    v1 = p2m - p1m
                    v2 = p3m - p2m
                    m1 = float(np.linalg.norm(v1))
                    m2 = float(np.linalg.norm(v2))
                    if m1 > 1 and m2 > 1:
                        cos_a = float(np.clip(np.dot(v1, v2) / (m1 * m2), -1, 1))
                        theta = float(np.arccos(cos_a))
                        if theta > np.radians(5):
                            T = min(m1, m2) / 2.0
                            half_theta = theta / 2.0
                            R = T / np.tan(half_theta) if half_theta > 1e-6 else float("inf")
                            worst_road_radius = min(worst_road_radius, R)

    road_curve_penalty = 1.0
    if worst_road_radius < APM_MIN_CURVE_RADIUS_M:
        road_curve_penalty = 0.3

    # Combined penalty applied after ridership estimation
    scoring_penalty = reversal_penalty * road_curve_penalty * dogleg_penalty

    # Mode choice parameters
    car_speed_kph = 30.0
    avg_parking = 0.0
    if in_catchment.any():
        avg_parking = float(np.average(
            p_parking[in_catchment], weights=dp_weights[in_catchment]
        ))

    # Effective APM speed (accounts for dwell + accel/decel + curve slowdowns)
    from scripts.generate_improved_ridership import compute_effective_apm_speed
    _base_eff_speed = compute_effective_apm_speed(length_km, n_stops)
    # Reduce effective speed by physics-based curve delay
    if length_km > 0 and curve_delay_s > 0:
        base_time_h = length_km / _base_eff_speed
        curve_delay_h = curve_delay_s / 3600.0
        apm_eff_speed = length_km / (base_time_h + curve_delay_h)
    else:
        apm_eff_speed = _base_eff_speed

    # Reject if curve delays degrade effective speed below floor
    if apm_eff_speed < APM_SPEED_KPH * EFFECTIVE_SPEED_FLOOR_FRACTION:
        return None

    # --- Per-OD Along-Route APM Distance (Option A) ---
    # Uses stop distance matrix (SDM) for actual APM route distances per OD pair
    # instead of a single representative trip distance for all pairs.

    # Pre-compute OD catchment state for reuse in first/second pass
    od_cache = station_data.get("od_cache")
    _od_near = None
    o_dists = d_dists = o_stop_idx = d_stop_idx = t_all = None
    if od_cache is not None:
        o_all = od_cache["o_coords"]
        d_all = od_cache["d_coords"]
        t_all = od_cache["trips"]
        o_dists_ft, o_stop_idx = _cached_kdtree_query(
            station_key, "od_origin", pts, o_all,
        )
        d_dists_ft, d_stop_idx = _cached_kdtree_query(
            station_key, "od_dest", pts, d_all,
        )
        o_dists = o_dists_ft * US_SURVEY_FT_TO_M
        d_dists = d_dists_ft * US_SURVEY_FT_TO_M
        _od_near = (o_dists <= max_walk_m) & (d_dists <= max_walk_m)

    def _vectorized_od_ridership(hw, speed):
        """Per-OD vectorized 4-mode MNL using stop distance matrix."""
        if _od_near is None or not _od_near.any():
            return 0.0
        near = _od_near
        o_wt = np.exp(-DECAY_BETA * o_dists[near])
        d_wt = np.exp(-DECAY_BETA * d_dists[near])
        access_wt = np.sqrt(o_wt * d_wt)
        # Per-OD APM route distance from SDM
        apm_route_km = sdm[o_stop_idx[near], d_stop_idx[near]] / 1000.0
        apm_route_km = np.maximum(apm_route_km, 0.3)  # floor for same-stop ODs
        # APM utility per OD pair (actual walk distance for access time)
        apm_access_min = o_dists[near] / (WALK_SPEED_KPH * 1000.0 / 60.0)
        u_apm = (BETA_IVT * (apm_route_km / speed) * 60
                 + BETA_WAIT * (hw / 2.0)
                 + BETA_ACCESS * apm_access_min
                 + BETA_COST * APM_FARE + ASC_APM)
        e_apm = np.exp(u_apm)
        # Competing modes at OD straight-line distance (route / circuity)
        trip_km = apm_route_km / max(corridor_circuity, 1.0)
        u_bus = (BETA_IVT * (trip_km / 20.0) * 60
                 + BETA_WAIT * bus_headway_eff / 2.0
                 + BETA_ACCESS * 5.0 + BETA_COST * 2.0 + (-0.10))
        u_car = (BETA_IVT * (trip_km / car_speed_kph) * 60
                 + BETA_ACCESS * 2.0
                 + BETA_COST * (trip_km * 0.621371 * 0.60 + avg_parking) + (-0.05))
        u_walk = BETA_IVT * (trip_km / WALK_SPEED_KPH) * 60 + 0.05
        total_exp = (e_apm + np.exp(u_bus) + np.exp(u_car)
                     + np.where(trip_km <= 2.0, np.exp(u_walk), 0.0))
        share = np.where(total_exp > 0, e_apm / total_exp, 0.0)
        return float(np.sum(t_all[near] * access_wt * share * 0.5))

    def _demand_apm_share(hw, speed):
        """APM share at Option 3b weighted-mean along-route distance."""
        if total_stop_demand <= 0:
            return 0.0
        u_apm = (BETA_IVT * (wmean_dist_km / speed) * 60
                 + BETA_WAIT * (hw / 2.0)
                 + BETA_ACCESS * 5.0 + BETA_COST * APM_FARE + ASC_APM)
        e_apm = np.exp(u_apm)
        tk = wmean_dist_km / max(corridor_circuity, 1.0)
        u_bus = (BETA_IVT * (tk / 20.0) * 60
                 + BETA_WAIT * bus_headway_eff / 2.0
                 + BETA_ACCESS * 5.0 + BETA_COST * 2.0 + (-0.10))
        u_car = (BETA_IVT * (tk / car_speed_kph) * 60
                 + BETA_ACCESS * 2.0
                 + BETA_COST * (tk * 0.621371 * 0.60 + avg_parking) + (-0.05))
        u_walk = BETA_IVT * (tk / WALK_SPEED_KPH) * 60 + 0.05
        exp_walk = np.exp(u_walk) if tk <= 2.0 else 0.0
        total = e_apm + np.exp(u_bus) + np.exp(u_car) + exp_walk
        return float(e_apm / total) if total > 0 else 0.0

    # Option 3b: weighted-mean along-route distance for demand-based path
    total_stop_demand = float(stop_demand.sum())
    if total_stop_demand > 0:
        demand_outer = np.outer(stop_demand, stop_demand)
        wmean_dist_m = float(np.sum(sdm * demand_outer)) / (total_stop_demand ** 2)
        wmean_dist_km = max(wmean_dist_m / 1000.0, 0.3)
    else:
        wmean_dist_km = max(length_km * 0.5, 0.3)

    # Length-dependent ridership blend weights.
    # Short corridors (<5km): demand-based is more reliable (concentrated
    # catchment, walk-access dominates).
    # Long corridors (>8km): OD-based is more reliable (dispersed flows,
    # cross-corridor commutes).
    # Replaces max(demand, od) which systematically biased upward.
    _od_weight = float(np.clip((length_km - 5.0) / 3.0, 0.3, 0.7))
    _demand_weight = 1.0 - _od_weight

    # First pass: initial headway
    od_ridership = _vectorized_od_ridership(APM_HEADWAY_MIN, apm_eff_speed)
    restructure_bonus = 1.0 + 0.15 * bus_competition
    apm_share = _demand_apm_share(APM_HEADWAY_MIN, apm_eff_speed)
    demand_based = weighted_demand * apm_share * 0.5 * restructure_bonus
    od_based = od_ridership * restructure_bonus
    ridership_est = _demand_weight * demand_based + _od_weight * od_based
    if weighted_demand > 0 and ridership_est < 10:
        ridership_est = max(ridership_est, weighted_demand * 0.001)

    # Second pass: demand-responsive headway + updated effective speed
    responsive_hw = compute_apm_headway(
        ridership_est, corridor_length_km=length_km, n_stops=n_stops,
    )
    apm_eff_speed = compute_effective_apm_speed(
        length_km, n_stops, daily_ridership=ridership_est,
    )
    if length_km > 0 and curve_delay_s > 0:
        _t_h = length_km / apm_eff_speed
        apm_eff_speed = length_km / (_t_h + curve_delay_s / 3600.0)
    if abs(responsive_hw - APM_HEADWAY_MIN) > 0.5:
        od_ridership_2 = _vectorized_od_ridership(responsive_hw, apm_eff_speed)
        apm_share = _demand_apm_share(responsive_hw, apm_eff_speed)
        demand_based_2 = weighted_demand * apm_share * 0.5 * restructure_bonus
        od_based_2 = od_ridership_2 * restructure_bonus
        ridership_est = _demand_weight * demand_based_2 + _od_weight * od_based_2
        if weighted_demand > 0 and ridership_est < 10:
            ridership_est = max(ridership_est, weighted_demand * 0.001)

    # Apply bearing-reversal and road-curve penalties to ridership estimate
    if scoring_penalty < 1.0:
        ridership_est *= scoring_penalty
        weighted_demand *= scoring_penalty

    # --- Student ridership (Option A.3: corridor-specific logit share) ---
    # The feedback loop model gets ~48% of mature ridership from students.
    # Use enrollment-based campus_pop_catch with a corridor-specific MNL
    # share (not flat 0.20) matching the full model's approach.
    campus_weight_arr = np.maximum(p_inst_wt - 1.0, 0.0)  # 0 for non-campus
    total_campus_weight = float(campus_weight_arr.sum())
    student_ridership = 0.0
    if total_campus_weight > 0:
        from src.data.purdue_transit_demand import (
            PURDUE_ENROLLMENT, PURDUE_FACULTY_STAFF,
            STUDENT_ASC_ADJUSTMENTS,
            CAMPUS_CAR_ACCESS_MIN, DEFAULT_CAR_ACCESS_MIN,
        )
        STUDENT_CAMPUS_TRIP_DIST_KM = 1.5  # representative intra-campus trip (km)
        CAMPUS_TOTAL = (
            PURDUE_ENROLLMENT * STUDENT_PRESENCE_FACTOR
            + PURDUE_FACULTY_STAFF * FACULTY_PRESENCE_FACTOR
        )  # ~14,560
        # Distance-weighted campus presence in walk catchment
        weighted_campus_catch = float(np.sum(campus_weight_arr * dp_weights))
        campus_pop_catch = CAMPUS_TOTAL * (weighted_campus_catch / total_campus_weight)

        # Corridor-specific student APM share via 4-mode logit
        # (replaces flat STUDENT_APM_SHARE_ESTIMATE = 0.20)
        _s_dist_km = STUDENT_CAMPUS_TRIP_DIST_KM  # 1.5km typical campus trip
        _s_apm_ivt = (_s_dist_km / apm_eff_speed) * 60
        _s_apm_wait = responsive_hw / 2.0
        _s_u_apm = (BETA_IVT * _s_apm_ivt + BETA_WAIT * _s_apm_wait
                    + BETA_ACCESS * 5.0 + BETA_COST * APM_FARE + ASC_APM
                    + STUDENT_ASC_ADJUSTMENTS["apm"])
        _s_u_bus = (BETA_IVT * (_s_dist_km / 18.0) * 60
                    + BETA_WAIT * bus_headway_eff / 2.0
                    + BETA_ACCESS * 5.0 + BETA_COST * 2.0 + (-0.10)
                    + STUDENT_ASC_ADJUSTMENTS["bus"])
        _s_u_car = (BETA_IVT * (_s_dist_km / 30.0) * 60
                    + BETA_ACCESS * CAMPUS_CAR_ACCESS_MIN
                    + BETA_COST * 0.40 + (-0.05)
                    + STUDENT_ASC_ADJUSTMENTS["car"])
        _s_u_walk = (BETA_IVT * (_s_dist_km / WALK_SPEED_KPH) * 60 + 0.05
                     + STUDENT_ASC_ADJUSTMENTS["walk"])
        _s_exps = np.exp(np.array([_s_u_apm, _s_u_bus, _s_u_car, _s_u_walk]))
        _student_apm_share = float(_s_exps[0] / _s_exps.sum()) if _s_exps.sum() > 0 else 0.10

        student_ridership = (
            campus_pop_catch * STUDENT_CAMPUS_TRIP_RATE * _student_apm_share
        )
    ridership_est += student_ridership

    # --- Feeder-zone demand (moved before mini forward sim so it compounds) ---
    # The full model has a two-layer catchment: walk (0-1200m) + feeder
    # (1200-7000m).  The search only scores walk-zone.  Add discounted
    # feeder-zone demand to better predict full-model outcomes.
    _feeder_dists = p_dists  # reuse cached parcel distances (already in meters)
    feeder_mask = (_feeder_dists > max_walk_m) & (_feeder_dists <= FEEDER_MAX_M)
    feeder_demand = 0.0
    if feeder_mask.any():
        FEEDER_DECAY_BETA = 0.0005  # gentler decay for feeder zone (50% at ~1400m into ring)
        _f_weights = np.exp(-FEEDER_DECAY_BETA * _feeder_dists[feeder_mask])
        feeder_demand = float(np.sum(p_demand[feeder_mask] * _f_weights))
        FEEDER_TRANSFER_DISCOUNT = 0.30
        FEEDER_COVERAGE_ESTIMATE = 0.40
        feeder_ridership = (
            feeder_demand * apm_share * 0.5 * FEEDER_TRANSFER_DISCOUNT
            * FEEDER_COVERAGE_ESTIMATE
        )
        ridership_est += feeder_ridership

    # --- Mini forward simulation (3-step development projection) ---
    # Replaces the static dev_bonus multiplier with a simplified 3-step
    # simulation that approximates the feedback loop's compounding:
    #   Step 1 (Year 0): ridership_est already computed above (walk+feeder+student).
    #   Step 2 (Year 5): induced pop -> higher demand -> re-run MNL.
    #   Step 3 (Year 15): re-compute headway from yr5 ridership -> re-run MNL.
    # POP_PER_RIDER calibrated from feedback loop results (9 corridors):
    #   Year 5:  cum_pop / yr0_riders ≈ 1.0
    #   Year 15: cum_pop / yr0_riders ≈ 2.5
    POP_PER_RIDER_YEAR5 = 1.0
    POP_PER_RIDER_YEAR15 = 2.5

    p_dev = station_data.get("parcel_dev_potential")
    dev_potential = 0.0
    if p_dev is not None:
        dev_potential = float(np.sum(p_dev * dp_weights))

    ridership_y5 = ridership_est
    ridership_y15 = ridership_est
    if dev_potential > 0 and ridership_est > 0:
        # Walk-zone development potential weights for allocating induced pop
        dev_wt = p_dev * dp_weights
        dev_wt_sum = float(dev_wt.sum())
        if dev_wt_sum > 0:
            dev_alloc = dev_wt / dev_wt_sum  # per-parcel allocation fraction

            # Step 2: Year 5 projection
            pop_growth_5 = ridership_est * POP_PER_RIDER_YEAR5
            demand_boost_5 = p_demand + pop_growth_5 * dev_alloc
            wd_5 = float(np.sum(demand_boost_5 * dp_weights))
            od_r_5 = _vectorized_od_ridership(responsive_hw, apm_eff_speed)
            pop_scale_5 = 1.0 + pop_growth_5 / max(weighted_demand, 1.0)
            demand_based_5 = wd_5 * apm_share * 0.5 * restructure_bonus
            od_based_5 = od_r_5 * restructure_bonus * pop_scale_5
            ridership_y5 = max(demand_based_5, od_based_5) + student_ridership

            # Step 3: Year 15 — re-compute headway from yr5 ridership (captures
            # the headway feedback loop: higher ridership -> lower headway ->
            # higher mode share -> even higher ridership)
            hw_15 = compute_apm_headway(
                ridership_y5, corridor_length_km=length_km, n_stops=n_stops,
            )
            speed_15 = compute_effective_apm_speed(
                length_km, n_stops, daily_ridership=ridership_y5,
            )
            if length_km > 0 and curve_delay_s > 0:
                _t15 = length_km / speed_15
                speed_15 = length_km / (_t15 + curve_delay_s / 3600.0)
            pop_growth_15 = ridership_y5 * POP_PER_RIDER_YEAR15
            demand_boost_15 = p_demand + pop_growth_15 * dev_alloc
            wd_15 = float(np.sum(demand_boost_15 * dp_weights))
            # Re-run MNL with updated headway/speed. OD trip table is static
            # (LODES base year) — induced pop is captured via pop_scale_15
            # multiplier rather than recomputing OD flows, which would require
            # a full synthetic population and is too expensive for search.
            od_r_15 = _vectorized_od_ridership(hw_15, speed_15)
            apm_share_15 = _demand_apm_share(hw_15, speed_15)
            pop_scale_15 = 1.0 + pop_growth_15 / max(weighted_demand, 1.0)
            demand_based_15 = wd_15 * apm_share_15 * 0.5 * restructure_bonus
            od_based_15 = od_r_15 * restructure_bonus * pop_scale_15
            ridership_y15 = max(demand_based_15, od_based_15) + student_ridership

    # Weighted average: 20% year 0, 30% year 5, 50% year 15
    # Emphasizes long-term performance since that determines viability.
    ridership_est = 0.20 * ridership_est + 0.30 * ridership_y5 + 0.50 * ridership_y15

    # Barrier crossings — use road-graph path when available, else straight line
    path_cache = station_data.get("path_node_cache", {})
    G = station_data.get("graph")
    if G is not None and path_cache:
        road_pts_proj = []
        for i in range(len(station_locals) - 1):
            seg_key = (station_locals[i], station_locals[i + 1])
            seg_path = path_cache.get(seg_key)
            if seg_path and len(seg_path) >= 2:
                for nid in (seg_path if i == 0 else seg_path[1:]):
                    _proj = _node_to_proj(nid, G)
                    if _proj is not None:
                        road_pts_proj.append(_proj)
            else:
                # Fallback: straight line for this segment
                if not road_pts_proj:
                    road_pts_proj.append(tuple(pts[i]))
                road_pts_proj.append(tuple(pts[i + 1]))
        if len(road_pts_proj) >= 2:
            barrier_line = LineString(road_pts_proj)
        else:
            barrier_line = LineString(pts.tolist())
    else:
        barrier_line = LineString(pts.tolist())
    barrier_cost_usd = barrier_crossing_cost_usd(barrier_line)

    # --- TIF potential ---
    # Two components: (A) uplift on existing AV from transit proximity,
    # and (B) new development on underbuilt TIF-eligible parcels.
    #
    # Uses conservative capture rate (85%, matching production finance model)
    # and circuit-breaker-capped effective tax rate.  Under EDA designation
    # (IC 36-7-14-39(a)), only commercial increment generates TIF —
    # res_share is set to 0 for the effective rate calculation.
    _tif_res_share = 0.0 if TIF_AREA_TYPE_DEFAULT == "eda" else 0.5
    _tif_eff_rate = effective_tif_tax_rate(PROPERTY_TAX_RATE, _tif_res_share)

    # (A) Uplift on existing taxable AV
    taxable_av = p_av.copy()
    taxable_av[p_exempt] = 0.0
    catchment_av = float(np.sum(taxable_av * dp_weights))
    MATURE_TARGET = 5000.0
    tif_uplift = 0.05 + 0.15 * min(ridership_est / MATURE_TARGET, 1.0)
    tif_av_component = catchment_av * tif_uplift * _tif_eff_rate * TIF_CAPTURE_RATE_CONSERVATIVE

    # (B) Developable increment: new sqft on TIF-eligible underbuilt parcels.
    # Parcels with high unused zoning capacity near stations generate the
    # most TIF from new construction.
    p_max_sqft = station_data.get("parcel_max_sqft")
    p_tif_eligible = station_data.get("parcel_tif_eligible")
    p_imp_ratio = station_data.get("parcel_improvement_ratio")
    tif_dev_component = 0.0
    if p_max_sqft is not None and p_tif_eligible is not None and p_imp_ratio is not None:
        # Developable increment = (max_sqft - current_sqft) for eligible parcels
        current_sqft = p_max_sqft * p_imp_ratio
        increment_sqft = np.maximum(p_max_sqft - current_sqft, 0.0)
        # Weight by walk-catchment decay and TIF eligibility
        tif_weighted_sqft = float(np.sum(
            increment_sqft * dp_weights * p_tif_eligible
        ))
        # Convert sqft → assessed value → annual TIF revenue
        TIF_RENT_PER_SQFT = 15.0        # $/sqft/year (mixed res/commercial)
        TIF_CAP_RATE = 0.07             # assessed value = NOI / cap_rate
        TIF_BUILDOUT_Y15 = 0.40         # 40% of capacity built by year 15
        tif_new_av = tif_weighted_sqft * TIF_RENT_PER_SQFT / TIF_CAP_RATE * TIF_BUILDOUT_Y15
        tif_dev_component = tif_new_av * _tif_eff_rate * TIF_CAPTURE_RATE_CONSERVATIVE

    tif_potential = tif_av_component + tif_dev_component

    # --- TIF viability flag (Fix 3) ---
    # TIF is a viability constraint, not a Pareto objective.
    # Corridors below the floor are penalized but not eliminated here
    # (the NSGA-II objectives no longer include TIF).
    tif_viable = tif_potential >= MIN_TIF_VIABILITY_USD

    # Cost efficiency — MECE capital cost decomposition
    from src.financial_params import (
        O_AND_M_FIXED_USD, O_AND_M_PER_KM_USD, O_AND_M_PER_STATION_USD,
        OPERATING_DAYS_PER_YEAR,
    )
    capital_cost = compute_capital_cost(
        length_km, n_stops, curve_cost_mult=curve_cost_mult,
        escalation=False,  # constant-dollar for relative ranking
    ) + barrier_cost_usd
    annual_ops = (O_AND_M_FIXED_USD + length_km * O_AND_M_PER_KM_USD
                  + n_stops * O_AND_M_PER_STATION_USD)
    r, n_yr = BOND_RATE, BOND_TERM
    annual_capital = capital_cost * (r * (1 + r)**n_yr) / ((1 + r)**n_yr - 1)
    total_annual_cost = annual_capital + annual_ops
    cost_per_rider = total_annual_cost / max(ridership_est * OPERATING_DAYS_PER_YEAR, 1)

    # --- DCR estimate (3rd NSGA-II objective) ---
    # Revenue / cost ratio using mature-year (y15) ridership for farebox,
    # since financial viability depends on long-term revenue not year-0.
    # TIF ramps up via S-curve phasing — apply average of years 1-10 to
    # reflect critical early-years DCR shortfall.
    ANNUAL_AVG_FACTOR = 0.860  # seasonal adjustment
    annual_fare_revenue = ridership_y15 * OPERATING_DAYS_PER_YEAR * APM_FARE * ANNUAL_AVG_FACTOR
    _tif_phasing_avg = sum(
        min(1.0, 0.05 + 0.28 * max(y - 2, 0) / 3.0 if y <= 5
            else 0.33 + 0.52 * (y - 5) / 10.0)
        for y in range(1, 11)
    ) / 10.0  # ~0.30 average over years 1-10
    dcr_est = (
        (annual_fare_revenue + tif_potential * _tif_phasing_avg)
        / max(total_annual_cost, 1.0)
    )
    # Keep viability_indicator as alias for backward compat
    viability_indicator = dcr_est

    # Marginal terminal cost-benefit diagnostic.
    # Estimates riders-per-$M for each terminus to help identify
    # whether terminal extensions are justified.
    _CAPITAL_PER_KM = 100e6  # $100M/km guideway
    _STATION_COST = 15e6     # $15M per station
    _terminal_mcr = {}
    if n_stops >= 4 and weighted_demand > 0:
        _demand_arr = station_data["demand_coverage"]
        for _label, _seg_i, _stn_i in [("west", 0, 0), ("east", -1, -1)]:
            _seg_km = seg_dist_m[_seg_i] / 1000.0
            _marginal_cost_m = _seg_km * _CAPITAL_PER_KM + _STATION_COST
            _term_share = _demand_arr[station_locals[_stn_i]] / max(weighted_demand, 1)
            _term_riders = ridership_est * _term_share
            _terminal_mcr[_label] = round(
                _term_riders / max(_marginal_cost_m / 1e6, 0.01), 1)

    return {
        "ridership_est": ridership_est,
        "ridership_y15": ridership_y15,
        "student_ridership": student_ridership,
        "weighted_demand": weighted_demand,
        "bus_competition": bus_competition,
        "apm_share": apm_share,
        "tif_potential": tif_potential,
        "tif_av_component": tif_av_component,
        "tif_dev_component": tif_dev_component,
        "tif_viable": tif_viable,
        "dcr_est": dcr_est,
        "cost_efficiency": 1.0 / max(cost_per_rider, 0.001),
        "viability_indicator": viability_indicator,
        "length_km": length_km,
        "barrier_cost_usd": barrier_cost_usd,
        "curve_cost_mult": curve_cost_mult,
        "n_turns": curve_info["n_speed_restricted_curves"],
        "turn_angles": curve_info["turn_angles"],
        "corridor_circuity": corridor_circuity,
        "min_curve_radius_m": curve_info["min_curve_radius_m"],
        "curve_delay_s": curve_info["total_curve_delay_s"],
        "effective_speed_kph": apm_eff_speed,
        "n_reversals": n_reversals,
        "n_doglegs": n_doglegs,
        "terminal_mcr_west": _terminal_mcr.get("west", 0.0),
        "terminal_mcr_east": _terminal_mcr.get("east", 0.0),
        "terminal_weak": bool(min(
            _terminal_mcr.get("west", 999),
            _terminal_mcr.get("east", 999),
        ) < 50),  # riders per $M
        "_weight_vector": dp_weights,
    }


# ---------------------------------------------------------------------------
# Station-set genetic operators
# ---------------------------------------------------------------------------


def _generate_dense_core_corridors(
    station_data: dict,
    top_k: int = 20,
    max_euclid_m: float = 4000.0,
    min_length_km: float = 3.0,
    max_length_km: float = 8.0,
    min_stations: int = 3,
    max_stations: int = 7,
) -> List[dict]:
    """Generate short, high-density corridors from demand hotspots.

    Instead of connecting distant anchors, this method identifies the
    highest-demand cluster of stations and builds minimal corridors
    within it.  These corridors target the campus-downtown core where
    ridership density (riders/km) is highest and capital cost is lowest.

    Returns a list of candidate dicts with "stations" and "source" keys.
    """
    import networkx as nx

    coords = station_data["coords_proj"]
    demand = station_data["demand_coverage"]
    feeder_cov = station_data.get("feeder_coverage")
    node_ids = station_data["node_ids"]
    node_to_local = station_data["node_to_local"]
    G = station_data["graph"]
    n_stations_total = len(node_ids)

    if n_stations_total < min_stations:
        return []

    # Step 1: Find top-K stations by demand
    actual_k = min(top_k, n_stations_total)
    top_idx = np.argsort(demand)[-actual_k:]
    top_set = set(int(i) for i in top_idx)

    max_euclid_ft = max_euclid_m / US_SURVEY_FT_TO_M
    insert_radius_ft = 300.0 / US_SURVEY_FT_TO_M

    candidates = []

    # Step 2: Enumerate short paths between pairs of top-K stations
    for i_pos in range(len(top_idx)):
        a = int(top_idx[i_pos])
        a_nid = int(node_ids[a])
        for j_pos in range(i_pos + 1, len(top_idx)):
            b = int(top_idx[j_pos])
            b_nid = int(node_ids[b])

            # Skip pairs too far apart
            euclid_ft = float(np.hypot(
                coords[a, 0] - coords[b, 0],
                coords[a, 1] - coords[b, 1],
            ))
            if euclid_ft > max_euclid_ft:
                continue

            # Road-graph path
            try:
                path_nodes = nx.shortest_path(G, a_nid, b_nid, weight="apm_cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            # Extract candidate stations along path
            path_stations = []
            for nid in path_nodes:
                li = node_to_local.get(int(nid))
                if li is not None:
                    path_stations.append(li)

            if len(path_stations) < 2:
                continue

            # Step 3: Insert nearby high-demand stations not on path
            path_set = set(path_stations)
            path_coords = coords[path_stations]
            insertions = []
            for s in top_set - path_set:
                s_coord = coords[s]
                # Distance from station to nearest point on the path polyline
                dists_to_path = np.sqrt(
                    (path_coords[:, 0] - s_coord[0]) ** 2
                    + (path_coords[:, 1] - s_coord[1]) ** 2
                )
                min_dist = float(dists_to_path.min())
                if min_dist < insert_radius_ft:
                    # Find insertion position: project onto main axis
                    main_vec = path_coords[-1] - path_coords[0]
                    proj_s = float(np.dot(s_coord - path_coords[0], main_vec))
                    insertions.append((proj_s, s))

            # Insert in order along the main axis
            if insertions:
                # Compute projections for existing stations too
                main_vec = path_coords[-1] - path_coords[0]
                existing_projs = [
                    (float(np.dot(coords[li] - path_coords[0], main_vec)), li)
                    for li in path_stations
                ]
                all_projs = existing_projs + insertions
                all_projs.sort(key=lambda x: x[0])
                # Deduplicate
                seen = set()
                path_stations = []
                for _, li in all_projs:
                    if li not in seen:
                        seen.add(li)
                        path_stations.append(li)

            # DP station selection with relaxed min_k for short corridors
            dist_m = _pairwise_distances(path_stations, coords)
            spaced = dp_select_stations(
                path_stations, demand if feeder_cov is None
                else demand + 0.12 * feeder_cov,
                dist_m,
                min_k=min_stations,
                max_k=max_stations,
            )
            if spaced is None:
                continue

            # Estimate length via Euclidean × circuity
            pts = coords[spaced]
            seg_ft = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            length_km = float(seg_ft.sum()) * US_SURVEY_FT_TO_M * ROAD_CIRCUITY_FACTOR / 1000.0

            if length_km < min_length_km or length_km > max_length_km:
                continue

            # Light validation (skip full validate_station_set for speed —
            # NSGA-II will validate later)
            if len(spaced) < min_stations:
                continue

            candidates.append({
                "stations": spaced,
                "source": "dense_core",
            })

    # Deduplicate by station overlap (>70% = duplicate)
    if candidates:
        candidates = deduplicate_station_sets(
            candidates, station_data, overlap_threshold=0.70,
        )

    return candidates


def generate_initial_station_sets(
    station_data: dict,
    od_flows: Optional[pd.DataFrame] = None,
    parcel_lookup: Optional[dict] = None,
    n_random_walks: int = 60,
) -> List[dict]:
    """Generate initial corridor candidates as station sets.

    1. Anchor-pair corridors (road-graph path -> extract intermediate stations)
    2. Farthest-point seeds -> demand-biased random walks on graph
    3. Random high-demand station chains
    """
    import networkx as nx
    from itertools import combinations

    G = station_data["graph"]
    coords = station_data["coords_proj"]
    demand = station_data["demand_coverage"]
    feeder_cov = station_data.get("feeder_coverage")
    node_ids = station_data["node_ids"]
    node_to_local = station_data["node_to_local"]
    anchor_local = station_data["anchor_station_local"]
    adjacency = station_data["adjacency"]
    n_stations = len(node_ids)

    candidates = []

    # --- 1. Anchor-pair corridors ---
    logger.debug("  Generating anchor-pair corridors...")
    anchor_pairs = list(combinations(range(len(anchor_local)), 2))
    n_anchor_corridors = 0
    for ai, bi in anchor_pairs:
        a_li = anchor_local[ai]
        b_li = anchor_local[bi]
        a_nid = int(node_ids[a_li])
        b_nid = int(node_ids[b_li])

        try:
            # Shortest path on road graph
            path_nodes = nx.shortest_path(G, a_nid, b_nid, weight="apm_cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        # Extract candidate stations along the path, enforcing min spacing
        all_path_stations = []
        for nid in path_nodes:
            li = node_to_local.get(int(nid))
            if li is not None:
                all_path_stations.append(li)

        if len(all_path_stations) < 2:
            continue

        # DP demand-maximizing station selection (replaces greedy spacing)
        spaced = _apply_dp_selection(all_path_stations, coords, demand, feeder_cov)

        # Fall back to greedy if DP finds no valid selection
        if spaced is None:
            min_sep = MIN_STOP_SPACING_M / US_SURVEY_FT_TO_M
            spaced = [all_path_stations[0]]
            for li in all_path_stations[1:-1]:
                last = spaced[-1]
                d = np.hypot(
                    coords[li, 0] - coords[last, 0],
                    coords[li, 1] - coords[last, 1],
                )
                if d >= min_sep:
                    spaced.append(li)
            end_li = all_path_stations[-1]
            if spaced[-1] != end_li:
                d_end = np.hypot(
                    coords[end_li, 0] - coords[spaced[-1], 0],
                    coords[end_li, 1] - coords[spaced[-1], 1],
                )
                if d_end >= min_sep:
                    spaced.append(end_li)
                else:
                    spaced[-1] = end_li

        valid, _ = validate_station_set(spaced, station_data)
        if valid:
            candidates.append({"stations": spaced, "source": "anchor_pair"})
            n_anchor_corridors += 1

    logger.debug(f"    {n_anchor_corridors} anchor-pair corridors")

    # --- 1b. Anchor-triplet corridors ---
    logger.debug("  Generating anchor-triplet corridors...")
    n_triplet_corridors = 0
    for ai, bi, ci in combinations(range(len(anchor_local)), 3):
        tri_locals = [anchor_local[ai], anchor_local[bi], anchor_local[ci]]
        tri_pts = coords[tri_locals]
        centered = tri_pts - tri_pts.mean(axis=0)
        if len(centered) >= 2:
            _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ Vt[0]
        else:
            proj = centered[:, 0]
        order = np.argsort(proj)
        ordered_locals = [tri_locals[i] for i in order]

        # Route: A -> B shortest path + B -> C shortest path
        all_path_stations: List[int] = []
        valid_route = True
        for seg_idx in range(2):
            src_nid = int(node_ids[ordered_locals[seg_idx]])
            dst_nid = int(node_ids[ordered_locals[seg_idx + 1]])
            try:
                path_nodes = nx.shortest_path(G, src_nid, dst_nid, weight="apm_cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                valid_route = False
                break
            seg_stations = [node_to_local[int(nid)] for nid in path_nodes
                           if int(nid) in node_to_local]
            if seg_idx == 0:
                all_path_stations.extend(seg_stations)
            else:
                # Skip first (it's the middle anchor, already included)
                all_path_stations.extend(seg_stations[1:])

        if not valid_route or len(all_path_stations) < 3:
            continue

        # Deduplicate while preserving order
        seen_li: set = set()
        deduped_path: List[int] = []
        for li in all_path_stations:
            if li not in seen_li:
                seen_li.add(li)
                deduped_path.append(li)

        spaced = _apply_dp_selection(deduped_path, coords, demand, feeder_cov)
        if spaced is None:
            continue

        valid, _ = validate_station_set(spaced, station_data)
        if valid:
            candidates.append({"stations": spaced, "source": "anchor_triplet"})
            n_triplet_corridors += 1

    logger.debug(f"    {n_triplet_corridors} anchor-triplet corridors")

    # --- 1c. Anchor-quadruplet corridors ---
    if len(anchor_local) >= 4:
        logger.debug("  Generating anchor-quadruplet corridors...")
        n_quad_corridors = 0
        all_quads = list(combinations(range(len(anchor_local)), 4))
        # Cap at 500 to avoid combinatorial blowup with many anchors
        _n_total_quads = len(all_quads)
        if _n_total_quads > 500:
            rng = np.random.default_rng(42)
            idx = rng.choice(_n_total_quads, 500, replace=False)
            all_quads = [all_quads[i] for i in idx]
            logger.debug(f"    Sampled 500 of {_n_total_quads} quadruplets")
        for quad in all_quads:
            quad_locals = [anchor_local[i] for i in quad]
            quad_pts = coords[quad_locals]
            centered = quad_pts - quad_pts.mean(axis=0)
            _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ Vt[0]
            order = np.argsort(proj)
            ordered_locals = [quad_locals[i] for i in order]

            all_path_stations = []
            valid_route = True
            for seg_idx in range(3):
                src_nid = int(node_ids[ordered_locals[seg_idx]])
                dst_nid = int(node_ids[ordered_locals[seg_idx + 1]])
                try:
                    path_nodes = nx.shortest_path(G, src_nid, dst_nid, weight="apm_cost")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    valid_route = False
                    break
                seg_stations = [node_to_local[int(nid)] for nid in path_nodes
                               if int(nid) in node_to_local]
                if seg_idx == 0:
                    all_path_stations.extend(seg_stations)
                else:
                    all_path_stations.extend(seg_stations[1:])

            if not valid_route or len(all_path_stations) < 4:
                continue

            seen_li = set()
            deduped_path = []
            for li in all_path_stations:
                if li not in seen_li:
                    seen_li.add(li)
                    deduped_path.append(li)

            spaced = _apply_dp_selection(deduped_path, coords, demand, feeder_cov)
            if spaced is None:
                continue

            valid, _ = validate_station_set(spaced, station_data)
            if valid:
                candidates.append({"stations": spaced, "source": "anchor_quadruplet"})
                n_quad_corridors += 1

        logger.debug(f"    {n_quad_corridors} anchor-quadruplet corridors")

    # --- 1d. Dense core corridors (short, high-density) ---
    logger.debug("  Generating dense core corridors...")
    dense_core = _generate_dense_core_corridors(station_data)
    candidates.extend(dense_core)
    logger.debug(f"    {len(dense_core)} dense core corridors")

    # --- 2. Demand-biased random walks ---
    logger.debug("  Generating random-walk corridors...")
    demand_sorted = np.argsort(-demand)

    # Seed stations: farthest-point sampling weighted by demand (vectorized)
    seeds = [int(demand_sorted[0])]
    # Track min distance from each station to any seed (updated incrementally)
    _min_dist_to_seeds = np.linalg.norm(
        coords - coords[seeds[0]], axis=1
    )  # (n_stations,)
    _n_target_seeds = min(20, n_stations)
    while len(seeds) < _n_target_seeds:
        # Score = min_dist_to_any_seed × demand × jitter
        _jitter = np.random.uniform(0.9, 1.1, size=n_stations)
        _scores = _min_dist_to_seeds * demand * _jitter
        # Zero out existing seeds so they can't be re-selected
        _scores[seeds] = -1.0
        best_idx = int(np.argmax(_scores))
        if _scores[best_idx] <= 0:
            break
        seeds.append(best_idx)
        # Incrementally update min distances with new seed
        _new_dists = np.linalg.norm(coords - coords[best_idx], axis=1)
        np.minimum(_min_dist_to_seeds, _new_dists, out=_min_dist_to_seeds)

    # Always include anchors as seeds
    for ali in anchor_local:
        if ali not in seeds:
            seeds.append(ali)

    n_walk_corridors = 0
    _walk_fail_reasons = {}  # diagnostic counts
    # Convert meter thresholds to feet for EPSG:2965 coordinate comparisons
    min_sep = MIN_STOP_SPACING_M / US_SURVEY_FT_TO_M
    max_sep = MAX_STOP_SPACING_M * 2.0 / US_SURVEY_FT_TO_M
    max_walk_dist = MAX_LENGTH_KM * 1000 / ROAD_CIRCUITY_FACTOR / US_SURVEY_FT_TO_M
    road_cache = station_data["road_dist_cache"]

    for _ in range(n_random_walks):
        seed = random.choice(seeds)
        # Pick a distant target anchor for strong directional pull
        target = random.choice(demand_sorted[:30].tolist())
        # Ensure target is at least 2km away for meaningful direction
        d_to_target = np.hypot(
            coords[target, 0] - coords[seed, 0],
            coords[target, 1] - coords[seed, 1],
        )
        if d_to_target < 2000 / US_SURVEY_FT_TO_M:
            # Pick a random far-away station instead
            dists_from_seed = np.hypot(
                coords[:, 0] - coords[seed, 0],
                coords[:, 1] - coords[seed, 1],
            )
            far_enough = np.where(dists_from_seed > 3000 / US_SURVEY_FT_TO_M)[0]
            if len(far_enough) > 0:
                target = int(random.choice(far_enough))

        stations_walk = [seed]
        total_dist = 0.0
        visited = {seed}
        _cum_road_m = 0.0  # running cumulative road distance for circuity check

        for _step in range(MAX_WALK_CANDIDATES - 1):
            current = stations_walk[-1]
            nearby = station_data["tree"].query_ball_point(
                coords[current], r=max_sep,
            )
            all_cands = [c for c in nearby if c not in visited]
            if not all_cands:
                break

            cx, cy = coords[current, 0], coords[current, 1]

            # Momentum direction: use last 2 stations if available, else target
            if len(stations_walk) >= 2:
                prev = stations_walk[-2]
                mom_dx = cx - coords[prev, 0]
                mom_dy = cy - coords[prev, 1]
                mom_len = np.hypot(mom_dx, mom_dy)
                if mom_len > 1e-6:
                    mom_angle = math.atan2(mom_dy, mom_dx)
                else:
                    mom_angle = math.atan2(
                        coords[target, 1] - cy, coords[target, 0] - cx,
                    )
            else:
                mom_angle = math.atan2(
                    coords[target, 1] - cy, coords[target, 0] - cx,
                )

            target_angle = math.atan2(
                coords[target, 1] - cy, coords[target, 0] - cx,
            )
            # Blend: 60% momentum, 40% target pull
            fwd_angle = math.atan2(
                0.6 * math.sin(mom_angle) + 0.4 * math.sin(target_angle),
                0.6 * math.cos(mom_angle) + 0.4 * math.cos(target_angle),
            )

            scores = []
            for c in all_cands:
                d_euclid = np.hypot(coords[c, 0] - cx, coords[c, 1] - cy)
                if d_euclid < min_sep:
                    continue

                # Road-graph circuity pre-check
                seg_key = (min(current, c), max(current, c))
                road_d = road_cache.get(seg_key)
                if road_d is not None:
                    d_euclid_m = d_euclid * US_SURVEY_FT_TO_M
                    seg_circ = road_d / max(d_euclid_m, 1.0)
                    if seg_circ > 1.45:  # tighter than 1.6 validation limit
                        continue
                elif d_euclid > max_sep * 0.8:
                    # Not in cache and far apart — likely too circuitous
                    continue

                # Road-bearing reversal pre-check: skip candidates that would
                # create a U-turn on the actual road path at the current station.
                if len(stations_walk) >= 2:
                    prev_li = stations_walk[-2]
                    arrive_key = (prev_li, current)
                    depart_key = (current, c)
                    brg = station_data.get("path_bearing_cache", {})
                    if arrive_key in brg and depart_key in brg:
                        _, arrive_brg = brg[arrive_key]
                        depart_brg, _ = brg[depart_key]
                        brg_diff = abs(arrive_brg - depart_brg)
                        if brg_diff > 180:
                            brg_diff = 360 - brg_diff
                        if brg_diff > PATH_BEARING_REVERSAL_DEG:
                            continue  # would fail validation anyway
                    else:
                        # Euclidean-bearing fallback when road-path cache
                        # is missing — avoids committing to gross reversals.
                        px, py = coords[prev_li, 0], coords[prev_li, 1]
                        arrive_brg_e = math.degrees(
                            math.atan2(cx - px, cy - py)
                        ) % 360
                        depart_brg_e = math.degrees(
                            math.atan2(coords[c, 0] - cx, coords[c, 1] - cy)
                        ) % 360
                        brg_diff_e = abs(arrive_brg_e - depart_brg_e)
                        if brg_diff_e > 180:
                            brg_diff_e = 360 - brg_diff_e
                        # Slightly looser threshold for Euclidean (roads
                        # bend more than straight lines)
                        if brg_diff_e > PATH_BEARING_REVERSAL_DEG + 15:
                            continue

                # Direction alignment with blended forward angle
                cand_angle = math.atan2(coords[c, 1] - cy, coords[c, 0] - cx)
                angle_diff = abs(fwd_angle - cand_angle)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                # Hard reject candidates more than 100° from forward direction
                if angle_diff > math.radians(100):
                    continue
                direction_score = math.cos(angle_diff)  # 1.0 = forward
                # Direction dominates (70%), demand secondary (30%)
                score = 0.3 * demand[c] / max(demand.max(), 1) + 0.7 * direction_score
                scores.append((c, max(score, 0.01), d_euclid))

            if not scores:
                break

            # Weighted random selection from top candidates
            scores.sort(key=lambda x: x[1], reverse=True)
            top_n = min(5, len(scores))
            top_scores = scores[:top_n]
            total_score = sum(s for _, s, _ in top_scores)
            if total_score <= 0:
                break

            probs = [s / total_score for _, s, _ in top_scores]
            chosen_idx = np.random.choice(top_n, p=probs)
            chosen_li, _, chosen_dist = top_scores[chosen_idx]

            total_dist += chosen_dist
            if total_dist > max_walk_dist:
                break

            stations_walk.append(chosen_li)
            visited.add(chosen_li)

            # Incrementally update cumulative road distance
            prev_li = stations_walk[-2]
            _seg_key = (min(prev_li, chosen_li), max(prev_li, chosen_li))
            _seg_rd = road_cache.get(_seg_key)
            if _seg_rd is not None:
                _cum_road_m += _seg_rd
            else:
                _cum_road_m += chosen_dist * US_SURVEY_FT_TO_M * ROAD_CIRCUITY_FACTOR

            # Running corridor circuity check — abandon walk if it meanders
            if len(stations_walk) >= 3:
                endpoint_euclid_m = float(np.linalg.norm(
                    coords[stations_walk[-1]] - coords[stations_walk[0]]
                )) * US_SURVEY_FT_TO_M
                if endpoint_euclid_m > 100:
                    if _cum_road_m / endpoint_euclid_m > 1.65:
                        break  # Abandon this walk early

        # Apply DP to select optimal subset from walk candidates
        dp_result = _apply_dp_selection(stations_walk, coords, demand, feeder_cov)
        submit = dp_result if dp_result is not None else stations_walk

        valid, reason = validate_station_set(submit, station_data)
        if valid:
            candidates.append({"stations": submit, "source": "random_walk"})
            n_walk_corridors += 1
        else:
            _walk_fail_reasons[reason.split("(")[0].strip()] = (
                _walk_fail_reasons.get(reason.split("(")[0].strip(), 0) + 1
            )

    logger.debug(f"    {n_walk_corridors} random-walk corridors")
    if _walk_fail_reasons:
        top_fails = sorted(_walk_fail_reasons.items(), key=lambda x: -x[1])[:3]
        logger.debug(f"    Walk failures: {', '.join(f'{r}={c}' for r, c in top_fails)}")

    # --- 3. Directional high-demand chains ---
    logger.debug("  Generating directional demand chains...")
    n_chain_corridors = 0
    # Centroid of all candidate stations
    centroid = coords.mean(axis=0)
    _BAND_HALF_WIDTH_FT = 1200.0 / US_SURVEY_FT_TO_M  # 1200m band (±600m from line)

    for _ in range(60):
        # Random direction through the station cloud (0-180°; reverse is symmetric)
        theta = random.uniform(0, math.pi)
        dx = math.cos(theta)
        dy = math.sin(theta)

        # Random lateral offset: shift the line ±2km from centroid
        offset_dist = random.uniform(-2000 / US_SURVEY_FT_TO_M, 2000 / US_SURVEY_FT_TO_M)
        # Perpendicular to direction
        origin = centroid + offset_dist * np.array([-dy, dx])

        # Find stations within band: perpendicular distance < BAND_HALF_WIDTH
        rel = coords - origin  # (N, 2)
        d_perp = np.abs(rel[:, 0] * (-dy) + rel[:, 1] * dx)
        in_band = np.where(d_perp < _BAND_HALF_WIDTH_FT)[0]

        if len(in_band) < 6:
            continue

        # From those in band, pick top 8-16 by demand
        band_demand = demand[in_band]
        k = min(random.randint(8, 16), len(in_band))
        top_k_idx = np.argsort(-band_demand)[:k]
        selected = in_band[top_k_idx].tolist()

        # Sort along the directional axis
        proj_along = coords[selected, 0] * dx + coords[selected, 1] * dy
        order = np.argsort(proj_along)
        selected = [selected[i] for i in order]

        dp_result = _apply_dp_selection(selected, coords, demand, feeder_cov)
        submit = dp_result if dp_result is not None else selected
        valid, _ = validate_station_set(submit, station_data)
        if valid:
            candidates.append({"stations": submit, "source": "demand_chain"})
            n_chain_corridors += 1

    logger.debug(f"    {n_chain_corridors} demand-chain corridors")

    # --- 4. Radial corridors from anchors ---
    # For each anchor, walk outward in 6 compass directions (60° apart)
    logger.debug("  Generating radial corridors from anchors...")
    n_radial_corridors = 0
    for ali in anchor_local:
        for bearing_deg in range(0, 360, 60):
            bearing_rad = math.radians(bearing_deg)
            stations_radial = [ali]
            visited_r = {ali}
            for _step in range(MAX_WALK_CANDIDATES - 1):
                current = stations_radial[-1]
                nearby = station_data["tree"].query_ball_point(
                    coords[current], r=max_sep,
                )
                cands = [c for c in nearby if c not in visited_r]
                if not cands:
                    break
                cx, cy = coords[current, 0], coords[current, 1]
                best_c, best_score = None, -999
                for c in cands:
                    d_c = np.hypot(coords[c, 0] - cx, coords[c, 1] - cy)
                    if d_c < min_sep:
                        continue
                    cand_angle = math.atan2(coords[c, 1] - cy, coords[c, 0] - cx)
                    angle_diff = abs(bearing_rad - cand_angle)
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff
                    if angle_diff > math.radians(70):
                        continue
                    sc = math.cos(angle_diff) + 0.3 * demand[c] / max(demand.max(), 1)
                    if sc > best_score:
                        best_score = sc
                        best_c = c
                if best_c is None:
                    break
                stations_radial.append(best_c)
                visited_r.add(best_c)
            dp_result = _apply_dp_selection(stations_radial, coords, demand, feeder_cov)
            submit = dp_result if dp_result is not None else stations_radial
            valid, _ = validate_station_set(submit, station_data)
            if valid:
                candidates.append({"stations": submit, "source": "radial"})
                n_radial_corridors += 1

    logger.debug(f"    {n_radial_corridors} radial corridors")

    # Deduplicate
    candidates = deduplicate_station_sets(candidates, station_data)
    logger.debug(f"  Total initial candidates: {len(candidates)}")
    return candidates


def run_station_first_search(
    data: dict,
    use_network: bool = True,
    n_iterations: int = 10,
    population_size: int = 50,
    n_output: int = 17,
    evaluation_mode: str = "isolated",
    network_synergy_weight: float = NETWORK_SYNERGY_WEIGHT_DEFAULT,
    network_anchor_top_k: int = 12,
    network_transfer_radius_m: float = NETWORK_TRANSFER_RADIUS_M,
    n_seeds: int = 1,
) -> Tuple[List[dict], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Run the station-first corridor search pipeline.

    Stages:
    1. Extract candidate stations from OSM road graph
    2. Generate initial station-set corridors
    3. NSGA-II with station-set genetic operators
    4. Diversity selection
    5. Road-graph alignment for final corridors
    6. Output GeoJSON
    """
    parcels = data["parcels"]
    od_flows = data["od_flows"]
    bus_stops = data["bus_stops"]

    # Step 1: Load road network + extract candidate stations
    bounds_4326 = parcels.to_crs(4326).total_bounds
    road_graph = load_road_network(bounds_4326, use_network=use_network)
    if road_graph is None:
        raise RuntimeError("Road graph required for station-first search")

    # Prepare bus stop spatial index
    if len(bus_stops) > 0:
        bus_proj = bus_stops.to_crs(PROJECT_CRS)
        bus_coords = np.column_stack([bus_proj.geometry.x, bus_proj.geometry.y])
        bus_tree_obj = cKDTree(bus_coords)
    else:
        bus_coords = None
        bus_tree_obj = None

    # Attach institutional weights to parcels for student ridership proxy
    parcels["institutional_weight"] = data["institutional_weights"]

    logger.info("\nBuilding candidate stations from road graph...")
    station_data = build_candidate_stations(
        road_graph, parcels,
        bus_stops_proj=bus_coords,
        bus_tree=bus_tree_obj,
    )

    # Pre-compute all-pairs road distances for nearby station pairs.
    # Uses disk cache to avoid redundant Dijkstra recomputation.
    logger.info("\nPre-computing station-pair distances...")
    _load_or_compute_station_distances(station_data)

    # Parcel lookup for OD scoring (both bare and ST-prefixed IDs)
    parcels_proj = parcels.to_crs(PROJECT_CRS)
    _pid_arr = parcels["PARCEL_ID"].astype(str).values
    _cx_arr = parcels_proj.geometry.centroid.x.values
    _cy_arr = parcels_proj.geometry.centroid.y.values
    parcel_lookup = {}
    for _k in range(len(_pid_arr)):
        _xy = (_cx_arr[_k], _cy_arr[_k])
        _bare = _pid_arr[_k]
        parcel_lookup[_bare] = _xy
        parcel_lookup["ST" + _bare] = _xy  # OD flows use ST prefix

    # Pre-build OD coordinate arrays once (avoids 510K dict lookups per candidate)
    if od_flows is not None and len(od_flows) > 0:
        _origins = od_flows["origin_parcel"].astype(str).values
        _dests = od_flows["dest_parcel"].astype(str).values
        _trips = od_flows["trips"].values
        _nan2 = (np.nan, np.nan)
        _o_xy = np.array([parcel_lookup.get(o, _nan2) for o in _origins])
        _d_xy = np.array([parcel_lookup.get(d, _nan2) for d in _dests])
        _od_valid = (
            (_trips >= 0.01) &
            ~np.isnan(_o_xy[:, 0]) &
            ~np.isnan(_d_xy[:, 0])
        )
        station_data["od_cache"] = {
            "o_coords": np.ascontiguousarray(_o_xy[_od_valid]),
            "d_coords": np.ascontiguousarray(_d_xy[_od_valid]),
            "trips": _trips[_od_valid].copy(),
        }
        logger.debug(f"  OD cache: {int(_od_valid.sum()):,} valid flows of {len(_trips):,}")

    # Synergy scoring config (shared across seeds)
    _synergy_kwargs = dict(
        evaluation_mode=evaluation_mode,
        synergy_weight=network_synergy_weight,
        anchor_top_k=network_anchor_top_k,
        transfer_radius_m=network_transfer_radius_m,
    )

    # Multi-seed loop: generate + evolve independently per seed, pool results
    _effective_seeds = max(n_seeds, 1)
    all_evolved: List[dict] = []
    _seen_signatures: set = set()

    for seed_idx in range(_effective_seeds):
        if _effective_seeds > 1:
            random.seed(42 + seed_idx)
            np.random.seed(42 + seed_idx)
            logger.info(f"\n{'=' * 50}")
            logger.debug(f"  SEED {seed_idx + 1}/{_effective_seeds}")
            logger.info(f"{'=' * 50}")

        # Step 2: Generate initial station-set candidates
        logger.info("\nGenerating initial station-set candidates...")
        initial = generate_initial_station_sets(
            station_data, od_flows=od_flows, parcel_lookup=parcel_lookup,
        )

        # Diagnostic: candidates by source
        _sources: dict = {}
        for _c in initial:
            _src = _c.get("source", "unknown")
            _sources[_src] = _sources.get(_src, 0) + 1
        if _sources:
            logger.debug("  Candidates by source:")
            for _src, _cnt in sorted(_sources.items(), key=lambda x: -x[1]):
                logger.debug(f"    {_src}: {_cnt}")

        # Step 3: Score initial population
        logger.info("\nScoring initial candidates...")
        population: List[dict] = []
        for i, cand in enumerate(initial):
            if (i + 1) % 20 == 0:
                logger.debug(f"  Scored {i + 1}/{len(initial)}")
            score = score_station_set(
                cand["stations"], station_data,
                od_flows=od_flows, parcel_lookup=parcel_lookup,
            )
            if score is None:
                continue  # infeasible curve or speed — skip
            cand["score"] = score
            population.append(cand)

        logger.debug(f"  Scored {len(population)} candidates")
        if not population:
            logger.debug("  WARNING: No valid candidates in this seed")
            continue

        population = apply_station_synergy_scores(
            population, station_data, **_synergy_kwargs,
        )

        # Step 4: NSGA-II evolutionary search
        _WEAK_POP_THRESHOLD = 8
        effective_iterations = n_iterations
        initial_pop_size = len(population)
        if initial_pop_size < _WEAK_POP_THRESHOLD:
            effective_iterations = max(n_iterations, 30)
            logger.info(f"\n  Weak initial population ({initial_pop_size}), "
                  f"extending iterations: {n_iterations} -> {effective_iterations}")
        logger.info(f"\nRunning NSGA-II optimization ({effective_iterations} iterations, "
              f"early-stop after 4 stagnant)...")
        best_ridership_ever = 0
        _last_improve = 0
        _cached_fronts = None

        for iteration in range(effective_iterations):
            if len(population) > population_size:
                population = nsga2_select(population, population_size, precomputed_fronts=_cached_fronts, station_data=station_data)
                _cached_fronts = None

            # Generate unscored offspring, then deduplicate before scoring
            raw_offspring: List[dict] = []
            n_mutations = min(population_size, len(population) * 2)
            n_crossovers = min(population_size // 2, len(population))

            for _ in range(n_mutations):
                parent = random.choice(population)
                child = mutate_station_set(parent, station_data)
                if child is not None:
                    raw_offspring.append(child)

            for _ in range(n_crossovers):
                if len(population) < 2:
                    break
                p1, p2 = random.sample(population, 2)
                child = crossover_station_sets(p1, p2, station_data)
                if child is not None:
                    raw_offspring.append(child)

            # Deduplicate before expensive scoring
            raw_offspring = deduplicate_station_sets(raw_offspring, station_data)

            offspring = []
            for child in raw_offspring:
                score = score_station_set(
                    child["stations"], station_data,
                    od_flows=od_flows, parcel_lookup=parcel_lookup,
                )
                if score is not None:
                    child["score"] = score
                    offspring.append(child)
            population.extend(offspring)
            population = apply_station_synergy_scores(
                population, station_data, **_synergy_kwargs,
            )

            if population:
                best_r = max(c["score"]["ridership_est"] for c in population)
                avg_r = np.mean([c["score"]["ridership_est"] for c in population])
                _cached_fronts = fast_non_dominated_sort(population)
                fronts = _cached_fronts
                pareto_size = len(fronts[0]) if fronts else 0

                logger.debug(
                    f"  Iter {iteration + 1}/{effective_iterations}: "
                    f"pop={len(population)}, "
                    f"best={best_r:.0f}, avg={avg_r:.0f}, "
                    f"fronts={len(fronts)}, pareto={pareto_size}, "
                    f"offspring={len(offspring)}"
                )

                ridership_improved = best_r > best_ridership_ever * 1.005
                if iteration > 4 and not ridership_improved:
                    stagnant = iteration - _last_improve
                    if stagnant >= 4:
                        logger.debug(f"  Stopping: stable for {stagnant} iterations")
                        break
                else:
                    _last_improve = iteration
                best_ridership_ever = max(best_ridership_ever, best_r)

        # Accumulate evolved candidates, deduplicating across seeds
        _seed_added = 0
        for cand in population:
            sig = tuple(sorted(cand["stations"]))
            if sig not in _seen_signatures:
                _seen_signatures.add(sig)
                all_evolved.append(cand)
                _seed_added += 1
        if _effective_seeds > 1:
            logger.debug(f"  Seed {seed_idx + 1}: {_seed_added} unique corridors added "
                  f"(pool: {len(all_evolved)} total)")

    population = all_evolved
    if _effective_seeds > 1:
        logger.info(f"\n  Combined pool: {len(population)} unique corridors from {_effective_seeds} seeds")

    if not population:
        logger.debug("  WARNING: No valid candidates generated!")
        empty_gdf = gpd.GeoDataFrame(
            columns=["corridor_id", "geometry"], geometry="geometry", crs="EPSG:4326",
        )
        return [], empty_gdf, empty_gdf

    # Step 5: Select diverse corridors (progressive relaxation if too few)
    logger.info(f"\nSelecting diverse corridors (max overlap fraction={OVERLAP_THRESHOLD:.0%}, buffer={OVERLAP_BUFFER_M:.0f}m)...")
    final_selection = select_diverse_station_sets(
        population, station_data,
        max_overlap=OVERLAP_THRESHOLD,
        max_select=n_output,
    )
    if len(final_selection) < n_output:
        for relax_step in (0.10, 0.20, 0.30):
            relaxed_threshold = min(OVERLAP_THRESHOLD + relax_step, 0.85)
            relaxed = select_diverse_station_sets(
                population, station_data,
                max_overlap=relaxed_threshold,
                max_select=n_output,
            )
            if len(relaxed) > len(final_selection):
                final_selection = relaxed
                logger.debug(f"  Relaxed overlap to {relaxed_threshold:.0%} -> {len(relaxed)} corridors")
            if len(final_selection) >= n_output:
                break
    logger.debug(f"  Selected {len(final_selection)} diverse corridors")

    # Remove exact-duplicate station sets
    _seen_keys: set = set()
    _deduped_selection: List[dict] = []
    for _cand in final_selection:
        _key = tuple(sorted(_cand["stations"]))
        if _key not in _seen_keys:
            _seen_keys.add(_key)
            _deduped_selection.append(_cand)
    if len(_deduped_selection) < len(final_selection):
        logger.debug(f"  Removed {len(final_selection) - len(_deduped_selection)} exact-duplicate station sets")
        final_selection = _deduped_selection

    # Step 5b: Local hill-climbing refinement on selected corridors
    logger.info("\nRefining station placements (hill-climbing)...")
    final_selection = refine_station_placements(
        final_selection, station_data,
        od_flows=od_flows, parcel_lookup=parcel_lookup,
    )

    # Post-refinement dedup: hill-climbing can converge different corridors
    _seen_keys2: set = set()
    _deduped2: List[dict] = []
    for _cand in final_selection:
        _key = tuple(sorted(_cand["stations"]))
        if _key not in _seen_keys2:
            _seen_keys2.add(_key)
            _deduped2.append(_cand)
    if len(_deduped2) < len(final_selection):
        logger.debug(f"  Post-refinement: removed {len(final_selection) - len(_deduped2)} converged duplicates")
        final_selection = _deduped2

    # Step 6: Generate road-graph alignment for final corridors
    logger.info("\nGenerating road-graph alignments for final corridors...")
    corridors_out = []
    stops_out = []
    coords_proj = station_data["coords_proj"]
    coords_4326 = station_data["coords_4326"]
    node_ids = station_data["node_ids"]

    # Build bench of backup corridors from the GA population for post-routing
    # replacement.  If a corridor is rejected after routing (infeasible curves
    # on actual road geometry), the next-best bench corridor takes its slot.
    _selected_keys = {tuple(sorted(c["stations"])) for c in final_selection}
    bench = [
        c for c in population
        if "score" in c and tuple(sorted(c["stations"])) not in _selected_keys
    ]
    bench.sort(key=lambda c: c["score"].get("ridership_est", 0), reverse=True)
    n_curve_rejected = 0
    n_replaced = 0

    # Pre-compute arterial node set for station snapping.
    # Stations on minor streets get snapped to the nearest primary/secondary
    # node within 200m so the display alignment follows major roads naturally.
    _ARTERIAL_CLASSES = frozenset({"primary", "secondary", "primary_link", "secondary_link"})
    _SNAP_MAX_M = 200.0  # max snap distance in meters
    _SNAP_MAX_FT = _SNAP_MAX_M / US_SURVEY_FT_TO_M
    _arterial_nodes = set()
    for nid in station_data["graph"].nodes():
        for _u, _v, edata in station_data["graph"].edges(nid, data=True):
            hw = edata.get("highway", "")
            if isinstance(hw, list):
                hw = hw[0] if hw else ""
            if hw in _ARTERIAL_CLASSES:
                _arterial_nodes.add(nid)
                break
    # Build KDTree of arterial node positions for fast nearest-neighbor
    _nid_list = list(station_data["graph"].nodes())
    _nid_to_idx = {nid: i for i, nid in enumerate(_nid_list)}
    _arterial_local_indices = []
    _arterial_proj_coords = []
    for nid in _arterial_nodes:
        if nid in _nid_to_idx:
            _global_idx = _nid_to_idx[nid]
            # Map graph node to station_data local index if present
            for _li, _snid in enumerate(node_ids):
                if _snid == nid:
                    _arterial_local_indices.append(_li)
                    _arterial_proj_coords.append(coords_proj[_li])
                    break
    _arterial_tree = None
    if _arterial_proj_coords:
        _arterial_proj_arr = np.array(_arterial_proj_coords)
        _arterial_tree = cKDTree(_arterial_proj_arr)

    def _snap_to_arterial(station_locals_list):
        """Snap stations to nearest arterial node within _SNAP_MAX_FT."""
        if _arterial_tree is None:
            return station_locals_list
        snapped = []
        for li in station_locals_list:
            pt = coords_proj[li]
            dist, idx = _arterial_tree.query(pt)
            if dist < _SNAP_MAX_FT:
                snapped.append(_arterial_local_indices[idx])
            else:
                snapped.append(li)
        # Deduplicate consecutive identical stations
        deduped = [snapped[0]]
        for s in snapped[1:]:
            if s != deduped[-1]:
                deduped.append(s)
        return deduped

    for i, corridor in enumerate(final_selection):
        cid = f"C{i + 1}"
        sl = _snap_to_arterial(corridor["stations"])

        # Generate alignment
        alignment = route_through_stations(sl, station_data)
        if alignment is None:
            # Fallback: straight line through station coordinates
            pts_proj = coords_proj[sl]
            alignment = LineString(pts_proj.tolist())

        length_km = alignment.length * US_SURVEY_FT_TO_M / 1000

        # Post-routing path quality check
        path_warnings = check_routed_path_quality(alignment, coords_proj[sl])
        if path_warnings:
            logger.debug(f"  WARNING {cid}: {'; '.join(path_warnings)}")

        # Post-routing curve validation on simplified routed path.
        # The raw road-graph alignment has many vertices at intersections
        # with sub-meter spacing, creating artificial sharp angles.  An APM
        # guideway uses engineered curves, not the exact road-grid vertices.
        # Simplify with Douglas-Peucker (3× min curve radius) before checking.
        _VALIDATE_TOLERANCE_M = APM_MIN_CURVE_RADIUS_M * 3.0
        _simplified = alignment.simplify(
            _VALIDATE_TOLERANCE_M / US_SURVEY_FT_TO_M,
            preserve_topology=True,
        )
        routed_pts_m = np.array(_simplified.coords) * US_SURVEY_FT_TO_M
        if len(routed_pts_m) >= 3:
            routed_curve_info = compute_curve_speed_penalties(routed_pts_m)
            _rejected = False
            if routed_curve_info["has_infeasible_curve"]:
                logger.debug(f"  REJECT {cid}: routed path has infeasible curve "
                      f"(min radius {routed_curve_info['min_curve_radius_m']:.1f}m "
                      f"< {APM_MIN_CURVE_RADIUS_M}m)")
                _rejected = True
            if not _rejected:
                # Compute effective speed from curve delay on routed path
                routed_delay_s = routed_curve_info["total_curve_delay_s"]
                if length_km > 0 and routed_delay_s > 0:
                    base_time_h = length_km / APM_SPEED_KPH
                    routed_eff_speed = length_km / (base_time_h + routed_delay_s / 3600.0)
                else:
                    routed_eff_speed = APM_SPEED_KPH
                if routed_eff_speed < APM_SPEED_KPH * EFFECTIVE_SPEED_FLOOR_FRACTION:
                    logger.debug(f"  REJECT {cid}: routed path effective speed "
                          f"{routed_eff_speed:.1f} kph < floor "
                          f"{APM_SPEED_KPH * EFFECTIVE_SPEED_FLOOR_FRACTION:.0f} kph")
                    _rejected = True
            if _rejected:
                n_curve_rejected += 1
                # Try bench replacements
                _replaced = False
                while bench and not _replaced:
                    replacement = bench.pop(0)
                    r_sl = replacement["stations"]
                    r_align = route_through_stations(r_sl, station_data)
                    if r_align is None:
                        r_pts = coords_proj[r_sl]
                        r_align = LineString(r_pts.tolist())
                    r_len = r_align.length * US_SURVEY_FT_TO_M / 1000
                    r_simp = r_align.simplify(
                        _VALIDATE_TOLERANCE_M / US_SURVEY_FT_TO_M,
                        preserve_topology=True,
                    )
                    r_pts_m = np.array(r_simp.coords) * US_SURVEY_FT_TO_M
                    if len(r_pts_m) >= 3:
                        r_ci = compute_curve_speed_penalties(r_pts_m)
                        if r_ci["has_infeasible_curve"]:
                            continue
                        r_delay = r_ci["total_curve_delay_s"]
                        if r_len > 0 and r_delay > 0:
                            r_btime = r_len / APM_SPEED_KPH
                            r_espd = r_len / (r_btime + r_delay / 3600.0)
                        else:
                            r_espd = APM_SPEED_KPH
                        if r_espd < APM_SPEED_KPH * EFFECTIVE_SPEED_FLOOR_FRACTION:
                            continue
                    # Replacement passes — swap in
                    corridor = replacement
                    sl = r_sl
                    alignment = r_align
                    length_km = r_len
                    path_warnings = check_routed_path_quality(r_align, coords_proj[r_sl])
                    _replaced = True
                    n_replaced += 1
                    logger.debug(f"  Replaced {cid} with bench corridor "
                          f"(ridership={replacement['score']['ridership_est']:.0f})")
                if not _replaced:
                    continue

        # Station coordinates in 4326
        stop_coords_4326 = []
        for li in sl:
            lon, lat = coords_4326[li]
            stop_coords_4326.append([round(float(lon), 10), round(float(lat), 10)])

        # Corridor geometry: station-to-station LineString.
        # Matches what the viewer draws and what Stage 2 uses for
        # catchment computation (via stop_coords, not geometry interpolation).
        # The display-simplified road path (_display_geom) is no longer
        # stored — it was misleading because Stage 2 didn't use it.
        line_4326 = LineString(stop_coords_4326)

        # Recompute barrier cost on actual alignment
        barrier_cost = barrier_crossing_cost_usd(alignment)

        corridors_out.append({
            "corridor_id": cid,
            "length_km": length_km,
            "n_stops": len(sl),
            "stop_coords": json.dumps(stop_coords_4326),
            "source": corridor.get("source", "unknown"),
            "ridership_est": corridor["score"]["ridership_est"],
            "student_ridership": corridor["score"].get("student_ridership", 0.0),
            "tif_potential": corridor["score"]["tif_potential"],
            "tif_viable": corridor["score"].get("tif_viable", True),
            "cost_efficiency": corridor["score"]["cost_efficiency"],
            "viability_indicator": corridor["score"].get("viability_indicator", 0.0),
            "bus_competition": corridor["score"]["bus_competition"],
            "apm_share": corridor["score"]["apm_share"],
            "weighted_demand": corridor["score"]["weighted_demand"],
            "barrier_cost_usd": barrier_cost,
            "curve_cost_mult": corridor["score"].get("curve_cost_mult", 1.0),
            "n_turns": corridor["score"].get("n_turns", 0),
            "corridor_circuity": corridor["score"].get("corridor_circuity", 0.0),
            "min_curve_radius_m": corridor["score"].get("min_curve_radius_m", 9999),
            "curve_delay_s": corridor["score"].get("curve_delay_s", 0),
            "effective_speed_kph": corridor["score"].get("effective_speed_kph", APM_SPEED_KPH),
            "cumulative_turn_degrees": round(sum(corridor["score"].get("turn_angles", [])), 1),
            "network_synergy": corridor["score"].get("network_synergy", 0.0),
            "ridership_est_base": corridor["score"].get("ridership_est_base", corridor["score"]["ridership_est"]),
            "ridership_network_adjusted": corridor["score"].get("ridership_network_adjusted", corridor["score"]["ridership_est"]),
            "evaluation_mode": corridor["score"].get("evaluation_mode", "isolated"),
            "geometry": line_4326,
        })

        for j, li in enumerate(sl):
            lon, lat = coords_4326[li]
            stops_out.append({
                "corridor_id": cid,
                "stop_id": f"{cid}_S{j + 1}",
                "sequence": j + 1,
                "geometry": Point(float(lon), float(lat)),
            })

        logger.debug(
            f"  {cid}: {length_km:.1f}km, {len(sl)} stops, "
            f"ridership_est={corridor['score']['ridership_est']:.0f}, "
            f"source={corridor.get('source', '?')}"
        )

    if n_curve_rejected > 0:
        logger.debug(f"  Post-routing: {n_curve_rejected} rejected, {n_replaced} replaced from bench")

    # Step 7: Deduplicate by routed geometry overlap.
    # After evolution, corridors with different station sets can cover
    # nearly identical service areas.  Cluster by bidirectional polyline
    # overlap on the station-to-station geometry and keep only the best
    # representative (highest ridership_est) from each cluster.
    if len(corridors_out) > 1:
        _DEDUP_OVERLAP = 0.90  # symmetric overlap threshold
        _DEDUP_BUFFER_M = 400.0
        _dedup_buffer_ft = _DEDUP_BUFFER_M / US_SURVEY_FT_TO_M
        _dedup_spacing_ft = 200.0 / US_SURVEY_FT_TO_M

        # Sample each routed geometry into dense points (projected CRS)
        _dedup_polys = []
        for _co in corridors_out:
            _line_proj = LineString(
                [_to_proj.transform(x, y) for x, y in _co["geometry"].coords]
            )
            _pts = [
                np.array(_line_proj.interpolate(float(d)).coords[0])
                for d in np.arange(0, _line_proj.length + 1e-3, _dedup_spacing_ft)
            ]
            _dedup_polys.append(np.array(_pts) if _pts else np.empty((0, 2)))

        _dedup_trees = [cKDTree(p) if len(p) > 0 else None for p in _dedup_polys]

        # Greedy clustering: sorted by ridership (best first)
        _order = sorted(
            range(len(corridors_out)),
            key=lambda k: corridors_out[k].get("ridership_est", 0),
            reverse=True,
        )
        _kept_indices: list[int] = []
        _removed_indices: list[int] = []
        for _idx in _order:
            _is_dup = False
            for _ki in _kept_indices:
                _ovlp = _bidirectional_overlap(
                    _dedup_polys[_idx], _dedup_polys[_ki],
                    tree_a=_dedup_trees[_idx], tree_b=_dedup_trees[_ki],
                )
                if _ovlp > _DEDUP_OVERLAP:
                    _is_dup = True
                    break
            if _is_dup:
                _removed_indices.append(_idx)
            else:
                _kept_indices.append(_idx)

        if _removed_indices:
            _removed_cids = [corridors_out[j]["corridor_id"] for j in _removed_indices]
            logger.info(f"\n  Geometry dedup: removed {len(_removed_indices)} near-duplicate corridors "
                  f"({', '.join(_removed_cids)}) at {_DEDUP_OVERLAP:.0%} overlap threshold")
            # Keep survivors, renumber corridor IDs
            corridors_out = [corridors_out[j] for j in _kept_indices]
            stops_out = [
                s for s in stops_out
                if s["corridor_id"] not in set(_removed_cids)
            ]
            # Renumber
            _old_to_new = {}
            for _ni, _co in enumerate(corridors_out):
                _old_cid = _co["corridor_id"]
                _new_cid = f"C{_ni + 1}"
                _old_to_new[_old_cid] = _new_cid
                _co["corridor_id"] = _new_cid
            for _so in stops_out:
                _so["corridor_id"] = _old_to_new[_so["corridor_id"]]
                _so["stop_id"] = f"{_so['corridor_id']}_S{_so['sequence']}"
            logger.debug(f"  {len(corridors_out)} distinct corridors remain")

    # Build output GeoDataFrames
    if corridors_out:
        corridors_gdf = gpd.GeoDataFrame(corridors_out, geometry="geometry", crs="EPSG:4326")
    else:
        corridors_gdf = gpd.GeoDataFrame(
            columns=["corridor_id", "length_km", "n_stops", "source",
                      "ridership_est", "tif_potential", "cost_efficiency",
                      "bus_competition", "apm_share", "weighted_demand",
                      "barrier_cost_usd", "network_synergy", "ridership_est_base",
                      "ridership_network_adjusted", "evaluation_mode", "geometry"],
            geometry="geometry", crs="EPSG:4326",
        )

    if stops_out:
        stops_gdf = gpd.GeoDataFrame(stops_out, geometry="geometry", crs="EPSG:4326")
    else:
        stops_gdf = gpd.GeoDataFrame(
            columns=["corridor_id", "stop_id", "sequence", "geometry"],
            geometry="geometry", crs="EPSG:4326",
        )

    return final_selection, corridors_gdf, stops_gdf

def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Optimized APM Corridor Search")
    parser.add_argument("--no-network", action="store_true",
                        help="Disable OSM network routing (use straight lines)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="NSGA-II iterations (default: 50)")
    parser.add_argument("--population", type=int, default=200,
                        help="NSGA-II population size (default: 200)")
    parser.add_argument("--output", type=int, default=17,
                        help="Number of corridors to output (default: 17)")
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        default="isolated",
        choices=["isolated", "network_aware"],
        help="Corridor evaluation mode: isolated or network_aware",
    )
    parser.add_argument(
        "--network-synergy-weight",
        type=float,
        default=NETWORK_SYNERGY_WEIGHT_DEFAULT,
        help="Weight for network synergy uplift in network_aware mode",
    )
    parser.add_argument(
        "--network-anchor-top-k",
        type=int,
        default=12,
        help="Top-K isolated candidates used as network anchors",
    )
    parser.add_argument(
        "--network-transfer-radius-m",
        type=float,
        default=NETWORK_TRANSFER_RADIUS_M,
        help="Endpoint transfer proximity radius (meters) for synergy scoring",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of random seeds for generation diversity (default: 1)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("STATION-FIRST APM CORRIDOR SEARCH")
    logger.info("=" * 70)
    logger.info("")
    logger.debug("  Mode: STATION-FIRST")
    logger.debug("  1. Station siting from OSM intersections (MCLP)")
    logger.debug("  2. NSGA-II on station sets (not line geometry)")
    logger.debug("  3. Road-graph alignment for final corridors only")
    logger.debug(f"  Evaluation mode: {args.evaluation_mode}")
    logger.info("")

    # Load data
    data = load_all_data()

    # Run search
    results, corridors_gdf, stops_gdf = run_station_first_search(
        data,
        use_network=not args.no_network,
        n_iterations=args.iterations,
        population_size=args.population,
        n_output=args.output,
        evaluation_mode=args.evaluation_mode,
        network_synergy_weight=args.network_synergy_weight,
        network_anchor_top_k=args.network_anchor_top_k,
        network_transfer_radius_m=args.network_transfer_radius_m,
        n_seeds=args.seeds,
    )

    # Save outputs
    logger.info("\n" + "=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)

    corridors_path = PROC_DIR / "apm_phase2a_corridors.geojson"
    stops_path = PROC_DIR / "apm_phase2a_stops.geojson"
    results_csv = PROC_DIR / "apm_optimized_search_results.csv"

    # Backup existing files
    for path in [corridors_path, stops_path]:
        if path.exists():
            backup = path.with_name(path.stem + "_pre_optimization" + path.suffix)
            if not backup.exists():
                import shutil
                shutil.copy2(path, backup)
                logger.debug(f"  Backed up: {path.name} -> {backup.name}")

    corridors_gdf.to_file(corridors_path, driver="GeoJSON")
    stops_gdf.to_file(stops_path, driver="GeoJSON")

    # Save summary CSV
    summary_df = corridors_gdf.drop(columns=["geometry"])
    summary_df.to_csv(results_csv, index=False)

    logger.info(f"\n  Corridors: {corridors_path}")
    logger.debug(f"  Stops: {stops_path}")
    logger.debug(f"  Results CSV: {results_csv}")

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"\nGenerated {len(corridors_gdf)} corridors with {len(stops_gdf)} stops")
    logger.info(f"Evaluation mode: {args.evaluation_mode}")
    logger.info("")
    logger.info(f"{'ID':>5s} {'Length':>7s} {'Stops':>5s} {'Ridership':>10s} {'TIF':>8s} "
          f"{'Efficiency':>10s} {'BusComp':>7s} {'Source':>12s}")
    logger.info("-" * 75)

    for _, row in corridors_gdf.iterrows():
        logger.info(
            f"{row['corridor_id']:>5s} {row['length_km']:>6.1f}km {row['n_stops']:>5d} "
            f"{row['ridership_est']:>10.0f} {row['tif_potential']:>8.1f} "
            f"{row['cost_efficiency']:>10.4f} {row['bus_competition']:>6.2f} "
            f"{row['source']:>12s}"
        )

    # Pareto front summary
    fronts = fast_non_dominated_sort(results)
    logger.info(f"\nPareto front: {len(fronts[0])} non-dominated solutions")
    logger.info(f"Ridership range: {corridors_gdf['ridership_est'].min():.0f} - "
          f"{corridors_gdf['ridership_est'].max():.0f}")
    logger.info(f"Variation ratio: {corridors_gdf['ridership_est'].max() / max(corridors_gdf['ridership_est'].min(), 1):.1f}x")


# ---------------------------------------------------------------------------
# Re-exports from extracted modules (backward compatibility)
# These imports are placed AFTER all function definitions to avoid circular
# imports — corridor_evolution.py lazy-imports validate_station_set and
# score_station_set from this module.
# ---------------------------------------------------------------------------
from scripts.corridor_geometry import *   # noqa: F401,F403
from scripts.corridor_evolution import *  # noqa: F401,F403

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

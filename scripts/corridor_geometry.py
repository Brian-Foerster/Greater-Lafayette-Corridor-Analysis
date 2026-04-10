"""
Corridor Geometry Utilities
============================

Geometry helpers extracted from optimized_corridor_search.py:
  - Barrier/bridge crossing detection and costing
  - Path bearing and road-graph validation helpers
  - Curve physics (speed penalties)
  - Road network loading, routing, and smoothing
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point

import logging
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.spatial_constants import PROJECT_CRS
from src.financial_params import CAPITAL_COST_GUIDEWAY_PER_KM

__all__ = [
    # Barrier / bridge geometry
    "count_barrier_crossings",
    "barrier_crossing_cost_usd",
    "barrier_cost_multiplier",
    "_build_barrier_lines_proj",
    "_build_river_line",
    "_node_side_of_river",
    "add_bridge_zone_edges",
    "_node_to_proj",
    # Path bearing / validation helpers
    "_node_touches_major_road",
    "_has_major_road_edge",
    "_path_bearing_at",
    "_road_graph_distance",
    # Curve physics
    "compute_curve_speed_penalties",
    # Road network loading / routing
    "route_through_stations",
    "check_routed_path_quality",
    "_augment_graph_weights",
    "load_road_network",
    "route_on_network",
    "_chaikin_cut",
    "_simplify_with_angle_constraint",
]

# ── Coordinate transformers ──────────────────────────────────────────
_to_proj = Transformer.from_crs("EPSG:4326", PROJECT_CRS, always_xy=True)
_to_4326 = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)

# ── Path constants ───────────────────────────────────────────────────
DATA_DIR = REPO_ROOT / "data"
PROC_DIR = DATA_DIR / "processed"

# ── APM Vehicle Specifications ───────────────────────────────────────
APM_SPEED_KPH = 40                                         # urban service speed (80 kph design max)
APM_LINE_SPEED_MS = APM_SPEED_KPH / 3.6                   # 11.11 m/s
APM_SERVICE_ACCEL_MS2 = 1.0                                # service accel (Innovia/Crystal Mover spec)
APM_SERVICE_DECEL_MS2 = 1.0                                # service braking (symmetric)
APM_LATERAL_ACCEL_LIMIT_MS2 = 0.687                        # 0.07g comfort target (ASCE 21 max is 0.10g)
APM_MIN_CURVE_RADIUS_M = 50.0                              # ASCE 21 restricted service min (25m is depot-only)
# Derived: radius at which line speed can be maintained
APM_FULL_SPEED_RADIUS_M = APM_LINE_SPEED_MS ** 2 / APM_LATERAL_ACCEL_LIMIT_MS2  # ~180m
# Curved guideway construction premium (industry 1.5-2.5× for tight curves)
CURVE_CONSTRUCTION_PREMIUM = 1.5
# Effective speed floor: reject if curves degrade avg speed below this × line speed
EFFECTIVE_SPEED_FLOOR_FRACTION = 0.50                      # reject if eff_speed < 20 kph

# EPSG:2965 (Indiana State Plane East) uses US survey feet, not meters.
US_SURVEY_FT_TO_M = 0.3048006096012192

# Corridor constraints (distances in meters)
MIN_LENGTH_KM = 3.0  # APM minimum: 3 km (~6 min ride); shorter is walkable
MAX_LENGTH_KM = 25.0

# Road-path bearing reversal threshold
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

# Station-first search parameters (used by _node_touches_major_road / _has_major_road_edge)
STATION_ROAD_CLASSES = frozenset({
    "primary", "secondary", "tertiary",
    "primary_link", "secondary_link", "tertiary_link",
})

# ---------------------------------------------------------------------------
# Geographic barriers (approximate geometries in EPSG:4326)
# ---------------------------------------------------------------------------
_WABASH_RIVER_COORDS_4326 = [
    (-86.8920, 40.4530),  # Sagamore Pkwy bridge midpoint
    (-86.8970, 40.4250),  # SR-26 / Old US-231 bridge midpoint
    (-86.8980, 40.4190),  # Columbia St / South St bridge midpoint
    (-86.9000, 40.4050),  # south of Brown St (river bends west)
    (-86.9020, 40.3940),  # Old US-231 south bridge midpoint
]
_I65_COORDS_4326 = [
    (-86.8450, 40.5100),
    (-86.8480, 40.4600),
    (-86.8500, 40.4100),
    (-86.8520, 40.3600),
    (-86.8540, 40.3400),
]
_RAILROAD_COORDS_4326 = [
    (-86.8950, 40.4500),
    (-86.8920, 40.4350),
    (-86.8900, 40.4200),
    (-86.8880, 40.4000),
]

# Fixed cost per barrier crossing (USD, added to capital cost).
BARRIER_RIVER_COST_USD = 35_000_000    # $35M per river crossing (Wabash-scale)
BARRIER_HIGHWAY_COST_USD = 20_000_000  # $20M per highway crossing (I-65)
BARRIER_RAILROAD_COST_USD = 5_000_000  # $5M per railroad crossing (CSX/NS)

# ---------------------------------------------------------------------------
# BRIDGE ZONE FREE-MOVEMENT AREAS
# ---------------------------------------------------------------------------
BRIDGE_ZONE_RADIUS_M = 800.0
GUIDEWAY_COST_PER_M = 2.0  # penalise off-road shortcuts during search; bridge
                           # zones exist to let corridors *reach* bridges, not to
                           # prefer cutting through blocks over following roads

# Cache projected coordinates by OSM node ID
_NODE_PROJ_CACHE: Dict[int, Tuple[float, float]] = {}


def _node_to_proj(nid: int, G) -> Optional[Tuple[float, float]]:
    """Get projected (x, y) for a graph node, with caching."""
    result = _NODE_PROJ_CACHE.get(nid)
    if result is not None:
        return result
    nd = G.nodes.get(int(nid))
    if nd and "x" in nd:
        xp, yp = _to_proj.transform(nd["x"], nd["y"])
        _NODE_PROJ_CACHE[nid] = (xp, yp)
        return (xp, yp)
    return None


# ============================================================================
# BARRIER CROSSING DETECTION
# ============================================================================

def _build_barrier_lines_proj():
    """Convert barrier coordinates from EPSG:4326 to projected LineStrings.

    Returns dict of {barrier_name: LineString_proj}.  Cached on first call.
    """
    if hasattr(_build_barrier_lines_proj, "_cache"):
        return _build_barrier_lines_proj._cache

    barriers = {}
    for name, coords in [
        ("river", _WABASH_RIVER_COORDS_4326),
        ("highway", _I65_COORDS_4326),
        ("railroad", _RAILROAD_COORDS_4326),
    ]:
        pts_proj = [_to_proj.transform(lon, lat) for lon, lat in coords]
        barriers[name] = LineString(pts_proj)

    _build_barrier_lines_proj._cache = barriers
    return barriers


def count_barrier_crossings(line_proj: LineString) -> dict:
    """Count how many times *line_proj* crosses each geographic barrier.

    Returns ``{barrier_name: n_crossings}``.
    """
    barriers = _build_barrier_lines_proj()
    crossings = {}
    for name, barrier_line in barriers.items():
        try:
            intersection = line_proj.intersection(barrier_line)
            if intersection.is_empty:
                crossings[name] = 0
            elif intersection.geom_type == "Point":
                crossings[name] = 1
            elif intersection.geom_type == "MultiPoint":
                crossings[name] = len(intersection.geoms)
            else:
                # LineString overlap or GeometryCollection
                crossings[name] = 1
        except Exception:
            crossings[name] = 0
    return crossings


def barrier_crossing_cost_usd(line_proj: LineString) -> float:
    """Compute total capital cost premium from barrier crossings.

    Returns additional cost in USD (e.g. one river crossing -> $80M).
    """
    crossings = count_barrier_crossings(line_proj)
    cost = 0.0
    cost += crossings.get("river", 0) * BARRIER_RIVER_COST_USD
    cost += crossings.get("highway", 0) * BARRIER_HIGHWAY_COST_USD
    cost += crossings.get("railroad", 0) * BARRIER_RAILROAD_COST_USD
    return cost


def barrier_cost_multiplier(line_proj: LineString) -> float:
    """Legacy wrapper: compute crossing cost as multiplier for short corridors.

    Kept for backward compatibility with code that expects a multiplier.
    Now returns 1.0 — crossing costs are applied additively via
    barrier_crossing_cost_usd().
    """
    return 1.0


# ---------------------------------------------------------------------------
# BRIDGE ZONE FREE-MOVEMENT AREAS
# ---------------------------------------------------------------------------

def _build_river_line():
    """Build a Shapely LineString of the Wabash centerline (EPSG:4326)."""
    from shapely.geometry import LineString as _LS
    return _LS(_WABASH_RIVER_COORDS_4326)


def _node_side_of_river(node_data, river_line):
    """Return 'N' if node is north of river centerline, 'S' if south."""
    from shapely.geometry import Point as _Pt
    pt = _Pt(node_data["x"], node_data["y"])
    nearest_river_pt = river_line.interpolate(river_line.project(pt))
    return "N" if node_data["y"] > nearest_river_pt.y else "S"


def add_bridge_zone_edges(G, bridge_zone_radius_m: float = BRIDGE_ZONE_RADIUS_M):
    """Add free-movement zones around existing Wabash river bridges.

    For each bridge crossing, creates hub-and-spoke synthetic edges
    connecting all road nodes within *bridge_zone_radius_m* on each bank
    to the bridge entry node on that bank.  This lets corridors approach
    and leave the bridge from any direction without following the road grid.

    Topology per bridge::

        north road nodes --> north bridge entry -- bridge -- south bridge entry <-- south road nodes

    Adds O(N) edges per bridge (not O(N²)) — a corridor from any
    north-zone road reaches any south-zone road in exactly 3 hops.

    Bridge detection uses the river centerline (not a fixed lat band) so
    it works everywhere the Wabash curves.  It detects crossings two ways:

    1. Edges with ``bridge=yes`` OSM attribute near the river
    2. Long edges whose endpoints are on opposite sides of the river
    """
    from shapely.geometry import Point as _Pt

    river_line = _build_river_line()
    # Buffer: ~300m in degrees (generous to catch bridges near bends)
    RIVER_PROXIMITY_DEG = 0.004

    # Precompute side-of-river for every node (cheap — one shapely call each)
    node_side: dict = {}
    for nid, nd in G.nodes(data=True):
        node_side[nid] = _node_side_of_river(nd, river_line)

    # --- 1. Find crossing edges (bridge=yes near river OR endpoints on opposite sides) ---
    bridge_crossings = []  # (north_node, south_node, mid_lon, mid_lat)

    for u, v, edata in G.edges(data=True):
        u_side = node_side[u]
        v_side = node_side[v]
        if u_side == v_side:
            continue  # both on same side — not a crossing

        mid_lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        mid_lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        mid_pt = _Pt(mid_lon, mid_lat)

        # Must be near the river (exclude edges that cross the extrapolated
        # line far from the actual river)
        if river_line.distance(mid_pt) > RIVER_PROXIMITY_DEG:
            continue

        # Accept if: (a) tagged bridge=yes, or (b) edge is on a primary/
        # secondary road (real bridges, even if not tagged)
        is_bridge_tagged = edata.get("bridge") not in (None, "", "no")
        hw = str(edata.get("highway", ""))
        is_major_road = any(c in hw for c in ("primary", "secondary", "tertiary"))
        if not (is_bridge_tagged or is_major_road):
            continue

        n_node = u if u_side == "N" else v
        s_node = v if v_side == "S" else u
        bridge_crossings.append((n_node, s_node, mid_lon, mid_lat))

    # --- 2. Deduplicate by location (~100m clusters) ---
    seen_keys: set = set()
    unique_bridges = []
    for b in bridge_crossings:
        loc_key = (round(b[2], 3), round(b[3], 3))  # ~100m resolution
        if loc_key not in seen_keys:
            seen_keys.add(loc_key)
            unique_bridges.append(b)

    if not unique_bridges:
        logger.debug("  Bridge zones: 0 bridges detected near river")
        return

    # --- 3. Create hub-and-spoke synthetic edges around each bridge ---
    radius_ft = bridge_zone_radius_m / US_SURVEY_FT_TO_M
    n_edges_added = 0

    for n_bridge_node, s_bridge_node, bridge_lon, bridge_lat in unique_bridges:
        bridge_proj = np.array(_to_proj.transform(bridge_lon, bridge_lat))

        # Collect road nodes within radius on each bank
        north_zone = []
        south_zone = []
        for node_id, node_data in G.nodes(data=True):
            node_proj = np.array(
                _to_proj.transform(node_data["x"], node_data["y"])
            )
            dist_ft = np.hypot(*(node_proj - bridge_proj))
            if dist_ft > radius_ft:
                continue
            if node_side[node_id] == "N":
                north_zone.append((node_id, node_proj))
            else:
                south_zone.append((node_id, node_proj))

        # Hub node projections for distance calculation
        n_hub_proj = np.array(
            _to_proj.transform(G.nodes[n_bridge_node]["x"], G.nodes[n_bridge_node]["y"])
        )
        s_hub_proj = np.array(
            _to_proj.transform(G.nodes[s_bridge_node]["x"], G.nodes[s_bridge_node]["y"])
        )

        # North bank spokes -> north bridge entry
        for node_id, node_proj in north_zone:
            if node_id == n_bridge_node:
                continue
            dist_m = np.hypot(*(node_proj - n_hub_proj)) * US_SURVEY_FT_TO_M
            edge_cost = dist_m * GUIDEWAY_COST_PER_M
            G.add_edge(
                node_id, n_bridge_node,
                length=dist_m, highway="apm_guideway",
                apm_cost=edge_cost, synthetic=True,
            )
            G.add_edge(
                n_bridge_node, node_id,
                length=dist_m, highway="apm_guideway",
                apm_cost=edge_cost, synthetic=True,
            )
            n_edges_added += 2

        # South bank spokes -> south bridge entry
        for node_id, node_proj in south_zone:
            if node_id == s_bridge_node:
                continue
            dist_m = np.hypot(*(node_proj - s_hub_proj)) * US_SURVEY_FT_TO_M
            edge_cost = dist_m * GUIDEWAY_COST_PER_M
            G.add_edge(
                node_id, s_bridge_node,
                length=dist_m, highway="apm_guideway",
                apm_cost=edge_cost, synthetic=True,
            )
            G.add_edge(
                s_bridge_node, node_id,
                length=dist_m, highway="apm_guideway",
                apm_cost=edge_cost, synthetic=True,
            )
            n_edges_added += 2

        # Ensure the bridge crossing itself exists
        if not G.has_edge(n_bridge_node, s_bridge_node):
            bridge_dist_m = 200.0  # approximate Wabash width
            G.add_edge(
                n_bridge_node, s_bridge_node,
                length=bridge_dist_m, highway="bridge",
                apm_cost=bridge_dist_m * 1.0, synthetic=True,
            )
            G.add_edge(
                s_bridge_node, n_bridge_node,
                length=bridge_dist_m, highway="bridge",
                apm_cost=bridge_dist_m * 1.0, synthetic=True,
            )
            n_edges_added += 2

    logger.debug(f"  Bridge zones: {len(unique_bridges)} bridges, {n_edges_added} synthetic edges added")


# ============================================================================
# PATH BEARING / VALIDATION HELPERS
# ============================================================================

def _node_touches_major_road(G, nid: int) -> bool:
    """Check if a graph node has at least one adjacent edge in STATION_ROAD_CLASSES."""
    for _u, _v, edata in G.edges(nid, data=True):
        hw = edata.get("highway", "")
        if isinstance(hw, list):
            hw = hw[0] if hw else ""
        if hw in STATION_ROAD_CLASSES:
            return True
    if hasattr(G, "in_edges"):
        for _u, _v, edata in G.in_edges(nid, data=True):
            hw = edata.get("highway", "")
            if isinstance(hw, list):
                hw = hw[0] if hw else ""
            if hw in STATION_ROAD_CLASSES:
                return True
    return False


def _has_major_road_edge(G, nid: int, neighbor_radius_m: float = 50.0) -> bool:
    """Check if a node or a close neighbor touches a major road.

    OSM tagging is inconsistent at intersections: the arterial classification
    sometimes starts at the next node downstream.  Checking one-hop neighbors
    within ``neighbor_radius_m`` handles this without broadly relaxing the
    constraint.
    """
    if _node_touches_major_road(G, nid):
        return True
    # Check neighbors reachable via short edges
    for _u, nbr, edata in G.edges(nid, data=True):
        if edata.get("length", 999) < neighbor_radius_m:
            if _node_touches_major_road(G, nbr):
                return True
    if hasattr(G, "in_edges"):
        for nbr, _v, edata in G.in_edges(nid, data=True):
            if edata.get("length", 999) < neighbor_radius_m:
                if _node_touches_major_road(G, nbr):
                    return True
    return False


def _path_bearing_at(G, path_nodes, from_end: bool, sample_m: float = 150.0) -> float:
    """Compute the bearing at one end of a road-network path.

    Uses edge geometries (when available) for accuracy — matches
    route_through_stations() which also extracts edge geometries.

    Parameters
    ----------
    G : networkx graph
    path_nodes : list of OSM node IDs forming the path
    from_end : if False, compute departure bearing (start of path);
               if True, compute arrival bearing (end of path).
    sample_m : accumulate this many meters of edge length to average
               out short jittery edges at intersections.

    Returns bearing in degrees (0 = North, clockwise).
    """
    if len(path_nodes) < 2:
        return 0.0

    # Collect dense coordinate list from edge geometries (same as route_through_stations)
    dense_coords = []  # (lon, lat) list
    edge_iter = range(len(path_nodes) - 1)
    if from_end:
        edge_iter = reversed(edge_iter)

    acc_m = 0.0
    for idx in edge_iter:
        u, v = path_nodes[idx], path_nodes[idx + 1]
        edata = G.get_edge_data(u, v)
        if edata:
            if isinstance(edata, dict) and 0 in edata:
                edata = edata[0]
            edge_len = edata.get("length", 0)
            geom = edata.get("geometry")
            if geom is not None:
                edge_coords = list(geom.coords)
                if from_end:
                    edge_coords = list(reversed(edge_coords))
            else:
                ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
                vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
                edge_coords = [(ux, uy), (vx, vy)]
                if from_end:
                    edge_coords = list(reversed(edge_coords))
        else:
            ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
            vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
            edge_coords = [(ux, uy), (vx, vy)]
            if from_end:
                edge_coords = list(reversed(edge_coords))
            edge_len = 0

        if not dense_coords:
            dense_coords.extend(edge_coords)
        else:
            dense_coords.extend(edge_coords[1:])

        acc_m += edge_len
        if acc_m >= sample_m:
            break

    if len(dense_coords) < 2:
        return 0.0

    x0, y0 = dense_coords[0]
    x1, y1 = dense_coords[-1]

    # Correct for longitude compression at this latitude
    lat_mid = (y0 + y1) / 2
    dx = (x1 - x0) * math.cos(math.radians(lat_mid))
    dy = y1 - y0

    if from_end:
        # Coordinates were collected backward (endpoint first);
        # flip so bearing = direction of travel arriving at endpoint.
        dx, dy = -dx, -dy

    return math.degrees(math.atan2(dx, dy)) % 360


def _road_graph_distance(li_a: int, li_b: int, station_data: dict) -> float:
    """Cached road-graph shortest-path distance between two candidate stations.

    Returns distance in meters, or inf if unreachable.  Symmetric caching
    with (min, max) key for O(1) repeated lookups.

    Side effect: caches departure/arrival bearings in
    station_data["path_bearing_cache"][(li_a, li_b)] as
    (departure_bearing, arrival_bearing) in degrees (0=N, CW).
    """
    import networkx as nx

    cache = station_data["road_dist_cache"]
    key = (min(li_a, li_b), max(li_a, li_b))

    # Always ensure directional bearings are cached for this ordered pair,
    # even if the symmetric distance was already computed for the reverse.
    bearing_cache = station_data.setdefault("path_bearing_cache", {})
    bearing_key = (li_a, li_b)
    need_bearings = bearing_key not in bearing_cache

    if key in cache and not need_bearings:
        return cache[key]

    G = station_data.get("graph")
    if G is None:
        cache[key] = float("inf")
        return float("inf")

    node_ids = station_data["node_ids"]
    dist_from_cache = cache.get(key)

    try:
        path = nx.shortest_path(
            G, int(node_ids[li_a]), int(node_ids[li_b]), weight="apm_cost"
        )

        # Compute distance only if not already cached
        if dist_from_cache is None:
            dist_m = 0.0
            for i in range(len(path) - 1):
                edata = G.get_edge_data(path[i], path[i + 1])
                if edata:
                    if isinstance(edata, dict) and 0 in edata:
                        edata = edata[0]
                    dist_m += edata.get("length", 0)
            cache[key] = dist_m
        else:
            dist_m = dist_from_cache

        # Cache directional bearings for path-hairpin detection
        if need_bearings and len(path) >= 2:
            depart = _path_bearing_at(G, path, from_end=False)
            arrive = _path_bearing_at(G, path, from_end=True)
            bearing_cache[bearing_key] = (depart, arrive)

        # Cache path nodes for barrier checking on actual road alignment
        path_cache = station_data.setdefault("path_node_cache", {})
        path_cache[bearing_key] = path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        dist_m = float("inf")
        if dist_from_cache is None:
            cache[key] = dist_m

    return dist_m


# ============================================================================
# CURVE PHYSICS
# ============================================================================

def compute_curve_speed_penalties(pts: np.ndarray) -> dict:
    """Physics-based curve speed and cost penalties at station vertices.

    Uses lateral acceleration comfort limit (ASCE 21.2-2008, 0.1g for standing
    passengers) and the maximum inscribed curve radius from station geometry
    to compute actual speed reductions and time penalties.

    For each interior vertex, the maximum curve radius is:
        R = T / tan(θ/2)
    where T = min(d_before, d_after) / 2 (available tangent length) and θ is
    the deflection angle.  The curve speed is:
        V_curve = min(V_line, sqrt(a_lateral × R))

    Parameters
    ----------
    pts : (N, 2) array of station coordinates in projected CRS (meters)

    Returns
    -------
    dict with physics-based curve diagnostics
    """
    n = len(pts)
    V_line = APM_LINE_SPEED_MS
    a_lat = APM_LATERAL_ACCEL_LIMIT_MS2
    a_decel = APM_SERVICE_DECEL_MS2
    a_accel = APM_SERVICE_ACCEL_MS2

    # ASCE 21: max lateral jerk 0.3 m/s³ for passenger comfort.
    # Each curve needs entry + exit clothoid transition spirals.
    _JERK_LIMIT_MS3 = 0.30

    total_delay_s = 0.0
    total_curve_extra_cost = 0.0
    min_radius = float("inf")
    has_infeasible = False
    n_restricted = 0
    angles = []
    radii = []
    speeds_kph = []
    delays_s = []

    for i in range(n - 2):
        v1x = pts[i + 1, 0] - pts[i, 0]
        v1y = pts[i + 1, 1] - pts[i, 1]
        v2x = pts[i + 2, 0] - pts[i + 1, 0]
        v2y = pts[i + 2, 1] - pts[i + 1, 1]
        dot = v1x * v2x + v1y * v2y
        m1 = np.hypot(v1x, v1y)
        m2 = np.hypot(v2x, v2y)
        if m1 < 1e-6 or m2 < 1e-6:
            angles.append(0.0)
            radii.append(float("inf"))
            speeds_kph.append(APM_SPEED_KPH)
            delays_s.append(0.0)
            continue

        cos_a = np.clip(dot / (m1 * m2), -1.0, 1.0)
        theta = np.arccos(cos_a)  # radians (0 = straight, π = reversal)
        angle_deg = np.degrees(theta)
        angles.append(angle_deg)

        if theta < np.radians(5):
            # Essentially straight — no curve
            radii.append(float("inf"))
            speeds_kph.append(APM_SPEED_KPH)
            delays_s.append(0.0)
            continue

        # Available tangent: half the shorter adjacent segment (meters)
        d_before_m = m1
        d_after_m = m2
        T = min(d_before_m, d_after_m) / 2.0

        # Maximum inscribed curve radius
        half_theta = theta / 2.0
        R = T / np.tan(half_theta) if half_theta > 1e-6 else float("inf")

        min_radius = min(min_radius, R)
        radii.append(R)

        if R < APM_MIN_CURVE_RADIUS_M:
            has_infeasible = True
            speeds_kph.append(0.0)
            delays_s.append(float("inf"))
            continue

        # Curve speed from lateral acceleration limit
        V_curve = min(V_line, (a_lat * R) ** 0.5)
        speeds_kph.append(V_curve * 3.6)

        if V_curve >= V_line:
            # Full speed through curve — no penalty
            delays_s.append(0.0)
            continue

        n_restricted += 1

        # Trapezoidal speed profile: decel -> curve traverse -> accel
        d_decel = (V_line ** 2 - V_curve ** 2) / (2.0 * a_decel)
        t_decel = (V_line - V_curve) / a_decel
        s_arc = R * theta  # arc length (meters)
        t_arc = s_arc / V_curve
        d_accel = (V_line ** 2 - V_curve ** 2) / (2.0 * a_accel)
        t_accel = (V_line - V_curve) / a_accel

        # Transition spiral: jerk-limited entry/exit clothoid.
        # Time to ramp lateral acceleration from 0 to a_curve.
        a_curve = V_curve ** 2 / R
        t_spiral = a_curve / _JERK_LIMIT_MS3  # seconds per transition
        spiral_delay = 2.0 * t_spiral           # entry + exit

        # Time that same distance would take at line speed
        spiral_length = V_curve * t_spiral * 2.0  # distance consumed by spirals
        total_affected_m = d_decel + s_arc + d_accel + spiral_length
        t_straight = total_affected_m / V_line

        delay = (t_decel + t_arc + t_accel + spiral_delay) - t_straight
        delay = max(delay, 0.0)
        delays_s.append(delay)
        total_delay_s += delay

        # Construction cost premium: proportional to how far below full-speed radius.
        # Applied to guideway cost only (curves don't affect station/vehicle costs).
        premium_frac = max(0.0, 1.0 - R / APM_FULL_SPEED_RADIUS_M)
        curve_extra_cost = s_arc * (CAPITAL_COST_GUIDEWAY_PER_KM / 1000.0) * CURVE_CONSTRUCTION_PREMIUM * premium_frac
        total_curve_extra_cost += curve_extra_cost

    # Cost multiplier on guideway component only
    total_length_m = 0.0
    for i in range(n - 1):
        total_length_m += np.hypot(
            pts[i + 1, 0] - pts[i, 0], pts[i + 1, 1] - pts[i, 1]
        )
    total_length_km = total_length_m / 1000.0
    if total_length_km > 0:
        straight_cost = total_length_km * CAPITAL_COST_GUIDEWAY_PER_KM
        curve_cost_mult = (straight_cost + total_curve_extra_cost) / straight_cost
    else:
        curve_cost_mult = 1.0

    return {
        "total_curve_delay_s": total_delay_s,
        "curve_cost_mult": curve_cost_mult,
        "min_curve_radius_m": min_radius if min_radius < float("inf") else 9999.0,
        "has_infeasible_curve": has_infeasible,
        "n_speed_restricted_curves": n_restricted,
        "turn_angles": angles,
        "curve_radii": radii,
        "curve_speeds_kph": speeds_kph,
        "curve_delays_s": delays_s,
    }


# ============================================================================
# ROAD NETWORK LOADING / ROUTING
# ============================================================================

def route_through_stations(
    station_locals: List[int],
    station_data: dict,
) -> Optional[LineString]:
    """Generate road-graph alignment for a final selected corridor.

    Routes between consecutive station pairs on the **real** road graph
    (synthetic bridge-zone edges excluded), concatenates edge geometries,
    applies light Chaikin smoothing.

    Returns a LineString in projected CRS, or None if routing fails.
    """
    import networkx as nx

    G_full = station_data["graph"]
    node_ids = station_data["node_ids"]
    coords_proj = station_data["coords_proj"]

    # Build a roads-only view: exclude synthetic bridge-zone edges so the
    # final alignment follows real streets instead of cutting through blocks.
    # Keep the actual bridge crossing edges (highway="bridge") so the route
    # can still cross the river on real bridges.
    def _real_road_edge(u, v, key):
        ed = G_full.edges[u, v, key]
        if not ed.get("synthetic", False):
            return True  # real OSM edge — always keep
        return ed.get("highway") == "bridge"  # keep bridge crossings only

    G = nx.subgraph_view(G_full, filter_edge=_real_road_edge)

    all_coords_4326 = []
    total_length_m = 0.0

    for i in range(len(station_locals) - 1):
        a_nid = int(node_ids[station_locals[i]])
        b_nid = int(node_ids[station_locals[i + 1]])

        try:
            path_nodes = nx.shortest_path(G, a_nid, b_nid, weight="apm_cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Real-road routing failed — try full graph (including synthetic
            # bridge-zone edges) so the alignment follows plausible infrastructure
            # paths instead of cutting through buildings.
            try:
                path_nodes = nx.shortest_path(G_full, a_nid, b_nid, weight="apm_cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Ultimate fallback: Euclidean straight line.
                # WARNING: This cuts through buildings and is only used
                # for visualization when no road path exists between
                # stations.  Corridors with Euclidean segments should
                # be flagged for manual review.
                import warnings
                warnings.warn(
                    f"Euclidean fallback for segment {i}->{i+1} "
                    f"(nodes {a_nid}->{b_nid}): no road path found in "
                    f"either real-road or full graph. Visualization "
                    f"may cut through buildings.",
                    stacklevel=2,
                )
                a_lon, a_lat = _to_4326.transform(
                    coords_proj[station_locals[i], 0],
                    coords_proj[station_locals[i], 1],
                )
                b_lon, b_lat = _to_4326.transform(
                    coords_proj[station_locals[i + 1], 0],
                    coords_proj[station_locals[i + 1], 1],
                )
                if all_coords_4326:
                    all_coords_4326.append((b_lon, b_lat))
                else:
                    all_coords_4326.extend([(a_lon, a_lat), (b_lon, b_lat)])
                total_length_m += np.hypot(
                    coords_proj[station_locals[i + 1], 0] - coords_proj[station_locals[i], 0],
                    coords_proj[station_locals[i + 1], 1] - coords_proj[station_locals[i], 1],
                )
                continue

        # Extract edge geometries
        seg_coords = []
        seg_length = 0.0
        for j in range(len(path_nodes) - 1):
            u, v = path_nodes[j], path_nodes[j + 1]
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                if isinstance(edge_data, dict) and 0 in edge_data:
                    edge_data = edge_data[0]
                geom = edge_data.get("geometry")
                edge_len = edge_data.get("length", 0)
                seg_length += edge_len
                if geom is not None:
                    edge_coords = list(geom.coords)
                    if seg_coords:
                        seg_coords.extend(edge_coords[1:])
                    else:
                        seg_coords.extend(edge_coords)
                    continue
            # Fallback: node coordinates
            if not seg_coords:
                seg_coords.append((G.nodes[u]["x"], G.nodes[u]["y"]))
            seg_coords.append((G.nodes[v]["x"], G.nodes[v]["y"]))

        if all_coords_4326:
            all_coords_4326.extend(seg_coords[1:])  # skip duplicate junction
        else:
            all_coords_4326.extend(seg_coords)
        total_length_m += seg_length

    if len(all_coords_4326) < 2:
        return None

    # Convert to projected CRS
    coords_proj = np.array([_to_proj.transform(x, y) for x, y in all_coords_4326])

    # Chaikin smoothing (4 passes) — rounds sharp intersection corners into
    # arcs approximating the engineered curve a real guideway would use.
    # 2 passes produce subtle rounding; 4 passes produce visible arcs that
    # survive downstream Douglas-Peucker simplification at 10m tolerance.
    smoothed = coords_proj
    for _ in range(4):
        smoothed = _chaikin_cut(smoothed)

    # Angle-aware Douglas-Peucker: remove minor road-grid wiggles while
    # preserving intentional turns (L-shapes, curves).
    smoothed = _simplify_with_angle_constraint(
        smoothed,
        distance_tolerance_m=5.0 / US_SURVEY_FT_TO_M,  # 5m in project CRS units
        max_angle_deg=PATH_BEARING_REVERSAL_DEG,  # stay consistent
    )

    return LineString(smoothed.tolist())


def check_routed_path_quality(
    alignment: LineString,
    station_coords_proj: np.ndarray,
    sample_m: float = 200.0 / 0.3048006096012192,  # 200m in US survey feet
) -> List[str]:
    """Post-routing quality check on the actual road-network alignment.

    Returns a list of warning strings (empty = all ok).
    Checks:
      1. Bearing reversals at each station (arrival vs departure on routed path)
      2. Overall path direction consistency
    """
    warnings = []
    if alignment is None or len(station_coords_proj) < 3:
        return warnings

    path_coords = np.array(alignment.coords)
    n_pts = len(path_coords)
    if n_pts < 4:
        return warnings

    # Find the index in path_coords closest to each station
    station_indices = []
    for sc in station_coords_proj:
        dists = np.hypot(path_coords[:, 0] - sc[0], path_coords[:, 1] - sc[1])
        station_indices.append(int(np.argmin(dists)))

    # For each interior station, check bearing reversal on the path
    for si in range(1, len(station_indices) - 1):
        idx = station_indices[si]

        # Walk backwards from station to find arrival bearing (~sample_m back)
        acc_m = 0.0
        arrive_idx = max(idx - 1, 0)
        for j in range(idx, 0, -1):
            seg_len = np.hypot(
                path_coords[j, 0] - path_coords[j - 1, 0],
                path_coords[j, 1] - path_coords[j - 1, 1],
            )
            acc_m += seg_len
            arrive_idx = j - 1
            if acc_m >= sample_m:
                break

        # Walk forward from station to find departure bearing (~sample_m ahead)
        acc_m = 0.0
        depart_idx = min(idx + 1, n_pts - 1)
        for j in range(idx, n_pts - 1):
            seg_len = np.hypot(
                path_coords[j + 1, 0] - path_coords[j, 0],
                path_coords[j + 1, 1] - path_coords[j, 1],
            )
            acc_m += seg_len
            depart_idx = j + 1
            if acc_m >= sample_m:
                break

        # Arrival bearing = direction of travel at arrival (toward station)
        adx = path_coords[idx, 0] - path_coords[arrive_idx, 0]
        ady = path_coords[idx, 1] - path_coords[arrive_idx, 1]
        arrive_brg = math.degrees(math.atan2(adx, ady)) % 360

        # Departure bearing = direction of travel leaving station
        ddx = path_coords[depart_idx, 0] - path_coords[idx, 0]
        ddy = path_coords[depart_idx, 1] - path_coords[idx, 1]
        depart_brg = math.degrees(math.atan2(ddx, ddy)) % 360

        diff = abs(arrive_brg - depart_brg)
        if diff > 180:
            diff = 360 - diff
        if diff > PATH_BEARING_REVERSAL_DEG:
            warnings.append(
                f"station {si}: routed-path reversal {diff:.0f}° "
                f"(arrive={arrive_brg:.0f}°, depart={depart_brg:.0f}°)"
            )

    return warnings


def _augment_graph_weights(G):
    """Add 'apm_cost' edge weight = length x road_class_multiplier.

    Makes shortest-path routing prefer primary/secondary roads (multiplier 1.0)
    over residential (2.5x) or service roads (3.0x).  Motorways are effectively
    blocked (5.0x).  The actual ``length`` attribute is preserved for distance
    calculations — ``apm_cost`` is used only for pathfinding direction.
    """
    for _, _, _, data in G.edges(data=True, keys=True):
        hw = data.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0] if hw else "unclassified"
        mult = ROAD_CLASS_COST_SURFACE.get(hw, 2.0)
        data["apm_cost"] = data.get("length", 0) * mult


def load_road_network(bounds_4326: tuple, use_network: bool = True) -> Optional[object]:
    """Download OSM road network for the study area.

    Uses the known corridor study area bounds (Lafayette/West Lafayette)
    rather than full county parcel extent to avoid massive downloads.

    Returns an osmnx graph or None if unavailable.
    """
    if not use_network:
        logger.debug("  Network routing disabled, using straight lines")
        return None

    # Use corridor-relevant bounds, not full county
    # osmnx 2.x bbox format: (left, bottom, right, top) = (west, south, east, north)
    STUDY_BBOX = (-86.94, 40.33, -86.84, 40.53)

    cache_path = PROC_DIR / "osm_road_network.graphml"

    try:
        import networkx as nx

        if cache_path.exists():
            logger.debug("  Loading cached road network...")
            # Use nx.read_graphml directly (osmnx import + load_graphml hang on this system)
            G = nx.read_graphml(cache_path)
            # Coerce node x/y to float (graphml stores strings)
            for _nid, _nd in G.nodes(data=True):
                _nd["x"] = float(_nd["x"])
                _nd["y"] = float(_nd["y"])
            # Coerce edge attributes from strings (graphml stores everything as text)
            from shapely import wkt as _wkt
            for _u, _v, _k, _ed in G.edges(data=True, keys=True):
                if "length" in _ed:
                    _ed["length"] = float(_ed["length"])
                if "geometry" in _ed and isinstance(_ed["geometry"], str):
                    try:
                        _ed["geometry"] = _wkt.loads(_ed["geometry"])
                    except Exception:
                        del _ed["geometry"]
            # Relabel nodes from strings to integers (graphml stores
            # all IDs as strings; osmnx uses int node IDs throughout)
            G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})
            logger.debug(f"  Road network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            _augment_graph_weights(G)
            add_bridge_zone_edges(G)
            return G

        logger.debug("  Downloading OSM road network (Lafayette/West Lafayette area)...")
        import osmnx as ox
        G = ox.graph_from_bbox(
            bbox=STUDY_BBOX,
            network_type="drive",
            simplify=True,
        )

        # Cache for future runs
        ox.save_graphml(G, cache_path)
        logger.debug(f"  Road network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        logger.debug(f"  Cached to: {cache_path.name}")
        _augment_graph_weights(G)
        add_bridge_zone_edges(G)
        return G
    except Exception as e:
        logger.debug(f"  Could not load road network: {e}")
        logger.debug("  Falling back to straight-line routing")
        return None


def route_on_network(
    G,
    start_xy_proj: tuple,
    end_xy_proj: tuple,
) -> Optional[LineString]:
    """Route between two points on the road network.

    Returns a LineString in projected CRS, or None if no route found.
    """
    if G is None:
        return None

    import osmnx as ox
    import networkx as nx

    try:
        # Convert to 4326 for osmnx (graph is in 4326)
        sx, sy = _to_4326.transform(start_xy_proj[0], start_xy_proj[1])
        ex, ey = _to_4326.transform(end_xy_proj[0], end_xy_proj[1])

        # nearest_nodes expects (X=lon, Y=lat)
        orig_node = ox.nearest_nodes(G, X=sx, Y=sy)
        dest_node = ox.nearest_nodes(G, X=ex, Y=ey)

        if orig_node == dest_node:
            return None

        route = nx.shortest_path(G, orig_node, dest_node, weight="apm_cost")

        if len(route) < 2:
            return None

        # Trim leading/trailing residential segments (up to 150m per end).
        # NSGA-II mutation can place terminals on minor streets; the router
        # follows the shortest path from those nodes to the arterial network,
        # creating visual hooks.  Trimming cleans these up regardless of
        # how the terminal was generated.
        _TRIM_CLASSES = {"residential", "service", "living_street", "unclassified"}
        for _end in ("start", "end"):
            _trim_dist = 0.0
            while len(route) > 2:
                _u, _v = (route[0], route[1]) if _end == "start" else (route[-2], route[-1])
                _ed = G.get_edge_data(_u, _v)
                if _ed and isinstance(_ed, dict) and 0 in _ed:
                    _ed = _ed[0]
                _hw = (_ed or {}).get("highway", "")
                if isinstance(_hw, list):
                    _hw = _hw[0]
                if _hw not in _TRIM_CLASSES:
                    break
                _seg_len = float((_ed or {}).get("length", 0))
                if _trim_dist + _seg_len > 150:
                    break
                _trim_dist += _seg_len
                if _end == "start":
                    route.pop(0)
                else:
                    route.pop()

        if len(route) < 2:
            return None

        # Extract route geometry from edge geometries where available
        coords_4326 = []
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            # Get edge data (may have 'geometry' attribute)
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                # Multi-edge graph: get first edge
                if isinstance(edge_data, dict) and 0 in edge_data:
                    edge_data = edge_data[0]
                geom = edge_data.get("geometry")
                if geom is not None:
                    edge_coords = list(geom.coords)
                    # Avoid duplicating the junction point
                    if coords_4326:
                        coords_4326.extend(edge_coords[1:])
                    else:
                        coords_4326.extend(edge_coords)
                    continue

            # Fallback: use node coordinates
            if not coords_4326:
                coords_4326.append((G.nodes[u]["x"], G.nodes[u]["y"]))
            coords_4326.append((G.nodes[v]["x"], G.nodes[v]["y"]))

        if len(coords_4326) < 2:
            return None

        # Convert to projected CRS
        coords_proj = [_to_proj.transform(x, y) for x, y in coords_4326]
        line = LineString(coords_proj)

        length_km = line.length * US_SURVEY_FT_TO_M / 1000
        if length_km < MIN_LENGTH_KM or length_km > MAX_LENGTH_KM:
            return None

        return line

    except Exception:
        return None


def _chaikin_cut(coords: np.ndarray) -> np.ndarray:
    """One pass of Chaikin corner cutting (quadratic B-spline subdivision).

    For each segment, replaces the two endpoints with points at 25% and 75%
    along the segment. Endpoints are preserved. This rounds off sharp corners
    while keeping the path close to the original — maximum deviation from the
    original path is bounded to 25% of the longest segment length per pass.
    """
    if len(coords) < 3:
        return coords
    # Q[i] = 0.75*P[i] + 0.25*P[i+1], R[i] = 0.25*P[i] + 0.75*P[i+1]
    q = 0.75 * coords[:-1] + 0.25 * coords[1:]
    r = 0.25 * coords[:-1] + 0.75 * coords[1:]
    # Interleave: Q0, R0, Q1, R1, ...
    new = np.empty((2 * len(q), 2), dtype=coords.dtype)
    new[0::2] = q
    new[1::2] = r
    # Preserve original endpoints
    new[0] = coords[0]
    new[-1] = coords[-1]
    return new


def _simplify_with_angle_constraint(
    coords: np.ndarray,
    distance_tolerance_m: float = 5.0,
    max_angle_deg: float = 55.0,
) -> np.ndarray:
    """Douglas-Peucker simplification that preserves turn vertices.

    First marks vertices with turn angle > *max_angle_deg* as forced-keep,
    then runs Shapely Douglas-Peucker on each segment between forced vertices.
    This removes minor road-grid wiggles while preserving intentional turns.
    """
    from shapely.geometry import LineString as _LS

    n = len(coords)
    if n < 3:
        return coords

    # Identify vertices with sharp turns that must be preserved
    keep = set([0, n - 1])
    for i in range(1, n - 1):
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 < 1e-6 or m2 < 1e-6:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1, 1)
        angle = np.degrees(np.arccos(cos_a))
        if angle > max_angle_deg:
            keep.add(i)

    # Run D-P on segments between forced-keep vertices
    keep_sorted = sorted(keep)
    result = [coords[keep_sorted[0]]]
    for j in range(len(keep_sorted) - 1):
        seg = coords[keep_sorted[j]:keep_sorted[j + 1] + 1]
        if len(seg) > 2:
            simplified = np.array(
                _LS(seg.tolist()).simplify(distance_tolerance_m).coords
            )
            result.extend(simplified[1:])
        else:
            result.append(seg[-1])
    return np.array(result)

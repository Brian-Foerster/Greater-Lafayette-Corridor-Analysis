"""Synthetic feeder bus route generator — arterial-spine algorithm.

Generates street-following feeder bus routes connecting APM stations to
population concentrations in the 0.8-5 km feeder ring.  Routes follow
arterial roads (trunk/primary/secondary/tertiary) so they look realistic
and avoid winding through residential neighborhoods.

Algorithm
---------
1. Extract arterial road segments that pass through the feeder ring
2. Score each arterial corridor by population within 400m walk shed
3. Select top corridors with angular diversity (no two within 30 deg)
4. Route from nearest APM station along each arterial spine
5. Cap at MAX_SYNTHETIC_ROUTES (4) per corridor
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

ox = None  # lazy-loaded to avoid 30s import at module level


def _get_ox():
    global ox
    if ox is None:
        try:
            import osmnx as _ox
            ox = _ox
        except ImportError:
            pass
    return ox

from src.bus_network import BusRoute, DEFAULT_BUS_SPEED_KPH
from src.spatial_constants import PROJECT_CRS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEEDER_INNER_M = 1200.0       # walk-zone boundary
FEEDER_OUTER_M = 5000.0       # feeder ring outer boundary
POP_PER_PARCEL_CAP = 2000     # clip mega-parcels
STOP_SPACING_M = 400.0        # target stop spacing along route
BUS_CIRCUITY = 1.30           # fallback circuity if routing fails
DEFAULT_FEEDER_HEADWAY = 20.0 # minutes
FEEDER_SERVICE_SPAN_HOURS = 16.0
ROAD_NETWORK_CACHE = "data/processed/lafayette_road_network.pkl"

# Arterial spine algorithm parameters
MAX_SYNTHETIC_ROUTES = 6
MAX_CROSS_CORRIDOR_ROUTES = 2  # separate budget for perpendicular arterials
MIN_SPINE_POP = 1000          # minimum walk-shed population to justify a route
ANGULAR_DIVERSITY_RAD = 0.52  # ~30 degrees minimum between spines
PARALLEL_REJECT_RAD = 0.52   # ~30 degrees — reject spines parallel to corridor
ARTERIAL_TYPES = {"trunk", "primary", "secondary", "tertiary"}
SPINE_WALK_SHED_M = 400.0     # walk distance to bus stop
MIN_SPINE_LENGTH_M = 800.0    # ignore very short arterial fragments


# ---------------------------------------------------------------------------
# Road network helpers
# ---------------------------------------------------------------------------

HIGHWAY_PENALTY = {
    "motorway": 5.0,
    "motorway_link": 3.0,
    "trunk": 1.0,
    "trunk_link": 1.0,
    "primary": 1.0,
    "primary_link": 1.0,
    "secondary": 1.0,
    "secondary_link": 1.0,
    "tertiary": 1.2,
    "tertiary_link": 1.2,
    "unclassified": 2.0,
    "residential": 3.0,
}


def _add_transit_weights(G: nx.MultiDiGraph) -> None:
    """Add transit-friendly edge weights that penalise residential streets."""
    for u, v, k, d in G.edges(keys=True, data=True):
        hw = d.get("highway", "residential")
        if isinstance(hw, list):
            hw = hw[0]
        penalty = HIGHWAY_PENALTY.get(hw, 2.0)
        d["transit_weight"] = d.get("length", 100.0) * penalty


def build_graph_node_index(G: nx.MultiDiGraph) -> Tuple[np.ndarray, np.ndarray]:
    """Build a KDTree-compatible index of graph node coordinates.

    Returns (node_ids, node_xy) where node_xy is (N, 2) array of [x, y]
    (lon, lat) for use with cKDTree.  Stored alongside the graph so that
    ``route_on_network`` can skip the slow ``ox.nearest_nodes()`` call.
    """
    node_ids = np.array(list(G.nodes()), dtype=np.int64)
    node_xy = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_ids],
                        dtype=np.float64)
    return node_ids, node_xy


def load_road_network(cache_path: str = ROAD_NETWORK_CACHE) -> Optional[nx.MultiDiGraph]:
    """Load cached osmnx road network graph and add transit routing weights.

    Also builds a fast spatial index (``_node_ids``, ``_node_xy``,
    ``_node_tree``) attached to the graph object for O(log n) nearest-node
    lookups.
    """
    p = Path(cache_path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        G = pickle.load(f)
    _add_transit_weights(G)
    # Attach fast spatial index for nearest-node queries
    from scipy.spatial import cKDTree
    node_ids, node_xy = build_graph_node_index(G)
    G.graph["_node_ids"] = node_ids
    G.graph["_node_xy"] = node_xy
    G.graph["_node_tree"] = cKDTree(node_xy)
    return G


def _fast_nearest_node(G: nx.MultiDiGraph, lon: float, lat: float) -> int:
    """Find nearest graph node using pre-built KDTree (O(log n) vs O(n))."""
    tree = G.graph.get("_node_tree")
    if tree is not None:
        _, idx = tree.query([lon, lat])
        return int(G.graph["_node_ids"][idx])
    # Fallback to osmnx if index not built
    _ox = _get_ox()
    if _ox is None:
        raise RuntimeError("osmnx required for _fast_nearest_node fallback")
    return _ox.nearest_nodes(G, lon, lat)


def route_on_network(
    G: nx.MultiDiGraph,
    origin_lonlat: Tuple[float, float],
    dest_lonlat: Tuple[float, float],
) -> Tuple[List[Tuple[float, float]], float]:
    """Route between two lon/lat points using transit-weighted shortest path.

    Returns (coords_lonlat, length_metres).
    """
    if G is None or _get_ox() is None:
        return [], 0.0

    try:
        orig_node = _fast_nearest_node(G, origin_lonlat[0], origin_lonlat[1])
        dest_node = _fast_nearest_node(G, dest_lonlat[0], dest_lonlat[1])

        if orig_node == dest_node:
            return [], 0.0

        path = nx.shortest_path(G, orig_node, dest_node, weight="transit_weight")

        coords: List[Tuple[float, float]] = []
        total_length = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge_data = G[u][v][0]
            total_length += edge_data.get("length", 0.0)
            if "geometry" in edge_data:
                edge_coords = list(edge_data["geometry"].coords)
                u_data = G.nodes[u]
                u_xy = (u_data["x"], u_data["y"])
                if _dist2(edge_coords[0], u_xy) > _dist2(edge_coords[-1], u_xy):
                    edge_coords = edge_coords[::-1]
                if coords:
                    edge_coords = edge_coords[1:]
                coords.extend(edge_coords)
            else:
                pt = (G.nodes[v]["x"], G.nodes[v]["y"])
                if not coords:
                    coords.append((G.nodes[u]["x"], G.nodes[u]["y"]))
                coords.append(pt)

        return coords, total_length

    except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
        return [], 0.0


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# ---------------------------------------------------------------------------
# Arterial spine extraction
# ---------------------------------------------------------------------------

@dataclass
class ArterialSpine:
    """A candidate feeder route corridor along an arterial road."""
    nodes: List[int]               # osmnx node IDs along the spine
    endpoint_node: int             # far end of spine (route destination)
    angle_rad: float               # bearing from corridor center to spine midpoint
    travel_bearing_rad: float      # direction the route actually travels (first→last node)
    length_m: float                # total spine length
    pop_within_walkshed: float     # population within SPINE_WALK_SHED_M
    highway_class: str             # dominant road class


def _get_edge_highway(G: nx.MultiDiGraph, u: int, v: int) -> str:
    """Get highway class for an edge, handling list values."""
    if not G.has_edge(u, v):
        return "residential"
    d = G[u][v][0]
    hw = d.get("highway", "residential")
    if isinstance(hw, list):
        hw = hw[0]
    return hw


def _extract_arterial_spines(
    G: nx.MultiDiGraph,
    station_xy: np.ndarray,
    parcel_xy: np.ndarray,
    parcel_pop: np.ndarray,
    inner_m: float = FEEDER_INNER_M,
    outer_m: float = FEEDER_OUTER_M,
) -> List[ArterialSpine]:
    """Extract arterial corridors passing through the feeder ring.

    1. Finds all arterial edges whose midpoint falls in the feeder ring
    2. Builds an arterial-only subgraph from these edges
    3. Extracts connected components as potential spines
    4. For each component, finds the longest path (linear corridor)
    5. Scores by population within walk shed
    """
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    # Project graph nodes from WGS-84 to local CRS (EPSG:2965, US survey feet)
    # Keep in feet to match station_xy and parcel_xy coordinate system
    from src.spatial_constants import US_SURVEY_FT_TO_M
    tx = Transformer.from_crs("EPSG:4326", PROJECT_CRS, always_xy=True)
    node_xy: Dict[int, Tuple[float, float]] = {}
    for n, d in G.nodes(data=True):
        px, py = tx.transform(d["x"], d["y"])
        node_xy[n] = (px, py)  # feet, same CRS as station_xy/parcel_xy

    # Convert metre thresholds to feet for distance comparisons
    inner_ft = inner_m / US_SURVEY_FT_TO_M
    outer_ft = outer_m / US_SURVEY_FT_TO_M

    # Corridor center (centroid of all stations)
    center_x, center_y = station_xy.mean(axis=0)

    # Find arterial edges in the feeder ring
    ring_edges = []
    for u, v, k, d in G.edges(keys=True, data=True):
        hw = d.get("highway", "residential")
        if isinstance(hw, list):
            hw = hw[0]
        if hw not in ARTERIAL_TYPES:
            continue

        # Check if edge midpoint is in the feeder ring
        if u not in node_xy or v not in node_xy:
            continue
        ux, uy = node_xy[u]
        vx, vy = node_xy[v]
        mx, my = (ux + vx) / 2, (uy + vy) / 2

        # Distance from corridor center to edge midpoint
        dist = math.sqrt((mx - center_x) ** 2 + (my - center_y) ** 2)
        if inner_ft * 0.5 <= dist <= outer_ft:  # slightly relaxed inner to catch spines starting near stations
            ring_edges.append((u, v, hw, d.get("length", 0.0)))

    if not ring_edges:
        return []

    # Build arterial subgraph (undirected for component finding)
    H = nx.Graph()
    for u, v, hw, length in ring_edges:
        if H.has_edge(u, v):
            continue
        H.add_edge(u, v, highway=hw, length=length / US_SURVEY_FT_TO_M)

    # --- Split arterial network at junctions to get linear corridors ---
    # Junctions = nodes with degree >= 3 in the arterial subgraph
    junction_nodes = {n for n in H.nodes() if H.degree(n) >= 3}

    # Remove junctions to break the graph into chains
    H_cut = H.copy()
    H_cut.remove_nodes_from(junction_nodes)
    chain_components = list(nx.connected_components(H_cut))

    # For each chain, re-attach the junction nodes at the ends to get full paths
    chains: List[List[int]] = []
    for comp in chain_components:
        if len(comp) < 2:
            continue
        sub = H.subgraph(comp)
        # Chain endpoints = degree-1 nodes in the subgraph
        endpoints = [n for n in comp if sub.degree(n) == 1]
        if len(endpoints) < 2:
            # Cycle — pick any two nodes far apart
            start = next(iter(comp))
            lengths = nx.single_source_dijkstra_path_length(sub, start, weight="length")
            endpoints = [start, max(lengths, key=lengths.get)]
        try:
            path = nx.shortest_path(sub, endpoints[0], endpoints[-1], weight="length")
        except nx.NetworkXNoPath:
            continue
        # Extend path to include adjacent junctions for better connectivity
        for jn in junction_nodes:
            if H.has_edge(jn, path[0]):
                path = [jn] + path
            if H.has_edge(jn, path[-1]):
                path = path + [jn]
        chains.append(path)

    # Merge collinear chains (same bearing within 15°, sharing a junction endpoint)
    def _chain_angle(chain):
        """Bearing from first to last node relative to corridor center."""
        p0 = node_xy[chain[0]]
        p1 = node_xy[chain[-1]]
        mx = (p0[0] + p1[0]) / 2
        my = (p0[1] + p1[1]) / 2
        return math.atan2(my - center_y, mx - center_x)

    def _chain_length(chain):
        total = 0.0
        for i in range(len(chain) - 1):
            if H.has_edge(chain[i], chain[i + 1]):
                total += H[chain[i]][chain[i + 1]].get("length", 0.0)
            else:
                # Euclidean fallback
                a, b = node_xy[chain[i]], node_xy[chain[i + 1]]
                total += math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        return total

    merged = True
    while merged:
        merged = False
        new_chains = []
        used = set()
        for i, c1_chain in enumerate(chains):
            if i in used:
                continue
            best_j = None
            for j, c2_chain in enumerate(chains):
                if j <= i or j in used:
                    continue
                # Check if they share an endpoint (junction)
                shared = None
                if c1_chain[-1] == c2_chain[0]:
                    shared = "tail_head"
                elif c1_chain[-1] == c2_chain[-1]:
                    shared = "tail_tail"
                elif c1_chain[0] == c2_chain[0]:
                    shared = "head_head"
                elif c1_chain[0] == c2_chain[-1]:
                    shared = "head_tail"
                if shared is None:
                    continue
                # Check angle similarity (within 15°)
                a1 = _chain_angle(c1_chain)
                a2 = _chain_angle(c2_chain)
                diff = abs(a1 - a2)
                if diff > math.pi:
                    diff = 2 * math.pi - diff
                if diff < 0.26:  # ~15 degrees
                    best_j = (j, shared)
                    break
            if best_j is not None:
                j, how = best_j
                c2_chain = chains[j]
                if how == "tail_head":
                    new_chains.append(c1_chain + c2_chain[1:])
                elif how == "tail_tail":
                    new_chains.append(c1_chain + c2_chain[-2::-1])
                elif how == "head_head":
                    new_chains.append(c1_chain[::-1] + c2_chain[1:])
                elif how == "head_tail":
                    new_chains.append(c2_chain + c1_chain[1:])
                used.add(i)
                used.add(j)
                merged = True
            else:
                new_chains.append(c1_chain)
                used.add(i)
        # Add remaining unused
        for i, c in enumerate(chains):
            if i not in used:
                new_chains.append(c)
        chains = new_chains

    # Build parcel KD-tree for population scoring
    parcel_tree = cKDTree(parcel_xy) if len(parcel_xy) > 0 else None

    spines = []
    for path in chains:
        # Compute spine length
        spine_length = _chain_length(path)
        if spine_length < MIN_SPINE_LENGTH_M / US_SURVEY_FT_TO_M:
            continue

        # Compute bearing from corridor center to spine midpoint
        mid_idx = len(path) // 2
        mid_node = path[mid_idx]
        if mid_node not in node_xy:
            continue
        mid_x, mid_y = node_xy[mid_node]
        angle = math.atan2(mid_y - center_y, mid_x - center_x)

        # Score by population within walk shed
        pop_score = _score_pop_walkshed(path, node_xy, parcel_tree, parcel_pop)

        # Dominant highway class
        hw_counts: Dict[str, int] = {}
        for i in range(len(path) - 1):
            if H.has_edge(path[i], path[i + 1]):
                edge_hw = H[path[i]][path[i + 1]].get("highway", "tertiary")
            else:
                edge_hw = "tertiary"
            hw_counts[edge_hw] = hw_counts.get(edge_hw, 0) + 1
        dominant_hw = max(hw_counts, key=hw_counts.get) if hw_counts else "tertiary"

        # Determine which end is farther from corridor center
        far_dists = [
            math.sqrt((node_xy[path[0]][0] - center_x) ** 2 + (node_xy[path[0]][1] - center_y) ** 2),
            math.sqrt((node_xy[path[-1]][0] - center_x) ** 2 + (node_xy[path[-1]][1] - center_y) ** 2),
        ]
        endpoint = path[-1] if far_dists[1] >= far_dists[0] else path[0]

        # Travel bearing: direction from first to last node (route travel direction)
        p0 = node_xy[path[0]]
        p1 = node_xy[path[-1]]
        travel_bearing = math.atan2(p1[1] - p0[1], p1[0] - p0[0])

        spines.append(ArterialSpine(
            nodes=path,
            endpoint_node=endpoint,
            angle_rad=angle,
            travel_bearing_rad=travel_bearing,
            length_m=spine_length * US_SURVEY_FT_TO_M,
            pop_within_walkshed=pop_score,
            highway_class=dominant_hw,
        ))

    return spines


def _score_pop_walkshed(
    path: List[int],
    node_xy: Dict[int, Tuple[float, float]],
    parcel_tree,
    parcel_pop: np.ndarray,
) -> float:
    """Score a spine path by population within walk shed."""
    if parcel_tree is None:
        return 0.0
    spine_pts = np.array([node_xy[n] for n in path if n in node_xy])
    if len(spine_pts) < 2:
        return 0.0
    # Sample points along spine at ~200m intervals (converted to feet)
    from src.spatial_constants import US_SURVEY_FT_TO_M
    sample_interval_ft = 200.0 / US_SURVEY_FT_TO_M
    walkshed_ft = SPINE_WALK_SHED_M / US_SURVEY_FT_TO_M
    sample_pts = []
    for i in range(len(spine_pts) - 1):
        seg_len = math.sqrt(
            (spine_pts[i + 1, 0] - spine_pts[i, 0]) ** 2
            + (spine_pts[i + 1, 1] - spine_pts[i, 1]) ** 2
        )
        n_samples = max(1, int(seg_len / sample_interval_ft))
        for s in range(n_samples):
            t = s / n_samples
            px = spine_pts[i, 0] + t * (spine_pts[i + 1, 0] - spine_pts[i, 0])
            py = spine_pts[i, 1] + t * (spine_pts[i + 1, 1] - spine_pts[i, 1])
            sample_pts.append([px, py])
    sample_pts.append(spine_pts[-1].tolist())
    sample_arr = np.array(sample_pts)

    nearby_indices = parcel_tree.query_ball_point(sample_arr, r=walkshed_ft)
    unique_parcels: set = set()
    for idx_list in nearby_indices:
        unique_parcels.update(idx_list)
    return float(parcel_pop[list(unique_parcels)].sum()) if unique_parcels else 0.0


# ---------------------------------------------------------------------------
# Spine selection with angular diversity
# ---------------------------------------------------------------------------

def _select_spines(
    spines: List[ArterialSpine],
    max_routes: int = MAX_SYNTHETIC_ROUTES,
    min_pop: float = MIN_SPINE_POP,
    angular_min: float = ANGULAR_DIVERSITY_RAD,
    existing_angles: Optional[List[float]] = None,
    corridor_bearing_rad: Optional[float] = None,
    station_xy: Optional[np.ndarray] = None,
    corridor_buffer_m: float = 400.0,
    road_graph: Optional[nx.MultiDiGraph] = None,
) -> List[ArterialSpine]:
    """Select top spines with angular diversity constraint.

    Greedy: pick highest-pop spine, then skip any within angular_min of
    already-selected spines. Also avoids angles already covered by
    existing CityBus feeder routes (if provided).

    If *station_xy* is given, uses spatial overlap test: rejects spines
    whose endpoint falls within *corridor_buffer_m* of the corridor line.
    Falls back to bearing-only rejection via PARALLEL_REJECT_RAD.
    """
    # Filter by minimum population
    candidates = [s for s in spines if s.pop_within_walkshed >= min_pop]

    # Reject spines spatially redundant with the corridor
    if corridor_bearing_rad is not None:
        _use_spatial = (
            station_xy is not None
            and len(station_xy) >= 2
            and road_graph is not None
        )
        filtered = []
        _transformer = None
        if _use_spatial:
            from pyproj import Transformer
            from src.spatial_constants import PROJECT_CRS
            _transformer = Transformer.from_crs(
                "EPSG:4326", PROJECT_CRS, always_xy=True
            )
        for s in candidates:
            if _use_spatial and hasattr(s, 'endpoint_node') and s.endpoint_node in road_graph.nodes:
                _ep = road_graph.nodes[s.endpoint_node]
                _epx, _epy = _transformer.transform(_ep["x"], _ep["y"])
                if _point_to_polyline_dist(_epx, _epy, station_xy) < corridor_buffer_m:
                    continue  # spine endpoint inside corridor buffer
                filtered.append(s)
            else:
                corr_diff = abs(s.travel_bearing_rad - corridor_bearing_rad)
                if corr_diff > math.pi:
                    corr_diff = 2 * math.pi - corr_diff
                if corr_diff > math.pi / 2:
                    corr_diff = math.pi - corr_diff
                if corr_diff >= PARALLEL_REJECT_RAD:
                    filtered.append(s)
        candidates = filtered

    # Sort by population served (descending)
    candidates.sort(key=lambda s: -s.pop_within_walkshed)

    selected: List[ArterialSpine] = []
    used_angles: List[float] = list(existing_angles or [])

    for spine in candidates:
        if len(selected) >= max_routes:
            break

        # Check angular diversity against all used angles
        too_close = False
        for used_a in used_angles:
            # Angular distance (handle wrap-around)
            diff = abs(spine.angle_rad - used_a)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            if diff < angular_min:
                too_close = True
                break

        if not too_close:
            selected.append(spine)
            used_angles.append(spine.angle_rad)

    return selected


# ---------------------------------------------------------------------------
# Station-demand sector analysis (Tier 2: demand-responsive routing)
# ---------------------------------------------------------------------------

DEMAND_DECAY_BETA = 0.0005   # same as FEEDER_DECAY_BETA in ridership model
N_SECTORS = 8                # 45° sectors
MIN_FEEDER_HEADWAY = 10.0    # highest-demand feeder (minutes)
MAX_FEEDER_HEADWAY = 30.0    # lowest-demand feeder (minutes)


def _compute_station_demand_sectors(
    station_xy: np.ndarray,
    parcel_xy: np.ndarray,
    parcel_pop: np.ndarray,
    od_origins_idx: Optional[np.ndarray] = None,
    od_dests_idx: Optional[np.ndarray] = None,
    od_flows: Optional[np.ndarray] = None,
    inner_m: float = FEEDER_INNER_M,
    outer_m: float = FEEDER_OUTER_M,
) -> np.ndarray:
    """Compute demand by station and angular sector.

    Returns
    -------
    demand_matrix : (N_stations, 8) array of demand scores.
    """
    from scipy.spatial import cKDTree

    n_stations = len(station_xy)
    demand = np.zeros((n_stations, N_SECTORS), dtype=np.float64)

    if len(parcel_xy) == 0:
        return demand

    parcel_tree = cKDTree(parcel_xy)
    sector_width = 2 * math.pi / N_SECTORS

    for si in range(n_stations):
        sx, sy = station_xy[si]

        # Find parcels in feeder ring
        inner_idx = set(parcel_tree.query_ball_point([sx, sy], r=outer_m))
        too_close = set(parcel_tree.query_ball_point([sx, sy], r=inner_m))
        ring_idx = sorted(inner_idx - too_close)

        if not ring_idx:
            continue

        ring_arr = np.array(ring_idx)
        dx = parcel_xy[ring_arr, 0] - sx
        dy = parcel_xy[ring_arr, 1] - sy
        dists = np.sqrt(dx * dx + dy * dy)
        bearings = np.arctan2(dy, dx)  # -pi to pi
        sectors = ((bearings / sector_width + 0.5) % N_SECTORS).astype(int)
        weights = parcel_pop[ring_arr] * np.exp(-DEMAND_DECAY_BETA * dists)

        for k in range(N_SECTORS):
            mask = sectors == k
            demand[si, k] = float(weights[mask].sum())

    # Add OD-weighted demand if available
    if (od_origins_idx is not None and od_dests_idx is not None
            and od_flows is not None and len(od_flows) > 0):
        # Filter out unmapped parcels (index == -1 from OD cache building)
        _od_valid = (od_origins_idx >= 0) & (od_dests_idx >= 0) & (od_flows >= 0.01)
        od_origins_idx = od_origins_idx[_od_valid]
        od_dests_idx = od_dests_idx[_od_valid]
        od_flows = od_flows[_od_valid]

        # For each station, find which OD origins are in its feeder ring
        # and whose destinations are near any APM station (corridor-bound trips)
        station_tree = cKDTree(station_xy)
        dest_dists, _ = station_tree.query(parcel_xy[od_dests_idx])
        # Destination is "near corridor" if within walk catchment
        dest_near = dest_dists <= inner_m

        for si in range(n_stations):
            sx, sy = station_xy[si]
            orig_xy = parcel_xy[od_origins_idx]
            orig_dx = orig_xy[:, 0] - sx
            orig_dy = orig_xy[:, 1] - sy
            orig_dists = np.sqrt(orig_dx * orig_dx + orig_dy * orig_dy)
            in_ring = (orig_dists > inner_m) & (orig_dists <= outer_m)
            valid = in_ring & dest_near

            if not valid.any():
                continue

            bearings = np.arctan2(orig_dy[valid], orig_dx[valid])
            sectors = ((bearings / sector_width + 0.5) % N_SECTORS).astype(int)
            flows = od_flows[valid] * 2.0  # round-trip

            for k in range(N_SECTORS):
                mask = sectors == k
                demand[si, k] += float(flows[mask].sum())

    return demand


def _point_to_polyline_dist(px: float, py: float, polyline: np.ndarray) -> float:
    """Minimum distance from point (px, py) to a polyline (N, 2) array."""
    min_d_sq = float("inf")
    for i in range(len(polyline) - 1):
        ax, ay = float(polyline[i, 0]), float(polyline[i, 1])
        bx, by = float(polyline[i + 1, 0]), float(polyline[i + 1, 1])
        abx, aby = bx - ax, by - ay
        ab_sq = abx * abx + aby * aby
        if ab_sq < 1e-12:
            d_sq = (px - ax) ** 2 + (py - ay) ** 2
        else:
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_sq))
            proj_x = ax + t * abx
            proj_y = ay + t * aby
            d_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
        if d_sq < min_d_sq:
            min_d_sq = d_sq
    return math.sqrt(min_d_sq) if min_d_sq < float("inf") else float("inf")


def _select_feeder_targets(
    demand_matrix: np.ndarray,
    corridor_bearing_rad: Optional[float],
    station_bearings: Optional[np.ndarray] = None,
    max_routes: int = MAX_SYNTHETIC_ROUTES,
    existing_angles: Optional[List[float]] = None,
    station_xy: Optional[np.ndarray] = None,
    corridor_buffer_m: float = 400.0,
) -> List[Tuple[int, int, float]]:
    """Rank and select station-sector pairs for feeder routes.

    Parameters
    ----------
    demand_matrix : (N_stations, 8) demand per station per sector.
    corridor_bearing_rad : overall corridor bearing (used if station_bearings
        is None).
    station_bearings : (N_stations,) per-station local corridor bearing.
        Used for L-shaped/curved corridors.  Falls back to corridor_bearing_rad.
    max_routes : maximum number of feeder routes to generate.
    station_xy : (N, 2) projected station coordinates. When provided, enables
        spatial overlap test: a sector is rejected as parallel only if its
        demand centroid falls within *corridor_buffer_m* of the corridor line.
    corridor_buffer_m : buffer distance for spatial overlap test (meters).
    existing_angles : bearings already served by existing feeders.

    Returns
    -------
    targets : list of (station_idx, sector_idx, demand_score), sorted by
        demand descending, subject to diversity constraints.
    """
    n_stations, n_sectors = demand_matrix.shape
    sector_width = 2 * math.pi / n_sectors

    # Flatten to ranked list
    triples = []
    for si in range(n_stations):
        for ki in range(n_sectors):
            d = demand_matrix[si, ki]
            if d <= 0:
                continue
            # Sector center bearing must match the assignment formula in
            # _compute_station_demand_sectors: ki = floor((bearing/sw + 0.5) % N).
            # Sector ki covers bearings in [(ki-0.5)*sw, (ki+0.5)*sw), so its
            # center is ki * sw.  Normalize to [-pi, pi] for atan2 consistency.
            sector_center_bearing = ki * sector_width
            if sector_center_bearing > math.pi:
                sector_center_bearing -= 2 * math.pi
            triples.append((si, ki, d, sector_center_bearing))

    # Sort by demand descending
    triples.sort(key=lambda x: -x[2])

    # Determine per-station corridor bearing for parallel rejection
    def _get_corridor_bearing(si: int) -> Optional[float]:
        if station_bearings is not None and si < len(station_bearings):
            return float(station_bearings[si])
        return corridor_bearing_rad

    selected: List[Tuple[int, int, float]] = []
    used_station_sectors: Dict[int, List[int]] = {}  # station -> list of selected sectors
    used_bearings: List[float] = list(existing_angles or [])

    for si, ki, d, bearing in triples:
        if len(selected) >= max_routes:
            break

        # Reject sectors whose demand centroid falls within the corridor
        # buffer.  Bearing-only rejection wrongly blocks sectors that point
        # along the corridor but serve population beyond its endpoints.
        cb = _get_corridor_bearing(si)
        if cb is not None:
            if station_xy is not None and len(station_xy) >= 2:
                # Spatial overlap test: project demand centroid outward from
                # the station and check if it lands inside the corridor buffer.
                avg_feeder_dist_m = 3000.0  # midpoint of feeder ring
                sx, sy = float(station_xy[si, 0]), float(station_xy[si, 1])
                cx = sx + avg_feeder_dist_m * math.cos(bearing)
                cy = sy + avg_feeder_dist_m * math.sin(bearing)
                if _point_to_polyline_dist(cx, cy, station_xy) < corridor_buffer_m:
                    continue  # centroid inside corridor buffer = redundant
            else:
                # Fallback: bearing-only rejection when station_xy unavailable
                corr_diff = abs(bearing - cb)
                if corr_diff > math.pi:
                    corr_diff = 2 * math.pi - corr_diff
                if corr_diff > math.pi / 2:
                    corr_diff = math.pi - corr_diff
                if corr_diff < PARALLEL_REJECT_RAD:
                    continue

        # No two from same station within 45° of each other
        prev_sectors = used_station_sectors.get(si, [])
        too_close_station = False
        for pk in prev_sectors:
            sector_diff = abs(ki - pk)
            if sector_diff > n_sectors // 2:
                sector_diff = n_sectors - sector_diff
            if sector_diff < 1:  # within same or adjacent sector
                too_close_station = True
                break
        if too_close_station:
            continue

        # Global angular diversity: no two within 30°
        too_close_global = False
        for ub in used_bearings:
            diff = abs(bearing - ub)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            if diff < ANGULAR_DIVERSITY_RAD:
                too_close_global = True
                break
        if too_close_global:
            continue

        selected.append((si, ki, d))
        used_station_sectors.setdefault(si, []).append(ki)
        used_bearings.append(bearing)

    return selected


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

@dataclass
class FeederRouteResult:
    """Result of generating feeder routes for one corridor."""
    routes: List[BusRoute]
    geojson_features: List[Dict]
    stops_by_route: Dict[str, np.ndarray]  # route_id -> (N,2) projected coords
    summary: Dict


def generate_feeder_routes(
    station_xy: np.ndarray,
    station_lonlat: np.ndarray,
    parcel_xy: np.ndarray,
    parcel_pop: np.ndarray,
    corridor_id: str = "C1",
    road_graph: Optional[nx.MultiDiGraph] = None,
    headway_min: float = DEFAULT_FEEDER_HEADWAY,
    budget_per_corridor: Optional[float] = None,
    existing_feeder_stops_xy: Optional[np.ndarray] = None,
    existing_feeder_angles: Optional[List[float]] = None,
    gap_fill_only: bool = False,
    od_origins_idx: Optional[np.ndarray] = None,
    od_dests_idx: Optional[np.ndarray] = None,
    od_flows: Optional[np.ndarray] = None,
) -> FeederRouteResult:
    """Generate synthetic feeder bus routes using demand-responsive algorithm.

    Station-first routing: compute demand by station × angular sector using
    parcel population + optional LODES OD flows, then route from each
    station outward along the best arterial in the selected sector.

    Parameters
    ----------
    station_xy : (N, 2) projected station coordinates (EPSG:2965)
    station_lonlat : (N, 2) WGS-84 station coordinates (lon, lat)
    parcel_xy : (P, 2) projected parcel centroids
    parcel_pop : (P,) population per parcel
    corridor_id : corridor identifier
    road_graph : osmnx MultiDiGraph with transit_weight
    headway_min : default feeder headway (used as fallback; demand-proportional
        headways override when demand data is available)
    budget_per_corridor : annual budget for feeder routes (optional)
    existing_feeder_stops_xy : (M, 2) projected coords of existing CityBus
        feeder stops. Used to avoid generating routes in already-served sectors.
    existing_feeder_angles : list of angular bearings (degrees, from center of
        APM stations) of already-existing feeder routes (from truncated GTFS).
    gap_fill_only : if True, only generate routes in angular sectors not
        already covered by existing feeders.
    od_origins_idx : (F,) parcel indices of LODES trip origins
    od_dests_idx : (F,) parcel indices of LODES trip destinations
    od_flows : (F,) trip counts per OD pair
    """
    from pyproj import Transformer
    transformer_to_lonlat = Transformer.from_crs(
        PROJECT_CRS, "EPSG:4326", always_xy=True
    )

    # Clip mega-parcels
    parcel_pop = np.minimum(parcel_pop, POP_PER_PARCEL_CAP)

    routes: List[BusRoute] = []
    geojson_features: List[Dict] = []
    stops_by_route: Dict[str, np.ndarray] = {}

    if road_graph is None:
        return FeederRouteResult(routes, geojson_features, stops_by_route,
                                 {"corridor_id": corridor_id, "n_routes": 0})

    # --- Step 1: Determine existing coverage angles ---
    existing_angles: List[float] = []
    if existing_feeder_stops_xy is not None and len(existing_feeder_stops_xy) > 0:
        center_x, center_y = station_xy.mean(axis=0)
        for sx, sy in existing_feeder_stops_xy:
            a = math.atan2(sy - center_y, sx - center_x)
            existing_angles.append(a)
        binned = set(round(a / 0.26, 0) for a in existing_angles)
        existing_angles = [b * 0.26 for b in binned]

    if existing_feeder_angles:
        for a_deg in existing_feeder_angles:
            existing_angles.append(math.radians(a_deg))

    if gap_fill_only and len(existing_angles) >= MAX_SYNTHETIC_ROUTES * 2:
        return FeederRouteResult([], [], {},
                                 {"corridor_id": corridor_id, "n_routes": 0})

    # --- Step 1b: Compute corridor bearing ---
    if len(station_lonlat) >= 2:
        _corr_dx = station_lonlat[-1, 0] - station_lonlat[0, 0]
        _corr_dy = station_lonlat[-1, 1] - station_lonlat[0, 1]
        corridor_bearing = math.atan2(_corr_dy, _corr_dx)
    else:
        corridor_bearing = None

    # Per-station local corridor bearing (for L-shaped/curved corridors)
    station_bearings = None
    if len(station_lonlat) >= 3:
        station_bearings = np.zeros(len(station_lonlat), dtype=np.float64)
        for si in range(len(station_lonlat)):
            si_prev = max(0, si - 1)
            si_next = min(len(station_lonlat) - 1, si + 1)
            dx = station_lonlat[si_next, 0] - station_lonlat[si_prev, 0]
            dy = station_lonlat[si_next, 1] - station_lonlat[si_prev, 1]
            station_bearings[si] = math.atan2(dy, dx)

    # --- Step 2: Compute demand sectors and select targets ---
    demand_matrix = _compute_station_demand_sectors(
        station_xy, parcel_xy, parcel_pop,
        od_origins_idx=od_origins_idx,
        od_dests_idx=od_dests_idx,
        od_flows=od_flows,
    )

    max_routes = MAX_SYNTHETIC_ROUTES
    if gap_fill_only and existing_angles:
        covered_sectors = len(set(int(a / 0.52) for a in existing_angles))
        max_routes = max(MAX_SYNTHETIC_ROUTES - covered_sectors, 1)

    targets = _select_feeder_targets(
        demand_matrix,
        corridor_bearing_rad=corridor_bearing,
        station_bearings=station_bearings,
        max_routes=max_routes,
        existing_angles=existing_angles if existing_angles else None,
        station_xy=station_xy,
    )

    # --- Step 2b: Extract arterial spines (used for routing targets) ---
    all_spines = _extract_arterial_spines(
        road_graph, station_xy, parcel_xy, parcel_pop
    )
    # Index spines by angle for sector matching
    sector_width = 2 * math.pi / N_SECTORS

    # Also identify cross-corridor spines (perpendicular arterials connecting 2+ stations)
    cross_corridor_spines: List[Tuple[ArterialSpine, List[int]]] = []
    if corridor_bearing is not None and len(station_xy) >= 2:
        from scipy.spatial import cKDTree
        _station_tree_ll = cKDTree(station_lonlat)
        perp_bearing = corridor_bearing + math.pi / 2
        for spine in all_spines:
            # Check if travel bearing is within 20° of perpendicular
            tb = spine.travel_bearing_rad
            perp_diff = abs(tb - perp_bearing)
            if perp_diff > math.pi:
                perp_diff = 2 * math.pi - perp_diff
            if perp_diff > math.pi / 2:
                perp_diff = math.pi - perp_diff
            if perp_diff > 0.35:  # ~20°
                continue
            # Check how many stations are within 200m of any spine node
            nearby_stations = set()
            for nid in spine.nodes:
                if nid not in road_graph.nodes:
                    continue
                nd = road_graph.nodes[nid]
                nll = np.array([nd["x"], nd["y"]])
                dists, idxs = _station_tree_ll.query(nll, k=min(3, len(station_lonlat)))
                if np.isscalar(dists):
                    dists, idxs = [dists], [idxs]
                for d, si in zip(dists, idxs):
                    if d * 111_000 < 200:  # rough degree-to-meter
                        nearby_stations.add(int(si))
            if len(nearby_stations) >= 2:
                cross_corridor_spines.append((spine, sorted(nearby_stations)))

    # Compute max demand for headway scaling
    max_demand = max((d for _, _, d in targets), default=1.0)

    # --- Step 3: Route from each station along target sector ---
    from src.spatial_constants import US_SURVEY_FT_TO_M as _FT2M
    transformer_to_proj = Transformer.from_crs(
        "EPSG:4326", PROJECT_CRS, always_xy=True
    )

    for ri, (anchor_si, sector_idx, demand_score) in enumerate(targets):
        route_id = f"{corridor_id}_F{ri + 1}"
        station_ll = (float(station_lonlat[anchor_si, 0]),
                      float(station_lonlat[anchor_si, 1]))

        # Find best arterial spine in this sector
        target_bearing = sector_idx * sector_width
        if target_bearing > math.pi:
            target_bearing -= 2 * math.pi
        best_spine = None
        best_pop = -1.0
        for spine in all_spines:
            # Check if spine midpoint bearing is in this sector (±22.5°)
            diff = abs(spine.angle_rad - target_bearing)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            if diff > sector_width:
                continue
            # Spatial parallel rejection: check if spine endpoint falls
            # within the corridor buffer (= redundant with APM).
            if corridor_bearing is not None:
                if len(station_xy) >= 2 and hasattr(spine, 'endpoint_node'):
                    _ep = road_graph.nodes[spine.endpoint_node]
                    _epx, _epy = transformer_to_proj.transform(_ep["x"], _ep["y"])
                    if _point_to_polyline_dist(_epx, _epy, station_xy) < 400.0:
                        continue
                else:
                    corr_diff = abs(spine.travel_bearing_rad - corridor_bearing)
                    if corr_diff > math.pi:
                        corr_diff = 2 * math.pi - corr_diff
                    if corr_diff > math.pi / 2:
                        corr_diff = math.pi - corr_diff
                    if corr_diff < PARALLEL_REJECT_RAD:
                        continue
            if spine.pop_within_walkshed > best_pop:
                best_spine = spine
                best_pop = spine.pop_within_walkshed

        # Route destination
        if best_spine is not None:
            endpoint_data = road_graph.nodes[best_spine.endpoint_node]
            far_ll = (endpoint_data["x"], endpoint_data["y"])
            pop_served = best_spine.pop_within_walkshed
            hw_class = best_spine.highway_class
        else:
            # Fallback: population centroid in the sector
            sx_proj, sy_proj = station_xy[anchor_si]
            parcels_in_sector = []
            for pi in range(len(parcel_xy)):
                dx = parcel_xy[pi, 0] - sx_proj
                dy = parcel_xy[pi, 1] - sy_proj
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < FEEDER_INNER_M or dist > FEEDER_OUTER_M:
                    continue
                b = math.atan2(dy, dx)
                b_diff = abs(b - target_bearing)
                if b_diff > math.pi:
                    b_diff = 2 * math.pi - b_diff
                if b_diff < sector_width:
                    parcels_in_sector.append((pi, dist))
            if parcels_in_sector:
                # Population-weighted centroid
                idxs = [p[0] for p in parcels_in_sector]
                pops = parcel_pop[idxs]
                total_pop = pops.sum()
                if total_pop > 0:
                    cx_f = float(np.average(parcel_xy[idxs, 0], weights=pops))
                    cy_f = float(np.average(parcel_xy[idxs, 1], weights=pops))
                    far_lon, far_lat = transformer_to_lonlat.transform(
                        cx_f / _FT2M, cy_f / _FT2M
                    )
                    far_ll = (far_lon, far_lat)
                else:
                    far_ll = (station_ll[0] + 0.02 * math.cos(target_bearing),
                              station_ll[1] + 0.02 * math.sin(target_bearing))
            else:
                far_ll = (station_ll[0] + 0.02 * math.cos(target_bearing),
                          station_ll[1] + 0.02 * math.sin(target_bearing))
            pop_served = demand_score
            hw_class = "tertiary"

        route_coords, route_length_m = route_on_network(
            road_graph, station_ll, far_ll
        )

        # Ensure route starts at the station point
        if route_coords and len(route_coords) >= 2:
            _gap_m = _haversine_m(route_coords[0], station_ll)
            if _gap_m < 50:
                route_coords[0] = station_ll
            elif _gap_m < 200:
                route_coords = [station_ll] + route_coords
            else:
                corner = (station_ll[0], route_coords[0][1])
                route_coords = [station_ll, corner] + route_coords

        if not route_coords or route_length_m < 200:
            route_coords = [station_ll, far_ll]
            route_length_m = _haversine_m(station_ll, far_ll) * BUS_CIRCUITY

        route_length_km = route_length_m / 1000.0

        stop_coords_ll = _interpolate_stops(route_coords, STOP_SPACING_M)
        # Keep stop_xy in EPSG:2965 feet (same CRS as station_xy) so
        # classify_routes_for_corridor proximity checks work correctly.
        stop_xy = np.array([
            transformer_to_proj.transform(lon, lat)
            for lon, lat in stop_coords_ll
        ])

        avg_speed_mps = DEFAULT_BUS_SPEED_KPH * 1000.0 / 3600.0
        one_way_min = (route_length_m / avg_speed_mps) / 60.0
        n_stops = len(stop_coords_ll)
        cycle_time = 2 * one_way_min + n_stops * 0.5 + 5.0

        # Direction label from sector bearing
        angle_deg = math.degrees(target_bearing) % 360
        dir_labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        dir_idx = int((angle_deg + 22.5) / 45) % 8
        direction = dir_labels[dir_idx]

        # Demand-proportional headway
        demand_rank = demand_score / max_demand if max_demand > 0 else 0.5
        route_headway = MAX_FEEDER_HEADWAY - demand_rank * (MAX_FEEDER_HEADWAY - MIN_FEEDER_HEADWAY)

        route_name = f"{corridor_id} Feeder {direction}"

        bus_route = BusRoute(
            route_id=route_id,
            name=route_name,
            length_km=route_length_km,
            baseline_headway_min=route_headway,
            current_headway_min=route_headway,
            cycle_time_min=cycle_time,
            n_stops=n_stops,
            trips_per_day=FEEDER_SERVICE_SPAN_HOURS * 60.0 / route_headway,
            observed_daily_riders=0.0,
            classification="feeder",
        )
        routes.append(bus_route)
        stops_by_route[route_id] = stop_xy

        geojson_features.append({
            "type": "Feature",
            "properties": {
                "route_id": route_id,
                "route_name": route_name,
                "corridor_id": corridor_id,
                "direction": direction,
                "pop_served": round(pop_served),
                "demand_score": round(demand_score, 1),
                "length_km": round(route_length_km, 2),
                "n_stops": n_stops,
                "headway_min": round(route_headway, 1),
                "highway_class": hw_class,
                "is_synthetic": True,
                "anchor_station": anchor_si,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for lon, lat in route_coords],
            },
        })

    # --- Step 3b: Cross-corridor routes ---
    # Perpendicular arterials connecting 2+ stations — common pattern
    # (e.g., east-west bus crossing north-south APM at multiple stations).
    n_cross = 0
    for spine, connected_stations in cross_corridor_spines:
        if n_cross >= MAX_CROSS_CORRIDOR_ROUTES:
            break
        # Score by demand at connected stations
        cross_demand = sum(
            demand_matrix[si].max() for si in connected_stations
            if si < demand_matrix.shape[0]
        )
        if cross_demand <= 0:
            continue

        # Route along the spine between the two most distant connected stations
        s0, s1 = connected_stations[0], connected_stations[-1]
        origin_ll = (float(station_lonlat[s0, 0]), float(station_lonlat[s0, 1]))
        dest_ll = (float(station_lonlat[s1, 0]), float(station_lonlat[s1, 1]))

        endpoint_data = road_graph.nodes[spine.endpoint_node]
        spine_far_ll = (endpoint_data["x"], endpoint_data["y"])
        route_coords, route_length_m = route_on_network(
            road_graph, origin_ll, spine_far_ll
        )

        if not route_coords or route_length_m < 200:
            continue

        # Prepend station start
        if route_coords:
            _gap_m = _haversine_m(route_coords[0], origin_ll)
            if _gap_m < 50:
                route_coords[0] = origin_ll
            elif _gap_m < 200:
                route_coords = [origin_ll] + route_coords

        route_length_km = route_length_m / 1000.0
        n_cross += 1
        route_id = f"{corridor_id}_FX{n_cross}"

        stop_coords_ll = _interpolate_stops(route_coords, STOP_SPACING_M)
        # Keep stop_xy in EPSG:2965 feet (same CRS as station_xy) so
        # classify_routes_for_corridor proximity checks work correctly.
        stop_xy = np.array([
            transformer_to_proj.transform(lon, lat)
            for lon, lat in stop_coords_ll
        ])

        avg_speed_mps = DEFAULT_BUS_SPEED_KPH * 1000.0 / 3600.0
        one_way_min = (route_length_m / avg_speed_mps) / 60.0
        n_stops = len(stop_coords_ll)
        cycle_time = 2 * one_way_min + n_stops * 0.5 + 5.0

        demand_rank = cross_demand / max_demand if max_demand > 0 else 0.5
        route_headway = MAX_FEEDER_HEADWAY - demand_rank * (MAX_FEEDER_HEADWAY - MIN_FEEDER_HEADWAY)

        bus_route = BusRoute(
            route_id=route_id,
            name=f"{corridor_id} Cross-corridor",
            length_km=route_length_km,
            baseline_headway_min=route_headway,
            current_headway_min=route_headway,
            cycle_time_min=cycle_time,
            n_stops=n_stops,
            trips_per_day=FEEDER_SERVICE_SPAN_HOURS * 60.0 / route_headway,
            observed_daily_riders=0.0,
            classification="feeder",
        )
        routes.append(bus_route)
        stops_by_route[route_id] = stop_xy
        geojson_features.append({
            "type": "Feature",
            "properties": {
                "route_id": route_id,
                "route_name": bus_route.name,
                "corridor_id": corridor_id,
                "direction": "cross",
                "pop_served": round(spine.pop_within_walkshed),
                "demand_score": round(cross_demand, 1),
                "length_km": round(route_length_km, 2),
                "n_stops": n_stops,
                "headway_min": round(route_headway, 1),
                "highway_class": spine.highway_class,
                "is_synthetic": True,
                "is_cross_corridor": True,
                "connected_stations": connected_stations,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for lon, lat in route_coords],
            },
        })

    # --- Step 4: Budget adjustment ---
    if budget_per_corridor and budget_per_corridor > 0 and routes:
        from src.bus_network import BusOperatingCostModel
        cost_model = BusOperatingCostModel(annual_budget=budget_per_corridor)
        total_cost = sum(cost_model.route_annual_cost(r) for r in routes)
        if total_cost > 0:
            scale = total_cost / budget_per_corridor
            if scale > 1.0:
                for r in routes:
                    r.baseline_headway_min = min(r.baseline_headway_min * scale, 60.0)
                    r.current_headway_min = r.baseline_headway_min
            else:
                for r in routes:
                    r.baseline_headway_min = max(r.baseline_headway_min * scale, 5.0)
                    r.current_headway_min = r.baseline_headway_min

    # --- Step 5: Validate station connectivity ---
    if routes:
        from src.bus_network import classify_routes_for_corridor
        classify_routes_for_corridor(
            routes, station_xy, stops_by_route,
        )
        valid_ids = {r.route_id for r in routes
                     if r.classification == "feeder"}
        if len(valid_ids) < len(routes):
            routes = [r for r in routes if r.route_id in valid_ids]
            geojson_features = [f for f in geojson_features
                                if f["properties"]["route_id"] in valid_ids]
            stops_by_route = {k: v for k, v in stops_by_route.items()
                              if k in valid_ids}

    _dir_labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    summary = {
        "corridor_id": corridor_id,
        "n_routes": len(routes),
        "n_spines_found": len(all_spines),
        "n_targets_selected": len(targets),
        "n_cross_corridor": n_cross,
        "total_demand_score": sum(d for _, _, d in targets),
        "directions_served": [
            _dir_labels[int((math.degrees(ki * sector_width) % 360 + 22.5) / 45) % 8]
            for _, ki, _ in targets
        ],
        "total_length_km": sum(r.length_km for r in routes),
        "avg_headway_min": (
            float(np.mean([r.baseline_headway_min for r in routes]))
            if routes else 0.0
        ),
    }

    return FeederRouteResult(
        routes=routes,
        geojson_features=geojson_features,
        stops_by_route=stops_by_route,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _interpolate_stops(
    coords: List[Tuple[float, float]],
    spacing_m: float,
) -> List[Tuple[float, float]]:
    """Place stops at regular intervals along a (lon, lat) coordinate sequence."""
    if len(coords) < 2:
        return list(coords)

    stops = [coords[0]]
    accum = 0.0

    for i in range(1, len(coords)):
        seg_len = _haversine_m(coords[i - 1], coords[i])
        accum += seg_len
        while accum >= spacing_m:
            overshoot = accum - spacing_m
            frac = 1.0 - overshoot / seg_len if seg_len > 0 else 1.0
            lon = coords[i - 1][0] + frac * (coords[i][0] - coords[i - 1][0])
            lat = coords[i - 1][1] + frac * (coords[i][1] - coords[i - 1][1])
            stops.append((lon, lat))
            accum -= spacing_m

    if _haversine_m(stops[-1], coords[-1]) > 50:
        stops.append(coords[-1])

    return stops


def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Haversine distance in metres between two (lon, lat) points."""
    R = 6_371_000.0
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))

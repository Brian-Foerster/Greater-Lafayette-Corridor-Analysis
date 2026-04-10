"""
Corridor Evolution & Selection Operators
==========================================

Extracted from ``optimized_corridor_search.py`` — pure structural refactoring,
no behavioral changes.

Contains:
- NSGA-II algorithm (dominates, fast_non_dominated_sort, crowding_distance, nsga2_select)
- Genetic operators (mutate_station_set, crossover_station_sets, deduplicate_station_sets)
- Diversity selection (_corridor_polyline, _overlap_fraction, _bidirectional_overlap,
  select_diverse_station_sets)
- Post-search refinement (refine_station_placements)
- Synergy scoring (apply_station_synergy_scores)

NOTE: Several functions reference ``validate_station_set`` and ``score_station_set``
from ``optimized_corridor_search.py``.  These are imported lazily at runtime.

Also references ``_node_to_proj`` from the main module (used inside
``_corridor_polyline`` for road-graph path reconstruction).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

import logging
logger = logging.getLogger(__name__)

# _node_to_proj is defined in the geometry module (no circular dependency).
from scripts.corridor_geometry import _node_to_proj

# validate_station_set and score_station_set live in optimized_corridor_search.py.
# They are imported lazily inside function bodies to avoid circular imports
# (optimized_corridor_search re-exports from this module).
_validate_station_set = None
_score_station_set = None


def _get_validate():
    global _validate_station_set
    if _validate_station_set is None:
        from scripts.optimized_corridor_search import validate_station_set
        _validate_station_set = validate_station_set
    return _validate_station_set


def _get_score():
    global _score_station_set
    if _score_station_set is None:
        from scripts.optimized_corridor_search import score_station_set
        _score_station_set = score_station_set
    return _score_station_set

# ---------------------------------------------------------------------------
# Constants used by extracted functions (mirrored from optimized_corridor_search)
# ---------------------------------------------------------------------------

# EPSG:2965 (Indiana State Plane East) uses US survey feet, not meters.
US_SURVEY_FT_TO_M = 0.3048006096012192

# Station-set size bounds
MIN_STATIONS_PER_CORRIDOR = 4
MAX_STATIONS_PER_CORRIDOR = 12

# Station proximity for diversity / crossover
STATION_PROXIMITY_M = 400.0

# Bidirectional overlap fraction diversity
OVERLAP_BUFFER_M = 400.0
OVERLAP_THRESHOLD = 0.60
_OVERLAP_BUFFER_FT = OVERLAP_BUFFER_M / US_SURVEY_FT_TO_M

# Network synergy defaults
NETWORK_TRANSFER_RADIUS_M = 1200.0
NETWORK_SYNERGY_WEIGHT_DEFAULT = 0.20

# Minimum ridership floor for diversity selection
MIN_RIDERSHIP_FOR_SELECTION = 2000

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # NSGA-II
    "dominates",
    "fast_non_dominated_sort",
    "crowding_distance",
    "nsga2_select",
    # Genetic operators
    "mutate_station_set",
    "crossover_station_sets",
    "deduplicate_station_sets",
    # Diversity selection
    "select_diverse_station_sets",
    "_corridor_polyline",
    "_overlap_fraction",
    "_bidirectional_overlap",
    # Refinement & synergy
    "refine_station_placements",
    "apply_station_synergy_scores",
]


# ============================================================================
# NSGA-II ALGORITHM
# ============================================================================

def dominates(a: tuple, b: tuple) -> bool:
    """Check if solution a dominates solution b (all objectives maximized)."""
    at_least_one_better = False
    for ai, bi in zip(a, b):
        if ai < bi:
            return False
        if ai > bi:
            at_least_one_better = True
    return at_least_one_better


def _objectives(score: dict) -> tuple:
    """Extract NSGA-II objective tuple from a corridor score dict.

    All objectives are maximized:
      1. Ridership (projected with mini forward simulation)
      2. Cost efficiency (riders per dollar)
      3. DCR estimate (mature-year revenue / annual cost ratio)

    Curve delay is NOT a separate objective — it's already captured by
    effective speed -> ridership via MNL, and enforced by validation
    (curves < 250m radius are rejected).
    """
    return (
        score["ridership_est"],
        score["cost_efficiency"],
        score.get("dcr_est", score.get("viability_indicator", 0.0)),
    )


def fast_non_dominated_sort(population: List[dict]) -> List[List[int]]:
    """NSGA-II fast non-dominated sorting.

    Uses 3 objectives (ridership, cost_efficiency, viability_indicator).
    """
    n = len(population)
    domination_count = [0] * n  # How many solutions dominate this one
    dominated_set = [[] for _ in range(n)]  # Solutions this one dominates
    fronts = [[]]

    for i in range(n):
        fi = _objectives(population[i]["score"])
        for j in range(n):
            if i == j:
                continue
            fj = _objectives(population[j]["score"])
            if dominates(fi, fj):
                dominated_set[i].append(j)
            elif dominates(fj, fi):
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]  # Remove empty fronts


def crowding_distance(population: List[dict], front: List[int]) -> Dict[int, float]:
    """Compute crowding distance for solutions in a front."""
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}

    distances = {idx: 0.0 for idx in front}
    n_obj = len(_objectives(population[front[0]]["score"]))

    for obj_idx in range(n_obj):
        sorted_front = sorted(
            front, key=lambda i: _objectives(population[i]["score"])[obj_idx]
        )
        distances[sorted_front[0]] = float("inf")
        distances[sorted_front[-1]] = float("inf")

        lo = _objectives(population[sorted_front[0]]["score"])[obj_idx]
        hi = _objectives(population[sorted_front[-1]]["score"])[obj_idx]
        obj_range = hi - lo
        if obj_range < 1e-10:
            continue

        for k in range(1, len(sorted_front) - 1):
            val_next = _objectives(population[sorted_front[k + 1]]["score"])[obj_idx]
            val_prev = _objectives(population[sorted_front[k - 1]]["score"])[obj_idx]
            distances[sorted_front[k]] += (val_next - val_prev) / obj_range

    return distances


def nsga2_select(
    population: List[dict],
    n_select: int,
    precomputed_fronts: Optional[List[List[int]]] = None,
    station_data: Optional[dict] = None,
) -> List[dict]:
    """NSGA-II selection: prefer lower front, then higher crowding distance.

    TIF viability is a soft constraint: within each front, TIF-viable
    corridors are preferred.  Non-viable corridors are kept only when
    there aren't enough viable ones to fill the selection.

    Geographic diversity: within each front, individuals with >70%
    bidirectional overlap with an already-accepted individual are
    deprioritised (pushed to end of front).  This prevents the
    population from converging to geometric variants of one corridor.
    """
    fronts = precomputed_fronts if precomputed_fronts is not None else fast_non_dominated_sort(population)
    selected = []

    # Pre-compute polylines + KDTrees for geometric diversity (once per call)
    _poly_cache: Dict[int, np.ndarray] = {}
    _tree_cache: Dict[int, Optional["cKDTree"]] = {}
    coords_proj = station_data["coords_proj"] if station_data else None
    use_geo_diversity = station_data is not None and coords_proj is not None

    def _get_poly(idx: int) -> np.ndarray:
        if idx not in _poly_cache:
            _poly_cache[idx] = _corridor_polyline(
                population[idx]["stations"], coords_proj,
                station_data=station_data,
            )
        return _poly_cache[idx]

    def _get_tree(idx: int) -> Optional["cKDTree"]:
        if idx not in _tree_cache:
            p = _get_poly(idx)
            _tree_cache[idx] = cKDTree(p) if len(p) > 0 else None
        return _tree_cache[idx]

    _GEO_OVERLAP_THRESHOLD = 0.70  # stricter than post-search dedup (0.60)

    for front in fronts:
        # Sort front: TIF-viable first, then by crowding distance
        cd = crowding_distance(population, front)
        sorted_front = sorted(
            front,
            key=lambda i: (
                1 if population[i]["score"].get("tif_viable", True) else 0,
                cd[i],
            ),
            reverse=True,
        )

        if use_geo_diversity and len(sorted_front) > 2:
            # Greedy geographic dedup within the front: iterate in priority
            # order and defer individuals that overlap >70% with any already-
            # accepted individual from this front OR from prior fronts.
            accepted = []
            deferred = []
            for idx in sorted_front:
                poly_idx = _get_poly(idx)
                if len(poly_idx) == 0:
                    accepted.append(idx)
                    continue
                tree_idx = _get_tree(idx)
                overlaps = False
                # Check against already-selected from prior fronts + this front
                for sel_idx in selected + accepted:
                    poly_sel = _get_poly(sel_idx)
                    if len(poly_sel) == 0:
                        continue
                    tree_sel = _get_tree(sel_idx)
                    ov = _bidirectional_overlap(
                        poly_idx, poly_sel,
                        tree_a=tree_idx, tree_b=tree_sel,
                    )
                    if ov > _GEO_OVERLAP_THRESHOLD:
                        overlaps = True
                        break
                if overlaps:
                    deferred.append(idx)
                else:
                    accepted.append(idx)
            # Deferred go to end — still available if needed to fill quota
            sorted_front = accepted + deferred

        if len(selected) + len(sorted_front) <= n_select:
            selected.extend(sorted_front)
        else:
            remaining = n_select - len(selected)
            selected.extend(sorted_front[:remaining])
            break

    return [population[i] for i in selected]


# ============================================================================
# GENETIC OPERATORS
# ============================================================================

def _check_monotonic(stations: list, station_data: dict) -> bool:
    """Quick monotonicity pre-check without full validation overhead.

    Projects station coordinates onto the main axis (first->last) and
    checks that projections are non-decreasing (within 10% tolerance).
    ~100x cheaper than validate_station_set() since it skips road-graph
    distance and path bearing computations.
    """
    coords = station_data["coords_proj"]
    pts = coords[stations]
    main_vec = pts[-1] - pts[0]
    main_len_sq = float(np.dot(main_vec, main_vec))
    if main_len_sq < 100**2:
        return True  # too short to judge
    projs = [float(np.dot(pts[i] - pts[0], main_vec) / main_len_sq) for i in range(len(pts))]
    max_proj = projs[0]
    for p in projs[1:]:
        if p < max_proj - 0.10:
            return False
        max_proj = max(max_proj, p)
    return True


def mutate_station_set(
    corridor: dict,
    station_data: dict,
) -> Optional[dict]:
    """Mutate a station-set corridor for NSGA-II evolution.

    Strategies: swap (35%), add (20%), remove (15%), shift (20%), reorder (10%).
    """
    stations = list(corridor["stations"])
    n = len(stations)
    coords = station_data["coords_proj"]
    demand = station_data["demand_coverage"]
    adjacency = station_data["adjacency"]
    tree = station_data["tree"]

    roll = random.random()

    # --- Swap (35%): replace one station with a high-demand neighbor ---
    if roll < 0.35:
        idx = random.randint(0, n - 1)
        old_li = stations[idx]
        # Find candidate replacements within 600m
        nearby = tree.query_ball_point(coords[old_li], r=600.0 / US_SURVEY_FT_TO_M)
        candidates = [li for li in nearby if li != old_li and li not in stations]
        if candidates:
            # Pick weighted by demand
            cand_demand = demand[candidates]
            if cand_demand.sum() > 0:
                probs = cand_demand / cand_demand.sum()
                chosen = candidates[np.random.choice(len(candidates), p=probs)]
            else:
                chosen = random.choice(candidates)
            new_stations = list(stations)
            new_stations[idx] = chosen
            if not _check_monotonic(new_stations, station_data):
                pass  # skip expensive validation
            else:
                valid, _ = _get_validate()(new_stations, station_data)
                if valid:
                    return {"stations": new_stations, "source": "mutation_swap"}

    # --- Add (20%): insert in largest gap ---
    elif roll < 0.55 and n < MAX_STATIONS_PER_CORRIDOR:
        # Find the largest gap
        gaps = []
        for i in range(n - 1):
            d = np.hypot(
                coords[stations[i + 1], 0] - coords[stations[i], 0],
                coords[stations[i + 1], 1] - coords[stations[i], 1],
            )
            gaps.append((d, i))
        gaps.sort(reverse=True)
        for gap_d, gap_i in gaps[:3]:
            # Midpoint of the gap
            mx = (coords[stations[gap_i], 0] + coords[stations[gap_i + 1], 0]) / 2
            my = (coords[stations[gap_i], 1] + coords[stations[gap_i + 1], 1]) / 2
            nearby = tree.query_ball_point([mx, my], r=gap_d / 2)
            candidates = [li for li in nearby if li not in stations]
            if candidates:
                cand_demand = demand[candidates]
                if cand_demand.sum() > 0:
                    best = candidates[np.argmax(cand_demand)]
                else:
                    best = random.choice(candidates)
                new_stations = list(stations)
                new_stations.insert(gap_i + 1, best)
                valid, _ = _get_validate()(new_stations, station_data)
                if valid:
                    return {"stations": new_stations, "source": "mutation_add"}

    # --- Remove (15%): drop lowest-demand station ---
    elif roll < 0.70 and n > MIN_STATIONS_PER_CORRIDOR:
        # Allow terminal removal if the terminal scores below the
        # corridor's 25th-percentile demand.  This prevents corridors
        # from extending into low-demand fringe areas while preserving
        # strong termini.  Interior stations are always removable.
        station_demands = np.array([demand[stations[i]] for i in range(n)])
        p25 = float(np.percentile(station_demands, 25))
        removable = []
        for i in range(n):
            if i == 0 or i == n - 1:
                # Terminal: only removable if below 25th percentile
                if station_demands[i] < p25:
                    removable.append(i)
            else:
                removable.append(i)
        if removable:
            worst = removable[int(np.argmin([station_demands[i] for i in removable]))]
            new_stations = stations[:worst] + stations[worst + 1:]
            valid, _ = _get_validate()(new_stations, station_data)
            if valid:
                return {"stations": new_stations, "source": "mutation_remove"}

    # --- Shift (20%): move one station to adjacent intersection ---
    elif roll < 0.90:
        idx = random.randint(0, n - 1)
        old_li = stations[idx]
        neighbors = adjacency.get(old_li, [])
        candidates = [li for li in neighbors if li not in stations]
        if candidates:
            # Prefer higher demand
            cand_demand = demand[candidates]
            if cand_demand.sum() > 0:
                probs = cand_demand / cand_demand.sum()
                chosen = candidates[np.random.choice(len(candidates), p=probs)]
            else:
                chosen = random.choice(candidates)
            new_stations = list(stations)
            new_stations[idx] = chosen
            if not _check_monotonic(new_stations, station_data):
                pass  # skip expensive validation
            else:
                valid, _ = _get_validate()(new_stations, station_data)
                if valid:
                    return {"stations": new_stations, "source": "mutation_shift"}

    # --- Reorder (10%): reverse 2-3 consecutive interior stations ---
    else:
        if n >= 5:
            seg_len = random.randint(2, min(3, n - 2))
            start = random.randint(1, n - seg_len - 1)
            new_stations = list(stations)
            new_stations[start:start + seg_len] = reversed(new_stations[start:start + seg_len])
            if not _check_monotonic(new_stations, station_data):
                return None
            valid, _ = _get_validate()(new_stations, station_data)
            if valid:
                return {"stations": new_stations, "source": "mutation_reorder"}

    return None


def crossover_station_sets(
    parent_a: dict,
    parent_b: dict,
    station_data: dict,
) -> Optional[dict]:
    """Crossover two station-set corridors.

    Strategy 1: Geographic split at median x-coordinate.
    Strategy 2: Shared-station splice (if parents share nearby stations).
    """
    sa = parent_a["stations"]
    sb = parent_b["stations"]
    coords = station_data["coords_proj"]

    # Strategy 1: Shared-station splice
    # Find stations in A that are within STATION_PROXIMITY_M of stations in B
    pts_a = coords[sa]
    pts_b = coords[sb]
    tree_b = cKDTree(pts_b)
    shared_pairs = []  # (index_in_a, index_in_b, distance)
    for ai, pt in enumerate(pts_a):
        d, bi = tree_b.query(pt, k=1)
        if d < STATION_PROXIMITY_M / US_SURVEY_FT_TO_M:
            shared_pairs.append((ai, bi, d))

    if shared_pairs:
        # Use the shared point closest to the middle of both parents
        mid_a = len(sa) // 2
        mid_b = len(sb) // 2
        shared_pairs.sort(key=lambda x: abs(x[0] - mid_a) + abs(x[1] - mid_b))
        ai, bi, _ = shared_pairs[0]
        # First half of A up to splice + second half of B from splice
        child_stations = list(sa[:ai + 1]) + list(sb[bi + 1:])
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for s in child_stations:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        if _check_monotonic(deduped, station_data):
            valid, _ = _get_validate()(deduped, station_data)
            if valid:
                return {"stations": deduped, "source": "crossover_splice"}

    # Strategy 2: Geographic split at median x-coordinate
    all_stations = list(set(sa) | set(sb))
    all_x = coords[all_stations, 0]
    median_x = float(np.median(all_x))

    west_a = [s for s in sa if coords[s, 0] <= median_x]
    east_b = [s for s in sb if coords[s, 0] > median_x]

    if west_a and east_b:
        child_stations = west_a + east_b
        # Sort by projection onto principal axis (handles diagonal corridors)
        _axis = coords[child_stations[-1]] - coords[child_stations[0]]
        if np.dot(_axis, _axis) > 0:
            child_stations.sort(key=lambda s: np.dot(coords[s], _axis))
        else:
            child_stations.sort(key=lambda s: coords[s, 0])
        valid, _ = _get_validate()(child_stations, station_data)
        if valid:
            return {"stations": child_stations, "source": "crossover_geographic"}

    # Try reverse direction
    west_b = [s for s in sb if coords[s, 0] <= median_x]
    east_a = [s for s in sa if coords[s, 0] > median_x]
    if west_b and east_a:
        child_stations = west_b + east_a
        _axis = coords[child_stations[-1]] - coords[child_stations[0]]
        if np.dot(_axis, _axis) > 0:
            child_stations.sort(key=lambda s: np.dot(coords[s], _axis))
        else:
            child_stations.sort(key=lambda s: coords[s, 0])
        valid, _ = _get_validate()(child_stations, station_data)
        if valid:
            return {"stations": child_stations, "source": "crossover_geographic_rev"}

    return None


def deduplicate_station_sets(
    candidates: List[dict],
    station_data: dict,
    proximity_m: float = 200.0,
    overlap_threshold: float = 0.80,
) -> List[dict]:
    """Remove near-duplicate station-set corridors.

    Two corridors are duplicates if >overlap_threshold of stations in the
    smaller set have a match within proximity_m in the larger set.
    """
    if not candidates:
        return []

    coords = station_data["coords_proj"]
    _prox_ft = proximity_m / US_SURVEY_FT_TO_M
    deduped = [candidates[0]]
    # Cache KDTrees for deduped corridors to avoid rebuilding per comparison
    deduped_trees = [cKDTree(coords[candidates[0]["stations"]])]

    for cand in candidates[1:]:
        is_dup = False
        sc = cand["stations"]
        pts_c = coords[sc]
        tree_c = cKDTree(pts_c)

        for j, existing in enumerate(deduped):
            se = existing["stations"]
            # Pick the cached tree for whichever side is "larger"
            if len(sc) <= len(se):
                smaller_pts = pts_c
                tree_larger = deduped_trees[j]
            else:
                smaller_pts = coords[se]
                tree_larger = tree_c
            dists, _ = tree_larger.query(smaller_pts, k=1)
            n_matched = int(np.sum(dists < _prox_ft))
            if n_matched / max(len(smaller_pts), 1) > overlap_threshold:
                is_dup = True
                break

        if not is_dup:
            deduped.append(cand)
            deduped_trees.append(tree_c)

    return deduped


# ============================================================================
# DIVERSITY SELECTION
# ============================================================================

def _corridor_polyline(
    station_indices: List[int],
    coords_proj: np.ndarray,
    sample_spacing_ft: float = 200.0 / US_SURVEY_FT_TO_M,
    station_data: Optional[dict] = None,
) -> np.ndarray:
    """Sample a corridor into a dense polyline for spatial overlap computation.

    If *station_data* contains a ``path_node_cache`` with road-graph paths,
    uses actual road node coordinates instead of straight lines.  Falls back
    to straight-line interpolation for uncached segments.
    """
    pts = coords_proj[station_indices]
    if len(pts) < 2:
        return pts

    # Try road-graph path if available
    path_cache = None
    G = None
    if station_data is not None:
        path_cache = station_data.get("path_node_cache")
        G = station_data.get("graph")

    all_points: list = []
    for i in range(len(station_indices) - 1):
        seg_key = (station_indices[i], station_indices[i + 1])
        seg_path = path_cache.get(seg_key) if path_cache else None

        if seg_path and G is not None and len(seg_path) >= 2:
            # Use road-graph node coordinates, subsampled to target spacing
            road_pts: list = []
            nodes_to_add = seg_path if i == 0 else seg_path[1:]
            for nid in nodes_to_add:
                proj = _node_to_proj(nid, G)
                if proj is not None:
                    road_pts.append(proj)
            if road_pts:
                all_points.append(road_pts[0])
                accum_dist = 0.0
                for j in range(1, len(road_pts)):
                    dx = road_pts[j][0] - road_pts[j - 1][0]
                    dy = road_pts[j][1] - road_pts[j - 1][1]
                    accum_dist += (dx * dx + dy * dy) ** 0.5
                    if accum_dist >= sample_spacing_ft:
                        all_points.append(road_pts[j])
                        accum_dist = 0.0
                # Always include segment endpoint
                if road_pts[-1] != all_points[-1]:
                    all_points.append(road_pts[-1])
            else:
                # All _node_to_proj calls failed — fall back to station coord
                if not all_points:
                    all_points.append(tuple(pts[i]))
                seg_vec = pts[i + 1] - pts[i]
                seg_len = np.linalg.norm(seg_vec)
                if seg_len >= 1e-6:
                    n_samples = max(int(seg_len / sample_spacing_ft), 1)
                    for k in range(1, n_samples + 1):
                        frac = k / n_samples
                        all_points.append(tuple(pts[i] + frac * seg_vec))
        else:
            # Fallback: straight-line interpolation
            if not all_points:
                all_points.append(tuple(pts[i]))
            seg_vec = pts[i + 1] - pts[i]
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-6:
                continue
            n_samples = max(int(seg_len / sample_spacing_ft), 1)
            for k in range(1, n_samples + 1):
                frac = k / n_samples
                all_points.append(tuple(pts[i] + frac * seg_vec))

    if not all_points:
        return pts
    return np.array(all_points)


def _overlap_fraction(
    poly_a: np.ndarray,
    poly_b: np.ndarray,
    tree_b: Optional["cKDTree"] = None,
) -> float:
    """Fraction of points in poly_a within _OVERLAP_BUFFER_FT of poly_b."""
    if len(poly_a) == 0 or len(poly_b) == 0:
        return 0.0
    if tree_b is None:
        tree_b = cKDTree(poly_b)
    dists, _ = tree_b.query(poly_a, k=1)
    return float(np.mean(dists < _OVERLAP_BUFFER_FT))


def _bidirectional_overlap(
    poly_a: np.ndarray,
    poly_b: np.ndarray,
    tree_a: Optional["cKDTree"] = None,
    tree_b: Optional["cKDTree"] = None,
) -> float:
    """Max of overlap(A->B) and overlap(B->A).

    Using max (not mean) catches containment: a short corridor entirely inside
    a long one has overlap_ab ~ 0.3 but overlap_ba ~ 1.0.  max gives 1.0,
    correctly flagging the near-duplicate.
    """
    return max(
        _overlap_fraction(poly_a, poly_b, tree_b=tree_b),
        _overlap_fraction(poly_b, poly_a, tree_b=tree_a),
    )


def select_diverse_station_sets(
    candidates: List[dict],
    station_data: dict,
    max_overlap: float = OVERLAP_THRESHOLD,
    max_select: int = 25,
) -> List[dict]:
    """Select corridors using MMR with bidirectional overlap fraction diversity.

    Sorts by blended score (60% ridership density + 40% total ridership) rather
    than total ridership alone, so short dense corridors compete with long ones.
    Filters corridors below MIN_RIDERSHIP_FOR_SELECTION before selection.

    Samples each corridor into a dense polyline and measures geographic overlap
    as the fraction of one polyline within OVERLAP_BUFFER_M of the other.
    Uses max(A->B, B->A) to catch containment (short corridor inside long one).
    """
    if not candidates:
        return []

    # Filter out corridors below minimum ridership floor
    candidates = [
        c for c in candidates
        if c["score"]["ridership_est"] >= MIN_RIDERSHIP_FOR_SELECTION
    ]
    if not candidates:
        return []

    coords_proj = station_data["coords_proj"]

    # Blended quality: 60% ridership density (riders/km) + 40% total ridership
    # Both normalized to 0-1 so they contribute proportionally.
    _riderships = np.array([c["score"]["ridership_est"] for c in candidates])
    _lengths = np.array([max(c["score"]["length_km"], 0.5) for c in candidates])
    _densities = _riderships / _lengths

    _r_lo, _r_hi = float(_riderships.min()), float(_riderships.max())
    _r_range = max(_r_hi - _r_lo, 1.0)
    _r_norm = (_riderships - _r_lo) / _r_range

    _d_lo, _d_hi = float(_densities.min()), float(_densities.max())
    _d_range = max(_d_hi - _d_lo, 1.0)
    _d_norm = (_densities - _d_lo) / _d_range

    _blended = 0.6 * _d_norm + 0.4 * _r_norm
    _sort_order = np.argsort(-_blended)
    sorted_cands = [candidates[int(i)] for i in _sort_order]
    _blended_sorted = _blended[_sort_order]

    # Pre-compute polylines + KDTrees for all candidates
    polylines = [
        _corridor_polyline(cand["stations"], coords_proj, station_data=station_data)
        for cand in sorted_cands
    ]
    poly_trees = [cKDTree(p) if len(p) > 0 else None for p in polylines]

    # Greedy MMR with decaying lambda
    # Start quality-heavy (lambda=0.3), shift to diversity-heavy (lambda=0.6)
    # as the selection fills.
    LAMBDA_START = 0.3
    LAMBDA_END = 0.6

    selected = [0]  # Best blended-score corridor is always first
    remaining = set(range(1, len(sorted_cands)))

    while remaining and len(selected) < max_select:
        best_idx = None
        best_mmr = -1.0

        # Decay lambda linearly from LAMBDA_START to LAMBDA_END
        progress = len(selected) / max_select
        mmr_lambda = LAMBDA_START + (LAMBDA_END - LAMBDA_START) * progress

        for i in remaining:
            # Max overlap with any already-selected corridor
            max_ol = max(
                _bidirectional_overlap(
                    polylines[i], polylines[s],
                    tree_a=poly_trees[i], tree_b=poly_trees[s],
                )
                for s in selected
            )
            if max_ol > max_overlap:
                continue  # near-duplicate, skip

            quality = float(_blended_sorted[i])  # already 0-1 normalized
            mmr = (1 - mmr_lambda) * quality + mmr_lambda * (1 - max_ol)
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    # Print overlap statistics for diagnostics (reuses pre-built trees)
    if len(selected) > 1:
        overlaps = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                si, sj = selected[i], selected[j]
                ol = _bidirectional_overlap(
                    polylines[si], polylines[sj],
                    tree_a=poly_trees[si], tree_b=poly_trees[sj],
                )
                overlaps.append(ol)
        logger.debug(f"  Overlap stats: min={min(overlaps):.2f} median={np.median(overlaps):.2f} "
              f"max={max(overlaps):.2f} (threshold={max_overlap:.2f})")

    return [sorted_cands[i] for i in selected]


# ============================================================================
# POST-SEARCH REFINEMENT & SYNERGY SCORING
# ============================================================================

def refine_station_placements(
    corridors: List[dict],
    station_data: dict,
    od_flows: Optional["pd.DataFrame"] = None,
    parcel_lookup: Optional[dict] = None,
    search_radius_m: float = 300.0,
    max_rounds: int = 3,
) -> List[dict]:
    """Improve station positions via greedy hill-climbing on the full scorer.

    For each corridor, iterates over every station and tries swapping it with
    each nearby candidate station (within *search_radius_m*).  If the swap
    produces a higher ``ridership_est`` *and* still passes
    ``validate_station_set``, the swap is kept.  Repeats up to *max_rounds*
    or until no further improvement is found.

    This post-search pass is cheap (~5-15 evaluations per station per round,
    ~800 total) and captures micro-siting gains the NSGA-II's coarser
    mutations may miss.
    """
    if not corridors:
        return corridors

    tree = station_data["tree"]
    coords = station_data["coords_proj"]
    search_r = search_radius_m / US_SURVEY_FT_TO_M  # convert to feet for EPSG:2965
    total_improved = 0
    total_evals = 0

    for ci, corridor in enumerate(corridors):
        stations = list(corridor["stations"])
        original_score = corridor["score"]["ridership_est"]
        best_score = original_score
        improved_this_corridor = False

        for _round in range(max_rounds):
            any_swap = False

            for si in range(len(stations)):
                current_li = stations[si]
                neighbors = tree.query_ball_point(
                    coords[current_li], r=search_r,
                )
                # Skip self
                neighbors = [n for n in neighbors if n != current_li]
                if not neighbors:
                    continue

                best_neighbor = None
                best_neighbor_score = best_score

                for ni in neighbors:
                    # Build trial station list
                    trial = list(stations)
                    trial[si] = ni

                    # Quick duplicate check
                    if len(set(trial)) != len(trial):
                        continue

                    # Validate spacing / length constraints
                    valid, _msg = _get_validate()(trial, station_data)
                    if not valid:
                        continue

                    total_evals += 1
                    trial_result = _get_score()(
                        trial, station_data,
                        od_flows=od_flows, parcel_lookup=parcel_lookup,
                    )
                    if trial_result["ridership_est"] > best_neighbor_score:
                        best_neighbor = ni
                        best_neighbor_score = trial_result["ridership_est"]
                        best_trial_result = trial_result

                if best_neighbor is not None:
                    stations[si] = best_neighbor
                    best_score = best_neighbor_score
                    corridor["score"] = best_trial_result
                    any_swap = True
                    improved_this_corridor = True

            if not any_swap:
                break  # converged this corridor

        if improved_this_corridor:
            corridor["stations"] = stations
            total_improved += 1
            logger.debug(f"  C{ci+1}: {original_score:.0f} -> {best_score:.0f} "
                  f"(+{(best_score/max(original_score,1)-1)*100:.1f}%)")

    logger.debug(f"  Refined {total_improved}/{len(corridors)} corridors "
          f"({total_evals} evaluations)")
    return corridors


def apply_station_synergy_scores(
    candidates: List[dict],
    station_data: dict,
    evaluation_mode: str = "isolated",
    synergy_weight: float = NETWORK_SYNERGY_WEIGHT_DEFAULT,
    anchor_top_k: int = 12,
    transfer_radius_m: float = NETWORK_TRANSFER_RADIUS_M,
) -> List[dict]:
    """Apply network synergy scoring to station-set candidates."""
    if not candidates:
        return candidates

    mode = str(evaluation_mode).strip().lower()
    coords = station_data["coords_proj"]

    for cand in candidates:
        score = cand.get("score", {})
        base = float(score.get("ridership_est_base", score.get("ridership_est", 0.0)))
        score["ridership_est_base"] = base
        score["network_synergy"] = 0.0
        score["ridership_network_adjusted"] = base
        score["evaluation_mode"] = mode
        score["ridership_est"] = base

    if mode == "isolated" or len(candidates) <= 1:
        return candidates

    # Anchors = top-K by isolated ridership
    ranked = sorted(
        candidates,
        key=lambda c: float(c.get("score", {}).get("ridership_est_base", 0.0)),
        reverse=True,
    )
    anchors = ranked[:max(1, int(anchor_top_k))]

    for cand in candidates:
        score = cand["score"]
        c_stations = set(cand["stations"])
        c_pts = coords[list(c_stations)]

        pair_scores = []
        for anchor in anchors:
            if anchor is cand:
                continue
            a_stations = set(anchor["stations"])
            a_pts = coords[list(a_stations)]

            # Complementarity (1 - station overlap)
            intersection = len(c_stations & a_stations)
            union = len(c_stations | a_stations)
            overlap = intersection / max(union, 1)
            complementarity = 1.0 - overlap

            # Transfer opportunity (endpoint proximity)
            c_endpoints = np.array([c_pts[0], c_pts[-1]])
            a_endpoints = np.array([a_pts[0], a_pts[-1]])
            dmat = np.sqrt(((c_endpoints[:, None, :] - a_endpoints[None, :, :]) ** 2).sum(axis=2))
            min_dist_ft = float(np.min(dmat))
            min_dist_m = min_dist_ft * US_SURVEY_FT_TO_M
            transfer = float(np.exp(-min_dist_m / max(transfer_radius_m, 1.0))) if min_dist_m < transfer_radius_m else 0.0

            pair_synergy = 0.6 * complementarity + 0.4 * transfer
            pair_scores.append(float(np.clip(pair_synergy, 0.0, 1.0)))

        synergy = float(np.max(pair_scores)) if pair_scores else 0.0
        base = float(score["ridership_est_base"])
        adjusted = base * (1.0 + float(synergy_weight) * synergy)
        score["network_synergy"] = synergy
        score["ridership_network_adjusted"] = adjusted
        score["ridership_est"] = adjusted

    return candidates

#!/usr/bin/env python
"""Build a routable network from roads and compute simple parcel accessibility metrics.

Outputs:
- data/processed/network_nodes.geojson
- data/processed/network_edges.geojson
- data/processed/parcels_accessibility.csv (PARCEL_ID, dist_to_nearest_node_m)
- Optionally, if pandana available, write a pickled network or keep node/edge csv for reconstruction
"""
from pathlib import Path
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import logging

logger = logging.getLogger(__name__)

RAW_ROADS = Path('data/raw/roads.geojson')
OUT_DIR = Path('data/processed')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_nodes_edges_from_roads(roads_gdf):
    """Extract nodes (endpoints) and edges from road centerlines."""
    nodes = []
    edges = []
    node_index = {}

    def add_node(pt):
        key = (round(pt.x, 6), round(pt.y, 6))
        if key in node_index:
            return node_index[key]
        idx = len(nodes)
        nodes.append({'node_id': idx, 'x': pt.x, 'y': pt.y, 'geometry': pt})
        node_index[key] = idx
        return idx

    for i, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        # handle multilinestring or linestring
        if geom.geom_type == 'MultiLineString':
            parts = list(geom)
        else:
            parts = [geom]
        for seg in parts:
            coords = list(seg.coords)
            if len(coords) < 2:
                continue
            u_pt = Point(coords[0])
            v_pt = Point(coords[-1])
            u = add_node(u_pt)
            v = add_node(v_pt)
            length = seg.length
            edges.append({'u': u, 'v': v, 'length': length, 'geometry': seg})

    nodes_gdf = gpd.GeoDataFrame(nodes, geometry=[n['geometry'] for n in nodes], crs=roads_gdf.crs)
    edges_gdf = gpd.GeoDataFrame(edges, geometry=[e['geometry'] for e in edges], crs=roads_gdf.crs)
    return nodes_gdf, edges_gdf


def add_travel_time_to_edges(edges_gdf, speed_kph=50):
    """Compute edge length in meters and travel time in seconds using a constant speed (kph).

    Projects geometries to WebMercator (EPSG:3857) for metric lengths. Returns a copy with
    columns `length_m` and `travel_time_s`.
    """
    if edges_gdf.crs is None or edges_gdf.crs.to_string().startswith('EPSG:4326'):
        edges_metric = edges_gdf.to_crs(epsg=3857).copy()
    else:
        edges_metric = edges_gdf.copy()

    edges_metric['length_m'] = edges_metric.geometry.length
    # convert speed from kph to meters per second
    v_mps = speed_kph * 1000.0 / 3600.0
    # avoid division by zero
    if v_mps <= 0:
        raise ValueError("speed_kph must be positive")
    edges_metric['travel_time_s'] = edges_metric['length_m'] / v_mps
    return edges_metric


def try_build_pandana_network(nodes_gdf, edges_gdf):
    try:
        import pandana as pdna
    except Exception:
        warnings.warn("pandana not available; skipping pan-based accessibility calculations")
        return None

    # Require edges to build a meaningful network with weights
    if edges_gdf is None:
        warnings.warn("Edges not provided; cannot build pandana network without edges")
        return None

    # pandana expects node x,y arrays and link-from, link-to and link_length
    # If travel_time_s is available, build network in metric coordinates so link_length is in seconds
    if 'travel_time_s' in edges_gdf.columns:
        # project nodes to metric to provide numeric coordinates; link_length is travel_time_s (seconds)
        if nodes_gdf.crs is None or nodes_gdf.crs.to_string().startswith('EPSG:4326'):
            nodes_proj = nodes_gdf.to_crs(epsg=3857)
        else:
            nodes_proj = nodes_gdf
        node_x = nodes_proj.geometry.x.values
        node_y = nodes_proj.geometry.y.values
        link_length = edges_gdf['travel_time_s'].values
    else:
        # fallback: use geometric length in metric units
        if edges_gdf.crs is None or edges_gdf.crs.to_string().startswith('EPSG:4326'):
            edges_proj = edges_gdf.to_crs(epsg=3857)
        else:
            edges_proj = edges_gdf
        if nodes_gdf.crs is None or nodes_gdf.crs.to_string().startswith('EPSG:4326'):
            nodes_proj = nodes_gdf.to_crs(epsg=3857)
        else:
            nodes_proj = nodes_gdf
        node_x = nodes_proj.geometry.x.values
        node_y = nodes_proj.geometry.y.values
        # use metric geometry length if available in projected edges
        if 'length_m' in edges_proj.columns:
            link_length = edges_proj['length_m'].values
        elif 'length' in edges_proj.columns:
            link_length = edges_proj['length'].values
        else:
            link_length = edges_proj.geometry.length.values

    links_from = edges_gdf['u'].astype(int).values
    links_to = edges_gdf['v'].astype(int).values

    net = pdna.Network(node_x, node_y, links_from, links_to, link_length)
    return net


def compute_nearest_node_distances(parcels_gdf, nodes_gdf):
    # Handle case with no nodes gracefully
    if nodes_gdf is None or len(nodes_gdf) == 0:
        # Project parcels to metric for consistency
        if parcels_gdf.crs is None or parcels_gdf.crs.to_string().startswith('EPSG:4326'):
            parcels = parcels_gdf.to_crs(epsg=3857).copy()
        else:
            parcels = parcels_gdf.copy()
        parcels['dist_to_node_m'] = np.nan
        parcels['nearest_node_id'] = -1
        return parcels[['PARCEL_ID', 'dist_to_node_m', 'nearest_node_id']]

    # Ensure same CRS. If geographic, project to WebMercator for distance in meters
    if parcels_gdf.crs is None or parcels_gdf.crs.to_string().startswith('EPSG:4326'):
        parcels = parcels_gdf.to_crs(epsg=3857)
        nodes = nodes_gdf.to_crs(epsg=3857)
    else:
        parcels = parcels_gdf.copy()
        nodes = nodes_gdf.copy()

    node_coords = np.vstack([nodes.geometry.x.values, nodes.geometry.y.values]).T
    from scipy.spatial import cKDTree
    tree = cKDTree(node_coords)

    parcel_points = np.vstack([parcels.geometry.centroid.x.values, parcels.geometry.centroid.y.values]).T
    dists, idx = tree.query(parcel_points, k=1)
    parcels = parcels.copy()
    parcels['dist_to_node_m'] = dists
    parcels['nearest_node_id'] = nodes.iloc[idx].node_id.values
    return parcels[['PARCEL_ID', 'dist_to_node_m', 'nearest_node_id']]

def compute_accessibilities(nodes_gdf, opportunities_gdf, parcels_access_df, radii_meters=(400,800,1600), pandana_net=None):
    """Compute opportunity counts within radii around nodes and attribute them to parcels.

    If `pandana_net` is provided (a prepared pandana.Network), it will be used for network-based aggregation.
    Otherwise, the function will attempt to import `pandana` and build a network; if that fails, it falls back to Euclidean KDTree neighbor sums.

    Returns a DataFrame keyed by PARCEL_ID with columns `opp_count_{r}` for each radius r in meters.
    """
    # Project to metric coordinates
    if nodes_gdf.crs is None or nodes_gdf.crs.to_string().startswith('EPSG:4326'):
        nodes = nodes_gdf.to_crs(epsg=3857).copy()
        opps = opportunities_gdf.to_crs(epsg=3857).copy()
    else:
        nodes = nodes_gdf.copy()
        opps = opportunities_gdf.copy()

    node_coords = np.vstack([nodes.geometry.x.values, nodes.geometry.y.values]).T
    from scipy.spatial import cKDTree
    node_tree = cKDTree(node_coords)

    # map opportunities to nearest node
    opp_points = np.vstack([opps.geometry.x.values, opps.geometry.y.values]).T
    _, opp_node_idx = node_tree.query(opp_points, k=1)
    # counts per node
    node_counts = np.zeros(len(nodes), dtype=int)
    for idx in opp_node_idx:
        node_counts[idx] += 1

    results = {}

    # If a pandana_net is explicitly provided, prefer it
    use_pandana = False
    net = None
    if pandana_net is not None:
        use_pandana = True
        net = pandana_net
    else:
        try:
            import pandana as pdna
            # build a pandana network from nodes+edges if possible
            # caller may provide an already-built network via pandana_net
            use_pandana = True
        except Exception:
            use_pandana = False

    if use_pandana and net is None:
        # attempt to build a pandana network from node positions only (no topology) - skip if not possible
        try:
            net = try_build_pandana_network(nodes_gdf, None)
            if net is None:
                use_pandana = False
        except Exception:
            use_pandana = False

    if use_pandana and net is not None:
        try:
            # set counts at nodes for POIs
            node_ids = np.arange(len(nodes))
            poi_node_ids = np.repeat(node_ids, node_counts)
            if len(poi_node_ids) > 0:
                net.set_pois('opps', poi_node_ids)
            for r in radii_meters:
                agg = net.aggregate(r, type='count', name='opps')
                counts = agg['opps'].values
                results[r] = counts
        except Exception:
            # fallback
            use_pandana = False

    if not use_pandana:
        # Euclidean fallback: for each node, sum node_counts of neighbors within radius
        for r in radii_meters:
            neigh_lists = node_tree.query_ball_point(node_coords, r)
            counts = np.array([node_counts[np.array(nb)].sum() if len(nb) > 0 else 0 for nb in neigh_lists], dtype=int)
            results[r] = counts

    if not use_pandana:
        # Euclidean fallback: for each node, sum node_counts of neighbors within radius
        for r in radii_meters:
            neigh_lists = node_tree.query_ball_point(node_coords, r)
            counts = np.array([node_counts[np.array(nb)].sum() if len(nb) > 0 else 0 for nb in neigh_lists], dtype=int)
            results[r] = counts

    # Map node results to parcels via their nearest_node_id from parcels_access_df
    parcels = parcels_access_df.copy()
    for r, counts in results.items():
        col = f"opp_count_{int(r)}"
        parcels[col] = parcels['nearest_node_id'].map(lambda nid: counts[nid] if (nid is not None and nid >= 0 and nid < len(counts)) else 0)

    # Optionally compute travel-time based accessibility if thresholds are provided in seconds
    # (delegates to compute_travel_time_accessibilities)
    return parcels[['PARCEL_ID'] + [f"opp_count_{int(r)}" for r in radii_meters]]


def compute_travel_time_accessibilities(nodes_gdf, edges_gdf, opportunities_gdf, parcels_access_df, thresholds_s=(300,600,1200), pandana_net=None, speed_kph=50, parcels_gdf=None):
    """Compute counts of opportunities reachable within travel-time thresholds (seconds) from each parcel.

    Uses pandana when available and a network (with `travel_time_s` edge attribute) is provided. If pandana is not available or network can't be built, falls back to an Euclidean time approximation using `speed_kph`.

    `parcels_access_df` may be a DataFrame with `PARCEL_ID` and `nearest_node_id` (no geometry); if so, provide `parcels_gdf` (GeoDataFrame with geometry) so centroids can be computed.

    Returns a DataFrame keyed by PARCEL_ID with columns `opp_count_tt_{t}` for each threshold t (seconds).
    """
    # If no nodes are provided, fall back to Euclidean travel-time approximation between parcels and POIs
    if nodes_gdf is None or len(nodes_gdf) == 0:
        # Euclidean fallback using parcels geometry directly
        opps = opportunities_gdf.copy()
        parcels = parcels_gdf.copy() if parcels_gdf is not None else parcels_access_df.copy()
        if opps.crs is None or opps.crs.to_string().startswith('EPSG:4326'):
            opps = opps.set_crs(epsg=4326, allow_override=True).to_crs(epsg=3857)
        else:
            opps = opps.to_crs(epsg=3857)
        if parcels.crs is None or parcels.crs.to_string().startswith('EPSG:4326'):
            parcels = parcels.set_crs(epsg=4326, allow_override=True).to_crs(epsg=3857)
        else:
            parcels = parcels.to_crs(epsg=3857)

        # Use KDTree on opportunities and parcel centroids
        from scipy.spatial import cKDTree
        opp_coords = np.vstack([opps.geometry.x.values, opps.geometry.y.values]).T
        opp_tree = cKDTree(opp_coords)

        # speed in m/s
        v_mps = (speed_kph * 1000.0) / 3600.0

        results = {}
        parcel_centroids = np.vstack([parcels.geometry.centroid.x.values, parcels.geometry.centroid.y.values]).T
        for t in thresholds_s:
            radius_m = v_mps * float(t)
            neigh_lists = opp_tree.query_ball_point(parcel_centroids, r=radius_m)
            counts = np.array([len(nb) for nb in neigh_lists], dtype=int)
            results[t] = counts

        out = parcels[['PARCEL_ID']].copy()
        for t, counts in results.items():
            col = f"opp_count_tt_{int(t)}"
            out[col] = counts
        return out

    # Prepare metric projections
    if nodes_gdf.crs is None or nodes_gdf.crs.to_string().startswith('EPSG:4326'):
        nodes = nodes_gdf.to_crs(epsg=3857).copy()
        opps = opportunities_gdf.to_crs(epsg=3857).copy()
    else:
        nodes = nodes_gdf.copy()
        opps = opportunities_gdf.copy()

    # parcels_access_df may be a DataFrame (no geometry) or a GeoDataFrame
    if hasattr(parcels_access_df, 'geometry'):
        parcels = parcels_access_df.copy()
        # Ensure parcels are in metric CRS
        if parcels.crs is None or parcels.crs.to_string().startswith('EPSG:4326'):
            parcels = parcels.to_crs(epsg=3857)
    else:
        # No geometry; require parcels_gdf to be provided
        if parcels_gdf is None:
            raise ValueError('parcels_access_df does not contain geometry; pass parcels_gdf GeoDataFrame to compute travel-time accessibilities')
        parcels = parcels_gdf.copy()
        if parcels.crs is None or parcels.crs.to_string().startswith('EPSG:4326'):
            parcels = parcels.to_crs(epsg=3857)
        # merge access metadata (nearest_node_id) into parcels by PARCEL_ID if present
        if 'PARCEL_ID' in parcels_access_df.columns:
            parcels = parcels.merge(parcels_access_df[['PARCEL_ID', 'nearest_node_id']], on='PARCEL_ID', how='left')

    # map opportunities to nearest node to simplify POI assignment
    node_coords = np.vstack([nodes.geometry.x.values, nodes.geometry.y.values]).T
    from scipy.spatial import cKDTree
    node_tree = cKDTree(node_coords)
    opp_points = np.vstack([opps.geometry.x.values, opps.geometry.y.values]).T
    _, opp_node_idx = node_tree.query(opp_points, k=1)

    # Attempt to use pandana network if available
    use_pandana = False
    net = None
    if pandana_net is not None:
        use_pandana = True
        net = pandana_net
    else:
        try:
            import pandana as pdna
            net = try_build_pandana_network(nodes_gdf, edges_gdf)
            if net is not None:
                use_pandana = True
        except Exception:
            use_pandana = False

    results = {}

    if use_pandana and net is not None:
        # assign POIs to nodes by replicating node indices by counts of POIs per node
        node_ids = np.arange(len(nodes))
        node_counts = np.zeros(len(nodes), dtype=int)
        for idx in opp_node_idx:
            node_counts[idx] += 1
        poi_node_ids = np.repeat(node_ids, node_counts)
        if len(poi_node_ids) > 0:
            net.set_pois('opps', poi_node_ids)
        # aggregate for each threshold (units are seconds when travel_time_s used)
        for t in thresholds_s:
            # when using travel_time_s, pandana expects distances as travel time units (seconds)
            agg = net.aggregate(t, type='count', name='opps')
            counts = agg['opps'].values
            results[t] = counts

        # Additionally, compute shortest travel-time to nearest POI for each node using pandana's shortest-path
        try:
            if hasattr(net, 'shortest_path') or hasattr(net, 'get_travel_time'):
                # pandana supports shortest path via `net.shortest_path` in some versions; else use `net.get_travel_time`
                # We'll attempt to compute shortest travel time from each node to nearest POI by setting POIs and querying
                # Create an array of POI counts per node; then compute distance to nearest POI per node
                # Pandana provides `nearest_poi` or `get_travel_time` depending on version; attempt both
                node_tt_to_poi = np.zeros(len(nodes), dtype=float)
                # Attempt to use `net.shortest_path` if available (older versions may have `networkx`-style API)
                try:
                    # Use aggregated POIs as 'opps' and compute nearest poi travel-time using network methods
                    # `net._get_graph` may expose edge weights in seconds; fallback handled below
                    for nid in range(len(nodes)):
                        # pandana doesn't always expose a direct 'distance to nearest POI' API; use aggregate radius search as proxy
                        # Here we set a large radius and then query the nearest
                        res = net.aggregate(1e9, type='count', name='opps')
                        # This returns counts; to get travel times we need to attempt get_travel_time if present
                    # If we reached here, but could not compute node_tt_to_poi, leave zeros
                    pass
                except Exception:
                    pass
        except Exception:
            # ignore pandana-specific short-path failures; fallback will be used elsewhere
            pass
    else:
        # Euclidean fallback: approximate travel time by distance / speed
        v_mps = speed_kph * 1000.0 / 3600.0
        # node locations and a KDTree for opportunities
        opp_coords = np.vstack([opps.geometry.x.values, opps.geometry.y.values]).T
        opp_tree = cKDTree(opp_coords)
        # compute parcel centroids in metric
        # Ensure parcels geometries are in metric CRS
        # parcels_access_df may be in geographic CRS; project centroids accordingly
        # We expect parcels to contain PARCEL_ID and nearest_node_id already
        # For counting, use parcel centroids
        # Build array of centroid coords
        parcel_centroids = np.vstack([parcels.geometry.centroid.x.values, parcels.geometry.centroid.y.values]).T
        for t in thresholds_s:
            radius_m = v_mps * float(t)
            neigh_lists = opp_tree.query_ball_point(parcel_centroids, r=radius_m)
            counts = np.array([len(nb) for nb in neigh_lists], dtype=int)
            results[t] = counts

    # Map results to parcels via row order (parcels order must correspond to counts order)
    # Ensure parcels are in the same order as computed (we used parcel_centroids order)
    out = parcels[['PARCEL_ID']].copy()
    for t, counts in results.items():
        col = f"opp_count_tt_{int(t)}"
        # If counts length matches parcels rows, map directly, else map via nearest_node_id
        if len(counts) == len(out):
            out[col] = counts
        else:
            # Map node-based counts using nearest_node_id
            out[col] = parcels['nearest_node_id'].map(lambda nid: counts[nid] if (nid is not None and nid >= 0 and nid < len(counts)) else 0)
    return out


def compute_job_accessibilities(nodes_gdf, edges_gdf, jobs_gdf, parcels_access_df, thresholds_s=(300,600,1200), pandana_net=None, speed_kph=50, parcels_gdf=None, weight_col='jobs'):
    """Compute counts and weighted sums of jobs reachable within travel-time thresholds (seconds) from each parcel.

    Prefers pandana network-based aggregation when a pandana network is available and `pandana_net` is provided or can be built.
    Falls back to Euclidean/KDTree time approximation using `speed_kph` when pandana is unavailable.

    Parameters:
    - nodes_gdf, edges_gdf: network node/edge GeoDataFrames (CRS may be geographic)
    - jobs_gdf: GeoDataFrame of POIs with at least geometry and a jobs weight column (name given by `weight_col`). If no weight_col is present, each POI counts as 1.
    - parcels_access_df: DataFrame or GeoDataFrame containing `PARCEL_ID` and optional `nearest_node_id`. If it lacks geometry, supply `parcels_gdf`.
    - thresholds_s: iterable of travel-time thresholds in seconds
    - pandana_net: optional pre-built pandana network to use (preferred when provided)
    - speed_kph: fallback travel speed used to convert distance to time (meters/sec)
    - parcels_gdf: GeoDataFrame with geometry for parcels when `parcels_access_df` has no geometry

    Returns a DataFrame keyed by `PARCEL_ID` with columns like `job_count_tt_{t}` and `job_sum_tt_{t}`.
    """
    # Validate inputs
    if weight_col not in jobs_gdf.columns:
        # treat each POI as one job if weight column missing
        jobs = jobs_gdf.copy()
        jobs[weight_col] = 1
    else:
        jobs = jobs_gdf.copy()

    # Prepare metric projection for nodes / jobs / parcels when needed
    # We'll attempt a pandana-based approach first
    use_pandana = False
    net = None
    try:
        import pandana as pdna
    except Exception:
        pdna = None

    if pandana_net is not None and pdna is not None:
        use_pandana = True
        net = pandana_net
    else:
        if pdna is not None:
            # try to build a pandana network from nodes+edges
            try:
                net = try_build_pandana_network(nodes_gdf, edges_gdf)
                if net is not None:
                    use_pandana = True
            except Exception:
                use_pandana = False

    results = None

    if use_pandana and net is not None:
        # Ensure jobs have x/y
        j = jobs.copy()
        if 'x' not in j.columns or 'y' not in j.columns:
            j['x'] = j.geometry.x
            j['y'] = j.geometry.y
        # map jobs to node ids
        try:
            job_node_ids = net.get_node_ids_from_xy(j['x'].values, j['y'].values)
        except Exception:
            job_node_ids = None

        # Build per-node job counts and sums
        node_job_counts = np.zeros(len(net.xs), dtype=int) if hasattr(net, 'xs') else None
        node_job_sums = np.zeros(len(net.xs), dtype=float) if hasattr(net, 'xs') else None
        if job_node_ids is not None:
            # When net.xs exists, length equals number of nodes
            n_nodes = len(net.xs) if hasattr(net, 'xs') else None
            # Safe fallback size estimate
            if n_nodes is None:
                # try to infer node count from nodes_gdf
                n_nodes = len(nodes_gdf)
            node_job_counts = np.zeros(n_nodes, dtype=int)
            node_job_sums = np.zeros(n_nodes, dtype=float)
            for nid, w in zip(job_node_ids, j[weight_col].values):
                if nid is None:
                    continue
                if nid >= 0 and nid < n_nodes:
                    node_job_counts[nid] += 1
                    node_job_sums[nid] += float(w)
        # Add node attribute for job sums if possible so we can call aggregate(type='sum')
        can_add_attr = False
        try:
            # Some pandana versions support `add_attribute` or `set_node_values`; we try add_attribute first
            if hasattr(net, 'add_attribute'):
                attr = {i: float(node_job_sums[i]) for i in range(len(node_job_sums))}
                net.add_attribute(attr, name='job_sum_tmp')
                can_add_attr = True
            elif hasattr(net, 'set_node_values'):
                net.set_node_values('job_sum_tmp', node_job_sums)
                can_add_attr = True
        except Exception:
            can_add_attr = False

        # Build parcel list for mapping
        if hasattr(parcels_access_df, 'geometry'):
            parcels = parcels_access_df.copy()
            if parcels.crs is None or parcels.crs.to_string().startswith('EPSG:4326'):
                parcels = parcels.to_crs(epsg=3857)
        else:
            if parcels_gdf is None:
                raise ValueError('parcels_access_df lacks geometry; pass parcels_gdf GeoDataFrame or ensure parcels_access_df has geometry')
            parcels = parcels_gdf.copy()
            if parcels.crs is None or parcels.crs.to_string().startswith('EPSG:4326'):
                parcels = parcels.to_crs(epsg=3857)
            if 'PARCEL_ID' in parcels_access_df.columns:
                parcels = parcels.merge(parcels_access_df[['PARCEL_ID', 'nearest_node_id']], on='PARCEL_ID', how='left')

        # prepare output structure
        out = parcels[['PARCEL_ID']].copy()

        try:
            # For each threshold t, use net.aggregate for counts and node attribute sum for sums when available
            for t in thresholds_s:
                count_col = f'job_count_tt_{int(t)}'
                sum_col = f'job_sum_tt_{int(t)}'
                try:
                    # set POIs for counting
                    if job_node_ids is not None and len(job_node_ids) > 0:
                        net.set_pois('jobs', job_node_ids)
                        agg_counts = net.aggregate(t, type='count', name='jobs')
                        counts = agg_counts['jobs'].values
                    else:
                        counts = np.zeros(len(nodes_gdf), dtype=int)
                except Exception:
                    counts = None

                sums = None
                if can_add_attr:
                    try:
                        agg_sums = net.aggregate(t, type='sum', name='job_sum_tmp')
                        sums = agg_sums['job_sum_tmp'].values
                    except Exception:
                        sums = None

                # map node-based arrays to parcels via nearest_node_id if counts length differs from parcels
                if counts is not None and len(counts) == len(parcels):
                    out[count_col] = counts
                else:
                    out[count_col] = parcels['nearest_node_id'].map(lambda nid: counts[nid] if (counts is not None and nid is not None and nid >= 0 and nid < len(counts)) else 0)

                if sums is not None and len(sums) == len(parcels):
                    out[sum_col] = sums
                elif sums is not None:
                    out[sum_col] = parcels['nearest_node_id'].map(lambda nid: sums[nid] if (nid is not None and nid >= 0 and nid < len(sums)) else 0)
                else:
                    # fallback to mapping node_job_sums directly
                    out[sum_col] = parcels['nearest_node_id'].map(lambda nid: float(node_job_sums[nid]) if (nid is not None and nid >= 0 and nid < len(node_job_sums)) else 0.0)

            # cleanup temporary attributes if any
            try:
                if can_add_attr and hasattr(net, 'remove_attribute'):
                    net.remove_attribute('job_sum_tmp')
            except Exception:
                pass

            results = out
            return results
        except Exception:
            # pandana attempt failed -- fallback to Euclidean
            pass

    # Euclidean fallback: use KDTree on job locations and approximate travel time by distance/speed
    jobs_proj = jobs.copy()
    parcels_proj = None
    if jobs_proj.crs is None or jobs_proj.crs.to_string().startswith('EPSG:4326'):
        jobs_proj = jobs_proj.set_crs(epsg=4326, allow_override=True).to_crs(epsg=3857)
    else:
        jobs_proj = jobs_proj.to_crs(epsg=3857)

    if hasattr(parcels_access_df, 'geometry'):
        parcels_proj = parcels_access_df.copy()
        if parcels_proj.crs is None or parcels_proj.crs.to_string().startswith('EPSG:4326'):
            parcels_proj = parcels_proj.set_crs(epsg=4326, allow_override=True).to_crs(epsg=3857)
        else:
            parcels_proj = parcels_proj.to_crs(epsg=3857)
    else:
        if parcels_gdf is None:
            raise ValueError('parcels_access_df does not contain geometry; pass parcels_gdf GeoDataFrame to compute job accessibilities')
        parcels_proj = parcels_gdf.copy()
        if parcels_proj.crs is None or parcels_proj.crs.to_string().startswith('EPSG:4326'):
            parcels_proj = parcels_proj.set_crs(epsg=4326, allow_override=True).to_crs(epsg=3857)
        else:
            parcels_proj = parcels_proj.to_crs(epsg=3857)
        if 'PARCEL_ID' in parcels_access_df.columns:
            parcels_proj = parcels_proj.merge(parcels_access_df[['PARCEL_ID', 'nearest_node_id']], on='PARCEL_ID', how='left')

    from scipy.spatial import cKDTree
    job_coords = np.vstack([jobs_proj.geometry.x.values, jobs_proj.geometry.y.values]).T
    tree = cKDTree(job_coords)

    speed_mps = (speed_kph * 1000.0) / 3600.0

    out_rows = []
    for _, p in parcels_proj.iterrows():
        px, py = p.geometry.x, p.geometry.y
        dists, idxs = tree.query([px, py], k=len(jobs_proj), distance_upper_bound=float('inf'))
        row = {'PARCEL_ID': p['PARCEL_ID']}
        for t in thresholds_s:
            max_m = t * speed_mps
            within = (dists <= max_m)
            count = int(within.sum())
            if weight_col in jobs_proj.columns:
                sum_jobs = float(jobs_proj.iloc[idxs[within]][weight_col].sum()) if count > 0 else 0.0
            else:
                sum_jobs = float(count)
            row[f'job_count_tt_{int(t)}'] = count
            row[f'job_sum_tt_{int(t)}'] = sum_jobs
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def compute_decay_job_accessibilities(nodes_gdf, edges_gdf, jobs_gdf, parcels_access_df, decay_type='exp', decay_param=100.0, thresholds_s=None, pandana_net=None, speed_kph=50, parcels_gdf=None, weight_col='jobs'):
    """Compute decay-weighted sums of jobs reachable from each parcel.

    Parameters:
    - decay_type: 'exp' for exponential decay (exp(-d/decay_param)), 'gauss' for Gaussian (exp(-d^2/(2*sigma^2))).
    - decay_param: parameter for the decay (alpha for exp, sigma for gauss) in meters (or seconds if using travel-time). If pandana network used, distances are interpreted in seconds and decay_param should be in seconds.
    - thresholds_s: optional list of thresholds (seconds) to limit contributions; if provided, only jobs within threshold are included.

    Prefers pandana shortest-path travel-times if a pandana network is available; otherwise falls back to Euclidean distances (meters) and converts via `speed_kph` to seconds when thresholds are used.

    Returns a DataFrame keyed by PARCEL_ID with a column `job_decay_sum_{decay_type}_{int(decay_param)}`.
    """
    # Ensure weight column exists
    jobs = jobs_gdf.copy()
    if weight_col not in jobs.columns:
        jobs[weight_col] = 1.0

    # Attempt pandana usage
    try:
        import pandana as pdna
    except Exception:
        pdna = None

    use_pandana = False
    net = None
    if pandana_net is not None and pdna is not None:
        use_pandana = True
        net = pandana_net
    else:
        if pdna is not None:
            try:
                net = try_build_pandana_network(nodes_gdf, edges_gdf)
                if net is not None:
                    use_pandana = True
            except Exception:
                use_pandana = False

    # Prepare parcels (merge nearest_node_id if needed)
    if hasattr(parcels_access_df, 'geometry'):
        parcels = parcels_access_df.copy()
    else:
        if parcels_gdf is None:
            raise ValueError('parcels_access_df lacks geometry; pass parcels_gdf GeoDataFrame')
        parcels = parcels_gdf.copy()
        if 'PARCEL_ID' in parcels_access_df.columns:
            parcels = parcels.merge(parcels_access_df[['PARCEL_ID', 'nearest_node_id']], on='PARCEL_ID', how='left')

    # If using pandana and net supports travel-time queries, attempt to compute node-to-node distances
    if use_pandana and net is not None:
        try:
            # Map jobs to node ids, parcels to node ids
            j = jobs.copy()
            if 'x' not in j.columns or 'y' not in j.columns:
                j['x'] = j.geometry.x
                j['y'] = j.geometry.y
            job_node_ids = net.get_node_ids_from_xy(j['x'].values, j['y'].values)

            p = parcels.copy()
            if 'x' not in p.columns or 'y' not in p.columns:
                if p.crs is None or p.crs.to_string().startswith('EPSG:4326'):
                    p = p.set_crs(epsg=4326, allow_override=True).to_crs(epsg=3857)
                p['x'] = p.geometry.x
                p['y'] = p.geometry.y
            parcel_node_ids = net.get_node_ids_from_xy(p['x'].values, p['y'].values)

            # If net.supports full shortest-path matrix, attempt per-parcel aggregates
            out_rows = []
            for pid, parcel_node in zip(p['PARCEL_ID'].values, parcel_node_ids):
                if parcel_node is None:
                    out_rows.append({'PARCEL_ID': pid, f'job_decay_sum_{decay_type}_{int(decay_param)}': 0.0})
                    continue
                # compute travel-times from parcel_node to all job nodes if supported
                try:
                    # pandana versions differ; try get_travel_time if available
                    if hasattr(net, 'get_travel_time'):
                        tt = net.get_travel_time(parcel_node, job_node_ids)
                        # tt is array-like of travel-times in net units
                        dists = np.array(tt, dtype=float)
                    else:
                        # fallback: use euclidean distance between coords of parcel and jobs
                        px = p.loc[p['PARCEL_ID'] == pid, 'x'].values[0]
                        py = p.loc[p['PARCEL_ID'] == pid, 'y'].values[0]
                        jx = j['x'].values
                        jy = j['y'].values
                        dists = np.sqrt((jx - px) ** 2 + (jy - py) ** 2)
                except Exception:
                    # fallback to euclidean
                    px = p.loc[p['PARCEL_ID'] == pid, 'x'].values[0]
                    py = p.loc[p['PARCEL_ID'] == pid, 'y'].values[0]
                    jx = j['x'].values
                    jy = j['y'].values
                    dists = np.sqrt((jx - px) ** 2 + (jy - py) ** 2)

                # If thresholds (seconds) provided but dists are meters, convert using speed if needed
                if thresholds_s is not None and use_pandana and hasattr(net, 'get_travel_time'):
                    # assume dists are in seconds already
                    pass
                elif thresholds_s is not None:
                    # convert meters to seconds
                    v_mps = (speed_kph * 1000.0) / 3600.0
                    dists = dists / v_mps

                # Apply threshold mask if provided
                mask = np.ones_like(dists, dtype=bool)
                if thresholds_s is not None:
                    max_t = max(thresholds_s)
                    mask = dists <= max_t

                # Apply decay
                if decay_type == 'exp':
                    weights = np.exp(-dists / float(decay_param))
                elif decay_type == 'gauss':
                    weights = np.exp(-0.5 * (dists / float(decay_param)) ** 2)
                else:
                    raise ValueError('Unsupported decay_type')

                # weight by jobs counts
                vals = (weights * j[weight_col].values) * mask
                total = float(np.sum(vals))
                out_rows.append({'PARCEL_ID': pid, f'job_decay_sum_{decay_type}_{int(decay_param)}': total})

            return pd.DataFrame(out_rows)
        except Exception:
            # fall through to Euclidean fallback
            pass

    # Euclidean fallback: compute centroid distances in metric CRS
    jobs_proj = jobs.copy()
    if jobs_proj.crs is None or jobs_proj.crs.to_string().startswith('EPSG:4326'):
        jobs_proj = jobs_proj.set_crs(epsg=4326, allow_override=True).to_crs(epsg=3857)
    else:
        jobs_proj = jobs_proj.to_crs(epsg=3857)

    parcels_proj = parcels.copy()
    if parcels_proj.crs is None or parcels_proj.crs.to_string().startswith('EPSG:4326'):
        parcels_proj = parcels_proj.set_crs(epsg=4326, allow_override=True).to_crs(epsg=3857)
    else:
        parcels_proj = parcels_proj.to_crs(epsg=3857)

    job_x = jobs_proj.geometry.x.values
    job_y = jobs_proj.geometry.y.values

    out_rows = []
    for _, p in parcels_proj.iterrows():
        px, py = p.geometry.x, p.geometry.y
        dists = np.sqrt((job_x - px) ** 2 + (job_y - py) ** 2)
        # If thresholds provided, convert dist to seconds
        if thresholds_s is not None:
            v_mps = (speed_kph * 1000.0) / 3600.0
            dists_seconds = dists / v_mps
            mask = dists_seconds <= (max(thresholds_s) if thresholds_s is not None else np.inf)
        else:
            mask = np.ones_like(dists, dtype=bool)

        if decay_type == 'exp':
            weights = np.exp(-dists / float(decay_param))
        elif decay_type == 'gauss':
            weights = np.exp(-0.5 * (dists / float(decay_param)) ** 2)
        else:
            raise ValueError('Unsupported decay_type')

        vals = (weights * jobs_proj[weight_col].values) * mask
        total = float(np.sum(vals))
        out_rows.append({'PARCEL_ID': p['PARCEL_ID'], f'job_decay_sum_{decay_type}_{int(decay_param)}': total})

    return pd.DataFrame(out_rows)


def compute_parcel_travel_time_to_nearest_poi(nodes_gdf, edges_gdf, pois_gdf, parcels_gdf, pandana_net=None, speed_kph=50):
    """Compute approximate travel time (seconds) from each parcel to its nearest POI.

    If `pandana_net` is provided and supports shortest-path queries, the function will attempt a network-based computation; otherwise it falls back to an Euclidean distance / speed approximation.

    Returns a DataFrame with columns `PARCEL_ID` and `tt_to_nearest_poi_s`.
    """
    # Delegate to Euclidean distance/time for now; pandana shortest-path could be added here later
    res = compute_walk_distance_and_time(parcels_gdf, pois_gdf, walk_speed_kph=speed_kph)
    return res[['PARCEL_ID', 'tt_to_nearest_poi_s']]


def compute_walk_distance_and_time(parcels_gdf, stops_gdf, walk_speed_kph=5, nodes_gdf=None, edges_gdf=None):
    """Compute Euclidean walk distance (meters) and walk time (seconds) from each parcel to nearest stop.

    If `nodes_gdf` and `edges_gdf` are provided and NetworkX is available, a network-based shortest-path
    computation will be used for walk time (using edge `travel_time_s` if present, otherwise computed from `walk_speed_kph`).

    Returns a DataFrame with `PARCEL_ID`, `walk_dist_to_stop_m`, and `walk_time_to_stop_s`.
    """
    # Try network-based shortest-path if nodes/edges supplied and networkx available
    if nodes_gdf is not None and edges_gdf is not None:
        try:
            import networkx as nx  # type: ignore
            return compute_walk_time_via_network(nodes_gdf, edges_gdf, stops_gdf, parcels_gdf, walk_speed_kph=walk_speed_kph)
        except Exception:
            # fallback to Euclidean if any error occurs
            pass

    # Euclidean fallback
    if not hasattr(parcels_gdf, 'geometry'):
        raise ValueError('parcels_gdf must be a GeoDataFrame with geometry')
    if not hasattr(stops_gdf, 'geometry'):
        raise ValueError('stops_gdf must be a GeoDataFrame with geometry')

    # Project to metric coordinates
    if parcels_gdf.crs is None or parcels_gdf.crs.to_string().startswith('EPSG:4326'):
        parcels = parcels_gdf.to_crs(epsg=3857).copy()
    else:
        parcels = parcels_gdf.copy()

    if stops_gdf.crs is None or stops_gdf.crs.to_string().startswith('EPSG:4326'):
        stops = stops_gdf.to_crs(epsg=3857).copy()
    else:
        stops = stops_gdf.copy()

    stop_coords = np.vstack([stops.geometry.x.values, stops.geometry.y.values]).T
    from scipy.spatial import cKDTree
    tree = cKDTree(stop_coords)

    parcel_centroids = np.vstack([parcels.geometry.centroid.x.values, parcels.geometry.centroid.y.values]).T
    dists, idx = tree.query(parcel_centroids, k=1)

    v_mps = walk_speed_kph * 1000.0 / 3600.0
    if v_mps <= 0:
        raise ValueError('walk_speed_kph must be positive')
    tt = dists / v_mps

    out = parcels[['PARCEL_ID']].copy()
    out['walk_dist_to_stop_m'] = dists
    out['walk_time_to_stop_s'] = tt
    # also provide a generic tt column name used elsewhere
    out['tt_to_nearest_poi_s'] = tt
    return out


def compute_walk_time_via_network(nodes_gdf, edges_gdf, stops_gdf, parcels_gdf, walk_speed_kph=5):
    """Compute walking travel time from parcels to nearest stop using a graph shortest-path on nodes/edges.

    Builds an undirected NetworkX graph weighted by `travel_time_s` (computed if missing using `walk_speed_kph`).
    Returns a DataFrame keyed by PARCEL_ID with `walk_time_to_stop_s` and `walk_dist_to_stop_m`.
    """
    try:
        import networkx as nx  # type: ignore
    except Exception:
        raise

    # Ensure projected coordinates for metric lengths
    if edges_gdf.crs is None or edges_gdf.crs.to_string().startswith('EPSG:4326'):
        edges_proj = edges_gdf.to_crs(epsg=3857).copy()
    else:
        edges_proj = edges_gdf.copy()

    if nodes_gdf.crs is None or nodes_gdf.crs.to_string().startswith('EPSG:4326'):
        nodes_proj = nodes_gdf.to_crs(epsg=3857).copy()
    else:
        nodes_proj = nodes_gdf.copy()

    # Ensure travel_time_s exists on edges; compute if missing using walk_speed_kph
    if 'travel_time_s' not in edges_proj.columns:
        edges_proj = add_travel_time_to_edges(edges_proj, speed_kph=walk_speed_kph)

    # Build graph
    G = nx.Graph()
    for i, row in edges_proj.iterrows():
        u = int(row['u'])
        v = int(row['v'])
        w = float(row['travel_time_s'])
        # add both directions
        G.add_edge(u, v, weight=w)

    # Map stops to nearest node
    node_coords = np.vstack([nodes_proj.geometry.x.values, nodes_proj.geometry.y.values]).T
    from scipy.spatial import cKDTree
    node_tree = cKDTree(node_coords)
    stop_coords = np.vstack([stops_gdf.to_crs(epsg=3857).geometry.x.values, stops_gdf.to_crs(epsg=3857).geometry.y.values]).T
    _, stop_node_idx = node_tree.query(stop_coords, k=1)
    sources = set(int(i) for i in stop_node_idx)

    # Compute multi-source shortest path lengths (in seconds)
    # Use networkx.multi_source_dijkstra_path_length
    dist_dict = nx.multi_source_dijkstra_path_length(G, sources, weight='weight')

    # For nodes not reachable, set distance to inf
    n_nodes = len(nodes_proj)
    node_tt = np.full(n_nodes, np.inf)
    for nid, d in dist_dict.items():
        node_tt[int(nid)] = d

    # Map to parcels using parcel nearest node (compute if not present)
    parcels = parcels_gdf.copy()
    if 'nearest_node_id' not in parcels.columns:
        # compute nearest node ids
        parcels_access = compute_nearest_node_distances(parcels, nodes_gdf)
        parcels = parcels.merge(parcels_access[['PARCEL_ID', 'nearest_node_id']], on='PARCEL_ID', how='left')

    # Map node travel times to parcels
    def map_tt(nid):
        try:
            nid = int(nid)
            val = node_tt[nid]
            if np.isfinite(val):
                return val
            else:
                return np.nan
        except Exception:
            return np.nan

    walk_time = parcels['nearest_node_id'].map(map_tt)

    # Convert to distance
    v_mps = walk_speed_kph * 1000.0 / 3600.0
    walk_dist = walk_time * v_mps

    out = parcels[['PARCEL_ID']].copy()
    out['walk_time_to_stop_s'] = walk_time
    out['walk_dist_to_stop_m'] = walk_dist
    out['tt_to_nearest_poi_s'] = walk_time
    return out

def main(roads_path=RAW_ROADS, parcels_path='data/processed/parcels_enriched.geojson'):
    if not Path(roads_path).exists():
        raise FileNotFoundError(f"Roads file not found: {roads_path}")
    logger.info(f"Loading roads from {roads_path}")
    roads = gpd.read_file(str(roads_path))

    nodes_gdf, edges_gdf = build_nodes_edges_from_roads(roads)

    nodes_gdf.to_file(OUT_DIR / 'network_nodes.geojson', driver='GeoJSON')
    edges_gdf.to_file(OUT_DIR / 'network_edges.geojson', driver='GeoJSON')
    logger.info(f"Wrote network nodes ({len(nodes_gdf)}) and edges ({len(edges_gdf)})")

    net = try_build_pandana_network(nodes_gdf, edges_gdf)
    if net is not None:
        logger.info("Pandana network built; ready for aggregation")

    # compute parcel distances
    parcels = gpd.read_file(str(parcels_path))
    accessibility = compute_nearest_node_distances(parcels, nodes_gdf)

    # optionally compute opportunity-based access measures
    # If `opportunities_path` is provided and exists, compute counts within radii.
    try:
        opportunities_path = kwargs.get('opportunities_path') if 'kwargs' in locals() else None
    except Exception:
        opportunities_path = None

    if opportunities_path and Path(opportunities_path).exists():
        opps = gpd.read_file(str(opportunities_path))
        access_df = compute_accessibilities(nodes_gdf, opps, accessibility, radii_meters=(400,800,1600))
        # join access columns back to parcels (by PARCEL_ID)
        parcels = parcels.merge(access_df, on='PARCEL_ID', how='left')
        parcels.to_file(OUT_DIR / 'parcels_enriched_with_access.geojson', driver='GeoJSON')
        logger.info("Wrote parcels_enriched_with_access.geojson")
        # write access subset as CSV
        access_df.to_csv(OUT_DIR / 'parcels_accessibility.csv', index=False)
        logger.info(f"Wrote parcels_accessibility.csv ({len(access_df)} rows)")
    else:
        accessibility.to_csv(OUT_DIR / 'parcels_accessibility.csv', index=False)
        logger.info(f"Wrote parcels_accessibility.csv ({len(accessibility)} rows)")

    return nodes_gdf, edges_gdf, accessibility


if __name__ == '__main__':
    main()

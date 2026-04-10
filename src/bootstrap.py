"""Orca bootstrap utilities."""
from pathlib import Path

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import orca
except Exception:
    orca = None


def setup_orca_from_zones(zones_geojson_path="data/processed/zones.json", id_prop="ZONE_ID"):
    """Register an orca table named 'zones' from a zones GeoJSON."""
    if gpd is None:
        raise RuntimeError("geopandas is required to run setup_orca_from_zones")
    if orca is None:
        raise RuntimeError("orca is required to run setup_orca_from_zones")

    path = Path(zones_geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"Zones GeoJSON not found: {path}")

    gdf = gpd.read_file(str(path))
    if id_prop not in gdf.columns:
        raise KeyError(f"Expected id property '{id_prop}' in zones file")

    df = gdf.drop(columns="geometry", errors="ignore").copy()
    df[id_prop] = df[id_prop].astype("int64")
    df = df.set_index(id_prop)
    orca.add_table("zones", df)
    return "zones"


def build_views_from_zones(
    zones_geojson_path="data/processed/zones.json",
    id_prop="ZONE_ID",
    join_name="zone_id",
):
    """Build a views dict suitable for dframe explorer."""
    if gpd is None:
        raise RuntimeError("geopandas is required to build views")

    path = Path(zones_geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"Zones GeoJSON not found: {path}")

    gdf = gpd.read_file(str(path))
    if id_prop not in gdf.columns:
        raise KeyError(f"Expected id property '{id_prop}' in zones file")

    gdf[id_prop] = gdf[id_prop].astype("int64")
    df = gdf.drop(columns="geometry", errors="ignore").copy()
    df = df.rename(columns={id_prop: join_name})
    return {"zones": df}

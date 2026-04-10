"""
GTFS Loading and Parsing for Bus Network
==========================================

Extracted from ``bus_network.py`` — pure structural refactoring, no
behaviour changes.  All functions that read GTFS text files and return
``BusRoute`` / ``ServiceProfile`` objects live here.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.bus_network import BusRoute, DEFAULT_BUS_SPEED_KPH, ServiceProfile
import logging

logger = logging.getLogger(__name__)

__all__ = ["load_bus_routes_from_gtfs"]


# ---------------------------------------------------------------------------
# GTFS Loader
# ---------------------------------------------------------------------------

def load_bus_routes_from_gtfs(
    gtfs_dir: Path,
    ridership_csvs: Optional[Sequence[Path]] = None,
    avg_bus_speed_kph: float = 20.0,
) -> List[BusRoute]:
    """Load per-route data from GTFS and optional ridership CSVs.

    Returns a list of BusRoute objects with baseline headways, lengths,
    cycle times, and observed ridership.
    """
    gtfs_dir = Path(gtfs_dir)

    routes_df = pd.read_csv(gtfs_dir / "routes.txt")
    trips_df = pd.read_csv(gtfs_dir / "trips.txt")
    stop_times_df = pd.read_csv(gtfs_dir / "stop_times.txt")

    # Filter to weekday service_ids only (Mon-Fri) using calendar.txt
    calendar_path = gtfs_dir / "calendar.txt"
    if calendar_path.exists() and "service_id" in trips_df.columns:
        cal = pd.read_csv(calendar_path)
        weekday_cols = ["monday", "tuesday", "wednesday", "thursday", "friday"]
        existing_cols = [c for c in weekday_cols if c in cal.columns]
        if existing_cols:
            # A service is weekday if it runs on any weekday
            weekday_mask = cal[existing_cols].sum(axis=1) > 0
            weekday_sids = set(cal.loc[weekday_mask, "service_id"].astype(str))
            if weekday_sids:
                trips_df = trips_df[trips_df["service_id"].astype(str).isin(weekday_sids)]
                # Re-filter stop_times to only include weekday trips
                stop_times_df = stop_times_df[
                    stop_times_df["trip_id"].isin(trips_df["trip_id"])
                ]

    # Route lengths from shapes.txt if available
    shapes_path = gtfs_dir / "shapes.txt"
    route_lengths_km: Dict[str, float] = {}
    if shapes_path.exists():
        shapes_df = pd.read_csv(shapes_path)
        if "shape_dist_traveled" in shapes_df.columns:
            shape_lens = shapes_df.groupby("shape_id")["shape_dist_traveled"].max()
        else:
            # Compute from lat/lon using Haversine
            shape_lens = _compute_shape_lengths(shapes_df)

        # Map shape_id → route_id via trips
        shape_route = trips_df[["route_id", "shape_id"]].drop_duplicates()
        for _, row in shape_route.iterrows():
            sid = row["shape_id"]
            rid = str(row["route_id"])
            if sid in shape_lens.index:
                dist = float(shape_lens[sid])
                # shape_dist_traveled may be in meters or km; normalize
                if dist > 500:  # likely meters
                    dist /= 1000.0
                route_lengths_km[rid] = max(route_lengths_km.get(rid, 0.0), dist)

    # Per-route headways from stop_times
    headways = _compute_route_headways(trips_df, stop_times_df)

    # Per-period service profiles from GTFS departure times
    service_profiles = _compute_route_service_profiles(
        trips_df, stop_times_df, headways,
    )

    # Per-route average speed from GTFS (shape_length / trip_runtime)
    route_speeds_kph: Dict[str, float] = {}
    try:
        # Trip runtime: last departure_time - first departure_time per trip
        st = stop_times_df.copy()
        st["time_s"] = st["departure_time"].apply(_parse_gtfs_time)
        trip_times = st.groupby("trip_id")["time_s"].agg(["min", "max"])
        trip_times["runtime_h"] = (trip_times["max"] - trip_times["min"]) / 3600.0
        trip_times = trip_times[trip_times["runtime_h"] > 0.01]  # skip <36s trips

        trip_route = trips_df[["trip_id", "route_id"]].drop_duplicates()
        trip_times = trip_times.merge(trip_route, left_index=True, right_on="trip_id")

        for rid_raw, grp in trip_times.groupby("route_id"):
            rid = str(rid_raw)
            length_km = route_lengths_km.get(rid, 0.0)
            if length_km > 0 and len(grp) > 0:
                median_runtime = float(grp["runtime_h"].median())
                if median_runtime > 0:
                    route_speeds_kph[rid] = length_km / median_runtime
    except Exception:
        pass  # speed computation is best-effort; fall back to default

    # Per-route stop counts (normalize keys to str for alphanumeric IDs like "4B")
    trip_stops = stop_times_df.merge(
        trips_df[["trip_id", "route_id"]], on="trip_id"
    )
    route_stop_counts = {
        str(k): v
        for k, v in trip_stops.groupby("route_id")["stop_id"].nunique().to_dict().items()
    }

    # Per-route trips per day (normalize keys to str)
    route_trip_counts = {
        str(k): v
        for k, v in trips_df.groupby("route_id")["trip_id"].count().to_dict().items()
    }

    # Load observed ridership
    daily_riders: Dict[str, float] = {}
    # Auto-discover ridership CSVs if none explicitly provided
    _csvs = list(ridership_csvs) if ridership_csvs else []
    if not _csvs:
        for p in sorted(gtfs_dir.glob("*ridership*.csv")):
            _csvs.append(p)
        if _csvs:
            logger.info(f"  GTFS: auto-discovered {len(_csvs)} ridership CSV(s)")
    if _csvs:
        for csv_path in _csvs:
            p = Path(csv_path)
            if not p.exists():
                continue
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            route_col = next(
                (c for c in ["Route", "route", "route_id"] if c in df.columns),
                None,
            )
            pax_col = next(
                (c for c in df.columns if "passenger" in c.lower()),
                None,
            )
            if route_col and pax_col:
                # Detect ridership period from magnitude:
                #   daily < 500 max/route, monthly 500-50K, annual > 50K
                _raw_vals = pd.to_numeric(df[pax_col], errors="coerce").dropna()
                _max_val = float(_raw_vals.max()) if len(_raw_vals) > 0 else 0.0
                if _max_val > 50_000:
                    _divisor = 260.0  # annual → daily weekday (52 weeks × 5 days)
                    _period = "annual"
                elif _max_val < 500:
                    _divisor = 1.0    # already daily
                    _period = "daily"
                else:
                    _divisor = 22.0   # monthly → daily weekday
                    _period = "monthly"

                logger.info(
                    f"  GTFS ridership: detected {_period} data "
                    f"(max={_max_val:.0f}, divisor={_divisor})"
                )

                for _, rr in df.iterrows():
                    rid = str(rr[route_col]).strip()
                    try:
                        val = float(rr[pax_col])
                    except (ValueError, TypeError):
                        continue
                    daily_riders[rid] = daily_riders.get(rid, 0.0) + val / _divisor

        # If multiple CSVs were loaded, average across them
        if len(_csvs) > 1:
            for rid in daily_riders:
                daily_riders[rid] /= len(_csvs)

    # Build BusRoute objects
    bus_routes: List[BusRoute] = []
    for _, row in routes_df.iterrows():
        rid = str(row["route_id"])
        rname = str(row.get("route_short_name", row.get("route_long_name", rid)))

        length_km = route_lengths_km.get(rid, 0.0)
        headway = headways.get(rid, 30.0)
        speed = route_speeds_kph.get(rid, avg_bus_speed_kph)
        n_stops = route_stop_counts.get(rid, 0)
        trips_day = route_trip_counts.get(rid, 0)
        riders = daily_riders.get(rid, 0.0)

        # Cycle time = round-trip travel + layover (10% buffer)
        if length_km > 0 and speed > 0:
            cycle_min = (length_km * 2.0 / speed) * 60.0 * 1.10
        else:
            cycle_min = headway * 2.0  # fallback

        profile = service_profiles.get(rid)

        route = BusRoute(
            route_id=rid,
            name=rname,
            length_km=length_km,
            baseline_headway_min=headway,
            current_headway_min=headway,
            cycle_time_min=round(cycle_min, 1),
            n_stops=n_stops,
            trips_per_day=float(trips_day),
            observed_daily_riders=riders,
            avg_speed_kph=speed,
            service_profile=profile,
        )
        bus_routes.append(route)

    logger.info(
        f"  GTFS: loaded {len(bus_routes)} routes, "
        f"{sum(1 for r in bus_routes if r.observed_daily_riders > 0)} with ridership"
    )
    return bus_routes


# ---------------------------------------------------------------------------
# Helper: Compute shape lengths from lat/lon points
# ---------------------------------------------------------------------------

def _compute_shape_lengths(shapes_df: pd.DataFrame) -> pd.Series:
    """Compute total length per shape_id from lat/lon sequences (meters).

    Returns a Series indexed by shape_id with total length in meters.
    """
    result: Dict[str, float] = {}
    for shape_id, grp in shapes_df.groupby("shape_id"):
        grp = grp.sort_values("shape_pt_sequence")
        lats = grp["shape_pt_lat"].values
        lons = grp["shape_pt_lon"].values
        total_m = 0.0
        for i in range(1, len(lats)):
            total_m += _haversine_m(lats[i - 1], lons[i - 1], lats[i], lons[i])
        result[shape_id] = total_m
    return pd.Series(result)


# ---------------------------------------------------------------------------
# Helper: Haversine distance in metres
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in metres."""
    R = 6_371_000.0  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ---------------------------------------------------------------------------
# Helper: Parse GTFS time strings (supports >24:00:00)
# ---------------------------------------------------------------------------

def _parse_gtfs_time(time_str: str) -> float:
    """Parse a GTFS time string like '25:30:00' into seconds since midnight."""
    try:
        parts = str(time_str).strip().split(":")
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
        return h * 3600.0 + m * 60.0 + s
    except (ValueError, IndexError):
        return float("nan")


# ---------------------------------------------------------------------------
# Helper: First departure per trip
# ---------------------------------------------------------------------------

def _first_departure_per_trip(stop_times_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns [trip_id, first_departure_s].

    Parses departure_time strings and keeps the earliest per trip.
    """
    st = stop_times_df[["trip_id", "departure_time"]].copy()
    st["dep_s"] = st["departure_time"].apply(_parse_gtfs_time)
    st = st.dropna(subset=["dep_s"])
    first = st.groupby("trip_id")["dep_s"].min().reset_index()
    first.columns = ["trip_id", "first_departure_s"]
    return first


# ---------------------------------------------------------------------------
# Helper: Compute average headway per route
# ---------------------------------------------------------------------------

def _compute_route_headways(
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute average weekday headway per route (minutes).

    Uses first-departure-per-trip, sorted chronologically, then takes
    median inter-departure gap.  Falls back to 30 min if insufficient data.
    """
    first_deps = _first_departure_per_trip(stop_times_df)
    merged = first_deps.merge(trips_df[["trip_id", "route_id"]], on="trip_id")

    headways: Dict[str, float] = {}
    for rid_raw, grp in merged.groupby("route_id"):
        rid = str(rid_raw)
        dep_times = np.sort(grp["first_departure_s"].values)
        if len(dep_times) < 2:
            headways[rid] = 60.0  # single-trip route fallback
            continue
        gaps = np.diff(dep_times) / 60.0  # minutes
        gaps = gaps[(gaps > 1.0) & (gaps < 180.0)]  # filter outliers
        if len(gaps) > 0:
            headways[rid] = float(np.median(gaps))
        else:
            headways[rid] = 30.0
    return headways


# ---------------------------------------------------------------------------
# Helper: Compute ServiceProfile objects with peak/offpeak headways
# ---------------------------------------------------------------------------

def _compute_route_service_profiles(
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    fallback_headways: Dict[str, float],
) -> Dict[str, ServiceProfile]:
    """Build per-route ServiceProfile from GTFS departure times.

    Time-of-day periods (weekday):
      AM peak   07:00-10:00 (25200-36000s)
      Midday    10:00-16:00 (36000-57600s)
      PM peak   16:00-19:00 (57600-68400s)
      Evening   19:00-22:00 (68400-79200s)

    Weekend headways are estimated from weekday as:
      Saturday = midday headway * 1.5
      Sunday   = midday headway * 2.0
    """
    PERIODS = {
        "am_peak":  (25200, 36000),
        "midday":   (36000, 57600),
        "pm_peak":  (57600, 68400),
        "evening":  (68400, 79200),
    }

    first_deps = _first_departure_per_trip(stop_times_df)
    merged = first_deps.merge(trips_df[["trip_id", "route_id"]], on="trip_id")

    profiles: Dict[str, ServiceProfile] = {}
    for rid_raw, grp in merged.groupby("route_id"):
        rid = str(rid_raw)
        dep_times = grp["first_departure_s"].values
        fallback = fallback_headways.get(rid, 30.0)

        period_headways: Dict[str, float] = {}
        for pname, (t_start, t_end) in PERIODS.items():
            mask = (dep_times >= t_start) & (dep_times < t_end)
            period_deps = np.sort(dep_times[mask])
            if len(period_deps) >= 2:
                gaps = np.diff(period_deps) / 60.0
                gaps = gaps[(gaps > 1.0) & (gaps < 180.0)]
                if len(gaps) > 0:
                    period_headways[pname] = float(np.median(gaps))
                else:
                    period_headways[pname] = fallback
            elif len(period_deps) == 1:
                # Single trip in period — estimate from period duration
                duration_h = (t_end - t_start) / 3600.0
                period_headways[pname] = duration_h * 60.0  # one trip = period-length headway
            else:
                period_headways[pname] = fallback * 2.0  # no service → double fallback

        midday_hw = period_headways.get("midday", fallback)
        profiles[rid] = ServiceProfile(
            am_peak_headway_min=round(period_headways.get("am_peak", fallback * 0.67), 1),
            midday_headway_min=round(midday_hw, 1),
            pm_peak_headway_min=round(period_headways.get("pm_peak", fallback * 0.75), 1),
            evening_headway_min=round(period_headways.get("evening", fallback * 1.5), 1),
            saturday_headway_min=round(midday_hw * 1.5, 1),
            sunday_headway_min=round(midday_hw * 2.0, 1),
        )

    return profiles
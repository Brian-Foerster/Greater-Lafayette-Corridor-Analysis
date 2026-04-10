#!/usr/bin/env python
"""Week 25 decision package generator (one-command final outputs)."""

from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.reproducibility import get_git_commit_short, utc_timestamp_compact

import logging
logger = logging.getLogger(__name__)



def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}")
    return pd.read_csv(path)


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def build_scenario_summary(zoning_df: pd.DataFrame, no_zoning_df: pd.DataFrame) -> pd.DataFrame:
    """Build scenario-level summary table for final recommendation package.

    Backward-compatible: accepts 2 positional DataFrames.  Use
    build_scenario_summary_multi() for N-scenario comparison.
    """
    return build_scenario_summary_multi({"zoning": zoning_df, "no_zoning": no_zoning_df})


def build_scenario_summary_multi(
    scenarios: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build scenario-level summary table for N scenarios.

    Parameters
    ----------
    scenarios : dict mapping scenario_name → DataFrame with corridor-level metrics.

    Returns summary rows + pairwise deltas for all scenario pairs.
    """
    _METRICS = [
        "mean_debt_coverage_ratio", "median_debt_coverage_ratio",
        "mean_daily_ridership", "mean_project_npv_dynamic_musd",
        "mean_tif_revenue_cumulative", "financially_viable_share",
        "mean_financial_viability_probability",
    ]

    rows = []
    for scenario_name, df in scenarios.items():
        debt = _safe_col(df, "debt_coverage_ratio")
        riders = _safe_col(df, "daily_ridership")
        npv = _safe_col(df, "project_npv_dynamic_musd")
        tif = _safe_col(df, "tif_revenue_cumulative")
        viability = _safe_col(df, "financially_viable")
        viability_prob = _safe_col(df, "financial_viability_probability", default=np.nan)
        rows.append(
            {
                "scenario": scenario_name,
                "corridor_count": int(len(df)),
                "mean_debt_coverage_ratio": float(debt.mean()),
                "median_debt_coverage_ratio": float(debt.median()),
                "mean_daily_ridership": float(riders.mean()),
                "mean_project_npv_dynamic_musd": float(npv.mean()),
                "mean_tif_revenue_cumulative": float(tif.mean()),
                "financially_viable_share": float(viability.mean()),
                "mean_financial_viability_probability": float(viability_prob.mean()) if viability_prob.notna().any() else np.nan,
            }
        )
    out = pd.DataFrame(rows)

    # Pairwise deltas for all scenario pairs
    names = list(scenarios.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = out[out["scenario"] == names[i]].iloc[0]
            b = out[out["scenario"] == names[j]].iloc[0]
            delta = {"scenario": f"delta_{names[i]}_minus_{names[j]}",
                     "corridor_count": int(a["corridor_count"] - b["corridor_count"])}
            for m in _METRICS:
                delta[m] = float(a[m] - b[m])
            out = pd.concat([out, pd.DataFrame([delta])], ignore_index=True)
    return out


def build_corridor_delta(zoning_df: pd.DataFrame, no_zoning_df: pd.DataFrame) -> pd.DataFrame:
    """Corridor-level zoning vs no-zoning comparison."""
    key_cols = ["corridor_id", "daily_ridership", "debt_coverage_ratio", "project_npv_dynamic_musd", "tif_revenue_cumulative"]
    for maybe in ["financial_viability_probability", "financially_viable", "rank"]:
        if maybe in zoning_df.columns:
            key_cols.append(maybe)
    z = zoning_df[[c for c in key_cols if c in zoning_df.columns]].copy()
    nz = no_zoning_df[[c for c in key_cols if c in no_zoning_df.columns]].copy()
    merged = z.merge(nz, on="corridor_id", suffixes=("_zoning", "_no_zoning"))

    for metric in ["daily_ridership", "debt_coverage_ratio", "project_npv_dynamic_musd", "tif_revenue_cumulative"]:
        if f"{metric}_zoning" in merged.columns and f"{metric}_no_zoning" in merged.columns:
            merged[f"{metric}_delta"] = merged[f"{metric}_zoning"] - merged[f"{metric}_no_zoning"]

    sort_col = "debt_coverage_ratio_delta" if "debt_coverage_ratio_delta" in merged.columns else "tif_revenue_cumulative_delta"
    merged = merged.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return merged


def select_recommendations(zoning_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Build recommendation table from zoning scenario."""
    df = zoning_df.copy()
    if "rank" in df.columns:
        df = df.sort_values("rank", ascending=True)
    elif "debt_coverage_ratio" in df.columns:
        df = df.sort_values("debt_coverage_ratio", ascending=False)
    cols = [
        "corridor_id",
        "rank",
        "daily_ridership",
        "debt_coverage_ratio",
        "tif_revenue_cumulative",
        "project_npv_dynamic_musd",
        "financially_viable",
        "financial_viability_probability",
    ]
    out_cols = [c for c in cols if c in df.columns]
    return df.head(int(top_n))[out_cols].reset_index(drop=True)


def build_equity_ranking(
    df: pd.DataFrame,
    financial_weight: float = 0.60,
    equity_weight: float = 0.40,
) -> pd.DataFrame:
    """Equity-weighted corridor ranking (Component 5, Addition 4).

    Computes a composite score = financial_weight × norm(DCR)
    + equity_weight × equity_score, where equity_score is composed of:
    - 0.30 × norm(equity_ratio)           # accessibility equity
    - 0.25 × norm(se01_riders_share)      # low-income ridership share
    - 0.25 × norm(-cumulative_displaced)  # displacement (lower = better)
    - 0.20 × norm(dac_benefit_share)      # Justice40 alignment

    Parameters
    ----------
    df : DataFrame with corridor-level metrics including equity columns.
    financial_weight : weight for financial score (default 0.60).
    equity_weight : weight for equity score (default 0.40).
    """
    result = df.copy()

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        rng = hi - lo
        if rng < 1e-12:
            return pd.Series(0.5, index=s.index)
        return (s - lo) / rng

    # Financial component
    dcr = _safe_col(result, "debt_coverage_ratio")
    financial_score = _norm(dcr)

    # Equity sub-components (all optional — use 0 if not available)
    _equity_cols = ["equity_ratio", "se01_riders_share", "cumulative_displaced", "dac_benefit_share"]
    _has_equity = any(c in result.columns and result[c].abs().sum() > 1e-12 for c in _equity_cols)
    if not _has_equity:
        import warnings
        warnings.warn(
            "No equity data found — equity component will be constant (0.5) "
            "across all corridors. Ranking is effectively financial-only.",
            stacklevel=2,
        )

    eq_ratio = _norm(_safe_col(result, "equity_ratio"))
    se01_share = _norm(_safe_col(result, "se01_riders_share"))
    displaced = _norm(-_safe_col(result, "cumulative_displaced"))  # lower is better
    dac_benefit = _norm(_safe_col(result, "dac_benefit_share"))

    equity_score = (
        0.30 * eq_ratio
        + 0.25 * se01_share
        + 0.25 * displaced
        + 0.20 * dac_benefit
    )

    composite = financial_weight * financial_score + equity_weight * equity_score

    result["financial_score"] = financial_score.round(4)
    result["equity_score"] = equity_score.round(4)
    result["composite_score"] = composite.round(4)
    result = result.sort_values("composite_score", ascending=False).reset_index(drop=True)
    result["composite_rank"] = range(1, len(result) + 1)

    return result


def _svg_header(width: int, height: int) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>"
    )


def _write_svg(path: Path, body: str, width: int = 900, height: int = 520) -> None:
    lines = [
        _svg_header(width, height),
        "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
        body,
        "</svg>",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_debt_coverage_distribution(z: pd.DataFrame, nz: pd.DataFrame, out_path: Path) -> None:
    width, height = 900, 520
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    zvals = _safe_col(z, "debt_coverage_ratio").to_numpy(dtype=float)
    nzvals = _safe_col(nz, "debt_coverage_ratio").to_numpy(dtype=float)
    bins = np.linspace(min(np.min(zvals), np.min(nzvals), 0.0), max(np.max(zvals), np.max(nzvals), 1.5), 16)
    zh, _ = np.histogram(zvals, bins=bins)
    nzh, _ = np.histogram(nzvals, bins=bins)
    maxh = max(float(np.max(zh)), float(np.max(nzh)), 1.0)

    body = []
    body.append(f"<text x='{margin}' y='32' font-size='20' font-family='Arial'>Debt Coverage Distribution by Scenario</text>")
    body.append(f"<line x1='{margin}' y1='{height-margin}' x2='{width-margin}' y2='{height-margin}' stroke='black'/>")
    body.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height-margin}' stroke='black'/>")
    bar_w = plot_w / (len(zh) * 2.2)
    for i in range(len(zh)):
        x_base = margin + (i + 0.1) * (plot_w / len(zh))
        h1 = (zh[i] / maxh) * plot_h
        h2 = (nzh[i] / maxh) * plot_h
        body.append(f"<rect x='{x_base:.1f}' y='{height-margin-h1:.1f}' width='{bar_w:.1f}' height='{h1:.1f}' fill='#2a9d8f' opacity='0.75'/>")
        body.append(f"<rect x='{x_base+bar_w+2:.1f}' y='{height-margin-h2:.1f}' width='{bar_w:.1f}' height='{h2:.1f}' fill='#e76f51' opacity='0.75'/>")

    # Viability threshold at x=1.0
    x_v = margin + ((1.0 - bins[0]) / max((bins[-1] - bins[0]), 1e-9)) * plot_w
    body.append(f"<line x1='{x_v:.1f}' y1='{margin}' x2='{x_v:.1f}' y2='{height-margin}' stroke='red' stroke-dasharray='4,4'/>")
    body.append(f"<text x='{width-240}' y='{margin+20}' font-size='13' font-family='Arial' fill='#2a9d8f'>zoning</text>")
    body.append(f"<text x='{width-240}' y='{margin+40}' font-size='13' font-family='Arial' fill='#e76f51'>no_zoning</text>")
    _write_svg(out_path, "\n".join(body), width=width, height=height)


def _plot_ridership_vs_coverage(z: pd.DataFrame, out_path: Path) -> None:
    width, height = 900, 520
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    x = _safe_col(z, "daily_ridership").to_numpy(dtype=float)
    y = _safe_col(z, "debt_coverage_ratio").to_numpy(dtype=float)
    viable = _safe_col(z, "financially_viable").to_numpy(dtype=float)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    body = []
    body.append(f"<text x='{margin}' y='32' font-size='20' font-family='Arial'>Zoning Scenario: Ridership vs Debt Coverage</text>")
    body.append(f"<line x1='{margin}' y1='{height-margin}' x2='{width-margin}' y2='{height-margin}' stroke='black'/>")
    body.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height-margin}' stroke='black'/>")
    for xi, yi, vi in zip(x, y, viable):
        sx = margin + ((xi - xmin) / (xmax - xmin)) * plot_w
        sy = height - margin - ((yi - ymin) / (ymax - ymin)) * plot_h
        color = "#2a9d8f" if vi >= 0.5 else "#e76f51"
        body.append(f"<circle cx='{sx:.1f}' cy='{sy:.1f}' r='5' fill='{color}' opacity='0.8'/>")
    yref = height - margin - ((1.0 - ymin) / (ymax - ymin)) * plot_h
    body.append(f"<line x1='{margin}' y1='{yref:.1f}' x2='{width-margin}' y2='{yref:.1f}' stroke='black' stroke-dasharray='4,4'/>")
    _write_svg(out_path, "\n".join(body), width=width, height=height)


def _plot_top_corridor_delta(delta_df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    d = delta_df.head(int(top_n)).copy()
    metric = "debt_coverage_ratio_delta" if "debt_coverage_ratio_delta" in d.columns else "tif_revenue_cumulative_delta"
    labels = d["corridor_id"].astype(str).tolist()
    vals = pd.to_numeric(d[metric], errors="coerce").fillna(0.0).to_numpy()
    width, height = 960, 520
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    vmax = max(float(np.max(np.abs(vals))), 1e-9)

    body = []
    body.append(f"<text x='{margin}' y='32' font-size='20' font-family='Arial'>Top Corridor Deltas ({metric})</text>")
    body.append(f"<line x1='{margin}' y1='{height-margin}' x2='{width-margin}' y2='{height-margin}' stroke='black'/>")
    body.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height-margin}' stroke='black'/>")
    n = max(len(vals), 1)
    bar_w = plot_w / (n * 1.4)
    for i, (lbl, v) in enumerate(zip(labels, vals)):
        x = margin + (i + 0.2) * (plot_w / n)
        h = (abs(v) / vmax) * (plot_h * 0.9)
        y = height - margin - h
        color = "#264653" if v >= 0 else "#b56576"
        body.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='{color}'/>")
        body.append(f"<text x='{x:.1f}' y='{height-margin+14}' font-size='10' font-family='Arial'>{lbl}</text>")
    _write_svg(out_path, "\n".join(body), width=width, height=height)


def write_report_markdown(
    out_path: Path,
    *,
    run_meta: Dict[str, str],
    scenario_summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    corridor_delta: pd.DataFrame,
    executive_summary_md: Optional[str] = None,
    uncertainty_dashboard: Optional[pd.DataFrame] = None,
) -> None:
    """Write final recommendation report markdown."""
    def _df_to_markdown(df: pd.DataFrame) -> str:
        if df is None or len(df) == 0:
            return "_No rows_"
        cols = [str(c) for c in df.columns]
        lines_local = []
        lines_local.append("| " + " | ".join(cols) + " |")
        lines_local.append("|" + "|".join(["---"] * len(cols)) + "|")
        for row in df.itertuples(index=False):
            vals = []
            for v in row:
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            lines_local.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines_local)

    lines = [
        "# Greater Lafayette APM Final Decision Package",
        "",
        f"- Generated UTC: `{run_meta['generated_at_utc']}`",
        f"- Git commit: `{run_meta['git_commit']}`",
        f"- Package run id: `{run_meta['run_id']}`",
        "",
    ]

    # Executive summary (structured, from build_executive_summary)
    if executive_summary_md:
        lines.append(executive_summary_md)
        lines.append("")
    else:
        top = recommendations.iloc[0] if len(recommendations) else None
        lines.append("## Executive Recommendation")
        lines.append("")
        if top is not None:
            lines.extend(
                [
                    f"- Recommended lead corridor: `{top['corridor_id']}`",
                    f"- Estimated debt coverage: `{float(top.get('debt_coverage_ratio', float('nan'))):.3f}x`",
                    f"- Estimated daily ridership: `{float(top.get('daily_ridership', float('nan'))):,.0f}`",
                ]
            )
        else:
            lines.append("- No corridor recommendation available (empty input).")

    lines.extend(
        [
            "",
            "## Scenario Summary Table",
            "",
            _df_to_markdown(scenario_summary),
            "",
            "## Top Recommendations (Zoning Scenario)",
            "",
            _df_to_markdown(recommendations),
            "",
        ]
    )

    # Uncertainty dashboard table
    if uncertainty_dashboard is not None and not uncertainty_dashboard.empty:
        lines.extend([
            "## Uncertainty Dashboard",
            "",
            _df_to_markdown(uncertainty_dashboard),
            "",
        ])

    lines.extend(
        [
            "## Largest Corridor-Level Deltas (Zoning - No Zoning)",
            "",
            _df_to_markdown(corridor_delta.head(15)),
            "",
            "## Generated Plots",
            "",
            "- `plots/debt_coverage_distribution.svg`",
            "- `plots/ridership_vs_coverage_zoning.svg`",
            "- `plots/top_corridor_delta.svg`",
            "",
            "## Corridor Maps",
            "",
            "See `maps/` directory for top corridor alignment maps with stations",
            "and walk catchment areas.",
            "",
            "## Technical Appendix",
            "",
            "See `TECHNICAL_APPENDIX.md` for auto-generated parameter tables,",
            "data sources, and model description.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_cba_artifacts(tables_dir: Path, zoning_df: pd.DataFrame) -> None:
    """Generate CBA / economic impact artifacts if feedback loop results exist.

    Looks for feedback_loop_results.csv and corridor metadata in standard
    locations.  If not found, skips silently — CBA is optional.
    """
    feedback_path = Path("data/processed/feedback_loop_results.csv")
    meta_path = Path("data/processed/apm_optimized_search_results.csv")

    if not feedback_path.exists() or not meta_path.exists():
        logger.info("[CBA] Skipping economic impact — feedback_loop_results.csv or corridor metadata not found")
        return

    try:
        from src.economic_impact import compute_corridor_cba, format_taxpayer_summary
    except ImportError as e:
        logger.info(f"[CBA] Skipping — import error: {e}")
        return

    logger.info("[CBA] Generating economic impact analysis...")
    feedback_df = pd.read_csv(feedback_path)
    meta_df = pd.read_csv(meta_path)

    # Normalize metadata columns
    col_map = {}
    for c in meta_df.columns:
        cl = c.lower().strip()
        if cl in ("id", "corridor_id"):
            col_map[c] = "corridor_id"
        elif cl in ("length", "length_km"):
            col_map[c] = "length_km"
        elif cl in ("stops", "n_stops", "stations"):
            col_map[c] = "n_stops"
    meta_df = meta_df.rename(columns=col_map)
    if "length_km" not in meta_df.columns or "n_stops" not in meta_df.columns:
        logger.info("[CBA] Skipping — corridor metadata missing length_km or n_stops columns")
        return
    if meta_df["length_km"].dtype == object:
        meta_df["length_km"] = meta_df["length_km"].str.replace("km", "").astype(float)

    corridor_ids = sorted(feedback_df["corridor_id"].unique())
    results = []
    for cid in corridor_ids:
        cid_str = str(cid)
        cdata = feedback_df[feedback_df["corridor_id"] == cid].copy()
        mrow = meta_df[meta_df["corridor_id"].astype(str) == cid_str]
        if cdata.empty or mrow.empty:
            continue
        if "year" not in cdata.columns:
            cdata["year"] = range(len(cdata))

        cba = compute_corridor_cba(
            results_df=cdata,
            corridor_id=cid_str,
            scenario="current_zoning",
            length_km=float(mrow.iloc[0]["length_km"]),
            n_stations=int(mrow.iloc[0]["n_stops"]),
            run_monte_carlo=True,
            n_mc_draws=200,
        )
        results.append(cba)

    if not results:
        logger.info("[CBA] No corridors processed")
        return

    # Write economic impact CSV
    summary_df = pd.DataFrame([r.to_dict() for r in results])
    summary_df.to_csv(tables_dir / "economic_impact_summary.csv", index=False)
    logger.info(f"[CBA] Written: {tables_dir / 'economic_impact_summary.csv'}")

    # Taxpayer summaries for top-5
    sorted_results = sorted(results, key=lambda r: r.bcr_3pct, reverse=True)
    summaries = [format_taxpayer_summary(r) for r in sorted_results[:5]]
    (tables_dir / "taxpayer_summary.txt").write_text(
        "\n\n".join(summaries), encoding="utf-8",
    )
    logger.info(f"[CBA] Written: {tables_dir / 'taxpayer_summary.txt'}")
    logger.info(f"[CBA] Done — {len(results)} corridors, best BCR={sorted_results[0].bcr_3pct:.2f}")


def _plot_corridor_maps(
    recommendations: pd.DataFrame,
    maps_dir: Path,
    corridor_geojson_path: Path = Path("data/processed/apm_phase2a_corridors.geojson"),
    top_n: int = 5,
) -> None:
    """Generate static map images for top corridors.

    Uses matplotlib + contextily for basemap tiles.  Falls back to plain
    matplotlib if contextily is unavailable or tiles fail to download.
    Each map shows the corridor alignment, station locations with labels,
    and 1200m walk catchment circles.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("[MAPS] Skipping — matplotlib not available")
        return

    try:
        import contextily as ctx
        has_contextily = True
    except ImportError:
        has_contextily = False
        logger.info("[MAPS] contextily not available — maps will have no basemap")

    if not corridor_geojson_path.exists():
        logger.info(f"[MAPS] Skipping — corridor GeoJSON not found: {corridor_geojson_path}")
        return

    # Load corridor geometries
    with open(corridor_geojson_path, encoding="utf-8") as f:
        corridors_gj = json.load(f)

    features_by_id = {}
    for feat in corridors_gj.get("features", []):
        cid = feat.get("properties", {}).get("corridor_id", "")
        if cid:
            features_by_id[cid] = feat

    top_cids = recommendations["corridor_id"].head(top_n).tolist()
    generated = 0

    for cid in top_cids:
        feat = features_by_id.get(str(cid))
        if feat is None:
            continue

        geom = feat.get("geometry", {})
        geom_type = geom.get("type", "")
        props = feat.get("properties", {})

        # Extract coordinates (handle LineString and MultiLineString)
        if geom_type == "LineString":
            coords = np.array(geom["coordinates"])
        elif geom_type == "MultiLineString":
            coords = np.array([c for line in geom["coordinates"] for c in line])
        else:
            continue

        if len(coords) < 2:
            continue

        lons = coords[:, 0]
        lats = coords[:, 1]

        # Extract station coords if available
        station_lons = props.get("station_lons", [])
        station_lats = props.get("station_lats", [])

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Plot corridor alignment
        ax.plot(lons, lats, color="#e63946", linewidth=3, zorder=5, label="APM Alignment")

        # Plot stations and walk catchments
        if station_lons and station_lats:
            for i, (slon, slat) in enumerate(zip(station_lons, station_lats)):
                # 1200m walk catchment ellipse (corrected for latitude distortion)
                # At lat ~40.4°: 1° lat ≈ 111km, 1° lon ≈ 111km × cos(40.4°) ≈ 84.5km
                import math
                radius_lat_deg = 1200.0 / 111_000.0
                radius_lon_deg = 1200.0 / (111_000.0 * math.cos(math.radians(40.4)))
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(
                    (slon, slat), width=2 * radius_lon_deg, height=2 * radius_lat_deg,
                    fill=True, facecolor="#457b9d", alpha=0.15,
                    edgecolor="#457b9d", linewidth=1, zorder=3,
                )
                ax.add_patch(ellipse)
                ax.plot(slon, slat, "o", color="#1d3557", markersize=8, zorder=6)
                ax.annotate(
                    f"S{i+1}", (slon, slat),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points",
                    zorder=7,
                )

        # Set extent with padding
        pad_lon = max((lons.max() - lons.min()) * 0.3, 0.01)
        pad_lat = max((lats.max() - lats.min()) * 0.3, 0.01)
        ax.set_xlim(lons.min() - pad_lon, lons.max() + pad_lon)
        ax.set_ylim(lats.min() - pad_lat, lats.max() + pad_lat)

        # Add basemap
        if has_contextily:
            try:
                ctx.add_basemap(
                    ax, crs="EPSG:4326",
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zoom=14,
                )
            except Exception:
                pass  # network error or tile issue — proceed without basemap

        # Title and labels
        length_km = props.get("length_km", "?")
        n_stops = props.get("n_stops", props.get("stations", "?"))
        ax.set_title(
            f"Corridor {cid}\n"
            f"Length: {length_km} km | Stations: {n_stops}",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

        out_path = maps_dir / f"corridor_{cid}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated += 1

    logger.info(f"[MAPS] Generated {generated} corridor maps in {maps_dir}")


def build_uncertainty_dashboard(
    uncertainty_wide_df: pd.DataFrame,
    robust_ranking_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build uncertainty dashboard table for the decision package.

    For each corridor, shows DCR percentile bands (p10/p50/p90), P(viable),
    and robust ranking metrics (P(top-5), downside DCR) if available.

    Parameters
    ----------
    uncertainty_wide_df : wide-format DataFrame from compute_uncertainty_bands()
        with columns like corridor_id, debt_coverage_ratio_p10, etc.
    robust_ranking_df : optional DataFrame from robust_corridor_ranking()
        with p_top_k and expected_shortfall_dcr columns.
    """
    if uncertainty_wide_df is None or uncertainty_wide_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in uncertainty_wide_df.iterrows():
        cid = row.get("corridor_id", "")
        entry = {
            "corridor_id": cid,
            "dcr_p10": round(float(row.get("debt_coverage_ratio_p10", 0)), 3),
            "dcr_p50": round(float(row.get("debt_coverage_ratio_p50", 0)), 3),
            "dcr_p90": round(float(row.get("debt_coverage_ratio_p90", 0)), 3),
            "p_viable": round(float(row.get("viability_probability", 0)), 3),
        }
        # Merge robust ranking if available
        if robust_ranking_df is not None and not robust_ranking_df.empty:
            rmatch = robust_ranking_df[
                robust_ranking_df["corridor_id"].astype(str) == str(cid)
            ]
            if not rmatch.empty:
                rrow = rmatch.iloc[0]
                entry["p_top5"] = round(float(rrow.get("p_top_k", 0)), 3)
                entry["downside_dcr"] = round(
                    float(rrow.get("expected_shortfall_dcr", 0)), 3
                )
                entry["robust_rank"] = int(rrow.get("robust_rank", 0))
        rows.append(entry)

    df = pd.DataFrame(rows)
    sort_col = "robust_rank" if "robust_rank" in df.columns else "dcr_p50"
    ascending = True if sort_col == "robust_rank" else False
    return df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)


def generate_technical_appendix(output_path: Path) -> None:
    """Auto-generate a technical appendix from code constants and module docstrings.

    Extracts calibrated constants from financial_params, mode_choice, and
    bus_network modules.  Includes data sources from source_manifest if available.
    """
    sections: List[str] = [
        "# Technical Appendix (Auto-Generated)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    # --- Model Description ---
    sections.append("## Model Description")
    sections.append("")
    sections.append(
        "This model evaluates APM (Automated People Mover) corridors using a "
        "dynamic feedback loop that couples land use, transport, and financial "
        "sub-models over a 25-year horizon.  Key sub-models include:"
    )
    sections.append("")
    sections.append("- **Mode choice**: 4-mode MNL (APM, bus, car, walk) with income-segmented coefficients")
    sections.append("- **Development model**: Relocation MNL + SqFtProForma with vacancy-rent feedback")
    sections.append("- **Bus network**: Budget-constrained headway optimization with productivity ranking")
    sections.append("- **Finance**: TIF-backed debt service with Monte Carlo uncertainty")
    sections.append("")

    # --- Parameter Table ---
    sections.append("## Calibrated Parameters")
    sections.append("")
    sections.append("| Module | Parameter | Value | Description |")
    sections.append("|---|---|---|---|")

    param_modules = [
        ("financial_params", "src.financial_params"),
        ("mode_choice", "src.mode_choice"),
        ("bus_network", "src.bus_network"),
    ]

    for display_name, module_path in param_modules:
        try:
            mod = __import__(module_path, fromlist=[""])
            members = inspect.getmembers(mod)
            for name, value in sorted(members):
                if name.startswith("_"):
                    continue
                if not isinstance(value, (int, float, str, bool)):
                    continue
                if callable(value):
                    continue
                # Skip very long strings
                val_str = str(value)
                if len(val_str) > 60:
                    val_str = val_str[:57] + "..."
                sections.append(f"| {display_name} | `{name}` | {val_str} | |")
        except ImportError:
            sections.append(f"| {display_name} | _(import failed)_ | | |")

    sections.append("")

    # --- Data Sources ---
    sections.append("## Data Sources")
    sections.append("")
    manifest_path = Path("data/raw/source_manifest.csv")
    if manifest_path.exists():
        try:
            manifest = pd.read_csv(manifest_path)
            sections.append("| Dataset | Source URL | Vintage | Status |")
            sections.append("|---|---|---|---|")
            for _, row in manifest.iterrows():
                did = row.get("dataset_id", "")
                url = row.get("source_url", "")
                vintage = row.get("data_vintage", "")
                status = row.get("status", "")
                sections.append(f"| {did} | {url} | {vintage} | {status} |")
        except Exception:
            sections.append("_(source manifest could not be parsed)_")
    else:
        sections.append("_(source manifest not found at data/raw/source_manifest.csv)_")
    sections.append("")

    # --- Known Limitations ---
    sections.append("## Known Limitations")
    sections.append("")
    weakness_path = Path("docs/model_weakness_triage.md")
    if weakness_path.exists():
        try:
            content = weakness_path.read_text(encoding="utf-8")
            # Extract first 50 lines as summary
            lines = content.strip().split("\n")[:50]
            sections.extend(lines)
        except Exception:
            sections.append("_(could not read model weakness triage)_")
    else:
        sections.append(
            "- Online-data-only constraint limits behavioral calibration precision\n"
            "- No revealed preference survey data for mode choice validation\n"
            "- LODES commute data is 2-3 years lagged\n"
            "- Property tax assessment values may differ from market values\n"
            "- Student population modeled via enrollment formula, not household survey"
        )
    sections.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    logger.info(f"[APPENDIX] Written: {output_path}")


def build_executive_summary(
    recommendations: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    uncertainty_dashboard: Optional[pd.DataFrame] = None,
    equity_ranking: Optional[pd.DataFrame] = None,
) -> str:
    """Generate a structured executive summary as markdown text.

    Includes lead recommendation, key numbers, scenario sensitivity,
    risk statement, equity statement, and next steps.
    """
    lines = [
        "## Executive Summary",
        "",
    ]
    _eq_filtered = None

    # 1. Lead recommendation
    if len(recommendations) > 0:
        top = recommendations.iloc[0]
        cid = top.get("corridor_id", "unknown")
        dcr = float(top.get("debt_coverage_ratio", 0))
        riders = float(top.get("daily_ridership", 0))

        # Use composite ranking if available, filtered to recommended corridors
        rank_source = "DCR"
        if equity_ranking is not None and not equity_ranking.empty:
            rec_ids = set(recommendations["corridor_id"]) if "corridor_id" in recommendations.columns else set()
            _eq_filtered = equity_ranking[equity_ranking["corridor_id"].isin(rec_ids)] if rec_ids else equity_ranking
            if not _eq_filtered.empty:
                eq_top = _eq_filtered.iloc[0]
                cid = eq_top.get("corridor_id", cid)
                dcr = float(eq_top.get("debt_coverage_ratio", dcr))
                riders = float(eq_top.get("daily_ridership", riders))
                rank_source = "composite (60% financial / 40% equity)"

        lines.append(f"**Lead recommendation:** Corridor `{cid}` ranks first by {rank_source} score")
        lines.append(f"with an estimated {riders:,.0f} daily riders and {dcr:.3f}x debt coverage ratio.")
        lines.append("")
    else:
        lines.append("**Lead recommendation:** No corridor data available.")
        lines.append("")

    # 2. Key numbers table (top 3) — always from recommended corridors
    if _eq_filtered is not None and not _eq_filtered.empty:
        top3 = _eq_filtered.head(3)
    else:
        top3 = recommendations.head(3)
    if len(top3) > 0:
        lines.append("### Key Metrics (Top 3 Corridors)")
        lines.append("")
        lines.append("| Corridor | Ridership | DCR | TIF Revenue ($M) |")
        lines.append("|---|---|---|---|")
        for _, row in top3.iterrows():
            cid = row.get("corridor_id", "")
            riders = float(row.get("daily_ridership", 0))
            dcr = float(row.get("debt_coverage_ratio", 0))
            tif = float(row.get("tif_revenue_cumulative", 0)) / 1e6
            lines.append(f"| {cid} | {riders:,.0f} | {dcr:.3f}x | {tif:.1f} |")
        lines.append("")

    # 3. Scenario sensitivity
    if scenario_summary is not None and len(scenario_summary) > 0:
        base_scenarios = scenario_summary[
            ~scenario_summary["scenario"].str.startswith("delta_")
        ]
        if len(base_scenarios) >= 2:
            lines.append("### Scenario Sensitivity")
            lines.append("")
            for _, srow in base_scenarios.iterrows():
                sname = srow.get("scenario", "")
                mean_riders = float(srow.get("mean_daily_ridership", 0))
                mean_dcr = float(srow.get("mean_debt_coverage_ratio", 0))
                lines.append(
                    f"- **{sname}**: mean {mean_riders:,.0f} riders/day, "
                    f"mean DCR {mean_dcr:.3f}x"
                )
            lines.append("")

    # 4. Risk statement
    if uncertainty_dashboard is not None and not uncertainty_dashboard.empty:
        lines.append("### Risk Profile")
        lines.append("")
        top_row = uncertainty_dashboard.iloc[0]
        p_viable = float(top_row.get("p_viable", 0))
        dcr_p10 = float(top_row.get("dcr_p10", 0))
        dcr_p50 = float(top_row.get("dcr_p50", 0))
        top_cid = top_row.get("corridor_id", "top corridor")
        lines.append(
            f"The top corridor (`{top_cid}`) has a {p_viable:.0%} probability of "
            f"achieving financial viability (DCR >= 1.0). "
            f"Median DCR is {dcr_p50:.3f}x; downside (p10) is {dcr_p10:.3f}x."
        )
        lines.append("")

    # 5. Equity statement
    if equity_ranking is not None and not equity_ranking.empty:
        lines.append("### Equity")
        lines.append("")
        eq_score = float(equity_ranking.iloc[0].get("equity_score", 0))
        lines.append(
            f"The top corridor achieves an equity score of {eq_score:.3f} "
            f"(normalized 0-1 scale combining accessibility equity, low-income "
            f"ridership share, displacement risk, and Justice40 alignment)."
        )
        lines.append("")

    # 6. Next steps
    lines.extend([
        "### Recommended Next Steps",
        "",
        "1. **Preliminary Engineering**: Commission PE study for top 2-3 corridors",
        "2. **Environmental Review**: Initiate NEPA categorical exclusion or EA",
        "3. **Public Engagement**: Conduct community meetings along recommended alignment",
        "4. **FTA Coordination**: Submit project profile for Small Starts evaluation",
        "5. **Funding Strategy**: Develop multi-source financing plan (TIF + federal + state)",
        "",
    ])

    return "\n".join(lines)


def build_decision_package(
    *,
    zoning_path: Path,
    no_zoning_path: Path,
    output_root: Path,
    top_n: int,
    scenario_paths: Optional[Dict[str, Path]] = None,
    model_options: Optional[Dict[str, object]] = None,
) -> Tuple[Path, Dict[str, str]]:
    """Generate all Week 25 package artifacts and return package dir + metadata.

    Parameters
    ----------
    zoning_path, no_zoning_path : backward-compatible 2-scenario inputs.
    scenario_paths : optional dict mapping scenario_name → CSV path.
        When provided, uses N-scenario comparison instead of just 2 scenarios.
        Falls back to {zoning, no_zoning} from positional args when not given.
    """
    ts = utc_timestamp_compact()
    git_commit = get_git_commit_short()
    run_id = f"decision_pkg_v1_{ts}_{git_commit}"
    package_dir = output_root / run_id
    plots_dir = package_dir / "plots"
    tables_dir = package_dir / "tables"
    package_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Build scenario dict: use explicit scenario_paths if given, else 2-scenario fallback
    if scenario_paths:
        all_scenarios = {name: _load_csv(p) for name, p in scenario_paths.items()}
    else:
        all_scenarios = {
            "zoning": _load_csv(zoning_path),
            "no_zoning": _load_csv(no_zoning_path),
        }

    # Pick primary scenario DataFrame (zoning > current_zoning > first available)
    if "zoning" in all_scenarios:
        zoning_df = all_scenarios["zoning"]
    elif "current_zoning" in all_scenarios:
        zoning_df = all_scenarios["current_zoning"]
    else:
        zoning_df = next(iter(all_scenarios.values()))
    no_zoning_df = all_scenarios.get("no_zoning", zoning_df)

    scenario_summary = build_scenario_summary_multi(all_scenarios)
    corridor_delta = build_corridor_delta(zoning_df, no_zoning_df)
    recommendations = select_recommendations(zoning_df, top_n=top_n)

    scenario_summary.to_csv(tables_dir / "scenario_summary.csv", index=False)
    corridor_delta.to_csv(tables_dir / "corridor_scenario_delta.csv", index=False)
    recommendations.to_csv(tables_dir / "top_recommendations.csv", index=False)

    _plot_debt_coverage_distribution(zoning_df, no_zoning_df, plots_dir / "debt_coverage_distribution.svg")
    _plot_ridership_vs_coverage(zoning_df, plots_dir / "ridership_vs_coverage_zoning.svg")
    _plot_top_corridor_delta(corridor_delta, plots_dir / "top_corridor_delta.svg")

    # --- Corridor maps for top recommendations (gated by config flag) ---
    _opts = model_options or {}
    if _opts.get("decision_package_maps", False):
        maps_dir = package_dir / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        _plot_corridor_maps(recommendations, maps_dir, top_n=min(top_n, 5))

    # --- Equity ranking (if equity columns present) ---
    equity_ranking = None
    equity_cols = {"equity_ratio", "se01_riders_share", "cumulative_displaced", "dac_benefit_share"}
    if equity_cols.intersection(zoning_df.columns):
        equity_ranking = build_equity_ranking(zoning_df)
        equity_ranking.to_csv(tables_dir / "equity_ranking.csv", index=False)

    # --- Uncertainty dashboard (if percentile columns present) ---
    uncertainty_dashboard = None
    if "debt_coverage_ratio_p50" in zoning_df.columns:
        uncertainty_dashboard = build_uncertainty_dashboard(zoning_df)
        uncertainty_dashboard.to_csv(tables_dir / "uncertainty_dashboard.csv", index=False)

    # --- Executive summary ---
    exec_summary = build_executive_summary(
        recommendations=recommendations,
        scenario_summary=scenario_summary,
        uncertainty_dashboard=uncertainty_dashboard,
        equity_ranking=equity_ranking,
    )

    run_meta = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "zoning_input": str(zoning_path),
        "no_zoning_input": str(no_zoning_path),
    }
    with (package_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    write_report_markdown(
        package_dir / "FINAL_DECISION_PACKAGE.md",
        run_meta=run_meta,
        scenario_summary=scenario_summary,
        recommendations=recommendations,
        corridor_delta=corridor_delta,
        executive_summary_md=exec_summary,
        uncertainty_dashboard=uncertainty_dashboard,
    )

    # --- Economic impact / CBA (if feedback loop results available) ---
    _generate_cba_artifacts(tables_dir, zoning_df)

    # --- FTA cost-effectiveness table (gated by config flag) ---
    if _opts.get("fta_cost_effectiveness", False):
        try:
            from src.economic_impact import build_fta_metrics_table
            fta_table = build_fta_metrics_table(zoning_df)
            if fta_table is not None and not fta_table.empty:
                fta_table.to_csv(tables_dir / "fta_cost_effectiveness.csv", index=False)
                logger.info(f"[FTA] Written: {tables_dir / 'fta_cost_effectiveness.csv'}")
        except (ImportError, Exception) as e:
            logger.info(f"[FTA] Skipping cost-effectiveness table: {e}")

    # --- CEJST / EJ community screening (gated by cejst_tracts_path) ---
    _cejst_path = _opts.get("cejst_tracts_path")
    if _cejst_path and Path(_cejst_path).exists():
        try:
            from src.economic_impact import screen_ej_communities
            cejst_df = pd.read_csv(_cejst_path)
            logger.info(f"[EJ] CEJST tracts loaded: {len(cejst_df)} tracts from {_cejst_path}")
            # Screening results would be computed per-corridor in the evaluation
            # script; here we just note availability for the report.
        except (ImportError, Exception) as e:
            logger.info(f"[EJ] Skipping CEJST screening: {e}")

    # --- Auto-generated technical appendix (gated by config flag) ---
    if _opts.get("technical_appendix_auto", False):
        generate_technical_appendix(package_dir / "TECHNICAL_APPENDIX.md")

    # Also publish a stable pointer to latest.
    latest_dir = output_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    with (latest_dir / "LATEST_DECISION_PACKAGE_PATH.txt").open("w", encoding="utf-8") as f:
        f.write(str(package_dir) + "\n")

    return package_dir, run_meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate final Week 25 decision package")
    parser.add_argument(
        "--zoning-input",
        type=str,
        default="data/processed/corridors_integrated_financial_zoning_with_uncertainty.csv",
    )
    parser.add_argument(
        "--no-zoning-input",
        type=str,
        default="data/processed/corridors_integrated_financial_no_zoning_with_uncertainty.csv",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/processed/decision_package",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--scenario",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        help="Additional scenario: --scenario no_zoning path/to/nz.csv (repeatable)",
    )
    parser.add_argument(
        "--maps", action="store_true",
        help="Generate corridor alignment maps (requires matplotlib + contextily)",
    )
    parser.add_argument(
        "--appendix", action="store_true",
        help="Generate auto-extracted technical appendix",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scenarios_config.json",
        help="Scenario config JSON path (reads model_options for feature flags)",
    )
    args = parser.parse_args()

    # Build scenario_paths dict from --scenario args
    scenario_paths = None
    if args.scenario:
        scenario_paths = {}
        for name, path in args.scenario:
            scenario_paths[name] = Path(path)
        # Also include the positional zoning/no_zoning if not already covered
        if "zoning" not in scenario_paths and "current_zoning" not in scenario_paths:
            scenario_paths["zoning"] = Path(args.zoning_input)
        if "no_zoning" not in scenario_paths:
            scenario_paths["no_zoning"] = Path(args.no_zoning_input)

    # Load model_options from config, merge CLI flags
    _pkg_opts: Dict[str, object] = {}
    _cfg_path = Path(args.config)
    if _cfg_path.exists():
        with open(_cfg_path, encoding="utf-8") as _f:
            _cfg = json.load(_f)
            _pkg_opts = dict(_cfg.get("model_options", {}))
    if args.maps:
        _pkg_opts["decision_package_maps"] = True
    if args.appendix:
        _pkg_opts["technical_appendix_auto"] = True

    package_dir, meta = build_decision_package(
        zoning_path=Path(args.zoning_input),
        no_zoning_path=Path(args.no_zoning_input),
        output_root=Path(args.output_root),
        top_n=max(int(args.top_n), 1),
        scenario_paths=scenario_paths,
        model_options=_pkg_opts,
    )
    logger.info(f"[DONE] Decision package generated: {package_dir}")
    logger.info(f"[DONE] Report: {package_dir / 'FINAL_DECISION_PACKAGE.md'}")
    logger.info(f"[DONE] Run id: {meta['run_id']}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())

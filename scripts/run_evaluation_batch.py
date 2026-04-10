"""Batch runners for financial and network-aware corridor optimization workflows.

Week 13 additions:
- Batch orchestration for isolated vs network-aware optimization runs.
- Versioned artifact naming conventions and run metadata manifests.
- Ranking-delta comparison outputs across evaluation modes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import pandas as pd

from src.source_manifest import validate_source_manifest_file

import logging
logger = logging.getLogger(__name__)


DEFAULT_NETWORK_BATCH_DIR = Path("data/processed/network_batches")
DEFAULT_FINANCIAL_BATCH_DIR = Path("data/processed/financial_batches")


def utc_timestamp_compact(now: Optional[datetime] = None) -> str:
    """Return compact UTC timestamp token (YYYYMMDDTHHMMSSZ)."""
    ts = now or datetime.now(timezone.utc)
    return ts.strftime("%Y%m%dT%H%M%SZ")


def get_git_commit_short() -> str:
    """Best-effort short git hash for artifact provenance."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            val = res.stdout.strip()
            if val:
                return val
    except Exception:
        pass
    return "nogit"


def build_batch_tag(
    *,
    iterations: int,
    population: int,
    n_output: int,
    timestamp_token: str,
    git_commit: str,
) -> str:
    """Build normalized batch tag shared by all run artifacts."""
    return (
        f"batch_v1_{timestamp_token}_it{int(iterations)}_"
        f"pop{int(population)}_out{int(n_output)}_{git_commit}"
    )


def build_mode_run_tag(mode: str, mode_version: str = "v1") -> str:
    """Build normalized mode run tag."""
    mode_norm = str(mode).strip().lower()
    return f"{mode_norm}_{mode_version}"


def normalize_corridor_ranking(corridors_df: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic corridor rankings from a scored corridor table."""
    if "corridor_id" not in corridors_df.columns or "ridership_est" not in corridors_df.columns:
        raise ValueError("corridors_df must contain corridor_id and ridership_est")

    out = corridors_df.copy()
    out["corridor_id"] = out["corridor_id"].astype(str)
    out["ridership_est"] = pd.to_numeric(out["ridership_est"], errors="coerce").fillna(0.0)
    out = out.sort_values(
        ["ridership_est", "corridor_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    out["rank"] = out.index + 1
    cols = [
        "corridor_id",
        "rank",
        "ridership_est",
        "length_km",
        "n_stops",
        "evaluation_mode",
        "network_synergy",
        "ridership_est_base",
        "ridership_network_adjusted",
    ]
    keep = [c for c in cols if c in out.columns]
    return out[keep].copy()


def compute_ranking_deltas(
    isolated_rank_df: pd.DataFrame,
    network_rank_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute rank/ridership deltas between isolated and network-aware runs."""
    req_cols = {"corridor_id", "rank", "ridership_est"}
    if not req_cols.issubset(set(isolated_rank_df.columns)):
        raise ValueError("isolated_rank_df missing required columns")
    if not req_cols.issubset(set(network_rank_df.columns)):
        raise ValueError("network_rank_df missing required columns")

    iso = isolated_rank_df[["corridor_id", "rank", "ridership_est"]].copy()
    net = network_rank_df[["corridor_id", "rank", "ridership_est"]].copy()
    merged = iso.merge(net, on="corridor_id", suffixes=("_isolated", "_network_aware"))
    merged["rank_delta"] = merged["rank_network_aware"] - merged["rank_isolated"]
    merged["rank_delta_abs"] = merged["rank_delta"].abs()
    merged["ridership_delta"] = (
        merged["ridership_est_network_aware"] - merged["ridership_est_isolated"]
    )
    denom = merged["ridership_est_isolated"].replace(0, pd.NA)
    merged["ridership_delta_pct"] = (merged["ridership_delta"] / denom) * 100.0
    merged["direction"] = merged["rank_delta"].map(
        lambda d: "up" if d < 0 else ("down" if d > 0 else "unchanged")
    )
    merged = merged.sort_values(
        ["rank_delta_abs", "ridership_delta"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return merged


def _parse_corridor_num(value: str | int) -> int:
    """Parse corridor number from int or C-prefixed string."""
    if isinstance(value, int):
        return int(value)
    return int(str(value).strip().upper().replace("C", ""))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _artifact_manifest_rows(run_dir: Path, run_meta: dict) -> List[dict]:
    rows = []
    created = datetime.now(timezone.utc).isoformat()
    for p in sorted(run_dir.glob("*")):
        if p.is_dir():
            continue
        row_count = None
        if p.suffix.lower() == ".csv":
            try:
                # Count lines without parsing full CSV into DataFrame
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    row_count = sum(1 for _ in fh) - 1  # subtract header
                if row_count < 0:
                    row_count = 0
            except Exception:
                row_count = None
        rows.append(
            {
                "run_tag": run_meta.get("run_tag"),
                "batch_tag": run_meta.get("batch_tag"),
                "evaluation_mode": run_meta.get("evaluation_mode"),
                "artifact_name": p.name,
                "relative_path": str(p),
                "file_format": p.suffix.lower().replace(".", ""),
                "row_count": row_count,
                "created_utc": created,
                "version": "v1",
                "git_commit": run_meta.get("git_commit"),
            }
        )
    return rows


def run_financial_batch_evaluation(
    start_corridor: str | int,
    end_corridor: str | int,
    *,
    scenario: str = "current_zoning",
    demand_results_path: Path = Path("data/processed/phase2b_corridor_results.csv"),
    phase2a_path: Path = Path("data/processed/apm_phase2a_results.csv"),
    output_dir: Path = DEFAULT_FINANCIAL_BATCH_DIR,
) -> Optional[pd.DataFrame]:
    """Evaluate a financial batch and archive result with versioned naming.

    Capital cost uses MECE decomposition (compute_capital_cost) — not a flat
    per-km rate.  See src/financial_params.py for cost breakdown.
    """
    from scripts.apm_corridor_evaluation_integrated import evaluate_corridor_with_financial_analysis

    start_num = _parse_corridor_num(start_corridor)
    end_num = _parse_corridor_num(end_corridor)

    logger.info(f"\n{'=' * 78}")
    logger.info(f"FINANCIAL BATCH EVALUATION: C{start_num} to C{end_num}")
    logger.info(f"Scenario: {scenario}")
    logger.info(f"{'=' * 78}\n")

    demand_df = pd.read_csv(demand_results_path)
    phase2a_df = pd.read_csv(phase2a_path)
    demand_df = demand_df.merge(
        phase2a_df[["corridor_id", "length_km"]],
        on="corridor_id",
        how="left",
    )

    corridors_to_eval = [f"C{i}" for i in range(start_num, end_num + 1)]
    batch_df = demand_df[demand_df["corridor_id"].isin(corridors_to_eval)].copy()
    logger.info(f"Found {len(batch_df)} corridors in batch\n")

    results = []
    for _, row in batch_df.iterrows():
        corridor_id = row["corridor_id"]
        daily_ridership = row["phase2b_daily_riders"]
        length_km = row["length_km"]
        try:
            result = evaluate_corridor_with_financial_analysis(
                corridor_id=corridor_id,
                daily_ridership=daily_ridership,
                length_km=length_km,
                scenario=scenario,
                use_property_tax_model=False,
            )
            results.append(result)
            viable = "YES" if result.get("financially_viable", False) else "NO"
            logger.info(f"OK {corridor_id}: {result['debt_coverage_ratio']:.2f}x coverage - {viable}")
        except Exception as exc:
            logger.info(f"ERR {corridor_id}: {exc}")

    if not results:
        logger.info("No results generated")
        return None

    results_df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_timestamp_compact()
    git_commit = get_git_commit_short()
    out_name = (
        f"financial_batch_v1_C{start_num:02d}_C{end_num:02d}_"
        f"{scenario}_{ts}_{git_commit}.csv"
    )
    out_path = output_dir / out_name
    results_df.to_csv(out_path, index=False)

    meta = {
        "workflow": "financial",
        "version": "v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "start_corridor": f"C{start_num}",
        "end_corridor": f"C{end_num}",
        "scenario": scenario,
        "result_path": str(out_path),
    }
    _write_json(out_path.with_suffix(".meta.json"), meta)

    viable_count = int((results_df["financially_viable"] == True).sum())  # noqa: E712
    logger.info(f"\nSaved financial batch: {out_path}")
    logger.info(f"Viable corridors: {viable_count}/{len(results_df)}")
    return results_df


def run_optimization_batch(
    *,
    evaluation_modes: Iterable[str],
    n_iterations: int,
    population_size: int,
    n_output: int,
    use_network: bool,
    network_synergy_weight: float,
    network_anchor_top_k: int,
    network_transfer_radius_m: float,
    output_root: Path = DEFAULT_NETWORK_BATCH_DIR,
    validate_manifest: bool = True,
    source_manifest_path: Path = Path("data/processed/source_manifest.csv"),
) -> Dict[str, dict]:
    """Run isolated/network-aware optimization modes and archive versioned artifacts."""
    from optimized_corridor_search import load_all_data, run_optimized_search

    modes = [str(m).strip().lower() for m in evaluation_modes]
    allowed = {"isolated", "network_aware"}
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise ValueError(f"Unsupported evaluation modes: {bad}")
    if not modes:
        raise ValueError("No evaluation modes requested")

    if validate_manifest:
        validate_source_manifest_file(source_manifest_path, max_source_age_days=3650)
        logger.info(f"Source manifest validated: {source_manifest_path}")

    timestamp_token = utc_timestamp_compact()
    git_commit = get_git_commit_short()
    batch_tag = build_batch_tag(
        iterations=n_iterations,
        population=population_size,
        n_output=n_output,
        timestamp_token=timestamp_token,
        git_commit=git_commit,
    )
    batch_dir = output_root / batch_tag
    batch_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 78}")
    logger.info("NETWORK-AWARE OPTIMIZATION BATCH")
    logger.info(f"Batch tag: {batch_tag}")
    logger.info(f"Modes: {modes}")
    logger.info(f"{'=' * 78}\n")

    data = load_all_data()
    outputs: Dict[str, dict] = {}

    for mode in modes:
        run_tag = build_mode_run_tag(mode, mode_version="v1")
        run_dir = batch_dir / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running optimization mode: {mode}")
        final_selection, corridors_gdf, stops_gdf = run_optimized_search(
            data,
            use_network=use_network,
            n_iterations=n_iterations,
            population_size=population_size,
            n_output=n_output,
            evaluation_mode=mode,
            network_synergy_weight=network_synergy_weight,
            network_anchor_top_k=network_anchor_top_k,
            network_transfer_radius_m=network_transfer_radius_m,
        )

        ranking_df = normalize_corridor_ranking(pd.DataFrame(corridors_gdf.drop(columns=["geometry"])))
        summary = {
            "workflow": "optimization",
            "version": "v1",
            "batch_tag": batch_tag,
            "run_tag": run_tag,
            "evaluation_mode": mode,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "n_iterations": int(n_iterations),
            "population_size": int(population_size),
            "n_output": int(n_output),
            "use_network_routing": bool(use_network),
            "network_synergy_weight": float(network_synergy_weight),
            "network_anchor_top_k": int(network_anchor_top_k),
            "network_transfer_radius_m": float(network_transfer_radius_m),
            "n_selected_corridors": int(len(corridors_gdf)),
            "n_generated_stops": int(len(stops_gdf)),
        }

        corridors_path = run_dir / f"corridors_{run_tag}.geojson"
        stops_path = run_dir / f"stops_{run_tag}.geojson"
        ranking_path = run_dir / f"ranking_{run_tag}.csv"
        summary_path = run_dir / f"summary_{run_tag}.json"
        manifest_path = run_dir / f"artifact_manifest_{run_tag}.csv"

        corridors_gdf.to_file(corridors_path, driver="GeoJSON")
        stops_gdf.to_file(stops_path, driver="GeoJSON")
        ranking_df.to_csv(ranking_path, index=False)
        _write_json(summary_path, summary)

        rows = _artifact_manifest_rows(run_dir, summary)
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

        outputs[mode] = {
            "final_selection": final_selection,
            "corridors_gdf": corridors_gdf,
            "stops_gdf": stops_gdf,
            "ranking_df": ranking_df,
            "run_dir": run_dir,
            "summary": summary,
        }
        logger.info(f"Archived mode '{mode}' outputs to: {run_dir}")

    # Comparative ranking delta between isolated and network_aware.
    if "isolated" in outputs and "network_aware" in outputs:
        delta_df = compute_ranking_deltas(
            outputs["isolated"]["ranking_df"],
            outputs["network_aware"]["ranking_df"],
        )
        delta_path = batch_dir / "ranking_delta_isolated_vs_network_aware_v1.csv"
        delta_df.to_csv(delta_path, index=False)
        logger.info(f"Wrote ranking deltas: {delta_path}")

        delta_summary = {
            "batch_tag": batch_tag,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "n_corridors": int(len(delta_df)),
            "n_rank_changed": int((delta_df["rank_delta"] != 0).sum()),
            "max_rank_shift": int(delta_df["rank_delta_abs"].max()) if not delta_df.empty else 0,
        }
        _write_json(batch_dir / "ranking_delta_summary_v1.json", delta_summary)

    # Batch-level manifest over all artifacts.
    batch_manifest_rows = []
    for mode, payload in outputs.items():
        run_dir = payload["run_dir"]
        summary = payload["summary"]
        batch_manifest_rows.extend(_artifact_manifest_rows(run_dir, summary))
    if batch_manifest_rows:
        pd.DataFrame(batch_manifest_rows).to_csv(
            batch_dir / "batch_artifact_manifest_v1.csv",
            index=False,
        )

    _write_json(
        batch_dir / "batch_metadata_v1.json",
        {
            "workflow": "optimization",
            "version": "v1",
            "batch_tag": batch_tag,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "evaluation_modes": modes,
        },
    )
    logger.info(f"\nBatch complete. Root archive: {batch_dir}")
    return outputs


def run_batch_evaluation(
    start_corridor: str | int,
    end_corridor: str | int,
    capital_cost_per_km: int = 20_000_000,
) -> Optional[pd.DataFrame]:
    """Backward-compatible wrapper for financial batch evaluation."""
    return run_financial_batch_evaluation(
        start_corridor=start_corridor,
        end_corridor=end_corridor,
        capital_cost_per_km=capital_cost_per_km,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch corridor evaluation workflows")
    parser.add_argument(
        "--workflow",
        type=str,
        default="optimization",
        choices=["financial", "optimization", "both"],
        help="Workflow to run",
    )

    # Financial workflow args.
    parser.add_argument("--start", type=str, default="C20", help="Start corridor (e.g., C20)")
    parser.add_argument("--end", type=str, default="C29", help="End corridor (e.g., C29)")
    parser.add_argument("--cost", type=float, default=20.0, help="Capital cost (millions per km)")

    # Optimization workflow args.
    parser.add_argument(
        "--evaluation-modes",
        nargs="+",
        default=["isolated", "network_aware"],
        help="Optimization evaluation modes to run",
    )
    parser.add_argument("--iterations", type=int, default=10, help="NSGA-II iterations")
    parser.add_argument("--population", type=int, default=50, help="NSGA-II population size")
    parser.add_argument("--output", type=int, default=25, help="Number of corridors to output")
    parser.add_argument("--no-network", action="store_true", help="Disable OSM network routing")
    parser.add_argument(
        "--network-synergy-weight",
        type=float,
        default=0.20,
        help="Network synergy uplift weight for network-aware objective",
    )
    parser.add_argument(
        "--network-anchor-top-k",
        type=int,
        default=12,
        help="Top-K anchor corridors for synergy comparison",
    )
    parser.add_argument(
        "--network-transfer-radius-m",
        type=float,
        default=1200.0,
        help="Transfer proximity radius in meters",
    )
    parser.add_argument(
        "--network-output-root",
        type=str,
        default=str(DEFAULT_NETWORK_BATCH_DIR),
        help="Root folder for optimization batch archives",
    )
    parser.add_argument(
        "--skip-source-manifest-validation",
        action="store_true",
        help="Skip source manifest freshness validation",
    )
    parser.add_argument(
        "--source-manifest",
        type=str,
        default="data/processed/source_manifest.csv",
        help="Source manifest path used for optimization workflow validation",
    )
    args = parser.parse_args()

    if args.workflow in {"financial", "both"}:
        run_financial_batch_evaluation(
            args.start,
            args.end,
            capital_cost_per_km=int(args.cost * 1_000_000),
        )

    if args.workflow in {"optimization", "both"}:
        run_optimization_batch(
            evaluation_modes=args.evaluation_modes,
            n_iterations=args.iterations,
            population_size=args.population,
            n_output=args.output,
            use_network=not args.no_network,
            network_synergy_weight=args.network_synergy_weight,
            network_anchor_top_k=args.network_anchor_top_k,
            network_transfer_radius_m=args.network_transfer_radius_m,
            output_root=Path(args.network_output_root),
            validate_manifest=not args.skip_source_manifest_validation,
            source_manifest_path=Path(args.source_manifest),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

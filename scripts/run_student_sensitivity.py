#!/usr/bin/env python
"""Student parameter sensitivity analysis (3D Latin Hypercube).

Sweeps three student parameters via the feedback loop:
  - student_presence_factor:      [0.20, 0.30]
  - student_trip_rate_multiplier: [0.85, 1.15]
  - student_mode_bias:            [-0.10, +0.10]

Uses scipy LHS with 12 draws (adequate 3D coverage in ~4 hours).
Each draw patches model_options and runs the full feedback loop.

Usage:
    python scripts/run_student_sensitivity.py --scenario current_zoning
    python scripts/run_student_sensitivity.py --scenario current_zoning --n-draws 9 --adaptive-stop
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


OUTPUT_DIR = Path("data/processed")
SCENARIOS_CONFIG = Path("scenarios_config.json")

# Parameter ranges for LHS
PARAM_RANGES = {
    "student_presence_factor":      (0.20, 0.30),
    "student_trip_rate_multiplier": (0.85, 1.15),
    "student_mode_bias":            (-0.10, 0.10),
}
PARAM_NAMES = list(PARAM_RANGES.keys())
N_PARAMS = len(PARAM_NAMES)


def _generate_lhs_samples(n_draws: int, seed: int = 42) -> np.ndarray:
    """Generate LHS samples scaled to parameter ranges. Shape (n_draws, N_PARAMS)."""
    from scipy.stats.qmc import LatinHypercube

    sampler = LatinHypercube(d=N_PARAMS, seed=seed)
    unit_samples = sampler.random(n=n_draws)

    scaled = np.empty_like(unit_samples)
    for j, pname in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[pname]
        scaled[:, j] = lo + unit_samples[:, j] * (hi - lo)
    return scaled


def _patch_config(params: dict) -> dict:
    """Inject student_presence_factor into scenarios_config.json, return original.

    Only patches config for parameters that lack CLI flags.
    student_trip_rate_multiplier and student_mode_bias are passed via CLI args
    to run_feedback_loop.py, so they are NOT duplicated here.
    """
    cfg = json.loads(SCENARIOS_CONFIG.read_text())
    original = json.loads(json.dumps(cfg))
    if "model_options" not in cfg:
        cfg["model_options"] = {}
    # Only student_presence_factor needs config patching (no CLI flag)
    cfg["model_options"]["student_presence_factor"] = params["student_presence_factor"]
    SCENARIOS_CONFIG.write_text(json.dumps(cfg, indent=2))
    return original


def _restore_config(original: dict):
    """Restore original scenarios_config.json."""
    SCENARIOS_CONFIG.write_text(json.dumps(original, indent=2))


def _parse_final_year_results(output_dir: Path) -> dict:
    """Parse viewer data JSON to extract final-year per-corridor results."""
    viewer_path = output_dir / "feedback_loop_viewer_data.json"
    if not viewer_path.exists():
        return {}
    data = json.loads(viewer_path.read_text())
    corridors = data.get("corridors", {})
    summary = {}
    for cid, cdata in corridors.items():
        total = cdata.get("daily_riders_y25", 0)
        non_campus = cdata.get("non_campus_daily", 0)
        student = total - non_campus
        dscr = cdata.get("dscr", 0)
        summary[cid] = {
            "daily_riders": round(total),
            "student_daily": round(student),
            "non_campus_daily": round(non_campus),
            "student_share": round(student / max(total, 1), 3),
            "dscr": round(dscr, 4) if dscr else 0.0,
        }
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Student parameter sensitivity analysis (3D LHS)"
    )
    parser.add_argument(
        "--scenario", default="current_zoning",
        help="Scenario name (default: current_zoning)",
    )
    parser.add_argument(
        "--n-draws", type=int, default=12,
        help="Number of LHS draws (default: 12)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for LHS (default: 42)",
    )
    parser.add_argument(
        "--adaptive-stop", action="store_true",
        help="Enable adaptive early stopping",
    )
    parser.add_argument(
        "--extra-args", nargs="*", default=[],
        help="Additional args to pass to run_feedback_loop.py",
    )
    args = parser.parse_args()

    samples = _generate_lhs_samples(args.n_draws, args.seed)

    all_results = {}
    sample_params = []
    for i in range(args.n_draws):
        params = {PARAM_NAMES[j]: float(samples[i, j]) for j in range(N_PARAMS)}
        sample_params.append(params)

        logger.info(f"\n{'=' * 70}")
        logger.debug(f"  Draw {i + 1}/{args.n_draws}: "
              f"spf={params['student_presence_factor']:.3f}  "
              f"trip_mult={params['student_trip_rate_multiplier']:.3f}  "
              f"mode_bias={params['student_mode_bias']:+.3f}")
        logger.info(f"{'=' * 70}\n")

        original_cfg = _patch_config(params)
        try:
            cmd = [
                sys.executable, "scripts/run_feedback_loop.py",
                "--scenario", args.scenario,
                "--student-trip-rate-mult", str(params["student_trip_rate_multiplier"]),
                "--student-mode-bias", str(params["student_mode_bias"]),
            ]
            if args.adaptive_stop:
                cmd.append("--adaptive-stop")
            cmd.extend(args.extra_args)

            subprocess.run(cmd, check=True)
            all_results[f"draw_{i}"] = _parse_final_year_results(OUTPUT_DIR)
        except subprocess.CalledProcessError as e:
            logger.info(f"WARNING: Draw {i + 1} failed: {e}")
            all_results[f"draw_{i}"] = {}
        finally:
            _restore_config(original_cfg)

    # Build analysis table
    if not all_results:
        logger.info("No results collected.")
        return

    rows = []
    for i, params in enumerate(sample_params):
        draw_key = f"draw_{i}"
        draw_data = all_results.get(draw_key, {})
        for cid, cdata in draw_data.items():
            rows.append({
                "draw": i,
                "corridor_id": cid,
                **params,
                **cdata,
            })

    if not rows:
        logger.info("No corridor data collected.")
        return

    df = pd.DataFrame(rows)

    # Summary: partial correlations per corridor
    logger.info(f"\n{'=' * 70}")
    logger.info("STUDENT SENSITIVITY: Parameter Impact Summary")
    logger.info(f"{'=' * 70}")

    # Aggregate: correlation of each parameter with daily_riders
    for pname in PARAM_NAMES:
        corr = df[pname].corr(df["daily_riders"])
        logger.debug(f"  {pname:40s} r={corr:+.3f}")

    # Per-corridor summary
    logger.info(f"\n{'=' * 70}")
    logger.info("STUDENT SENSITIVITY: Per-Corridor Ridership Range")
    logger.info(f"{'=' * 70}")

    corridor_summary = (
        df.groupby("corridor_id")["daily_riders"]
        .agg(["min", "mean", "max", "std"])
        .sort_values("mean", ascending=False)
    )
    corridor_summary["cv"] = corridor_summary["std"] / corridor_summary["mean"].clip(lower=1)
    logger.info(corridor_summary.to_string())

    # Ranking stability
    rank_cols = []
    for i in range(args.n_draws):
        draw_df = df[df["draw"] == i].set_index("corridor_id")["daily_riders"]
        ranks = draw_df.rank(ascending=False, method="min").astype(int)
        rank_cols.append(ranks.rename(f"rank_{i}"))
    if rank_cols:
        rank_df = pd.concat(rank_cols, axis=1)
        rank_df["rank_range"] = rank_df.max(axis=1) - rank_df.min(axis=1)
        unstable = rank_df[rank_df["rank_range"] >= 3]
        if not unstable.empty:
            logger.info(f"\nWARNING: {len(unstable)} corridors shift >= 3 rank positions:")
            for cid, row in unstable.iterrows():
                logger.debug(f"  {cid}: range {int(row['rank_range'])}")
        else:
            logger.info("\nAll corridors stable (< 3 rank positions shift).")

    # Save results
    out_path = OUTPUT_DIR / "student_sensitivity_results.json"
    output = {
        "n_draws": args.n_draws,
        "seed": args.seed,
        "param_ranges": {k: list(v) for k, v in PARAM_RANGES.items()},
        "samples": [dict(zip(PARAM_NAMES, [float(x) for x in samples[i]]))
                    for i in range(args.n_draws)],
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

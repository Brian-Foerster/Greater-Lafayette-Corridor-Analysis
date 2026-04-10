#!/usr/bin/env python
"""Detect corridor geometry/stops changes and map them to rerun pipelines."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import logging
logger = logging.getLogger(__name__)



DEFAULT_GEOMETRY_PATTERNS = [
    "data/processed/*corridor*.geojson",
    "data/processed/*stops*.geojson",
    "data/processed/*geometry*.csv",
    "data/processed/*alignment*.csv",
]

DEFAULT_PIPELINES = [
    "optimized_corridor_search",
    "run_feedback_loop",
    "apm_corridor_evaluation_integrated",
    "run_uncertainty_sensitivity",
]


def _run_git_cmd(args: Sequence[str], cwd: Path | None = None) -> list[str]:
    """Run git command and return non-empty stdout lines."""
    res = subprocess.run(
        list(args),
        check=False,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )
    if res.returncode != 0:
        return []
    return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]


def _normalize_path(path: str) -> str:
    return str(path).strip().replace("\\", "/")


def is_geometry_path(path: str, patterns: Sequence[str] | None = None) -> bool:
    """Return true when path likely contains corridor geometry/stops artifacts."""
    pats = list(patterns or DEFAULT_GEOMETRY_PATTERNS)
    p = _normalize_path(path)
    lower = p.lower()

    if any(fnmatch.fnmatch(lower, pat.lower()) for pat in pats):
        return True

    suffix = Path(lower).suffix
    if suffix in {".geojson", ".gpkg", ".shp", ".wkt", ".wkb"}:
        key_hit = any(k in lower for k in ("corridor", "stop", "station", "geometry", "alignment"))
        if key_hit:
            return True

    if suffix == ".csv":
        if any(k in lower for k in ("geometry", "alignment", "stops", "station", "wkt", "shape")):
            return True

    return False


def infer_required_pipelines(geometry_files: Iterable[str]) -> list[str]:
    """Map geometry-change files to required rerun pipeline targets."""
    files = [_normalize_path(f).lower() for f in geometry_files]
    if not files:
        return []

    pipelines = set(DEFAULT_PIPELINES)
    # Geometry-specific boost: optimization/search should rerun if corridor lines move.
    if any("corridor" in f for f in files):
        pipelines.add("run_evaluation_batch")
    return sorted(pipelines)


def _is_empty_or_null_ref(ref: str | None) -> bool:
    if ref is None:
        return True
    v = str(ref).strip()
    if not v:
        return True
    if v.lower() == "none":
        return True
    if set(v) == {"0"}:
        return True
    return False


def get_changed_files(
    *,
    base_ref: str | None,
    head_ref: str | None = "HEAD",
    repo_root: Path = Path("."),
) -> list[str]:
    """Get changed file paths between two refs with robust fallbacks."""
    head = str(head_ref or "HEAD")

    if not _is_empty_or_null_ref(base_ref):
        files = _run_git_cmd(["git", "diff", "--name-only", str(base_ref), head], cwd=repo_root)
        if files:
            return [_normalize_path(p) for p in files]

    # Fallback for local/head-only usage.
    files = _run_git_cmd(["git", "diff", "--name-only", "HEAD~1", "HEAD"], cwd=repo_root)
    if files:
        return [_normalize_path(p) for p in files]

    return []


def build_geometry_change_report(
    *,
    changed_files: Sequence[str],
    base_ref: str | None,
    head_ref: str | None,
    geometry_patterns: Sequence[str] | None = None,
) -> dict:
    """Build canonical detection report payload."""
    normalized = [_normalize_path(p) for p in changed_files]
    geometry_files = [p for p in normalized if is_geometry_path(p, patterns=geometry_patterns)]
    required_pipelines = infer_required_pipelines(geometry_files)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_ref": base_ref,
        "head_ref": head_ref,
        "changed_file_count": len(normalized),
        "geometry_changed_file_count": len(geometry_files),
        "geometry_changed": bool(geometry_files),
        "changed_files": normalized,
        "geometry_changed_files": geometry_files,
        "required_pipelines": required_pipelines,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    geom_set = set(report.get("geometry_changed_files", []))
    for p in report.get("changed_files", []):
        rows.append(
            {
                "path": p,
                "is_geometry_change": p in geom_set,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "is_geometry_change"])
        writer.writeheader()
        writer.writerows(rows)


def _write_github_outputs(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"geometry_changed={'true' if report.get('geometry_changed') else 'false'}\n")
        f.write(f"geometry_changed_file_count={int(report.get('geometry_changed_file_count', 0))}\n")
        pipelines = ",".join(report.get("required_pipelines", []))
        f.write(f"required_pipelines={pipelines}\n")


def _load_changed_files_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Changed-files file not found: {path}")
    return [_normalize_path(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect geometry changes and required rerun pipelines")
    parser.add_argument("--base-ref", type=str, default=None, help="Git base ref/sha")
    parser.add_argument("--head-ref", type=str, default="HEAD", help="Git head ref/sha")
    parser.add_argument("--repo-root", type=str, default=".", help="Repository root")
    parser.add_argument(
        "--changed-files-file",
        type=str,
        default=None,
        help="Optional newline-separated changed-files file (bypasses git diff)",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Additional glob pattern for geometry-related files (repeatable)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/processed/geometry_change_detection.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/processed/geometry_change_detection_files.csv",
        help="Output CSV file list path",
    )
    parser.add_argument(
        "--github-output",
        type=str,
        default=None,
        help="GitHub Actions output file path ($GITHUB_OUTPUT)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    extra_patterns = list(args.pattern or [])

    if args.changed_files_file:
        changed_files = _load_changed_files_file(Path(args.changed_files_file))
    else:
        changed_files = get_changed_files(
            base_ref=args.base_ref,
            head_ref=args.head_ref,
            repo_root=repo_root,
        )

    report = build_geometry_change_report(
        changed_files=changed_files,
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        geometry_patterns=DEFAULT_GEOMETRY_PATTERNS + extra_patterns,
    )

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    _write_json(out_json, report)
    _write_csv(out_csv, report)

    if args.github_output:
        _write_github_outputs(Path(args.github_output), report)

    logger.info(
        f"[DONE] Geometry change detection: changed={report['changed_file_count']}, "
        f"geometry_changed={report['geometry_changed_file_count']}"
    )
    logger.info(f"[DONE] Report JSON: {out_json}")
    logger.info(f"[DONE] Report CSV: {out_csv}")
    if report["geometry_changed"]:
        logger.info(f"[DONE] Required pipelines: {', '.join(report['required_pipelines'])}")
    else:
        logger.info("[DONE] No geometry-triggered reruns required.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

"""Run reproducibility and artifact packaging utilities (Week 22)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from src.source_manifest import validate_source_manifest_file


def utc_timestamp_compact(now: Optional[datetime] = None) -> str:
    """Return compact UTC timestamp token (YYYYMMDDTHHMMSSZ)."""
    ts = now or datetime.now(timezone.utc)
    return ts.strftime("%Y%m%dT%H%M%SZ")


def get_git_commit_short(default: str = "nogit") -> str:
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
    return str(default)


def _is_git_dirty(default: bool = False) -> bool:
    """Check whether the working tree has uncommitted changes."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            return bool(res.stdout.strip())
    except Exception:
        pass
    return default


def environment_fingerprint() -> dict:
    """Capture full environment state for reproducibility.

    Returns a dict with:
    - python_version, platform, machine, os
    - git_commit, git_dirty
    - pip_freeze_sha256 (hash of ``pip freeze`` output)
    - key_packages (version strings for critical dependencies)
    """
    import platform as _platform
    import sys as _sys

    fingerprint: dict = {
        "python_version": _sys.version,
        "platform": _platform.platform(),
        "machine": _platform.machine(),
        "os": _platform.system(),
        "git_commit": get_git_commit_short(),
        "git_dirty": _is_git_dirty(),
    }

    # Installed packages: hash of full freeze for exact reproducibility,
    # plus explicit versions of the packages that affect model output.
    _KEY_PACKAGES = {
        "numpy", "pandas", "geopandas", "scipy", "scikit-learn",
        "shapely", "pyproj", "statsmodels", "joblib", "orca",
    }
    try:
        res = subprocess.run(
            [_sys.executable, "-m", "pip", "freeze", "--local"],
            capture_output=True, text=True, check=False,
            timeout=30,
        )
        if res.returncode == 0:
            deps = res.stdout.strip()
            fingerprint["pip_freeze_sha256"] = hashlib.sha256(
                deps.encode()
            ).hexdigest()[:16]
            key_pkgs: dict = {}
            for line in deps.splitlines():
                if "==" in line:
                    name, ver = line.split("==", 1)
                    if name.lower().replace("-", "_") in _KEY_PACKAGES:
                        key_pkgs[name] = ver
            fingerprint["key_packages"] = key_pkgs
    except Exception:
        fingerprint["pip_freeze_sha256"] = "unavailable"
        fingerprint["key_packages"] = {}

    return fingerprint


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 digest for one file."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def source_manifest_version_info(
    source_manifest_path: str | Path = Path("data/processed/source_manifest.csv"),
) -> dict:
    """
    Build deterministic source-manifest version metadata.

    Includes manifest hash and latest retrieval date for traceability.
    """
    manifest_path = Path(source_manifest_path)
    df = validate_source_manifest_file(
        manifest_path,
        max_source_age_days=3650,
        fail_on_error=True,
    )
    hash_full = sha256_file(manifest_path)
    latest_retrieval = (
        pd.to_datetime(df["retrieval_date"], errors="coerce").max()
        if "retrieval_date" in df.columns and len(df) > 0
        else None
    )
    latest_retrieval_str = (
        latest_retrieval.date().isoformat()
        if latest_retrieval is not None and pd.notna(latest_retrieval)
        else "unknown"
    )
    version_id = f"manifest_v1_{latest_retrieval_str}_{hash_full[:12]}"

    return {
        "source_manifest_path": str(manifest_path),
        "source_manifest_rows": int(len(df)),
        "source_manifest_latest_retrieval_date": latest_retrieval_str,
        "source_manifest_sha256": hash_full,
        "source_manifest_version": version_id,
    }


def build_run_id(
    *,
    workflow: str,
    timestamp_token: str,
    git_commit: str,
    scenario_ids: Optional[Sequence[str]] = None,
    version: str = "v1",
) -> str:
    """Build normalized run id for reproducible artifact packaging."""
    workflow_norm = str(workflow).strip().lower().replace(" ", "_")
    commit_norm = str(git_commit).strip() or "nogit"
    parts = [f"run_{version}", workflow_norm, timestamp_token, commit_norm]
    if scenario_ids:
        uniq = sorted({str(s).strip().lower() for s in scenario_ids if str(s).strip()})
        if len(uniq) == 1:
            parts.append(f"scn_{uniq[0]}")
        elif len(uniq) > 1:
            parts.append(f"scn_multi{len(uniq)}")
    return "_".join(parts)


def initialize_run_directory(
    *,
    run_root: str | Path,
    run_id: str,
) -> dict:
    """Create standardized run directory structure and return key paths."""
    run_dir = Path(run_root) / run_id
    metadata_dir = run_dir / "metadata"
    inputs_dir = run_dir / "inputs"
    outputs_dir = run_dir / "outputs"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "metadata_dir": metadata_dir,
        "inputs_dir": inputs_dir,
        "outputs_dir": outputs_dir,
    }


def write_environment_json(
    run_dir: str | Path,
    extra: Optional[dict] = None,
) -> Path:
    """Write environment.json into a run's metadata directory.

    Captures Python version, OS, dependency hashes, and git state.
    """
    env = environment_fingerprint()
    if extra:
        env.update(extra)
    out_path = Path(run_dir) / "metadata" / "environment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    return out_path


def write_json(path: str | Path, payload: dict) -> Path:
    """Write JSON payload and return path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return p


def copy_artifacts_to_output_dir(
    artifact_paths: Iterable[str | Path],
    output_dir: str | Path,
) -> list[Path]:
    """Copy artifacts to standardized outputs folder."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for item in artifact_paths:
        src = Path(item)
        if not src.exists() or not src.is_file():
            continue
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def build_artifact_manifest_rows(
    *,
    run_dir: str | Path,
    artifact_paths: Iterable[str | Path],
    run_metadata: dict,
) -> list[dict]:
    """Build artifact manifest rows for copied output files."""
    base = Path(run_dir)
    rows: list[dict] = []
    created = datetime.now(timezone.utc).isoformat()
    for pth in artifact_paths:
        p = Path(pth)
        if not p.exists() or not p.is_file():
            continue
        row_count = None
        if p.suffix.lower() == ".csv":
            try:
                row_count = int(len(pd.read_csv(p)))
            except Exception:
                row_count = None
        rows.append(
            {
                "run_id": run_metadata.get("run_id"),
                "workflow": run_metadata.get("workflow"),
                "scenario_ids": ",".join(run_metadata.get("scenario_ids", [])),
                "git_commit": run_metadata.get("git_commit", "nogit"),
                "source_manifest_version": run_metadata.get("source_manifest_version", ""),
                "artifact_name": p.name,
                "relative_path": str(p.relative_to(base)),
                "file_size_bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
                "file_format": p.suffix.lower().lstrip("."),
                "row_count": row_count,
                "created_utc": created,
                "version": "v1",
            }
        )
    return rows


def write_artifact_manifest(
    *,
    run_dir: str | Path,
    artifact_paths: Iterable[str | Path],
    run_metadata: dict,
    manifest_filename: str = "artifact_manifest_v1.csv",
) -> Path:
    """Write artifact manifest CSV and return path."""
    base = Path(run_dir)
    rows = build_artifact_manifest_rows(
        run_dir=base,
        artifact_paths=artifact_paths,
        run_metadata=run_metadata,
    )
    manifest_path = base / "metadata" / manifest_filename
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def check_required_artifacts(
    *,
    run_dir: str | Path,
    required_relative_paths: Sequence[str],
) -> dict:
    """Validate required artifact set for one run package."""
    base = Path(run_dir)
    missing = []
    for rel in required_relative_paths:
        p = base / rel
        if not p.exists() or not p.is_file():
            missing.append(str(rel))
    return {
        "run_dir": str(base),
        "required_count": int(len(required_relative_paths)),
        "missing_count": int(len(missing)),
        "missing": missing,
        "is_complete": len(missing) == 0,
    }

"""Source manifest loading and validation utilities."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


REQUIRED_SOURCE_MANIFEST_COLUMNS = (
    "dataset_id",
    "source_url",
    "data_vintage",
    "retrieval_date",
    "refresh_cadence_days",
    "required_for_pipeline",
    "status",
)

ALLOWED_STATUS = {"active", "deprecated", "pending"}
TRUE_VALUES = {"true", "1", "yes", "y", "t"}
FALSE_VALUES = {"false", "0", "no", "n", "f"}


def _coerce_required_flag(series: pd.Series) -> tuple[pd.Series, list[int]]:
    """Coerce required flag values to booleans and report invalid row positions."""
    coerced = []
    invalid_positions = []
    for idx, value in enumerate(series.tolist()):
        if isinstance(value, bool):
            coerced.append(value)
            continue
        text = str(value).strip().lower()
        if text in TRUE_VALUES:
            coerced.append(True)
        elif text in FALSE_VALUES:
            coerced.append(False)
        else:
            coerced.append(False)
            invalid_positions.append(idx)
    return pd.Series(coerced, index=series.index, dtype=bool), invalid_positions


def load_source_manifest(path: str | Path) -> pd.DataFrame:
    """Load source manifest CSV."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def validate_source_manifest(
    df: pd.DataFrame,
    *,
    as_of_date: Optional[date] = None,
    max_source_age_days: Optional[int] = 3650,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate manifest schema and freshness. Returns normalized DataFrame + errors."""
    errors: list[str] = []
    normalized = df.copy()

    missing_cols = [c for c in REQUIRED_SOURCE_MANIFEST_COLUMNS if c not in normalized.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return normalized, errors

    normalized = normalized[list(REQUIRED_SOURCE_MANIFEST_COLUMNS)].copy()

    for col in ("dataset_id", "source_url", "data_vintage", "status"):
        normalized[col] = normalized[col].astype(str).str.strip()

    empty_id_mask = normalized["dataset_id"] == ""
    if empty_id_mask.any():
        errors.append(f"Empty dataset_id values at rows: {empty_id_mask[empty_id_mask].index.tolist()}")

    duplicate_ids = normalized["dataset_id"][normalized["dataset_id"].duplicated()].tolist()
    if duplicate_ids:
        errors.append(f"Duplicate dataset_id values: {sorted(set(duplicate_ids))}")

    bad_url_mask = ~normalized["source_url"].str.lower().str.startswith(("http://", "https://"))
    if bad_url_mask.any():
        bad_ids = normalized.loc[bad_url_mask, "dataset_id"].tolist()
        errors.append(f"Invalid source_url (must start with http/https) for datasets: {bad_ids}")

    parsed_retrieval = pd.to_datetime(normalized["retrieval_date"], errors="coerce")
    if parsed_retrieval.isna().any():
        bad_ids = normalized.loc[parsed_retrieval.isna(), "dataset_id"].tolist()
        errors.append(f"Invalid retrieval_date for datasets: {bad_ids}")

    cadence = pd.to_numeric(normalized["refresh_cadence_days"], errors="coerce")
    if cadence.isna().any() or (cadence <= 0).any():
        bad_ids = normalized.loc[cadence.isna() | (cadence <= 0), "dataset_id"].tolist()
        errors.append(f"Invalid refresh_cadence_days for datasets: {bad_ids}")
    normalized["refresh_cadence_days"] = cadence

    required_for_pipeline, invalid_required = _coerce_required_flag(normalized["required_for_pipeline"])
    if invalid_required:
        bad_ids = normalized.iloc[invalid_required]["dataset_id"].tolist()
        errors.append(f"Invalid required_for_pipeline values for datasets: {bad_ids}")
    normalized["required_for_pipeline"] = required_for_pipeline

    normalized["status"] = normalized["status"].str.lower()
    bad_status_mask = ~normalized["status"].isin(ALLOWED_STATUS)
    if bad_status_mask.any():
        bad_ids = normalized.loc[bad_status_mask, "dataset_id"].tolist()
        errors.append(f"Invalid status values for datasets: {bad_ids}")

    if parsed_retrieval.notna().all() and cadence.notna().all():
        today = pd.Timestamp(as_of_date or date.today())
        age_days = (today - parsed_retrieval).dt.days

        thresholds = cadence.copy()
        if max_source_age_days is not None:
            thresholds = thresholds.clip(upper=int(max_source_age_days))

        stale_mask = (
            normalized["required_for_pipeline"]
            & (normalized["status"] == "active")
            & (age_days > thresholds)
        )
        if stale_mask.any():
            stale = normalized.loc[stale_mask, ["dataset_id"]].copy()
            stale["age_days"] = age_days[stale_mask].values
            stale["max_allowed_days"] = thresholds[stale_mask].astype(int).values
            details = [
                f"{row.dataset_id}(age={int(row.age_days)}, limit={int(row.max_allowed_days)})"
                for row in stale.itertuples(index=False)
            ]
            errors.append(f"Stale required sources detected: {details}")

    return normalized, errors


def validate_source_manifest_file(
    path: str | Path,
    *,
    as_of_date: Optional[date] = None,
    max_source_age_days: Optional[int] = 3650,
    fail_on_error: bool = True,
) -> pd.DataFrame:
    """Load and validate manifest file; raise ValueError on errors by default."""
    df = load_source_manifest(path)
    normalized, errors = validate_source_manifest(
        df,
        as_of_date=as_of_date,
        max_source_age_days=max_source_age_days,
    )
    if errors and fail_on_error:
        raise ValueError("Source manifest validation failed: " + " | ".join(errors))
    return normalized

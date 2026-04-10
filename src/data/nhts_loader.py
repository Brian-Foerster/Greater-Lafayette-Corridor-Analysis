"""
Download NHTS trip rate tables (person-type by purpose) and normalize them
into the format expected by trip_generation.

Usage examples:
  # Download from a provided URL containing columns: person_type,purpose,rate
  python -m src.data.nhts_loader --source-url https://example.com/nhts_trip_rates.csv \
      --out data/processed/nhts_trip_rates.csv

  # Or point to a local CSV
  python -m src.data.nhts_loader --source-file path/to/nhts_rates.csv \
      --out data/processed/nhts_trip_rates.csv

Expected columns in source: person_type,purpose,rate
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from io import StringIO


def _download_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    # Allow small CSV responses
    content = resp.content.decode("utf-8")
    return pd.read_csv(StringIO(content))


def load_rates(source_url: Optional[str], source_file: Optional[Path]) -> pd.DataFrame:
    if source_url:
        df = _download_csv(source_url)
    elif source_file:
        df = pd.read_csv(source_file)
    else:
        raise ValueError("Provide either --source-url or --source-file")

    required = {"person_type", "purpose", "rate"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns {required - set(df.columns)} in source")

    df["rate"] = df["rate"].astype(float)
    return df


def write_rates(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fetch/normalize NHTS trip rates")
    p.add_argument("--source-url", help="URL to CSV with person_type,purpose,rate")
    p.add_argument("--source-file", type=Path, help="Local CSV with person_type,purpose,rate")
    p.add_argument("--out", default="data/processed/nhts_trip_rates.csv", help="Output CSV path")
    args = p.parse_args()

    df = load_rates(args.source_url, args.source_file)
    write_rates(df, Path(args.out))
    print(f"Wrote NHTS trip rates to {args.out}")

# src/ — Data Processing & Model Infrastructure

## Overview
Core scripts for data ingestion, validation, normalization, and enrichment. All scripts are CLI-capable (run with `--help` for usage) and designed to be chained in a pipeline.

## Modules

### `ingest.py`
- **Purpose:** Read GeoJSON/shapefile, coerce id field to integer, reproject to canonical CRS (EPSG:4326), write GeoJSON.
- **Usage:** `python -m src.ingest --input data/raw/zones.geojson --output data/processed/zones.json --id-prop ZONE_ID`
- **Key Functions:** `ingest(input_path, output_path, id_prop, crs)` — reads, validates, reprojects, writes
- **Last Updated:** Nov 15, 2025; stable

### `validate.py`
- **Purpose:** Validate GeoJSON against data contract (id presence, type coercion, uniqueness, geometries, CRS).
- **Key Functions:** `validate_geojson(path, id_prop, crs)` — raises KeyError/ValueError/RuntimeError with actionable messages
- **Handles:** Int64 and float id types; integrality check via safe dropna + lambda
- **Tests:** `tests/test_validate.py`
- **Last Updated:** Dec 30, 2025 (added numpy.integer type support)

### `validate_cli.py`
- **Purpose:** CLI wrapper for validation; runs on multiple files, outputs JSON report, returns exit code.
- **Usage:** `python -m src.validate_cli data/raw/parcels.geojson data/raw/zones.geojson --output report.json`
- **Key Functions:** `run(paths, id_prop, crs)` — validates list of files, returns dict of results
- **Tests:** `tests/test_validate_cli.py`
- **Last Updated:** Nov 15, 2025; stable

### `download_wfs.py`
- **Purpose:** Download GeoJSON from Tippecanoe County WFS (Schneider Corp endpoint), handle pagination (resultOffset), retries, delays.
- **Usage:** `python -m src.download_wfs --all --force` (downloads 15 layers); `--layer 58 --out data/raw/parcels.geojson` (single layer)
- **Key Functions:** `download_layer(layer_id, output_path, force=False)` — fetches paged GeoJSON with retries
- **Built-in Layers:** 58=parcels, 56=roads, 57=highways, 52=zoning, 62–64=AV, 24–28=sales 2021–2025, 44=TIF, 54=cities, 55=county
- **Tests:** None (WFS download requires network; integration testing deferred)
- **Last Updated:** Nov 15, 2025; ran successfully (all 15 layers downloaded)

### `normalize_parcels.py`
- **Purpose:** Clean parcel data: coerce ParcelID to integer, extract/flag duplicates, optionally enrich with AV/sales.
- **Usage:** `python -m src.normalize_parcels --input data/raw/parcels.geojson --output data/processed/parcels_clean.geojson`
- **Key Functions:** `normalize_parcels(input_path, output_path, ...)` — coerces id, splits duplicates, writes clean + duplicates
- **Outputs:**
  - `parcels_clean.geojson` — 3,689 unique parcels (no duplicates, no missing geometries)
  - `parcels_duplicates.geojson` — 78,160 duplicate records (77,566 invalid NaN ids; 594 valid condo/split)
- **Tests:** None (complex multifile output; use inspect_duplicates.py for ad hoc checks)
- **Last Updated:** Dec 30, 2025 (completed full run); stable

### `enrich_parcels.py`
- **Purpose:** Enrich cleaned parcels with Assessed Values (land, improvements, total) and sales records (temporal).
- **Usage:** `python -m src.enrich_parcels --parcels data/processed/parcels_clean.geojson --land-av data/raw/land_av.geojson --sales data/raw/sales_2021.geojson`
- **Key Functions:**
  - `load_and_join_av(av_path, parcels_gdf, av_type)` — loads AV layer, merges on PARCEL_ID, renames columns to avoid collision
  - `link_sales_to_parcels(sales_path, parcels_gdf)` — detects ParcelID/PIN field, merges on attribute, attempts spatial join for unmatched
- **Outputs:**
  - `parcels_enriched.geojson` — 3,689 parcels + land_av, improvements_av, total_av columns (29 cols total)
  - `sales_YYYY_linked.geojson` — sales records with PARCEL_ID column (attribute-linked + spatial join fallback)
- **Recent Results (Dec 30):**
  - AV layers: 3 joined successfully (2,464, 2,421, 2,472 records)
  - Sales 2021: 1,795 of 16,768 linked by PARCEL_ID (10.7%); 0 additional by spatial join
- **Tests:** None yet (complex multifile workflow; manual validation performed)
- **Last Updated:** Dec 30, 2025 (fixed sjoin output handling); stable

### `consolidate_sales.py`
- **Purpose:** Consolidate sales records across multiple years (2021–2025) into single time-series GeoJSON with year column.
- **Usage:** `python -m src.consolidate_sales --parcels data/processed/parcels_enriched.geojson --output data/processed/sales_complete.geojson`
- **Key Functions:** `consolidate_sales(parcels_path, output_path, sales_pattern)` — loads all sales files, links to parcels, adds year column, merges
- **Outputs:** `data/processed/sales_complete.geojson` (5,267 records, 39 columns including year, sales amount, property class)
- **Recent Execution (Dec 30):**
  - Processed 46,798 total sales across 5 years
  - Linked 5,267 records (100% of consolidated set)
  - Breakdown: 2021=1,795, 2022=1,561, 2023=1,241, 2024=383, 2025=287
- **Tests:** None (multi-file consolidation; manual validation performed)
- **Last Updated:** Dec 30, 2025; stable

### `orca_setup.py`
- **Purpose:** Register parcels and sales tables as orca-compatible tables for model estimation; compute summary statistics.
- **Usage:** `python -m src.orca_setup --parcels data/processed/parcels_enriched.geojson --sales data/processed/sales_complete.geojson --stats`
- **Key Functions:**
  - `setup_orca_from_parcels(parcels_path, sales_path, compute_stats)` — registers orca tables, returns dataframes + stats
  - `print_stats(stats)` — pretty-prints summary statistics
- **Outputs:** Orca tables ('parcels' and 'sales') registered and ready for model use
- **Recent Statistics (Dec 30):**
  - Parcels: 3,689 records, current total AV mean $462,674 (median $281,000)
  - Sales: 5,267 records, mean sale amount $484,976 (median $378,900)
  - Parcel distribution: 39 tax districts (largest: District 030 with 504, District 017 with 408)
- **Tests:** None yet (orca registration requires runtime evaluation)
- **Last Updated:** Dec 30, 2025; stable

### `network_accessibility.py` (planned)
- **Purpose:** Build routable network from roads, compute shortest-path distances from parcels to roads and transit stations.
- **Expected Functions:** `build_network()`, `compute_parcel_distances()`
- **Status:** Not yet created; high priority

## Data Flow

```
WFS Download (download_wfs.py)
  ↓
Validation (validate_cli.py)
  ↓
Normalization (normalize_parcels.py)
  ├─ Parcels (clean)
  └─ Duplicates (flagged condo/split)
  ↓
Enrichment (enrich_parcels.py)
  ├─ AV Join (land, improvements, total)
  └─ Sales Link (2021, individual year)
  ↓
Sales Consolidation (consolidate_sales.py)
  └─ Merge 2021–2025 into time-series with year column
  ↓
Orca Registration (orca_setup.py)
  ├─ Parcels table with assessed values
  └─ Sales table with temporal data
  ↓
Network Accessibility (network_accessibility.py — planned)
  └─ Distance to roads, transit stations
```

## Running Tests

```powershell
# All src tests
python -m pytest tests/test_ingest.py tests/test_validate.py tests/test_validate_cli.py tests/test_bootstrap.py -v

# Single module
python -m pytest tests/test_validate.py::test_validate_good_fixture -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## Notes for Contributors

1. **CLI Interface:** All modules should support `--help` and key `--input`, `--output` arguments for pipeline chaining.
2. **Error Handling:** Raise informative errors (KeyError, ValueError, RuntimeError) with context; avoid silent failures.
3. **Logging:** Use print() for simple status messages; consider adding logging module for verbose/debug modes.
4. **Testing:** Write unit tests in `tests/` for new functions; use fixtures for sample data.
5. **Documentation:** Update this file whenever adding or modifying a script; include usage examples.

Last updated: Dec 30, 2025

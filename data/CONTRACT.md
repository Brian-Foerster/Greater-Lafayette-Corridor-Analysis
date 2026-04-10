DATA CONTRACT — Tippecanoe transit modeling

Overview
This document defines the minimal data contract for spatial and tabular inputs used
by the project. Keep this file updated when data sources or canonical column names change.

Canonical CRS
- All spatial files must be provided or reprojected to EPSG:4326 (WGS84) for
  explorer compatibility and stable serializations.

Datasets
1) Zones (GeoJSON / Shapefile)
   - File location (example): data/processed/zones.json
   - Geometry: Polygon or MultiPolygon preferred; Point is acceptable for tiny fixtures.
   - CRS: EPSG:4326
   - Required properties (feature.properties):
     - ZONE_ID: integer — primary id used for joins with model tables. Must be unique.
     - Optional: NAME (string), AREA (float), other metadata fields.
   - Join discipline: When serializing for map queries, keys must be integers (not floats with ".0") or responses should return arrays of objects with explicit integer id fields.

2) Buildings / Parcels / Households (tabular or spatial)
   - Primary keys: BUILDING_ID, PARCEL_ID, HOUSEHOLD_ID — integers.
   - Foreign keys referencing zones should be named zone_id (lowercase) in orca tables.

General rules
- All id fields must use integer types. If ingest receives numeric fields with trailing decimals (1001.0), coerce to integer after verifying they are integral.
- Validate uniqueness of primary keys before exporting to orca HDF5 or GeoJSON.
- Provide sample fixtures under data/raw/ for tests; do not commit sensitive or large raw datasets.

Validation expectations
- Ingest pipeline must raise informative errors for:
  - Missing required properties (KeyError with clear message)
  - Duplicate ids (ValueError listing duplicated ids)
  - CRS mismatches (RuntimeError or warning; prefer to reproject automatically)
  - Missing geometries (ValueError listing features without geometry)

Success criteria
- `data/processed/zones.json` present, CRS=EPSG:4326, each feature has integer `ZONE_ID`, and values are unique.

Change log
- 2025-11-15: Initial contract created. Updated as ingest and models evolve.

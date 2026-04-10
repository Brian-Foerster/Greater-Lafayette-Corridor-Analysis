Layer download notes — Tippecanoe County WFS

Links and usage
- The WFS endpoint pattern is:
  https://wfs.schneidercorp.com/arcgis/rest/services/TippecanoeCountyIN_WFS/MapServer/{LAYER}/query?where=1%3D1&outFields=*&f=geojson&resultOffset={OFFSET}
- Each request returns up to ~2000 features. Use `resultOffset` 0, 2000, 4000, ... until no features are returned.
- The repo includes `src/download_wfs.py` which automates paging, retries, and writes combined GeoJSON files into `data/raw/`.

Priority layers (recommended first downloads)
- Parcels — Layer 58 -> data/raw/parcels.geojson
- Roads — Layer 56 -> data/raw/roads.geojson
- Highways — Layer 57 -> data/raw/highways.geojson
- Zoning — Layer 52 -> data/raw/zoning.geojson
- Land AV — Layer 62 -> data/raw/land_av.geojson
- Improvements AV — Layer 63 -> data/raw/improvements_av.geojson
- Total AV — Layer 64 -> data/raw/total_av.geojson
- Sales 2021–2025 — Layers 24–28 -> data/raw/sales_YYYY.geojson
- TIF Districts — Layer 44 -> data/raw/tif.geojson
- Cities — Layer 54 -> data/raw/cities.geojson
- County Boundary — Layer 55 -> data/raw/county_boundary.geojson

How to run the downloader (PowerShell examples)
- Download a single layer:
```powershell
python -m src.download_wfs --layer 58 --out data\raw\parcels.geojson
```
- Download the recommended default list (one-by-one):
```powershell
python -m src.download_wfs --all
```
- If you want to be cautious and not overwrite existing files, omit `--force`.
- To speed up you can increase `--page-size` if the server allows it (default 2000) and adjust `--delay` between requests.

Notes and cautions
- The downloader uses modest delays and retries but be mindful of server load and usage policies.
- Some layers may return attributes that are arrays or nested objects; we store raw GeoJSON and later clean/normalize via ingestion pipeline.
- If the service returns fewer than `page_size` features for a page, the script stops (end-of-data). For very large layers you may need to run until no features are returned.

Next steps after download
1. Run `src/validate.py` on the downloaded GeoJSON to enforce the contract (e.g., `ZONE_ID` presence for zone files).
2. Convert and link parcel assessment and sales tables to the parcels by `PARCEL_ID` where possible.
3. Build a routable network from `data/raw/roads.geojson` for catchment and accessibility calculations.

If you'd like, I can:
- Run the downloader for a chosen subset (parcels, roads, zoning) and add the downloaded files to `data/raw/` in the repo (note: network access is required and may be rate-limited by the environment).
- Add normalization steps that read the raw GeoJSON, normalize column names, coerce types, and write `data/processed/` outputs ready for orca ingestion.

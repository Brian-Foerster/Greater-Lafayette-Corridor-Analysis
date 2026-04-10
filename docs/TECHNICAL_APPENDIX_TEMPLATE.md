# Technical Appendix Template

Use this template to accompany the Week 25 decision package report.

## A. Run Provenance

- `run_id`:
- generated UTC:
- git commit:
- scenario config path/version:
- source manifest path/version:

## B. Input Data Inventory

| Dataset | Path/URL | Vintage | Retrieval Date | Notes |
|---|---|---|---|---|
| Example | data/processed/source_manifest.csv | YYYY | YYYY-MM-DD | |

## C. Modeling Configuration

- cashflow horizon:
- discount rate:
- fare assumptions:
- uncertainty draws + seed:
- key scenario switches:

## D. Core Output Tables

Include or reference:

1. scenario summary table
2. top recommendations table
3. corridor-level scenario delta table

## E. Plot Catalog

| Plot | File | Interpretation |
|---|---|---|
| Debt coverage distribution | `plots/debt_coverage_distribution.png` | Compare scenario risk profile |
| Ridership vs coverage | `plots/ridership_vs_coverage_zoning.png` | Identify high-demand viable candidates |
| Top corridor delta | `plots/top_corridor_delta.png` | Compare policy impact concentration |

## F. QA and Drift Gates

- behavioral validation status:
- KPI drift gate status:
- geometry rerun smoke status:
- end-to-end smoke status:

## G. Assumptions and Limitations

Document:

- online-data-only constraints,
- proxy assumptions,
- stale-data risk areas,
- known model limitations still open.

## H. Reproducibility Commands

```bash
python scripts/run_end_to_end_smoke.py
python scripts/generate_decision_package.py
```

## I. Recommendation Rationale

Summarize why the selected corridor/scenario is preferred:

- financial viability,
- ridership strength,
- uncertainty posture,
- policy dependency,
- implementation risk.

# Behavioral Validation Report Template

Use this template each time benchmark validation is run (Week 8 gate and later).

## Run Metadata

- Run date:
- Commit hash:
- Scenario/config:
- Validation command:
- Source manifest version:

## Summary

- PASS count:
- WARN count:
- FAIL count:
- Gate outcome (`PASS` or `BLOCKED`):

## Benchmark Results

| Check | Status | Model Value | Benchmark | Deviation | Source | Notes |
|---|---|---:|---:|---:|---|---|
| walk_access_decay_400m |  |  |  |  |  |  |
| walk_access_decay_800m |  |  |  |  |  |  |
| peak_concentration |  |  |  |  |  |  |
| headway_elasticity |  |  |  |  |  |  |
| fare_elasticity |  |  |  |  |  |  |
| parking_transit_sensitivity |  |  |  |  |  |  |
| income_segment_fare_sensitivity_gap |  |  |  |  |  |  |

## Interpretation

- Primary concerns:
- Expected causes:
- Confidence level:

## Required Follow-Up (only if WARN/FAIL exists)

1. Parameter or code areas to inspect:
2. Proposed adjustment:
3. Re-run plan and acceptance target:

## Artifact Paths

- `data/processed/validation_results.json`
- `data/processed/validation_results.csv`
- `data/processed/validation_summary.md`

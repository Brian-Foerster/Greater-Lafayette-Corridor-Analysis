# Network Validation Phase Summary (Week 14)

## Scope

Week 14 objective from `WEEKLY_IMPLEMENTATION_CHECKLIST.md`:
- Add network scenario QA checks in `tests/`.
- Add CI smoke run for network-aware mode in `.github/workflows/`.
- Publish phase-level network validation summary in `docs/`.

## Implemented

1. Network QA test suite
- Added `tests/test_network_validation_gate.py` with:
  - optimization-batch smoke validation using a stubbed optimizer module,
  - artifact-set assertions (`ranking_*.csv`, `ranking_delta_*.csv`, `batch_artifact_manifest_v1.csv`),
  - transfer-assignment reproducibility check (identical inputs -> identical outputs).

2. CI smoke coverage for network-aware mode
- Updated `.github/workflows/test-with-network.yml`.
- Added `network-aware-smoke` job that runs:
  - `tests.test_network_validation_gate`
  - `tests.test_network_aware_scoring`
  - `tests.test_transit_transfer_assignment`
  - `tests.test_evaluation_batch_artifacts`

3. Data-loading resilience for optimization preflight
- Updated `scripts/optimized_corridor_search.py` to locate parcels from a candidate list:
  - `parcels_enriched_final.geojson`
  - `parcels_enriched_with_access_test2.geojson`
  - `parcels_clean.geojson`
- Added explicit error messaging when required parcel inputs are unavailable.
- Added fallback logic in demand-point generation for empty study-area intersection to avoid opaque `arange` failures.

## Validation Results

Executed locally (2026-02-22):

```bash
python -m unittest -v tests.test_network_validation_gate tests.test_network_aware_scoring tests.test_transit_transfer_assignment tests.test_evaluation_batch_artifacts
```

Result: `OK` (11 tests passed).

## Known Blocker for Full Optimization Smoke

Direct CLI optimization smoke run in this workspace still cannot complete with current local parcel content:

```bash
python scripts/run_evaluation_batch.py --workflow optimization --evaluation-modes isolated --iterations 1 --population 8 --output 3 --no-network --network-output-root data/processed/network_batches_smoke --skip-source-manifest-validation
```

Current failure condition:
- available parcel fallback file has too few records and no positive mapped demand weights,
- candidate generation stops with a clear error (`No parcels with positive demand_wt available for candidate generation`).

This is a data readiness issue, not a network-scoring logic issue.

## Exit-Gate Status (Week 14)

- CI smoke coverage for network-aware path: Implemented.
- Transfer-aware output stability/reproducibility checks: Implemented.
- Phase summary: Implemented.
- Full-data optimization smoke: Blocked by local dataset readiness.

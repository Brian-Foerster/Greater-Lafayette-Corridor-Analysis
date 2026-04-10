# Long-Term Plan: Greater Lafayette APM Corridor Program (Online-Data-Only)

## Constraint (Non-Negotiable)

This project will use only existing online data and already-downloaded local files.
No real-world data collection is allowed, including:
- surveys,
- interviews,
- manual field counts,
- custom agency data requests,
- stakeholder workshops for data generation.

Execution companion:
- Detailed weekly delivery plan: `WEEKLY_IMPLEMENTATION_CHECKLIST.md`

## Purpose

Upgrade the current corridor analysis into a reproducible decision-support platform with:
- stable land-use/transport equilibrium behavior,
- realistic development and affordability dynamics,
- stronger multimodal demand representation from online sources,
- corridor-to-network evaluation,
- finance outputs re-based on dynamic ridership,
- automated validation and rerun workflows.

## Current Baseline

- Ridership model bug fixes completed and validated.
- Corridor search re-optimized with geometry constraints.
- 25-year iterative feedback loop implemented in `src/land_use_transport_model.py`.
- Equity, bus restructuring, and awareness ramp are implemented but simplified.

## Program Goals and Success Criteria

1. Technical validity
- Loop convergence is measured and reported.
- Parcel development state depletes capacity over time.
- Results are reproducible with regression tests.

2. Behavioral realism
- Income-segment mode choice captures value-of-time and cost sensitivity differences.
- Parking price effects and transfer penalties are represented.
- Institutional demand (students/staff/healthcare) is represented via online datasets.

3. Equity and policy realism
- Displacement and affordability feedback is modeled from online rent/value proxies.
- Equity metrics evolve over time with modeled household/job redistribution.

4. Decision support
- TIF/NPV/IRR are recalculated using dynamic ridership outputs.
- Corridor and network scenarios are comparable with uncertainty ranges.

## Workstreams

### W1. Equilibrium and State Dynamics

Target files:
- `src/land_use_transport_model.py`
- `scripts/run_feedback_loop.py`
- `tests/`

Deliverables:
- Convergence checks (ridership/development deltas by corridor and year).
- Adaptive stopping and divergence flags.
- Parcel-level built/remaining capacity ledger.
- Run diagnostics output (`data/processed/feedback_loop_diagnostics.csv`).

Acceptance criteria:
- Every run reports convergence status.
- No parcel exceeds cumulative modeled capacity.

### W2. Development and Affordability (Proxy-Driven)

Target files:
- `src/land_use_transport_model.py`
- `scripts/parcel_development_allocation.py`
- `scripts/development_pattern_integration.py`

Deliverables:
- Capacity depletion by parcel and use type.
- Construction lag assumptions parameterized from published benchmarks.
- Displacement module using rent/value growth proxies from online series.

Acceptance criteria:
- Capacity is consumed once and tracked across time steps.
- Equity/displacement outputs vary meaningfully by scenario.

### W3. Demand and Mode Choice Modernization

Target files:
- `src/transit_model.py`
- `src/mode_choice.py`
- `scripts/run_lodes_mode_choice.py`
- `src/data/institutional_generators.py`
- `src/data/purdue_transit_demand.py`

Deliverables:
- Income-specific value-of-time and generalized cost coefficients.
- Parking pricing utility terms using publicly available downtown/campus rates.
- Institutional demand overlays from published enrollment/employment data.
- LODES/QWI refresh process using newest online releases available at run time.

Acceptance criteria:
- Segment-level elasticities fall within literature benchmark ranges.
- Output diagnostics show institutional and non-commute demand contributions.

### W4. Bus and Network Effects

Target files:
- `src/land_use_transport_model.py`
- `scripts/calibrate_bus_routes_pre2025.py`
- `src/gtfs_ridership.py`
- `scripts/optimized_corridor_search.py`

Deliverables:
- Bus restructuring logic based on public GTFS and published service productivity.
- Multi-corridor assignment with transfer penalties and network effects.
- Corridor optimization updated for network synergy impacts.

Acceptance criteria:
- Network-aware results are materially different from isolated-corridor runs.
- Service plan constraints are documented from online/public operations assumptions.

### W5. Finance Re-baseline and Policy Scenarios

Target files:
- `src/finance.py`
- `scripts/financial_corridor_ranking.py`
- `scripts/apm_corridor_evaluation_integrated.py`
- `scripts/tif_financing_model.py`

Deliverables:
- Dynamic annual ridership feed into farebox and TIF trajectories.
- Updated NPV/IRR/debt coverage from feedback-loop ridership.
- Scenario pack: base, conservative, optimistic, anti-displacement.

Acceptance criteria:
- Finance outputs trace to dynamic model outputs and source assumptions.
- Sensitivity table includes top online-data uncertainty drivers.

### W6. Pipeline, QA, and Automation

Target files:
- `.github/workflows/`
- `scripts/run_full_evaluation.py`
- `tests/`

Deliverables:
- Auto rerun trigger when corridor geometry files change.
- Golden-run metric regression checks.
- Source manifest recording dataset URL, vintage, and retrieval date.

Acceptance criteria:
- CI fails on key metric drift beyond threshold.
- One-command reproducible run from inputs to ranked outputs.

## Phased Timeline (26 Weeks)

### Phase 0 (Weeks 1-3): Stabilize Core Loop

Scope:
- Convergence diagnostics.
- Capacity depletion ledger.
- Loop invariants tests.

Exit gate:
- Stable feedback-loop baseline with diagnostics artifact.

### Phase 1 (Weeks 4-8): Behavioral Upgrades

Scope:
- Income-segment utility updates.
- Parking pricing terms from online/public sources.
- Institutional demand overlays from published data.

Exit gate:
- Elasticity and segment sanity checks pass.

### Phase 2 (Weeks 9-14): Network and Operations

Scope:
- GTFS-based restructuring logic.
- Transfer-aware multi-corridor effects.
- Re-run optimization with network-aware scoring.

Exit gate:
- Network-aware ranking outputs generated and validated.

### Phase 3 (Weeks 15-20): Equity and Housing Dynamics

Scope:
- Proxy-based displacement and affordability feedback.
- Time-varying equity metrics.

Exit gate:
- Policy scenarios show distinguishable equity trajectories.

### Phase 4 (Weeks 21-26): Finance Re-run and Decision Package

Scope:
- Re-run TIF/NPV/IRR with dynamic demand outputs.
- Publish uncertainty and sensitivity ranges.
- Produce final reproducible recommendation package.

Exit gate:
- Final package tied to tagged code and dataset manifest.

## Week-by-Week Descriptions (Coming Weeks)

### Week 7: Source Manifest and Data Freshness Controls
- Add a source manifest utility and schema checks so every required online dataset records URL, vintage, retrieval date, and status.
- Create and maintain `data/processed/source_manifest.csv` as the single audited source inventory.
- Add automated validation tests so runs fail fast when required source metadata is missing or stale.

### Week 8: Behavioral Validation Gate
- Add benchmark validation scripts for elasticities and mode-share sanity ranges.
- Integrate benchmark checks into CI so out-of-range behavior blocks merges.
- Publish a short behavioral validation report template in `docs/`.

### Week 9: GTFS-Informed Bus Restructuring Logic
- Replace purely mechanical bus headway updates with GTFS-informed competitiveness and productivity rules.
- Add explicit restructuring decision logic in the feedback model.
- Add tests covering route competition and fallback behavior.

### Week 10: Operating Constraint Enforcement
- Add operating budget/service-hour constraints for restructuring decisions.
- Write diagnostics showing where constraints bind and how service is clipped.
- Add tests for violations, clipping, and guardrail behavior.

### Week 11: Multi-Corridor Transfer Assignment Core
- Add transfer-aware assignment with explicit transfer penalties.
- Support one-transfer path evaluation in the transit assignment step.
- Add synthetic tests for transfer utility and ridership split behavior.

### Week 12: Network-Aware Corridor Scoring
- Extend corridor scoring to include network synergy metrics.
- Add toggles for isolated-corridor versus network-aware evaluation.
- Add regression tests for ranking reproducibility under both modes.

### Week 13: Network-Aware Optimization Re-run
- Run optimization batch workflows with the network-aware objective.
- Version and archive network-aware outputs with clear artifact naming.
- Document interpretation rules for ranking changes versus isolated runs.

### Week 14: Network Validation Milestone
- Add network scenario QA checks and CI smoke coverage for network mode.
- Validate that transfer-aware outputs are stable and reproducible.
- Publish a phase-level network validation summary in `docs/`.

### Week 15: Affordability Pressure Index
- Implement affordability pressure metrics driven by online rent/value proxy series.
- Add configurable proxy parameters and guardrails in scenario config.
- Add tests for pressure response under growth and policy variation.

### Week 16: Displacement Transition Logic
- Add displacement transition modeling for vulnerable households under rent pressure.
- Output displacement diagnostics by corridor and modeled year.
- Add conservation tests to ensure moved/remaining totals are internally consistent.

### Week 17: Equity Coupling with Displacement
- Feed post-displacement household distribution into equity metrics.
- Produce time-varying equity trajectories for each scenario.
- Add tests for ratio sanity bounds and expected directional behavior.

### Week 18: Equity Policy Scenario Pack
- Add anti-displacement policy levers to scenario configuration.
- Wire policy toggles into scenario execution and reporting.
- Document policy interpretation and caveats in `docs/`.

### Week 19: Dynamic Ridership to Finance Wiring
- Update finance calculations to consume year-by-year ridership series.
- Integrate dynamic demand into corridor financial ranking scripts.
- Add tests for annual cashflow consistency and input validation.

### Week 20: Re-baselined NPV/IRR/DCR Outputs
- Recompute NPV/IRR/debt metrics using dynamic ridership outputs.
- Ensure TIF and integrated evaluation scripts trace to modeled annual demand.
- Add regression tests for financial metric stability and traceability.

### Week 21: Uncertainty and Sensitivity Framework
- Add uncertainty ranges for key online-data-driven assumptions.
- Implement structured sensitivity or Monte Carlo runner for scenario outputs.
- Produce percentile-band outputs for ridership and finance metrics.

### Week 22: Reproducibility and Artifact Packaging
- Capture run metadata (commit hash, manifest version, timestamp, scenario id).
- Standardize output structure for reproducible run comparisons.
- Add tests that verify required artifact completeness.

### Week 23: Geometry-Change Automation
- Add corridor geometry change detection utilities.
- Add workflow triggers so geometry edits automatically rerun required pipelines.
- Add CI smoke workflow for geometry-driven reruns.

### Week 24: Final QA Hardening
- Add end-to-end integration smoke targets and tighten KPI drift thresholds.
- Stabilize flaky tests and reduce non-deterministic behaviors.
- Lock a final pre-release QA gate for full pipeline reliability.

### Week 25: Decision Package Generation
- Implement one-command generation of final scenario comparison outputs.
- Produce recommendation-ready tables/plots and technical appendix artifacts.
- Add final report assembly script in `scripts/`.

### Week 26: Release and Handoff
- Tag release candidate and freeze source manifests.
- Publish final metrics snapshot, changelog, and handoff runbook.
- Complete maintenance checklist for post-release updates.

## Online Data Stack (Allowed Sources)

Primary sources:
- US Census LEHD/LODES.
- Census ACS.
- BLS QWI (optional calibration).
- Public GTFS feeds (current and archived).
- County assessor and parcel/zoning data already online.
- Purdue and major employer published enrollment/employment figures.
- Published parking rates and policy documents.
- Published construction cost indexes and market reports.

## Validation Strategy Under Online-Only Constraint

- Unit and integration tests for model correctness and pipeline stability.
- Benchmark validation against published comparable-system metrics.
- Back-cast validation where historical public data exists (ridership, service levels, values).
- Sensitivity analysis expanded where direct local calibration data is unavailable.

## Tradeoffs Introduced by Online-Only Constraint

1. Lower local specificity in some parameters:
- Construction costs, parking behavior, and displacement responses rely on published proxies.

2. Higher uncertainty bands:
- More scenario and sensitivity emphasis is required where direct observation is unavailable.

3. Slower calibration closure:
- Some parameters cannot be tightly fit without original local survey/transaction data.

4. Better reproducibility and auditability:
- All inputs remain publicly traceable and rerunnable by any reviewer.

## Governance and Reproducibility

- Weekly technical checkpoint: drift, failed tests, blockers.
- Bi-weekly milestone review: phase completion against exit gates.
- Monthly release: tagged code + dataset manifest + metrics snapshot.

## Immediate Next Actions (Next 10 Working Days)

1. Implement source manifest tooling and create `data/processed/source_manifest.csv` (Week 7).
2. Add manifest completeness/freshness validation tests in `tests/`.
3. Add benchmark elasticity validation scripts and CI checks (Week 8).
4. Publish a behavioral validation report template in `docs/`.
5. Run and store a Week 8 benchmark baseline for future regression checks.

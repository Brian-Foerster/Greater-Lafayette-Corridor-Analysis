# Weekly Implementation Checklist (26 Weeks, PR-Sized)

This checklist operationalizes `LONG_TERM_PLAN.md` under the online-data-only constraint.

## PR Sizing Standard

- Target each PR to one coherent behavior change.
- Keep code changes reviewable in one session (roughly 200-600 net LOC when possible).
- Include tests and docs in the same PR.
- Require reproducible command examples in PR description.

## Week-by-Week Execution

## Week 1

Objective: Add convergence scaffolding to the feedback loop.

PR tasks:
- PR W01-1: Add convergence config and thresholds in `src/land_use_transport_model.py`.
- PR W01-2: Add convergence status columns to outputs in `scripts/run_feedback_loop.py`.
- PR W01-3: Add unit tests for threshold logic in `tests/`.

Done when:
- Run output includes convergence fields and tests pass.

## Week 2

Objective: Implement adaptive stop/divergence behavior.

PR tasks:
- PR W02-1: Add adaptive stopping and max-iteration guards in `src/land_use_transport_model.py`.
- PR W02-2: Add diagnostics artifact writer (`feedback_loop_diagnostics.csv`) in `scripts/run_feedback_loop.py`.
- PR W02-3: Add integration test for stable vs divergent synthetic cases in `tests/`.

Done when:
- Stable case stops early; divergent case is flagged.

## Week 3

Objective: Add parcel capacity depletion ledger.

PR tasks:
- PR W03-1: Add parcel-level remaining-capacity fields and update logic in `src/land_use_transport_model.py`.
- PR W03-2: Add no-double-counting tests in `tests/`.
- PR W03-3: Add baseline diagnostics export schema docs in `docs/`.

Done when:
- No parcel exceeds cumulative capacity in baseline run.

## Week 4

Objective: Parameterize income-segment utility terms.

PR tasks:
- PR W04-1: Add segment coefficient schema in `src/mode_choice.py`.
- PR W04-2: Wire segment coefficients into LODES mode-choice path in `src/transit_model.py`.
- PR W04-3: Add segment elasticity unit tests in `tests/`.

Done when:
- SE01/SE02/SE03 produce distinct utility responses.

## Week 5

Objective: Add parking pricing utility effects (online sources only).

PR tasks:
- PR W05-1: Add parking-cost inputs and defaults in `src/mode_choice.py`.
- PR W05-2: Add parking scenario parameters to `scenarios_config.json`.
- PR W05-3: Add parking sensitivity tests and CLI examples in `scripts/run_lodes_mode_choice.py`.

Done when:
- Car utility and mode share shift with parking price scenarios.

## Week 6

Objective: Integrate institutional demand overlays.

PR tasks:
- PR W06-1: Implement overlay ingest path in `src/data/institutional_generators.py`.
- PR W06-2: Connect overlay weighting in `src/data/purdue_transit_demand.py` and `src/transit_model.py`.
- PR W06-3: Add diagnostics columns for institutional demand contribution in outputs.

Done when:
- Model reports institutional share contribution by corridor.

## Week 7

Objective: Refreshable online dataset management.

PR tasks:
- PR W07-1: Add source manifest schema and loader in `src/` utilities.
- PR W07-2: Create `data/processed/source_manifest.csv` with URL, vintage, retrieval date.
- PR W07-3: Add tests validating manifest completeness and required columns.

Done when:
- Pipeline fails fast on missing or stale required source metadata.

## Week 8

Objective: Behavioral validation gate.

PR tasks:
- PR W08-1: Add benchmark elasticity check script updates in `scripts/validate_against_benchmarks.py`.
- PR W08-2: Add CI job for benchmark sanity checks in `.github/workflows/`.
- PR W08-3: Publish phase report template in `docs/`.

Done when:
- CI enforces elasticity sanity bounds.

## Week 9

Objective: Replace mechanical bus restructuring with GTFS-informed logic.

PR tasks:
- PR W09-1: Add GTFS-informed route competitiveness metrics in `src/gtfs_ridership.py`.
- PR W09-2: Add restructuring decision module in `src/land_use_transport_model.py`.
- PR W09-3: Add unit tests for restructure rules.

Done when:
- Headway changes come from explicit service-plan logic, not scalar formula only.

## Week 10

Objective: Add operating-constraint checks.

PR tasks:
- PR W10-1: Add service-hour and frequency budget constraints in `src/land_use_transport_model.py`.
- PR W10-2: Add constraint diagnostics output to `scripts/run_feedback_loop.py`.
- PR W10-3: Add tests for budget violations and clipping behavior.

Done when:
- Restructuring never violates configured operating constraints.

## Week 11

Objective: Build transfer-aware multi-corridor assignment core.

PR tasks:
- PR W11-1: Add transfer penalty parameters and path utility hooks in `src/transit_model.py`.
- PR W11-2: Add multi-corridor assignment method in `src/transit_model.py`.
- PR W11-3: Add synthetic network transfer tests.

Done when:
- Assignment supports at least one transfer with penalty.

## Week 12

Objective: Integrate network effects into corridor scoring.

PR tasks:
- PR W12-1: Add network synergy metrics to `scripts/optimized_corridor_search.py`.
- PR W12-2: Add scenario toggle for isolated vs network-aware evaluation.
- PR W12-3: Add regression tests for ranking stability.

Done when:
- Search can run both isolated and network-aware modes reproducibly.

## Week 13

Objective: Re-run optimization with network-aware objective.

PR tasks:
- PR W13-1: Add batch runner wiring in `scripts/run_evaluation_batch.py`.
- PR W13-2: Add artifacts naming/versioning conventions in `scripts/`.
- PR W13-3: Add documentation for interpretation of ranking deltas in `docs/`.

Done when:
- Network-aware ranked outputs are generated and archived.

## Week 14

Objective: Network validation gate.

PR tasks:
- PR W14-1: Add network scenario QA checks in `tests/`.
- PR W14-2: Add CI smoke run for network-aware mode in `.github/workflows/`.
- PR W14-3: Add phase summary report file in `docs/`.

Done when:
- CI passes for network-aware pipeline smoke run.

## Week 15

Objective: Implement affordability pressure index.

PR tasks:
- PR W15-1: Add affordability pressure calculation in `src/land_use_transport_model.py`.
- PR W15-2: Add online-data proxy parameter table in config files.
- PR W15-3: Add unit tests for pressure index behavior.

Done when:
- Affordability pressure values are produced for each modeled period.

## Week 16

Objective: Add displacement transition logic.

PR tasks:
- PR W16-1: Add household displacement state transitions in `src/land_use_transport_model.py`.
- PR W16-2: Add displacement diagnostics output columns in `scripts/run_feedback_loop.py`.
- PR W16-3: Add tests for conservation checks (moved vs remaining households).

Done when:
- Displacement transitions are internally consistent and scenario-sensitive.

## Week 17

Objective: Couple displacement with equity metrics.

PR tasks:
- PR W17-1: Update equity tracking to use post-displacement populations in `src/land_use_transport_model.py`.
- PR W17-2: Add equity trajectory outputs by scenario.
- PR W17-3: Add tests for monotonicity and ratio sanity bounds.

Done when:
- Equity metrics materially vary across policy scenarios.

## Week 18

Objective: Equity policy levers and scenario pack.

PR tasks:
- PR W18-1: Add anti-displacement policy toggles in `scenarios_config.json`.
- PR W18-2: Add scenario runner support in `src/scenario.py`.
- PR W18-3: Add docs for policy lever interpretation in `docs/`.

Done when:
- Scenario runs include policy-driven equity differences.

## Week 19

Objective: Finance model wiring to dynamic ridership.

PR tasks:
- PR W19-1: Add dynamic annual demand feed into `src/finance.py`.
- PR W19-2: Update `scripts/financial_corridor_ranking.py` for dynamic inputs.
- PR W19-3: Add tests for annual cashflow consistency.

Done when:
- Finance functions accept and validate year-by-year ridership series.

## Week 20

Objective: Recompute NPV/IRR/debt coverage with updated demand.

PR tasks:
- PR W20-1: Update `scripts/apm_corridor_evaluation_integrated.py` to consume dynamic finance outputs.
- PR W20-2: Update `scripts/tif_financing_model.py` scenario outputs.
- PR W20-3: Add regression tests for finance metric stability.

Done when:
- Updated finance metrics are traceable to feedback-loop outputs.

## Week 21

Objective: Uncertainty and sensitivity framework.

PR tasks:
- PR W21-1: Add uncertainty parameter ranges in config.
- PR W21-2: Add Monte Carlo or structured sensitivity runner in `scripts/`.
- PR W21-3: Add summary output format for percentile bands.

Done when:
- Corridor finance/ridership outputs include uncertainty bands.

## Week 22

Objective: Reproducibility and artifact packaging.

PR tasks:
- PR W22-1: Add run metadata capture (commit hash, source manifest version, timestamp).
- PR W22-2: Standardize output folder structure and naming.
- PR W22-3: Add tests for required artifact set.

Done when:
- Any run can be traced to code revision and source vintages.

## Week 23

Objective: Geometry-change automation.

PR tasks:
- PR W23-1: Add geometry diff detection utility in `scripts/`.
- PR W23-2: Add workflow trigger for corridor geometry changes in `.github/workflows/`.
- PR W23-3: Add smoke pipeline rerun target in CI.

Done when:
- Geometry edits automatically trigger required rerun jobs.

## Week 24

Objective: Final QA hardening.

PR tasks:
- PR W24-1: Add end-to-end integration test target in `tests/`.
- PR W24-2: Tighten drift thresholds for key KPIs.
- PR W24-3: Fix flaky tests and stabilize CI runtime.

Done when:
- Full pipeline passes with deterministic KPI checks.

## Week 25

Objective: Decision package generation.

PR tasks:
- PR W25-1: Add final report generation script in `scripts/`.
- PR W25-2: Add scenario comparison tables/plots into report outputs.
- PR W25-3: Add technical appendix template in `docs/`.

Done when:
- Final recommendation package builds from one command.

## Week 26

Objective: Release and handoff.

PR tasks:
- PR W26-1: Tag release candidate and freeze manifests.
- PR W26-2: Publish final metrics snapshot and changelog in `docs/`.
- PR W26-3: Add handoff runbook and maintenance checklist.

Done when:
- Tagged release is reproducible and fully documented.

## Ongoing Weekly Cadence

- Monday: define PR scope and test plan.
- Mid-week: merge implementation PRs with tests.
- Friday: run full smoke checks, publish metrics delta, and close week gate.

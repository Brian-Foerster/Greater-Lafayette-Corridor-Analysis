# Tier 3 Implementation: Components 2–6

Detailed implementation descriptions for the five model enhancements validated
by literature review and FTA guidance.  Component 1 (full agent
microsimulation) was rejected as over-engineered for a 230K metro — the
current Tier 2 (relocation MNL + SqFtProForma + formula students) already
exceeds typical practice for this population size.

Each component below follows the same structure: **what exists today**,
**what changes**, **implementation steps**, **files modified**, **testing
strategy**, and **dependencies**.

---

## Configuration Architecture: Per-Feature Flags, Not a Tier Toggle

### Why No Monolithic Toggle

Tier 2 was a coherent architectural replacement — relocation MNL, SqFtProForma,
vacancy-rent feedback, and per-parcel state all depend on each other.  It made
sense as a single switchover.

Tier 3 is different.  The five components are **independent enhancements**:

| Component | Nature | Replaces existing logic? |
|---|---|---|
| 2: Bus redesign | New restructuring engine | Yes (optimizer dispatch) |
| 3: Monte Carlo | Enhanced sampling + ranking | Partially (sampling fn) |
| 4: Alternatives | New run modes (BRT, TSM, No-Build) | No (additive) |
| 5: Equity | New output metrics + ranking | No (additive) |
| 6: Decision package | Enhanced output generator | No (additive) |

A user might want proactive bus restructuring (Component 2) without the
expensive behavioral sensitivity runs (Component 3B).  Or equity metrics
(Component 5) without BRT comparison (Component 4).  A global on/off switch
would force all-or-nothing adoption, which doesn't match how these features
are actually used.

### Configuration Schema

All Tier 3 features are controlled by a `model_options` block in
`scenarios_config.json`.  Every flag defaults to the **current behavior** —
running with an unmodified config produces identical results to today.

```json
{
  "model_options": {
    "bus_restructuring": "reactive",
    "uncertainty_correlation": false,
    "behavioral_sensitivity": false,
    "behavioral_sensitivity_lhs_points": 20,
    "transit_mode": "apm",
    "alternative": "build",
    "opening_delay_years": 0,
    "phase_1_stations": null,
    "phase_2_start_year": null,
    "equity_analysis": false,
    "equity_feeder_weighting": false,
    "equity_financial_weight": 0.60,
    "equity_uplift": 0.5,
    "cejst_tracts_path": null,
    "displacement_tracking": false,
    "decision_package_maps": false,
    "fta_cost_effectiveness": false,
    "robust_ranking": false,
    "technical_appendix_auto": false
  }
}
```

### How Features Read Their Flags

Each component reads its own flags at initialization and either activates
new logic or falls back to existing behavior.  The pattern is:

```python
# In the feedback loop or module entry point:
opts = config.get("model_options", {})

# Component 2: bus restructuring
if opts.get("bus_restructuring", "reactive") == "proactive":
    decisions = decide_route_restructuring(routes, se01_parcels, year)
    apply_restructuring_decisions(service_plan, decisions, budget)
else:
    service_plan.optimize_headways(restructure_pressure=pressure)

# Component 5: equity
if opts.get("equity_analysis", False):
    equity = compute_spatial_equity(accessibility_delta, se01_pop, se03_pop)
    viewer_data["equity_ratio"] = equity["equity_ratio"]
```

This means:
- **No feature changes the default code path.**  Existing tests pass
  without config changes.
- **Features can be enabled incrementally** as they're implemented and
  tested.  A user enables `"bus_restructuring": "proactive"` the day that
  component ships, without waiting for Components 3–6.
- **Expensive features are opt-in.**  `behavioral_sensitivity` (which
  triggers 20 LHS feedback-loop re-runs at ~30 min each) defaults to
  `false`.
- **Features that need external data declare their data path.**
  `cejst_tracts_path` is `null` by default; setting it to
  `"data/raw/cejst_tracts.csv"` enables CEJST screening.  The code
  checks for `null` and skips gracefully.

### Interaction Between Flags

Most flags are independent, but a few have soft dependencies:

| Flag | Requires | Behavior if dependency missing |
|---|---|---|
| `robust_ranking` | `uncertainty_correlation` | Works but ranking is less meaningful with uncorrelated draws |
| `fta_cost_effectiveness` | At least one `alternative: "tsm"` run | Omits incremental columns, reports absolute cost-per-trip only |
| `displacement_tracking` | Tier 2 development model active | Silently skipped if `_current_rents` is not populated |
| `equity_feeder_weighting` | LODES SE01 data loaded | Falls back to uniform weighting with a warning |
| `behavioral_sensitivity` | `bus_restructuring: "proactive"` | Works but behavioral reruns use reactive bus model |
| `decision_package_maps` | `contextily` installed | Skips maps with a warning if package missing |

No flag combination produces an error.  Missing dependencies degrade
gracefully to the baseline behavior with a logged warning.

### CLI Interface

The feedback loop script exposes the most commonly toggled flags as
command-line arguments that override the config file:

```bash
# Run with proactive bus restructuring
python scripts/run_feedback_loop.py --bus-restructuring proactive

# Run BRT alternative
python scripts/run_feedback_loop.py --transit-mode brt --alternative build

# Run TSM baseline (no APM, bus improvements only)
python scripts/run_feedback_loop.py --alternative tsm

# Run No-Build baseline
python scripts/run_feedback_loop.py --alternative no-build

# Enable equity analysis
python scripts/run_feedback_loop.py --equity

# Full Tier 3 run (all features)
python scripts/run_feedback_loop.py \
    --bus-restructuring proactive \
    --equity \
    --displacement-tracking \
    --cejst-tracts data/raw/cejst_tracts.csv
```

The decision package and uncertainty scripts read the same
`model_options` from config:

```bash
# Decision package with maps and FTA table
python scripts/generate_decision_package.py --maps --fta

# Uncertainty with correlated sampling and robust ranking
python scripts/run_uncertainty_sensitivity.py --correlated --robust-ranking

# Behavioral sensitivity (expensive — runs 20 LHS feedback loops)
python scripts/run_behavioral_sensitivity.py --lhs-points 20
```

### Migration Path

Since every flag defaults to current behavior, the migration is:

1. **Day 0**: Merge all Tier 3 code.  Zero config changes needed.
   All existing runs produce identical results.
2. **Per-feature adoption**: Enable features one at a time in
   `scenarios_config.json` or via CLI flags.  Validate each against
   baseline before enabling the next.
3. **Production config**: Once all desired features are validated,
   commit the updated `scenarios_config.json` with the target flags.

There is no "switch to Tier 3" moment.  The model gradually acquires
capabilities as features are enabled.

---

## Component 2: Proactive Bus Network Redesign

### What Exists Today

`src/bus_network.py` (~2,550 lines) provides a full dynamic bus network
model with per-route headway optimization within CityBus's $13.5M annual
budget.  Routes are classified as parallel, feeder, or independent by
geometric overlap with the APM corridor.  Headway adjustments are driven by
a `restructure_pressure` ramp — a function of APM ridership maturity that
ranges from 0 (no restructuring) to 1 (full restructuring).

The current model is **reactive**: it adjusts headways in response to
ridership, but doesn't proactively plan which routes to cut, reroute, or
extend.  There is no route elimination — `truncate_parallel_routes()`
degrades parallel headways but never removes a route entirely.  And there is
no equity guard: restructuring doesn't check whether degrading a route
harms SE01 (low-income) riders disproportionately.

### Config Flag

```json
"bus_restructuring": "reactive"   // "reactive" (default/current) | "proactive"
```

When `"proactive"`, `optimize_headways()` dispatches to the new decision
engine.  When `"reactive"`, the existing pressure-ramp logic runs unchanged.

### What Changes

Replace the pressure-ramp reactive model with a **productivity-ranked
restructuring engine** that makes explicit cut/keep/reroute decisions for
each bus route, subject to budget, coverage, and equity constraints.

### Implementation Steps

#### Step 1: Route Productivity Scoring

Add a `route_productivity_score()` function to `bus_network.py` that
computes, for each bus route:

```
productivity = daily_ridership / daily_vehicle_hours
```

This is the standard transit metric (passengers per revenue-hour) used by
CityBus and peer agencies.  Routes below a minimum threshold
(`MIN_PRODUCTIVITY = 8.0` passengers/revenue-hour, typical for small-urban
agencies per NTD data) are candidates for elimination or restructuring.

Source the `daily_ridership` from the GTFS ridership CSVs already loaded by
`load_bus_routes_from_gtfs()`.  The `daily_vehicle_hours` is already
computed on each `BusRoute` object via the `daily_vehicle_hours` property.

#### Step 2: Restructuring Decision Matrix

Add a `RestructuringDecision` enum and a `decide_route_restructuring()`
function that maps each route to one of four actions:

| Classification | Productivity | Action | Budget Effect |
|---|---|---|---|
| Parallel | Any | **Eliminate** | Savings → feeder pool |
| Feeder | ≥ MIN_PRODUCTIVITY | **Enhance** (reduce headway) | Cost from feeder pool |
| Feeder | < MIN_PRODUCTIVITY | **Reroute** to serve APM station | Neutral (same veh-hrs) |
| Independent | ≥ MIN_PRODUCTIVITY | **Keep** (no change) | No effect |
| Independent | < MIN_PRODUCTIVITY | **Reduce** (increase headway 1.5×) | Savings → feeder pool |

The `decide_route_restructuring()` function takes the list of classified
`BusRoute` objects and returns a dict mapping `route_id → action`.

#### Step 3: Coverage Equity Guard

Before executing any elimination or reduction, check that the affected
route's walk catchment (400m buffer around stops) doesn't contain a
disproportionate share of SE01 (low-income) parcels.

Define a function `check_coverage_equity()`:

1. For each route slated for elimination or reduction, compute the set of
   parcels within 400m of the route's stops.
2. Compute the SE01 share of those parcels (from LODES WAC data, already
   loaded).
3. If `se01_share > METRO_SE01_SHARE * 1.5` (i.e., the route serves 50%
   more low-income workers than the metro average), **downgrade** the
   action from "eliminate" to "reduce" or from "reduce" to "keep".

This prevents restructuring from disproportionately harming low-income
transit-dependent riders.  The 1.5× threshold comes from FTA Title VI
guidance on disparate impact analysis.

The metro-average SE01 share is computed once from the full LODES WAC
dataset (already loaded in `build_candidate_stations()`).

#### Step 4: Budget-Constrained Execution

Replace the current `optimize_headways()` ramp-based logic with a
budget-allocation loop:

1. Compute savings from all eliminations and reductions.
2. Allocate savings to feeder enhancements in priority order
   (highest-productivity feeders first).
3. If budget mode is `combined`, allocate `BUS_SAVINGS_APM_OFFSET_FRACTION`
   (30%) of savings to APM O&M.
4. If total feeder enhancement cost exceeds savings pool, enhance only as
   many routes as the pool covers (greedy by productivity).

The existing `RouteServicePlan` class provides the budget-accounting
infrastructure.  The change is to replace the continuous pressure-ramp
with discrete cut/keep/enhance decisions.

#### Step 5: Phased Restructuring Timeline

Restructuring doesn't happen at Year 0.  Add a phasing schedule:

| Year | Action |
|---|---|
| 0–2 | No bus changes (APM construction / ramp-up) |
| 3 | Eliminate highest-overlap parallel routes |
| 5 | Reroute low-productivity feeders to serve stations |
| 7 | Enhance remaining feeders to target headways |

This is controlled by a `RESTRUCTURING_PHASES` dict mapping year thresholds
to allowed actions.  The feedback loop already tracks year index, so the
phasing check is a simple `if year >= threshold` gate.

### Files Modified

- `src/bus_network.py` — add `route_productivity_score()`,
  `RestructuringDecision`, `decide_route_restructuring()`,
  `check_coverage_equity()`, `RESTRUCTURING_PHASES`.  Modify
  `optimize_headways()` to dispatch to new decision engine.
- `src/land_use_transport_model.py` — pass LODES SE01 parcel data to
  `check_coverage_equity()` during each feedback loop year.
- `tests/test_bus_network.py` — add tests for productivity scoring,
  decision matrix, equity guard, phased timeline.

### Testing Strategy

- **Unit tests**: Mock routes with known ridership/veh-hours, verify
  productivity scores and decision mapping.
- **Equity guard test**: Create a route where SE01 share exceeds threshold,
  verify action is downgraded.
- **Budget test**: With fixed savings pool and feeder costs, verify
  allocation is greedy-by-productivity and doesn't exceed pool.
- **Integration**: Run feedback loop for 1 corridor × 10 years, verify
  bus headways change at correct year thresholds.

### Dependencies

None — this component is self-contained within the bus network module.

---

## Component 3: Enhanced Monte Carlo Uncertainty Framework

### What Exists Today

`scripts/run_uncertainty_sensitivity.py` and
`scripts/apm_corridor_evaluation_integrated.py::compute_uncertainty_bands()`
provide a post-hoc Monte Carlo engine.  It samples 9 parameters from
triangular distributions (ridership, TIF, capital cost, O&M, fare, discount
rate, TIF capture rate, confidence ramp midpoint, confidence ramp floor),
runs 500 draws, and produces percentile bands (p10/p50/p90) for financial
metrics.

Key limitations:
- **Parameters are independent** — no correlation structure (e.g., high
  capital cost should correlate with high O&M).
- **Confidence ramp parameters are documented but not sampled** — they
  require a feedback loop re-run and are skipped in post-hoc MC.
- **No behavioral sensitivity** — mode choice elasticity, car ownership
  trends, employment growth are all fixed at baseline values.
- **No scenario-cross-uncertainty** — each scenario (zoning/no_zoning) is
  sampled independently with no modeling of how policy choices interact
  with uncertainty.

### Config Flags

```json
"uncertainty_correlation": false,        // true = Gaussian copula sampling
"behavioral_sensitivity": false,         // true = LHS feedback-loop reruns
"behavioral_sensitivity_lhs_points": 20, // number of LHS sample points
"robust_ranking": false                  // true = regret-minimizing ranking
```

Enhancement A is controlled by `uncertainty_correlation`.  Enhancement B
by `behavioral_sensitivity`.  Enhancement C by `robust_ranking`.  Each
can be enabled independently.

### What Changes

Three enhancements, in order of impact:

#### Enhancement A: Correlated Parameter Sampling

Replace independent triangular draws with a **rank-correlated sampling**
scheme using a Gaussian copula.

**Implementation:**

1. Define a correlation matrix `R` (9×9) encoding realistic parameter
   dependencies:

   | | ridership | capital | O&M | TIF | fare |
   |---|---|---|---|---|---|
   | ridership | 1.0 | 0.0 | 0.0 | 0.3 | 0.1 |
   | capital | 0.0 | 1.0 | 0.5 | 0.0 | 0.0 |
   | O&M | 0.0 | 0.5 | 1.0 | 0.0 | 0.0 |
   | TIF | 0.3 | 0.0 | 0.0 | 1.0 | 0.0 |
   | fare | 0.1 | 0.0 | 0.0 | 0.0 | 1.0 |

   Key correlations: capital↔O&M (+0.5, scope creep affects both),
   ridership↔TIF (+0.3, higher ridership drives more development and tax
   increment).  These values are not precisely calibrated but represent
   directional relationships that prevent nonsensical joint draws (e.g.,
   very high ridership with very low TIF).

2. Sample from multivariate normal with correlation matrix `R` using
   `np.random.multivariate_normal()`.

3. Convert each marginal to uniform via Φ (standard normal CDF), then
   invert through each parameter's triangular quantile function.

This is a standard Iman-Conover procedure and adds ~20 lines to the
sampling function.  The marginal distributions remain triangular — only
the joint dependence structure changes.

**Where to implement:** Add a `_correlated_sample()` function in
`scripts/apm_corridor_evaluation_integrated.py` alongside the existing
`_sample_parameters()`.  Add the correlation matrix to `scenarios_config.json`
under `uncertainty_framework.correlations`.

#### Enhancement B: Behavioral Parameter Sensitivity

Add three behavioral parameters to the MC sampling:

1. **Mode choice distance sensitivity** (`beta_distance`): ±20% around
   the calibrated value in `src/mode_choice.py`.  This captures
   uncertainty in how distance-sensitive mode choice is — a significant
   driver of ridership at the corridor margins.

2. **Employment growth rate**: ±1 percentage point around the baseline
   0.5%/yr.  Higher employment growth means more commute trips in the
   corridor catchment.  This interacts with the LODES OD matrix scaling
   already in the feedback loop.

3. **Car ownership trend**: ±5% around the baseline zero-car shares
   per income segment.  Captures Uber/Lyft effects (reducing car
   ownership) or suburban shift (increasing it).

These three parameters require **feedback loop re-run** to take effect
because they change the mode choice and development model outputs.  This
is expensive (~30 min per full run), so the implementation uses a
**Latin Hypercube Sample (LHS)** of N=20 parameter combinations (not
500 independent draws) and interpolates between them for the remaining
draws.

**Implementation:**

1. Generate N=20 LHS points in the 3-dimensional behavioral parameter
   space using `scipy.stats.qmc.LatinHypercube`.
2. Run the feedback loop once for each LHS point (20 runs × ~30 min =
   ~10 hours, parallelizable to ~2.5 hours on 4 cores).
3. Store the 20 ridership/development trajectories as a lookup table.
4. For each of the 500 MC draws, find the nearest LHS point (or
   interpolate between the 2 nearest) and use its trajectory as the
   base for post-hoc financial scaling.

**Where to implement:** New script `scripts/run_behavioral_sensitivity.py`
that orchestrates the LHS runs.  Modify `compute_uncertainty_bands()` to
accept an optional `behavioral_lookup` table and interpolate.

#### Enhancement C: Robust Corridor Ranking

Replace point-estimate ranking (sort by mean DCR) with a
**regret-minimizing** ranking:

1. For each corridor, compute the probability of being in the top-5
   across all MC draws: `P(rank ≤ 5)`.
2. Compute the **expected shortfall** (mean DCR in the worst 10% of
   draws) — this captures downside risk.
3. Compute **max regret**: for each draw, the gap between the best
   corridor's DCR and this corridor's DCR.  Take the maximum across draws.
4. Rank corridors by a composite:
   `0.40 × norm(P(top-5)) + 0.35 × norm(expected_shortfall) + 0.25 × norm(-max_regret)`.

This is a standard robust decision-making approach (Lempert et al., RAND)
that prevents selecting corridors that look good on average but have
catastrophic downside scenarios.  The three-component formulation adds
minimax regret (a standard DMDU metric) to the original two components.

**Where to implement:** Add `robust_corridor_ranking()` to
`scripts/apm_corridor_evaluation_integrated.py`.  Output a new CSV
`corridors_robust_ranking.csv` alongside the existing percentile bands.

### Files Modified

- `scripts/apm_corridor_evaluation_integrated.py` — add
  `_correlated_sample()`, modify `compute_uncertainty_bands()`, add
  `robust_corridor_ranking()`.
- `scenarios_config.json` — add `correlations` matrix under
  `uncertainty_framework`.
- New: `scripts/run_behavioral_sensitivity.py` — LHS orchestrator.
- `scripts/generate_decision_package.py` — include robust ranking in
  output.
- `tests/test_week21_uncertainty_framework.py` — add correlation and
  ranking tests.

### Testing Strategy

- **Correlation test**: Sample 10,000 draws with copula, verify that
  `np.corrcoef(capital_draws, om_draws)` ≈ 0.5 and
  `np.corrcoef(ridership_draws, tif_draws)` ≈ 0.3.
- **Marginal preservation test**: Verify that each marginal distribution
  still matches the triangular parameters (KS test, p > 0.05).
- **Ranking test**: Create 5 synthetic corridors with known draw
  distributions, verify that the robust ranking correctly penalizes
  high-variance corridors.
- **LHS coverage test**: Verify 20 LHS points span the parameter space
  with maximum dispersion (check minimum pairwise distance).

### Dependencies

- Enhancement B depends on Component 2 (bus network redesign) being
  complete so that behavioral reruns include the updated bus network.
- Enhancement C depends on Enhancement A (correlated sampling) for
  meaningful ranking.

---

## Component 4: Structured Alternatives Analysis

### What Exists Today

`scenarios_config.json` defines 6 named scenarios (status_quo through
zoning_dependent) plus 2 development scenarios (current_zoning,
no_zoning).  The feedback loop runs each development scenario independently,
and `generate_decision_package.py` compares zoning vs. no_zoning with delta
tables.

Limitations:
- Scenarios are **manually defined** with ad-hoc multipliers, not
  systematically designed.
- Only **2** development scenarios are compared in the decision
  package (current_zoning vs. no_zoning).
- No **phasing scenarios** (delayed opening, phased buildout).
- No **technology alternatives** (BRT, enhanced bus) as comparisons.
- No **FTA-standard alternatives** format (TSM/Baseline, Build, Enhanced).

### Config Flags

```json
"transit_mode": "apm",              // "apm" (default) | "brt"
"alternative": "build",            // "no-build" | "tsm" | "build" (default)
"opening_delay_years": 0,          // 0 = Year 0 opening (default)
"phase_1_stations": null,          // null = all stations from Year 0 (default)
"phase_2_start_year": null,        // null = no phasing (default)
"fta_cost_effectiveness": false    // true = compute FTA metrics table
```

The `transit_mode` and `alternative` flags control which variant the
feedback loop runs.  A full alternatives analysis requires **four
separate runs** with different flag combinations (no-build, tsm,
build/apm, build/brt), then the decision package compares all results.
Phasing flags are only relevant for `alternative: "build"`.

### What Changes

Restructure the scenario framework to follow **FTA Alternatives Analysis
guidance** (23 CFR 611, Small Starts / New Starts) with systematic
scenario definition, both development scenarios compared, and a
no-build/TSM baseline.

#### Step 1: Define FTA-Standard Alternatives

Create four alternatives that match FTA's expected analysis structure:

1. **No-Build**: Current transit network (CityBus only), no APM, current
   zoning.  This is the comparison baseline.  Implementation: run the
   feedback loop with `apm_ridership = 0` at every year (skip APM mode
   choice, skip TIF, skip development response).  The bus network
   operates at current headways.  This produces a 25-year trajectory
   of population, jobs, and transit ridership under status quo.

2. **TSM (Transportation System Management)**: No APM, but implement
   bus network improvements — enhanced feeder headways, signal priority,
   proof-of-payment boarding.  This represents the "best bus" alternative
   that FTA requires as a baseline for incremental benefit calculation.
   Implementation: run the feedback loop with `apm_ridership = 0` but
   apply the bus speed improvements (TSP 12%, all-door 6%) from
   `bus_network.py` and reallocate 20% of the APM capital budget to bus
   fleet expansion.

3. **Build (APM)**: The current model output — APM corridor with bus
   restructuring and TOD zoning.  Run under both zoning scenarios.

4. **Build-Lite (BRT)**: A bus rapid transit alternative using the same
   corridor alignment.  Implementation: use the same station locations
   but with BRT parameters:
   - Capital cost: $30M/km (vs. $100M/km for APM)
   - Operating cost: $135/veh-hr (standard bus, vs. $85 for APM)
   - Speed: 25 km/h (vs. 40 km/h for APM)
   - Headway: 5–10 min (same range)
   - Capacity: 60 pax/vehicle (articulated bus)
   - No grade separation (mixed traffic with signal priority)

   This requires adding a `TransitMode` parameter to the feedback loop
   that switches between APM and BRT parameter sets.  The mode choice
   model already uses generic speed/headway/capacity inputs, so the
   change is in the parameter values, not the model structure.

#### Step 2: Add the TransitMode Abstraction

Create a `TransitMode` dataclass in `src/financial_params.py`:

```python
@dataclass
class TransitMode:
    name: str                    # "APM" or "BRT"
    capital_cost_per_km: float   # $M/km
    operating_cost_per_vhr: float # $/vehicle-hour
    speed_kph: float             # average operating speed
    capacity_per_vehicle: int    # passengers per vehicle
    min_headway_min: float       # physical minimum headway
    max_headway_min: float       # policy maximum
    grade_separated: bool        # affects reliability, mode choice
    vehicles_per_consist: int    # for fleet sizing
```

Define `APM_MODE` and `BRT_MODE` as module-level constants with the
values above.  The feedback loop accepts `transit_mode: TransitMode`
and passes the relevant parameters to mode choice, finance, and bus
network modules.

#### Step 3: Incremental Metrics (Build vs. TSM)

FTA evaluates projects on **incremental** benefit over the TSM baseline,
not absolute performance.  Add an `incremental_metrics()` function to
`src/economic_impact.py` that computes:

- Incremental ridership: `build_riders - tsm_riders`
- Incremental cost per trip: `(build_annual_cost - tsm_annual_cost) / incremental_annual_trips`
- Incremental BCR: `incremental_benefits / incremental_costs`

These are the metrics FTA uses for Small Starts project justification
(Medium or higher rating required on cost-effectiveness).

#### Step 4: Include Both Zoning Scenarios

Modify `generate_decision_package.py` to accept an arbitrary list of
scenario CSVs (not just zoning and no_zoning).  The comparison table
becomes an N×M matrix where N = corridors and M = scenarios, with
pairwise deltas available.

Implementation: change `build_scenario_summary()` to accept a
`Dict[str, pd.DataFrame]` instead of two positional DataFrames.

#### Step 5: Phasing Scenarios

Add two phasing variants to the Build alternative:

1. **Phased buildout**: Build the highest-ridership segment (e.g.,
   Purdue campus to downtown) first (Years 0–3), then extend to
   full corridor (Years 4–7).  Implementation: in the feedback loop,
   use a reduced station list for years 0–3 and the full list from
   year 4 onward.  Capital cost is front-loaded for segment 1,
   with segment 2 costs starting at year 4.

2. **Delayed opening**: APM opens at Year 3 instead of Year 0.
   Bus improvements (TSM) operate alone for Years 0–2.  This
   models construction delay risk and tests whether early bus
   improvements provide enough ridership to justify the wait.

These are implemented as scenario parameters in `scenarios_config.json`
(`phase_1_stations`, `phase_2_start_year`, `opening_delay_years`) and
handled by a `_phasing_gate()` function in the feedback loop that
controls which stations are active at each year.

### Files Modified

- `src/financial_params.py` — add `TransitMode` dataclass, `APM_MODE`,
  `BRT_MODE` constants.
- `src/land_use_transport_model.py` — accept `transit_mode` parameter,
  pass to mode choice and finance.
- `src/economic_impact.py` — add `incremental_metrics()`.
- `scripts/run_feedback_loop.py` — add `--mode brt|apm`, `--alternative
  no-build|tsm|build`, phasing parameters.
- `scripts/generate_decision_package.py` — generalize to N scenarios.
- `scenarios_config.json` — add `alternatives` section with no-build,
  TSM, build, build-lite definitions and phasing parameters.
- `tests/test_alternatives_analysis.py` — new test file.

### Testing Strategy

- **TransitMode test**: Verify BRT parameters produce lower capital cost
  but higher operating cost than APM for the same corridor.
- **Incremental metrics test**: With known build/TSM ridership, verify
  incremental cost-per-trip calculation.
- **No-build test**: Run 1 corridor × 5 years with `apm_ridership = 0`,
  verify zero TIF and zero APM mode share.
- **Phasing test**: Run with `phase_2_start_year = 4`, verify that only
  phase-1 stations are active in years 0–3.
- **N-scenario comparison**: Generate decision package with 2 scenarios,
  verify table has 2 rows plus pairwise delta.

### Dependencies

- Depends on Component 2 (bus network redesign) for the TSM alternative
  (which applies bus improvements without APM).
- BRT mode shares the same mode choice model as APM — no dependency on
  mode choice changes.

---

## Component 5: Comprehensive Equity Analysis

### What Exists Today

The model has three equity-relevant capabilities:

1. **Income-segmented mode choice** (`src/mode_choice.py`): Walk-zone
   MNL is weighted by LODES SE01/SE02/SE03 income segments with
   segment-specific zero-car shares.

2. **Equity metrics** (`src/economic_impact.py::compute_equity_metrics()`):
   Computes annual transport savings for SE01 households and counts
   zero-car households served.  But these are aggregate numbers (one
   value per corridor), not spatial or distributional.

3. **Income segment outputs** in the feedback loop viewer data:
   `riders_SE01`, `riders_SE02`, `riders_SE03`, `latent_SE01`,
   `low_income_access_ratio`.

Limitations:
- No **displacement analysis** — the relocation model exists but isn't
  coupled to equity reporting.
- No **spatial equity mapping** — benefits are aggregate, not
  geographically distributed.
- No **Title VI / Environmental Justice** screening against disadvantaged
  community designations.
- Equity metrics aren't included in the decision package or corridor
  ranking.

### Config Flags

```json
"equity_analysis": false,           // true = spatial equity mapping + ratio
"equity_feeder_weighting": false,   // true = SE01-weighted feeder coverage
"equity_financial_weight": 0.60,    // weight of DCR vs equity in composite ranking
"equity_uplift": 0.5,              // feeder coverage uplift for SE01 sectors
"cejst_tracts_path": null,         // path to CEJST CSV; null = skip EJ screening
"displacement_tracking": false      // true = rent burden + displacement count
```

Each addition has its own flag.  `equity_analysis` enables Additions 1
and 4 (spatial mapping and ranking).  `displacement_tracking` enables
Addition 2.  `cejst_tracts_path` enables Addition 3 when pointed at
the downloaded CSV.  `equity_feeder_weighting` enables Addition 5.

### What Changes

Five additions that bring the equity analysis to FTA Title VI and
Executive Order 14008 (Justice40) standards.

#### Addition 1: Spatial Equity Mapping

For each corridor, compute per-parcel accessibility change and
aggregate by income segment and geography.

**Implementation:**

1. In `src/land_use_transport_model.py`, after computing ridership at
   each year, compute a per-parcel `accessibility_delta`:
   ```
   accessibility_delta[p] = accessibility_with_apm[p] - accessibility_baseline[p]
   ```
   The baseline accessibility is already computed in
   `_snapshot_baseline()`.  The with-APM accessibility is the current
   year's value.

2. Aggregate `accessibility_delta` by income segment:
   ```
   mean_delta_SE01 = weighted_mean(accessibility_delta, weights=se01_pop)
   mean_delta_SE03 = weighted_mean(accessibility_delta, weights=se03_pop)
   equity_ratio = mean_delta_SE01 / max(mean_delta_SE03, 1e-6)
   ```
   An `equity_ratio > 1.0` means low-income parcels benefit more than
   high-income parcels — a pro-equity outcome.

3. Output per-parcel deltas to the viewer data for spatial visualization.
   Add a `parcel_equity` layer to the corridor viewer GeoJSON with
   columns: `parcel_id`, `accessibility_delta`, `income_segment`,
   `pop_weight`.

#### Addition 2: Displacement Risk Index

Couple the relocation model (`src/relocation_model.py`) to an
affordability tracker.

**Implementation:**

1. After each development year in the feedback loop, compute a
   per-parcel `rent_burden`:
   ```
   rent_burden[p] = current_rent[p] / median_income_SE01
   ```
   where `current_rent[p]` comes from the Tier 2 vacancy-rent feedback
   model (already computed as `self._current_rents`) and
   `median_income_SE01` is from ACS data (~$15,000/yr for the
   lowest-earning third of workers, per LODES definition).

2. Flag parcels where `rent_burden > 0.30` (the standard HUD
   affordability threshold) as "at-risk."

3. Track the **count of SE01 households displaced** per year:
   households that were on at-risk parcels in year t-1 and are no
   longer present in year t (based on relocation model population
   allocation).  This uses the existing `_pop_by_parcel` arrays.

4. Output a per-corridor displacement trajectory:
   `cumulative_se01_displaced` over 25 years.  Add to the viewer data
   and decision package.

#### Addition 3: Title VI / EJ Community Screening

Screen corridors against disadvantaged community designations.

**Data source:** The CEJST (Climate & Economic Justice Screening Tool)
dataset from the White House Council on Environmental Quality provides
census-tract-level disadvantaged community (DAC) designations for all
US tracts.  Download the tract-level CSV (~15 MB) which flags tracts as
disadvantaged based on 8 burden categories.

**Implementation:**

1. Add a `data/raw/cejst_tracts.csv` input file (one-time download from
   screeningtool.geoplatform.gov/en/downloads).

2. In `src/economic_impact.py`, add `screen_ej_communities()`:
   - Spatial join corridor station catchments (1200m walk zone) with
     census tracts.
   - For each corridor, compute:
     - `dac_tract_count`: number of disadvantaged tracts in catchment
     - `dac_pop_share`: share of catchment population in DAC tracts
     - `dac_benefit_share`: share of ridership benefit accruing to
       DAC tracts (from spatial equity mapping in Addition 1)

3. FTA Justice40 requires that ≥40% of benefits flow to disadvantaged
   communities.  Compute `justice40_compliant = dac_benefit_share >= 0.40`
   as a boolean flag per corridor.

#### Addition 4: Equity-Weighted Corridor Ranking

Add equity to the corridor selection / ranking process.

**Implementation:**

In `scripts/generate_decision_package.py`, add an equity-adjusted
ranking alongside the existing DCR ranking:

```
equity_score = (
    0.30 * normalize(equity_ratio)           # accessibility equity
  + 0.25 * normalize(se01_riders_share)      # low-income ridership share
  + 0.25 * normalize(-cumulative_displaced)  # displacement (lower = better)
  + 0.20 * normalize(dac_benefit_share)      # Justice40 alignment
)

composite_score = 0.60 * normalize(dcr) + 0.40 * equity_score
```

The 60/40 weighting between financial viability and equity is a policy
choice that should be configurable in `scenarios_config.json`.  Output
three ranking tables: by DCR, by equity_score, and by composite_score.

#### Addition 5: Feeder Coverage Equity Weighting

Modify `compute_sector_coverage()` in `src/bus_network.py` to accept an
optional `equity_weight` array that up-weights sectors containing more
SE01 parcels.

**Implementation:**

1. After computing population-weighted sector coverage (existing), apply
   an equity multiplier:
   ```
   equity_weighted_coverage[s] = coverage[s] * (1.0 + EQUITY_UPLIFT * se01_share[s])
   ```
   where `EQUITY_UPLIFT = 0.5` (sectors with 100% SE01 population get
   50% more weight in the coverage score).

2. This nudges feeder bus optimization toward better serving low-income
   areas within the feeder ring, without requiring separate equity-tagged
   scenarios.

### Files Modified

- `src/land_use_transport_model.py` — compute per-parcel
  `accessibility_delta`, displacement tracking, add equity fields to
  viewer data.
- `src/economic_impact.py` — add `screen_ej_communities()`,
  `compute_spatial_equity()`, `compute_displacement_risk()`.
- `src/bus_network.py` — add `equity_weight` parameter to
  `compute_sector_coverage()`.
- `scripts/generate_decision_package.py` — add equity ranking tables,
  Justice40 compliance flag.
- `scenarios_config.json` — add `equity_weights` configuration
  (financial/equity split, EQUITY_UPLIFT).
- New data: `data/raw/cejst_tracts.csv` (one-time download).
- `tests/test_economic_impact.py` — add equity metric tests.
- `tests/test_bus_network.py` — add equity-weighted coverage test.

### Testing Strategy

- **Equity ratio test**: Create 2 corridors — one serving high-SE01
  area, one serving high-SE03 area — verify equity_ratio > 1 for
  corridor 1 and < 1 for corridor 2.
- **Displacement test**: Set rents above affordability threshold on
  specific parcels, verify displacement count matches relocated
  SE01 households.
- **CEJST screening test**: Mock 5 tracts (3 DAC, 2 non-DAC), verify
  `dac_pop_share` calculation.
- **Justice40 test**: Corridor with 50% DAC benefit → compliant;
  corridor with 30% → non-compliant.
- **Equity-weighted coverage**: Sector with 80% SE01 should have higher
  weighted coverage than sector with 10% SE01 at same headway.
- **Ranking test**: Corridor with low DCR but high equity_score should
  rank higher in composite than corridor with mid DCR and zero equity.

### Dependencies

- Addition 1 (spatial equity) requires the feedback loop to be running
  (uses per-year accessibility data).
- Addition 2 (displacement) requires Tier 2 development model to be
  active (uses `_current_rents` and relocation model).
- Addition 3 (CEJST screening) requires one-time data download.
- Addition 4 (ranking) depends on Additions 1–3 for input metrics.
- Addition 5 (feeder coverage) is independent.

---

## Component 6: Enhanced Decision Package

### What Exists Today

`scripts/generate_decision_package.py` produces a markdown report, CSV
tables, SVG plots, and JSON metadata.  It compares zoning vs. no_zoning,
ranks corridors by DCR, and optionally includes CBA metrics from
`src/economic_impact.py`.

Limitations:
- Only 2 scenarios compared (zoning vs. no_zoning).
- No interactive outputs — decision-makers must manually cross-reference
  static files.
- No equity ranking.  No FTA cost-effectiveness ranking.
- Technical appendix is a placeholder template.
- No corridor maps or geographic visualizations in the package.
- Uncertainty integration is minimal — percentile bands exist but
  aren't used for robust ranking.

### Config Flags

```json
"decision_package_maps": false,     // true = generate corridor maps (needs contextily)
"fta_cost_effectiveness": false,    // true = FTA Small Starts metrics table
"robust_ranking": false,            // true = include P(top-5) and expected shortfall
"technical_appendix_auto": false    // true = auto-generate appendix from code
```

Most enhancements activate automatically when their upstream data is
available (e.g., multi-scenario comparison works as soon as ≥2 scenario
CSVs are provided).  The flags above control the optional/expensive
features.

### What Changes

Transform the decision package from a static report into a comprehensive
decision-support artifact that a city council, transit board, or FTA
reviewer can use directly.

#### Enhancement 1: Multi-Scenario Comparison Matrix

Replace the 2-scenario comparison with an N-scenario matrix.

**Implementation:**

1. Modify `build_scenario_summary()` to accept `Dict[str, DataFrame]`.
   Loop over all scenario names, compute the same aggregate metrics
   for each.

2. Add pairwise delta computation: for each pair of scenarios (i, j),
   compute `delta_ij = metrics_i - metrics_j`.  Output as a separate
   CSV `scenario_pairwise_deltas.csv`.

3. Add a "scenario recommendation" section to the markdown report that
   identifies which scenario is best under different objectives
   (highest ridership, best DCR, most equitable, lowest risk).

4. Add No-Build and TSM from Component 4
   when available.

#### Enhancement 2: FTA Cost-Effectiveness Table

Add an FTA-standard cost-effectiveness summary for each corridor.

**Implementation:**

Add a `build_fta_metrics_table()` function that computes, per corridor:

| Metric | Formula | Source |
|---|---|---|
| Annual operating cost ($M) | From finance model | `src/finance.py` |
| Annualized capital cost ($M) | Capital / useful life (30 yr) | `src/financial_params.py` |
| Total annualized cost ($M) | Operating + annualized capital | Sum |
| Annual trips | Daily ridership × operating days | Feedback loop |
| Incremental trips vs. TSM | Build trips − TSM trips | Component 4 |
| Cost per trip ($) | Total annualized cost / annual trips | Division |
| Incremental cost per trip ($) | Incremental cost / incremental trips | Division |
| FTA rating | Based on incremental cost per trip thresholds | FTA guidance |

FTA Small Starts cost-effectiveness ratings (2024 thresholds):
- High: < $4.00/trip
- Medium-High: $4.00–$8.00
- Medium: $8.00–$12.00
- Low: > $12.00

Output as `fta_cost_effectiveness.csv` and include in the markdown
report.

#### Enhancement 3: Corridor Maps in Package

Generate static map images for the top-5 corridors showing alignment,
stations, walk catchment, and feeder routes.

**Implementation:**

1. Add a `_plot_corridor_map()` function using matplotlib with
   contextily for basemap tiles.

2. For each top-5 corridor, generate a map showing:
   - Corridor alignment (from the GeoJSON already produced)
   - Station locations with labels
   - 1200m walk catchment circles
   - 7000m feeder zone (shaded)
   - Color-coded parcels by development intensity (from feedback loop)
   - SE01 concentration overlay (from LODES)

3. Output as PNG files in `decision_package/maps/`.

4. Embed in the markdown report using relative image links.

This requires `contextily` and `matplotlib` (both already available in
the environment).  The corridor GeoJSON and parcel data are already
produced by the feedback loop.

#### Enhancement 4: Uncertainty Dashboard Table

Integrate Monte Carlo results directly into the decision package.

**Implementation:**

1. Add a `build_uncertainty_dashboard()` function that, for each
   corridor, shows:

   | Corridor | DCR (p10) | DCR (p50) | DCR (p90) | P(viable) | P(top-5) | Downside DCR |
   |---|---|---|---|---|---|---|
   | C1 | 0.22 | 0.35 | 0.52 | 0% | 45% | 0.18 |

2. Include the robust ranking from Component 3 (Enhancement C).

3. Add a "risk profile" plot: stacked bar chart showing p10/p50/p90
   bands for each corridor's DCR, sorted by p50.

#### Enhancement 5: Automated Technical Appendix

Replace the placeholder template with an auto-generated appendix.

**Implementation:**

1. Add a `generate_technical_appendix()` function that writes:

   - **Model description**: Paragraph describing the feedback loop,
     mode choice, development model, bus network — generated from
     module docstrings and constants.
   - **Parameter table**: All calibrated constants from
     `src/financial_params.py`, `src/mode_choice.py`, and
     `src/bus_network.py` with their values, units, and sources.
     Auto-extracted using `inspect` module to read module-level
     constants.
   - **Data sources table**: From `src/source_manifest.py` — lists
     all input datasets with dates, URLs, and versions.
   - **Sensitivity results**: Summary statistics from Monte Carlo
     (which parameters have the largest impact on DCR).
   - **Limitations**: Known model weaknesses from
     `docs/model_weakness_triage.md` (if it exists), or a standard
     limitations paragraph.

2. This appendix is regenerated on every decision package build,
   so it always reflects the current model state.

#### Enhancement 6: Executive Summary with Recommendation

Add a structured executive summary at the top of the markdown report.

**Implementation:**

Generate a 1-page executive summary section containing:

1. **Lead recommendation**: Top corridor by composite score (from
   Component 5, Addition 4), with 1-sentence justification.
2. **Key numbers table**: Ridership (p50), DCR (p50), capital cost,
   25-year TIF revenue, BCR, equity score — for the top-3 corridors.
3. **Scenario sensitivity**: "Under current zoning, the best corridor
   achieves X riders/day.  With TOD zoning, this increases to Y
   (+Z%)."  Auto-generated from scenario comparison data.
4. **Risk statement**: "There is a P% probability that the top
   corridor achieves financial viability (DCR ≥ 1.0)."  From Monte
   Carlo.
5. **Equity statement**: "X% of ridership benefits accrue to
   low-income (SE01) households.  Y corridors meet Justice40
   thresholds."  From Component 5.
6. **Next steps**: Standardized list (environmental review,
   preliminary engineering, public engagement, FTA coordination).

### Files Modified

- `scripts/generate_decision_package.py` — all 6 enhancements modify
  this file.  Refactor into a class `DecisionPackageBuilder` with
  methods for each section.
- `src/economic_impact.py` — add `build_fta_metrics_table()`.
- `scenarios_config.json` — add `decision_package` section with
  configurable weights and thresholds.
- `tests/test_decision_package_generation.py` — extend with tests for
  multi-scenario, FTA table, appendix generation.

### Testing Strategy

- **Multi-scenario test**: Generate package with 3 mock scenario
  DataFrames, verify summary table has 3 rows + 3 pairwise deltas.
- **FTA metrics test**: Known ridership + cost → verify cost-per-trip
  and rating assignment.
- **Appendix test**: Verify that auto-generated appendix includes all
  constants from `financial_params.py` (count check).
- **Executive summary test**: Verify that recommendation matches the
  top corridor by composite score from the input data.
- **Map generation test**: Generate map for 1 corridor, verify PNG
  file exists and has non-zero size.

### Dependencies

- Enhancement 1 depends on Component 4 (alternatives analysis) for
  additional scenarios.
- Enhancement 2 depends on Component 4 for incremental metrics
  (TSM baseline).
- Enhancement 4 depends on Component 3 (enhanced Monte Carlo) for
  robust ranking and P(top-5).
- Enhancement 5 depends on Components 2–5 for complete model
  description.
- Enhancement 6 depends on Component 5 (equity) for equity statement.
- Enhancement 3 (maps) is independent.

---

## Implementation Order

The components have interdependencies that suggest this sequence.
Because every feature defaults to off, each phase can be merged and
validated independently — there is no "big bang" switchover.

```
Phase 1 (parallel, no dependencies):
  Component 2: Bus Network Redesign
    Flag: bus_restructuring = "proactive"
  Component 5, Addition 5: Feeder Coverage Equity Weighting
    Flag: equity_feeder_weighting = true
  Component 6, Enhancement 3: Corridor Maps
    Flag: decision_package_maps = true

Phase 2 (after Phase 1):
  Component 3, Enhancement A: Correlated Sampling
    Flag: uncertainty_correlation = true
  Component 4, Steps 1-3: FTA Alternatives (No-Build, TSM, BRT)
    Flags: transit_mode, alternative
  Component 5, Additions 1-3: Spatial Equity, Displacement, CEJST
    Flags: equity_analysis, displacement_tracking, cejst_tracts_path

Phase 3 (after Phase 2):
  Component 3, Enhancement B: Behavioral Sensitivity
    Flag: behavioral_sensitivity = true
  Component 4, Steps 4-5: All Scenarios + Phasing
    Flags: phase_1_stations, phase_2_start_year, opening_delay_years
  Component 5, Addition 4: Equity-Weighted Ranking
    Flag: equity_financial_weight (tune the 60/40 split)

Phase 4 (after Phase 3):
  Component 3, Enhancement C: Robust Ranking
    Flag: robust_ranking = true
  Component 6, Enhancements 1-2, 4-6: Full Decision Package
    Flags: fta_cost_effectiveness, technical_appendix_auto
```

### Validation Protocol Per Phase

After merging each phase:

1. **Baseline regression**: Run feedback loop with **no config changes**.
   Verify output matches pre-merge baseline exactly (byte-identical CSVs
   or numeric diff < 1e-6).  This confirms the new code doesn't alter
   the default path.

2. **Feature validation**: Enable the phase's flags one at a time.
   For each flag, verify:
   - The feature activates (check logs for feature-specific output)
   - Disabling the flag returns to baseline behavior
   - The feature produces reasonable outputs (sanity checks, not full
     calibration)

3. **Cross-feature test**: Enable all flags from this phase and all
   prior phases simultaneously.  Verify no interaction bugs.

### Example: Minimal vs. Full Tier 3 Configs

**Minimal adoption** (just proactive bus + equity metrics):
```json
{
  "model_options": {
    "bus_restructuring": "proactive",
    "equity_analysis": true
  }
}
```

**Full Tier 3** (all features enabled):
```json
{
  "model_options": {
    "bus_restructuring": "proactive",
    "uncertainty_correlation": true,
    "behavioral_sensitivity": true,
    "behavioral_sensitivity_lhs_points": 20,
    "robust_ranking": true,
    "transit_mode": "apm",
    "alternative": "build",
    "equity_analysis": true,
    "equity_feeder_weighting": true,
    "equity_financial_weight": 0.60,
    "equity_uplift": 0.5,
    "cejst_tracts_path": "data/raw/cejst_tracts.csv",
    "displacement_tracking": true,
    "decision_package_maps": true,
    "fta_cost_effectiveness": true,
    "technical_appendix_auto": true
  }
}
```

Note that the full config still requires **separate runs** for the
alternatives analysis (no-build, tsm, build/brt).  These are different
invocations of the feedback loop, not config toggles within a single
run:

```bash
# Run all four alternatives for one development scenario
python scripts/run_feedback_loop.py --alternative no-build
python scripts/run_feedback_loop.py --alternative tsm
python scripts/run_feedback_loop.py --alternative build --transit-mode apm
python scripts/run_feedback_loop.py --alternative build --transit-mode brt

# Then generate the comparison package
python scripts/generate_decision_package.py --maps --fta
```

Estimated scope: ~1,200 lines of new code across 8 files, plus ~400
lines of tests.  The largest single addition is the equity analysis
(Component 5, ~350 lines) followed by the decision package refactor
(Component 6, ~300 lines).

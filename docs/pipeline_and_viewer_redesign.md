# Pipeline Stages & Viewer Redesign

## Overview

The model pipeline has 5 stages (1, 2a, 2b, 3, 4). Each stage filters the
corridor set and adds information. The viewer should reflect all available
data, progressively enriching its display as later stages complete. This
document describes what each stage produces, how the viewer consumes it, and
what viewer changes are needed.

---

## Stage 1

**Script**: `scripts/optimized_corridor_search.py`
**Time**: ~3-10 min (cached distances)
**Input**: OSM road graph, enriched parcels, LODES OD flows, GTFS bus stops
**Output**:
- `data/processed/apm_phase2a_corridors.geojson` — corridor geometries +
  properties (length_km, n_stops, barrier_cost_usd, curve_cost_mult,
  ridership_est, cost_efficiency, stations list)
- `data/processed/apm_phase2a_stops.geojson` — station points with demand
  coverage values

**Corridor count**: ~25 (from diversity selection of ~40-60 evolved candidates)

**What the viewer shows after Stage 1 only**:
- Corridor lines on map with station markers
- Static ridership estimate (from search scoring, not feedback loop)
- Capital cost estimate (from corridor properties)
- Cost efficiency rank
- No trajectories, no development, no bus network, no financials

**Current state**: The viewer requires feedback loop data to populate most
fields. If only Stage 1 output exists, the corridor list shows but detail
panels are mostly empty.

**Change needed**: The viewer should have a "search-only" fallback mode that
populates cards with the static ridership estimate, capital cost, length,
stops, and barriers from the GeoJSON properties. This gives immediate visual
feedback after the search completes without requiring a feedback loop run.

### Implementation

In `corridor_viewer.html`, the `buildCard()` function currently reads from
`feedbackData[cid]`. Add a fallback path:

```javascript
function buildCard(cid, props, fb, color, rank) {
  // fb may be null if only Stage 1 has run
  const riders = fb ? fb.year25_riders : (props.ridership_est || 0);
  const capCost = capitalCost(props);
  const sparkHtml = fb ? sparklineSVG(fb.ridership_trajectory, ...) : '';
  // ...
}
```

The corridor list should be sortable by:
- Ridership (static estimate or year-25 feedback)
- Capital cost
- Cost efficiency (riders / $M capital)
- Length

Add a banner at the top of the sidebar when no feedback data exists:
```
"Showing search estimates only. Run feedback loop for dynamic trajectories."
```

---

## Stage 2a / 2b

**Script**: `scripts/run_feedback_loop.py`
**Time**: ~15-30 min per scenario (parallelized across corridors)
**Input**: Stage 1 corridors + enriched parcels + LODES + GTFS
**Output**:
- `data/processed/feedback_loop_results.csv` — long format:
  corridor_id × year × {daily_riders, new_units, new_pop, new_jobs,
  new_comm_sqft, bus_headway, bus_feeder_headway, bus_restructure_pressure,
  bus_restructure_phase, work_commute_daily, local_nonwork_daily,
  campus_daily, destination_daily, equity_daily, riders_SE01/SE02/SE03,
  lodes_commute_apm_share, length_km, n_stops, ...}
- `data/processed/feedback_loop_viewer_data.json` — pre-aggregated per-corridor
  data with trajectories, components, bus state, inline financials
- `data/processed/corridor_viewer.html` — viewer with embedded data
- `data/processed/screening_survivors.json` (if `--screening`)
- `data/processed/feedback_loop_diagnostics.csv`

**Corridor count**: All 25 from Stage 1 (or survivors from screening)

**Run modes**:

| Mode | Command | Time | Purpose |
|---|---|---|---|
| Screening | `--screening --adaptive-stop` | ~3 min | Fast filter, 6 time steps |
| Single scenario | `--scenario current_zoning` | ~20 min | Baseline evaluation |
| All scenarios | `--all-scenarios` | ~40 min | Full 2-scenario comparison |

### What the viewer shows after Stage 2b

**Ranking tab** (existing, needs fixes):
- Corridor cards sorted by year-25 ridership (default) or DSCR
- Sparkline showing 25-year ridership trajectory
- Opening ridership, final ridership, growth %
- Click to zoom on map + show detail panel

**Detail panel** (existing, needs additions):
- 9-cell summary grid: Y25 ridership, capital cost, bus phase, housing units,
  population, jobs, DSCR (min), annual revenue, cost/rider
- Ridership trajectory line chart (25 years, bus phase markers)
- Cumulative development timeline (units, pop, jobs multi-line)
- New units per period bar chart
- Ridership component breakdown bar (both-ends, origin-only, student,
  generator, induced, latent)

**Finance tab** (existing, needs redesign):
- Currently: flat list of corridors with DSCR values
- Should become: per-corridor financial summary with revenue breakdown

**Bus Network tab** (existing, working):
- Phase timeline (discrete phases: pre-APM → opening → early → mature)
- Feeder/parallel headway trajectories
- Restructure pressure trajectory
- Bus ridership trajectory (if available)

**Development tab** (existing, working):
- Cumulative development for selected corridor
- Per-period breakdown

**Equity tab** (existing, minimal):
- SE01/SE02/SE03 rider counts
- Low-income access ratio

### Changes needed for Stage 2b

#### 1. Unhide the scenario selector

The `<select id="scenario-select">` is already in the HTML at line 134 with
`display:none`. The JavaScript already handles multi-scenario data — the
`feedbackData` object is keyed by scenario name, and the scenario selector
switches between them.

**Fix**: In the initialization code where scenarios are detected, change:
```javascript
// Current: selector always hidden
document.getElementById('scenario-select').style.display = 'none';

// Change to: show when multiple scenarios exist
const scenarioKeys = Object.keys(allData);
const sel = document.getElementById('scenario-select');
if (scenarioKeys.length > 1) {
  sel.style.display = 'inline-block';
  sel.innerHTML = scenarioKeys.map(s =>
    `<option value="${s}">${s.replace(/_/g, ' ')}</option>`
  ).join('');
}
```

When the user switches scenarios, the corridor list, detail panel, and all
charts should update. The corridor geometries on the map stay the same (they're
scenario-independent), but the colors/thickness could reflect scenario-specific
ridership.

#### 2. Fix stale legend values

Line 183 of `corridor_viewer.html`:
```
River: $80M | Highway: $40M | Railroad: $25M
```

Should be:
```
River: $35M | Highway: $20M | Railroad: $5M
```

These match the current values in `optimized_corridor_search.py`:
- `BARRIER_RIVER_COST_USD = 35_000_000` (line 227)
- `BARRIER_HIGHWAY_COST_USD = 20_000_000` (line 228)
- `BARRIER_RAILROAD_COST_USD = 5_000_000` (line 229)

#### 3. Add corridor sort controls

Above the corridor list, add a sort dropdown:
```html
<select id="sort-select">
  <option value="ridership">Ridership (Y25)</option>
  <option value="dscr">DSCR</option>
  <option value="cost_eff">Cost Efficiency</option>
  <option value="growth">Growth %</option>
  <option value="development">Development (units)</option>
</select>
```

The corridor cards are currently generated in a fixed order. Change
`renderCorridorList()` to sort by the selected metric before rendering.

#### 4. Add financial trajectory chart to detail panel

The detail panel shows DSCR as a single number. Add a 25-year line chart
showing cumulative revenue vs cumulative debt service + O&M.

Data required (compute in JavaScript from existing viewer data):
```javascript
function financialTrajectory(fb, props) {
  const fp = EMBEDDED_FINANCIAL_PARAMS;
  const capCost = capitalCost(props);
  const debtService = annualDebtService(capCost, fp);
  const omAnnual = fp.om_fixed + fp.om_per_km * props.length_km
                 + fp.om_per_station * props.n_stops;

  return fb.years.map((yr, i) => {
    const farebox = fb.ridership_trajectory[i] * fp.fare_per_trip * fp.operating_days;
    const tifEst = (fb.cum_units_trajectory[i] || 0) * 250000 * 0.0232 * 0.85;
    const revenue = farebox + tifEst;
    const costs = debtService + omAnnual;
    return { year: yr, revenue, costs, dscr: revenue / costs };
  });
}
```

Display as a dual-line chart:
- Blue line: cumulative annual revenue (farebox + TIF)
- Red line: annual costs (debt service + O&M)
- Green shading where revenue > costs
- Red shading where costs > revenue
- Horizontal dashed line at DSCR = 1.0x

This shows when (if ever) the corridor becomes self-sustaining and how the
gap closes over time as ridership and TIF revenue grow.

#### 5. Add revenue breakdown stacked bar

In the Finance tab, replace the flat DSCR list with a stacked bar per corridor
showing revenue composition:

```
|████ TIF ████|██ Farebox ██|█ Campus █|  vs  |████ Debt Service ████|██ O&M ██|
```

Colors:
- TIF: `#22c55e` (green)
- Farebox: `#0ea5e9` (blue)
- Campus payment: `#a855f7` (purple)
- Debt service: `#ef4444` (red)
- O&M: `#f97316` (orange)

The data keys already exist in the viewer JSON: `annual_tif_musd`,
`farebox_musd`, `campus_payment_musd`, `annual_debt_service_musd`,
`annual_om_musd`.

#### 6. Add capital cost breakdown to popup

The popup currently shows total capital cost. Expand to show components:

```
Capital cost: $823M
  Guideway:    $495M (55M × 9.0km)
  Stations:    $105M (15M × 7)
  Vehicles:     $48M (3M × 16)
  Systems:      $15M (fixed)
  Prof. svcs:   $66M (10%)
  Escalation:   $59M (4.5% × 2yr)
  Barriers:     $35M (river)
  Curves:         $0 (1.00x)
```

The `capitalCost()` function in the viewer already computes most of these
internally. Change it to return an object with components instead of a single
number:

```javascript
function capitalCostBreakdown(p) {
  const fp = EMBEDDED_FINANCIAL_PARAMS || {};
  const guideway = (fp.guideway_per_km || 55e6) * p.length_km * (p.curve_cost_mult || 1.0);
  const stations = (fp.station_cost || 15e6) * p.n_stops;
  const fleet = fleetSize(p.length_km, p.n_stops);
  const vehicles = (fp.vehicle_cost || 3e6) * fleet;
  const systems = fp.systems_fixed || 15e6;
  const direct = guideway + stations + vehicles + systems;
  const profServices = direct * (fp.prof_services_rate || 0.10);
  const escFactor = Math.pow(1 + (fp.construction_escalation_rate || 0.045),
                             (fp.construction_period_years || 4) / 2);
  const escalation = (direct + profServices) * (escFactor - 1);
  const barriers = p.barrier_cost_usd || 0;
  return {
    guideway, stations, vehicles, systems, profServices, escalation, barriers,
    total: (direct + profServices) * escFactor + barriers,
    fleet,
  };
}
```

---

## Stage 3

**Script**: `scripts/run_full_evaluation.py` → `apm_corridor_evaluation_integrated.py`
**Time**: ~2-5 min for top 8 corridors per scenario
**Input**: Stage 2 feedback loop results CSV + corridor GeoJSON
**Output**:
- `data/processed/FULL_evaluation_{scenario}_uncertainty_metadata.json`
- `data/processed/FULL_evaluation_{scenario}.csv` — per-corridor financials:
  corridor_id, daily_ridership, debt_coverage_ratio, financially_viable,
  npv, irr, cost_per_rider, tif_revenue_cumulative, annual_tif_revenue,
  annual_farebox, annual_om, annual_debt_service, capex_musd,
  p10/p50/p90 ridership, p10/p50/p90 DSCR

**Corridor count**: Top 5-8 by Stage 2 DSCR (per scenario)

### Filtering logic

After Stage 2 completes, select corridors for Stage 3:

```python
# In run_full_evaluation.py main():
feedback_df = pd.read_csv(args.dynamic_ridership_path)
# Get final-year DSCR from inline financials (or ridership as proxy)
final_year = feedback_df.groupby("corridor_id").last()
top_n = final_year.nlargest(args.top_n, "daily_riders")["daily_riders"].index.tolist()
# Filter evaluation to top_n only
```

Add `--top-n` argument (default 8) to `run_full_evaluation.py`. When set to 0
or omitted with `--all`, evaluate everything.

### What the viewer shows after Stage 3

Stage 3 output must be merged back into the viewer data. Currently this
requires re-running the viewer embedding step. The cleanest approach:

**Option A** (recommended): `run_full_evaluation.py` writes a sidecar JSON
(`evaluation_overlay.json`) that the viewer loads alongside the feedback data:

```json
{
  "current_zoning": {
    "C1": {
      "p10_ridership": 9200,
      "p50_ridership": 12500,
      "p90_ridership": 15800,
      "p10_dscr": 0.28,
      "p50_dscr": 0.38,
      "p90_dscr": 0.52,
      "irr": -0.02,
      "npv_musd": -420,
      "uncertainty_n_draws": 500
    }
  }
}
```

**Option B**: Re-run the viewer embedding step after Stage 3, passing
evaluation results to `_build_scenario_data()` via `evaluation_df`. This
already works — `_build_scenario_data` merges evaluation fields when
`evaluation_df` is provided. But it requires re-reading all feedback data
and re-embedding the entire HTML.

#### Viewer changes for Stage 3

**Uncertainty bands on ridership trajectory chart**:

When evaluation overlay data is available, draw p10-p90 bands behind the
ridership trajectory line:

```javascript
// In lineChartSVG, add shaded band before the main line:
if (p10Data && p90Data) {
  const bandPts = p10Data.map((v, i) => {
    const x = padL + (i/(p10Data.length-1))*cw;
    const y10 = padT + ch - (v/max)*ch;
    return `${x},${y10}`;
  }).join(' ');
  const bandPtsRev = p90Data.map((v, i) => {
    const x = padL + (i/(p90Data.length-1))*cw;
    const y90 = padT + ch - (v/max)*ch;
    return `${x},${y90}`;
  }).reverse().join(' ');
  svg += `<polygon points="${bandPts} ${bandPtsRev}"
           fill="${color}" opacity="0.15"/>`;
}
```

**DSCR with uncertainty in detail grid**:

Currently shows `DSCR (min): 0.35x`. With Stage 3 data, show:
```
DSCR: 0.38x
p10: 0.28x | p90: 0.52x
```

Color the cell based on p90 (best case):
- p90 >= 1.25: green (viable even in pessimistic scenarios)
- p90 >= 1.0: yellow (viable in optimistic scenarios)
- p90 < 1.0: red (not viable in any scenario)

**Probability of viability badge**:

From Monte Carlo draws, compute P(DSCR >= 1.0) and display as a percentage:
```
Viability: 12% chance
```

This is more informative than a binary viable/not-viable flag.

**NPV and IRR display**:

Add to the detail grid:
```
NPV: -$420M          IRR: -2.1%
(p10: -$580M, p90: -$260M)
```

---

## Stage 4

**Scripts**:
- `scripts/generate_decision_package.py` — narrative summary + charts
- `scripts/compute_economic_impact.py` — jobs, tax revenue, GDP impact
- `scripts/run_behavioral_sensitivity.py` — parameter sensitivity tornado

**Time**: ~5-15 min total
**Input**: Stage 2 + Stage 3 results
**Output**:
- `data/processed/decision_package/` — per-corridor summary files
- Economic impact estimates
- Sensitivity tornado data

**Corridor count**: Top 3-5 only (presentation-ready outputs)

### What the viewer shows after Stage 4

Stage 4 outputs are primarily for export (PDF, slides). However, two elements
should feed back into the viewer:

**Economic impact summary** in the detail panel:

```
Economic Impact (25-year cumulative)
  Direct jobs created:     2,400
  Indirect/induced jobs:   1,800
  Property tax increment:  $45M
  Sales tax increment:     $12M
  GDP contribution:        $380M
```

This goes in a new section of the detail panel, below the existing financial
metrics. Only shown for corridors that have Stage 4 output.

**Sensitivity tornado** as an expandable chart:

Show which parameters have the most impact on DSCR for the selected corridor:
```
ridership_multiplier    ████████████████  ±0.12x
capital_cost_multiplier ██████████        ±0.08x
tif_multiplier          ████████          ±0.06x
operating_cost_mult     ████              ±0.03x
fare_multiplier         ███               ±0.02x
discount_rate_delta     ██                ±0.01x
```

This helps decision-makers understand which assumptions matter most.

---

## Viewer Architecture Changes

### Data loading

Currently the viewer loads all data at startup from embedded JavaScript
variables. With the overlay approach, the loading sequence becomes:

```javascript
// 1. Core data (always present after Stage 1)
const corridorsGeoJSON = EMBEDDED_CORRIDORS || await fetch('apm_phase2a_corridors.geojson').then(r => r.json());
const stopsGeoJSON = EMBEDDED_STOPS || await fetch('apm_phase2a_stops.geojson').then(r => r.json());

// 2. Feedback data (present after Stage 2)
const feedbackData = EMBEDDED_FEEDBACK || await fetch('feedback_loop_viewer_data.json').then(r => r.json()).catch(() => null);

// 3. Evaluation overlay (present after Stage 3)
const evalOverlay = EMBEDDED_EVALUATION || await fetch('evaluation_overlay.json').then(r => r.json()).catch(() => null);

// 4. Economic impact overlay (present after Stage 4)
const econOverlay = EMBEDDED_ECONOMIC || await fetch('economic_impact_overlay.json').then(r => r.json()).catch(() => null);

// Determine which stage has completed
const stage = econOverlay ? 4 : evalOverlay ? 3 : feedbackData ? 2 : 1;
```

Each stage adds capabilities to the viewer without breaking earlier stages.
The viewer degrades gracefully: Stage 1 shows static estimates, Stage 2 adds
trajectories, Stage 3 adds uncertainty, Stage 4 adds economic impact.

### Tab visibility by stage

| Tab | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| Ranking | Static estimates | Trajectories + sparklines | + uncertainty bands | Same |
| Finance | Capital cost only | + DSCR, revenue breakdown | + p10/p50/p90, NPV, IRR, viability % | + sensitivity |
| Bus Network | Hidden | Full | Same | Same |
| Development | Hidden | Full | Same | + economic impact |
| Equity | Hidden | Full | Same | Same |

### Embedding pipeline

The embedding step in `run_feedback_loop.py` (`_embed_feedback_in_viewer`)
should be extended to also embed Stage 3/4 overlays when they exist:

```python
def _embed_all_overlays(output_dir: Path):
    """Embed any available overlay files into the viewer HTML."""
    overlays = {
        "EMBEDDED_EVALUATION": output_dir / "evaluation_overlay.json",
        "EMBEDDED_ECONOMIC": output_dir / "economic_impact_overlay.json",
    }
    # Same pattern as existing GeoJSON embedding...
```

This means re-running `--serve` after Stage 3 or 4 automatically picks up the
new data. Alternatively, the viewer could fetch overlay files via HTTP when
served (no embedding needed for `python -m http.server` usage).

---

## Summary: Full Pipeline Command Sequence

```bash
# Stage 1 (~5 min)
python scripts/optimized_corridor_search.py --seeds 3 --output 25

# Stage 2a (~3 min)
python scripts/run_feedback_loop.py --screening --adaptive-stop

# Stage 2b, all scenarios (~60 min)
python scripts/run_feedback_loop.py \
  --all-scenarios \
  --screen-results data/processed/screening_survivors.json \
  --serve

# Stage 3, top 8 per scenario (~15 min)
python scripts/run_full_evaluation.py \
  --scenario current_zoning --ridership-source dynamic --top-n 8
python scripts/run_full_evaluation.py \
  --scenario no_zoning --ridership-source dynamic --top-n 8

# Stage 4, top 3 (~5 min)
python scripts/generate_decision_package.py --top-n 3
python scripts/compute_economic_impact.py

# Re-serve with all overlays
python scripts/run_feedback_loop.py --serve  # re-embeds overlays
```

**Total time**: ~85 min for full pipeline (Stages 1-4).
**Baseline (Stage 2b only)**: ~20 min.
**Quick check (Stage 2a only)**: ~3 min.

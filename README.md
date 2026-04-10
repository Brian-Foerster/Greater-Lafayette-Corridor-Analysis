# Greater Lafayette APM Corridor Evaluation

**[Interactive Viewer](https://brian-foerster.github.io/Tippecanoe-Urbansim/viewer.html)** |
**[Project Website](https://brian-foerster.github.io/Tippecanoe-Urbansim/)** |
**[FAQ](docs/TECHNICAL_FAQ.md)**

An UrbanSim fork repurposed as a land-use/transport interaction model for
evaluating Automated People Mover (APM) and Bus Rapid Transit (BRT) corridors
in Greater Lafayette, Indiana.

## What This Does

The model generates candidate transit corridors, simulates 25 years of
land-use and ridership feedback, stress-tests results under uncertainty, and
produces a decision-ready package comparing corridors across financial,
ridership, and equity dimensions.

### Pipeline Stages

| Stage | What happens | Key script |
|-------|-------------|------------|
| **1** | Generate ~25 candidate corridors via DP station selection on the road network | `scripts/optimized_corridor_search.py` |
| **2a** | Screen corridors on cost, ridership, and geometric feasibility | (internal to feedback loop) |
| **2b** | 25-year dynamic feedback loop (ridership ↔ development ↔ bus restructuring) | `scripts/run_feedback_loop.py` |
| **3** | Dynamic financial evaluation, BRT comparison, Monte Carlo uncertainty, stress testing | `scripts/apm_corridor_evaluation_integrated.py` |
| **4** | Generate decision package with maps, financials, and equity metrics | `scripts/generate_decision_package.py` |

### Scenarios

- **current_zoning** — existing Tippecanoe County UZO lot coverage, height,
  and use restrictions
- **no_zoning** — all FAR caps and dwelling-unit-per-acre limits removed
  (market determines building size, up to ~10 stories)

## Installation

Requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Or with pinned versions for exact reproducibility:
```bash
pip install -r requirements.txt
```

See [DATA.md](DATA.md) for data file acquisition before running.

## Running

Stages run sequentially — each depends on the previous stage's outputs.

**Stage 1 — corridor search:**
```bash
python scripts/optimized_corridor_search.py
```

**Stage 2b — feedback loop (both zoning scenarios + BRT comparison):**
```bash
python scripts/run_feedback_loop.py --all-scenarios --brt-compare --serve
```

**Stage 3 — financial evaluation + uncertainty (must run after Stage 2b):**
```bash
python scripts/apm_corridor_evaluation_integrated.py
```
Reads `feedback_loop_results_{scenario}.csv` from Stage 2b. Computes
demand-responsive O&M, evaluates BRT alternative, runs 500-draw Monte Carlo
with Gaussian copula correlation, stress tests, and regenerates the viewer.

**Stage 4 — decision package:**
```bash
python scripts/generate_decision_package.py
```

Both `optimized_corridor_search.py` and `run_feedback_loop.py` auto-generate
`data/processed/parcels_enriched_final.geojson` from raw parcels + zones if it
doesn't exist.

## Project Layout

```
src/                            Core model modules
  land_use_transport_model.py     25-year feedback loop engine
  ridership_engine.py             5-component ridership computation
  mode_choice.py                  MNL mode choice (canonical coefficients)
  bus_network.py                  Bus headway, APM service, cost model
  bus_service_planning.py         Proactive bus restructuring engine
  finance.py                      NPV/IRR, TIF revenue
  financial_params.py             Cost constants, TransitMode dataclass
  model_constants.py              Calibration constants
  demand_driven_development.py    Demand-driven development model
  relocation_model.py             Household location choice MNL
  vacancy_rent_feedback.py        Vacancy-driven rent adjustment
  developer_proforma.py           Zoning matrix, market config
  economic_impact.py              Economic/equity impact metrics
  spatial_constants.py            CRS, catchment radii
  feeder_route_generator.py       Synthetic feeder bus routes
scripts/                        Pipeline entry points
  optimized_corridor_search.py    Stage 1: corridor generation (NSGA-II + DP)
  corridor_geometry.py            Curve physics, road-graph routing
  corridor_evolution.py           NSGA-II selection and mutation
  run_feedback_loop.py            Stage 2: feedback loop orchestrator
  apm_corridor_evaluation_integrated.py  Stage 3: financial evaluation
  generate_decision_package.py    Stage 4: decision package
  generate_improved_ridership.py  LODES ridership and mode choice
tests/                          Test suite (pytest, ~486 tests)
urbansim/developer/             Upstream SqFtProForma (only library dependency)
data/raw/                       Input data (parcels, zones, GTFS, LODES)
data/processed/                 Generated outputs (gitignored)
docs/                           Website and documentation
  index.html                      Landing page (GitHub Pages)
  viewer.html                     Interactive corridor viewer
  faq.html                        FAQ page
  TECHNICAL_FAQ.md                FAQ source (plain-language + technical + glossary)
scenarios_config.json           Scenario definitions
pyproject.toml                  Package metadata and dependencies
Makefile                        make install / test / run / deploy / clean
```

## Tests

```bash
pytest tests/ -x -q
```

Fast unit tests run in under 2 seconds. Tests prefixed `test_week2*` are
heavyweight smoke tests that spawn full pipeline runs — skip unless needed.

## Key Design Decisions

- **Online data only** — no surveys or field data; parcels from county GIS,
  jobs from LODES, transit from GTFS
- **Income-segmented mode choice** — MNL weighted by LODES SE01/SE02/SE03
  earnings segments
- **Two-layer catchment** — walk zone (0–800 m) with 4-mode MNL, feeder zone
  (800–7000 m) with bus-to-APM transfer
- **100% local funding assumption** — no federal match; TIF with Indiana
  IC 36-7-14 adjustments (SB 1 erosion, circuit breaker, assessment lag)
- **Student population segment** — enrollment-based campus ridership with
  seasonal adjustment (2.28× academic/summer ratio)

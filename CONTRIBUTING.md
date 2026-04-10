# Contributing

This is the Greater Lafayette APM/BRT corridor evaluation model. It uses a
custom land-use-transport feedback loop to evaluate automated people mover and
bus rapid transit corridors for the Lafayette, Indiana metro area.

## Architecture overview

The pipeline runs in four stages:

| Stage | Script | What it does |
|-------|--------|-------------|
| 1 | `scripts/optimized_corridor_search.py` | NSGA-II station-first corridor search (~15-25 iterations) |
| 2 | `scripts/run_feedback_loop.py` | 25-year land-use-transport feedback loop (per-corridor, per-scenario) |
| 3 | `scripts/apm_corridor_evaluation_integrated.py` | Financial evaluation (TIF, debt service, O&M, DCR) |
| 4 | `scripts/generate_decision_package.py` | Decision package generation (ranking, maps, appendix) |

Key design decisions:

- **Corridor independence** — each corridor is evaluated in isolation with
  baseline state snapshot/restore between corridors
- **Online-data-only** — no surveys or field data; all inputs are from public
  sources (LODES, ACS, GTFS, county GIS)
- **Two scenarios** — `current_zoning` (existing FAR limits) and `no_zoning`
  (FAR caps removed), each with APM and BRT mode comparison

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -e .
```

For development (adds pytest-cov and pandana):
```bash
pip install -r requirements-dev.txt
```

## Running the pipeline

```bash
# Stage 1: corridor search
python scripts/optimized_corridor_search.py --iterations 15 --output 40

# Stage 2: feedback loop (all scenarios + BRT comparison + viewer)
python scripts/run_feedback_loop.py --all-scenarios --brt-compare --serve

# Stage 3: financial evaluation
python scripts/apm_corridor_evaluation_integrated.py
```

All output goes to `data/processed/` (gitignored, fully regenerable).

## Tests

Tests live in `tests/` (not `urbansim/tests/`, which is the upstream library's
test suite and is not relevant to the APM project).

```bash
python -m pytest tests/ -q                  # fast unit tests (~15s)
python -m pytest tests/ -q -m "not slow"    # skip heavy smoke tests
```

Key test files:
- `test_model_integration.py` — end-to-end model with synthetic data
- `test_finance.py` — TIF, debt service, O&M calculations
- `test_mode_choice_*.py` — MNL mode choice coefficients
- `test_bus_network.py` — bus restructuring and headway optimization
- `test_proforma_developer.py` — SqFtProForma feasibility

## Source layout

- `src/` — model core (importable as `from src.X import Y`)
- `scripts/` — pipeline scripts (importable as `from scripts.X import Y`)
- `urbansim/developer/` — upstream `SqFtProForma` (only library dependency)
- `data/raw/` — input data (parcels, zones, GTFS, LODES)
- `data/processed/` — all pipeline outputs (gitignored)

## Contributing code

- Branch from `dev`, open PRs against `dev`
- Run `python -m pytest tests/ -q` before pushing
- Follow existing code style (no linter enforced, but keep it consistent)
- Add tests for new functionality in `tests/`

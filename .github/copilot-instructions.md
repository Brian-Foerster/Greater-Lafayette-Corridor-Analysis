# Copilot Instructions for UrbanSim APM Corridor Analysis

This codebase is a **specialized urban planning analysis** for evaluating Automated People Mover (APM) corridor viability. It is NOT a production UrbanSim library—it's a research project grounded in the UrbanSim platform but focused on a single infrastructure question.

## Project Context

**Mission:** Identify the optimal APM corridor in a given city by evaluating 17 road-following corridors using calibrated demand models, mode choice analysis, and a 25-year land use/transport feedback loop.

**Key Phases:**
- **Phase 1 (Complete):** Baseline corridor search with synthetic demand, uniform competition, fixed stop spacing → Recommendation: C23 (11,301 daily riders)
- **Phase 2a (In Progress):** Reality-grounded improvements (gravity model, time-of-day variation, zone-based bus competition) → Expected: C20 (11,491 riders) with HIGH confidence
- **Phase 2b (Planned):** OD flow integration from validation survey → VERY HIGH confidence (if executed)
- **Phase 3 (Planned):** Environmental, equity, financial analysis for implementation planning

**Read [README_START_HERE.md](../README_START_HERE.md) for documentation navigation.**

## Essential Architecture

### Model Realism: 5 Issues, 5 Levels of Improvement

| Level | Issue | Status | Script | Complexity |
|-------|-------|--------|--------|------------|
| 1 | No time-of-day variation | ✅ Done | `improved_demand_model_v2.py` L35-85 | <1 min |
| 2 | Uniform bus competition | ✅ Done | `improved_demand_model_v2.py` L92-155 | 1-2 min |
| 3 | No gravity model | ✅ Done | `improved_demand_model_v2.py` L170-220 | 5-10 min |
| 4 | Stop spacing dilution | ⏳ Framework | `improved_demand_model_v2.py` L292-345 | 30+ min |
| 5 | Uniform synthetic demand (OD flows) | ⏳ Planned | Need validation survey first | 8+ hrs |

**Critical Pattern:** Each level addresses a specific realism gap. Higher levels are more computationally expensive but increase confidence. Phase 2a implements L1-3 (15 min total). Phase 2b would add L5 (OD validation survey).

### Data Pipeline

```
Parcels + LODES + GTFS → Corridor Search → Feedback Loop → Results
  (GIS + Census)         (17 corridors)   (25-year sim)    (Ranked)
       ↓                      ↓                ↓               ↓
  parcels_enriched    apm_phase2a_corridors  ridership      financial
  zones, OD flows     apm_phase2a_stops      development    viability
```

**Key Files:**
- `scripts/improved_demand_model_v2.py`: All demand model improvements (L1-5 frameworks)
- `scripts/iterative_apm_search_phase2a.py`: Runs corridor search WITH Phase 2a improvements
- `scripts/iterative_apm_search_improved.py`: Base search algorithm (don't modify lightly)
- `data/processed/`: All intermediate data (trips, parcels, competitors)

### Mode Choice Logic

UrbanSim uses **Multinomial Logit (MNL) with 4 modes:** APM, Car, Bus, Walk

```python
Mode Utility = Intercept + β₁×Travel_Time + β₂×Cost + β₃×Reliability + ...
Mode Probability = exp(Utility) / Σ(exp(Utilities))
```

**Project Override:** Phase 2a replaces Phase 1's generic national coefficients with local **time-period specific** evaluation:
- AM Peak (7-10am): 1.40x demand concentration
- PM Peak (4-7pm): 1.35x demand concentration  
- Off-Peak: 0.80x demand (spread out)

**Don't:** Assume uniform demand across hours. **Do:** Check `time_period` column in trip data.

## Developer Workflows

### Run Phase 2a Pipeline (Quick Integration Test)
```bash
# Activate environment
source .venv/bin/activate

# 1. Generate improved demand data (Levels 1-3)
python scripts/improved_demand_model_v2.py
# Output: synthetic_trips_improved_v2.csv, parcels_improved_gravity_v2.geojson

# 2. Run corridor search with improvements
python scripts/iterative_apm_search_phase2a.py
# Output: corridor_results_phase2a.csv, phase2a_comparison.csv

# Total runtime: ~11 minutes
```

### Validate Demand Model Changes
- Check `data/processed/synthetic_trips_improved_v2.csv` columns: `time_period`, `demand_factor`, gravity weights
- Spot-check: AM peak trips should concentrate 7am-10am; off-peak should flatten
- Verify: Gravity weights correlate with employment data in `parcels_improved_gravity_v2.geojson`

### Add a New Improvement Level
1. **Update `improved_demand_model_v2.py`** with new function (follow L1-3 structure)
   - Include clear docstring with computational complexity
   - Add before the "return" statement that saves data
2. **Update `iterative_apm_search_phase2a.py`** to apply new weights to corridor search
3. **Document in [REALITY_GROUNDING_ROADMAP.md](../REALITY_GROUNDING_ROADMAP.md)** with timeline and impact estimate
4. **Don't modify `iterative_apm_search_improved.py`** directly—wrap it, don't change the base algorithm

## Project-Specific Conventions

### Naming & Conventions
- **Corridors:** `C1`, `C2`, ... `C17` (road-following corridors from NSGA-II + cost-surface routing)
- **Phases:** Phase 1 (done), Phase 2a (current), Phase 2b (conditional OD survey), Phase 3 (planning)
- **Confidence Levels:** MEDIUM-HIGH (Phase 1) → HIGH (Phase 2a) → VERY HIGH (Phase 2b)
- **Computation Tiers:** Easy (<1min), Medium (1-5min), Hard (5-30min), Very Hard (30+min), Extreme (8+hrs)

### Critical Decision Points (Require User Approval)
1. **Week 8:** Accept Phase 2a HIGH confidence OR pursue Phase 2b for VERY HIGH (adds 12 weeks)
2. **If Phase 2b:** Three methodology choices (OD matrix type, distance decay, mode choice re-estimation)
3. **Phase 3:** Environmental/equity/financial analysis needed before implementation

**These are NOT technical decisions—they're project scope decisions. Don't simplify Phase 2b without explicit approval.**

### Uncertainty Documentation
Every comparison must include confidence intervals or uncertainty ranges:
- Phase 1: MEDIUM-HIGH confidence (synthetic demand)
- Phase 2a: HIGH confidence (realistic demand distribution)
- Phase 2b: VERY HIGH confidence (empirical OD patterns)

**Example (from code):**
```
C1: 8,922 ± 500 riders/day (25-year feedback loop, current zoning)
```

## External Dependencies & Integration Points

### UrbanSim Library
This project imports from the base `urbansim` package (DCM, mode choice models). **Don't assume** UrbanSim handles all demand modeling—it doesn't. That's why we built `improved_demand_model_v2.py`.

### Key Data Sources
- **Population synthesis:** Synthetic person/household microdata (NHTS-based)
- **Employment data:** `parcels_enriched.geojson` with job estimates
- **Transit network:** `CityBusOld/stops.txt` (existing bus stops for competition modeling)
- **Validation:** `validation_final.json` (pilot survey data for Phase 2a)

### Spatial Data Format
All geographic data uses **GeoJSON (WGS84, EPSG:4326)**. If adding new spatial analysis:
- Use `geopandas` (not raw shapely)
- Save as `.geojson`, not `.shp` (easier to share)
- Include all attributes needed for downstream analysis

## Testing & Validation

### Before Making Changes
1. Run Phase 2a pipeline end-to-end (`improved_demand_model_v2.py` → `iterative_apm_search_phase2a.py`)
2. Compare output to Phase 1 baseline: Does Phase 2a show expected improvement (C20 > C23)?
3. Check numerical stability: Are confidence intervals reasonable? Any NaN/Inf?

### Documentation Checks
- If modifying a level (L1-5), update both the code docstring AND [REALITY_GROUNDING_ROADMAP.md](../REALITY_GROUNDING_ROADMAP.md)
- If adding a phase decision, flag it in roadmap with explicit approval requirement
- If changing computational complexity, update the summary table

## Red Flags & Warnings

**Don't do these without explicit discussion:**
1. **Simplify the demand model** (e.g., ignore gravity weighting to "speed up")—This was Phase 1's mistake
2. **Use uniform competition penalties** instead of zone-based—Removes geographic realism
3. **Skip validation survey** and proceed with synthetic OD—Contradicts "data-driven" commitment
4. **Change the corridor search algorithm** without documenting impact—Base algorithm is stable; wrap it
5. **Accept Phase 2a results without understanding the improvements**—Understand what gravity + time-of-day + competition actually changed

## Key Files to Understand

| File | Purpose | Read When |
|------|---------|-----------|
| [REALITY_GROUNDING_ROADMAP.md](../REALITY_GROUNDING_ROADMAP.md) | 12-week strategic plan with decision gates | Planning new work |
| [COMPLETE_WORK_SUMMARY.md](../COMPLETE_WORK_SUMMARY.md) | What was accomplished in Phase 2a | Getting project status |
| [MODEL_IMPROVEMENTS_REFERENCE.md](../MODEL_IMPROVEMENTS_REFERENCE.md) | Technical reference for all 5 levels | Implementing improvements |
| `scripts/improved_demand_model_v2.py` | Demand model L1-5 frameworks | Modifying demand logic |
| `scripts/iterative_apm_search_phase2a.py` | Phase 2a corridor search runner | Running analysis |
| `scripts/iterative_apm_search_improved.py` | Base search algorithm | Understanding corridor evaluation |

---

**Last Updated:** January 12, 2026 | **Phase:** 2a (Phase 2b pending decision) | **Confidence:** HIGH

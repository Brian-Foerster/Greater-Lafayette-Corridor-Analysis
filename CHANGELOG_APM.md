# APM Corridor Evaluation - Changelog

## 1.0.0-apm (2026-03-22)

Initial release of the Greater Lafayette APM corridor evaluation pipeline,
built as a fork of UrbanSim 3.2.

### Pipeline Architecture
- Four-stage evaluation pipeline: Stage 1 (corridor search), Stage 2a
  (screening), Stage 2b (25-year feedback loop), Stage 3 (uncertainty),
  Stage 4 (decision package)
- Per-corridor independent evaluation with baseline snapshot/restore
- Annual time steps (years 0-25) with multi-year delivery pipeline
- Two-scenario framework: `current_zoning`, `no_zoning`
- Unified CLI entry point (`urbansim-apm`) with subcommands for each stage

### Corridor Search (Stage 1)
- NSGA-II multi-objective evolutionary optimization (ridership, cost
  efficiency, DCR estimate) with configurable population size and
  generations
- Dense core corridor generator targeting short, high-density corridors
  (2-5km, 3-7 stations)
- Dynamic programming station selection with demand maximization
- MMR (Maximal Marginal Relevance) diversity selection with bidirectional
  geographic overlap
- Blended quality score: 60% ridership density + 40% total ridership
- Minimum ridership floor filter (2,000 riders/day)
- Enhanced TIF estimation with existing AV uplift + developable increment

### Feedback Loop (Stage 2b)
- Integrated land use / transport model with 25-year annual simulation
- Demand-driven development model (DiPasquale-Wheaton gap targeting)
- Tier 2 development: relocation MNL, SqFtProForma (non-student
  residential), formula-driven students (Option C), proforma commercial
- Per-parcel vacancy-rent feedback with asymmetric stickiness
- County-capacity cost escalation and developer confidence ramp
- Congestion feedback: car speed degrades with population growth
  (CONGESTION_ELASTICITY = 0.30)
- Regional growth reallocation (elasticity = 0.05)
- Demand-responsive APM headway (two-regime continuous model)
- Convergence detection: 1% ridership, 2% development, 3 consecutive
  periods

### Ridership Model
- Income-segmented mode choice MNL (SE01/SE02/SE03 from LODES)
- Two-layer catchment: walk zone (0-1200m, 4-mode MNL) and feeder zone
  (1200-7000m, 5th APM+feeder mode)
- Location-sensitive parking costs (campus $0.40, downtown $4.00,
  suburban $0.00)
- Student population segment with enrollment-based campus catchment
- Latent demand model: zero-car HH suppressed trips with awareness ramp
- Five ridership components: work commute, local non-work, campus,
  destination, equity
- EPSG:3857 Mercator correction on all distances
- Seasonal adjustment: 2.28x academic/summer ratio, annual average
  factor 0.860

### Financial Model
- Fare $2.00, capital $55M/km guideway + $15M/station, O&M $2.5M +
  $350K/km/yr
- 25-year debt at 5% bond rate, 2.32% property tax rate
- TIF capture: statutory 1.00, conservative 0.85
- 100% local funding assumption
- DCR, NPV, total coverage ratio, self-sufficiency metrics

### Bus Network Integration
- Per-route headway optimization with spatial classification
  (parallel/feeder/independent)
- Budget-constrained restructuring ($13.5M CityBus annual budget)
- Reactive and proactive restructuring modes
- Tier 3: productivity scoring, Title VI equity guard

### Sensitivity & Uncertainty (Stage 3)
- Behavioral LHS with GP surrogate (Matern nu=2.5 kernel, joblib
  persistence)
- Warm-start batch runner for LHS parameter exploration
- Gaussian copula Monte Carlo with 11-parameter correlation structure
- Scenario-specific 11x11 correlation matrices
  (ridership-TIF coupling varies by zoning regime)
- Vectorized 500-draw simulation (numpy broadcasting, no Python loops)
- Data-calibrated parameter distributions with documented empirical
  sources (FTA Before & After, NTD, FRED, Lincoln Institute)
- Congestion elasticity as continuous sampled parameter
- Robust corridor ranking: P(top-k), CVaR/expected shortfall, max
  regret (composite 0.40/0.35/0.25 weights)
- Named stress scenarios: enrollment shock, cost blowout, transit boom,
  stagflation
- Behavioral validation gate with calibration feedback loop (damped mode
  adjustment from TCRP/NHTS benchmark deviations)

### Decision Package (Stage 4)
- Automated report generation with corridor maps, financial tables,
  uncertainty bands
- Economic impact and equity analysis modules
- FTA cost-effectiveness metrics

### Reproducibility
- Environment fingerprinting: Python version, OS, dependency hashes, git
  state
- Artifact manifest with SHA-256 checksums and row counts
- Source manifest validation with staleness checks
- Deterministic run IDs with git commit, timestamp, scenario tags

### Data Pipeline
- GeoJSON-based parcel system (81K parcels, STKEY-based IDs)
- Auto-enrichment from raw parcels + zones
  (`_ensure_enriched_parcels()`)
- LODES WAC/OD integration for employment and commute patterns
- CityBus GTFS for existing bus network
- OSM road network (GraphML) for routing
- Fiona bypass for GeoJSON loading (json module, performance fix)
- Vectorized 8-sector feeder loops with `np.bincount`/`np.dot`

### Configuration
- `scenarios_config.json` with full model_options, uncertainty framework,
  stress scenarios, and correlation matrices
- CLI flags for all model_options (transit mode, bus restructuring, equity,
  alternatives analysis, behavioral parameters)
- Per-scenario development parameters (zoning cost adjustment, corridor
  capture rates)

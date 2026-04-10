# Greater Lafayette APM Corridor Evaluation Model

## Technical Description for Peer Review

---

## 1. Project Goal

This model evaluates 40 candidate corridors for an Automated People Mover (APM) system serving the Greater Lafayette, Indiana metropolitan area (population 232,000; Tippecanoe County). The study area is anchored by Purdue University (54,651 students, 15,396 faculty/staff) and the twin cities of West Lafayette and Lafayette, separated by the Wabash River.

The model couples land use and transport in a 25-year annual feedback loop: ridership drives development, development drives population and employment growth near stations, and that growth feeds back into ridership. Each of 40 corridors is evaluated independently with state snapshot/restore between corridors to prevent inter-corridor competition bias. Two zoning scenarios (current zoning and unconstrained FAR) are threaded through the loop, producing scenario-differentiated ridership, development, financial viability, and equity outcomes.

The model operates under an **online-data-only constraint**: no travel surveys, stated preference data, traffic counts, or field collection were available. All behavioral parameters are calibrated from published research (primarily TCRP reports) and all spatial/demographic data come from public administrative sources. This constraint is a deliberate design choice reflecting the project's role as a planning-level screening tool rather than an FTA New Starts submission model.

---

## 2. Data Sources

### 2.1 Spatial Data

**Parcels**: ~61,593 parcels from the Tippecanoe County WFS endpoint (Schneider Corp GIS). Each parcel carries PARCEL_ID (STKEY tax key), geometry, and assessed values from the county assessor: CurLandAV, CurImpAV, CurTotAV. Building-level improvement data (ClassCode, Grade, finished living area) comes from a separate improvements layer, joined by STKEY. Sales transactions (2021–2025) are linked by STKEY with spatial-join fallback for unmatched records.

**Zoning**: Zone geometries with RefName codes, joined to a density estimates table providing FAR and height limits by zone classification. Parcels are assigned zone codes via spatial join (centroid-in-polygon). PropClass 600-series parcels (educational, religious, government) are marked exempt from development — 4,663 parcels including the Purdue campus core.

**Road network**: OSM via osmnx — 5,423 nodes, 13,641 edges. Used for corridor alignment (shortest-path routing with road-class cost surface), station siting (candidate stations at degree-≥3 intersections on primary/secondary/tertiary roads), and barrier identification. The Wabash River is represented by 43 bridge edges connecting 139 bridge-flagged edges. 229 traffic signal nodes are identified for potential TSP modeling (not yet implemented).

**TIF districts**: Existing Tax Increment Financing boundaries from county GIS. Used to determine which parcels fall within potential TIF capture zones for each corridor.

### 2.2 Employment and Commute Data

**LODES**: Longitudinal Employer-Household Dynamics (LEHD) Origin-Destination Employment Statistics, workplace-area characteristics (WAC) file for Tippecanoe County. Provides block-level employment by income segment: SE01 (<$1,250/month), SE02 ($1,250–$3,333/month), SE03 (>$3,333/month). Total county employment: 96,546 jobs (2023 WAC C000). OD flows are aggregated to parcel level using Census block-to-parcel spatial correspondence.

**Synthetic population**: Generated from ACS block group data. Provides household-level attributes including household size (mean 2.56, reflecting TOD-oriented family composition rather than dorm-inflated campus averages) and zero-vehicle household rates by income segment.

### 2.3 Transit Data

**GTFS**: CityBus 2025 feed. Routes, stops, stop_times, calendar, and shapes. Used to build per-route BusRoute objects with observed headways, span, and cycle times. System-wide: 5,982 daily boardings; Route 4B (Purdue campus) is the highest-ridership route at 2,997 daily boardings.

### 2.4 Institutional Data

**Purdue enrollment**: 54,651 students (Fall 2025 headcount, Purdue Data Digest). 15,396 faculty/staff. Campus boundary from OpenStreetMap Overpass API with 500m adjacency buffer. Building-level institutional weights derived from ClassCode analysis of the improvements layer: university buildings (ClassCode 699) receive weight 5.0; large multi-family student housing (ClassCode 520/530/550/551 within 2km of campus centroid) receives 3.0–4.0; single-family near campus receives 1.5–2.5.

---

## 3. Corridor Search and Station Placement

### 3.1 Candidate Station Identification

Candidate APM station locations are extracted from the OSM road graph: intersections with degree ≥ 3 (in the undirected sense) that are adjacent to at least one primary, secondary, or tertiary road. This produces approximately 500–1,000 candidate locations across the study area. Each candidate is pre-scored for demand coverage (distance-decay weighted parcel demand within 1,200m), TIF revenue potential (assessed value in catchment), and bus competition (CityBus stop density within 400m).

### 3.2 Corridor Generation

Corridors are generated as ordered station sets through three seeding strategies:

1. **Anchor-pair corridors**: Seven anchor destinations (downtown Lafayette, Tippecanoe Mall, IU Health Arnett, Wabash riverfront, Purdue campus core, north and south termini). All pairwise combinations are connected via Dijkstra shortest path on the road graph with a road-class cost surface. Candidate stations along each path are extracted with greedy 500m minimum spacing.

2. **Demand-biased random walks**: Starting from high-demand nodes, random walks on the road graph adjacency structure, biased toward high-demand neighbors. Produces diverse corridor geometries not captured by anchor pairs.

3. **Radial corridors from campus center**: Ensures campus-serving corridors are represented in the initial population.

### 3.3 NSGA-II Optimization

A two-objective NSGA-II evolves the initial population (100 candidates) over 10–15 generations. Objectives are ridership estimate (maximize) and cost efficiency (maximize ridership per dollar of capital cost). The scoring function (`score_station_set`) evaluates each station set using:

- Distance-decay weighted demand catchment (β = 0.00173, giving 50% retention at 400m)
- A simplified 5-mode MNL (APM, bus, car, walk, bike-to-APM) using the same β coefficients as the full model
- OD-based commute ridership from LODES flows where both ends are near stations
- A student ridership proxy using institutional weights and Purdue enrollment data
- Bus competition modeled as a headway-based quality function
- Barrier crossing costs: $80M per river crossing, $40M per highway (I-65), $25M per railroad

Five genetic operators mutate station sets: swap (35%, replace one station with a nearby high-demand intersection), add (20%, insert station in largest spacing gap), remove (15%, drop lowest-demand interior station), shift (20%, move station to adjacent road graph node), and reorder (10%, reverse a segment of interior stations). All mutations are validated against spacing constraints (500–1,500m) and geometric quality (circuity ≤ 1.60, no bearing reversals > 100°).

### 3.4 Diversity Selection

Final corridors are selected using Maximal Marginal Relevance on Jaccard overlap of station catchment areas (1km neighborhoods). Maximum 50% overlap threshold ensures the output set of 17 corridors covers distinct geographic markets.

### 3.5 Alignment

Selected station sets are converted to corridor geometries via road-graph shortest path between consecutive stations, with Chaikin smoothing. Physics-based curve speed penalties (ASCE 21.2-2008 lateral acceleration model) reduce effective APM speed where curve radii are tight.

**Known limitation**: The scoring function optimizes for current demand (CurTotAV-weighted catchment) and does not account for development potential, zoning headroom, or FAR capacity. Corridors with high future development potential but low current demand are systematically undervalued.

---

## 4. Ridership Model

### 4.1 Two-Layer Catchment

The catchment is divided into two zones around each station:

**Walk zone (0–800m)**: Direct walk access to APM. Five-mode MNL: APM, bus, car, walk, bike-to-APM. Walk access time computed with WALK_CIRCUITY = 1.20.

**Feeder zone (800–7,000m)**: Bus-to-APM transfer. Fifth mode "APM+feeder" added to the MNL. Feeder bus in-vehicle time uses full parcel-to-station distance with BUS_CIRCUITY = 1.30 (Manhattan distance approximation). Transfer penalty: 8 minutes (TCRP standard for untimed transfers; 4 minutes for timed). Integrated fare: $2.00 total (no additional transfer fare). Feeder zone trips are scaled by `feeder_coverage_fraction` from the bus restructuring phase (range 0.0–0.90 depending on restructure pressure).

All distances are computed in EPSG:3857 with a Mercator correction factor of cos(40.4°) ≈ 0.763 applied to convert projected distances to ground meters.

### 4.2 Mode Choice Specification

Multinomial logit with linear-in-parameters utility:

```
V_mode = β_IVT × IVT + β_WAIT × WAIT + β_ACCESS × ACCESS + β_COST × COST + ASC_mode
```

| Parameter | Value | Source |
|-----------|-------|--------|
| β_IVT | −0.055 /min | TCRP 165 |
| β_WAIT | −0.090 /min | TCRP 165 |
| β_ACCESS | −0.120 /min | TCRP 165 |
| β_COST | −0.035 /$ | TCRP 165 |
| ASC_APM | +0.18 | New transit boost, exp(0.18) ≈ 1.20 |
| ASC_BUS | −0.10 | Crowding/unreliability penalty |
| ASC_CAR | −0.05 | Baseline slight penalty |
| ASC_WALK | +0.05 | Short-trip walk preference |

**Income segmentation**: The MNL is evaluated separately for three LODES income segments (SE01, SE02, SE03) with segment-specific coefficient multipliers and ASC shifts:

- **SE01 (low income)**: cost_mult = 1.30 (higher cost sensitivity), ivt_mult = 1.05, small transit-favorable ASC shifts
- **SE02 (mid income)**: baseline (all multipliers = 1.0)
- **SE03 (high income)**: cost_mult = 0.65 (lower cost sensitivity), ivt_mult = 0.85, car-favorable ASC shift (+0.10)

Results are weighted by segment share from LODES OD data to produce blended mode shares. This means income composition affects ridership totals, not just post-hoc disaggregation.

**Reliability-aware wait time**: For headways ≤ 10 minutes (random arrival regime), wait = headway/2. For headways > 10 minutes (scheduled arrival), wait = headway/4 + 3 minutes reliability buffer. This is applied to bus wait; APM wait currently uses raw headway/2 (consistency fix identified but not yet implemented).

**Location-sensitive parking cost**: Integrated into car utility. Campus parcels (institutional_weight > 1.0): $0.40/trip, derived from Purdue's $100/year permit over 250 working days. Downtown parcels (zone codes C\*, B-, DT, CBD): $4.00/trip, based on Lafayette parking meter and garage rates. Suburban: $0.00.

### 4.3 Ridership Components

Total daily ridership is the sum of six components:

**Component 1a — Both-ends work trips (LODES logit)**: OD pairs from LODES where both origin and destination are within the walk zone (~7.3% of catchment pairs). The MNL produces `apm_prob` per OD pair; trips are `weighted_trips × apm_prob × 2` (round trip) × `WORK_OFFPEAK_EXPANSION` (1.15, covering reverse commute and off-peak work trips; NHTS 2017 200-500K metro). Non-work travel is handled entirely by Component 1b, eliminating the double-counting that occurred with the former flat 3.0 expansion factor.

**Component 1b — Purpose-specific non-work generation (NHTS)**: Catchment residents who work outside the corridor but make local non-work APM trips. Trip rates are disaggregated by purpose from NHTS 2017 for 200-500K urbanized areas: shopping (0.65 trips/person/day), social/recreation (0.45), personal business (0.40), escort/other (0.30). Each purpose has a purpose-specific APM share multiplier applied to the corridor's commute-derived `avg_apm_share`: shopping 0.35, social 0.25, personal 0.30, escort 0.15 (STOPS guidance: non-work transit share is 15-35% of commute). Campus-affiliated population is excluded (their non-work trips are in Component 2). This approach differentiates corridors by land use context — corridors near retail generate more shopping trips — rather than applying a flat rate uniformly.

**Component 2 — Student/campus**: Campus-affiliated population is estimated as PURDUE_ENROLLMENT × 0.25 + PURDUE_FACULTY_STAFF × 0.10 ≈ 14,560. The 25%/10% presence factors represent the fraction of the campus population physically present and potentially trip-making at any given time during the academic day. This population is distributed across campus-weighted parcels using distance-decay weights, then multiplied by a corridor-specific APM share computed via 5-mode logit with student ASC adjustments (APM +0.25, bus +0.20, car −0.40, walk +0.10, reflecting low car ownership and transit habit). Campus alignment is modeled as exponential decay from the corridor to campus buildings (ALIGNMENT_DECAY = 0.000866, half-credit at 800m). Approximately 48% of total ridership for campus-proximate corridors. Seasonal factor: 0.860 (academic calendar — 2.28× ratio between academic peak and summer).

**Component 3 — Self-selection — REMOVED**: Was TCRP 128 Michaelis-Menten decay; now `self_selection_mult = 1.0` (no-op). Removed because logit already captures mode share at each location, and catchment_scale grows trip opportunities with TOD development — separate self-selection double-counted residential sorting.

**Component 4 — Non-commute generators**: Attraction-based institutional trip generators with proximity-scaled APM share. These capture trips drawn from outside the residential catchment (unlike Component 1b which is residential generation):
- IU Health Arnett hospital: 350 daily trips
- Retail employment: 0.5 trips/retail-job/day
- Purdue events (sports, theater, convocation): 550 daily equivalent (annualized from event calendar)
- Generator APM share: 12%, scaled by proximity to corridor
- No overlap deduction needed: Component 1a is work-only and Component 1b is residential-based, so neither overlaps with attraction-based generator trips.

Approximately 14% of total ridership.

**Component 5 — Induced demand**: Trip generation elasticity from TCRP 95. Applied to origin-only and generator components only (LODES commute OD pairs and student trip rates are fixed). Signal: service quality = clip(pre_induced_riders / MATURE_RIDERSHIP_TARGET, 0, 1). Elasticity: 0.10. Activates only after year 5. Approximately 3% of total at maturity.

**Component 6 — Latent demand (zero-car households)**: ACS table B08201 zero-vehicle rates by income: SE01 = 25%, SE02 = 6%, SE03 = 2%. Suppressed trips: 1.0 trip/person/day (the unmet travel demand of carless households). Release rate: 40% (fraction of suppressed trips that materialize as APM ridership when service is available). Walk zone at full rate; feeder zone at LATENT_FEEDER_RELEASE_DISCOUNT = 0.35 (zero-car households less likely to complete a bus transfer). Approximately 3% of total at maturity.

**Combination**: `base_riders = (work_1a + nonwork_1b) + student + generators + induced + latent`

**Temporal ramp**: Logistic S-curve with RAMP_MIDPOINT = 0, RAMP_STEEPNESS = 0.8, giving 50% ridership at opening day and 98% by year 5. This is calibrated to FTA New Starts before/after studies showing most systems reach near-mature ridership within 3–5 years.

**Seasonal adjustment**: Component-specific factors blended by component mix: student 0.860, work commute 0.950, non-work 0.920, generator 0.90, induced 0.92, latent 0.95.

### 4.4 APM Speed Model

Effective APM speed accounts for line-haul cruise (40 km/h), intermediate stop dwell, and acceleration/deceleration:

```
cruise_time = length_km / 40
stop_time = (n_stops - 2) × (dwell_s + 15s) / 3600
effective_speed = length_km / (cruise_time + stop_time)
```

Dwell time is demand-responsive: `dwell = min(15 + 0.01 × daily_boardings_per_station, 45)` seconds. Low-demand stations: ~17s. High-demand campus hub: ~35s. Cap at 45s (ADA + crowding). Terminal stops are excluded (turnaround absorbs dwell).

---

## 5. Bus Network Model

### 5.1 GTFS-Based Route Classification

When GTFS data is available (`--gtfs-dir` flag), CityBus routes are loaded and spatially classified relative to each APM corridor:

- **Parallel**: ≥40% of route stops within 400m of corridor buffer. Redundant with APM; candidates for frequency reduction.
- **Feeder**: 15–40% overlap. Supplementary coverage; candidates for restructuring into APM feeders.
- **Independent**: <15% overlap. Separate travel market; retained at baseline service.

Classification drives the bus restructuring decision, which in turn determines feeder coverage and bus operating savings.

### 5.2 APM Headway

Two-regime continuous model:

**Service quality regime**: `hw = 10 − 4.5 × ln(1 + riders/2000)`. At ~1,000 daily riders → 8 min; ~3,000 → 6 min; ~6,000 → 3.8 min.

**Capacity constraint regime**: `hw = 60 / (peak_pphd / effective_capacity)`. Peak period passengers per hour per direction computed as `daily_riders × 0.14 (peak hour factor) × 0.60 (directional split)`. Train capacity: 100 passengers (2 cars × 50 pax).

The binding constraint (longer headway) is used, subject to: physical minimum 90s (AGT signaling), policy maximum 10 min, fleet maximum 20 trains.

The directional split is currently a global constant (0.60) but OD flow data could support corridor-specific values (identified improvement).

### 5.3 Bus Restructuring

Restructure pressure is a continuous 0–1 score: `0.70 × (riders/mature_target) + 0.20 × (1 − competitiveness) + 0.10 × (1 − productivity)`. This drives:

- **Parallel route headway**: increases with pressure (service reduction on redundant routes)
- **Feeder route headway**: decreases with pressure (improved feeder frequency)
- **Feeder coverage fraction**: ramps from 0.0 (no feeders) through 0.30, 0.60, to 0.90 (dominant feeder network)

Budget constraint: $13.5M CityBus annual operating budget. Vehicle-hour costs: $135/hr bus, $85/hr APM. Freed service hours from parallel route reduction are reallocated to feeder service.

### 5.4 Discrete Event Years

Full bus network redesign (spatial reclassification + budget optimization) occurs only at years 0, 3, and 8 — representing opening day, early operations adjustment, and mature network restructuring. Between event years, the bus network is frozen. An annual incremental feeder headway adjustment (responding to ridership growth) has been designed but not yet implemented.

---

## 6. Development Model

### 6.1 Demand-Driven Development

The development model follows DiPasquale & Wheaton (1992) market equilibrium logic, implemented in `DemandDrivenDevelopmentModel`:

1. **Metro-level growth**: Annual population growth of 1.5% and job growth of 1.8% (IBRC Lafayette MSA forecast) applied to the metro area.

2. **Corridor capture (logit location choice)**: A binary logit determines each corridor's share of metro growth: P_c = exp(V_c) / (exp(V_0) + exp(V_c)), where V_c = beta_acc × log1p(accessibility) + beta_cap × log1p(capacity) + beta_rent × log1p(rent) + beta_cost × log1p(cost), and V_0 = asc_metro + beta_acc × log1p(metro_accessibility) + beta_cap × log1p(metro_capacity). The alternative-specific constant asc_metro = 2.50 represents suburban inherent advantage, calibrated to produce 3–15% capture for Lafayette's 232K metro (Nelson & Hibberd 2024). Scenario effects are endogenous: upzoning increases developable capacity, which raises V_c and capture share. Multiplied by developer confidence ramp (logistic 0.30 to 0.97 over 25 years).

3. **Parcel-level pro forma**: For each parcel in the corridor catchment, a developer pro forma determines feasibility:
   - Land value from CurLandAV/Shape_Area (clipped $1–200/sqft; zone-median fallback for 6,537 zero-AV parcels)
   - Demolition cost: CurImpAV opportunity cost + $8/sqft physical demolition (RSMeans Midwest)
   - Construction cost: $150/sqft residential (wood-frame), $180/sqft commercial (steel-frame), RSMeans Midwest 2024
   - FAR utilization: heuristic from land value per sqft (land_psf/120, clipped 10–85%)
   - Developer margin: margin-on-cost (residential 17.5%, commercial 20%, mixed 22.5%)
   - Exempt parcels (PropClass 600-series) are excluded

4. **Supply-side constraints**: Height-cost escalation for tall buildings, absorption cap preventing oversupply, developer confidence ramp reflecting construction lag and market uncertainty.

5. **Multi-year delivery**: OCCUPANCY_SCHEDULE = (0.0, 0.0, 0.33, 0.67, 1.0) spreads each development project's occupancy across 5 years from construction start.

### 6.2 Vacancy and Rent Feedback

Target vacancy rates: 4% residential (ACS 2022 Lafayette weighted average: 1.4% homeowner + 5.8% rental), 8% commercial (CBRE Midwest Q4 2023). Rent adjustment speed is 0.50 per 5-year period, scaled by `step_years/5.0` for annual steps to prevent oscillation. Floor at 0.60× and ceiling at 1.40× of base rent. Accessibility-driven rent premium capped at MAX_RENT_PREMIUM = 0.40 (TCRP upper bound).

**Mode-specific permanence premium.** Fixed-guideway transit (APM/rail) generates a "permanence premium" over BRT: the irreversibility of track and station infrastructure signals long-term commitment, reducing developer risk and anchoring land-use expectations. The model applies a mode-specific multiplier on adjusted rents:

- `FIXED_GUIDEWAY_RENT_MULT = 1.12` (APM/rail: 12% above BRT baseline)
- `BRT_RENT_MULT = 1.00` (BRT: no permanence premium)

The adjusted rent formula is: `adj_rent = base_rent × (1.0 + accessibility_premium) × mode_mult`.

The same multiplier is applied to assessed values per square foot in the TIF revenue calculation (`_compute_endogenous_tif`), since assessed values track rents.

The 12% value is derived from a cross-study synthesis:

| Source | Finding |
|--------|---------|
| Cervero & Duncan 2002 | Rail land-value premium 10-25% over BRT (Santa Clara County commercial and residential) |
| Debrezion, Pels & Rietveld 2007 | Meta-analysis (73 studies): rail residential premium ~4.2%, commercial ~16.4% |
| Zhang & Yen 2020 | Meta-analysis (23 BRT studies): BRT residential premium ~5% (range 2-8%) |
| FTA Report No. 0022 | Developer interviews: permanence of fixed infrastructure cited as key investment factor |

Land-value premiums capitalize future rent growth, so rent premiums are approximately 60% of land-value premiums, giving a 6-15% rent differential for fixed-guideway over BRT. The 12% value is the central estimate for a blended residential/commercial context. A conservative estimate would be 8-10%; an aggressive one 15%. Sensitivity to this parameter can be tested via `run_behavioral_sensitivity.py`.

### 6.3 Scenario Zoning

Two scenarios control the FAR limits available to the development model:

- **current_zoning**: Existing FAR limits from the zoning density table. Development scenario cost multiplier: 1.000.
- **no_zoning**: All FAR caps removed within TIF boundary (NO_ZONING_FAR_CAP = 8.0). Cost multiplier: 0.936 (6.4% cost reduction; NAHB/NMHC 2022 adjusted for Indiana WRLURI). Maximum capacity increase produces highest endogenous capture.

---

## 7. Financial Model

### 7.1 Capital Cost

MECE decomposition (2025 dollars), sourced from Detroit People Mover, Miami Metromover, and Jacksonville Skyway inflation-adjusted costs, excluding airport APMs which have 3–10× premiums from terminal integration:

| Component | Unit Cost | Basis |
|-----------|-----------|-------|
| Guideway & civil works | $55M/route-km | Elevated dual-track: columns, precast, running surface, utilities |
| Stations | $15M/station | Platform, canopy, vertical circulation, HVAC, fare collection |
| Vehicles | $3M/car | Alstom Innovia, 2-car consists (50 pax/car) |
| Fixed systems | $15M lump | Signaling, control center, power distribution, comms |
| Professional services | 10% of direct | Design, CM, environmental, permitting (FTA SCC range: 8–12%) |

Cross-check: 8 km corridor, 6 stations, 40 vehicles (20 trains) = $731M (~$91M/km), consistent with the legacy blended rate of $100M/km.

Curvature cost multiplier from physics-based curve analysis is applied to the guideway component only. Barrier crossing surcharges are additive: $80M/river, $40M/highway (I-65), $25M/railroad.

### 7.2 Operating Cost

```
Annual O&M = $1.5M fixed + $200K/km/yr + $150K/station/yr + $65/vehicle-hour
```

Escalated at 3%/yr (BLS CPI transport services). Peer benchmarks: Detroit $11–22M/yr for 4.7km/13-station system; Morgantown ~$5M/yr for 14km.

### 7.3 Revenue

**Farebox**: daily_ridership × 300 operating days × $2.00 fare. The 300-day annualization follows FTA STOPS methodology: weekday × 255 + Saturday × 52 × 0.55 + Sunday × 58 × 0.40 ≈ weekday × 300.

**TIF revenue**: Endogenous — computed from feedback loop development outputs using a three-stream tenure decomposition: homestead (owner-occupied), rental (non-homestead residential), and commercial. Each stream has its own Indiana circuit breaker cap (IC 6-1.1-20.6): homestead 1% of AV, rental 2%, commercial 3% (vs. the 2.32% gross levy from WL-TSC district, DLGF 2024 certified). Effective capture: 85% of statutory (admin overhead, assessment lag, appeals). SB 1 (2025) erosion is applied as a year-by-year multiplier declining from 1.00 to 0.78 over 25 years.

In EDA (Economic Development Area) TIF districts, IC 36-7-14-39 (post-June 2025) defines "residential property" as homestead only — rental residential is capturable. This distinction matters in a university town where ~95-99% of new station-area construction is multifamily rental.

**Tenure classification (simplification).** New construction is classified as homestead or rental based on the zoning of the developed parcel, not by modeling individual unit tenure. SF zones (R1/R1A/R1B/R1T) produce homestead; all other zones produce rental. Student housing is always rental. Under the no_zoning scenario, a FAR ceiling of 1.5 reclassifies SF-zoned parcels as rental when the proforma builds above SF density, since the physical product is an apartment building regardless of the underlying zone code. This produces homestead shares of ~1-3% (current_zoning) and ~0% (no_zoning) on new construction — consistent with national TOD patterns (TCRP Report 128: 85-95% multifamily near stations) despite the county being 45.6% owner-occupied overall (ACS 2022). The low share arises because the proforma selects high-FAR parcels first; only ~37% of station-area parcels are SF-zoned, and their low FAR limits make them unattractive to the proforma. R2/R2U (duplex/townhome) zones are conservatively classified as rental; some units may be owner-occupied, but near Purdue the majority are investor-owned student rentals. Indiana HEA 1120 (2023) caps residential TIF at 20 years; commercial continues to year 25.

### 7.4 Financial Viability

Debt service: 25-year level amortization at 5% municipal bond rate. Viability threshold: Debt Service Coverage Ratio (DSCR) ≥ 1.25 (standard bond underwriting). Default funding assumption: 100% local (no federal or state contribution).

### 7.5 Benefit-Cost Analysis

Full USDOT BCA Guidance (2024) framework with seven MECE accounting rules. Benefits computed at both 3% and 7% discount rates:

- **Travel time savings**: VTTS at $18.80/hr personal, $31.00/hr business (USDOT 2024), growing at 1.2%/yr real
- **Vehicle operating cost savings**: $0.22/mile marginal (AAA 2024)
- **Safety**: $0.12/VMT external crash cost (FHWA; Parry & Small 2005)
- **Emissions**: $190/ton CO₂ social cost (EPA/OMB 2024, 3% near-term) + $0.02/VMT criteria pollutants
- **Health**: $0.15/min walking value (WHO HEAT), applied to net new walking from transit access
- **Parking avoidance**: $35,000/space structured parking cost (ITE 2024)
- **Residual value**: 50% of capital at year 25 (USDOT 40-year infrastructure guidance)

Car diversion rates by ridership component: commute 60%, student 30%, generator 55%, induced 0% (new trips), latent 0% (zero-car households).

Broader fiscal impacts computed separately (not in BCR): local income tax (Tippecanoe LIT 1.1%), state income tax (3.05%), sales tax (7% on retail expansion), road maintenance savings ($0.05/VMT). Employment impacts via APTA multipliers (30 construction jobs per $1M; Type II multiplier 2.0).

---

## 8. Feedback Loop Structure

### 8.1 Per-Corridor Independent Evaluation

Each corridor is evaluated in isolation over 25 annual time steps. Before evaluating corridor *i*, the model snapshots baseline state (population, jobs, metro growth metrics, bus headways, feeder coverage, pending deliveries). After completing all 25 years for corridor *i*, baseline state is restored before evaluating corridor *i+1*. This prevents a high-ridership corridor's induced development from inflating the catchment population seen by subsequent corridors.

### 8.2 Annual Loop

For each year *t* of each corridor:

1. **Ridership**: Compute 6-component daily ridership given current pop, jobs, bus headways, feeder coverage, APM headway
2. **Development**: Compute new residential and commercial units given ridership, market conditions, and available developable parcels (logit location choice determines corridor share of metro growth)
3. **Population/jobs allocation**: Assign new residents and jobs to specific parcels that received development (not spread uniformly)
4. **Bus restructuring**: At event years (0, 3, 8), run full GTFS-based network redesign. Otherwise frozen.
5. **APM headway update**: Continuous function of cumulative ridership
6. **Convergence check**: Track relative change in ridership, new_pop, new_jobs

### 8.3 Convergence

Per-corridor, per-year: relative delta = |current − previous| / max(|previous|, 25.0). Convergence declared when ridership delta ≤ 1% AND development delta ≤ 2% for 3 consecutive periods. Adaptive early stopping available when all corridors converge.

---

## 9. Validation

### 9.1 Behavioral Validation Gate

Seven checks against published benchmarks, run as CI gate:

| Check | Source | Benchmark | Tolerance | Severity |
|-------|--------|-----------|-----------|----------|
| Walk-access decay at 400m | TCRP 165 | 50% retention | ±15% | Hard |
| Walk-access decay at 800m | TCRP 165 | 25% retention | ±25% | Hard |
| Peak-hour concentration | NHTS 2017 | 55% AM+PM share | ±10% | Hard |
| Headway elasticity | TCRP 95 | −0.30 | ±0.25 | Hard |
| Fare elasticity | TCRP 95 | −0.35 | ±0.20 | Soft |
| Parking sensitivity | Project gate | ≥2pp transit increase at $8 | — | Hard |
| Income-segment fare gap | Project gate | SE01 > SE03 by ≥0.5pp | — | Hard |

### 9.2 CityBus Cross-Check

System-wide daily boardings: 5,982 observed (NTD). Route 4B (highest ridership): 2,997. The model's top campus-proximate corridors produce 2.1–3.6× Route 4B ridership, which is plausible for a dedicated guideway with higher speed, frequency, and coverage than a single bus route.

### 9.3 Uncertainty Quantification

Monte Carlo framework (500 draws, triangular distributions) on key parameters:

| Parameter | Distribution |
|-----------|-------------|
| Ridership multiplier | Tri(0.8, 1.0, 1.25) |
| TIF capture multiplier | Tri(0.75, 1.0, 1.3) |
| Capital cost multiplier | Tri(0.9, 1.0, 1.25) |
| Operating cost multiplier | Tri(0.9, 1.0, 1.2) |
| Fare multiplier | Tri(0.9, 1.0, 1.1) |
| Discount rate delta | Tri(−0.01, 0.0, +0.01) |

Produces corridor-level p5/p25/p50/p75/p95 bands for ridership, NPV, DSCR, and TIF revenue.

### 9.4 Reproducibility

Run packaging includes: git commit hash, source manifest with retrieval dates and SHA256 hashes, artifact manifest with per-file SHA256 and row counts, required-artifacts completeness gate. All outputs are traceable to specific data source versions and code commits.

---

## 10. Notable Design Choices

### 10.1 Why APM, Not LRT or BRT

The study area's characteristics — university-dominated ridership, short corridors (3–16 km), compact development pattern, Wabash River barrier — favor automated guideway transit over BRT (which shares road congestion and requires operators) or LRT (which has higher capital cost and longer headway constraints). APM allows 90-second minimum headway with full automation, providing high frequency at lower operating cost per passenger-km.

### 10.2 Online-Data-Only Constraint

No travel surveys, stated preference data, or traffic counts were available. All behavioral parameters are calibrated from TCRP reports and national datasets. This is explicitly a planning-level screening tool. The mode choice coefficients (β_IVT = −0.055, β_WAIT = −0.090, β_ACCESS = −0.120, β_COST = −0.035) are within standard ranges for mid-size US cities but have not been locally estimated.

### 10.3 LODES as Primary OD Source

LODES provides block-level commute OD flows with income segmentation — essential for the income-differentiated mode choice. The limitation is that LODES captures only home-to-work flows, not non-work travel. Component 1b uses NHTS 2017 purpose-specific trip generation rates (shopping, social/recreation, personal business, escort) applied to residential population, with purpose-specific APM share multipliers derived from STOPS guidance. Component 4 handles attraction-based institutional generators (medical, retail, events). Student non-work trips are cleanly separated into Component 2 via enrollment-based campus population. This purpose-disaggregated approach replaces the former flat 3.0 all-day expansion factor, which double-counted non-work trips already captured by Components 1b and 4.

### 10.4 Student Ridership Dominance

At approximately 48% of total ridership for campus-proximate corridors, student trips dominate. This reflects Purdue's scale (70,000 affiliates, system generating roughly one-third of CityBus's total ridership on a single route). The model uses enrollment-based campus population rather than residential population (which is near-zero on campus parcels), with building-level institutional weights distributing the campus population to specific parcels. The student ASC adjustments (car −0.40) reflect the empirical reality that only 32% of undergraduates have cars on campus.

### 10.5 Feeder Coverage as a Scalar

Bus feeder coverage is currently a scalar fraction (0.0–0.90) applied uniformly to all feeder-zone trips. A sector-based model (8 directional wedges with population-weighted, frequency-quality-adjusted coverage) has been designed but not yet implemented. The scalar model likely underestimates feeder coverage variation — corridors with bus service concentrated on one side (e.g., east of the Wabash River) receive the same coverage fraction as corridors with symmetric bus coverage.

### 10.6 Static Station Placement

Stations are fixed at the corridor design phase and never modified during the 25-year feedback loop. As TOD matures and demand patterns shift, optimal station locations may change. Infill station logic (checking for coverage gaps where new development has created demand between existing stations) has been designed but not implemented.

### 10.7 Per-Corridor Independence

Evaluating corridors independently (with state snapshot/restore) prevents inter-corridor competition bias but also prevents modeling of network effects — e.g., two corridors sharing a transfer station could produce more ridership together than the sum of their independent evaluations. A network-synergy scoring adjustment is applied during the search phase but not during feedback loop evaluation.

### 10.8 TIF Revenue Endogeneity

TIF revenue is not an input assumption — it emerges from the feedback loop. The model computes actual development quantities per year, classifies them by tenure (homestead/rental/commercial), applies mode-specific assessed values (see §6.2), and calculates property tax increments subject to per-class circuit breaker caps and SB 1 erosion. This means the two zoning scenarios produce genuinely different TIF revenue streams reflecting their different development outcomes and tenure mixes, rather than relying on assumed TIF capture amounts. The tenure classification is zone-based (see §7.3) and produces the expected pattern: higher zoning flexibility shifts development from SF to MF, reducing the homestead share and increasing the share subject to the more favorable rental circuit breaker cap.

### 10.9 Mega-Parcel Population Cap

One parcel in the dataset had an implied population of 39,000 (a large institutional parcel with aggregated housing). A per-parcel population cap of 2,000 prevents such outliers from dominating catchment calculations.

---

## 11. Known Limitations and Planned Improvements

1. **No congestion feedback**: Car travel time is static (30 km/h urban). In reality, APM-induced development would increase local traffic, and mode diversion to APM would reduce it. Net effect is ambiguous but potentially meaningful for dense corridors.

2. **Bus travel time uses aggregate speed**: Bus IVT = distance / 18 kph, where 18 kph already includes all dwell, signal, and fare collection delays (from GTFS observed speeds). There is no per-stop dwell decomposition, which means TSP and proof-of-payment speed improvements can only be modeled as aggregate speed adjustments rather than per-stop reductions.

3. **Three-year gap in bus restructuring**: After the year-8 redesign, bus network is frozen for 17 years. Annual incremental headway adjustment has been designed but not implemented.

4. **No induced development in corridor search**: The NSGA-II scoring function uses current demand. Corridors with high development potential but low current demand are systematically undervalued.

5. **Directional split is global**: APM_PEAK_DIR_SPLIT = 0.60 for all corridors. LODES OD data could support corridor-specific values (campus corridors likely show stronger directional imbalance).

6. **Bike-to-APM suppressed for short trips**: Bike-to-APM is modeled in the walk zone (5th mode) but suppressed for trips < 1.5 km. ASC_BIKE_APM = −0.60 targets ~8% annual bike mode share.

7. **Reliability-aware wait time partially wired**: `effective_wait_time()` is applied to bus wait but APM wait still uses raw headway/2. Low impact (APM headways are generally ≤ 10 min) but inconsistent.

---

## 12. Output Summary

The model produces, for each corridor × scenario × year:

- Daily ridership (total and by component: commute, student, generator, induced, latent)
- Income-segmented ridership (SE01, SE02, SE03) and equity metrics (low_income_access_ratio)
- New development (residential units, commercial sqft, population, jobs)
- Bus network state (headway, feeder headway, restructure pressure, feeder coverage)
- APM headway and effective speed
- Financial metrics (capital cost, annual O&M, fare revenue, TIF revenue, DSCR, NPV, IRR)
- BCA results (benefit categories at 3% and 7% discount, BCR, Monte Carlo uncertainty bands)
- Convergence diagnostics

Final deliverables include corridor rankings, scenario comparison tables, uncertainty bands, and a decision package with top-N recommendations for each scenario.

# Economic & Fiscal Impact Measurement Plan for APM Corridors

## Full Synthesis — Analytical Framework, Implementation Specification, and Accounting Safeguards

---

## 1. Current State — What the Model Already Computes

The codebase has strong **financial viability** analysis and **ridership/development** modeling, but limited **societal benefit-cost** and **broader fiscal/economic multiplier** analysis. What exists today:

| Category | Metric | Location | Status |
|----------|--------|----------|--------|
| Capital cost | $/km, total capex, federal/state/local splits | `finance.py` | Done |
| Operating cost | Fixed ($1.5M) + variable ($200K/km + $150K/station + $65/veh-hr), demand-responsive | `financial_params.py` | Done |
| Fare revenue | Daily riders x $2.00 x 300 FTA days | `finance.py` | Done |
| TIF revenue | 25-yr phased increment, 85% capture, breakeven years | `tif_financing_model.py` | Done |
| Debt coverage | (TIF + farebox) / (debt + ops), dynamic per year | `financial_corridor_ranking.py` | Done |
| NPV / IRR | Project-level DCF at 5% | `financial_corridor_ranking.py` | Done |
| Development | Units, sqft, pop, jobs from 26-year feedback loop | `demand_driven_development.py` | Done |
| Property uplift | Rent premium decay from stations (max 40%, β=0.00425) | `demand_driven_development.py` | Done |
| Equity | Income-segmented ridership (SE01/02/03), low_income_access_ratio | `land_use_transport_model.py` | Done |
| Mode choice | 5-mode logit, income-weighted, parking-sensitive | `mode_choice.py` | Done |
| Monte Carlo | 500-draw uncertainty on ridership, TIF, DCR, NPV (p10/p50/p90) | `run_uncertainty_sensitivity.py` | Done |
| Scenarios | current_zoning / no_zoning with cost multipliers | `scenarios_config.json` | Done |
| Decision package | Scenario comparison, corridor delta, top recommendations | `generate_decision_package.py` | Done |
| Bus restructure | Per-route headway optimization, budget-constrained | `bus_network.py` | Done |

**What's missing** is the societal benefit-cost case, broader fiscal/economic multiplier story, and public-facing narrative that persuades taxpayers and elected officials. The gap is between "does the project cover its debt?" and "does the project benefit society more than it costs?"

---

## 2. Accounting Boundaries — Preventing Double-Counting

> **This section must be read before any implementation work begins.** Double-counting is the most common credibility-destroying error in transit economic impact studies.

### 2.1 The Core Problem

Several benefit categories overlap:

| Benefit A | Benefit B | Overlap Risk |
|-----------|-----------|--------------|
| TIF revenue (property tax increment) | Property value uplift in BCR | TIF *is* the public capture of property uplift. Counting both overstates benefits. |
| Travel time savings → higher productivity | Agglomeration productivity gains | Agglomeration partially operates through reduced travel time. |
| Construction jobs from APM | Construction jobs from TOD | Both are real, but TOD construction is *induced* by APM — listing them separately then adding a "multiplier" triple-counts. |
| Farebox revenue (project cash flow) | Travel time savings (societal benefit) | Fares are a transfer from riders to the project; time savings are a net societal gain. Fares should appear only on the cost-offset side, not as a benefit. |
| Ridership-driven development | Regional reallocation | Development captured by the corridor may be *relocated* from elsewhere in the metro, not net new. |

### 2.2 Clean Accounting Rules

**Rule 1 — Separate the financial case from the societal case.**

| Analysis | Question It Answers | What Counts as "Benefit" | What Counts as "Cost" |
|----------|---------------------|--------------------------|------------------------|
| **Financial viability** (existing) | Can the project service its debt? | TIF revenue + farebox revenue | Capital cost + O&M + debt service |
| **Benefit-cost analysis** (new) | Does society gain more than it loses? | Travel time savings, VOC savings, safety, emissions, health, parking, agglomeration, **spillover** property uplift (outside TIF district only) | Capital cost + O&M + net public service cost for new residents |

**Rule 2 — Property uplift in the BCR counts only the spillover beyond the TIF boundary.**
- Property uplift *within* the TIF district is already captured as TIF revenue on the financial side.
- Only the uplift on parcels 400–800m from stations (outside the TIF boundary, typically ~400m) should appear as a BCR benefit.
- This avoids counting the same property value increase twice.

**Rule 3 — Farebox revenue is a cost offset, not a benefit.**
- In the BCR, fare payments are a transfer (riders pay, project receives). They net to zero in societal accounting.
- Farebox appears only in the financial viability analysis (reducing the net public subsidy).

**Rule 4 — Multiplier analysis is presented separately, never added to the BCR.**
- RIMS II / APTA multipliers estimate gross economic activity (output, earnings, jobs), not net welfare gains.
- Adding multiplier-derived benefits to a BCR would double-count, because the BCR already captures the *welfare* value of the underlying activity.
- Present multiplier results in their own table: "Economic Activity Generated" — distinct from "Benefit-Cost Ratio."

**Rule 5 — Regional reallocation is net-zero at the metro level.**
- Development captured by APM corridors is partially *redirected* from elsewhere in the 232K Lafayette metro, not entirely net new.
- The model's REGIONAL_REALLOCATION_ELASTICITY (0.05) already accounts for this — it's modest.
- In the fiscal impact analysis, present both "corridor-level" (gross) and "metro-level" (net) figures. The difference is the displacement from other areas.
- For the tax argument, corridor-level is appropriate (the TIF district captures value regardless of metro-level displacement). For the BCR, metro-level is more honest.

**Rule 6 — Agglomeration benefits use a conservative, non-overlapping specification.**
- Agglomeration productivity gains *partially* operate through reduced travel time (which is already counted in travel time savings).
- Use the "rule of half" correction: apply agglomeration elasticities only to the *residual* effective density change after removing the travel-time component.
- Alternatively, use a lower-bound agglomeration elasticity (0.04–0.07 instead of 0.10–0.22) when travel time savings are also counted.
- Document the conservative choice explicitly — reviewers will look for this.

**Rule 7 — Construction employment: separate APM from TOD, don't multiply TOD.**
- APM construction jobs: direct function of public capital cost. Apply APTA multiplier (including indirect/induced).
- TOD construction jobs: direct function of private development sqft × construction cost. Report as "private investment leveraged" — do NOT apply a second multiplier on top, because the APTA multiplier for APM already includes induced-development effects in its Type II specification.
- Present as: "X,XXX public-investment construction jobs + $Y million in private development activity."

### 2.3 Presentation Framework

Every output table must clearly label which accounting frame it belongs to:

```
[FINANCIAL]   — Project cash flow (TIF + farebox vs. debt + ops)
[SOCIETAL]    — Benefit-cost analysis (welfare gains vs. resource costs)
[ECONOMIC]    — Gross economic activity (multipliers, jobs, output) — NOT additive to BCR
[FISCAL]      — Government revenue impact (taxes generated across all streams)
[EQUITY]      — Distributional analysis (who benefits, who bears cost)
```

---

## 3. Layer 1 — Benefit-Cost Analysis (USDOT Framework)

### 3.1 Why This Is the Highest Priority

The BCR transforms "does the project pay for itself?" into "does the project benefit society more than it costs?" — a fundamentally different and more persuasive question. It is the gold standard metric for federal funding applications (FTA Small Starts) and the most rigorous framework for taxpayer communication.

### 3.2 Benefit Categories

#### A. Travel Time Savings (typically 50–70% of total benefits)

**What you already have:** The mode choice model produces trip diversions by mode (car→APM). The LODES data gives OD distances. The mode choice utility function computes IVT for each mode.

**Computation:**

```
For each diverted car trip:
  car_travel_time = od_distance_km × CAR_CIRCUITY / car_speed_kph × 60  [minutes]
  apm_travel_time = walk_access + wait_time + apm_ivt + walk_egress     [minutes]
  time_saved_per_trip = car_travel_time - apm_travel_time               [minutes]

annual_person_hours_saved = daily_diverted_trips × time_saved_per_trip / 60 × 300
annual_benefit_$ = person_hours_saved × VTTS
```

**Value of Travel Time Savings (VTTS):**
- Personal travel: $18.80/hr (USDOT BCA Guidance 2024, 2022$)
- Business travel: $31.00/hr (USDOT 2022$)
- Weighted average: Use LODES SE01/SE02/SE03 shares to weight. ~80% personal, ~20% business for commute trips.
- Non-commute trips: 100% personal rate.
- Real growth: VTTS grows at 1.2% real per year (USDOT guidance for 25-year horizon).

**Key modeling decision:** "Diverted trips" = daily APM riders who would otherwise have driven. The mode choice model already computes this — it's the car-to-APM shift. Riders who shift from bus or walking generate smaller time savings (bus→APM) or losses (walk→APM for very short trips). Compute separately by origin mode.

#### B. Vehicle Operating Cost Savings (typically 10–15%)

**Use marginal cost, not full cost.**

A rider who takes APM instead of driving saves marginal per-mile costs (fuel + tire wear + maintenance) but still owns the car and pays insurance/depreciation. For the BCR, use AAA marginal cost.

```
marginal_voc = $0.22/mile  (AAA 2024, fuel + maintenance + tires only)
annual_vmt_avoided = daily_diverted_car_trips × avg_car_trip_distance_mi × 2 × 300
annual_benefit_$ = annual_vmt_avoided × marginal_voc
```

For the **equity/cost-burden analysis** (Layer 6), use full AAA cost ($0.655/mile) — a low-income household that can shed a vehicle because of APM saves the full ownership cost. Document this distinction.

**Data source for trip distance:** Each LODES OD pair has a known distance (already in the spatial cache). For non-LODES trips (students, generators), use the average corridor catchment distance.

#### C. Safety Benefits (typically 5–10%)

```
annual_benefit_$ = annual_vmt_avoided × crash_cost_per_vmt
```

**Crash cost per VMT:** $0.12/VMT comprehensive (FHWA, includes fatal + injury + PDO, 2022$). This is the *external* crash cost — the portion borne by society, not just the driver. Using the lower $0.04/VMT figure (FHWA property-damage-only) would substantially undercount.

Decomposition for transparency:
- Fatal crashes: NHTSA VSL = $12.5M (2022$), Indiana fatal crash rate = 1.13/100M VMT (FARS 2022) → $0.141/VMT
- Injury crashes: ~$0.06/VMT (FHWA KABCO scale)
- Property damage: ~$0.03/VMT
- **External share:** ~50% of total crash cost is external (Parry & Small 2005) → use $0.12/VMT external

Present the decomposition in footnotes. Use $0.12/VMT as the primary figure, note that some analyses use $0.04 (PDO only), and show both.

#### D. Emission Reductions (typically 3–5%)

**Two approaches — use the simpler one, show the other for validation:**

*Approach 1 — Per-VMT social cost (simpler):*
```
CO2_per_vmt = 0.000348 tons  (EPA, avg light-duty vehicle 2024)
social_cost_CO2 = $190/ton   (EPA/OMB Interim SCC, 2024, 2020$, 3% near-term discount)
co2_benefit = annual_vmt_avoided × 0.000348 × $190 = ~$0.066/VMT

Criteria pollutants (NOx, PM2.5, SO2):
  Use USDOT BCA Tables (damage cost per ton by county, rural/urban):
  ~$0.015–0.035/VMT depending on urban density
  Lafayette (urban fringe): use $0.020/VMT

total_emission_benefit_per_vmt = ~$0.086/VMT
```

*Approach 2 — Blended (for cross-check):*
```
$0.035/VMT (Response 2's figure) is too low — it only captures CO2 at a lower SCC ($51/ton, pre-2024 revision).
$0.086/VMT is the updated figure with EPA's 2024 SC-CO2 and criteria pollutants.
```

**Use $0.086/VMT as primary.** The EPA raised the social cost of carbon substantially in 2024. Using the old figure would be indefensible in a post-2024 analysis.

**Real escalation:** SC-CO2 increases ~2.5%/year in real terms (EPA schedule). Apply to future-year benefits before discounting.

#### E. Health Benefits from Active Transport (typically 2–5%)

Each APM trip involves walking to/from stations. Quantify the health value of this induced physical activity.

```
walk_access_distance = avg 400m each end = 800m round trip = 0.5 miles
walk_time = 800m / 4.8kph = 10 minutes per trip

annual_walk_minutes = daily_apm_riders × 10 × 300 = X million minutes

Health benefit (WHO HEAT methodology):
  Relative risk reduction: 0.90 per 168 min/week of walking (all-cause mortality)
  Per-minute value: $0.30–0.50/minute of walking (WHO 2024, adjusted for US VSL)
  Conservative: $0.15/minute (US adaptation, lower bound)

annual_benefit_$ = annual_walk_minutes × $0.15
```

**Caution:** Only count *new* walking — riders who previously walked to a bus stop and now walk to an APM station get no incremental benefit. Apply only to car→APM and previously-sedentary→APM shifts. Estimate ~60% of riders generate net new walking.

#### F. Parking Cost Savings (typically 2–5%)

**Two components:**

*Public/institutional parking avoided:*
```
Purdue/city avoided parking spaces = car_trips_shifted × peak_hour_share × parking_duration_factor
Cost per space: surface $5–15K, structured $25–50K (ITE 2024)
Purdue context: structured parking at ~$35K/space
Annual amortized savings: avoided_spaces × $35,000 / 30yr life × occupancy_factor
```

*Individual parking cost savings:*
```
Already modeled in mode choice (campus $0.40/trip, downtown $4.00/trip, suburban $0.00)
This is a transfer (riders save, parking operators lose revenue) — nets to zero in BCR
EXCEPT for avoided construction of new parking, which is a real resource savings
```

**Only count avoided parking construction in the BCR.** Day-to-day parking fees are transfers.

### 3.3 Cost Categories for BCR

| Cost | Source | Notes |
|------|--------|-------|
| Capital cost (local share) | `finance.py` capex_musd() | Already computed. Use 100% local per user specification. |
| Annual O&M | `finance.py` annual_om_musd() | Already computed. Escalate at 3%/yr real (BLS CPI transport). |
| Net public service cost | **New** | New residents require police, fire, schools, water/sewer. |
| Residual value | Standard | At year 25, the infrastructure has remaining life. Credit 50% of capital as residual value (USDOT guidance for 40-year infrastructure). |

**Net public service cost computation:**
```
Tippecanoe County per-capita municipal service cost: ~$2,800/year (derived from county budget / population)
Net of property tax paid: new residents pay property tax, which partially offsets service cost
Net cost per new resident = $2,800 - (avg_home_value × 0.0232) / avg_household_size
For typical $200K home: $2,800 - ($200K × 0.0232 / 2.56) = $2,800 - $1,813 = $987/person/year net cost
```

This is the honest cost side. Presenting it proactively builds credibility with skeptical audiences.

### 3.4 BCR Calculation

```
BCR = NPV(benefits, r) / NPV(costs, r)
```

**Present at both 3% and 7% discount rates** per USDOT BCA Guidance (2024). The 3% rate reflects the social rate of time preference; 7% reflects the opportunity cost of capital. FTA and OMB expect both.

| BCR Range | Interpretation |
|-----------|----------------|
| < 1.0 | Benefits do not justify costs |
| 1.0–1.5 | Marginal — justifiable with equity or strategic arguments |
| 1.5–2.5 | Strong — typical for well-sited transit projects |
| > 2.5 | Very strong — may indicate aggressive assumptions |

**For Lafayette APM corridors, target a conservative 1.5–2.5× range with p10/p90 uncertainty bands.** If the BCR exceeds 3.0×, audit assumptions — it likely means an input is too optimistic.

### 3.5 Monte Carlo Extension

Propagate uncertainty through the BCR. The existing Monte Carlo framework (500 draws, triangular distributions) should be extended with:

| New Parameter | Distribution | Low | Mode | High |
|---------------|-------------|-----|------|------|
| VTTS multiplier | Triangular | 0.80 | 1.00 | 1.20 |
| Crash cost multiplier | Triangular | 0.70 | 1.00 | 1.30 |
| SC-CO2 multiplier | Triangular | 0.50 | 1.00 | 2.00 |
| Walk health value $/min | Triangular | 0.10 | 0.15 | 0.30 |
| Agglomeration elasticity | Triangular | 0.03 | 0.05 | 0.10 |

Output: `bcr_p10`, `bcr_p50`, `bcr_p90` per corridor per scenario.

---

## 4. Layer 2 — Broader Fiscal Impact (Beyond TIF)

### 4.1 Property Tax (Enhance Existing)

**Already modeled.** Two enhancements:

**A. "But-for" test presentation:**
Indiana TIF statute (IC 36-7-14) requires demonstrating development wouldn't occur without the investment. The model's `baseline_growth_rate = 2%` counterfactual is the right approach. Surface this comparison prominently:

```
property_value_with_apm = endogenous from feedback loop (existing)
property_value_without_apm = base_av × (1.02)^year (counterfactual)
but_for_increment = with_apm - without_apm (per year, per corridor)
```

This is already implicit in the TIF calculation. Make it an explicit output column.

**B. 2023 TIF law change (CRITICAL — this is a model bug):**
Indiana HEA 1120 (2023) reduced the maximum TIF term for **residential** allocation areas from 25 to 20 years. The current model uses 25 years for all uses.

**Fix required:** Split TIF revenue projection into residential and commercial components. Residential TIF truncates at year 20. Commercial TIF continues to year 25. This reduces cumulative TIF by approximately 15–20% for residential-heavy corridors.

Implementation:
```python
res_share = new_res_sqft / (new_res_sqft + new_comm_sqft)
com_share = 1 - res_share
for year in range(26):
    if year <= 20:
        tif_revenue[year] = total_increment × capture_rate × tax_rate
    else:
        tif_revenue[year] = total_increment × capture_rate × tax_rate × com_share
        # residential portion expires after year 20
```

### 4.2 Local Income Tax (LIT) — New

Indiana Local Income Tax (IC 6-3.6). Tippecanoe County rate: 1.10%.

```
new_permanent_jobs = new_comm_sqft / SQFT_PER_EMPLOYEE (200)  [already computed as new_jobs]
avg_wage_by_sector:
  SE01 (≤$1,250/mo): $12,000/yr
  SE02 ($1,251–$3,333/mo): $27,000/yr
  SE03 (>$3,333/mo): $58,000/yr
  Weighted average (LODES Lafayette MSA mix): ~$42,000/yr

annual_lit_revenue = new_jobs × weighted_avg_wage × 0.011
```

**Note:** This revenue flows to the county general fund, NOT the TIF district. It is a separate public benefit that partially offsets the net public service cost of new residents.

**State income tax (informational):** Indiana flat 3.05%. Not a local revenue source, but useful for state-level advocacy.

```
annual_state_income_tax = new_jobs × weighted_avg_wage × 0.0305
```

### 4.3 Sales Tax from TOD Retail — New

Indiana state sales tax: 7%. No local option in Tippecanoe County currently.

```
retail_share_of_comm = 0.35  (ULI benchmark for TOD mixed-use)
new_retail_sqft = new_comm_sqft × retail_share_of_comm
sales_per_sqft = $350/yr  (Census Annual Retail Trade Survey, neighborhood retail)
annual_retail_sales = new_retail_sqft × sales_per_sqft
annual_sales_tax = annual_retail_sales × 0.07
```

**Presentation note:** Sales tax accrues to the state, not Tippecanoe County. Present as "economic activity generated" rather than "local revenue." If a local option sales tax is ever adopted, this becomes local revenue — note the contingency.

### 4.4 Construction-Phase Tax Revenue — New

```
construction_spending_apm = capital_cost  [already computed]
construction_spending_tod = new_res_sqft × $150/sqft + new_comm_sqft × $180/sqft
  [RSMeans Midwest 2024, wood-frame res / steel-frame comm]

construction_wages = total_construction_spending × 0.40  [labor share of construction cost]
construction_lit = construction_wages × 0.011  [Tippecanoe LIT]
construction_state_income_tax = construction_wages × 0.0305
construction_sales_tax_materials = total_construction_spending × 0.35 × 0.07
  [materials share × state rate]
```

**Duration:** APM construction: 3–4 years. TOD construction: phased over 15–20 years per the existing development phasing model.

### 4.5 Reduced Infrastructure Costs — New

**Road maintenance savings:**
```
annual_vmt_avoided (from Layer 1)
road_maintenance_cost_per_vmt = $0.05  [FHWA, local road share]
annual_road_savings = annual_vmt_avoided × $0.05
```

**Deferred road capacity:**
More speculative. If VMT reduction defers a road widening project, the savings are substantial but project-specific. Present as a qualitative benefit unless a specific road project can be identified.

**Reduced public parking need:**
```
avoided_parking_spaces = peak_car_trips_shifted × 0.85  [occupancy adjustment]
avoided_parking_cost = avoided_spaces × $35,000  [one-time, structured parking]
annualized_savings = avoided_parking_cost / 30  [30-year parking structure life]
```

### 4.6 Fiscal Impact Summary Table

Label: `[FISCAL]`

| Revenue/Savings Stream | Annual (Yr 10) | 25-Year Cumulative | Recipient |
|------------------------|-----------------|---------------------|-----------|
| TIF property tax increment | $X.XM | $XXM | TIF district |
| LIT from new permanent jobs | $X.XM | $XXM | County general fund |
| LIT from construction workers | $X.XM | $XXM (years 1–20) | County general fund |
| State income tax (new jobs) | $X.XM | $XXM | State of Indiana |
| Sales tax (new retail) | $X.XM | $XXM | State of Indiana |
| Road maintenance savings | $X.XM | $XXM | City/county |
| Parking infrastructure avoided | — | $XXM (one-time) | Purdue/city |
| **Gross fiscal benefit** | | | |
| *Less:* Net public service cost | ($X.XM) | ($XXM) | City/county |
| **Net fiscal benefit** | | | |

---

## 5. Layer 3 — Economic Activity Analysis (Multipliers)

> **Accounting label: `[ECONOMIC]`** — These figures represent gross economic activity, NOT net welfare gains. They must NEVER be added to the BCR.

### 5.1 Construction Phase

```
APM construction:
  direct_jobs_apm = capital_cost_M × 30 jobs/$1M  [APTA 2020 benchmark]
  Type II multiplier (RIMS II, transit construction): 1.8–2.2×
  total_jobs_apm = direct_jobs_apm × 2.0  [midpoint]
  total_earnings_apm = total_jobs_apm × $55,000  [BLS, construction avg, Lafayette MSA]

TOD construction:
  private_investment = new_res_sqft × $150/sqft + new_comm_sqft × $180/sqft
  direct_jobs_tod = private_investment_M × 30 jobs/$1M
  [Do NOT apply Type II multiplier again — see Rule 7 in Section 2]
  total_earnings_tod = direct_jobs_tod × $55,000
```

### 5.2 Operations Phase

```
APM operations:
  annual_ops_spending = annual_om_musd (from finance.py)
  direct_ops_jobs = lookup by system size (50–100 FTEs for 5–15km system)
  Type II multiplier (transit operations): 2.0–2.5× [more labor-intensive than construction]
  total_ops_jobs = direct_ops_jobs × 2.25

TOD permanent employment:
  direct_permanent_jobs = new_comm_sqft / 200  [already computed as new_jobs]
  wages = by LODES sector mix (already available)
```

### 5.3 Private Investment Leverage Ratio

This is the single most powerful headline metric for elected officials.

```
leverage_ratio = cumulative_private_development_value / public_capital_cost
```

The feedback loop already tracks `new_res_sqft` and `new_comm_sqft`. Convert to dollar value:

```
private_investment = new_res_sqft × $150/sqft + new_comm_sqft × $180/sqft
```

**Comparable benchmarks:**
| Project | Leverage Ratio | Context |
|---------|----------------|---------|
| Cleveland HealthLine BRT | $47.50 : $1 | 20+ years post-opening, $200M investment |
| Portland MAX Blue Line | ~$46 : $1 | 35+ years, $214M (1986$) |
| IndyGo Red Line BRT | ~$25 : $1 | 5 years post-opening, $96M |
| **Lafayette APM (projected)** | **$X : $1** | 25-year projection |

**Honest disclosure:** The Cleveland and Portland ratios include decades of development that would partially have occurred anyway. Your model's "but-for" test (Section 4.1A) provides a more defensible comparison — report both the gross and net-of-baseline leverage ratios.

### 5.4 RIMS II vs. APTA Published Multipliers

For a first-pass analysis, use APTA's published per-billion multipliers. These are defensible and widely cited.

For localized precision, purchase Tippecanoe County RIMS II multipliers from BEA (~$275). This gives county-specific indirect/induced effects that account for Lafayette's economic structure (university town, manufacturing base).

**Recommendation:** Start with APTA benchmarks. Purchase RIMS II only if the project advances to a formal economic impact study for a ballot measure or FTA application.

---

## 6. Layer 4 — Agglomeration & Wider Economic Benefits

### 6.1 Why This Matters for Lafayette

Agglomeration economics captures productivity gains from increased economic density. For most small cities, this is a minor benefit. For Lafayette, it's potentially significant because:

- **Purdue University** is a major research institution with 14,000+ employees in knowledge-intensive sectors.
- The education/research agglomeration elasticity (0.15–0.22, Graham 2007) is the highest of any sector.
- Connecting campus to downtown and residential areas tightens the effective labor market for Purdue employees and creates knowledge-spillover proximity.
- This is the single strongest analytical differentiator between the Lafayette APM case and generic small-city transit proposals.

### 6.2 Methodology (Graham 2007 Framework)

**Step 1 — Compute effective density before APM:**
```
ED_i = SUM_j [ E_j / gc_ij^alpha ]
```
Where:
- `E_j` = employment in zone j (from LODES, already loaded)
- `gc_ij` = generalized travel cost from zone i to zone j (minutes, from mode choice model)
- `alpha` = distance decay parameter (~1.0 for services, ~1.5 for manufacturing)

**Step 2 — Compute effective density after APM:**
Same formula, but with reduced `gc_ij` for zone pairs connected by the APM corridor. The mode choice model already computes travel times by mode — use the composite (logsum) travel cost.

**Step 3 — Compute productivity gain:**
```
delta_GDP = SUM_sectors [ (delta_ED_s / ED_s) × elasticity_s × GDP_per_worker_s × workers_s ]
```

**Sector-specific agglomeration elasticities:**
| Sector | Elasticity | Relevance to Lafayette |
|--------|-----------|----------------------|
| Education & research | 0.15–0.22 | **Primary** — Purdue campus, 14K+ employees |
| Business services | 0.18–0.22 | Professional services in downtown WL/Lafayette |
| Retail & hospitality | 0.05–0.10 | Chauncey Hill, downtown Lafayette |
| Manufacturing | 0.07–0.08 | Subaru (SIA), Caterpillar, Wabash National |
| Healthcare | 0.10–0.15 | IU Health Arnett, Franciscan Health |

**Conservative specification (per Rule 6, Section 2):**
Since travel time savings are already counted in the BCR, use the **lower bound** of each elasticity to avoid double-counting the travel-time channel of agglomeration:
```
education: 0.08  (lower half of 0.15-0.22, since ~50% operates through travel time)
business: 0.10
retail: 0.04
manufacturing: 0.05
healthcare: 0.06
```

### 6.3 Data Requirements

| Input | Source | Available? |
|-------|--------|-----------|
| Zonal employment by sector | LODES WAC | Yes (already loaded) |
| Zone-to-zone travel cost (before) | Mode choice model (car IVT) | Yes |
| Zone-to-zone travel cost (after) | Mode choice model (composite, with APM) | Yes |
| GDP per worker by sector | BEA CAINC6N (Tippecanoe County) | Free download |
| Workers by sector | LODES WAC | Yes |

**Implementation effort:** Medium-high. Requires building a zone-to-zone generalized cost matrix (before/after APM), which the mode choice infrastructure can support but doesn't currently produce in matrix form.

### 6.4 Expected Magnitude

Typical agglomeration benefits for transit projects: 10–25% on top of conventional user benefits (Crossrail London: 24%, but that's a megacity). For Lafayette, conservatively estimate 5–10% due to smaller metro size but partially offset by Purdue's knowledge-economy concentration.

---

## 7. Layer 5 — Comparable Project Benchmarks

### 7.1 Comparables Table

| Project | Metro Pop | Daily Ridership | Capital Cost | BCR | Leverage | Key Lesson |
|---------|-----------|----------------|--------------|-----|----------|------------|
| **Morgantown PRT (WVU)** | ~70K | ~12,000 (acad yr) | $126M (2024$) | N/A | N/A | Closest analogue: university APM, small metro. Demonstrates very high ridership/capita. **Cautionary:** 3 years late, 3–4× over budget. |
| **Tempe Streetcar (ASU)** | ~5M (PHX) | ~2,400 | ~$200M | ~1.8 | ~$15:$1 | University-adjacent, exceeded projections by 28%. |
| **Cleveland HealthLine BRT** | ~2M | ~14,000 | $200M | ~2.5 | $47.50:$1 | Gold standard for BRT economic impact. Similar-scale city core. |
| **Portland MAX Blue Line** | ~2.5M | ~35,000 | $214M (1986$) | N/A | ~$46:$1 | University-adjacent (PSU), TIF-funded. 35+ years of data. |
| **IndyGo Red Line BRT** | ~2M | ~3,500 | ~$96M | N/A | ~$25:$1 | **Same state.** Same TIF legal framework (IC 36-7-14). Passed via 2016 ballot measure (0.25% LIT). |
| **Detroit QLine** | ~4M | ~2,460 | ~$187M | <1.0 | Disputed | **Cautionary.** Below projections, equity criticism. Claimed $8.1B economic impact widely disputed. Do not cite favorably. |
| **Tampa TECO Streetcar** | ~3M | ~3,600 | ~$55M | N/A | N/A | 57% tourism ridership — event/visitor component relevant. |

### 7.2 Framing Strategy

**Lead with Morgantown PRT** — closest technology and context match. At 12,000 daily riders in a 70K metro, it demonstrates that a university-oriented APM achieves very high ridership per capita. Lafayette's top corridors projecting 3,000–3,500 in a 232K metro are conservative by comparison.

**Use IndyGo Red Line** for Indiana-specific political context. Same state, same TIF framework, successful ballot measure. Lessons on voter messaging directly transferable.

**Include Detroit QLine as an honest cautionary example.** This builds credibility — you're not cherry-picking successes. The QLine's problems (below-projection ridership, mixed-traffic operation, equity criticism, disputed economic claims) are all avoidable in the Lafayette APM design (dedicated guideway, university anchor, income-segmented analysis).

---

## 8. Layer 6 — Equity & Access Impact

### 8.1 Mobility Cost Burden (New)

Low-income households spend 25–30% of income on transportation (BLS CEX 2023). APM access enables significant savings.

```
For SE01 households (≤$15,000/yr):
  current_transport_cost = $15,000 × 0.28 = $4,200/yr  [BLS CEX low-income avg]

  If household can shed one vehicle:
    annual_savings = $4,200 - ($2.00/trip × 2 trips/day × 300 days) = $4,200 - $1,200 = $3,000/yr
    savings_as_pct_income = $3,000 / $15,000 = 20% of income

  If household retains vehicle but substitutes some trips:
    annual_savings = diverted_trips × ($0.655/mi × avg_trip_mi - $2.00/trip)
    [Use full AAA cost here — low-income households deciding whether to keep a car]
```

**Output:** `annual_transport_savings_se01`, `households_eligible_for_vehicle_shedding`

### 8.2 Jobs Accessibility Improvement (Enhance Existing)

Count jobs reachable within 30/45/60 minutes by transit, before vs. after APM.

```
For each residential zone with SE01 population:
  jobs_accessible_baseline = count(jobs where transit_time_ij ≤ 45 min, using current CityBus)
  jobs_accessible_build = count(jobs where transit_time_ij ≤ 45 min, with APM + restructured bus)
  accessibility_improvement = (build - baseline) / baseline

Aggregate: weighted average across SE01-heavy zones
```

**Data available:** LODES OD employment, mode choice travel times (existing). CityBus GTFS for baseline transit times (loaded when `--gtfs-dir` specified).

**Presentation:** "X,000 additional jobs accessible within 45 minutes for residents of [Romig/New Chauncey/Wabash corridor]."

### 8.3 Zero-Car Household Mobility (Enhance Existing)

Already modeled as latent demand (Component 6). Surface prominently:

```
zero_car_hh_served = zero-car households within walk zone (ACS B08201, already loaded)
trips_enabled = latent_riders (from Component 6)
mobility_value = trips_enabled × avg_trip_distance × VTTS / speed
```

**Presentation:** "X,XXX households currently have no car. The APM provides X,XXX new trips per year of previously unavailable mobility."

### 8.4 Title VI / Environmental Justice Mapping

For any future FTA funding application, Title VI compliance requires demonstrating that the project does not disproportionately burden minority or low-income communities.

```
For each corridor:
  pct_minority_catchment = ACS B03002 (race/ethnicity) within 1200m
  pct_low_income_catchment = SE01 share within 1200m (already computed)
  pct_minority_metro = metro-wide baseline

  ej_index = (pct_minority_catchment / pct_minority_metro + pct_low_income_catchment / pct_low_income_metro) / 2

  If ej_index > 1.0: corridor disproportionately serves disadvantaged populations (beneficial)
  If ej_index < 1.0 AND corridor displaces existing services: flag for mitigation
```

**Displacement risk:** If APM replaces CityBus routes in low-income corridors without adequate feeder service, it could *worsen* access for some residents. The bus restructure model already handles this (retain_parallel → feeder_transition phases), but the equity analysis should verify that SE01 accessibility never *decreases* during the transition.

### 8.5 FTA Equity Metrics (December 2024 CIG Guidance)

FTA's updated Capital Investment Grants guidance (December 2024) increased weight for:

| FTA Metric | How to Compute | Data Source |
|-----------|----------------|-------------|
| Affordable housing units within ½ mile | ACS B25070 (gross rent as % income) within 805m | Census API + spatial cache |
| Station-area pop density | Population within 805m / area | `weights_1200` filtered to 805m |
| Station-area emp density | Employment within 805m / area | Same |
| Cost per rider by income | Allocate costs proportional to SE01 share / SE01 trips | Existing income-segmented output |
| Community risk indicators | FEMA NRI + CDC SVI within catchment | Public APIs |

---

## 9. Indiana-Specific Legal & Fiscal Tools

### 9.1 Applicable Revenue Mechanisms

| Tool | Statute | Current Status in Tippecanoe | Potential Annual Revenue | Notes |
|------|---------|------------------------------|--------------------------|-------|
| **TIF** | IC 36-7-14 | Actively used (Purdue Research Park TIF, etc.) | $X.XM (corridor-dependent) | **Already modeled.** Fix residential 20-yr cap per 2023 amendment. |
| **Local Income Tax (LIT)** | IC 6-3.6 | 1.10% certified rate | ~$X.XM from new jobs | New revenue stream to model. |
| **Transit-specific LIT** | IC 8-25-2 et seq. | Not used (only Marion County) | ~$11.8M/yr if adopted at 0.25% | Requires referendum. Model as a funding scenario. |
| **Wheel Tax** | IC 6-3.5-10 | Not levied in Tippecanoe | ~$3–5M/yr ($25–40 × ~120K vehicles) | Supplemental source. Political feasibility uncertain. |
| **County Economic Development Income Tax (CEDIT)** | IC 6-3.6 (subtype) | May be partially allocated | Varies | Could designate transit as eligible use. |
| **Federal Small Starts** | 49 USC 5309 | N/A | Up to 80% federal share for projects < $400M | Requires FTA 6-criteria justification. Model outputs should conform to FTA format. |

### 9.2 Legal Risks and Constraints

**TIF 20-year residential cap (HEA 1120, 2023):** As noted in Section 4.1B, this is a model bug that must be fixed. Residential TIF revenue truncates at year 20, reducing cumulative revenue for residential-heavy corridors by 15–20%.

**TIF "but-for" test (IC 36-7-14-15):** The declaratory resolution establishing a TIF must demonstrate that development would not occur "but for" the APM investment. The model's no-build baseline (2% background growth) provides the analytical foundation, but the legal threshold is higher than analytical — it requires specific findings by the redevelopment commission. Frame model outputs to support this finding.

**Transit LIT referendum (IC 8-25-2):** Only one precedent in Indiana (IndyGo, Marion County, 2016, passed 60-40). No precedent in a county as small as Tippecanoe (193K vs. Marion's 978K). Political feasibility is uncertain. Model the revenue as a scenario, not a baseline assumption.

**Purdue's tax-exempt status:** Purdue-owned properties (PropClass 600-series) are exempt from property tax and generate no TIF revenue. The model already handles this (4,663 exempt parcels marked undevelopable). However, if Purdue were to ground-lease campus-adjacent land for private development (as some universities do), those parcels would generate TIF. This is a potential upside scenario.

---

## 10. FTA Small Starts Metrics

Even if the project is 100% locally funded, framing outputs in FTA format adds credibility and preserves optionality for federal funding.

### 10.1 FTA Project Justification Criteria (49 CFR 611.203)

| Criterion | FTA Metric | How to Compute | Current Status |
|-----------|-----------|----------------|----------------|
| **Mobility improvements** | Annual linked trips per $M (annualized capex + O&M) | `annual_ridership / annualized_cost` | Inputs exist, formula needed |
| **Environmental benefits** | Annual VMT reduction, CO2 tons avoided | From Layer 1 | New |
| **Congestion relief** | Trips shifted from congested corridors | Mode shift from car trips | Partial |
| **Cost effectiveness** | Annualized cost per trip | `(amortized_capital + annual_O&M) / annual_trips` | Inputs exist, formula needed |
| **Transit-supportive land use** | Station-area pop+emp density, zoning, parking | Spatial cache, existing outputs | Partial |
| **Economic development** | Station-area development potential, regional growth | Development model outputs | Partial |

### 10.2 Cost-Effectiveness Index

FTA's primary screening metric:

```
annualized_capital = capital_cost × CRF(5%, 30yr)  [capital recovery factor, 30-yr FTA standard]
annualized_cost = annualized_capital + annual_O&M
cost_per_trip = annualized_cost / annual_linked_trips

FTA thresholds:
  Low: > $6.00/trip
  Medium-Low: $4.00–6.00
  Medium: $2.00–4.00
  Medium-High: $1.00–2.00
  High: < $1.00
```

---

## 11. Output Artifacts Specification

### 11.1 Economic Impact Summary CSV (`economic_impact_summary.csv`)

One row per corridor per scenario. Label each column with its accounting frame.

```
corridor_id, scenario,

# [FINANCIAL] — Project Cash Flow (existing, carried forward)
capital_cost_musd, annual_om_musd, annual_debt_service_musd,
annual_farebox_musd, tif_revenue_cumulative_musd,
debt_coverage_ratio, project_npv_musd, project_irr,

# [SOCIETAL] — Benefit-Cost Analysis (new)
annual_vmt_avoided_miles, annual_person_hours_saved,
benefit_travel_time_npv_3pct_musd, benefit_travel_time_npv_7pct_musd,
benefit_voc_savings_npv_3pct_musd, benefit_voc_savings_npv_7pct_musd,
benefit_safety_npv_3pct_musd, benefit_safety_npv_7pct_musd,
benefit_emissions_npv_3pct_musd, benefit_emissions_npv_7pct_musd,
benefit_health_walking_npv_3pct_musd, benefit_health_walking_npv_7pct_musd,
benefit_parking_avoided_npv_3pct_musd, benefit_parking_avoided_npv_7pct_musd,
benefit_property_spillover_npv_3pct_musd, benefit_property_spillover_npv_7pct_musd,
benefit_agglomeration_npv_3pct_musd, benefit_agglomeration_npv_7pct_musd,
cost_public_services_npv_3pct_musd, cost_public_services_npv_7pct_musd,
cost_capital_npv_3pct_musd, cost_capital_npv_7pct_musd,
cost_om_npv_3pct_musd, cost_om_npv_7pct_musd,
residual_value_npv_3pct_musd, residual_value_npv_7pct_musd,
total_benefits_npv_3pct_musd, total_benefits_npv_7pct_musd,
total_costs_npv_3pct_musd, total_costs_npv_7pct_musd,
bcr_3pct, bcr_7pct,
bcr_3pct_p10, bcr_3pct_p50, bcr_3pct_p90,
bcr_7pct_p10, bcr_7pct_p50, bcr_7pct_p90,

# [ECONOMIC] — Gross Activity (NOT additive to BCR)
construction_jobs_apm, construction_jobs_tod,
construction_earnings_musd, permanent_ops_jobs,
permanent_tod_jobs, permanent_tod_earnings_musd,
private_investment_musd, leverage_ratio_gross,
leverage_ratio_net_of_baseline,

# [FISCAL] — Government Revenue
tif_annual_yr10_musd, tif_cumulative_25yr_musd,
tif_residential_truncated_20yr_musd,
lit_new_jobs_annual_musd, lit_construction_annual_musd,
state_income_tax_annual_musd,
sales_tax_retail_annual_musd,
road_maintenance_savings_annual_musd,
parking_avoided_onetime_musd,
net_public_service_cost_annual_musd,
net_fiscal_benefit_annual_yr10_musd,
net_fiscal_benefit_cumulative_25yr_musd,

# [EQUITY]
riders_se01, riders_se02, riders_se03,
low_income_access_ratio,
transport_savings_se01_annual,
jobs_accessible_45min_se01_baseline, jobs_accessible_45min_se01_build,
accessibility_improvement_se01_pct,
zero_car_hh_served,
ej_index,

# [FTA]
fta_cost_per_trip,
fta_annual_trips_per_musd,
fta_station_area_pop_density,
fta_station_area_emp_density,
fta_annual_vmt_reduction,
fta_annual_co2_tons_avoided
```

### 11.2 One-Page Taxpayer Summary (per top-5 corridor)

A non-technical summary for public consumption. Generate as formatted text (or HTML) from model outputs:

```
═══════════════════════════════════════════════════════
  CORRIDOR [ID]: [NAME / DESCRIPTION]
  Economic Impact Summary
═══════════════════════════════════════════════════════

  FOR EVERY $1 OF PUBLIC INVESTMENT:
    $X.XX in community benefits (BCR at 3%)
    $X.XX in private development activity

  JOBS:
    X,XXX construction jobs over X years
    XXX permanent operations jobs
    X,XXX permanent jobs from new development

  TAX REVENUE:
    $X.XM in annual property tax from new development
    $X.XM in annual income tax from new jobs
    $XXM cumulative TIF revenue (25 years)

  TRANSPORTATION:
    X,XXX daily riders
    Equivalent to removing X,XXX cars from the road daily
    X.X million fewer vehicle miles per year
    X,XXX tons CO2 reduced annually

  EQUITY:
    X,XXX low-income residents gain new transit access
    $X,XXX annual transportation savings per eligible household
    X,XXX additional jobs accessible within 45 minutes

  HOW IT COMPARES:
    Morgantown PRT (WVU): 12,000 daily riders, university APM
    IndyGo Red Line (Indianapolis): $2.4B development, same state
    Cleveland HealthLine: $47.50 private investment per $1 public

  UNCERTAINTY RANGE (p10–p90):
    BCR: X.X – X.X    Ridership: X,XXX – X,XXX daily
    TIF Revenue: $XXM – $XXM    Jobs: X,XXX – X,XXX

═══════════════════════════════════════════════════════
```

### 11.3 Enhanced Decision Package

Extend `generate_decision_package.py` to include:

1. `economic_impact_summary.csv` (Section 11.1)
2. `fiscal_impact_by_year.csv` (annual fiscal flows for each revenue stream)
3. `bcr_sensitivity.csv` (BCR under different discount rates and assumption sets)
4. `comparable_projects.csv` (benchmarking table)
5. `equity_dashboard.csv` (income-segmented access metrics)
6. `fta_metrics.csv` (FTA Small Starts format)
7. `taxpayer_summary.txt` (Section 11.2)

---

## 12. Critical Risks and Honest Disclosures

A credible analysis must proactively acknowledge weaknesses. Omitting these invites critics to find them — and dismiss the entire study.

### 12.1 Mid-Size Metro Discount

Most transit property value and ridership studies are from metros >500K. Lafayette at 232K has less baseline demand pressure, a thinner real estate market, and less congestion (reducing the value of congestion relief).

**Mitigation:** Use conservative end of all ranges:
- Property premium: 4–6% peak (not 8–12% from large-metro studies)
- Agglomeration elasticity: lower-bound specification (Section 6.2)
- BCR: if result exceeds 3.0×, audit assumptions

### 12.2 CityBus Fragility

CityBus is facing 20–25% service cuts and lost Purdue campus routes. The feeder bus service assumed in the model may not be available without additional funding.

**Implication:** This is both a risk (less feeder coverage reduces ridership by 10–15%) and an opportunity (APM-as-CityBus-replacement narrative strengthens political case). Model a "degraded CityBus" sensitivity scenario with `feeder_coverage_fraction` reduced by 50%.

### 12.3 2023 TIF Law Change (Model Bug)

As detailed in Section 4.1B, the model currently uses 25-year TIF for all uses. Indiana HEA 1120 (2023) caps residential TIF at 20 years. This must be fixed before any results are presented externally.

**Impact:** Reduces cumulative TIF by 15–20% for residential-heavy corridors. May shift some corridors from "viable" to "marginal" on debt coverage.

### 12.4 "But-For" Legal Threshold

Indiana TIF requires a higher standard than "development is larger with APM." It requires demonstrating that the specific increment would not have occurred without the public investment. The model's 2% baseline growth counterfactual is analytically sound, but the legal finding requires a declaratory resolution by the redevelopment commission.

**Mitigation:** Frame model output as "analytical support for the but-for finding," not the finding itself. Show the delta explicitly.

### 12.5 No Precedent for Transit LIT in Small Counties

The 0.25% transit income tax (IC 8-25-2) has only been used in Marion County (978K). Tippecanoe County (193K) has no precedent. The revenue estimate ($11.8M/yr) is real, but political feasibility is uncertain.

**Mitigation:** Present as a "funding scenario" alongside TIF-only and TIF+federal scenarios. Do not assume transit LIT in the base case.

### 12.6 Morgantown PRT Cost Overrun

Morgantown PRT, the closest technology analogue, was 3 years late and 3–4× over original budget estimate. Any cost presentation must include Monte Carlo uncertainty bands prominently. The existing p10/p90 on capital cost (0.9–1.25×) may be too narrow given Morgantown's experience.

**Mitigation:** Add a "Morgantown scenario" sensitivity with capital cost at 2.0× baseline. Show BCR and DCR under this stress test.

### 12.7 Regional Displacement

Development captured by APM corridors is partially redirected from elsewhere in the Lafayette metro, not entirely net new. The model's REGIONAL_REALLOCATION_ELASTICITY (0.05) accounts for this modestly, but the fiscal impact analysis should distinguish:

- **Corridor-level impact** (gross): appropriate for the TIF district analysis
- **Metro-level impact** (net of displacement): appropriate for the BCR and countywide fiscal analysis

Presenting only corridor-level figures without disclosing displacement risk invites criticism.

### 12.8 Induced Demand Uncertainty

The model's induced demand component (INDUCED_TRIP_ELASTICITY = 0.10) and latent demand component together add ~6% to ridership at maturity. These are empirically grounded (TCRP 95, ACS B08201), but their interaction is uncertain. The Monte Carlo already varies ridership ±20%, which encompasses this uncertainty.

---

## 13. Implementation Priority and Phasing

### Phase 1 — Core Fiscal Impact Module (Highest Leverage)

**Effort:** 1–2 weeks. All inputs already exist in model outputs.

| New Metric | Derivation | Input Available? |
|-----------|-----------|-----------------|
| `construction_jobs_apm` | `capital_cost_M × 30` | Yes (`capex_musd`) |
| `construction_jobs_tod` | `(res_sqft × $150 + comm_sqft × $180) / 1M × 30` | Yes (feedback loop) |
| `permanent_ops_jobs` | Lookup by `length_km` (50–100 FTEs) | Yes |
| `permanent_tod_jobs` | `new_comm_sqft / 200` | Yes (already `new_jobs`) |
| `lit_revenue_annual` | `new_jobs × $42K × 0.011` | Yes |
| `annual_vmt_avoided` | `daily_car_diversions × avg_dist × 2 × 300` | Yes (mode choice) |
| `travel_time_savings_hrs` | `daily_riders × (car_ivt - apm_ivt) / 60 × 300` | Yes (mode choice) |
| `private_investment_musd` | `res_sqft × $150 + comm_sqft × $180` | Yes |
| `leverage_ratio` | `private_investment / capital_cost` | Yes |
| TIF 20-yr residential fix | Split res/comm TIF streams | Yes (need res/comm share) |

**Deliverable:** `fiscal_impact_summary.csv` with one row per corridor per scenario.

### Phase 2 — Benefit-Cost Ratio (Most Important Single Number)

**Effort:** 2–3 weeks. Requires monetizing benefits from Phase 1 inputs and building the NPV/discount-rate framework.

| New Metric | Derivation |
|-----------|-----------|
| `benefit_travel_time_npv` | `person_hours_saved × $18.80/hr`, NPV at 3% and 7% |
| `benefit_voc_savings_npv` | `vmt_avoided × $0.22/mi`, NPV |
| `benefit_safety_npv` | `vmt_avoided × $0.12/VMT`, NPV |
| `benefit_emissions_npv` | `vmt_avoided × $0.086/VMT` (with 2.5%/yr SC-CO2 escalation), NPV |
| `benefit_health_npv` | `walk_minutes × $0.15/min × 0.60`, NPV |
| `benefit_parking_npv` | `avoided_spaces × $35K / 30yr`, NPV |
| `benefit_spillover_npv` | Uplift on parcels 400–800m (outside TIF), NPV |
| `cost_public_services_npv` | `new_pop × $987/yr`, NPV |
| `residual_value_npv` | 50% of capital at year 25 |
| `bcr_3pct`, `bcr_7pct` | `sum(benefits) / sum(costs)` |
| `bcr_p10/p50/p90` | Monte Carlo propagation |

**Deliverable:** BCR columns added to integrated financial CSV. Monte Carlo extended.

### Phase 3 — Equity & FTA Framing

**Effort:** 1–2 weeks. Extends existing income-segmented outputs.

| New Metric | Derivation |
|-----------|-----------|
| `transport_savings_se01` | `(car_cost - apm_cost) × trips × SE01_hh` |
| `jobs_accessible_45min_baseline/build` | Zone-to-zone travel time matrix + LODES |
| `accessibility_improvement_se01_pct` | `(build - baseline) / baseline` |
| `fta_cost_per_trip` | `annualized_cost / annual_trips` |
| `fta_station_area_density` | Pop+emp within 805m |
| `ej_index` | ACS race/income within catchment vs. metro |

**Deliverable:** `equity_dashboard.csv`, `fta_metrics.csv`

### Phase 4 — Public-Facing Narrative & Decision Package

**Effort:** 1 week. Post-processing and formatting.

| Deliverable | Content |
|------------|---------|
| Taxpayer summary (per top corridor) | One-page formatted output (Section 11.2) |
| Comparable projects table | Section 7.1, embedded in decision package |
| Risk disclosures | Section 12, embedded in decision package |
| Enhanced `generate_decision_package.py` | All new CSVs + narrative output |

### Phase 5 — Agglomeration (Highest Analytical Sophistication)

**Effort:** 3–4 weeks. Requires zone-to-zone cost matrix infrastructure.

| Deliverable | Content |
|------------|---------|
| Zone-to-zone generalized cost matrix (before/after) | From mode choice model |
| Effective density by zone and sector | Graham framework computation |
| Productivity gain by sector | Elasticity × density change × GDP/worker |
| `benefit_agglomeration_npv` | Added to BCR (conservative specification) |

**This phase is optional for initial presentations** but strongly recommended for FTA applications or academic publication, particularly given Purdue's knowledge-economy concentration.

---

## 14. Data Gaps and Sources

| Gap | Solution | Cost | Priority |
|-----|----------|------|----------|
| USDOT VTTS / VSL / BCA values | Published in USDOT BCA Guidance 2024 — free | $0 | Phase 2 |
| AAA per-mile vehicle costs | Published annually (2024: $0.655 full, $0.22 marginal) | $0 | Phase 1 |
| EPA social cost of carbon | EPA SC-CO2 schedule (2024 update: $190/ton) | $0 | Phase 2 |
| Indiana crash rates by county | NHTSA FARS + Indiana ARIES database | $0 | Phase 2 |
| Tippecanoe County per-capita service cost | County budget / population (public record) | $0 | Phase 2 |
| BLS wages by sector (Lafayette MSA) | BLS OES / QCEW (supplements LODES) | $0 | Phase 1 |
| Retail sales per sqft by category | Census Annual Retail Trade Survey | $0 | Phase 1 |
| Agglomeration elasticities by sector | Graham (2007), UK DfT WebTAG — published | $0 | Phase 5 |
| BEA GDP per worker by sector (Tippecanoe) | BEA CAINC6N (free download) | $0 | Phase 5 |
| Tippecanoe County RIMS II multipliers | Purchase from BEA (bea.gov/resources/methodologies/RIMSII) | ~$275 | Optional |
| ACS B25070 (rent burden by income) | Census API | $0 | Phase 3 |
| ACS B08201 (vehicles available) | Census API (already partially loaded) | $0 | Phase 3 |
| ACS B03002 (race/ethnicity by tract) | Census API | $0 | Phase 3 |

**Total out-of-pocket cost:** $0–$275 (only RIMS II purchase is non-free, and is optional).

---

## 15. Methodological References

| Method | Source | Used In |
|--------|--------|---------|
| Benefit-Cost Analysis framework | USDOT BCA Guidance (2024 update) | Phase 2 |
| Value of Travel Time Savings | USDOT Revised Departmental Guidance (2024, 2022$) | Phase 2 |
| Value of Statistical Life | USDOT VSL = $12.5M (2022$) | Phase 2 |
| Social Cost of Carbon | EPA/OMB Interim SCC (2024), $190/ton at 3% near-term | Phase 2 |
| Construction employment multiplier | APTA Economic Impact of Public Transportation (2020) | Phase 1 |
| RIMS II regional multipliers | BEA Regional Input-Output Modeling System | Phase 1 (optional) |
| Vehicle operating costs | AAA Your Driving Costs (2024) | Phase 1–2 |
| Walking health benefits | WHO HEAT for Walking and Cycling (2024 update) | Phase 2 |
| Crash costs per VMT | FHWA Highway Statistics; NHTSA FARS | Phase 2 |
| Transit property value premium | Meta-analysis, JTLU 2022; Chatman & Noland 2014 | Phase 2 |
| Agglomeration elasticities | Graham (2007); UK DfT WebTAG Unit A2.4 | Phase 5 |
| FTA project justification criteria | 49 CFR 611.203; FTA CIG Policy Guidance (Dec 2024) | Phase 3 |
| Indiana TIF statute | IC 36-7-14; HEA 1120 (2023 amendment) | Phase 1 |
| Indiana LIT / transit LIT | IC 6-3.6; IC 8-25-2 et seq. | Phase 1 |
| Fiscal impact methodology | TCRP Report 78; Burchell & Listokin (1978) | Phase 1–2 |
| Equity / environmental justice | FTA Title VI Circular 4702.1B; EO 12898 | Phase 3 |
| DiPasquale-Wheaton demand model | DiPasquale & Wheaton (1992) | Existing |
| TCRP transit self-selection | TCRP Report 128 | Removed (self_selection_mult = 1.0; logit captures mode share directly) |
| Trip generation elasticities | TCRP Report 95 | Existing |

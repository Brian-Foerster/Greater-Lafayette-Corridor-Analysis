# Frequently Asked Questions

## For General Audiences

### What is an APM / people mover?

An Automated People Mover (APM) is a small, driverless transit system that runs
on its own elevated track — separate from cars and pedestrians. You've probably
ridden one at an airport (like the trains connecting terminals at Dallas/Fort
Worth or Orlando). City APMs work the same way but connect neighborhoods,
universities, and downtown areas. They run every few minutes, don't get stuck in
traffic, and operate rain or shine without a driver.

Bus Rapid Transit (BRT) is a less expensive alternative: dedicated bus lanes
with station platforms, pre-paid boarding, and signal priority. It's faster and
more reliable than regular bus service, but still shares the road to some degree.

### Why Greater Lafayette?

Lafayette and West Lafayette are twin cities separated by the Wabash River, with
Purdue University (54,000+ students, 15,000+ faculty/staff) on the west side and
the commercial core on the east. The geography creates natural transit demand:

- **River barrier**: Only a few bridges connect the two cities, creating
  bottleneck congestion — State Street and Northwestern Avenue are at or near
  capacity during peak hours
- **University growth**: Purdue's enrollment has grown 20% since 2020, driving
  a construction boom (Rise on Chauncey, Hub, The Standard) and increasing
  traffic pressure
- **Existing transit use**: CityBus carries ~1.2 million riders annually, and
  Purdue operates campus shuttles, demonstrating that transit demand already
  exists in this market
- **Two downtowns**: West Lafayette's Chauncey Village area and Lafayette's
  downtown are both densifying with mixed-use development
- **Linear geography**: The population centers align roughly east-west across
  the river, well-suited to a corridor-based transit system

### What corridors were evaluated?

The computer tested thousands of possible routes by combining different station
locations along the road network. An optimization algorithm (similar to how
nature evolves better solutions over generations) narrowed these down to
10-40 diverse corridors that represent different tradeoffs between ridership,
cost, and geographic coverage.

Each corridor was then simulated for 25 years to see how ridership would grow,
what development it would attract, and whether it could pay for itself.

### What do the financial numbers mean?

- **Capital cost**: How much it costs to build (e.g., $1.3 billion for a 13 km
  APM). This is the upfront construction price.
- **DCR (Debt Coverage Ratio)**: How much revenue the system generates compared
  to its annual debt payments. A DCR of 0.10x means revenue covers 10% of debt
  — the rest would need subsidies or grants. A DCR of 1.0x or higher means the
  system pays for itself.
- **NPV (Net Present Value)**: The total financial gain or loss over 25 years,
  adjusted for the time value of money. A negative NPV (like -$1.2 billion)
  means the project costs more than it earns — which is normal for transit
  projects that provide social benefits beyond fare revenue.
- **Cost per rider**: How much public subsidy each daily rider effectively
  receives. Lower is better.
- **TIF revenue**: Property tax income from new development near the transit
  line. When transit attracts new apartments and offices, their property taxes
  help pay for the system.

### Why does no corridor "break even"?

Because the model assumes 100% local funding — no federal grants, no state
contributions. This is intentionally conservative: it shows what each corridor
can generate on its own, from fares and local property taxes alone.

Real transit projects typically receive 50-80% federal funding through FTA
grants (New Starts / Small Starts programs). With 50% federal funding, several
corridors would approach financial viability. The model's "worst case" funding
assumption makes the financial comparison between corridors meaningful without
speculating about which federal programs might be available.

### What's the difference between current zoning and no zoning?

- **Current zoning** uses the actual building rules from Tippecanoe County's
  zoning code. Single-family neighborhoods can only build single-family homes.
  Multi-family zones have height and density limits.
- **No zoning** removes all building restrictions within the transit corridor,
  allowing the market to decide what gets built (up to about 10 stories).

These two scenarios bracket the range: current zoning shows what happens with
today's rules, and no zoning shows the theoretical maximum if all regulatory
barriers were removed. Reality would fall somewhere in between — a targeted
"transit overlay" district that allows more density near stations.

### How does the bus network change when transit is built?

When an APM or BRT line opens, the existing CityBus network gradually adapts:

- **Parallel bus routes** (running the same direction as the transit line)
  reduce frequency, since riders switch to the faster option
- **Feeder bus routes** (running perpendicular, bringing riders to stations)
  are created or improved using the hours freed from parallel reductions
- **Other routes** stay the same to maintain coverage for areas the transit
  line doesn't serve

This happens gradually over several years as ridership grows, not all at once.
The model tracks this year by year within the CityBus operating budget.

### Why do students matter so much for ridership?

Purdue students make up 46-53% of daily riders on campus-proximate corridors
because:

- Students have low car ownership (~35% drive alone, vs. ~85% for the general
  population)
- Campus generates concentrated, predictable travel patterns (classes, dining,
  recreation)
- Student housing clusters near campus create high-density catchment zones

A corridor that misses the campus area loses roughly half its potential
ridership.

### How should decision-makers use these results?

This is a screening tool, not a final design. It answers "which corridors
are worth studying further?" and "how do APM and BRT compare?" — not "exactly
where should we put stations" or "exactly how many riders will we get."

The model is most useful for:
- Comparing corridors against each other (relative ranking)
- Understanding the APM vs. BRT tradeoff (cost vs. ridership)
- Identifying which zoning changes would matter most for transit success
- Estimating the order of magnitude of ridership and financial outcomes

It should not be used to:
- Set exact fare prices or headway schedules
- Predict ridership to within 10% accuracy
- Make final route alignment decisions (those require engineering studies)

### Why APM instead of light rail or streetcar?

Three reasons favor APM for a metro area of this size:

- **Automation**: APMs run without drivers, which cuts operating costs by
  30-40% compared to light rail. For a 230,000-person metro that can't
  support high-frequency staffed service, this is decisive — it means
  trains every 3-5 minutes instead of every 15-20.
- **Grade separation**: APMs run on elevated guideway above traffic. They
  never wait at red lights, never hit cars, and never block intersections.
  Light rail at street level faces all of these issues and typically averages
  only 15-20 km/h in urban settings — barely faster than a bus.
- **Right-sized vehicles**: APM cars carry 50-100 passengers (vs 200+ for
  light rail). Smaller vehicles running more frequently provide a better
  passenger experience at the ridership levels this market generates.

The model also evaluates BRT (Bus Rapid Transit) as a lower-cost alternative
for every corridor, providing a direct cost-benefit comparison between the
two most realistic technology options.

### Why does the model use only online data?

Every input to the model comes from publicly available online sources — no
surveys, no traffic counts, no focus groups. This is a deliberate choice:

- **Reproducibility**: Anyone can download the same data and verify the results
- **Cost**: Travel surveys cost $500K-$2M and take 12-18 months to administer
- **Coverage**: LODES provides origin-destination flows for every worker in the
  county; a travel survey would sample only 1-3% of households
- **Timeliness**: Census/LODES data is updated annually; survey data is stale
  within 3-5 years

The tradeoff is precision: mode choice parameters come from national research
(TCRP, FTA) rather than local calibration, and parking costs are estimated by
location type rather than counted lot-by-lot. This makes the model better for
*relative* corridor comparison than for *absolute* ridership prediction.

### What happens to CityBus when transit is built?

CityBus continues operating with its full current budget ($13.5M/year for
fixed-route service). The model restructures bus routes around the APM/BRT
corridor — reducing parallel routes that duplicate the new service and
creating feeder routes that bring riders to stations — but keeps the total
bus budget constant. No CityBus routes are eliminated entirely; low-ridership
routes are maintained at reduced frequency as an equity floor.

### What are the model's limitations?

- **No travel behavior surveys**: Mode choice parameters are calibrated from
  national research, not local preferences
- **No parking inventory**: Parking costs are estimated by location type
  (campus, downtown, suburban), not counted lot-by-lot
- **No political feasibility**: The model doesn't know which routes would face
  community opposition or environmental review delays
- **No construction market**: Building costs assume normal market conditions,
  not labor shortages or materials inflation
- **Corridors are evaluated independently**: Building two transit lines would
  affect each other's ridership, but the model evaluates each one alone
- **No agglomeration**: Improved transit connectivity doesn't attract new
  employers in the model — jobs grow at a fixed regional rate

---

## Reading the Viewer

### What am I looking at?

The interactive viewer shows a map of Greater Lafayette with colored lines
representing candidate transit corridors and white dots marking station
locations. Click any corridor to see its detailed metrics in the sidebar panel.

- **Colored lines**: Each corridor is a different color. The line connects
  stations in a straight path showing the corridor's alignment. Thicker
  lines indicate higher ridership.
- **Station dots**: White circles mark where stations would be located. These
  are positioned at major road intersections and demand centers along the
  corridor.
- **Scenario selector**: Toggle between "Current Zoning" and "No Zoning" to
  see how relaxing building restrictions changes development outcomes and
  ridership.

### What do the sidebar numbers mean?

When you click a corridor, the sidebar shows three panels:

**Top panel (APM):**
- **Y25 Ridership**: Predicted daily riders in year 25 (academic-year weekday).
  Multiply by 0.86 for annual average.
- **Capital Cost**: Total construction cost including guideway, stations,
  vehicles, and barrier crossings (e.g., bridges over the Wabash River).
- **Bus Restructuring**: What percentage of existing CityBus routes would be
  restructured into feeder service, and the resulting feeder headway.
- **Housing Units / Population / Jobs**: Cumulative new development attracted
  to the corridor over 25 years.
- **DCR**: Debt Coverage Ratio — revenue ÷ debt payments. Below 1.0x means
  the corridor needs subsidies. Under the 100% local funding assumption,
  all corridors show DCR below 1.0x.
- **NPV**: Net Present Value — total financial outcome over 25 years. Negative
  values are normal for transit projects evaluated without federal grants.
- **Cost/Rider**: Annual public subsidy per daily rider. Lower is more
  efficient.

**Middle panel (BRT Alternative):**
Same metrics computed for a Bus Rapid Transit alternative on the same
corridor. BRT has lower capital cost but also lower ridership and less
development impact.

**Bottom panel (Charts):**
Three charts show the corridor's 25-year trajectory.

### What does the ridership trajectory chart show?

The ridership chart shows daily riders from year 0 (opening day) to year 25.
The S-shaped growth curve reflects three dynamics:

- **Years 0-3**: Low initial ridership as awareness builds. Residents and
  commuters are still learning the system exists and adjusting travel habits.
- **Years 3-10**: Rapid growth as awareness matures, new development near
  stations adds population, and the bus network restructures to feed the
  transit line.
- **Years 10-25**: Growth moderates as the corridor approaches its natural
  demand ceiling. Development continues but at a steady pace rather than a
  boom.

"Daily riders" means average weekday during the academic year (when Purdue
is in session). Multiply by 0.86 for the annual average including summer
and breaks.

### What does the development timeline chart show?

The development chart shows three cumulative curves:

- **Units** (blue): Total housing units built near stations over 25 years.
  Includes both market-rate apartments and student housing.
- **Population** (green): New residents attracted to the corridor. Rises
  faster than units because each unit houses ~2.6 people on average.
- **Jobs** (orange): New commercial employment. Smaller than residential
  because commercial development follows rooftops — offices and retail
  arrive after housing is built.

The "New Units per Period" bar chart below shows the year-by-year delivery
rate. A typical pattern: low in years 0-3 (developers waiting to see
ridership), peak in years 5-8 (confidence established, construction boom),
then settling to 150-250 units/year as the market matures.

### What does the ridership components chart show?

The stacked bar shows where riders come from:

- **Non-work** (light blue): Local trips — shopping, errands, recreation,
  medical appointments — from residents within walking distance of stations.
  Often the largest component for corridors away from campus.
- **Student** (purple): Purdue students traveling to/from campus for classes,
  dining, recreation, and off-campus activities. Dominant for corridors that
  pass through or near the university.
- **Destination** (green): Trips attracted by specific destinations —
  retail centers, entertainment, medical facilities — near stations.
- **Work Commute** (dark blue): LODES journey-to-work trips where both home
  and workplace are within walking distance of stations. Typically small
  in this model because most Lafayette employers are outside the corridor
  catchment.
- **Induced** (orange): New trips that wouldn't exist without transit —
  primarily zero-car households gaining mobility they didn't previously have.
- **Latent** (red): Suppressed demand from populations underserved by current
  transit, released as the new system matures and feeder coverage improves.

Components with zero riders are hidden from the chart.

---

## Technical Details

### How is ridership computed?

Ridership has five components:

1. **Work commute** — LODES origin-destination flows through the corridor,
   allocated by a multinomial logit (MNL) mode choice model with income
   segmentation (SE01/SE02/SE03 earnings brackets)
2. **Student** — Purdue enrollment-based campus catchment with purpose-split
   trip generation (class, dining, recreation, employment)
3. **Local non-work** — non-commute trips from walk-zone population
4. **Induced** — new trips that wouldn't occur without transit (zero-car
   households gaining mobility)
5. **Latent** — trips from populations underserved by current transit

Each component runs through a 4-mode MNL (APM/BRT, bus, car, walk) with
period-specific car speeds and bus headways (AM peak, midday, PM peak,
evening).

### Why is BRT ridership only 55-75% of APM ridership?

The MNL mode choice model produces a modest ridership gap because only a
small fraction of the catchment population is sensitive to the speed and
frequency differences between BRT and APM.

For a typical 3 km trip with a station 400m away:
- APM in-vehicle time: 6.7 minutes at 27 km/h
- BRT in-vehicle time: 8.0 minutes at 22.5 km/h
- Difference: 1.3 minutes

This 1.3-minute difference changes the mode choice for only about 6% of the
catchment population — the "car-competitive marginal riders" who would ride
APM but drive if the transit option is BRT. The other ~20% who ride BRT are
mode-insensitive: zero-car households, students with low car ownership,
downtown workers facing expensive parking, and preference riders.

The utility gap is actually dominated by the alternative-specific constant
(ASC), which captures the perceived quality, reliability, and modernity
differences between grade-separated transit and buses. The ASC accounts for
about 72% of the utility gap; actual speed and wait time differences account
for only 28%.

The ridership ratio varies with corridor length (longer corridors amplify the
speed gap): 74% for a 4 km corridor down to 55% for a 9 km corridor.

### What is the "permanence premium"?

Grade-separated transit infrastructure (concrete guideway, elevated stations)
is nearly impossible to remove once built. This "permanence" gives developers
confidence that transit service will persist for the 20-30 year horizon of
their investment. Bus routes, even dedicated-lane BRT, can theoretically be
rerouted or defunded.

The model captures this through two mechanisms:
- **Rent multiplier**: APM-adjacent properties are assessed at 12% above
  baseline (matching rail property value meta-analyses); BRT at 5% (matching
  high-quality BRT empirical premiums like Cleveland HealthLine)
- **Speculative discount**: Developers discount the transit rent premium before
  the system proves itself. APM developers realize 50% of the premium in year
  0; BRT developers realize 40%. Both converge to 100% by year 10.

The permanence premium is documented in GAO-12-811 (2012), which found
developers "emphasized the importance of physical features perceived as
permanent in helping to spur economic development."

### How does the model decide what gets built?

Development follows a four-step process each year:

1. **Demand**: A household relocation MNL computes how many metro households
   choose corridor-adjacent parcels. This accounts for rents, commute times,
   and transit accessibility by income segment.
2. **Feasibility**: Each candidate parcel is evaluated through the UrbanSim
   SqFtProForma, which tests 24 different building intensities (FAR 0.1 to
   11.0) and picks the most profitable design that fits within zoning limits.
3. **Selection**: The Developer module selects which feasible buildings to
   construct, targeting the demand from step 1.
4. **Delivery**: Buildings are delivered over 4-6 years using height-dependent
   occupancy schedules (low-rise fills in 4 years; high-rise takes 6 years
   and stabilizes at 92%).

Student housing follows a separate formula-driven path calibrated to observed
near-campus development in Lafayette (100 units/year base, FAR-responsive).

### Why do the two zoning scenarios produce different development?

**Current zoning** applies Tippecanoe County UZO lot coverage and height
limits, which effectively cap building intensity. An R1 (single-family) parcel
is limited to 1 dwelling unit by use restriction. An R3 (multi-family) parcel
can build to FAR 1.4 (~3.5 stories). Only CB (central business) zones allow
tall buildings (FAR 8.3, ~25 stories).

**No zoning** removes all FAR caps and dwelling-unit-per-acre limits,
replacing them with a market-determined ceiling of FAR 8.0 (~10 stories). This
represents the theoretical maximum development if all regulatory constraints
were removed.

The scenarios differ through several channels:
- **Supply headroom**: no_zoning allows 3.9x more theoretical building capacity
- **Demand**: no_zoning receives a 30% demand capture bonus (upzoning attracts
  more households to the corridor)
- **Cost**: no_zoning reduces regulatory costs by ~6% (fewer permits, less
  delay interest)
- **Developer hurdle**: no_zoning developers accept a 10% lower profit margin
  (reduced political risk)
- **Student housing**: FAR-responsive, so no_zoning allows taller student
  towers (~250 units/year vs ~100 under current zoning)

### Why doesn't development continue at high rates for all 25 years?

After an initial building boom (years 3-7), market-rate development moderates
because:

1. **Vacancy absorption**: New movers occupy both existing vacant units and new
   construction. The model subtracts existing vacancies from new-build demand,
   so a corridor with many vacant units sees less new construction.
2. **Rent feedback**: If construction outpaces absorption, vacancy rises and
   rents decline, making new projects less profitable.
3. **Transit rent floor**: Rents within 800m of stations cannot fall below 82%
   of initial values (empirically, transit-adjacent rents never crash to the
   levels the pure vacancy model would produce).
4. **Replacement demand**: Existing stock depreciates at 0.4%/year, creating
   steady replacement demand even when net growth slows.

The result is a trajectory that rises sharply in years 3-7, moderates to
steady-state delivery of 100-200 units/year, and continues through year 25 —
matching observed TOD development patterns in peer corridors (Minneapolis
Green Line: ~490 units/year sustained over 13 years).

### What is TIF and how is it computed?

Tax Increment Financing (TIF) captures the property tax revenue from
*increased* assessed values within a designated district around the transit
corridor. The model computes TIF endogenously:

1. New development produces assessed value (residential at $130/sqft,
   commercial at $110/sqft, adjusted by mode-specific premium)
2. Each year's new construction enters the tax rolls after a 1-year
   assessment lag (Indiana taxes paid in arrears)
3. Indiana circuit breaker caps limit effective tax rates by property class
   (1% homestead, 2% rental, 3% commercial)
4. SB 1 (2025) capture-rate erosion reduces effective TIF collection over time
5. Economic Development Area (EDA) rules exclude homestead increment from
   capture

The result is a more conservative TIF estimate than simple
"assessed value x tax rate" calculations, reflecting Indiana-specific
institutional constraints.

### How does BRT compare financially?

BRT has roughly 1/3 the capital cost of APM ($25M/km vs $100M/km) but also
lower ridership (55-75% of APM) and a smaller property value premium (5% vs
12%). The financial comparison often favors BRT on DSCR because the lower
capital cost reduces debt service more than the lower ridership reduces fare
revenue.

However, BRT produces less TIF revenue (lower rent premium = lower assessed
value increment) and generates less long-term development. The model provides
both APM and BRT evaluations for every corridor so decision-makers can compare
the full financial profile.

### How does the bus network respond to APM/BRT?

The model includes a dynamic bus restructuring engine that adjusts CityBus
service as the APM/BRT corridor matures:

- **Parallel routes** (running alongside the corridor) gradually increase
  headway as ridership transfers to APM
- **Feeder routes** (perpendicular, bringing riders to stations) are created
  or strengthened with hours freed from parallel reductions
- **Independent routes** (not overlapping the corridor) are held at current
  service levels as an equity floor

Restructuring is budget-constrained: the total bus operating budget ($13.5M,
matching CityBus FY2024 NTD-reported fixed-route operating expenses) is
reallocated between parallel and feeder service. A budget-feasibility ceiling
prevents the model from promising feeder service the budget can't fund.

### What is bus restructuring "pressure"?

Pressure is a 0-1 score measuring how strongly the APM corridor justifies bus
network changes. It depends on APM ridership relative to a maturity target
(2,500 riders/day), incumbent bus competitiveness, and route productivity. At
pressure 0.0, no restructuring occurs. At pressure 1.0, full feeder-dominant
configuration.

Restructuring triggers when ridership changes by more than 5% since the last
event, producing smooth headway transitions rather than sudden phase jumps.

### How are corridors generated?

Stage 1 uses a multi-objective evolutionary algorithm (NSGA-II) to search for
high-performing corridors:

1. **Candidate stations** are road-network intersections on primary/secondary
   roads, demand-qualified minor-road nodes, and explicit points of interest
2. **Dynamic programming** selects a subset of stations along each path to
   maximize demand coverage
3. **NSGA-II** evolves a population of corridors over 15+ generations,
   optimizing ridership, cost efficiency, and financial viability
4. **Diversity selection** ensures the final set covers different parts of the
   metro using geographic overlap thresholds

Corridors are validated against physical constraints: minimum curve radius
(50m for revenue service), maximum grade (6%), and effective speed floor (50%
of line speed). A dogleg detector penalizes corridors that zigzag to chase
demand pockets.

### Why do corridors sometimes follow unexpected paths?

The optimizer maximizes ridership within physical constraints, which can
produce corridors that deviate from the obvious straight-line path. Two
mechanisms can cause this:

- **Demand chasing**: The DP station selection optimizes total demand without
  a geometry penalty. A pocket of demand 200m off the main arterial can pull
  the corridor sideways.
- **Road-graph routing**: Stations are placed at road intersections, and the
  alignment follows the road network. Grid street patterns force right-angle
  turns at intersections, which are smoothed for display but reflect the
  underlying road geometry.

Post-selection arterial snapping moves stations to the nearest primary or
secondary road (within 200m), and Chaikin smoothing rounds intersection
corners into arcs approximating engineered guideway curves.

### How is uncertainty quantified?

Stage 3 runs a 500-draw Monte Carlo simulation for each corridor, sampling
from triangular and lognormal distributions for:
- Ridership multiplier (0.80-1.25)
- TIF revenue multiplier (0.60-1.40)
- Capital cost multiplier (0.75-1.50)
- Operating cost multiplier (0.85-1.25)
- Fare multiplier (0.85-1.15)
- Discount rate shift (+/- 1.5%)
- Student demand multiplier (0.65-1.40)

Parameters are correlated using a Gaussian copula with per-scenario
correlation matrices (e.g., ridership and TIF are more correlated under
permissive zoning because development responds more elastically to ridership
changes).

Results are reported as percentile bands (p10/p50/p90) for DSCR, NPV,
ridership, and financial viability probability.

### What are the stress tests?

Four named stress scenarios test corridor resilience:
- **Enrollment shock**: 30% student demand drop + negative employment growth
- **Cost blowout**: 40% capital cost overrun + 30% O&M increase
- **Transit boom**: 30% ridership increase + 20% higher TIF capture
- **Stagflation**: negative job growth + 25% capital cost increase + 10% fare
  reduction

Each stress scenario produces a modified DSCR, NPV, and self-sufficiency
ratio, showing which corridors are robust to adverse conditions.

### What does the model NOT capture?

- **Agglomeration effects**: No feedback from improved transit connectivity to
  employer location decisions or labor market size
- **Induced trips beyond mode shift**: Trip generation from improved
  accessibility is partially captured but not fully modeled
- **Construction market dynamics**: No labor/materials shortage modeling beyond
  a county-capacity cost escalation
- **Political feasibility**: Corridor routing ignores land acquisition
  difficulty, NIMBY opposition, and environmental review timelines
- **Maintenance downtime**: APM maintenance outages (5-10% of service hours
  for small systems) are not modeled, overstating APM reliability advantage
- **Parking supply changes**: The model uses static parking costs by location,
  not dynamic parking supply that could change with development
- **Network effects between corridors**: Each corridor is evaluated
  independently; synergies or competition between overlapping corridors are
  not captured

### How sensitive is the model to its assumptions?

The Monte Carlo uncertainty analysis (500 draws) quantifies parametric
sensitivity. The most influential parameters are:
1. **Ridership multiplier** — directly scales fare revenue and ridership-driven
   development
2. **Capital cost multiplier** — dominates the financial viability calculation
3. **TIF revenue multiplier** — determines whether property tax increment can
   service debt
4. **Student demand** — Purdue enrollment is the single largest ridership
   source for campus-proximate corridors

The model is relatively insensitive to fare assumptions (narrow range,
partially offset by ridership elasticity) and operating cost assumptions
(O&M is small relative to debt service).


### Why does the model assume 100% local funding?

All financial metrics (DCR, NPV, cost per rider) assume the full capital cost
is financed locally through 25-year municipal bonds at 5% interest. No
federal grants (FTA New Starts/Small Starts) or state contributions are
included. This produces uniformly low DCR values (0.06-0.34x) — every
corridor requires subsidy under this assumption.

This is intentional: it makes corridors financially comparable without
speculating about which federal programs might be available or what match
ratio a specific project might receive. Federal funding typically covers
50-80% of capital costs for qualifying projects. To estimate DCR with federal
funding, multiply the reported capital cost by (1 - federal_share) and
recompute debt service.

The 100% local assumption also stress-tests the TIF mechanism: if a corridor
can cover 25% of debt service from local TIF alone, it would likely be
financially viable with a typical 60% federal match.

### How are zoning FAR limits derived?

The Tippecanoe County Unified Zoning Ordinance (UZO) does not regulate
density through Floor Area Ratio (FAR) directly. Instead, it specifies lot
coverage percentage, maximum building height, and setback requirements per
zone. The model derives an effective FAR using the formula:
effective FAR = lot coverage % × (max height in feet / story height in feet).

For example, R3 (multi-family) has 40% lot coverage and 35ft height, giving
0.40 × (35/10) = 1.4. This means a 3.5-story building covering 40% of the
lot.

For single-family zones (R1, R2), the UZO also restricts use — only one or
two dwelling units per lot, regardless of what the building envelope allows.
The model enforces this through a dwelling units per acre (DUA) cap that
the proforma respects. An R1 lot with FAR 1.0 could theoretically hold an
8-unit building, but the 4.35 DUA cap limits it to 1 unit — matching the
UZO's single-family restriction.

Under the no_zoning scenario, both FAR caps and DUA restrictions are removed,
allowing the market to determine building size.

### Why is the BRT property value premium 5%, not zero?

Early versions of the model assumed BRT produces no property value premium
(BRT_RENT_MULT = 1.00). This was too conservative. Empirical evidence shows
high-quality BRT with dedicated lanes and substantial stations *does* produce
measurable premiums:

- **Cleveland HealthLine**: 18% higher office rents, 42% multi-family premium
- **Eugene EmX**: 12% residential premium
- **Kansas City MAX**: 12% office rent premium
- **Meta-analysis (2020)**: Mature BRT systems average ~4.3% property value
  increase

The model now uses 5% for BRT (matching the meta-analysis median for
high-quality systems) and 12% for APM (matching the rail meta-analysis).
The 7-percentage-point gap aligns with the literature finding that rail
premiums are roughly double BRT premiums.

This matters for TIF revenue: the endogenous TIF calculation multiplies new
construction assessed value by the mode-specific rent multiplier. A 5% BRT
premium produces meaningfully less TIF than a 12% APM premium — making the
financial case for APM stronger on the revenue side even though BRT wins on
capital cost.

### How does the model handle the Wabash River?

The Wabash River is the defining geographic barrier of the Greater Lafayette
metro. It affects the model in three ways:

1. **Bridge crossing cost**: Corridors that cross the river incur an $80M
   penalty per crossing (elevated guideway over water requires deeper
   foundations, longer spans, and marine construction). Highway crossings
   (I-65) add $40M; railroad crossings add $25M.

2. **Feeder coverage asymmetry**: The river limits bus access to stations.
   A station on the West Lafayette side can only draw feeder riders from
   West Lafayette; Lafayette residents across the river must reach a different
   station. The 8-sector feeder coverage model treats river-blocked sectors
   as having zero bus coverage.

3. **Development asymmetry**: West Lafayette (university-adjacent) and
   Lafayette (commercial core) have different zoning regimes, parcel sizes,
   assessed values, and demographic profiles. A corridor crossing the river
   serves both markets but faces different development potential on each side.

Corridors that stay on one side avoid the $35M bridge cost but serve only
half the metro. The model's evolutionary search naturally explores both
bridge-crossing and single-side corridors, letting the ridership/cost
tradeoff determine which survive.

### Why do some corridors extend into low-demand areas?

The Dynamic Programming (DP) station selection algorithm maximizes total
demand across selected stations but does not require minimum demand at
endpoints. The first and last stations on a corridor path are "forced
endpoints" — the DP must include them regardless of their individual demand.

This can produce corridors that extend into nature preserves (like Celery
Bog), agricultural areas, or highway interchanges where there is no adjacent
population or employment. The corridor reaches these areas because the road
graph path between two high-demand anchors happens to pass through them, and
the DP has no mechanism to trim low-demand tails.

The model partially mitigates this through flexible terminal selection (the
last 5 stations on the path are candidates for the endpoint, not just the
final one) and a demand floor (endpoints must exceed the 25th percentile of
station demand). But corridors with awkward termini can still survive if
their interior stations generate enough total ridership.

### How are curve speed penalties computed?

The model uses ASCE 21.2-2008 lateral acceleration limits to compute speed
reductions on curved guideway:

1. **Comfort limit**: 0.07g (0.687 m/s²) — the target for standing passenger
   comfort, below the ASCE 0.10g maximum
2. **Curve speed**: V = √(a_lateral × R). At radius 100m, V = 26 km/h
   (vs 27 km/h line speed). At radius 50m, V = 21 km/h.
3. **Trapezoidal profile**: Each curve involves deceleration from line speed,
   traversal at curve speed, and re-acceleration — modeled as a trapezoidal
   speed profile
4. **Transition spirals**: Each curve entry/exit requires a jerk-limited
   clothoid transition (ASCE max jerk 0.3 m/s³). This adds 2-4 seconds per
   curve depending on tightness.
5. **Minimum radius**: 50m for revenue service (ASCE 21 restricted minimum).
   Tighter curves are physically possible but only at depot-speed (~5 km/h).

The cumulative delay from all curves is added to the travel time from stop
dwell penalties. This composite effective speed feeds into the mode choice
MNL — curvy corridors are slower and attract fewer riders.

### What is the transit rent floor and why does it exist?

The model includes an 82% floor on rents for parcels within 800m of transit
stations. This means station-area rents can decline from vacancy pressure
but never fall below 82% of their initial (pre-transit) values.

Without this floor, a pure vacancy-rent model can produce an unrealistic
"death spiral." If construction temporarily outpaces absorption, vacancy
rises and rents fall, which makes new projects infeasible, which halts
development permanently. Real transit corridors self-correct: falling rents
attract more residents, developers pull back before vacancy spirals, and the
transit amenity premium creates a demand floor. No observed TOD corridor has
experienced a permanent rent crash.

The 82% value is based on empirical evidence. Singer (2025) found TOD rents
appreciated 30% vs 20% metro-wide during 2012-2016 post-recession recovery.
Jiang et al. (2020) documented persistent 8-17% rent premiums near rail
stations across multiple metros and time periods. Even during building booms,
transit-adjacent rents stabilize because of captive demand: zero-car
households, walkability amenity, and reduced transportation costs create a
floor on willingness to pay.

The 82% floor preserves the vacancy feedback signal for moderate oversupply
(6% → 15% vacancy produces real rent pressure) while preventing the
catastrophic collapse that the unconstrained model produces.

---

## Glossary

| Term | Definition |
|------|-----------|
| **APM** | Automated People Mover — driverless, grade-separated transit on a dedicated guideway |
| **BRT** | Bus Rapid Transit — enhanced bus service with dedicated lanes, platforms, and signal priority |
| **Bus restructuring** | The process of reorganizing existing bus routes when a new transit line opens — reducing parallel service and creating feeder connections |
| **Catchment zone** | The area around a station from which riders are drawn. Walk zone: 0-800m; feeder zone: 800-7,000m |
| **Corridor** | A candidate transit route connecting two or more activity centers, defined by its station locations and alignment |
| **DCR / DSCR** | Debt Coverage (Service) Ratio — annual revenue divided by annual debt payments. 1.0x = break-even |
| **FAR** | Floor Area Ratio — total building floor area divided by lot area. FAR 2.0 = a 2-story building covering the whole lot, or a 4-story building covering half |
| **Feeder bus** | A bus route that brings riders to/from a transit station, typically running perpendicular to the main line |
| **Feeder headway** | Time between feeder buses serving a station. Lower = more frequent = more riders from the feeder zone |
| **Guideway** | The physical track or beam that an APM vehicle runs on, typically elevated above street level |
| **Headway** | Time between consecutive vehicles at a station (e.g., 5-minute headway = a train every 5 minutes) |
| **MNL** | Multinomial Logit — a statistical model that predicts which transportation mode a person will choose based on travel time, cost, and other factors |
| **Mode share** | The percentage of trips made by each transportation mode (car, bus, walk, transit, etc.) |
| **NPV** | Net Present Value — the total financial gain or loss over 25 years, discounted to today's dollars |
| **NSGA-II** | Non-dominated Sorting Genetic Algorithm II — an optimization method that finds solutions balancing multiple competing objectives |
| **P(viable)** | Probability that a corridor achieves DCR ≥ 1.0x across Monte Carlo uncertainty draws. Higher = more financially robust |
| **Pressure** | A 0-1 score measuring how strongly APM ridership justifies bus network restructuring. 0 = no change; 1 = full feeder conversion |
| **Proforma** | A financial feasibility analysis that determines whether a building project would be profitable given construction costs, rents, and zoning limits |
| **Scenario** | A set of policy assumptions (current zoning or no zoning) that the model evaluates independently to show how outcomes change |
| **Self-sufficiency ratio** | Total revenue (fares + TIF + campus payments) divided by total costs (debt + O&M + net bus cost). 1.0 = fully self-sustaining |
| **Stress test** | A named adverse scenario (enrollment shock, cost blowout, etc.) that tests whether a corridor remains viable under unfavorable conditions |
| **TIF** | Tax Increment Financing — a mechanism that captures property tax revenue from new development near transit to help pay for the transit system |
| **Uncertainty bands** | The range of outcomes (p10 to p90) from Monte Carlo simulation, showing how sensitive results are to parameter assumptions |

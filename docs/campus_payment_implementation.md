# Campus Payment Revenue Stream — Implementation Description

## Overview

The campus payment is a new revenue stream in the financial model that captures
the economic value Purdue University receives when APM ridership displaces
automobile trips to campus.  Displaced drivers free parking spaces, which
avoids garage construction or liberates surface-lot land for higher-value use.
Purdue pays an annual fee to the APM operator proportional to the parking value
displaced.

This revenue enters the financial model alongside TIF and farebox, improving
the debt coverage ratio for campus-proximate corridors.  It does not make any
corridor viable on its own — the expected magnitude is $400–700K/year at
maturity, closing 5–10% of the funding gap for the best campus corridors.

## Data already produced by the model (per corridor, per year)

The feedback loop CSV (`_build_year_row()` in `land_use_transport_model.py:2538`)
already writes these fields for every corridor at every time step:

| Field | Source | What it tells you |
|---|---|---|
| `work_commute_daily` | Awareness-adjusted LODES commute component | Non-student APM commuters |
| `campus_daily` | Awareness-adjusted student component | Student APM riders |
| `destination_daily` | Awareness-adjusted generator component | Medical/retail/event riders |
| `equity_daily` | Awareness-adjusted induced + latent | Zero-car HH and new trips |
| `pop_catchment` | Walk-zone population | Total catchment population |
| `daily_riders` | Sum of all components | Total corridor ridership |

These sum correctly: `daily_riders = work_commute_daily + local_nonwork_daily +
campus_daily + destination_daily + equity_daily`.

## What needs to be added

### Step 1: Compute auto-diversion using existing component constants

`src/financial_params.py` already defines per-component car diversion fractions
(lines 289–296):

```python
CAR_DIVERSION_COMMUTE = 0.60      # LODES commute: ~60% diverted from car
CAR_DIVERSION_STUDENT = 0.30      # Students: lower car ownership
CAR_DIVERSION_GENERATOR = 0.55    # Medical/retail/event trips
CAR_DIVERSION_INDUCED = 0.00      # New trips — no prior mode to divert from
CAR_DIVERSION_LATENT = 0.00       # Zero-car HH — had no car to divert
```

**Do not create new diversion constants.**  Use these directly, applied
component-wise.  The `equity_daily` component (induced + latent) has 0%
car diversion because those riders either had no car or are making new trips
that didn't exist before.

```python
from src.financial_params import (
    CAR_DIVERSION_COMMUTE,
    CAR_DIVERSION_STUDENT,
    CAR_DIVERSION_GENERATOR,
)

# Component-wise auto diversion (per year, from feedback loop data)
total_diverted = (
    work_commute_daily * CAR_DIVERSION_COMMUTE    # 60% of commuters
    + campus_daily * CAR_DIVERSION_STUDENT         # 30% of students
    + destination_daily * CAR_DIVERSION_GENERATOR  # 55% of generator trips
    # equity_daily: 0% diversion — induced/latent had no car
)
```

This replaces the original plan's flat `daily_riders * AUTO_DIVERSION_FRACTION`
which over-counted by applying 60% diversion to induced/latent riders (should
be 0%).

### Step 2: Compute displaced campus parking spaces

Student auto-diversion IS the campus parking diversion — students who shift
from car to APM are exactly the people who stop parking on campus.  Non-student
commuters who shift from car were parking at their workplaces, not on campus.

```python
# Students who shifted from car = students who stopped parking on campus
campus_diverted_drivers = campus_daily * CAR_DIVERSION_STUDENT

# Each diverted round-trip driver ≈ 1 parking space freed
# But shared parking means some spaces serve multiple users
RIDERS_PER_DISPLACED_SPACE = 3.0  # conservative (CU Boulder study: 2.75)
displaced_spaces = campus_diverted_drivers / RIDERS_PER_DISPLACED_SPACE
```

This replaces the original plan's `total_diverted * campus_fraction` formula,
which conflated catchment population ratio with parking location.  The
component-wise approach is both simpler and more accurate.

### Step 3: Compute annual campus payment

Two value channels, take the dominant one:

**Channel A — Avoided garage construction:**  Purdue cancels the next planned
structured parking garage because demand dropped.

```python
# Use the existing constant from financial_params.py:247
from src.financial_params import STRUCTURED_PARKING_COST_PER_SPACE  # $35,000

parking_construction_avoided = displaced_spaces * STRUCTURED_PARKING_COST_PER_SPACE
```

**Channel B — Surface lot land liberation:**  Purdue demolishes surface lots
that are no longer needed and develops the land.

```python
SURFACE_SPACES_PER_ACRE = 130     # ITE surface lot standard
CAMPUS_LAND_VALUE_PER_ACRE = 3_000_000  # Discovery Park / peripheral campus

acres_freed = displaced_spaces / SURFACE_SPACES_PER_ACRE
land_value_liberated = acres_freed * CAMPUS_LAND_VALUE_PER_ACRE
```

**Why max() and not additive:**  These are alternative uses of the SAME freed
parking capacity.  If Purdue frees 250 spaces of demand, it can either
(a) cancel a garage, or (b) demolish surface lots and develop the land.  It
cannot count the same spaces twice.  Take the dominant value:

```python
one_time_value = max(parking_construction_avoided, land_value_liberated)
```

At typical numbers (250 spaces): construction avoided = 250 × $35K = $8.75M,
land liberation = 250/130 × $3M = $5.8M.  **Construction avoidance dominates**
because garage costs exceed land value per space.

**Convert to annual payment:**  Purdue amortizes the one-time value into an
annual fee to the APM operator.

```python
PURDUE_BORROWING_RATE = 0.04  # AA-rated university bond (Moody's Aa1)
PAYMENT_TERM_YEARS = 25

annuity_factor = (
    PURDUE_BORROWING_RATE * (1 + PURDUE_BORROWING_RATE) ** PAYMENT_TERM_YEARS
) / ((1 + PURDUE_BORROWING_RATE) ** PAYMENT_TERM_YEARS - 1)

# Parking O&M avoidance is a separate annual saving (not one-time)
PARKING_OM_PER_SPACE_ANNUAL = 600.0  # $/space/year, Midwest surface lot
annual_parking_om_avoided = displaced_spaces * PARKING_OM_PER_SPACE_ANNUAL

annual_campus_payment = one_time_value * annuity_factor + annual_parking_om_avoided
```

### Step 4: Build per-year campus payment series

The payment ramps with student ridership.  Since the relationship is linear
(riders → diverted → displaced → value), proportional scaling from the year-25
value is mathematically equivalent to recomputing at each year:

```python
def compute_campus_payment_series(
    campus_daily_series: np.ndarray,
    years: int,
) -> np.ndarray:
    """Annual campus payment series (USD), scaling with student ridership."""
    if len(campus_daily_series) == 0 or campus_daily_series[-1] <= 0:
        return np.zeros(years, dtype=float)

    # Compute year-25 (maturity) payment
    campus_daily_mature = campus_daily_series[-1]
    diverted_mature = campus_daily_mature * CAR_DIVERSION_STUDENT
    displaced_mature = diverted_mature / RIDERS_PER_DISPLACED_SPACE

    construction_avoided = displaced_mature * STRUCTURED_PARKING_COST_PER_SPACE
    acres = displaced_mature / SURFACE_SPACES_PER_ACRE
    land_value = acres * CAMPUS_LAND_VALUE_PER_ACRE
    one_time = max(construction_avoided, land_value)

    annuity = (
        PURDUE_BORROWING_RATE * (1 + PURDUE_BORROWING_RATE) ** PAYMENT_TERM_YEARS
    ) / ((1 + PURDUE_BORROWING_RATE) ** PAYMENT_TERM_YEARS - 1)
    om_avoided = displaced_mature * PARKING_OM_PER_SPACE_ANNUAL
    payment_mature = one_time * annuity + om_avoided

    # Scale proportionally by each year's student ridership
    series = np.zeros(years, dtype=float)
    for yr in range(min(years, len(campus_daily_series))):
        ratio = campus_daily_series[yr] / campus_daily_mature
        series[yr] = payment_mature * ratio

    return series
```

### Step 5: Wire into `_compute_dynamic_finance_metrics()`

**File:** `scripts/apm_corridor_evaluation_integrated.py`

The function signature needs a new parameter:

```python
def _compute_dynamic_finance_metrics(
    annual_ridership_series: np.ndarray,
    *,
    years: int,
    annual_tif_series_usd: np.ndarray,
    campus_payment_series_usd: np.ndarray | None = None,  # NEW
    capital_cost_usd: float,
    # ... rest unchanged
) -> Dict[str, float]:
```

At line 428, add the campus payment to total revenue:

```python
campus_series_musd = np.zeros(years, dtype=float)
if campus_payment_series_usd is not None:
    _cp = np.asarray(campus_payment_series_usd, dtype=float)
    if _cp.size >= years:
        campus_series_musd = _cp[:years] / 1_000_000.0
    elif _cp.size > 0:
        campus_series_musd[:_cp.size] = _cp / 1_000_000.0

annual_total_revenue_musd = (
    annual_tif_series_musd + farebox_series_musd + campus_series_musd
)
```

Add to the returned dict:

```python
"campus_payment_annual_mean_musd": float(np.mean(campus_series_musd)),
"campus_payment_npv_musd": float(np.sum(
    campus_series_musd / np.power(1 + discount_rate, np.arange(1, years + 1))
)),
```

### Step 6: Build campus payment series in evaluate_corridor_with_financial_analysis()

**File:** `scripts/apm_corridor_evaluation_integrated.py`, after line 991

The feedback DataFrame already has per-year `campus_daily` for each corridor.
Extract it to build the series:

```python
# Build campus payment series from per-year student ridership
campus_payment_series = np.zeros(years, dtype=float)
if feedback_df is not None and "campus_daily" in feedback_df.columns:
    cdf = feedback_df[feedback_df["corridor_id"] == corridor_id].sort_values("year")
    campus_daily_series = annualize_daily_ridership_series(
        # Convert daily campus riders to annual
        cdf["campus_daily"].values,
        years=len(cdf),
    )
    # Actually we need daily values, not annual — compute_campus_payment_series
    # expects daily riders
    _campus_daily_arr = cdf["campus_daily"].values.astype(float)
    campus_payment_series = compute_campus_payment_series(
        _campus_daily_arr, years=years,
    )
```

Pass into `_compute_dynamic_finance_metrics()`:

```python
dynamic_finance = _compute_dynamic_finance_metrics(
    annual_ridership_series,
    years=years,
    annual_tif_series_usd=_annual_tif_series,
    campus_payment_series_usd=campus_payment_series,  # NEW
    capital_cost_usd=capital_cost,
    # ... rest unchanged
)
```

Add to the returned result dict (after line 1087):

```python
"campus_payment_annual_musd": float(dynamic_finance.get("campus_payment_annual_mean_musd", 0.0)),
"campus_payment_npv_musd": float(dynamic_finance.get("campus_payment_npv_musd", 0.0)),
"displaced_parking_spaces": float(np.max(campus_payment_series) / ...) if ...,  # at maturity
```

### Step 7: Wire into Monte Carlo uncertainty

**File:** `scripts/apm_corridor_evaluation_integrated.py`, `_simulate_uncertainty_draws_for_row()`

The Monte Carlo function operates on summary row data, not per-year series.
Add `campus_payment_annual` to the corridor results DataFrame (Step 6 output),
then scale it by `ridership_mult` in the Monte Carlo draws:

```python
# Line ~628: add to base values
base_campus = max(float(row.get("campus_payment_annual_musd", 0.0)), 0.0) * 1_000_000.0

# Line ~648: add to revenue
annual_campus = base_campus * ridership_mult  # scales with ridership
annual_revenue = annual_tif_usd + annual_farebox_usd + annual_campus
```

This is a simplification — the campus payment should really scale with the
student share of ridership, not total ridership.  But since the Monte Carlo
varies ridership as a single multiplier (not per-component), and student share
is roughly constant across draws, this is an acceptable approximation.

### Step 8: Wire into financial_corridor_ranking.py

**File:** `scripts/financial_corridor_ranking.py`

The `FinancialCorridorRanker` reads from the evaluation output DataFrame.
The new columns (`campus_payment_annual_musd`, `campus_payment_npv_musd`) will
flow through automatically because the ranker reads whatever columns exist.

In `calculate_debt_coverage_ratio()` and cost_per_rider calculations, campus
payment is already included in the revenue totals from Step 5.  No structural
change needed — the ranker inherits the improved DSCR/NPV from the upstream
finance computation.

Add display columns in the ranking output:

```python
# In the summary table generation:
"campus_payment_musd": row.get("campus_payment_annual_musd", 0.0),
"displaced_spaces": row.get("displaced_parking_spaces", 0),
```

## Constants to add to `src/financial_params.py`

```python
# ===================================================================
# Campus Parking Payment (Purdue connection)
# ===================================================================

# Parking displacement (Step 2)
RIDERS_PER_DISPLACED_SPACE = 3.0          # CU Boulder study: 2.75, conservative
# Note: STRUCTURED_PARKING_COST_PER_SPACE = 35_000 already exists (line 247)

# Land liberation (Step 3, Channel B)
SURFACE_SPACES_PER_ACRE = 130             # ITE surface lot standard
CAMPUS_LAND_VALUE_PER_ACRE = 3_000_000    # Discovery Park / peripheral campus

# Annual O&M avoidance
PARKING_OM_PER_SPACE_ANNUAL = 600.0       # $/space/year, Midwest surface lot

# Purdue payment terms
PURDUE_BORROWING_RATE = 0.04              # Moody's Aa1 university bond rate
CAMPUS_PAYMENT_TERM_YEARS = 25            # Matches TIF/debt term
```

**Do NOT add** `AUTO_DIVERSION_FRACTION` or `STUDENT_AUTO_DIVERSION_FRACTION` —
these duplicate the existing `CAR_DIVERSION_*` constants.

## What does NOT change

| Component | Why unchanged |
|---|---|
| `src/finance.py` | Generic NPV/IRR functions — campus payment enters via revenue series, no API change needed |
| `src/land_use_transport_model.py` | Ridership model already produces all required component fields |
| `src/mode_choice.py` | MNL already captures the car→APM shift; no modification needed |
| Ridership components | campus_daily, work_commute_daily, etc. remain as-is |
| BCA calculations | See double-counting note below |

## Double-counting risk with BCA

The BCA (benefit-cost analysis) in `financial_params.py` already includes
parking cost avoidance as a societal benefit via `STRUCTURED_PARKING_COST_PER_SPACE`.
The campus payment is a **transfer** (Purdue → APM operator), not an additional
net benefit.  In the financial model, count it as revenue.  In the BCA, count
the parking savings as a benefit.  Do not sum both.

If a BCA module is added later, it should exclude campus payment from revenue
and instead count parking displacement directly as a benefit — or include the
campus payment as revenue and exclude parking from benefits.  Either way, the
parking value appears exactly once.

## Expected magnitudes

For a corridor with `campus_daily` = 2,000 at year 25:

| Step | Value |
|---|---|
| Student auto diversion | 2,000 × 0.30 = 600 trips/day |
| Displaced spaces | 600 / 3.0 = 200 spaces |
| Construction avoidance | 200 × $35,000 = $7.0M one-time |
| Land liberation | 200/130 × $3M = $4.6M one-time |
| Dominant channel | Construction avoidance ($7.0M) |
| Annuity payment | $7.0M × 0.0640 = $448K/year |
| O&M avoidance | 200 × $600 = $120K/year |
| **Total annual payment** | **$568K/year at maturity** |
| Year-5 payment (awareness ~30%) | ~$170K/year |

For comparison, annual debt service on a typical corridor is $30–50M/year.
Campus payment closes ~1–2% of the gap.  It matters more for DSCR at early
years when TIF is still ramping but student ridership matures faster
(student awareness ramp is independent of development).

## Implementation sequence

```
Step 1: Add constants to financial_params.py
Step 2: Add compute_campus_payment_series() function (new helper in
        apm_corridor_evaluation_integrated.py or a shared module)
Step 3: Wire into _compute_dynamic_finance_metrics() (signature + revenue line)
Step 4: Wire into evaluate_corridor_with_financial_analysis() (build series from
        feedback DataFrame, pass to finance, add to output dict)
Step 5: Wire into Monte Carlo (_simulate_uncertainty_draws_for_row)
Step 6: Add display columns to financial_corridor_ranking.py
```

Steps 1–4 are the core change.  Step 5 is mechanical.  Step 6 is cosmetic.

## Verification

1. `pytest tests/test_dynamic_finance_cashflows.py` — existing tests pass
   (campus payment defaults to None/zero, so backward-compatible)
2. Add `tests/test_campus_payment.py`:
   - `test_campus_payment_zero_when_no_students` — corridor with 0 campus_daily
     produces 0 campus payment
   - `test_campus_payment_magnitude` — 2,000 campus_daily → ~$568K/year
   - `test_campus_payment_ramps_with_ridership` — year-5 payment < year-25
   - `test_campus_payment_in_dscr` — DSCR improves for campus corridor,
     unchanged for non-campus corridor
   - `test_construction_dominates_land` — at $35K/space and 130 spaces/acre,
     construction avoidance > land liberation
3. Run `python scripts/run_feedback_loop.py --scenario current_zoning` for 5
   corridors, verify campus payment appears in output and is non-zero only for
   campus-proximate corridors

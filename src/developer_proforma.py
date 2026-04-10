"""
Lafayette market data constants for the developer proforma.

ZONING_MATRIX, MARKET_CONFIG, and NO_ZONING_FAR_CAP are imported by
realistic_developer_proforma.py (star-import) and consumed by
land_use_transport_model.py for SqFtProForma configuration.

The DeveloperProForma class that was here has been removed — all feasibility
analysis now runs through urbansim.developer.sqftproforma.SqFtProForma.
Archived copy: archive/src/developer_proforma.py
"""

# ============================================================================
# MARKET DATA CONFIGURATION
# ============================================================================

MARKET_CONFIG = {
    # Market rents ($/sqft/year) by zone type and use
    "zone_characteristics": {
        # Residential zones (R1, R2, R3, etc.)
        #
        # Rents calibrated to 2025 Tippecanoe County Rental Report and
        # RentCafe/Apartments.com market data:
        #   Lafayette average: $1,196/mo ($14.35/sqft/yr for avg unit)
        #   West Lafayette average: $1,915/mo ($22.98/sqft/yr for avg unit)
        #   New luxury near Purdue: $2,200-2,600/mo ($27-32/sqft/yr)
        #
        # Note: the model applies a distance-based rent premium (up to +40%)
        # for near-station parcels, so base rents represent county-wide
        # averages for each zone type, not peak near-campus values.
        #
        # Construction costs calibrated to RSMeans 2024 Midwest (Lafayette
        # location factor ~0.89) and local project permit valuations.
        #
        # operating_cost_pct: EXCLUDES property taxes.  Property tax
        # revenue is modeled separately in the TIF module (src/finance.py)
        # which computes incremental tax revenue from assessed value uplift.
        # Including taxes here would double-count.  IREM benchmarks for
        # Midwest multifamily (excl. taxes): 0.28-0.33.  With taxes the
        # IREM benchmark is 0.41 — DO NOT use the with-taxes number here.
        "residential_low": {
            "zones": ["R1", "R1A", "R1B", "R1T", "R1U", "R2", "R2U"],
            "rent_psf_year": 14.00,  # Lafayette suburban/single-family avg
            "vacancy_rate": 0.05,    # ACS 2022 Lafayette MSA; suburban stable
            "operating_cost_pct": 0.30,
            "cap_rate": 0.060,       # Tertiary market; CBRE H2 2024 secondary ~5.5%
            "construction_cost_psf": 165,  # Type V wood frame, RSMeans Midwest
            "land_cost_multiplier": 0.8,
        },
        "residential_medium": {
            "zones": ["R3", "R3W", "R3U", "R4W"],
            "rent_psf_year": 20.00,  # WL multifamily avg ~$22; county avg ~$18
            "vacancy_rate": 0.04,    # WL near-campus vacancy 1.5-2.3% (2025)
            "operating_cost_pct": 0.32,
            "cap_rate": 0.050,
            "construction_cost_psf": 190,  # 3-4 story wood frame, current costs
            "land_cost_multiplier": 1.0,
        },
        # Commercial zones
        "commercial_general": {
            "zones": ["GB", "HB", "NB", "OR", "MR", "MRU"],
            "rent_psf_year": 20.00,  # Office/retail NNN rents, Tippecanoe Co
            "vacancy_rate": 0.08,    # CBRE Midwest secondary Q4 2023
            "operating_cost_pct": 0.35,
            "cap_rate": 0.058,
            "construction_cost_psf": 195,  # Single-story commercial, RSMeans
            "land_cost_multiplier": 1.2,
        },
        # Special overlay districts (PD = Planned Development)
        "commercial_mixed_use": {
            "zones": ["PDRS", "PDCC", "PDMX", "PDNR", "CB", "CBW", "NBU"],
            "rent_psf_year": 26.00,  # New mixed-use near Purdue premium
            "vacancy_rate": 0.06,    # WL urban mixed-use vacancy ~2-6% (2025)
            "operating_cost_pct": 0.33,
            "cap_rate": 0.050,       # Tertiary market; CBRE H2 2024 secondary ~5.0-5.5%
            "construction_cost_psf": 250,  # 5-over-1 podium, comparable to Verve
            "land_cost_multiplier": 1.5,
        },
        # Industrial zones
        "industrial": {
            "zones": ["I1", "I2", "I3"],
            "rent_psf_year": 9.00,   # Industrial NNN, slight increase
            "vacancy_rate": 0.10,
            "operating_cost_pct": 0.25,
            "cap_rate": 0.065,
            "construction_cost_psf": 130,  # Tilt-up/pre-engineered, RSMeans
            "land_cost_multiplier": 0.6,
        },
        # Non-building uses (genuinely undevelopable)
        "conservation": {
            "zones": ["FP", "RE"],
            "rent_psf_year": 0.0,
            "vacancy_rate": 0.0,
            "operating_cost_pct": 0.0,
            "cap_rate": 0.0,
            "construction_cost_psf": 0,
            "land_cost_multiplier": 0,
            "developable": False,
        },
    },
    
    # Base land costs by location ($/sqft of land)
    # Calibrated from Tippecanoe County Assessor CurLandAV / Shape_Area
    # (2024 assessment year).  The main model uses per-parcel CurLandAV
    # data directly; these medians are only a fallback.
    #   Wabash Twp (West Lafayette) median: $4.45/sqft, p90: $14.45
    #   Fairfield Twp (Lafayette city) median: $2.90/sqft, p90: $16.58
    #   Rural townships: $0.10-$0.50/sqft median
    "land_cost_base_psf": {
        "downtown_core": 90,     # p95+ urban parcels
        "urban_residential": 15,
        "suburban": 8,
        "rural": 2,
    },
    
    # APM proximity bonus (% of property value increase at each distance)
    "apm_proximity_bonus": {
        0: 0.12,      # 12% bonus within 0-200m
        200: 0.08,    # 8% bonus 200-400m
        400: 0.04,    # 4% bonus 400-600m
        600: 0.01,    # 1% bonus 600-800m
        800: 0.00,    # No bonus beyond 800m
    },
    
    # Developer hurdle rates (required IRR) by use type
    "developer_irr_targets": {
        "residential": 0.12,      # 12% minimum IRR
        "commercial": 0.14,       # 14% minimum IRR
        "industrial": 0.13,       # 13% minimum IRR
        "mixed_use": 0.15,        # 15% minimum IRR (highest risk)
    },
    
    # Financing parameters
    "construction_financing": {
        "interest_rate": 0.06,     # 6% construction loan
        "duration_years": 2.0,     # Typical 2-year construction
        "closing_costs_pct": 0.02, # 2% of loan amount
    },
    
    # Project delivery assumptions
    "project_delivery": {
        "permit_approval_rate": 0.95,   # 95% of feasible projects get permitted
        "construction_completion_rate": 0.85,  # 85% of permitted complete
        "stabilization_success_rate": 0.90,    # 90% meet rent/occupancy targets
        "overall_completion_rate": 0.95 * 0.85 * 0.90,  # ~73%
    },
    
    # Market absorption rates (% of zoning capacity per year)
    # Much lower than zoning allows.  Calibrated to Census C-40 permits
    # for Tippecanoe County (2019-2023 avg 1,241 units/yr) relative to
    # total zoned capacity.  Student housing absorption is higher than
    # conventional due to Purdue enrollment growth (+20% since 2020).
    "market_absorption_rate": {
        "residential": 0.015,    # 1.5%/year
        "commercial": 0.012,     # 1.2%/year
        "industrial": 0.010,     # 1.0%/year
        "mixed_use": 0.018,      # 1.8%/year (highest demand)
    },
}

# ============================================================================
# HEIGHT-DEPENDENT CONSTRUCTION COST PREMIUM
# ============================================================================
#
# Construction costs escalate with building height due to structural system
# requirements mandated by the International Building Code (IBC).
#
# Empirical sources:
#   RSMeans 2024 Square Foot Costs, Midwest region (Gordian):
#     Apartments 1-3 story (Type V):   $145-185/sqft (Lafayette factor 0.89)
#     Apartments 4-7 story (Type III): $200-250/sqft  (+25% over baseline)
#     Apartments 8-12 story (Type I):  $275-350/sqft  (+65% over baseline)
#     High-rise 13+ story (Type I):    $350-450/sqft  (+110% over baseline)
#
#   Greater Lafayette tall-building projects:
#     Rise on Chauncey (16-story all-concrete, 2019): 458K sf, 283 units.
#       Brinkmann Constructors. Tallest in West Lafayette.
#     Hub Chauncey (13-story, under construction 2027): 681 units.
#     The Standard (13-story, under construction 2027): 253 units.
#     The Approach (11-story, approved 2026): 268 units.
#     Verve West Lafayette (7-story 5-over-1, 2023): 235 units, ~$250/sqft.
#     District at Tapawingo (5-6 story, approved): $350M / ~800K sf = ~$440/sqft
#       total development cost (incl. land, soft costs, margin).
#
#   Student housing dominates tall construction in the market.  All 8+ story
#   buildings in Tippecanoe County are student-oriented, supported by
#   Purdue enrollment growth (58K students, +20% since 2020) and per-bed
#   pricing ($1,200-1,800/bed/month) that exceeds conventional rent/sqft.
#
#   NAHB Construction Cost Survey 2023:
#     Wood-frame to concrete/steel transition: +40-60% cost increase
#     High-rise (12+) vs low-rise: approximately 2x cost
#
# IBC construction type thresholds (height limits):
#   Type V-A  (wood frame):    4 stories / 70 ft max
#   Type III-A (5-over-1):     7 stories / 85 ft max (IBC 510.2 podium)
#   Type I-A  (concrete/steel): unlimited height
#   Type I-A  high-rise (75+ ft): additional fire command, standpipe,
#     emergency power requirements (IBC Chapter 4 high-rise provisions)
#
# Note: a 15-story project on Chauncey Hill was denied by West Lafayette
# City Council (height/traffic concerns), indicating political resistance
# to the tallest buildings even when financially feasible.
#
# (max_stories_inclusive, fractional_cost_premium)
HEIGHT_COST_TIERS = [
    (4,   0.00),   # Type V-A wood frame: baseline
    (7,   0.25),   # Type III-A / 5-over-1 podium: +25%
    (12,  0.65),   # Type I-A concrete/steel mid-rise: +65%
    (999, 1.10),   # Type I-A high-rise: +110% (Rise on Chauncey scale)
]

# Fraction of lot area that is buildable floor plate (after setbacks,
# circulation cores, structured parking footprint).  Standard for
# multi-story development in Midwest secondary markets.
FLOOR_PLATE_EFFICIENCY = 0.80


# ============================================================================
# ZONING FAR MATRIX
# ============================================================================
#
# Derived FARs — Tippecanoe County UZO does NOT use FAR directly.
# These are back-calculated from the UZO dimensional tables as:
#
#   effective FAR = lot_coverage × (max_height_ft / story_height_ft)
#
# where story_height_ft = 10 (residential) or 12 (commercial/industrial).
#
# The UZO specifies lot coverage % + max building height + setbacks per zone.
# For zones without explicit height limits (e.g., A, AA), we use 35 ft
# (the common default in Tippecanoe County for agricultural zones).
#
# PD zones (Planned Development) are individually negotiated; FARs here
# are estimates from approved PD projects near Purdue campus.
#
# FAR derived from UZO dimensional tables:
#   effective FAR = lot_coverage × (max_height_ft / story_height_ft)
# The no_zoning scenario uses NO_ZONING_FAR_CAP (8.0) for all zones.

ZONING_MATRIX = {
    # Zone Code: (FAR, land_use_type, max_dua)
    # max_dua = maximum dwelling units per acre (None = FAR is the binding constraint).
    # R1/R2 zones have per-lot unit caps that translate to DUA limits.
    # R3+ zones allow multifamily by-right; DUA is effectively unlimited
    # (constrained only by FAR and parking requirements).

    # --- Residential zones ---
    # R1/R1A/R1B: 30% lot coverage, 35ft max → FAR 1.0
    # UZO restricts R1 to ONE dwelling per lot.  Typical R1 lot = 10,000 sqft
    # = 0.23 acres → 1 unit / 0.23 acres = 4.35 DUA.
    "R1":  (1.0, "residential", 4.35),
    "R1A": (1.0, "residential", 4.35),
    "R1B": (1.0, "residential", 4.35),
    # R1T: legacy zone — treat as R1-equivalent
    "R1T": (1.0, "residential", 4.35),
    # R1U: 35% lot coverage, 35ft → FAR 1.2
    # R1U (urban) allows smaller lots (4,000 sqft) but still 1 unit per lot.
    # 1 unit / 0.092 acres = 10.9 DUA.
    "R1U": (1.2, "residential", 10.9),
    # R2: 35% lot coverage, 35ft → FAR 1.2
    # UZO allows TWO dwellings per lot (duplex).  Typical R2 lot = 8,000 sqft.
    # 2 units / 0.184 acres = 10.9 DUA.
    "R2":  (1.2, "residential", 10.9),
    "R2U": (1.2, "residential", 10.9),
    # R3: 40% lot coverage, 35ft → 40% × (35/10) = 1.4
    # Multifamily by-right — no per-lot unit cap.
    "R3":  (1.4, "residential", None),
    # R3W: 40% lot coverage, 40ft max → 40% × (40/10) = 1.6
    # (Jan 2026 APC amendment aligned R3W height to 40ft roof peak;
    #  prior 14ft measurement was to highest finished floor, not roof.)
    "R3W": (1.6, "residential", None),
    # R3U: 60% lot coverage, 55ft → 60% × (55/10) = 3.3
    "R3U": (3.3, "residential", None),
    # R4W: 50% lot coverage, 40ft max → 50% × (40/10) = 2.0
    # (Same Jan 2026 amendment as R3W.)
    "R4W": (2.0, "residential", None),

    # --- Agricultural / open zones ---
    # UZO intent: reserved for agricultural use, no urbanization planned.
    # IC 6-1.1-4-13: assessed at use value (~$2,050/acre), not market value.
    # IC 6-1.1-20.6: receives 2% circuit breaker cap (same as residential).
    # Excluded from developable mask — no urban development permitted.
    # A: 20% lot coverage, 35ft → 20% × (35/12) = 0.6
    "A":  (0.6, "agricultural", None),
    # AA: 10% lot coverage, 35ft → 10% × (35/12) = 0.3
    "AA": (0.3, "agricultural", None),
    # AW: 10% lot coverage, 35ft → 0.3
    "AW": (0.3, "agricultural", None),

    # --- Office/Research ---
    # OR: 40% lot coverage, 55ft → 40% × (55/12) = 1.8
    "OR": (1.8, "commercial", None),

    # --- PD (Planned Development) zones ---
    # Negotiated case-by-case; estimates from approved projects near campus.
    "PDRS": (3.0, "residential", None),  # Planned Development Residential Special
    "PDCC": (4.0, "residential", None),  # Planned Development Condominium Conversion
    "PDMX": (3.5, "mixed_use", None),    # Planned Development Mixed Use
    "PDNR": (2.0, "commercial", None),   # Planned Development Non-Residential

    # --- Industrial zones ---
    # I1: 40% lot coverage, 35ft → 40% × (35/12) = 1.2
    "I1": (1.2, "industrial", None),
    # I2: 50% lot coverage, 45ft → 50% × (45/12) = 1.9
    "I2": (1.9, "industrial", None),
    # I3: 60% lot coverage, 55ft → 60% × (55/12) = 2.75
    "I3": (2.75, "industrial", None),

    # --- Central Business districts ---
    # CB: 100% lot coverage, 100ft → 100% × (100/12) = 8.3
    "CB":  (8.3, "mixed_use", None),
    # CBW: 100% lot coverage, 100ft → 8.3
    "CBW": (8.3, "mixed_use", None),

    # --- Business districts ---
    # GB: 80% lot coverage, 55ft → 80% × (55/12) = 3.7
    "GB":  (3.7, "commercial", None),
    # HB: 60% lot coverage, 45ft → 60% × (45/12) = 2.25
    "HB":  (2.25, "commercial", None),
    # NB: 50% lot coverage, 35ft → 50% × (35/12) = 1.5
    "NB":  (1.5, "commercial", None),
    # NBU: 70% lot coverage, 55ft → 70% × (55/12) = 3.2
    "NBU": (3.2, "mixed_use", None),
    # MR: 50% lot coverage, 45ft → 50% × (45/12) = 1.9
    "MR":  (1.9, "commercial", None),
    # MRU: 60% lot coverage, 55ft → 60% × (55/12) = 2.75
    "MRU": (2.75, "mixed_use", None),

    # --- Genuinely undevelopable ---
    "FP": (0.0, "undevelopable", None),
    "RE": (0.0, "undevelopable", None),
    "SHADELAND": (0.0, "undevelopable", None),
    "OTTERBEIN": (0.0, "undevelopable", None),
}

# Uncapped FAR used for the "no_zoning" scenario.
# FAR 8.0 produces ~10-story buildings (at 0.80 parcel coverage), which is the
# upper bound of what a developer would finance in a 230K metro.  The proforma
# lookup table maxes at FAR 11.0 (sqftproforma.py), so higher values are
# extrapolated.  Even Houston — the canonical "no zoning" city — caps effective
# FAR at 3-8 through fire code, setbacks, and parking requirements.
NO_ZONING_FAR_CAP = 8.0

# DeveloperProForma class removed — all feasibility analysis now runs through
# urbansim.developer.sqftproforma.SqFtProForma in land_use_transport_model.py.
# Archived copy: archive/src/developer_proforma.py


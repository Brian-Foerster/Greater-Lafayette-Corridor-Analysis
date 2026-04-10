"""Shared spatial constants for the Lafayette APM corridor model."""

# Project CRS — Indiana State Plane East (NAD83, US survey feet).
# Native CRS for Tippecanoe County GIS data.
# IMPORTANT: EPSG:2965 coordinates are in US survey feet, NOT meters.
# Always multiply projected distances by US_SURVEY_FT_TO_M before using
# with meter-calibrated constants (DECAY_BETA, catchment radii, etc.).
PROJECT_CRS = "EPSG:2965"

# US survey foot → meter conversion (exact definition)
US_SURVEY_FT_TO_M = 0.3048006096012192

# Catchment radii (meters — apply conversion when comparing to EPSG:2965 distances)
WALK_CATCHMENT_M = 800.0       # TCRP 165: 800-1000m for urban rail/metro
FEEDER_CATCHMENT_M = 7000.0    # Outer bound of feeder bus/DRT service area
# Extended from 5000m to 7000m per TCRP Report 165 guidance on demand-
# responsive transit (DRT) catchments (7-10km in low-density areas).
# With FEEDER_DECAY_BETA=0.0005, parcels at 7km from station get
# exp(-0.0005 × 5800) ≈ 5.6% weight — population signal is heavily
# attenuated, capturing a small tail of zero-car households without
# materially changing results for the majority of catchment population.

"""Household Relocation Model (Tier 2)
======================================

Generates residential demand endogenously via a multinomial logit (MNL)
location choice model.  Each year, a fraction of metro households become
"movers" and choose among candidate parcels based on:
  - Housing cost relative to income
  - Average commute time
  - Transit (APM) accessibility

Students are excluded — handled by formula in DeveloperSegment (Option C).

References:
  - Ben-Akiva & Lerman (1985): Discrete Choice Analysis
  - McFadden (1978): Modelling the Choice of Residential Location
  - Train (2009): Discrete Choice Methods with Simulation, Ch. 6
  - ACS B07003: Geographic Mobility (mover rates)
  - LODES SE01/SE02/SE03: Income segmentation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


# Shared constant — must match land_use_transport_model.py:102
AVG_HOUSEHOLD_SIZE = 2.56
METRO_POPULATION = 230_000  # Census 2024 est. (matches MetroGrowthParams)

# Structural equilibrium vacancy rate.  ACS 2022 observed rate for Lafayette
# MSA is ~4%, but that reflects a tight market.  The 6% target accounts for
# frictional vacancy (turnover, renovation, seasonal student cycles) and is
# the equilibrium the rent-adjustment model (Stage 5) converges toward.
# Must match TARGET_VAC_RES in tier2_stage5_vacancy_rents.md.
TARGET_VAC_RES = 0.06


@dataclass
class RelocationConfig:
    """Configuration for household relocation MNL."""

    # Mover pool — derived from established model constants
    annual_mover_rate: float = 0.05       # ACS B07003 Tippecanoe County
    metro_households: float = float(int(METRO_POPULATION / AVG_HOUSEHOLD_SIZE))  # ~89,844

    # Student households: off-campus Purdue students who form separate HH.
    # 49K enrolled × 0.30 off-campus share = 14,700 off-campus student HH.
    # NOTE: This is NOT the same as CAMPUS_TOTAL (~14,560) in the ridership
    # model, which measures campus-affiliated *population* within the 1200m
    # walk catchment (students + faculty/staff × presence factors).  The
    # near-identical numbers are coincidental.
    student_households: float = 14_700

    # Choice set sampling — stratified with importance weights (McFadden 1978).
    # Larger region/random strata reduce corridor overrepresentation bias;
    # importance weights on exp(V) correct for remaining sampling distortion.
    # Train (2009) recommends ≥100 alternatives for stable MNL estimates.
    near_corridor_draws: int = 15         # within 800m of any station
    in_region_draws: int = 50             # anywhere in MSA (developable)
    random_draws: int = 50                # uniform random (outside option)

    # MNL coefficients (no beta_size — only two unit_sqft values exist in
    # the current parcel data [900 vs 500], making it a binary campus
    # indicator that's collinear with beta_transit.  Will revisit when
    # building-level sqft data is available.)
    beta_price: float = -2.0             # ln(rent/income) sensitivity
    beta_commute: float = -0.05          # per minute
    beta_transit: float = 0.20           # APM accessibility premium (reduced from 0.3
                                         # per reviewer: +0.3 asserted very strong
                                         # residential sorting, risk of self-reinforcing
                                         # in feedback loop; +0.20 more conservative)

    # Income segmentation (LODES)
    income_segments: Dict[str, dict] = field(default_factory=lambda: {
        "SE01": {"share": 0.29, "median_annual_income": 12_000, "beta_price_mult": 1.5},
        "SE02": {"share": 0.42, "median_annual_income": 30_000, "beta_price_mult": 1.0},
        "SE03": {"share": 0.29, "median_annual_income": 55_000, "beta_price_mult": 0.7},
    })


class HouseholdRelocationModel:
    """MNL-based household location choice model."""

    def __init__(self, config: Optional[RelocationConfig] = None):
        self.config = config or RelocationConfig()
        # Validate income segment shares sum to ~1.0
        total_share = sum(s["share"] for s in self.config.income_segments.values())
        if abs(total_share - 1.0) > 0.01:
            raise ValueError(
                f"Income segment shares must sum to 1.0, got {total_share:.3f}"
            )

    @property
    def choice_set_size(self) -> int:
        c = self.config
        return c.near_corridor_draws + c.in_region_draws + c.random_draws

    def annual_movers(self) -> int:
        """Non-student households that relocate this year."""
        non_student = max(self.config.metro_households - self.config.student_households, 0)
        return int(non_student * self.config.annual_mover_rate)

    def sample_choice_set(
        self,
        near_800_idx: np.ndarray,
        developable_mask: np.ndarray,
        n_parcels: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample stratified choice set with importance sampling weights.

        Returns (indices, weights) where:
          indices — unique parcel position indices
          weights — expansion factor per alternative (stratum_pop / stratum_sample)

        The weights correct for overrepresentation of near-corridor parcels
        in the choice set (McFadden 1978, Train 2009 Ch. 6).
        """
        c = self.config

        # Near-corridor: developable parcels within 800m
        near_dev = near_800_idx[developable_mask[near_800_idx]] if len(near_800_idx) > 0 else np.array([], dtype=int)
        n_near = min(c.near_corridor_draws, len(near_dev))
        near = rng.choice(near_dev, size=n_near, replace=False) if n_near > 0 else np.array([], dtype=int)

        # In-region: any developable parcel
        all_dev = np.flatnonzero(developable_mask)
        n_region = min(c.in_region_draws, len(all_dev))
        region = rng.choice(all_dev, size=n_region, replace=False) if n_region > 0 else np.array([], dtype=int)

        # Random: any parcel (including non-developable, for realistic outside option)
        n_rand = min(c.random_draws, n_parcels)
        rand = rng.choice(n_parcels, size=n_rand, replace=False)

        # Build weight map: expansion factor = stratum_population / stratum_sample.
        # Lower-priority strata are set first so higher-priority overwrites.
        weight_map: Dict[int, float] = {}

        # Random stratum (lowest priority)
        w_rand = n_parcels / max(n_rand, 1)
        for idx in rand:
            weight_map[int(idx)] = w_rand

        # Region stratum
        w_region = len(all_dev) / max(n_region, 1)
        for idx in region:
            weight_map[int(idx)] = w_region

        # Near stratum (highest priority)
        w_near = len(near_dev) / max(n_near, 1)
        for idx in near:
            weight_map[int(idx)] = w_near

        unique_idx = np.unique(np.concatenate([near, region, rand]))
        weights = np.array([weight_map[int(idx)] for idx in unique_idx], dtype=np.float64)

        return unique_idx, weights

    def location_probabilities(
        self,
        rents: np.ndarray,
        commute_times: np.ndarray,
        accessibility: np.ndarray,
        income_segment: str,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """MNL probability for each parcel in choice set.

        Parameters
        ----------
        rents : yearly rent per sqft for each candidate parcel
        commute_times : average commute minutes from each parcel
        accessibility : transit accessibility score
        income_segment : "SE01", "SE02", or "SE03"
        weights : importance sampling expansion factors per alternative.
            If provided, exp(V) is multiplied by weights before normalizing
            to correct for stratified sampling bias (McFadden 1978).

        Returns
        -------
        probabilities : array of same length, sums to 1.0
        """
        seg = self.config.income_segments[income_segment]
        c = self.config

        # Clamp inputs to avoid log(0) or division by zero
        safe_rents = np.maximum(rents, 1.0)
        safe_income = max(seg["median_annual_income"], 1000)
        safe_access = np.maximum(accessibility, 0.0)

        # Annual rent for average unit / annual income
        # Use a fixed 900 sqft reference unit for price ratio (actual unit
        # size variation is not modeled — see RelocationConfig comment).
        REF_UNIT_SQFT = 900.0
        annual_unit_rent = safe_rents * REF_UNIT_SQFT
        price_ratio = annual_unit_rent / safe_income

        V = (c.beta_price * seg["beta_price_mult"] * np.log(price_ratio)
             + c.beta_commute * commute_times
             + c.beta_transit * np.log1p(safe_access))

        # Numerical stability: subtract max then clip.  -50 preserves
        # exp(-50) ≈ 2e-22 which is still representable in float64 and avoids
        # collapsing low-utility alternatives to exactly zero.
        V_shifted = V - V.max()
        exp_V = np.exp(np.clip(V_shifted, -50, 0))

        # Apply importance sampling weights
        if weights is not None:
            exp_V = exp_V * weights

        total = exp_V.sum()
        if total <= 0:
            # All utilities collapsed — uniform fallback (no information to
            # distinguish alternatives; weight-proportional would bias toward
            # oversampled strata).
            return np.ones(len(V)) / len(V)
        return exp_V / total

    def allocate_movers_to_corridor(
        self,
        near_800_idx: np.ndarray,
        developable_mask: np.ndarray,
        n_parcels: int,
        rents: np.ndarray,
        commute_times: np.ndarray,
        accessibility: np.ndarray,
        year: int,
        corridor_id: str = "",
    ) -> float:
        """Run relocation MNL, return expected households choosing corridor parcels.

        Uses expected-value calculation (not Monte Carlo) for determinism.
        Importance sampling weights correct for stratified choice set bias.

        Parameters
        ----------
        near_800_idx : parcel indices within 800m of corridor stations
        developable_mask : boolean mask of developable parcels
        n_parcels : total number of parcels
        rents, commute_times, accessibility : full parcel arrays (length n_parcels)
        year : simulation year (for RNG seed)
        corridor_id : corridor identifier (hashed into RNG seed so different
            corridors get independent random draws within the same year)

        Returns
        -------
        corridor_households : expected number of movers choosing near-corridor parcels

        Note for Stage 6 integration
        ----------------------------
        AbsorptionParams.confidence_factor() should be applied as a supply-side
        adjustment (cost multiplier or profit threshold) in _run_proforma_developer(),
        NOT as a demand scalar on corridor_households.  The MNL already models
        demand; confidence should only affect developer willingness to invest.

        Note on commute times
        ---------------------
        commute_times is currently static (from base-year ACS averages).
        For dynamic congestion feedback, pass year-specific commute arrays
        that reflect capacity changes from APM ridership diversion.
        """
        n_movers = self.annual_movers()
        if n_movers <= 0 or len(near_800_idx) == 0:
            return 0.0

        # Corridor-specific seed prevents identical draws across corridors
        seed = 42 + year + (hash(corridor_id) % 2**31)
        rng = np.random.default_rng(seed=seed)
        choice_set, weights = self.sample_choice_set(near_800_idx, developable_mask, n_parcels, rng)

        if len(choice_set) == 0:
            return 0.0

        # Which choice set members are near the corridor?
        # near_800_idx includes ALL parcels within 800m (demand-side catchment),
        # not just developable ones.  sample_choice_set internally filters to
        # developable for the near-corridor stratum (supply-side sampling), but
        # demand attribution here correctly counts any near-corridor parcel.
        near_set = set(near_800_idx.tolist())
        near_mask = np.array([idx in near_set for idx in choice_set])

        corridor_hh = 0.0
        for seg_name, seg_info in self.config.income_segments.items():
            seg_movers = n_movers * seg_info["share"]

            probs = self.location_probabilities(
                rents=rents[choice_set],
                commute_times=commute_times[choice_set],
                accessibility=accessibility[choice_set],
                income_segment=seg_name,
                weights=weights,
            )

            corridor_prob = probs[near_mask].sum()
            corridor_hh += seg_movers * corridor_prob

        return corridor_hh

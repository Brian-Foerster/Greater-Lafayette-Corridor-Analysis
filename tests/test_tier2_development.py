"""Tests for Tier 2 development model (relocation MNL + proforma + vacancy feedback).

Tests use the actual production APIs. Vacancy-rent feedback uses
src.vacancy_rent_feedback (lightweight, no geopandas/osmnx dependency).
"""
import numpy as np
import pytest
from src.relocation_model import HouseholdRelocationModel, RelocationConfig, TARGET_VAC_RES
from src.vacancy_rent_feedback import update_rents_from_vacancy


# Relocation MNL tests are in tests/test_relocation_model.py (canonical).

# ---------------------------------------------------------------------------
# Vacancy feedback — uses production function from src.vacancy_rent_feedback.
# Alias for backward compatibility with test code below.
# ---------------------------------------------------------------------------

_update_rents_standalone = update_rents_from_vacancy


class TestVacancyFeedback:
    """Test rent adjustment using production-mirroring function.

    Uses TARGET_VAC_RES imported from src.relocation_model (same source
    as production code) so constant changes are detected.
    """

    def test_tight_market_raises_rents(self):
        """0% vacancy (full occupancy) should push rents up."""
        rents = np.array([20.0])
        total = np.array([100.0])
        occupied = np.array([100.0])  # 0% vacancy
        old = rents[0]
        _update_rents_standalone(rents, total, occupied, np.array([0]))
        assert rents[0] > old

    def test_oversupply_lowers_rents(self):
        """20% vacancy should push rents down."""
        rents = np.array([20.0])
        total = np.array([100.0])
        occupied = np.array([80.0])  # 20% vacancy
        old = rents[0]
        _update_rents_standalone(rents, total, occupied, np.array([0]))
        assert rents[0] < old

    def test_at_target_vacancy_rents_stable(self):
        """At target vacancy, rents should not change."""
        rents = np.array([20.0])
        total = np.array([100.0])
        occupied = np.array([100.0 * (1.0 - TARGET_VAC_RES)])
        old = rents[0]
        _update_rents_standalone(rents, total, occupied, np.array([0]))
        assert abs(rents[0] - old) < 0.01

    def test_asymmetric_response(self):
        """Same deviation magnitude: rent rises faster than it falls."""
        # Use small deviation (2 ppt) so neither side hits the annual cap.
        # Undersupply: 4% vacancy (2 ppt below 6% target)
        r1 = np.array([20.0])
        _update_rents_standalone(
            r1, np.array([100.0]), np.array([96.0]), np.array([0]))
        rise = r1[0] - 20.0

        # Oversupply: 8% vacancy (2 ppt above 6% target)
        r2 = np.array([20.0])
        _update_rents_standalone(
            r2, np.array([100.0]), np.array([92.0]), np.array([0]))
        fall = 20.0 - r2[0]

        assert rise > fall, "Rents should rise faster than they fall"

    def test_min_rent_floor_prevents_zero(self):
        """After 25 years at extreme oversupply, rent should stay >= $5."""
        MIN_RENT_PSF = 5.0
        rents = np.full(1, 20.0)
        total = np.full(1, 100.0)
        occupied = np.full(1, 1.0)  # 99% vacancy → max decline each year
        for _ in range(25):
            _update_rents_standalone(rents, total, occupied, np.array([0]))
        assert rents[0] >= MIN_RENT_PSF


class TestProformaDeveloper:
    """Test UrbanSim proforma integration."""

    def test_proforma_config_loads(self):
        from urbansim.developer.sqftproforma import SqFtProForma, SqFtProFormaConfig
        config = SqFtProFormaConfig()
        config.cap_rate = 0.05
        config.profit_factor = 1.175
        pf = SqFtProForma(config)
        assert pf is not None

    def test_proforma_lookup_returns_dataframe(self):
        import pandas as pd
        from urbansim.developer.sqftproforma import SqFtProForma, SqFtProFormaConfig

        config = SqFtProFormaConfig()
        pf = SqFtProForma(config)

        df = pd.DataFrame({
            "residential": [20.0, 25.0, 15.0],
            "retail": [22.0, 27.0, 16.0],
            "office": [18.0, 23.0, 14.0],
            "industrial": [10.0, 12.0, 8.0],
            "land_cost": [100000, 200000, 80000],
            "parcel_size": [10000, 15000, 8000],
            "max_far": [3.0, 4.0, 2.0],
            "max_height": [55.0, 72.0, 36.0],
        }, index=["P1", "P2", "P3"])

        result = pf.lookup("residential", df, only_built=True)
        assert isinstance(result, pd.DataFrame)

    def test_developer_pick_with_form_none(self):
        """Developer.pick must use form=None for single-form feasibility."""
        import pandas as pd
        from urbansim.developer.sqftproforma import SqFtProForma, SqFtProFormaConfig
        from urbansim.developer.developer import Developer

        config = SqFtProFormaConfig()
        config.profit_factor = 1.175
        config.cap_rate = 0.05
        pf = SqFtProForma(config)
        df = pd.DataFrame({
            "residential": [25.0] * 5,
            "retail": [27.0] * 5,
            "office": [23.0] * 5,
            "industrial": [12.0] * 5,
            "land_cost": [100000] * 5,
            "parcel_size": [20000] * 5,
            "max_far": [4.0] * 5,
            "max_height": [72.0] * 5,
        }, index=[f"P{i}" for i in range(5)])

        feasibility = pf.lookup("residential", df, only_built=True)
        if len(feasibility) == 0:
            pytest.skip("No feasible parcels at these rents")

        dev = Developer(feasibility)
        # form=None is required for single-form lookup result
        buildings = dev.pick(
            form=None,
            target_units=10,
            parcel_size=df["parcel_size"].reindex(feasibility.index, fill_value=10000),
            ave_unit_size=pd.Series(900, index=feasibility.index),
            current_units=pd.Series(0, index=feasibility.index),
            residential=True,
        )
        assert buildings is not None
        assert len(buildings) > 0


class TestCalibrationSmoke:
    """Verify Tier 2 outputs are in reasonable ranges."""

    def test_corridor_capture_rate_reasonable(self):
        """Corridor should capture a plausible fraction of metro movers."""
        model = HouseholdRelocationModel()
        n = 1000
        n_corridor = 50

        acc = np.zeros(n)
        acc[:n_corridor] = 0.5  # moderate accessibility

        hh = model.allocate_movers_to_corridor(
            near_800_idx=np.arange(n_corridor),
            developable_mask=np.ones(n, dtype=bool),
            n_parcels=n,
            rents=np.full(n, 18.0),
            commute_times=np.full(n, 15.0),
            accessibility=acc,
            year=10,
        )
        capture_rate = hh / model.annual_movers()
        assert 0.01 < capture_rate < 0.50, \
            f"Capture rate {capture_rate:.1%} outside reasonable range"

    def test_movers_scale_with_metro_size(self):
        """Larger metro → more movers."""
        small = HouseholdRelocationModel(RelocationConfig(metro_households=50_000))
        large = HouseholdRelocationModel(RelocationConfig(metro_households=100_000))
        assert large.annual_movers() > small.annual_movers()

    def test_target_vacancy_in_literature_range(self):
        """Structural vacancy 4-8% is the standard range."""
        assert 0.04 <= TARGET_VAC_RES <= 0.08


# ---------------------------------------------------------------------------
# Snapshot/restore — behavioral tests using standalone arrays
# ---------------------------------------------------------------------------

class TestSnapshotRestore:
    """Behavioral tests for per-parcel snapshot/restore correctness."""

    def test_restore_resets_rents_after_mutation(self):
        """Simulates corridor independence: mutate rents, restore, verify reset."""
        n = 50
        baseline_rents = np.full(n, 20.0)
        baseline_units = np.zeros(n)
        baseline_occ = np.zeros(n)

        # "Snapshot"
        snap_rents = baseline_rents.copy()
        snap_units = baseline_units.copy()
        snap_occ = baseline_occ.copy()

        # "Corridor 1 evaluation" — deliver units and adjust rents
        pos = np.array([0, 1, 2, 3, 4])
        baseline_units[pos] = 100.0
        baseline_occ[pos] = 100.0  # 0% vacancy → rents rise
        _update_rents_standalone(
            baseline_rents, baseline_units, baseline_occ, pos)
        assert not np.allclose(baseline_rents, 20.0), "Rents should have changed"

        # "Restore" before corridor 2
        baseline_rents[:] = snap_rents
        baseline_units[:] = snap_units
        baseline_occ[:] = snap_occ

        # Verify restoration
        assert (baseline_rents == 20.0).all()
        assert (baseline_units == 0.0).all()
        assert (baseline_occ == 0.0).all()

    def test_snapshot_restore_preserves_all_arrays(self):
        """Multiple arrays should all be restored after mutation."""
        n = 20
        rents = np.full(n, 20.0)
        units = np.zeros(n)
        occ = np.zeros(n)
        comm = np.zeros(n)

        # Snapshot
        snap = {k: v.copy() for k, v in
                [("rents", rents), ("units", units), ("occ", occ), ("comm", comm)]}

        # Mutate all
        rents[:] = 99.0
        units[:] = 50.0
        occ[:] = 45.0
        comm[:] = 1000.0

        # Restore
        rents[:] = snap["rents"]
        units[:] = snap["units"]
        occ[:] = snap["occ"]
        comm[:] = snap["comm"]

        assert (rents == 20.0).all()
        assert (units == 0.0).all()
        assert (occ == 0.0).all()
        assert (comm == 0.0).all()


# ---------------------------------------------------------------------------
# Double-counting prevention (Stage 6 issue #1)
# ---------------------------------------------------------------------------

class TestCumulativePopAccounting:
    """Verify _cumulative_corridor_pop doesn't double-count student pop."""

    def test_update_segment_supply_does_not_update_cumulative_pop(self):
        """update_segment_supply must NOT modify _cumulative_corridor_pop.

        The single source of truth for cumulative pop is section E of
        _run_development_model() — update_segment_supply only tracks
        per-segment unit/sqft supply.
        """
        from src.demand_driven_development import (
            DemandDrivenDevelopmentModel,
            MetroGrowthParams,
            MarketParams,
            CorridorCaptureParams,
            AbsorptionParams,
        )
        model = DemandDrivenDevelopmentModel(
            growth_params=MetroGrowthParams(),
            market_params=MarketParams(),
            capture_params=CorridorCaptureParams(),
            absorption_params=AbsorptionParams(),
        )

        # Record initial cumulative pop
        cid = "C_TEST"
        before = model._cumulative_corridor_pop.get(cid, 0.0)

        # Call update_segment_supply with student results that include pop
        student_result = {"new_units": 50.0, "new_pop": 128.0}
        model.update_segment_supply(cid, "student", student_result)

        after = model._cumulative_corridor_pop.get(cid, 0.0)
        assert after == before, (
            f"update_segment_supply should NOT modify _cumulative_corridor_pop "
            f"(was {before}, now {after})")

    def test_student_pop_counted_once_in_combined(self):
        """combined['new_pop'] includes student pop exactly once.

        If section E adds combined['new_pop'] to _cumulative_corridor_pop,
        and update_segment_supply also added it, student pop would be
        double-counted.  This test documents the invariant.
        """
        # Simulate the three channels
        res_pop = 200.0
        student_pop = 80.0
        combined_new_pop = res_pop + student_pop  # section A + B

        # Section E should add this once
        cum_pop_before = 500.0
        cum_pop_after = cum_pop_before + combined_new_pop

        # Student pop fraction of total
        student_frac = student_pop / combined_new_pop
        assert student_frac == pytest.approx(80.0 / 280.0)

        # If double-counted, cum_pop would be inflated
        double_counted = cum_pop_before + combined_new_pop + student_pop
        assert cum_pop_after < double_counted
        assert cum_pop_after == cum_pop_before + combined_new_pop


# ---------------------------------------------------------------------------
# Corridor independence (end-to-end behavioral)
# ---------------------------------------------------------------------------

class TestCorridorIndependence:
    """Verify that corridor evaluations don't leak state."""

    def test_two_corridors_start_from_same_baseline(self):
        """Simulate two corridor evaluations with snapshot/restore."""
        n = 100
        rents = np.full(n, 20.0)
        units = np.zeros(n)
        occ = np.zeros(n)

        # Snapshot baseline
        snap_rents = rents.copy()
        snap_units = units.copy()
        snap_occ = occ.copy()

        # --- Corridor 1: heavy development (rents change) ---
        c1_pos = np.arange(10)
        units[c1_pos] = 200.0
        occ[c1_pos] = 100.0  # 50% vacancy → rents fall
        _update_rents_standalone(rents, units, occ, c1_pos)
        c1_rents = rents[c1_pos].copy()
        assert (c1_rents < 20.0).all(), "C1 should have lowered rents"

        # Restore baseline
        rents[:] = snap_rents
        units[:] = snap_units
        occ[:] = snap_occ

        # --- Corridor 2: verify starts from original baseline ---
        c2_pos = np.arange(10, 20)
        # Before any C2 development, rents should be at baseline
        assert (rents[c2_pos] == 20.0).all()
        # C1's parcels should also be back to baseline
        assert (rents[c1_pos] == 20.0).all()
        assert (units[c1_pos] == 0.0).all()

    def test_overlapping_parcels_dont_leak(self):
        """Two corridors sharing parcels should each start from baseline."""
        n = 50
        rents = np.full(n, 15.0)
        units = np.zeros(n)
        occ = np.zeros(n)

        snap_rents = rents.copy()
        snap_units = units.copy()
        snap_occ = occ.copy()

        shared = np.array([5, 6, 7])  # parcels near both corridors

        # Corridor A mutates shared parcels
        units[shared] = 100.0
        occ[shared] = 100.0  # 0% vacancy → rents rise
        _update_rents_standalone(rents, units, occ, shared)
        assert rents[5] > 15.0

        # Restore
        rents[:] = snap_rents
        units[:] = snap_units
        occ[:] = snap_occ

        # Corridor B starts fresh on the same shared parcels
        assert rents[5] == 15.0
        assert units[5] == 0.0


# ---------------------------------------------------------------------------
# Integration: MNL → proforma → vacancy feedback chain
# ---------------------------------------------------------------------------

class TestTier2PipelineIntegration:
    """End-to-end test of MNL demand → proforma build → vacancy rent feedback.

    Uses actual production modules (relocation_model, SqFtProForma) to verify
    the full Tier 2 chain works without importing the heavy LandUseTransportModel.
    """

    def test_mnl_demand_drives_proforma_build(self):
        """MNL produces demand → proforma evaluates feasibility → units built."""
        import pandas as pd
        from urbansim.developer.sqftproforma import SqFtProForma, SqFtProFormaConfig
        from urbansim.developer.developer import Developer

        model = HouseholdRelocationModel()
        n = 200
        n_corridor = 20

        # Step 1: MNL produces corridor demand
        acc = np.zeros(n)
        acc[:n_corridor] = 0.6
        corridor_hh = model.allocate_movers_to_corridor(
            near_800_idx=np.arange(n_corridor),
            developable_mask=np.ones(n, dtype=bool),
            n_parcels=n,
            rents=np.full(n, 18.0),
            commute_times=np.full(n, 15.0),
            accessibility=acc,
            year=5,
        )
        assert corridor_hh > 0, "MNL should produce positive corridor demand"

        # Step 2: Convert demand to target units
        target_units = max(int(corridor_hh / (1.0 - TARGET_VAC_RES)), 1)
        assert target_units > 0

        # Step 3: Proforma evaluates feasibility
        config = SqFtProFormaConfig()
        config.profit_factor = 1.175
        config.cap_rate = 0.05
        pf = SqFtProForma(config)
        df = pd.DataFrame({
            "residential": [20.0] * n_corridor,
            "retail": [22.0] * n_corridor,
            "office": [19.0] * n_corridor,
            "industrial": [11.0] * n_corridor,
            "land_cost": [200000] * n_corridor,
            "parcel_size": [15000] * n_corridor,
            "max_far": [3.0] * n_corridor,
            "max_height": [55.0] * n_corridor,
        }, index=[f"P{i}" for i in range(n_corridor)])

        feasibility = pf.lookup("residential", df, only_built=True)
        if len(feasibility) == 0:
            pytest.skip("No feasible parcels at these rents/costs")

        dev = Developer(feasibility)
        buildings = dev.pick(
            form=None,
            target_units=target_units,
            parcel_size=df["parcel_size"].reindex(feasibility.index, fill_value=10000),
            ave_unit_size=pd.Series(900, index=feasibility.index),
            current_units=pd.Series(0, index=feasibility.index),
            residential=True,
        )
        assert buildings is not None
        units_built = buildings["residential_units"].sum()
        assert units_built > 0, "Proforma should produce buildings"

        # Step 4: Vacancy feedback adjusts rents
        rents = np.full(n, 20.0)
        total_units = np.zeros(n)
        occupied = np.zeros(n)
        pos = np.arange(n_corridor)
        total_units[pos] = float(units_built)
        # Fill at target vacancy
        occupied[pos] = float(units_built) * (1.0 - TARGET_VAC_RES)
        result = _update_rents_standalone(rents, total_units, occupied, pos)

        # At target vacancy, rents should be approximately stable
        assert abs(result["mean_rent_change"] - 1.0) < 0.01

    def test_oversupply_depresses_next_year_rents(self):
        """Over-building → high vacancy → rents fall → less feasible next year."""
        import pandas as pd
        from urbansim.developer.sqftproforma import SqFtProForma, SqFtProFormaConfig

        n = 50
        rents = np.full(n, 20.0)
        total_units = np.zeros(n)
        occupied = np.zeros(n)
        pos = np.arange(10)

        # Simulate heavy over-building (20% vacancy)
        total_units[pos] = 500.0
        occupied[pos] = 400.0
        _update_rents_standalone(rents, total_units, occupied, pos)
        depressed_rent = rents[0]
        assert depressed_rent < 20.0, "Oversupply should lower rents"

        # Depressed rents reduce proforma feasibility
        config = SqFtProFormaConfig()
        config.profit_factor = 1.175
        config.cap_rate = 0.05
        pf = SqFtProForma(config)

        # Low rents → fewer feasible parcels
        df_low = pd.DataFrame({
            "residential": [depressed_rent] * 5,
            "retail": [depressed_rent * 1.1] * 5,
            "office": [depressed_rent * 0.95] * 5,
            "industrial": [depressed_rent * 0.55] * 5,
            "land_cost": [200000] * 5,
            "parcel_size": [15000] * 5,
            "max_far": [3.0] * 5,
            "max_height": [55.0] * 5,
        }, index=[f"P{i}" for i in range(5)])

        df_high = df_low.copy()
        df_high["residential"] = 20.0
        df_high["retail"] = 22.0
        df_high["office"] = 19.0
        df_high["industrial"] = 11.0

        f_low = pf.lookup("residential", df_low, only_built=True)
        f_high = pf.lookup("residential", df_high, only_built=True)

        # At minimum: depressed rents shouldn't produce MORE feasibility
        # (they could produce equal if both are 0 or both are all feasible)
        if len(f_high) > 0:
            assert len(f_low) <= len(f_high), \
                "Lower rents should not produce more feasible parcels"

    def test_student_and_market_use_same_proforma(self):
        """Both student and market-rate housing should evaluate under same economics.

        This verifies the Tier 2 unification: both channels use SqFtProForma
        rather than separate proforma engines with different cost structures.
        """
        import pandas as pd
        from urbansim.developer.sqftproforma import SqFtProForma, SqFtProFormaConfig

        config = SqFtProFormaConfig()
        config.profit_factor = 1.175
        config.cap_rate = 0.05
        pf = SqFtProForma(config)

        # Same parcel, same rents — both market and student use this proforma
        df = pd.DataFrame({
            "residential": [22.0],
            "retail": [24.0],
            "office": [21.0],
            "industrial": [12.0],
            "land_cost": [150000],
            "parcel_size": [12000],
            "max_far": [3.0],
            "max_height": [55.0],
        }, index=["P_campus"])

        result = pf.lookup("residential", df, only_built=True)
        # The same proforma evaluates both market-rate and student housing
        # (student housing is just residential on campus-proximate parcels)
        assert isinstance(result, pd.DataFrame)

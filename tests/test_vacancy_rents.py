"""Tests for Tier 2 Stage 5: Vacancy-driven rent feedback.

Tests use the production functions from src.vacancy_rent_feedback
(lightweight module, no geopandas/osmnx dependency).
"""
import numpy as np
import pytest

from src.relocation_model import TARGET_VAC_RES
from src.vacancy_rent_feedback import (
    update_rents_from_vacancy,
    track_unit_delivery,
    track_unit_delivery_fallback,
    MIN_RENT_PSF,
)

AVG_UNIT_SQFT = 900
AVG_HOUSEHOLD_SIZE = 2.56


# ---------------------------------------------------------------------------
# Helper to create arrays
# ---------------------------------------------------------------------------

def _make_arrays(n=100, base_rent=20.0):
    return {
        "rents": np.full(n, base_rent, dtype=np.float64),
        "total_units": np.zeros(n, dtype=np.float64),
        "occupied_units": np.zeros(n, dtype=np.float64),
        "total_comm_sqft": np.zeros(n, dtype=np.float64),
        "occupied_comm_sqft": np.zeros(n, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Tests: update_rents_from_vacancy
# ---------------------------------------------------------------------------

class TestRentFeedback:
    def test_at_target_vacancy_no_change(self):
        a = _make_arrays()
        pos = np.array([0, 1, 2])
        a["total_units"][pos] = 100.0
        a["occupied_units"][pos] = 100.0 * (1.0 - TARGET_VAC_RES)
        old_rents = a["rents"][pos].copy()

        result = update_rents_from_vacancy(
            a["rents"], a["total_units"], a["occupied_units"], pos)
        assert np.allclose(a["rents"][pos], old_rents, atol=0.01)
        assert abs(result["mean_rent_change"] - 1.0) < 0.001

    def test_undersupply_raises_rents(self):
        a = _make_arrays()
        pos = np.array([5])
        a["total_units"][pos] = 100.0
        a["occupied_units"][pos] = 100.0  # 0% vacancy
        old_rent = a["rents"][5]

        update_rents_from_vacancy(
            a["rents"], a["total_units"], a["occupied_units"], pos)
        assert a["rents"][5] > old_rent

    def test_oversupply_lowers_rents(self):
        a = _make_arrays()
        pos = np.array([10])
        a["total_units"][pos] = 100.0
        a["occupied_units"][pos] = 70.0  # 30% vacancy
        old_rent = a["rents"][10]

        update_rents_from_vacancy(
            a["rents"], a["total_units"], a["occupied_units"], pos)
        assert a["rents"][10] < old_rent

    def test_asymmetric_speed(self):
        """Rents rise faster than they fall for same deviation magnitude."""
        a1 = _make_arrays()
        a2 = _make_arrays()
        pos = np.array([0])

        # Use small deviation (2 ppt) so neither side hits the annual cap.
        # Undersupply: 4% vacancy (2 ppt below 6% target)
        a1["total_units"][pos] = 100.0
        a1["occupied_units"][pos] = 96.0
        r1 = update_rents_from_vacancy(
            a1["rents"], a1["total_units"], a1["occupied_units"], pos)
        rise = abs(r1["mean_rent_change"] - 1.0)

        # Oversupply: 8% vacancy (2 ppt above 6% target)
        a2["total_units"][pos] = 100.0
        a2["occupied_units"][pos] = 92.0
        r2 = update_rents_from_vacancy(
            a2["rents"], a2["total_units"], a2["occupied_units"], pos)
        fall = abs(r2["mean_rent_change"] - 1.0)

        assert rise > fall, "Rents should rise faster than they fall"

    def test_max_annual_cap(self):
        """Even extreme vacancy shouldn't exceed 15% annual change."""
        a = _make_arrays()
        pos = np.array([0])
        a["total_units"][pos] = 100.0
        a["occupied_units"][pos] = 1.0  # 99% vacancy
        old_rent = a["rents"][0]

        update_rents_from_vacancy(
            a["rents"], a["total_units"], a["occupied_units"], pos, step_years=1)
        assert a["rents"][0] >= old_rent * 0.85

    def test_empty_positions_noop(self):
        a = _make_arrays()
        old = a["rents"].copy()
        result = update_rents_from_vacancy(
            a["rents"], a["total_units"], a["occupied_units"],
            np.array([], dtype=int))
        assert np.array_equal(a["rents"], old)
        assert result["mean_rent_change"] == 1.0

    def test_no_units_uses_target_vacancy(self):
        a = _make_arrays()
        pos = np.array([0, 1])
        old = a["rents"][pos].copy()
        update_rents_from_vacancy(
            a["rents"], a["total_units"], a["occupied_units"], pos)
        assert np.allclose(a["rents"][pos], old, atol=0.01)

    def test_step_years_scaling(self):
        a1 = _make_arrays()
        a2 = _make_arrays()
        pos = np.array([0])
        for a in (a1, a2):
            a["total_units"][pos] = 100.0
            a["occupied_units"][pos] = 100.0

        r1 = update_rents_from_vacancy(
            a1["rents"], a1["total_units"], a1["occupied_units"], pos, step_years=1)
        r2 = update_rents_from_vacancy(
            a2["rents"], a2["total_units"], a2["occupied_units"], pos, step_years=3)

        assert abs(r2["mean_rent_change"] - 1.0) > abs(r1["mean_rent_change"] - 1.0)

    def test_diagnostics_keys(self):
        a = _make_arrays()
        pos = np.array([0, 1])
        a["total_units"][pos] = 50.0
        a["occupied_units"][pos] = 47.0
        result = update_rents_from_vacancy(
            a["rents"], a["total_units"], a["occupied_units"], pos)
        assert "mean_vacancy" in result
        assert "mean_rent_change" in result
        assert "parcels_with_units" in result
        assert "parcels_total" in result
        assert result["parcels_total"] == 2
        assert result["parcels_with_units"] == 2

    def test_min_rent_floor_prevents_zero(self):
        """After 25 years at extreme oversupply, rent should stay >= MIN_RENT_PSF."""
        rents = np.full(1, 20.0)
        total = np.full(1, 100.0)
        occupied = np.full(1, 1.0)  # 99% vacancy
        for _ in range(25):
            update_rents_from_vacancy(rents, total, occupied, np.array([0]))
        assert rents[0] >= MIN_RENT_PSF

    def test_commercial_blending(self):
        """When commercial arrays provided, blending should affect result."""
        a = _make_arrays()
        pos = np.array([0])
        a["total_units"][pos] = 100.0
        a["occupied_units"][pos] = 100.0 * (1.0 - TARGET_VAC_RES)
        a["total_comm_sqft"][pos] = 50000.0
        a["occupied_comm_sqft"][pos] = 25000.0  # 50% comm vacancy

        result = update_rents_from_vacancy(
            a["rents"], a["total_units"], a["occupied_units"], pos,
            total_comm_sqft=a["total_comm_sqft"],
            occupied_comm_sqft=a["occupied_comm_sqft"],
        )
        assert "mean_comm_vacancy" in result
        assert "parcels_with_comm" in result


# ---------------------------------------------------------------------------
# Tests: track_unit_delivery
# ---------------------------------------------------------------------------

class TestTrackUnitDelivery:
    def test_single_parcel_residential(self):
        a = _make_arrays()
        track_unit_delivery(
            a["total_units"], a["occupied_units"],
            a["total_comm_sqft"], a["occupied_comm_sqft"],
            pos=5, res_sqft=9000.0, comm_sqft=0.0)
        assert a["total_units"][5] == pytest.approx(10.0)
        # New units start vacant (lease-up lag); absorbed via absorb_vacant_units()
        assert a["occupied_units"][5] == pytest.approx(0.0)
        assert a["total_comm_sqft"][5] == 0.0

    def test_single_parcel_commercial(self):
        a = _make_arrays()
        track_unit_delivery(
            a["total_units"], a["occupied_units"],
            a["total_comm_sqft"], a["occupied_comm_sqft"],
            pos=3, res_sqft=0.0, comm_sqft=5000.0)
        assert a["total_units"][3] == 0.0
        assert a["total_comm_sqft"][3] == 5000.0
        assert a["occupied_comm_sqft"][3] == pytest.approx(5000.0 * 0.90)

    def test_single_parcel_mixed(self):
        a = _make_arrays()
        track_unit_delivery(
            a["total_units"], a["occupied_units"],
            a["total_comm_sqft"], a["occupied_comm_sqft"],
            pos=7, res_sqft=4500.0, comm_sqft=2000.0)
        assert a["total_units"][7] == pytest.approx(5.0)
        assert a["total_comm_sqft"][7] == 2000.0

    def test_array_positions(self):
        a = _make_arrays()
        pos = np.array([0, 1, 2])
        track_unit_delivery(
            a["total_units"], a["occupied_units"],
            a["total_comm_sqft"], a["occupied_comm_sqft"],
            pos=pos, res_sqft=2700.0, comm_sqft=0.0)
        assert np.allclose(a["total_units"][pos], 3.0)

    def test_cumulative(self):
        a = _make_arrays()
        track_unit_delivery(
            a["total_units"], a["occupied_units"],
            a["total_comm_sqft"], a["occupied_comm_sqft"],
            pos=0, res_sqft=900.0)
        track_unit_delivery(
            a["total_units"], a["occupied_units"],
            a["total_comm_sqft"], a["occupied_comm_sqft"],
            pos=0, res_sqft=1800.0)
        assert a["total_units"][0] == pytest.approx(3.0)

    def test_zero_sqft_noop(self):
        a = _make_arrays()
        track_unit_delivery(
            a["total_units"], a["occupied_units"],
            a["total_comm_sqft"], a["occupied_comm_sqft"],
            pos=0, res_sqft=0.0, comm_sqft=0.0)
        assert a["total_units"][0] == 0.0
        assert a["total_comm_sqft"][0] == 0.0


class TestTrackUnitDeliveryFallback:
    def test_distributes_by_frac(self):
        a = _make_arrays()
        near_idx = np.array([10, 11, 12])
        frac = np.array([0.5, 0.3, 0.2])
        pop = 25.6  # = 10 HH at 2.56 persons/HH
        track_unit_delivery_fallback(
            a["total_units"], a["occupied_units"],
            near_idx, frac, pop=pop)

        expected_units = pop / AVG_HOUSEHOLD_SIZE
        total_tracked = a["total_units"][near_idx].sum()
        assert total_tracked == pytest.approx(expected_units, rel=0.01)
        assert a["total_units"][10] > a["total_units"][11] > a["total_units"][12]

    def test_zero_pop_noop(self):
        a = _make_arrays()
        near_idx = np.array([0, 1])
        frac = np.array([0.5, 0.5])
        track_unit_delivery_fallback(
            a["total_units"], a["occupied_units"],
            near_idx, frac, pop=0.0)
        assert a["total_units"][0] == 0.0


# ---------------------------------------------------------------------------
# Integration: delivery → vacancy → rent adjustment cycle
# ---------------------------------------------------------------------------

class TestDeliveryToRentCycle:
    def test_delivery_then_rent_feedback(self):
        """Full cycle: deliver vacant units → absorb → check vacancy → rents."""
        from src.vacancy_rent_feedback import absorb_vacant_units
        a = _make_arrays(n=50, base_rent=20.0)
        pos = np.array([5, 6, 7])

        for p in pos:
            track_unit_delivery(
                a["total_units"], a["occupied_units"],
                a["total_comm_sqft"], a["occupied_comm_sqft"],
                pos=p, res_sqft=27000.0)  # 30 units, start vacant

        # Before absorption: 100% vacancy → rents should fall hard
        result_pre = update_rents_from_vacancy(
            a["rents"].copy(), a["total_units"], a["occupied_units"], pos)
        assert result_pre["mean_rent_change"] < 1.0

        # After absorption (65% fill): vacancy ~35% → rents still fall
        absorb_vacant_units(a["total_units"], a["occupied_units"], pos)
        result = update_rents_from_vacancy(
            a["rents"], a["total_units"], a["occupied_units"], pos)
        assert result["mean_rent_change"] < 1.0

    def test_oversupply_cycle(self):
        """Deliver units but don't fill them → rents fall."""
        a = _make_arrays(n=50, base_rent=25.0)
        pos = np.array([10])

        a["total_units"][pos] = 200.0
        a["occupied_units"][pos] = 100.0  # 50% vacancy
        old_rent = a["rents"][10]

        update_rents_from_vacancy(
            a["rents"], a["total_units"], a["occupied_units"], pos)
        assert a["rents"][10] < old_rent

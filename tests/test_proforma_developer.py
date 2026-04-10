"""Tests for Tier 2 Stage 4: UrbanSim SqFtProForma + Developer.pick() wrapper."""
import numpy as np
import pandas as pd
import pytest

from urbansim.developer.sqftproforma import SqFtProForma, SqFtProFormaConfig
from urbansim.developer.developer import Developer


# ---------------------------------------------------------------------------
# Lafayette-calibrated config (mirrors _make_lafayette_proforma_config)
# ---------------------------------------------------------------------------

def _make_test_config():
    """Reproduce the Lafayette config for isolated testing."""
    config = SqFtProFormaConfig()
    config.costs = {
        "retail":      [155.0, 170.0, 195.0, 225.0],
        "industrial":  [125.0, 155.0, 180.0, 210.0],
        "office":      [150.0, 170.0, 195.0, 225.0],
        "residential": [155.0, 175.0, 195.0, 225.0],
    }
    config.heights_for_costs = [15, 55, 120, np.inf]
    config.profit_factor = 1.175
    config.cap_rate = 0.05
    config.building_efficiency = 0.70
    config.parcel_coverage = 0.80
    config.parking_rates = {
        "retail": 2.0, "industrial": 0.6, "office": 1.0, "residential": 1.0,
    }
    config.parking_configs = ["surface", "deck"]
    config.parking_cost_d = {"surface": 30, "deck": 90}
    config.parking_sqft_d = {"surface": 300.0, "deck": 250.0}
    return config


def _make_parcel_df(n=10, rent=20.0, land_cost=500_000.0, parcel_size=15_000.0, max_far=3.0):
    """Create a small parcel DataFrame matching SqFtProForma.lookup() input schema."""
    idx = pd.Index([f"P{i:04d}" for i in range(n)], name="parcel_id")
    cfg = SqFtProFormaConfig()
    return pd.DataFrame({
        "residential": rent,
        "retail": rent * 1.10,
        "office": rent * 0.95,
        "industrial": rent * 0.55,
        "land_cost": land_cost,
        "parcel_size": parcel_size,
        "max_far": max_far,
        "max_height": max_far / 0.80 * cfg.height_per_story,
    }, index=idx)


# ---------------------------------------------------------------------------
# Tests: SqFtProForma config
# ---------------------------------------------------------------------------

class TestLafayetteConfig:
    def test_config_passes_validation(self):
        config = _make_test_config()
        config.check_is_reasonable()

    def test_proforma_initializes(self):
        pf = SqFtProForma(_make_test_config())
        assert pf.config.cap_rate == 0.05
        assert pf.config.profit_factor == 1.175
        assert set(pf.config.parking_configs) == {"surface", "deck"}

    def test_cost_tiers_match_height_breaks(self):
        config = _make_test_config()
        for use, costs in config.costs.items():
            assert len(costs) == len(config.heights_for_costs), \
                f"{use} cost tiers don't match height breaks"


# ---------------------------------------------------------------------------
# Tests: SqFtProForma.lookup()
# ---------------------------------------------------------------------------

class TestProFormaLookup:
    def test_lookup_returns_feasible_rows(self):
        pf = SqFtProForma(_make_test_config())
        df = _make_parcel_df(n=10, rent=20.0)
        result = pf.lookup("residential", df, only_built=True)
        assert isinstance(result, pd.DataFrame)
        # At $20/sqft residential rent, some parcels should be feasible
        assert len(result) > 0
        assert "max_profit" in result.columns
        assert "max_profit_far" in result.columns

    def test_lookup_low_rent_not_feasible(self):
        pf = SqFtProForma(_make_test_config())
        df = _make_parcel_df(n=10, rent=2.0)  # very low rent
        result = pf.lookup("residential", df, only_built=True)
        # At $2/sqft nothing should be profitable
        assert len(result) == 0

    def test_lookup_high_rent_all_feasible(self):
        pf = SqFtProForma(_make_test_config())
        df = _make_parcel_df(n=5, rent=50.0)  # high rent
        result = pf.lookup("residential", df, only_built=True)
        assert len(result) == 5

    def test_lookup_respects_max_far(self):
        pf = SqFtProForma(_make_test_config())
        df = _make_parcel_df(n=5, rent=30.0, max_far=0.5)
        result = pf.lookup("residential", df, only_built=True)
        if len(result) > 0:
            assert (result["max_profit_far"] <= 0.51).all()  # small float tolerance

    def test_lookup_multiple_forms(self):
        pf = SqFtProForma(_make_test_config())
        df = _make_parcel_df(n=10, rent=25.0)
        for form in ["residential", "retail", "office"]:
            result = pf.lookup(form, df, only_built=True)
            assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: Developer.pick()
# ---------------------------------------------------------------------------

class TestDeveloperPick:
    def test_pick_returns_buildings(self):
        pf = SqFtProForma(_make_test_config())
        df = _make_parcel_df(n=20, rent=25.0, parcel_size=20_000.0)
        feasibility = pf.lookup("residential", df, only_built=True)
        assert len(feasibility) > 0

        developer = Developer(feasibility)
        parcel_size = df["parcel_size"].reindex(feasibility.index, fill_value=10000)
        ave_unit_size = pd.Series(900.0, index=feasibility.index)
        current_units = pd.Series(0.0, index=feasibility.index)

        new_buildings = developer.pick(
            form=None,
            target_units=50,
            parcel_size=parcel_size,
            ave_unit_size=ave_unit_size,
            current_units=current_units,
            residential=True,
            drop_after_build=True,
            bldg_sqft_per_job=200.0,
        )
        assert new_buildings is not None
        assert len(new_buildings) > 0
        assert "residential_units" in new_buildings.columns
        assert (new_buildings["residential_units"] > 0).all()

    def test_pick_respects_target_units(self):
        pf = SqFtProForma(_make_test_config())
        df = _make_parcel_df(n=30, rent=25.0, parcel_size=20_000.0)
        feasibility = pf.lookup("residential", df, only_built=True)
        if len(feasibility) == 0:
            pytest.skip("No feasible parcels for this test")

        developer = Developer(feasibility)
        parcel_size = df["parcel_size"].reindex(feasibility.index, fill_value=10000)
        ave_unit_size = pd.Series(900.0, index=feasibility.index)
        current_units = pd.Series(0.0, index=feasibility.index)

        result = developer.pick(
            form=None,
            target_units=10,
            parcel_size=parcel_size,
            ave_unit_size=ave_unit_size,
            current_units=current_units,
            residential=True,
        )
        assert result is not None
        # Developer should build at least target_units net
        assert result["net_units"].sum() >= 10

    def test_pick_zero_target_returns_empty(self):
        pf = SqFtProForma(_make_test_config())
        df = _make_parcel_df(n=10, rent=25.0)
        feasibility = pf.lookup("residential", df, only_built=True)
        if len(feasibility) == 0:
            pytest.skip("No feasible parcels")

        developer = Developer(feasibility)
        parcel_size = df["parcel_size"].reindex(feasibility.index, fill_value=10000)
        ave_unit_size = pd.Series(900.0, index=feasibility.index)
        current_units = pd.Series(0.0, index=feasibility.index)

        result = developer.pick(
            form=None,
            target_units=0,
            parcel_size=parcel_size,
            ave_unit_size=ave_unit_size,
            current_units=current_units,
            residential=True,
        )
        # target_units=0 → empty build_idx → empty DataFrame
        assert result is not None
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: _get_parcel_max_far
# ---------------------------------------------------------------------------

class TestGetParcelMaxFar:
    """Test _get_parcel_max_far via the ZONING_MATRIX and NO_ZONING_FAR_CAP.

    Uses types.SimpleNamespace as a lightweight stand-in for the model instance
    (the method only reads self.development_scenario).
    """

    def _make_model(self, scenario="current_zoning"):
        from types import SimpleNamespace
        return SimpleNamespace(development_scenario=scenario)

    def test_current_zoning_uses_column_0(self):
        from src.land_use_transport_model import LandUseTransportModel
        from src.developer_proforma import ZONING_MATRIX
        m = self._make_model("current_zoning")
        result = LandUseTransportModel._get_parcel_max_far(m, "R1")
        assert result == ZONING_MATRIX["R1"][0]

    def test_no_zoning_returns_cap(self):
        from src.land_use_transport_model import LandUseTransportModel
        m = self._make_model("no_zoning")
        result = LandUseTransportModel._get_parcel_max_far(m, "R1")
        assert result == 8.0

    def test_unknown_zone_returns_zero(self):
        from src.land_use_transport_model import LandUseTransportModel
        m = self._make_model("current_zoning")
        result = LandUseTransportModel._get_parcel_max_far(m, "NONEXISTENT")
        assert result == 0.0

    def test_cb_zones_are_mixed_use(self):
        from src.land_use_transport_model import LandUseTransportModel
        from src.developer_proforma import ZONING_MATRIX
        m = self._make_model("current_zoning")
        assert LandUseTransportModel._get_parcel_max_far(m, "CB") == ZONING_MATRIX["CB"][0]
        assert LandUseTransportModel._get_parcel_max_far(m, "CBW") == ZONING_MATRIX["CBW"][0]

    def test_reclassified_zones_not_zero(self):
        """GB, HB, NB, NBU, MR, MRU should all have non-zero FAR."""
        from src.land_use_transport_model import LandUseTransportModel
        m = self._make_model("current_zoning")
        for zone in ("GB", "HB", "NB", "NBU", "MR", "MRU"):
            far = LandUseTransportModel._get_parcel_max_far(m, zone)
            assert far > 0, f"{zone} should have non-zero FAR, got {far}"

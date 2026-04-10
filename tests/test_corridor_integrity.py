"""Regression tests for corridor integrity and financial parameter consistency.

Guards against:
1. Duplicate corridor geometries reappearing in the GeoJSON
2. Financial parameter drift between scripts
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORRIDORS_PATH = PROJECT_ROOT / "data" / "processed" / "apm_phase2a_corridors.geojson"


# ============================================================================
# Corridor uniqueness
# ============================================================================


def _load_corridors():
    if not CORRIDORS_PATH.exists():
        pytest.skip(f"Corridors file not found: {CORRIDORS_PATH}")
    with open(CORRIDORS_PATH) as f:
        return json.load(f)


def test_no_surrogate_corridors():
    """No surrogate-cloned corridors should remain in the GeoJSON."""
    gj = _load_corridors()
    surrogates = [
        f["properties"]["corridor_id"]
        for f in gj["features"]
        if f["properties"].get("source", "").startswith("surrogate_from_")
    ]
    assert surrogates == [], f"Surrogate corridors found: {surrogates}"


def test_no_duplicate_geometries():
    """No two corridors should share identical or reversed geometry."""
    gj = _load_corridors()

    def _canon(coords):
        """Canonical form: forward or reversed, whichever is lexicographically smaller."""
        fwd = tuple(tuple(c) for c in coords)
        rev = fwd[::-1]
        return min(fwd, rev)

    seen = {}
    duplicates = []
    for feat in gj["features"]:
        cid = feat["properties"]["corridor_id"]
        canon = _canon(feat["geometry"]["coordinates"])
        if canon in seen:
            duplicates.append((cid, seen[canon]))
        else:
            seen[canon] = cid

    assert duplicates == [], f"Duplicate geometries: {duplicates}"


def test_corridor_ids_are_unique():
    """Each corridor_id should appear exactly once."""
    gj = _load_corridors()
    ids = [f["properties"]["corridor_id"] for f in gj["features"]]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"


def test_corridors_have_stop_coords():
    """Each corridor should have demand-responsive stop_coords."""
    gj = _load_corridors()
    missing = []
    for feat in gj["features"]:
        cid = feat["properties"]["corridor_id"]
        sc = feat["properties"].get("stop_coords")
        if not sc:
            missing.append(cid)
    assert missing == [], f"Corridors missing stop_coords: {missing}"


def test_stop_coords_are_valid():
    """stop_coords should be parseable JSON with at least 2 [lon, lat] pairs."""
    gj = _load_corridors()
    for feat in gj["features"]:
        cid = feat["properties"]["corridor_id"]
        sc = feat["properties"].get("stop_coords")
        if not sc:
            continue
        coords = json.loads(sc) if isinstance(sc, str) else sc
        assert isinstance(coords, list), f"{cid}: stop_coords is not a list"
        assert len(coords) >= 2, f"{cid}: fewer than 2 stops"
        for i, c in enumerate(coords):
            assert len(c) == 2, f"{cid} stop {i}: expected [lon, lat], got {c}"
            assert -180 <= c[0] <= 180, f"{cid} stop {i}: lon {c[0]} out of range"
            assert -90 <= c[1] <= 90, f"{cid} stop {i}: lat {c[1]} out of range"


# ============================================================================
# Financial parameter consistency
# ============================================================================


def test_financial_params_computations():
    """Verify financial parameter computations produce correct results."""
    from src.financial_params import (
        compute_capital_cost,
        effective_tif_tax_rate,
        CAPITAL_COST_PER_KM,
        CAPITAL_COST_GUIDEWAY_PER_KM,
        PROFESSIONAL_SERVICES_RATE,
        TIF_CAPTURE_RATE,
    )

    # compute_capital_cost: 5 km corridor, 4 stations
    total = compute_capital_cost(5.0, 4, escalation=False)
    assert total > 0, "compute_capital_cost should return positive value"

    # Escalated cost should exceed base (inflation markup)
    total_esc = compute_capital_cost(5.0, 4, escalation=True)
    assert total_esc >= total, "Escalated cost should not be less than base"

    # Blended rate should exceed guideway-only (includes stations, vehicles, etc)
    assert CAPITAL_COST_PER_KM > CAPITAL_COST_GUIDEWAY_PER_KM

    # Professional services is a fraction
    assert 0 < PROFESSIONAL_SERVICES_RATE < 1

    # TIF capture rate in valid range
    assert 0 < TIF_CAPTURE_RATE <= 1

    # effective_tif_tax_rate: blended rate should be between res and comm caps
    eff = effective_tif_tax_rate()
    assert 0 < eff < 0.05, f"Effective TIF rate {eff} outside reasonable range"

    # Mostly-residential base should give lower effective rate
    eff_res = effective_tif_tax_rate(res_share=0.90)
    eff_comm = effective_tif_tax_rate(res_share=0.10)
    assert eff_res < eff_comm, "Higher residential share should give lower effective rate"

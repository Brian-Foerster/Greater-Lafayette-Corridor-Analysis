"""Unit tests for Week 3 parcel capacity depletion logic."""
import unittest

try:
    from src.land_use_transport_model import update_capacity_state
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    update_capacity_state = None
    _IMPORT_ERROR = exc


class TestCapacityLedger(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_no_double_counting_when_requests_exceed_capacity(self):
        total = 0.0
        remaining = 0.0
        delivered_cumulative = 0.0

        # Step 1: create 100 sqft capacity and request 60
        state = update_capacity_state(total, remaining, theoretical_capacity_sqft=100.0, requested_delivery_sqft=60.0)
        total, remaining = state["total_capacity_sqft"], state["remaining_capacity_sqft"]
        delivered_cumulative += state["delivered_sqft"]

        # Step 2: request another 60 (only 40 should remain)
        state = update_capacity_state(total, remaining, theoretical_capacity_sqft=100.0, requested_delivery_sqft=60.0)
        total, remaining = state["total_capacity_sqft"], state["remaining_capacity_sqft"]
        delivered_cumulative += state["delivered_sqft"]

        # Step 3: request again (should deliver 0)
        state = update_capacity_state(total, remaining, theoretical_capacity_sqft=100.0, requested_delivery_sqft=60.0)
        total, remaining = state["total_capacity_sqft"], state["remaining_capacity_sqft"]
        delivered_cumulative += state["delivered_sqft"]

        self.assertAlmostEqual(total, 100.0, places=8)
        self.assertAlmostEqual(remaining, 0.0, places=8)
        self.assertAlmostEqual(delivered_cumulative, 100.0, places=8)
        self.assertLessEqual(delivered_cumulative, total)

    def test_capacity_can_expand_if_theoretical_limit_increases(self):
        # Existing parcel is exhausted at 100 sqft, then revised theoretical capacity is 150
        state = update_capacity_state(
            total_capacity_sqft=100.0,
            remaining_capacity_sqft=0.0,
            theoretical_capacity_sqft=150.0,
            requested_delivery_sqft=80.0,
        )
        self.assertAlmostEqual(state["total_capacity_sqft"], 150.0, places=8)
        self.assertAlmostEqual(state["added_capacity_sqft"], 50.0, places=8)
        self.assertAlmostEqual(state["delivered_sqft"], 50.0, places=8)
        self.assertAlmostEqual(state["remaining_capacity_sqft"], 0.0, places=8)

    def test_negative_requested_delivery_is_clamped_to_zero(self):
        state = update_capacity_state(
            total_capacity_sqft=200.0,
            remaining_capacity_sqft=100.0,
            theoretical_capacity_sqft=200.0,
            requested_delivery_sqft=-25.0,
        )
        self.assertAlmostEqual(state["delivered_sqft"], 0.0, places=8)
        self.assertAlmostEqual(state["remaining_capacity_sqft"], 100.0, places=8)


if __name__ == "__main__":
    unittest.main()

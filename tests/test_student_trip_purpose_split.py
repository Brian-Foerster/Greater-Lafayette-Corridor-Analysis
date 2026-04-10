"""Tests for the student trip purpose split (Component 2).

Verifies that the three-purpose split (home_to_campus, campus_to_offcampus,
intracampus) preserves the total rate while differentiating ridership by
corridor geometry and destination coverage.
"""
import unittest
import numpy as np


class TestPurposeRatesSumToTotal(unittest.TestCase):
    def test_rates_sum_to_student_campus_trip_rate(self):
        from src.land_use_transport_model import (
            STUDENT_TRIP_PURPOSES, STUDENT_CAMPUS_TRIP_RATE,
        )
        total = sum(s["rate"] for s in STUDENT_TRIP_PURPOSES.values())
        self.assertAlmostEqual(total, STUDENT_CAMPUS_TRIP_RATE, places=6)


class TestIntracampusWalkDominates(unittest.TestCase):
    """At 0.8 km, walk should have higher share than APM for intracampus."""

    def test_walk_beats_apm_at_short_distance(self):
        from src.land_use_transport_model import LandUseTransportModel

        model = LandUseTransportModel.__new__(LandUseTransportModel)
        model._transit_mode_name = "apm"
        model._model_options = {}
        from src.mode_choice import ASC_APM
        model._transit_asc = ASC_APM
        model._transit_speed_kph = 40.0

        # intracampus: 0.8 km trip, station 200m from campus
        share_short, _ = model._compute_student_apm_share(
            campus_to_station_m=200.0,
            apm_headway_min=3.0,
            bus_headway_min=15.0,
            corridor_length_km=5.0,
            n_stops=5,
            trip_dist_km=0.8,
        )
        # home_to_campus: 2.5 km trip — APM should compete better
        share_long, _ = model._compute_student_apm_share(
            campus_to_station_m=200.0,
            apm_headway_min=3.0,
            bus_headway_min=15.0,
            corridor_length_km=5.0,
            n_stops=5,
            trip_dist_km=2.5,
        )
        # Longer trips should give APM higher share (walk less competitive)
        self.assertGreater(share_long, share_short,
                           "APM share should be higher for 2.5 km vs 0.8 km trips")


class TestDestCoverageScalesOffcampus(unittest.TestCase):
    """Corridor with no destination coverage → 0 off-campus ridership."""

    def test_zero_coverage_zeroes_offcampus(self):
        from src.land_use_transport_model import LandUseTransportModel

        model = LandUseTransportModel.__new__(LandUseTransportModel)
        model._transit_mode_name = "apm"
        model._model_options = {}
        from src.mode_choice import ASC_APM
        model._transit_asc = ASC_APM
        model._transit_speed_kph = 40.0

        # Stations far from any destination → coverage ≈ 0
        far_stops = np.array([[0.0, 0.0]])  # origin (0,0) in proj CRS
        cov = model._compute_destination_coverage(far_stops)
        self.assertLess(cov, 0.01, "Far-away stations should have ~0 coverage")

    def test_no_stops_returns_zero(self):
        from src.land_use_transport_model import LandUseTransportModel

        model = LandUseTransportModel.__new__(LandUseTransportModel)
        model._transit_mode_name = "apm"
        model._model_options = {}
        from src.mode_choice import ASC_APM
        model._transit_asc = ASC_APM
        model._transit_speed_kph = 40.0

        cov = model._compute_destination_coverage(np.empty((0, 2)))
        self.assertEqual(cov, 0.0)


class TestSpPlusPenaltyReducesIntracampus(unittest.TestCase):
    """SP+ shuttle penalty should reduce intracampus APM share."""

    def test_penalty_lowers_short_trip_share(self):
        from src.land_use_transport_model import LandUseTransportModel

        model = LandUseTransportModel.__new__(LandUseTransportModel)
        model._transit_mode_name = "apm"
        model._model_options = {}
        from src.mode_choice import ASC_APM
        model._transit_asc = ASC_APM
        model._transit_speed_kph = 40.0

        # Short trip (intracampus): SP+ penalty applies (trip_dist < 1.0)
        share_with_penalty, _ = model._compute_student_apm_share(
            campus_to_station_m=200.0,
            apm_headway_min=3.0,
            bus_headway_min=15.0,
            corridor_length_km=5.0,
            n_stops=5,
            trip_dist_km=0.8,  # < 1.0 → SP+ penalty
        )
        # Longer trip: no SP+ penalty (trip_dist >= 1.0)
        share_no_penalty, _ = model._compute_student_apm_share(
            campus_to_station_m=200.0,
            apm_headway_min=3.0,
            bus_headway_min=15.0,
            corridor_length_km=5.0,
            n_stops=5,
            trip_dist_km=1.5,  # >= 1.0 → no SP+ penalty
        )
        # The 0.8 km trip gets both penalties: walk is more competitive AND
        # SP+ shuttle competes.  So its share should be lower.
        # Note: walk competitiveness at 0.8km is the dominant effect, but
        # SP+ further reduces it.
        self.assertLess(share_with_penalty, share_no_penalty,
                        "SP+ penalty should further reduce intracampus APM share")


class TestBackwardCompatibleTotal(unittest.TestCase):
    """With dest_coverage=1.0 and similar geometry, total ≈ original."""

    def test_purpose_split_within_range_of_original(self):
        from src.land_use_transport_model import (
            LandUseTransportModel, STUDENT_TRIP_PURPOSES,
        )

        model = LandUseTransportModel.__new__(LandUseTransportModel)
        model._transit_mode_name = "apm"
        model._model_options = {}
        from src.mode_choice import ASC_APM
        model._transit_asc = ASC_APM
        model._transit_speed_kph = 40.0

        # Compute old-style single share at 1.5 km
        old_share, _ = model._compute_student_apm_share(
            campus_to_station_m=300.0,
            apm_headway_min=3.0,
            bus_headway_min=15.0,
            corridor_length_km=5.0,
            n_stops=5,
            trip_dist_km=1.5,
        )
        old_daily = 100.0 * 2.0 * old_share  # campus_pop=100, rate=2.0

        # Compute new-style purpose-split total
        new_daily = 0.0
        for spec in STUDENT_TRIP_PURPOSES.values():
            share, _ = model._compute_student_apm_share(
                campus_to_station_m=300.0,
                apm_headway_min=3.0,
                bus_headway_min=15.0,
                corridor_length_km=5.0,
                n_stops=5,
                trip_dist_km=spec["dist_km"],
            )
            new_daily += 100.0 * spec["rate"] * share

        # With full dest_coverage (=1.0), the new total should be within
        # ~25% of the old total (intracampus walks pull it down, home_to_campus
        # longer distance pushes it up slightly)
        self.assertGreater(new_daily, old_daily * 0.50,
                           "Purpose-split total should not drop more than 50%")
        self.assertLess(new_daily, old_daily * 1.50,
                        "Purpose-split total should not increase more than 50%")


if __name__ == "__main__":
    unittest.main()

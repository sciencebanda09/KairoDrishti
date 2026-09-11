import numpy as np

from engine.search_rescue.models import AerialDetection, DetectionBand, GeoPoint, SearchCell, SearchMission
from engine.search_rescue.mission import SearchAndRescueMission
from engine.search_rescue.tracking import (
    DEFAULT_POSITION_COVARIANCE, DetectionTracker, Track, associate_detections,
    imm_predict, imm_update, mahalanobis_gate,
)


def detection(name, lat, lon, confidence=.8):
    return AerialDetection(name, GeoPoint(lat, lon), "candidate", confidence, DetectionBand.RGB)


def test_mahalanobis_gating_merges_near_moving_point_and_rejects_far_point():
    tracker = DetectionTracker("nearest_decay")
    tracker.ingest([detection("first", 30.0, 77.0)], 0)
    track = tracker.tracks[0]
    assert mahalanobis_gate(track, detection("near", 30.00008, 77.00005), DEFAULT_POSITION_COVARIANCE)
    assert not mahalanobis_gate(track, detection("far", 30.01, 77.01), DEFAULT_POSITION_COVARIANCE)
    matches, unmatched = associate_detections([track], [detection("near", 30.00008, 77.00005),
                                                         detection("far", 30.01, 77.01)])
    assert len(matches) == 1
    assert len(unmatched) == 1


def test_nearest_decay_is_monotonic_nonnegative_and_last_observation_based():
    tracker = DetectionTracker("nearest_decay")
    tracker.ingest([detection("first", 30.0, 77.0, .9)], 0)
    initial = tracker.tracks[0].confidence
    tracker.advance(2)
    middle = tracker.tracks[0].confidence
    tracker.advance(20)
    final = tracker.tracks[0].confidence
    assert initial >= middle >= final >= 0
    assert tracker.tracks[0].position == GeoPoint(30.0, 77.0)


def test_imm_constant_velocity_path_moves_toward_ground_truth():
    tracker = DetectionTracker("imm")
    truth = []
    for minute in range(6):
        lat = 30.0 + minute * .0004
        lon = 77.0 + minute * .0002
        truth.append((lat, lon))
        tracker.ingest([detection(f"d{minute}", lat, lon)], minute)
    state = tracker.tracks[0].state
    assert abs(state[0] - truth[-1][0]) < .00025
    assert abs(state[1] - truth[-1][1]) < .00025
    np.testing.assert_allclose(sum(tracker.tracks[0].model_probabilities.values()), 1.0, atol=1e-6)


def test_switching_tracking_modes_keeps_field_packet_valid():
    base = SearchMission("tracking", GeoPoint(30.0, 77.0),
                         [SearchCell("A", GeoPoint(30.001, 77.001), 1000)])
    for mode in ("nearest_decay", "imm"):
        mission = SearchAndRescueMission(base, tracking_mode=mode)
        mission.ingest_detections([detection(f"{mode}-1", 30.001, 77.001)])
        packet = mission.field_packet()
        assert packet["tracking_mode"] == mode
        assert packet["detections"][0]["track_id"]
        assert packet["tracks"][0]["tracking_mode"] == mode
        assert packet["offline"] is True

import numpy as np

from engine.search_rescue import (
    AerialDetection, DetectionBand, GeoPoint, SearchAndRescueMission,
    SearchCell, SearchMission, detect_changes, fuse_detections,
)


def test_fuses_cross_band_evidence():
    point = GeoPoint(20.0, 77.0)
    result = fuse_detections([
        AerialDetection("rgb", point, "person", .70, DetectionBand.RGB),
        AerialDetection("ir", GeoPoint(20.0001, 77.0001), "heat source", .75, DetectionBand.THERMAL),
    ])
    assert len(result) == 1
    assert result[0].confidence > .75
    assert "thermal" in result[0].evidence


def test_plan_respects_battery_and_returns_home():
    home = GeoPoint(20.0, 77.0)
    cells = [SearchCell(str(i), GeoPoint(20.001 * i, 77.0), 10000, movement_score=1)
             for i in range(1, 20)]
    packet = SearchAndRescueMission(SearchMission("m1", home, cells, battery_minutes=3)).field_packet()
    assert packet["offline"] is True
    assert packet["waypoints"][-1]["kind"] == "return_home"


def test_change_detection_finds_new_signal():
    before = np.zeros((4, 4), dtype=np.float32)
    after = before.copy()
    after[2, 3] = .9
    mask = detect_changes(before, after)
    assert int(mask.sum()) == 1


def test_change_detection_normalizes_uint8_and_raw_thermal_ranges():
    before = np.zeros((4, 4), dtype=np.uint8)
    after = before.copy()
    after[2, 3] = 255
    assert int(detect_changes(before, after).sum()) == 1

    thermal_before = np.full((4, 4), 1000, dtype=np.uint16)
    thermal_after = thermal_before.copy()
    thermal_after[2, 3] = 5000
    assert int(detect_changes(thermal_before, thermal_after).sum()) == 1

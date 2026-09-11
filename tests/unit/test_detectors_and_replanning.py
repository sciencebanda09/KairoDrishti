import numpy as np

from engine.search_rescue import (
    AerialDetection, DetectionBand, FrameMetadata, GeoPoint, SearchAndRescueMission,
    SearchCell, SearchMission, detect_rgb, detect_thermal,
)
from engine.search_rescue.detectors import detect_change, detect_multispectral
from engine.search_rescue.coverage import plan_coverage


def test_rgb_and_thermal_detectors_emit_geolocated_candidates():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[27:37, 29:35] = 255
    thermal = np.full((64, 64), 20, dtype=np.float32)
    thermal[27:37, 29:35] = 80
    metadata = FrameMetadata("frame-001", GeoPoint(30.0, 77.0, 120), ground_sample_distance_m=.5)
    rgb_result = detect_rgb(rgb, metadata)
    thermal_result = detect_thermal(thermal, metadata)
    assert rgb_result and rgb_result[0].band is DetectionBand.RGB
    assert thermal_result and thermal_result[0].band is DetectionBand.THERMAL
    assert thermal_result[0].location.lat != metadata.location.lat or thermal_result[0].location.lon != metadata.location.lon


def test_multispectral_and_change_detectors_emit_signals():
    metadata = FrameMetadata("frame-002", GeoPoint(30.0, 77.0, 120), ground_sample_distance_m=.5)
    multi = np.ones((32, 32, 4), dtype=np.float32)
    multi[12:20, 14:19, -1] = 8
    before = np.zeros((32, 32), dtype=np.float32)
    after = before.copy(); after[10:18, 15:22] = 1
    assert detect_multispectral(multi, metadata)
    assert detect_change(before, after, metadata)


def test_detection_inserts_investigation_before_return_home():
    home = GeoPoint(30.0, 77.0, 1200)
    mission = SearchAndRescueMission(SearchMission(
        "m1", home, [SearchCell("A", GeoPoint(30.001, 77.001), 10_000)], battery_minutes=10,
    ))
    detection = AerialDetection("d1", GeoPoint(30.001, 77.001), "person", .8, DetectionBand.RGB)
    route = mission.replan([detection])
    assert route[0].kind.value == "investigate"
    assert route[-1].kind.value == "return_home"
    assert route[0].location == detection.location
    assert mission.route_version == 1


def test_coverage_skips_cell_inside_airspace_polygon():
    home = GeoPoint(30.0, 77.0)
    blocked = GeoPoint(30.001, 77.001)
    polygon = [GeoPoint(29.999, 76.999), GeoPoint(30.002, 76.999),
               GeoPoint(30.002, 77.002), GeoPoint(29.999, 77.002)]
    route = plan_coverage([SearchCell("blocked", blocked, 1000)], home,
                          battery_minutes=10, airspace_polygon=polygon)
    assert route[-1].kind.value == "return_home"
    assert all(point.cell_id != "blocked" for point in route)

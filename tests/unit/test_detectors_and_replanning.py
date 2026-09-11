import numpy as np

from engine.search_rescue import (
    AerialDetection, DetectionBand, FrameMetadata, GeoPoint, SearchAndRescueMission,
    SearchCell, SearchMission, detect_rgb, detect_thermal,
)
from engine.search_rescue.detectors import detect_change, detect_multispectral
from engine.search_rescue.geolocation import geolocate_pixel
from engine.search_rescue.astar import plan_3d_detour
from engine.search_rescue.terrain import ElevationGrid
from engine.search_rescue.coverage import plan_coverage


def test_rgb_and_thermal_detectors_emit_geolocated_candidates():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[27:37, 29:35] = 255
    thermal = np.full((64, 64), 20, dtype=np.float32)
    thermal[27:37, 29:35] = 80
    metadata = FrameMetadata("frame-001", GeoPoint(30.0, 77.0, 120), ground_sample_distance_m=.5, image_width=64, image_height=64)
    rgb_result = detect_rgb(rgb, metadata)
    thermal_result = detect_thermal(thermal, metadata)
    assert rgb_result and rgb_result[0].band is DetectionBand.RGB
    assert thermal_result and thermal_result[0].band is DetectionBand.THERMAL
    assert thermal_result[0].location.lat != metadata.location.lat or thermal_result[0].location.lon != metadata.location.lon


def test_geolocation_uses_full_frame_center_not_box_dimensions():
    metadata = FrameMetadata("frame-geo", GeoPoint(30.0, 77.0), ground_sample_distance_m=1.0,
                             image_width=1000, image_height=1000)
    center = geolocate_pixel(500, 500, 1000, 1000, metadata)
    corner = geolocate_pixel(0, 0, 1000, 1000, metadata)
    assert abs(center.lat - metadata.location.lat) < 1e-9
    assert abs(center.lon - metadata.location.lon) < 1e-9
    assert corner.lat > metadata.location.lat
    assert corner.lon < metadata.location.lon


def test_multispectral_and_change_detectors_emit_signals():
    metadata = FrameMetadata("frame-002", GeoPoint(30.0, 77.0, 120), ground_sample_distance_m=.5, image_width=32, image_height=32)
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


def test_investigation_is_not_reissued_and_nearby_target_can_supersede():
    home = GeoPoint(30.0, 77.0, 1200)
    mission = SearchAndRescueMission(SearchMission("m1", home, [], battery_minutes=10))
    far = AerialDetection("far", GeoPoint(30.008, 77.008), "candidate", .90, DetectionBand.RGB)
    near = AerialDetection("near", GeoPoint(30.0001, 77.0001), "candidate", .70, DetectionBand.RGB)
    first = mission.replan([far])
    assert first[0].location == far.location
    second = mission.replan([near])
    assert second[0].location == near.location
    mission.complete_investigation("near")
    third = mission.replan()
    assert third[0].kind.value == "investigate"
    assert third[0].location == far.location
    mission.complete_investigation("far")
    assert all(point.kind.value != "investigate" for point in mission.replan())


def test_coverage_skips_cell_inside_airspace_polygon():
    home = GeoPoint(30.0, 77.0)
    blocked = GeoPoint(30.001, 77.001)
    polygon = [GeoPoint(29.999, 76.999), GeoPoint(30.002, 76.999),
               GeoPoint(30.002, 77.002), GeoPoint(29.999, 77.002)]
    route = plan_coverage([SearchCell("blocked", blocked, 1000)], home,
                          battery_minutes=10, airspace_polygon=polygon)
    assert route[-1].kind.value == "return_home"
    assert all(point.cell_id != "blocked" for point in route)


def test_dem_voxel_planner_returns_altitude_aware_path():
    dem = ElevationGrid(np.full((101, 101), 100, dtype=np.float32), 30.0, 77.0, 36)
    route = plan_3d_detour(GeoPoint(30.01, 77.01, 220), GeoPoint(30.015, 77.015, 220), dem,
                           min_altitude_m=80, max_altitude_m=300, step_m=40)
    assert route
    assert route[-1].location.lat > route[0].location.lat
    assert all(point.location.altitude_m >= 80 for point in route)

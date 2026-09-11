from engine.search_rescue import (
    AerialDetection, DetectionBand, Drone, DroneStatus, GeoPoint, SearchCell,
    SearchMission, SwarmMission, SwarmSearchAndRescueMission,
    RESERVE_MIN_BATTERY_PERCENT, assign_sectors, reassign_on_candidate,
    reassign_on_failure,
)


def _cells():
    return [
        SearchCell("A", GeoPoint(30.000, 77.000), 1000, required_sensor_bands={DetectionBand.RGB}),
        SearchCell("B", GeoPoint(30.010, 77.010), 1000, required_sensor_bands={DetectionBand.THERMAL}),
    ]


def _drones():
    return [
        Drone("low", GeoPoint(30.000, 77.000), battery_percent=20,
              sensor_bands={DetectionBand.RGB}, status=DroneStatus.SEARCHING),
        Drone("thermal", GeoPoint(30.010, 77.010), battery_percent=75,
              sensor_bands={DetectionBand.THERMAL}, status=DroneStatus.SEARCHING),
        Drone("reserve", GeoPoint(30.005, 77.005), battery_percent=95,
              sensor_bands={DetectionBand.RGB, DetectionBand.THERMAL}, status=DroneStatus.SEARCHING),
    ]


def test_assignment_respects_battery_sensor_and_best_battery_reserve():
    drones, cells = _drones(), _cells()
    assignments = assign_sectors(drones, cells, GeoPoint(30.0, 77.0))
    assert set(assignments) == {"A", "B"}
    assert assignments["B"] == "thermal"
    assert "reserve" not in assignments
    assert next(drone for drone in drones if drone.drone_id == "reserve").battery_percent >= RESERVE_MIN_BATTERY_PERCENT


def test_failure_reassigns_failed_sector():
    drones, cells = _drones(), _cells()
    assignments = assign_sectors(drones, cells, GeoPoint(30.0, 77.0))
    failed = next(drone for drone in drones if drone.drone_id == assignments["B"])
    failed.status = DroneStatus.OFFLINE
    updated = reassign_on_failure(assignments, drones, cells)
    assert updated.get("B") != failed.drone_id
    assert all(owner != failed.drone_id for owner in updated.values())


def test_candidate_handoff_reassigns_investigator_sector():
    drones = [
        Drone("near", GeoPoint(30.000, 77.000), battery_percent=90, status=DroneStatus.SEARCHING),
        Drone("far", GeoPoint(30.010, 77.010), battery_percent=80, status=DroneStatus.SEARCHING),
        Drone("free", GeoPoint(30.020, 77.020), battery_percent=70, status=DroneStatus.IDLE),
    ]
    assignments = {"A": "near", "B": "far"}
    updated, investigator = reassign_on_candidate(assignments, drones, GeoPoint(30.0001, 77.0001))
    assert investigator == "near"
    assert updated["A"] in {"far", "free"}
    assert drones[0].status is DroneStatus.INVESTIGATING


def test_field_packet_schema_and_ownership_are_explicit():
    mission = SwarmSearchAndRescueMission(SwarmMission(
        SearchMission("swarm-1", GeoPoint(30.0, 77.0), _cells()), _drones()))
    packet = mission.field_packet()
    assert packet["offline"] is True
    assert packet["mode"] == "swarm"
    assert packet["simulation_only"] is True
    assert "SIMULATION / PLANNING ONLY" in packet["framing"]
    assert set(packet) >= {"mission_id", "drones", "sectors", "detections", "waypoints", "swarm_coverage_fraction"}
    drone_ids = {drone["drone_id"] for drone in packet["drones"]}
    for sector in packet["sectors"]:
        assert sector["owning_drone_id"] in drone_ids or sector["owner_status"] in {"idle", "offline", "IDLE", "OFFLINE"}
    assert set(packet["waypoints"]) == drone_ids


def test_candidate_threshold_triggers_assignment_change():
    drones = [
        Drone("one", GeoPoint(30.0, 77.0), battery_percent=80, status=DroneStatus.SEARCHING),
        Drone("two", GeoPoint(30.01, 77.01), battery_percent=80, status=DroneStatus.SEARCHING),
    ]
    mission = SwarmSearchAndRescueMission(SwarmMission(
        SearchMission("swarm-2", GeoPoint(30.0, 77.0), _cells()), drones))
    before = dict(mission.swarm_mission.sector_assignments)
    mission.ingest_detections([AerialDetection("d", GeoPoint(30.0, 77.0), "person", .7)])
    assert before != mission.swarm_mission.sector_assignments or any(
        drone.status is DroneStatus.INVESTIGATING for drone in drones)

from math import hypot

from engine.search_rescue.flocking import flock_step
from engine.search_rescue.models import Drone, DroneStatus, GeoPoint, SearchWaypoint, WaypointKind


def _distance(a, b):
    return hypot((a.position.lat - b.position.lat) * 111_000,
                 (a.position.lon - b.position.lon) * 111_000)


def test_drones_maintain_spacing_during_simulated_ticks():
    drones = [
        Drone("D1", GeoPoint(30.0, 77.0), status=DroneStatus.SEARCHING),
        Drone("D2", GeoPoint(30.0, 77.0), status=DroneStatus.SEARCHING),
    ]
    target = GeoPoint(30.004, 77.004)
    for _ in range(80):
        flock_step(drones, {drone.drone_id: SearchWaypoint(1, target, WaypointKind.SEARCH)
                            for drone in drones}, .1,
                   separation_weight=3.0, alignment_weight=.25,
                   cohesion_weight=.2, goal_weight=7.0, radius_m=35)
    assert _distance(drones[0], drones[1]) > 8.0


def test_goal_weight_converges_each_drone_to_its_own_waypoint():
    drones = [
        Drone("D1", GeoPoint(30.0, 77.0), status=DroneStatus.SEARCHING),
        Drone("D2", GeoPoint(30.0, 77.002), status=DroneStatus.SEARCHING),
    ]
    targets = {
        "D1": GeoPoint(30.0015, 77.0),
        "D2": GeoPoint(30.0015, 77.002),
    }
    for _ in range(180):
        flock_step(drones, {key: SearchWaypoint(1, value, WaypointKind.SEARCH)
                            for key, value in targets.items()}, .1,
                   separation_weight=.5, alignment_weight=.2,
                   cohesion_weight=.2, goal_weight=12.0)
    assert abs(drones[0].position.lat - targets["D1"].lat) < .00035
    assert abs(drones[1].position.lat - targets["D2"].lat) < .00035


def test_no_fly_polygon_clamps_motion():
    drones = [Drone("D1", GeoPoint(29.999, 76.999), status=DroneStatus.SEARCHING)]
    polygon = [GeoPoint(29.9995, 76.9995), GeoPoint(30.0005, 76.9995),
               GeoPoint(30.0005, 77.0005), GeoPoint(29.9995, 77.0005)]
    flock_step(drones, {"D1": GeoPoint(30.002, 77.002)}, 1.0,
               separation_weight=0, alignment_weight=0,
               cohesion_weight=0, goal_weight=10, no_fly_polygon=polygon)
    assert not (polygon[0].lat < drones[0].position.lat < polygon[2].lat and
                polygon[0].lon < drones[0].position.lon < polygon[2].lon)

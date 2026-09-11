"""Reynolds-style motion simulation for coordinated drone replay only.

This layer changes simulated in-flight positions, not sector assignments or
route planning.  KairoDrishti does not control real aircraft.
"""
from __future__ import annotations

from math import cos, hypot, pi, radians, sin

from .coverage import _inside, _intersects
from .models import Drone, GeoPoint, SearchWaypoint


EARTH_METRES_PER_DEGREE = 111_000.0


def _local_delta(origin: GeoPoint, point: GeoPoint) -> tuple[float, float]:
    return ((point.lon - origin.lon) * EARTH_METRES_PER_DEGREE * max(cos(radians(origin.lat)), .1),
            (point.lat - origin.lat) * EARTH_METRES_PER_DEGREE)


def _vector_to_point(origin: GeoPoint, east: float, north: float) -> GeoPoint:
    return GeoPoint(
        origin.lat + north / EARTH_METRES_PER_DEGREE,
        origin.lon + east / (EARTH_METRES_PER_DEGREE * max(cos(radians(origin.lat)), .1)),
        origin.altitude_m,
    )


def _nearby(drone: Drone, neighbors: list[Drone], radius_m: float) -> list[tuple[Drone, float, float]]:
    result = []
    for neighbor in neighbors:
        if neighbor is drone or neighbor.drone_id == drone.drone_id:
            continue
        east, north = _local_delta(drone.position, neighbor.position)
        distance = hypot(east, north)
        if distance <= radius_m:
            result.append((neighbor, east, north))
    return result


def separation(drone: Drone, neighbors: list[Drone], radius_m: float) -> tuple[float, float]:
    """Return a repulsion vector away from nearby simulated aircraft."""
    if radius_m <= 0:
        return 0.0, 0.0
    dx = dy = 0.0
    for neighbor, east, north in _nearby(drone, neighbors, radius_m):
        distance = hypot(east, north)
        if distance < 1e-9:
            # Stable, deterministic nudge for coincident markers.
            angle = (sum(ord(char) for char in drone.drone_id) % 360) * pi / 180.0
            east, north, distance = cos(angle), sin(angle), 1.0
        strength = (radius_m - distance) / radius_m
        dx -= east / distance * strength
        dy -= north / distance * strength
    return dx, dy


def alignment(drone: Drone, neighbors: list[Drone], radius_m: float) -> tuple[float, float]:
    """Return the local average velocity difference."""
    nearby = _nearby(drone, neighbors, radius_m)
    if not nearby:
        return 0.0, 0.0
    average_east = sum(item[0].velocity[0] for item in nearby) / len(nearby)
    average_north = sum(item[0].velocity[1] for item in nearby) / len(nearby)
    return average_east - drone.velocity[0], average_north - drone.velocity[1]


def cohesion(drone: Drone, neighbors: list[Drone], radius_m: float) -> tuple[float, float]:
    """Return a vector from the drone to the local flock center."""
    nearby = _nearby(drone, neighbors, radius_m)
    if not nearby:
        return 0.0, 0.0
    east = sum(item[1] for item in nearby) / len(nearby)
    north = sum(item[2] for item in nearby) / len(nearby)
    return east, north


def _target_point(target: SearchWaypoint | GeoPoint | None) -> GeoPoint | None:
    if target is None:
        return None
    return target.location if isinstance(target, SearchWaypoint) else target


def _blocked(start: GeoPoint, end: GeoPoint, polygon: list[GeoPoint] | None) -> bool:
    if not polygon:
        return False
    if _inside(end, polygon):
        return True
    return any(_intersects(start, end, left, right)
               for left, right in zip(polygon, polygon[1:] + polygon[:1]))


def flock_step(
    drones: list[Drone],
    target_waypoints: dict[str, SearchWaypoint | GeoPoint | None],
    dt_seconds: float,
    *,
    separation_weight: float,
    alignment_weight: float,
    cohesion_weight: float,
    goal_weight: float,
    radius_m: float = 45.0,
    no_fly_polygon: list[GeoPoint] | None = None,
    airspace_polygon: list[GeoPoint] | None = None,
) -> list[Drone]:
    """Advance simulated positions by one tick while preserving route goals.

    ``goal_weight`` is intentionally required to dominate the social terms in
    callers.  The function mutates and returns the supplied drone objects for
    a low-allocation animation loop.
    """
    polygon = no_fly_polygon if no_fly_polygon is not None else airspace_polygon
    dt = max(0.0, float(dt_seconds))
    if dt == 0.0:
        return drones
    max_speed = 20.0
    next_states: list[tuple[Drone, tuple[float, float], GeoPoint]] = []
    for drone in drones:
        target = _target_point(target_waypoints.get(drone.drone_id))
        if target is None or drone.status.value in {"offline", "returning"}:
            goal = (0.0, 0.0)
        else:
            goal_east, goal_north = _local_delta(drone.position, target)
            distance = hypot(goal_east, goal_north)
            if distance < 1e-6:
                goal = (0.0, 0.0)
            else:
                goal_speed = min(max_speed, distance / dt)
                goal = (goal_east / distance * goal_speed,
                        goal_north / distance * goal_speed)
        sep = separation(drone, drones, radius_m)
        sep = (sep[0] * max_speed, sep[1] * max_speed)
        ali = alignment(drone, drones, radius_m)
        coh = cohesion(drone, drones, radius_m)
        coh_distance = hypot(*coh)
        if coh_distance > max_speed:
            coh = (coh[0] / coh_distance * max_speed,
                   coh[1] / coh_distance * max_speed)
        total = max(goal_weight + separation_weight + alignment_weight + cohesion_weight, 1e-9)
        desired = (
            (goal[0] * goal_weight + sep[0] * separation_weight + ali[0] * alignment_weight + coh[0] * cohesion_weight) / total,
            (goal[1] * goal_weight + sep[1] * separation_weight + ali[1] * alignment_weight + coh[1] * cohesion_weight) / total,
        )
        smoothing = min(1.0, dt * 3.0)
        velocity = (
            drone.velocity[0] + (desired[0] - drone.velocity[0]) * smoothing,
            drone.velocity[1] + (desired[1] - drone.velocity[1]) * smoothing,
        )
        speed = hypot(*velocity)
        if speed > max_speed:
            velocity = (velocity[0] / speed * max_speed, velocity[1] / speed * max_speed)
        proposed = _vector_to_point(drone.position, velocity[0] * dt, velocity[1] * dt)
        if _blocked(drone.position, proposed, polygon):
            # Clamp components independently.  If the combined position is
            # still blocked, stop both components at the safe current point.
            east_only = _vector_to_point(drone.position, velocity[0] * dt, 0.0)
            north_only = _vector_to_point(drone.position, 0.0, velocity[1] * dt)
            if _blocked(drone.position, east_only, polygon):
                velocity = (0.0, velocity[1])
            if _blocked(drone.position, north_only, polygon):
                velocity = (velocity[0], 0.0)
            proposed = _vector_to_point(drone.position, velocity[0] * dt, velocity[1] * dt)
            if _blocked(drone.position, proposed, polygon):
                velocity = (0.0, 0.0)
                proposed = drone.position
        next_states.append((drone, velocity, proposed))
    for drone, velocity, proposed in next_states:
        drone.velocity = velocity
        drone.position = proposed
    return drones

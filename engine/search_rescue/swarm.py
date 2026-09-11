"""Deterministic multi-drone sector planning for simulation / planning only.

KairoDrishti does not control real aircraft and does not infer sensor-fused
ground truth.  The functions in this module assign and replay simulated
aircraft for human-led search planning.
"""
from __future__ import annotations

from math import hypot

import numpy as np
from scipy.optimize import linear_sum_assignment

from .coverage import plan_coverage
from .models import (DetectionBand, Drone, DroneStatus, GeoPoint, SearchCell,
                     SearchWaypoint, SwarmMission, WaypointKind)


# A reserve needs enough simulated endurance to investigate a candidate or
# replace a failed aircraft.  It is not selected because it is the weakest
# asset; the best-battery unassigned aircraft is deliberately held back.
RESERVE_MIN_BATTERY_PERCENT = 35.0


def _distance_m(a: GeoPoint, b: GeoPoint) -> float:
    return hypot((a.lat - b.lat) * 111_000,
                 (a.lon - b.lon) * 111_000)


def _online(drone: Drone) -> bool:
    return drone.status is not DroneStatus.OFFLINE and drone.communication_ok


def _available(drone: Drone) -> bool:
    return (_online(drone) and drone.status in {DroneStatus.SEARCHING, DroneStatus.IDLE}
            and drone.battery_percent > 0.0)


def _required_bands(cell: SearchCell) -> set[DetectionBand]:
    requirements = getattr(cell, "required_sensor_bands", None)
    if requirements:
        return set(requirements)
    # Accept the natural alternate spelling used by integrations without
    # changing SearchCell's public field contract.
    return set(getattr(cell, "sensor_bands", set()) or set())


def _cost(drone: Drone, cell: SearchCell, home: GeoPoint) -> float:
    distance_cost = _distance_m(drone.position, cell.center) / 1_000.0
    # Low battery is a soft cost plus a strong guardrail when another asset
    # can safely cover the cell.
    battery_cost = max(0.0, (100.0 - drone.battery_percent) / 100.0) * 2.0
    if drone.battery_percent < RESERVE_MIN_BATTERY_PERCENT:
        battery_cost += 50.0
    required = _required_bands(cell)
    missing = len(required - set(drone.sensor_bands))
    sensor_cost = missing * 12.0
    # Home is intentionally a small tie-breaker: shorter returns are safer,
    # while sector distance remains the dominant planning term.
    return distance_cost + battery_cost + sensor_cost + _distance_m(drone.position, home) / 50_000.0


def _assign(drones: list[Drone], cells: list[SearchCell], home: GeoPoint,
            *, hold_reserve: bool) -> dict[str, str]:
    candidates = [drone for drone in drones if _available(drone)]
    for drone in drones:
        if drone.status is not DroneStatus.INVESTIGATING:
            drone.assigned_sector_id = None
    if not candidates or not cells:
        return {}

    reserve: Drone | None = None
    if hold_reserve and len(candidates) > len(cells):
        # Keep the strongest endurance asset free for investigation and
        # failure recovery.  Reserve is a capability choice, not a penalty
        # box for the weakest aircraft.
        reserve = max(candidates, key=lambda item: item.battery_percent)
        candidates = [drone for drone in candidates if drone is not reserve]
        reserve.assigned_sector_id = None
        if reserve.status is DroneStatus.SEARCHING:
            reserve.status = DroneStatus.IDLE

    if not candidates:
        return {}

    # Prefer drones above the reserve endurance floor if there are enough of
    # them to cover the sectors; otherwise retain every available asset so a
    # small mission never loses a sector merely because batteries are low.
    robust = [drone for drone in candidates
              if drone.battery_percent >= RESERVE_MIN_BATTERY_PERCENT]
    if len(robust) >= min(len(candidates), len(cells)):
        candidates = robust

    matrix = np.asarray([[_cost(drone, cell, home) for cell in cells]
                         for drone in candidates], dtype=float)
    rows, columns = linear_sum_assignment(matrix)
    assignments: dict[str, str] = {}
    for row, column in zip(rows, columns):
        drone, cell = candidates[int(row)], cells[int(column)]
        assignments[cell.cell_id] = drone.drone_id
        drone.assigned_sector_id = cell.cell_id
        if drone.status is DroneStatus.IDLE:
            drone.status = DroneStatus.SEARCHING
    return assignments


def assign_sectors(drones: list[Drone], cells: list[SearchCell], home: GeoPoint) -> dict[str, str]:
    """Assign sectors with Hungarian distance/battery/sensor costs.

    Offline or disconnected drones are excluded.  The returned mapping is
    ``sector_id -> drone_id`` and is deterministic for equal inputs.
    """
    return _assign(drones, cells, home, hold_reserve=True)


def reassign_on_failure(assignments: dict[str, str], drones: list[Drone],
                        cells: list[SearchCell]) -> dict[str, str]:
    """Recompute ownership after offline or disconnected aircraft failure."""
    healthy = [drone for drone in drones if _available(drone)]
    if not healthy:
        for drone in drones:
            if drone.status is not DroneStatus.OFFLINE:
                drone.assigned_sector_id = None
        return {}
    home = healthy[0].position
    # A failed owner cannot retain any sector.  Re-running the same cost
    # function over all cells also repairs stale mappings from a prior handoff.
    return _assign(healthy, cells, home, hold_reserve=len(healthy) > len(cells))


def reassign_on_candidate(assignments: dict[str, str], drones: list[Drone],
                          candidate_location: GeoPoint) -> tuple[dict[str, str], str | None]:
    """Give the candidate to the nearest free aircraft and orphan safely.

    The function has no cell geometry by design, so an orphan is assigned to
    the least-loaded healthy aircraft.  The next full planning pass can then
    refine that ownership using sector-center costs.
    """
    available = [drone for drone in drones if _available(drone)]
    if not available:
        return dict(assignments), None
    investigator = min(available, key=lambda drone: _distance_m(drone.position, candidate_location))
    orphaned = [sector for sector, owner in list(assignments.items())
                if owner == investigator.drone_id]
    updated = dict(assignments)
    for sector in orphaned:
        updated.pop(sector, None)
    investigator.assigned_sector_id = None
    investigator.status = DroneStatus.INVESTIGATING

    for sector in orphaned:
        replacements = [drone for drone in drones
                        if _available(drone) and drone is not investigator]
        if not replacements:
            break
        load = {drone.drone_id: sum(owner == drone.drone_id
                                    for owner in updated.values())
                for drone in replacements}
        replacement = min(replacements,
                          key=lambda drone: (load[drone.drone_id],
                                             _distance_m(drone.position, investigator.position)))
        updated[sector] = replacement.drone_id
        if replacement.assigned_sector_id is None:
            replacement.assigned_sector_id = sector
    return updated, investigator.drone_id


def plan_swarm_routes(swarm_mission: SwarmMission) -> dict[str, list[SearchWaypoint]]:
    """Plan one battery-aware coverage route per assigned simulated aircraft."""
    cells_by_id = {cell.cell_id: cell for cell in swarm_mission.mission.cells}
    routes: dict[str, list[SearchWaypoint]] = {}
    for drone in swarm_mission.drones:
        if drone.status is DroneStatus.OFFLINE or not drone.communication_ok:
            routes[drone.drone_id] = [SearchWaypoint(
                1, drone.position, WaypointKind.RETURN_HOME,
                rationale="offline/disconnected; simulation recovery hold")]
            continue
        owned = [cells_by_id[cell_id] for cell_id, owner in swarm_mission.sector_assignments.items()
                 if owner == drone.drone_id and cell_id in cells_by_id]
        budget = swarm_mission.mission.battery_minutes * max(drone.battery_percent, 0.0) / 100.0
        routes[drone.drone_id] = plan_coverage(
            owned, drone.position, battery_minutes=budget,
            airspace_polygon=swarm_mission.mission.airspace_polygon)
    return routes


def swarm_coverage_fraction(swarm_mission: SwarmMission) -> float:
    """Aggregate cell coverage once, even after ownership handoffs."""
    cells = {cell.cell_id: cell for cell in swarm_mission.mission.cells}
    total_area = sum(max(cell.area_m2, 0.0) for cell in cells.values())
    if total_area <= 0:
        return 0.0
    covered = sum(max(0.0, min(1.0, cell.covered_fraction)) * max(cell.area_m2, 0.0)
                   for cell in cells.values())
    return round(covered / total_area, 4)

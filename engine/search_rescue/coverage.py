from __future__ import annotations

from math import hypot

from .models import GeoPoint, SearchCell, SearchWaypoint, WaypointKind


def _distance_m(a: GeoPoint, b: GeoPoint) -> float:
    return hypot((a.lat - b.lat) * 111_000, (a.lon - b.lon) * 111_000)


def _inside(point: GeoPoint, polygon: list[GeoPoint]) -> bool:
    inside = False
    for left, right in zip(polygon, polygon[1:] + polygon[:1]):
        if (left.lon > point.lon) != (right.lon > point.lon):
            cross = (right.lat - left.lat) * (point.lon - left.lon) / (right.lon - left.lon) + left.lat
            if point.lat < cross:
                inside = not inside
    return inside


def _orientation(a: GeoPoint, b: GeoPoint, c: GeoPoint) -> float:
    return (b.lon - a.lon) * (c.lat - a.lat) - (b.lat - a.lat) * (c.lon - a.lon)


def _intersects(a: GeoPoint, b: GeoPoint, c: GeoPoint, d: GeoPoint) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    return ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0))


def _safe_leg(start: GeoPoint, end: GeoPoint, polygon: list[GeoPoint] | None) -> bool:
    if not polygon:
        return True
    if _inside(start, polygon) or _inside(end, polygon):
        return False
    return not any(_intersects(start, end, left, right)
                   for left, right in zip(polygon, polygon[1:] + polygon[:1]))


def plan_coverage(
    cells: list[SearchCell], home: GeoPoint, *, battery_minutes: float,
    cruise_mps: float = 12.0, dwell_seconds: int = 8,
    airspace_polygon: list[GeoPoint] | None = None,
) -> list[SearchWaypoint]:
    """Greedy priority route with an explicit safe-return reserve."""
    if battery_minutes <= 0 or cruise_mps <= 0:
        return [SearchWaypoint(1, home, WaypointKind.RETURN_HOME,
                               rationale="invalid flight budget")]
    budget_m = battery_minutes * 60 * cruise_mps
    reserve_m = max(500.0, budget_m * 0.18)
    route: list[SearchWaypoint] = []
    current = home
    spent = 0.0
    for cell in sorted(cells, key=lambda item: item.priority):
        if not _safe_leg(current, cell.center, airspace_polygon):
            continue
        leg = _distance_m(current, cell.center)
        return_leg = _distance_m(cell.center, home)
        task_cost = dwell_seconds * cruise_mps
        if spent + leg + task_cost + return_leg + reserve_m > budget_m:
            continue
        route.append(SearchWaypoint(len(route) + 1, cell.center,
                                    WaypointKind.SEARCH, cell.cell_id,
                                    dwell_seconds,
                                    f"priority rank {cell.priority}; likelihood {cell.likelihood:.0%}"))
        spent += leg + task_cost
        current = cell.center
    return_rationale = f"return reserve {reserve_m / cruise_mps / 60:.1f} min"
    if not _safe_leg(current, home, airspace_polygon):
        return_rationale += "; direct return intersects no-fly polygon"
    route.append(SearchWaypoint(len(route) + 1, home, WaypointKind.RETURN_HOME,
                                rationale=return_rationale))
    return route

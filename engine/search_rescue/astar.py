"""Bounded 3D voxel A* planner for DEM terrain and extruded no-fly polygons."""
from __future__ import annotations

import heapq
from math import hypot

from .models import GeoPoint, SearchWaypoint, WaypointKind
from .terrain import ElevationGrid


def plan_3d_detour(start: GeoPoint, goal: GeoPoint, dem: ElevationGrid,
                   *, clearance_m: float = 35.0, min_altitude_m: float = 80.0,
                   max_altitude_m: float = 260.0, step_m: float = 40.0,
                   no_fly: list[tuple[float, float, float, float, float, float]] | None = None) -> list[SearchWaypoint]:
    """Return a terrain-clear route; no_fly boxes are xmin,xmax,ymin,ymax,zmin,zmax in metres."""
    def xyz(point: GeoPoint) -> tuple[float, float, float]:
        return ((point.lon - start.lon) * 111_000, point.altitude_m, (point.lat - start.lat) * 111_000)
    sx, sy, sz = xyz(start); gx, gy, gz = xyz(goal)
    def key(x: float, y: float, z: float) -> tuple[int, int, int]:
        return round(x / step_m), round(y / step_m), round(z / step_m)
    start_key, goal_key = key(sx, max(sy, min_altitude_m), sz), key(gx, max(gy, min_altitude_m), gz)
    no_fly = no_fly or []
    def point(k: tuple[int, int, int]) -> tuple[float, float, float]: return k[0]*step_m, k[1]*step_m, k[2]*step_m
    def blocked(k: tuple[int, int, int]) -> bool:
        x, y, z = point(k); lat, lon = start.lat + z/111_000, start.lon + x/(111_000)
        try: terrain = dem.sample(lat, lon)
        except ValueError: return True
        if y < terrain - start.altitude_m + clearance_m: return True
        if y < min_altitude_m or y > max_altitude_m: return True
        return any(a <= x <= b and c <= y <= d and e <= z <= f for a,b,c,d,e,f in no_fly)
    if blocked(start_key) or blocked(goal_key):
        raise ValueError("start or goal is blocked by terrain/airspace")
    frontier = [(0.0, start_key)]; came: dict[tuple[int,int,int], tuple[int,int,int]] = {}; cost = {start_key: 0.0}
    moves = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1) if (dx,dy,dz)!=(0,0,0)]
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal_key: break
        for move in moves:
            nxt = tuple(current[i] + move[i] for i in range(3))
            if blocked(nxt): continue
            step_cost = hypot(hypot(move[0], move[1]), move[2])
            new_cost = cost[current] + step_cost
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost; came[nxt] = current
                h = hypot(hypot(nxt[0]-goal_key[0], nxt[1]-goal_key[1]), nxt[2]-goal_key[2])
                heapq.heappush(frontier, (new_cost + h, nxt))
    else:
        raise ValueError("no terrain-following path exists")
    path = [goal_key]
    while path[-1] != start_key: path.append(came[path[-1]])
    path.reverse()
    route = []
    for index, k in enumerate(path, 1):
        x, y, z = point(k)
        route.append(SearchWaypoint(index, GeoPoint(start.lat + z/111_000, start.lon + x/111_000, start.altitude_m + y), WaypointKind.SEARCH, rationale="3D DEM A* terrain-following detour"))
    return route

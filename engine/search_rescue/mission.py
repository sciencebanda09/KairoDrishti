from __future__ import annotations

from .coverage import plan_coverage
from .detection import fuse_detections
from .models import AerialDetection, GeoPoint, SearchMission, SearchWaypoint, WaypointKind
from .prioritization import rank_search_cells


class SearchAndRescueMission:
    """Offline mission coordinator: rank, fuse, and emit field waypoints."""

    def __init__(self, mission: SearchMission) -> None:
        self.mission = mission
        self.detections: list[AerialDetection] = []
        self.current_position = mission.last_known_position
        self.elapsed_minutes = 0.0
        self.completed_cell_ids: set[str] = set()
        self.route_version = 0
        self.last_route: list[SearchWaypoint] = []
        self.battery_percent = 100.0
        self.coverage_fraction = 0.0

    def ingest_detections(self, detections: list[AerialDetection]) -> list[AerialDetection]:
        self.detections = fuse_detections(self.detections + detections)
        return self.detections

    def update_telemetry(self, position: GeoPoint, elapsed_minutes: float = 0.0,
                         completed_cell_ids: set[str] | None = None,
                         battery_percent: float | None = None,
                         coverage_fraction: float | None = None) -> None:
        self.current_position = position
        self.elapsed_minutes = max(self.elapsed_minutes, elapsed_minutes)
        if completed_cell_ids:
            self.completed_cell_ids.update(completed_cell_ids)
        if battery_percent is not None:
            self.battery_percent = max(0.0, min(100.0, battery_percent))
        if coverage_fraction is not None:
            self.coverage_fraction = max(0.0, min(1.0, coverage_fraction))

    def replan(self, detections: list[AerialDetection] | None = None) -> list[SearchWaypoint]:
        """Fuse detections and generate an investigation-first route.

        A corroborated or sufficiently confident detection diverts the aircraft,
        then resumes remaining search cells before returning home when budget allows.
        """
        if detections:
            self.ingest_detections(detections)
        accepted = [d for d in self.detections if d.confidence >= (.55 if "thermal" in d.evidence and "rgb" in d.evidence else .65)]
        remaining = [cell for cell in self.mission.cells if cell.cell_id not in self.completed_cell_ids]
        budget = max(0.1, self.mission.battery_minutes * self.battery_percent / 100 - self.elapsed_minutes)
        route: list[SearchWaypoint] = []
        if accepted:
            target = accepted[0]
            route.append(SearchWaypoint(1, target.location, WaypointKind.INVESTIGATE,
                                        rationale=f"{target.label}; {target.evidence}; confidence {target.confidence:.0%}"))
            planned = plan_coverage(remaining, target.location, battery_minutes=budget,
                                    airspace_polygon=self.mission.airspace_polygon)
            # plan_coverage reserves a return leg to its supplied origin; use the
            # actual mission home for the final operational waypoint.
            for point in planned[:-1]:
                route.append(SearchWaypoint(len(route) + 1, point.location, point.kind,
                                            point.cell_id, point.dwell_seconds, point.rationale))
            route.append(SearchWaypoint(len(route) + 1, self.mission.last_known_position,
                                        WaypointKind.RETURN_HOME, rationale="protected return reserve"))
        else:
            route = plan_coverage(remaining, self.current_position, battery_minutes=budget,
                                  airspace_polygon=self.mission.airspace_polygon)
            if route:
                route[-1] = SearchWaypoint(route[-1].sequence, self.mission.last_known_position,
                                            WaypointKind.RETURN_HOME, rationale=route[-1].rationale)
        self.route_version += 1
        self.last_route = route
        return route

    def plan(self) -> list[SearchWaypoint]:
        rank_search_cells(self.mission.cells, self.current_position)
        return self.replan()

    def field_packet(self) -> dict:
        return {
            "mission_id": self.mission.mission_id,
            "offline": True,
            "route_version": self.route_version,
            "battery_percent": round(self.battery_percent, 2),
            "coverage_fraction": round(self.coverage_fraction, 4),
            "current_position": {"lat": self.current_position.lat, "lon": self.current_position.lon},
            "detections": [
                {"id": d.detection_id, "lat": d.location.lat, "lon": d.location.lon,
                 "confidence": round(d.confidence, 3), "label": d.label,
                 "evidence": d.evidence}
                for d in self.detections
            ],
            "waypoints": [
                {"sequence": point.sequence, "kind": point.kind.value,
                 "lat": point.location.lat, "lon": point.location.lon,
                 "cell_id": point.cell_id, "rationale": point.rationale}
                for point in self.plan()
            ],
        }

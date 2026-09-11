from __future__ import annotations

from math import exp, hypot

from .coverage import plan_coverage
from .detection import fuse_detections
from .models import AerialDetection, GeoPoint, SearchMission, SearchWaypoint, WaypointKind
from .prioritization import rank_search_cells
from .astar import plan_3d_detour


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
        self.active_investigation_id: str | None = None
        self.investigated_detection_ids: set[str] = set()
        self.battery_percent = 100.0
        self.coverage_fraction = 0.0
        self.dem = None
        self.satellite_context = None
        self.no_fly_volumes: list[tuple[float, float, float, float, float, float]] = []

    def configure_terrain(self, dem, no_fly_volumes: list[tuple[float, float, float, float, float, float]] | None = None) -> None:
        self.dem = dem
        self.no_fly_volumes = no_fly_volumes or []

    def configure_satellite_context(self, context) -> None:
        self.satellite_context = context

    def _update_satellite_features(self) -> None:
        if self.satellite_context is None:
            return
        for cell in self.mission.cells:
            if cell.satellite_context_score is not None:
                continue
            features = self.satellite_context.cell_features(cell.center.lat, cell.center.lon)
            if features is None:
                continue
            cell.vegetation_index = features["ndvi"]
            cell.satellite_context_score = features["visibility_score"]
            cell.visibility_score = round(.7 * cell.visibility_score + .3 * features["visibility_score"], 4)

    def _terrain_route(self, route: list[SearchWaypoint]) -> list[SearchWaypoint]:
        if self.dem is None or not route:
            return route
        expanded: list[SearchWaypoint] = []
        current = self.current_position
        for target in route:
            try:
                leg = plan_3d_detour(current, target.location, self.dem, no_fly=self.no_fly_volumes)
            except ValueError:
                expanded.append(SearchWaypoint(len(expanded) + 1, target.location, target.kind,
                                                target.cell_id, target.dwell_seconds,
                                                f"{target.rationale}; DEM coverage unavailable"))
                current = target.location
                continue
            for point in leg[:-1]:
                expanded.append(SearchWaypoint(len(expanded) + 1, point.location, point.kind,
                                                rationale=point.rationale))
            expanded.append(SearchWaypoint(len(expanded) + 1, target.location, target.kind,
                                           target.cell_id, target.dwell_seconds,
                                           f"{target.rationale}; DEM A* validated"))
            current = target.location
        return expanded

    def _update_dem_features(self) -> None:
        """Attach local elevation/slope evidence to cells when a DEM is loaded."""
        if self.dem is None:
            return
        for cell in self.mission.cells:
            try:
                features = self.dem.cell_features(cell.center.lat, cell.center.lon)
            except ValueError:
                # A mission can intentionally cover a larger area than one
                # uploaded tile; preserve the configured baseline in that case.
                continue
            if cell.dem_slope_score is None:
                cell.terrain_score = round(.5 * cell.terrain_score + .5 * features["slope_score"], 4)
            cell.elevation_m = features["elevation_m"]
            cell.slope_deg = features["slope_deg"]
            cell.dem_slope_score = features["slope_score"]

    def ingest_detections(self, detections: list[AerialDetection]) -> list[AerialDetection]:
        self.detections = fuse_detections(self.detections + detections)
        return self.detections

    def update_telemetry(self, position: GeoPoint, elapsed_minutes: float = 0.0,
                         completed_cell_ids: set[str] | None = None,
                         battery_percent: float | None = None,
                         coverage_fraction: float | None = None,
                         investigated_detection_ids: set[str] | None = None) -> None:
        self.current_position = position
        self.elapsed_minutes = max(self.elapsed_minutes, elapsed_minutes)
        if completed_cell_ids:
            self.completed_cell_ids.update(completed_cell_ids)
        if battery_percent is not None:
            self.battery_percent = max(0.0, min(100.0, battery_percent))
        if coverage_fraction is not None:
            self.coverage_fraction = max(0.0, min(1.0, coverage_fraction))
        if investigated_detection_ids:
            self.investigated_detection_ids.update(investigated_detection_ids)
            if self.active_investigation_id in investigated_detection_ids:
                self.active_investigation_id = None

    def complete_investigation(self, detection_id: str | None = None) -> None:
        detection_id = detection_id or self.active_investigation_id
        if detection_id:
            self.investigated_detection_ids.add(detection_id)
            if self.active_investigation_id == detection_id:
                self.active_investigation_id = None

    def clear_detections(self) -> None:
        """Dismiss all accepted detections and rebuild the ordinary search route."""
        self.detections.clear()
        self.active_investigation_id = None
        self.investigated_detection_ids.clear()
        self.replan()

    def replan(self, detections: list[AerialDetection] | None = None) -> list[SearchWaypoint]:
        """Fuse detections and generate an investigation-first route.

        A corroborated or sufficiently confident detection diverts the aircraft,
        then resumes remaining search cells before returning home when budget allows.
        """
        self._update_dem_features()
        self._update_satellite_features()
        rank_search_cells(self.mission.cells, self.current_position)
        if detections:
            self.ingest_detections(detections)
        accepted = [d for d in self.detections
                    if d.detection_id not in self.investigated_detection_ids
                    and d.detection_id != self.active_investigation_id
                    and d.confidence >= (.55 if "thermal" in d.evidence and "rgb" in d.evidence else .65)]
        remaining = [cell for cell in self.mission.cells if cell.cell_id not in self.completed_cell_ids]
        budget = max(0.1, self.mission.battery_minutes * self.battery_percent / 100 - self.elapsed_minutes)
        route: list[SearchWaypoint] = []
        if accepted:
            def urgency(detection: AerialDetection) -> tuple[float, float]:
                distance = hypot((detection.location.lat - self.current_position.lat) * 111_000,
                                 (detection.location.lon - self.current_position.lon) * 111_000)
                proximity = exp(-(distance / 1200) ** 2)
                return .65 * detection.confidence + .35 * proximity, detection.confidence
            target = max(accepted, key=urgency)
            self.active_investigation_id = target.detection_id
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
        route = self._terrain_route(route)
        self.route_version += 1
        self.last_route = route
        return route

    def plan(self) -> list[SearchWaypoint]:
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
                 "altitude": d.location.altitude_m, "confidence": round(d.confidence, 3), "label": d.label,
                 "band": d.band.value, "source_frame": d.source_frame,
                 "evidence": d.evidence, "crs": "EPSG:4326",
                 "investigated": d.detection_id in self.investigated_detection_ids,
                 "related_locations": [{"lat": p.lat, "lon": p.lon, "altitude": p.altitude_m}
                                       for p in d.related_locations]}
                for d in self.detections
            ],
            "waypoints": [
                {"sequence": point.sequence, "kind": point.kind.value,
                 "lat": point.location.lat, "lon": point.location.lon,
                 "altitude": point.location.altitude_m,
                 "cell_id": point.cell_id, "dwell_seconds": point.dwell_seconds,
                 "rationale": point.rationale, "crs": "EPSG:4326"}
                for point in self.plan()
            ],
        }

from __future__ import annotations

from math import exp, hypot

from .coverage import plan_coverage
from .detection import fuse_detections
from .models import (AerialDetection, DetectionBand, Drone, DroneStatus, GeoPoint,
                     SearchMission, SearchWaypoint, SwarmMission, WaypointKind)
from .prioritization import rank_search_cells
from .astar import plan_3d_detour
from .swarm import (assign_sectors, plan_swarm_routes, reassign_on_candidate,
                    reassign_on_failure, swarm_coverage_fraction)
from .tracking import (DetectionTracker, SIMULATION_TRACKING_FRAMING,
                       TRACKING_MODE)


SIMULATION_PLANNING_FRAMING = (
    "SIMULATION / PLANNING ONLY: KairoDrishti plans and simulates coordinated "
    "multi-drone search and candidate tracking; it does not control real aircraft "
    "and does not claim sensor-fused ground truth."
)


class SearchAndRescueMission:
    """Offline mission coordinator: rank, fuse, and emit field waypoints."""

    def __init__(self, mission: SearchMission, tracking_mode: str | None = None) -> None:
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
        self.tracker = DetectionTracker(tracking_mode)

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
        # Cross-band fusion happens before association; repeated passes are
        # then merged by Mahalanobis gating in the tracker.
        fused = fuse_detections(detections)
        self.tracker.ingest(fused, self.elapsed_minutes)
        self.detections = self.tracker.latest_detections()
        return self.detections

    def update_telemetry(self, position: GeoPoint, elapsed_minutes: float = 0.0,
                         completed_cell_ids: set[str] | None = None,
                         battery_percent: float | None = None,
                         coverage_fraction: float | None = None,
                         investigated_detection_ids: set[str] | None = None) -> None:
        self.current_position = position
        self.elapsed_minutes = max(self.elapsed_minutes, elapsed_minutes)
        self.tracker.advance(self.elapsed_minutes)
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
        self.tracker.reset()
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
            "tracking_mode": self.tracker.mode,
            "framing": SIMULATION_TRACKING_FRAMING,
            "tracks": self.tracker.packet_tracks(),
            "current_position": {"lat": self.current_position.lat, "lon": self.current_position.lon},
            "detections": [
                {"id": d.detection_id, "lat": d.location.lat, "lon": d.location.lon,
                 "altitude": d.location.altitude_m, "confidence": round(d.confidence, 3), "label": d.label,
                 "band": d.band.value, "source_frame": d.source_frame,
                 "evidence": d.evidence, "crs": "EPSG:4326",
                 "track_id": d.track_id,
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


class SwarmSearchAndRescueMission(SearchAndRescueMission):
    """Multi-drone planning coordinator for simulation / planning only.

    This subclasses the stable single-drone coordinator so DEM and satellite
    feature bookkeeping remains identical.  It never sends aircraft commands.
    """

    def __init__(self, swarm_mission: SwarmMission, tracking_mode: str | None = None) -> None:
        super().__init__(swarm_mission.mission, tracking_mode)
        self.swarm_mission = swarm_mission
        if not self.swarm_mission.sector_assignments:
            self.swarm_mission.sector_assignments = assign_sectors(
                self.swarm_mission.drones, self.mission.cells,
                self.mission.last_known_position)
        self.swarm_routes: dict[str, list[SearchWaypoint]] = {}

    @property
    def drones(self) -> list[Drone]:
        return self.swarm_mission.drones

    def update_drone_telemetry(self, drone_id: str, position: GeoPoint,
                               battery_percent: float, status: DroneStatus | str,
                               communication_ok: bool) -> None:
        for drone in self.drones:
            if drone.drone_id != drone_id:
                continue
            if isinstance(status, str):
                status = DroneStatus(status.lower())
            drone.position = position
            drone.battery_percent = max(0.0, min(100.0, float(battery_percent)))
            drone.status = status
            drone.communication_ok = bool(communication_ok)
            return
        raise KeyError(f"unknown drone: {drone_id}")

    @staticmethod
    def _candidate_threshold(detection: AerialDetection) -> float:
        evidence = detection.evidence.lower()
        return .55 if "rgb" in evidence and "thermal" in evidence else .65

    def ingest_detections(self, detections: list[AerialDetection]) -> list[AerialDetection]:
        fused = fuse_detections(detections)
        self.tracker.ingest(fused, self.elapsed_minutes)
        self.detections = self.tracker.latest_detections()
        for detection in fused:
            if detection.confidence >= self._candidate_threshold(detection):
                updated, _ = reassign_on_candidate(
                    self.swarm_mission.sector_assignments,
                    self.drones, detection.location)
                self.swarm_mission.sector_assignments = updated
        return self.detections

    def replan(self) -> dict[str, list[SearchWaypoint]]:
        """Repair failed ownership first, then compute per-drone routes."""
        self._update_dem_features()
        self._update_satellite_features()
        rank_search_cells(self.mission.cells, self.mission.last_known_position)
        self.swarm_mission.sector_assignments = reassign_on_failure(
            self.swarm_mission.sector_assignments, self.drones, self.mission.cells)
        self.swarm_routes = plan_swarm_routes(self.swarm_mission)
        self.route_version += 1
        return self.swarm_routes

    def plan(self) -> dict[str, list[SearchWaypoint]]:
        return self.replan()

    @staticmethod
    def _detection_json(detection: AerialDetection,
                        investigated: bool = False) -> dict:
        return {
            "id": detection.detection_id,
            "track_id": detection.track_id,
            "lat": detection.location.lat,
            "lon": detection.location.lon,
            "altitude": detection.location.altitude_m,
            "confidence": round(detection.confidence, 3),
            "label": detection.label,
            "band": detection.band.value,
            "source_frame": detection.source_frame,
            "evidence": detection.evidence,
            "crs": "EPSG:4326",
            "investigated": investigated or detection.investigated,
            "related_locations": [
                {"lat": point.lat, "lon": point.lon, "altitude": point.altitude_m}
                for point in detection.related_locations
            ],
        }

    def field_packet(self) -> dict:
        routes = self.replan()
        cells = {cell.cell_id: cell for cell in self.mission.cells}
        owner_status = {drone.drone_id: drone.status.value for drone in self.drones}
        sectors = []
        for cell in self.mission.cells:
            owner = self.swarm_mission.sector_assignments.get(cell.cell_id)
            sectors.append({
                "cell_id": cell.cell_id,
                "owning_drone_id": owner,
                "owner_status": owner_status.get(owner, "IDLE") if owner else "IDLE",
                "coverage_fraction": round(cell.covered_fraction, 4),
                "priority": cell.priority,
                "likelihood": round(cell.likelihood, 4),
                "crs": "EPSG:4326",
            })
        return {
            "mission_id": self.mission.mission_id,
            "offline": True,
            "mode": "swarm",
            "simulation_only": True,
            "planning_only": True,
            "framing": SIMULATION_PLANNING_FRAMING,
            "route_version": self.route_version,
            "home": {"lat": self.mission.last_known_position.lat,
                      "lon": self.mission.last_known_position.lon,
                      "altitude": self.mission.last_known_position.altitude_m},
            "cells": [{"id": cell.cell_id, "lat": cell.center.lat,
                       "lon": cell.center.lon, "altitude": cell.center.altitude_m,
                       "area_m2": cell.area_m2, "priority": cell.priority,
                       "likelihood": round(cell.likelihood, 4),
                       "coverage_fraction": round(cell.covered_fraction, 4)}
                      for cell in self.mission.cells],
            "no_fly": [],
            "tracking_mode": self.tracker.mode,
            "tracking_framing": SIMULATION_TRACKING_FRAMING,
            "tracks": self.tracker.packet_tracks(),
            "drones": [{
                "drone_id": drone.drone_id,
                "status": drone.status.value,
                "battery_percent": round(drone.battery_percent, 2),
                "lat": drone.position.lat,
                "lon": drone.position.lon,
                "assigned_sector_id": drone.assigned_sector_id or (
                    "CANDIDATE" if drone.status is DroneStatus.INVESTIGATING else None
                ),
                "communication_ok": drone.communication_ok,
            } for drone in self.drones],
            "sectors": sectors,
            "detections": [self._detection_json(
                detection, detection.detection_id in self.investigated_detection_ids)
                for detection in self.detections],
            "waypoints": {
                drone_id: [{
                    "sequence": point.sequence,
                    "kind": point.kind.value,
                    "lat": point.location.lat,
                    "lon": point.location.lon,
                    "altitude": point.location.altitude_m,
                    "cell_id": point.cell_id,
                    "dwell_seconds": point.dwell_seconds,
                    "rationale": point.rationale,
                    "crs": "EPSG:4326",
                } for point in route]
                for drone_id, route in routes.items()
            },
            "swarm_coverage_fraction": swarm_coverage_fraction(self.swarm_mission),
        }

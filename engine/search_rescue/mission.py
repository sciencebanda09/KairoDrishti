from __future__ import annotations

from .coverage import plan_coverage
from .detection import fuse_detections
from .models import AerialDetection, SearchMission, SearchWaypoint
from .prioritization import rank_search_cells


class SearchAndRescueMission:
    """Offline mission coordinator: rank, fuse, and emit field waypoints."""

    def __init__(self, mission: SearchMission) -> None:
        self.mission = mission
        self.detections: list[AerialDetection] = []

    def ingest_detections(self, detections: list[AerialDetection]) -> list[AerialDetection]:
        self.detections = fuse_detections(self.detections + detections)
        return self.detections

    def plan(self) -> list[SearchWaypoint]:
        rank_search_cells(self.mission.cells, self.mission.last_known_position)
        return plan_coverage(self.mission.cells, self.mission.last_known_position,
                             battery_minutes=self.mission.battery_minutes)

    def field_packet(self) -> dict:
        return {
            "mission_id": self.mission.mission_id,
            "offline": True,
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

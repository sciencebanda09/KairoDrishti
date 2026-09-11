from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DetectionBand(str, Enum):
    RGB = "rgb"
    THERMAL = "thermal"
    MULTISPECTRAL = "multispectral"


class WaypointKind(str, Enum):
    SEARCH = "search"
    INVESTIGATE = "investigate"
    RETURN_HOME = "return_home"


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float
    altitude_m: float = 0.0


@dataclass
class AerialDetection:
    detection_id: str
    location: GeoPoint
    label: str
    confidence: float
    band: DetectionBand = DetectionBand.RGB
    source_frame: str = ""
    evidence: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be in [0, 1]")


@dataclass
class SearchCell:
    cell_id: str
    center: GeoPoint
    area_m2: float
    terrain_score: float = 0.5
    visibility_score: float = 0.5
    covered_fraction: float = 0.0
    movement_score: float = 0.5
    likelihood: float = 0.0
    priority: int = 0


@dataclass
class SearchMission:
    mission_id: str
    last_known_position: GeoPoint
    cells: list[SearchCell]
    battery_minutes: float = 24.0
    airspace_polygon: Optional[list[GeoPoint]] = None
    sensor_bands: set[DetectionBand] = field(
        default_factory=lambda: {DetectionBand.RGB, DetectionBand.THERMAL}
    )


@dataclass(frozen=True)
class SearchWaypoint:
    sequence: int
    location: GeoPoint
    kind: WaypointKind
    cell_id: Optional[str] = None
    dwell_seconds: int = 0
    rationale: str = ""


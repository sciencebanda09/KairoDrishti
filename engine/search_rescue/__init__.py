"""Offline aerial search-and-rescue intelligence primitives."""

from .models import (
    AerialDetection, DetectionBand, GeoPoint, SearchCell, SearchMission,
    SearchWaypoint, WaypointKind,
)
from .detectors import FrameMetadata, detect_change, detect_multispectral, detect_rgb, detect_thermal
from .detection import fuse_detections
from .prioritization import rank_search_cells
from .coverage import plan_coverage
from .change import detect_changes
from .mission import SearchAndRescueMission

__all__ = [
    "AerialDetection", "DetectionBand", "GeoPoint", "SearchCell",
    "SearchMission", "SearchWaypoint", "WaypointKind", "fuse_detections",
    "rank_search_cells", "plan_coverage", "detect_changes",
    "SearchAndRescueMission",
]

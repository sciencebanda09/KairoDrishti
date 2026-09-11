"""Offline aerial search-and-rescue intelligence primitives."""

from .models import (
    AerialDetection, DetectionBand, GeoPoint, SearchCell, SearchMission,
    SearchWaypoint, WaypointKind,
)
from .detectors import FrameMetadata, detect_change, detect_change_with_report, detect_multispectral, detect_rgb, detect_thermal
from .yolo_detector import YoloRgbDetector
from .terrain import ElevationGrid, load_dem, load_geotiff, load_hgt
from .satellite import SatelliteContext, build_satellite_context
from .astar import plan_3d_detour
from .detection import fuse_detections
from .prioritization import rank_search_cells
from .coverage import plan_coverage
from .change import detect_changes, detect_registered_changes, register_previous, RegistrationReport
from .geospatial import IngestedFrame, ingest_frame
from .mission import SearchAndRescueMission

__all__ = [
    "AerialDetection", "DetectionBand", "GeoPoint", "SearchCell",
    "SearchMission", "SearchWaypoint", "WaypointKind", "fuse_detections",
    "rank_search_cells", "plan_coverage", "detect_changes",
    "detect_registered_changes", "register_previous", "RegistrationReport",
    "detect_change_with_report", "IngestedFrame", "ingest_frame",
    "SearchAndRescueMission",
]

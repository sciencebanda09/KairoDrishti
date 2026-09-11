"""Offline aerial search-and-rescue intelligence primitives."""

from .models import (
    AerialDetection, DetectionBand, Drone, DroneStatus, GeoPoint, SearchCell,
    SearchMission, SearchWaypoint, SwarmMission, WaypointKind,
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
from .mission import SearchAndRescueMission, SwarmSearchAndRescueMission
from .flocking import alignment, cohesion, flock_step, separation
from .tracking import (DetectionTracker, Track, TRACKING_MODE,
                       associate_detections, imm_predict, imm_update,
                       mahalanobis_gate)
from .swarm import (RESERVE_MIN_BATTERY_PERCENT, assign_sectors,
                    plan_swarm_routes, reassign_on_candidate,
                    reassign_on_failure, swarm_coverage_fraction)

__all__ = [
    "AerialDetection", "DetectionBand", "Drone", "DroneStatus", "GeoPoint",
    "SearchCell", "SearchMission", "SearchWaypoint", "SwarmMission",
    "WaypointKind", "fuse_detections",
    "rank_search_cells", "plan_coverage", "detect_changes",
    "detect_registered_changes", "register_previous", "RegistrationReport",
    "detect_change_with_report", "IngestedFrame", "ingest_frame",
    "SearchAndRescueMission", "SwarmSearchAndRescueMission",
    "RESERVE_MIN_BATTERY_PERCENT", "assign_sectors", "reassign_on_candidate",
    "reassign_on_failure", "plan_swarm_routes", "swarm_coverage_fraction",
    "separation", "alignment", "cohesion", "flock_step", "Track",
    "DetectionTracker", "TRACKING_MODE", "mahalanobis_gate",
    "associate_detections", "imm_predict", "imm_update",
]

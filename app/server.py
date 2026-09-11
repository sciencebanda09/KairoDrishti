"""Serve the deterministic KairoDrishti 3D mission dashboard offline."""
from __future__ import annotations

import argparse
import io
import json
import tempfile
import zipfile
from email import policy
from email.parser import BytesParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from engine.search_rescue import (AerialDetection, DetectionBand, Drone, DroneStatus,
                                  FrameMetadata, GeoPoint, SearchAndRescueMission,
                                  SearchCell, SearchMission, SwarmMission,
                                  SwarmSearchAndRescueMission, detect_change,
                                  detect_multispectral, detect_rgb, detect_thermal,
                                  load_dem, YoloRgbDetector)
from engine.search_rescue.detectors import detect_change_with_report
from engine.search_rescue.detector_labels import TERRAIN_CLASSES
from engine.search_rescue.exports import export_packet
from engine.search_rescue.geospatial import ingest_frame, metadata_json
from engine.search_rescue.satellite import build_satellite_context
from engine.search_rescue.vision import decode_array as decode_vision_array, opencv_available
from engine.search_rescue.mission import SIMULATION_PLANNING_FRAMING
from app.airspace import OpenSkyCache

ROOT = Path(__file__).parent
STATIC_ROOT = ROOT / "dist" if (ROOT / "dist").exists() else ROOT / "static"


class UploadPart:
    def __init__(self, value: bytes, filename: str = "") -> None:
        self.file = io.BytesIO(value)
        self.filename = filename


class MultipartForm:
    def __init__(self, fields: dict[str, list[str]], files: dict[str, UploadPart]) -> None:
        self.fields = fields
        self.files = files

    def __getitem__(self, name: str) -> UploadPart:
        if name not in self.files:
            raise KeyError(name)
        return self.files[name]

    def getfirst(self, name: str, default: str | None = None) -> str | None:
        values = self.fields.get(name)
        return values[0] if values else default


def parse_multipart(handler: SimpleHTTPRequestHandler) -> MultipartForm:
    """Parse a multipart upload without the removed Python 3.13 cgi module."""
    length = int(handler.headers.get("content-length", "0"))
    body = handler.rfile.read(length)
    content_type = handler.headers.get("content-type", "")
    if not body or not content_type.lower().startswith("multipart/"):
        return MultipartForm({}, {})
    header = (f"Content-Type: {content_type}\r\n"
              "MIME-Version: 1.0\r\n\r\n").encode("utf-8")
    message = BytesParser(policy=policy.default).parsebytes(header + body)
    fields: dict[str, list[str]] = {}
    files: dict[str, UploadPart] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        value = part.get_payload(decode=True) or b""
        filename = part.get_filename() or ""
        if filename:
            files[name] = UploadPart(value, filename)
        else:
            charset = part.get_content_charset() or "utf-8"
            fields.setdefault(name, []).append(value.decode(charset, errors="replace"))
    return MultipartForm(fields, files)


def build_mission() -> SearchAndRescueMission:
    home = GeoPoint(30.362, 78.082, 1200)
    cells = [
        SearchCell("RIDGE-01", GeoPoint(30.363, 78.083, 1180), 25_000, .80, .55, movement_score=.92),
        SearchCell("STREAM-02", GeoPoint(30.366, 78.086, 1120), 25_000, .92, .35, movement_score=.72),
        SearchCell("CLEARING-03", GeoPoint(30.371, 78.091, 1250), 25_000, .30, .90, movement_score=.38),
        SearchCell("TRAIL-04", GeoPoint(30.368, 78.080, 1140), 25_000, .62, .62, movement_score=.86),
    ]
    return SearchAndRescueMission(SearchMission("SAR-DEMO-001", home, cells, battery_minutes=18))


MISSION = build_mission()
YOLO_DETECTOR = None
DEM = None
PLANNER_MODE = "baseline"
DETECTOR_MODE = "opencv"
DETECTOR_ERROR = None
OPENSKY = OpenSkyCache()
SATELLITE_CONTEXT = None
SWARM_MISSION: SwarmSearchAndRescueMission | None = None
DEFAULT_NO_FLY = [{"id": "NFZ-A", "lat": 30.3665, "lon": 78.0835,
                   "radius": 0.0007, "height": 70}]


def _swarm_payload(payload: dict | None = None) -> SwarmSearchAndRescueMission:
    """Build a deterministic simulated swarm from optional JSON input."""
    payload = payload or {}
    base = build_mission().mission
    mission_id = str(payload.get("mission_id", base.mission_id))
    cells = base.cells
    if payload.get("cells"):
        cells = []
        for item in payload["cells"]:
            bands = {DetectionBand(str(value).lower())
                     for value in item.get("required_sensor_bands", [])}
            cells.append(SearchCell(
                str(item.get("cell_id", item.get("id"))),
                GeoPoint(float(item["lat"]), float(item["lon"]),
                         float(item.get("altitude", item.get("altitude_m", 0)))),
                float(item.get("area_m2", 25_000)),
                float(item.get("terrain_score", .5)),
                float(item.get("visibility_score", .5)),
                float(item.get("covered_fraction", 0.0)),
                float(item.get("movement_score", .5)),
                required_sensor_bands=bands,
            ))
    drone_payload = payload.get("drones")
    if drone_payload is None:
        count = max(1, int(payload.get("drone_count", 4)))
        default_batteries = [72.0, 64.0, 81.0, 22.0]
        drone_payload = [{
            "drone_id": f"DRONE-{index + 1:02d}",
            "lat": base.last_known_position.lat + (index - (count - 1) / 2) * .00015,
            "lon": base.last_known_position.lon + (index - (count - 1) / 2) * .00012,
            "altitude": base.last_known_position.altitude_m,
            "battery_percent": default_batteries[index] if index < len(default_batteries) else 70.0,
            "status": "searching",
            "sensor_bands": ["rgb", "thermal"],
        } for index in range(count)]
    drones = []
    for index, item in enumerate(drone_payload):
        bands = {DetectionBand(str(value).lower()) for value in item.get("sensor_bands", ["rgb", "thermal"])}
        status = DroneStatus(str(item.get("status", "searching")).lower())
        drones.append(Drone(
            str(item.get("drone_id", f"DRONE-{index + 1:02d}")),
            GeoPoint(float(item.get("lat", base.last_known_position.lat)),
                     float(item.get("lon", base.last_known_position.lon)),
                     float(item.get("altitude", item.get("altitude_m", base.last_known_position.altitude_m)))),
            tuple(item.get("velocity", (0.0, 0.0))),
            float(item.get("battery_percent", 100.0)), status, bands,
            item.get("assigned_sector_id"), bool(item.get("communication_ok", True)),
        ))
    return SwarmSearchAndRescueMission(SwarmMission(
        SearchMission(mission_id, base.last_known_position, cells,
                      battery_minutes=float(payload.get("battery_minutes", base.battery_minutes)),
                      airspace_polygon=base.airspace_polygon),
        drones,
    ))


def swarm_scenario() -> dict:
    if SWARM_MISSION is None:
        return {"mission_id": None, "offline": True, "mode": "swarm",
                "initialized": False, "simulation_only": True,
                "planning_only": True, "framing": SIMULATION_PLANNING_FRAMING,
                "drones": [], "sectors": [], "detections": [], "waypoints": {},
                "swarm_coverage_fraction": 0.0, "no_fly": DEFAULT_NO_FLY}
    payload = SWARM_MISSION.field_packet()
    payload["initialized"] = True
    payload["no_fly"] = DEFAULT_NO_FLY
    return payload


def scenario() -> dict:
    mission = MISSION
    home = mission.mission.last_known_position
    cells = mission.mission.cells
    packet = mission.field_packet()
    return {
        "mission_id": mission.mission.mission_id,
        "home": {"lat": home.lat, "lon": home.lon, "altitude": home.altitude_m},
        "detections": packet["detections"],
        "battery_percent": packet["battery_percent"],
        "coverage_fraction": packet["coverage_fraction"],
        "route_version": packet["route_version"],
        "tracking_mode": packet["tracking_mode"],
        "tracks": packet["tracks"],
        "framing": packet["framing"],
        "cells": [{"id": c.cell_id, "lat": c.center.lat, "lon": c.center.lon,
                   "altitude": c.center.altitude_m, "priority": c.priority,
                   "likelihood": c.likelihood, "terrain": c.terrain_score,
                   "visibility": c.visibility_score, "movement": c.movement_score,
                   "elevation_m": c.elevation_m, "slope_deg": c.slope_deg}
                  for c in cells],
        "waypoints": [
            point
            for point in packet["waypoints"]
        ],
        "no_fly": [{"id": "NFZ-A", "lat": 30.3665, "lon": 78.0835, "radius": 0.0007, "height": 70}],
        "truth": {"lat": 30.3668, "lon": 78.0852, "label": "POSSIBLE PERSON", "confidence": .84},
        "coordinate_system": "WGS84 → local equirectangular metres → scene metres",
        "simulation_only": True,
        "telemetry_source": "synthetic_replay",
        "satellite_context": SATELLITE_CONTEXT.summary(include_previews=False) if SATELLITE_CONTEXT else None,
        "readiness": readiness(),
    }


def readiness() -> dict:
    return {"detector_mode": DETECTOR_MODE,
            "model_loaded": bool(YOLO_DETECTOR and YOLO_DETECTOR.ready),
            "model_path": str(YOLO_DETECTOR.model_path) if YOLO_DETECTOR else None,
            "detector_classes": list(YOLO_DETECTOR.class_names) if YOLO_DETECTOR else list(TERRAIN_CLASSES),
            "terrain_detector_ready": bool(YOLO_DETECTOR and YOLO_DETECTOR.terrain_ready),
            "fallback_reason": DETECTOR_ERROR,
            "opencv_available": opencv_available(),
            "dem_loaded": DEM is not None, "planner_mode": PLANNER_MODE,
            "planner_ready": DEM is not None,
            "simulation_only": True,
            "opensky_enabled": OPENSKY.enabled,
            "satellite_context_loaded": SATELLITE_CONTEXT is not None,
            "satellite_products": list(SATELLITE_CONTEXT.products) if SATELLITE_CONTEXT else []}


def decode_array(raw: bytes, filename: str = "") -> np.ndarray:
    """Decode a NumPy or common raster upload for local detector workflows."""
    return decode_vision_array(raw, filename)


def terrain_payload() -> dict:
    if DEM is None:
        return {"source": "synthetic", "bounds": None, "grid": None,
                "message": "No DEM loaded; map uses clearly labelled synthetic terrain.",
                "statistics": None, "resolution_m": None, "crs": None,
                "vertical_units": None}
    target = 64
    rows = np.linspace(0, DEM.elevations.shape[0] - 1,
                       min(target, DEM.elevations.shape[0]), dtype=int)
    cols = np.linspace(0, DEM.elevations.shape[1] - 1,
                       min(target, DEM.elevations.shape[1]), dtype=int)
    grid = np.nan_to_num(DEM.elevations[np.ix_(rows, cols)], nan=0.0).round(1).tolist()
    return {"source": "dem", "format": DEM.source, "bounds": DEM.bounds,
            "grid": grid, "statistics": DEM.statistics,
            "resolution_m": list(DEM.resolution_m), "crs": DEM.crs,
            "vertical_units": DEM.vertical_units,
            "message": "Offline DEM elevation grid loaded."}


def load_dem_upload(raw: bytes, filename: str):
    suffix = Path(filename).suffix.lower() or ".tif"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(raw)
            temporary_path = Path(temporary.name)
        return load_dem(temporary_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def detect_rgb_frame(image: np.ndarray, metadata: FrameMetadata) -> list:
    global YOLO_DETECTOR, DETECTOR_MODE, DETECTOR_ERROR
    if YOLO_DETECTOR is not None:
        try:
            return YOLO_DETECTOR.detect(image, metadata)
        except Exception as error:
            DETECTOR_ERROR = f"YOLO inference failed: {error}"
            if DETECTOR_MODE == "yolo_explicit":
                raise RuntimeError(DETECTOR_ERROR) from error
            YOLO_DETECTOR = None
            DETECTOR_MODE = "opencv_fallback"
    return detect_rgb(image, metadata)


class Handler(SimpleHTTPRequestHandler):
    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, body: bytes, content_type: str, filename: str | None = None, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path == "/api/swarm":
            self._json(swarm_scenario())
            return
        if parsed.path == "/api/mission":
            self._json(scenario())
            return
        if parsed.path == "/api/readiness":
            self._json(readiness())
            return
        if parsed.path == "/api/terrain":
            self._json(terrain_payload())
            return
        if parsed.path == "/api/airspace":
            query = parse_qs(parsed.query)
            bounds = {}
            for source, target in (("south", "lamin"), ("west", "lomin"), ("north", "lamax"), ("east", "lomax")):
                if source in query:
                    bounds[target] = float(query[source][0])
            self._json(OPENSKY.snapshot(bounds or None))
            return
        if parsed.path == "/api/mission/export":
            format_name = parse_qs(parsed.query).get("format", ["geojson"])[0]
            try:
                content, content_type, extension = export_packet(scenario(), format_name)
                self._bytes(content.encode("utf-8"), content_type,
                            f"kairodristi-field-packet.{extension}")
            except ValueError as error:
                self._json({"error": str(error)}, 400)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        global MISSION, SATELLITE_CONTEXT, DEM, PLANNER_MODE, SWARM_MISSION
        try:
            if self.path == "/api/swarm/init":
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                SWARM_MISSION = _swarm_payload(payload)
                if DEM is not None:
                    SWARM_MISSION.configure_terrain(DEM)
                if SATELLITE_CONTEXT is not None:
                    SWARM_MISSION.configure_satellite_context(SATELLITE_CONTEXT)
                self._json(swarm_scenario())
                return
            if self.path == "/api/swarm/telemetry":
                if SWARM_MISSION is None:
                    self._json({"error": "swarm is not initialized", "offline": True,
                                "mode": "swarm", "simulation_only": True,
                                "planning_only": True, "framing": SIMULATION_PLANNING_FRAMING}, 409)
                    return
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                updates = payload.get("drones", [])
                if not updates and payload.get("drone_id"):
                    updates = [payload]
                for update in updates:
                    current = next(drone for drone in SWARM_MISSION.drones
                                   if drone.drone_id == str(update["drone_id"]))
                    SWARM_MISSION.update_drone_telemetry(
                        current.drone_id,
                        GeoPoint(float(update.get("lat", current.position.lat)),
                                 float(update.get("lon", current.position.lon)),
                                 float(update.get("altitude", current.position.altitude_m))),
                        float(update.get("battery_percent", current.battery_percent)),
                        update.get("status", current.status.value),
                        bool(update.get("communication_ok", current.communication_ok)),
                    )
                if payload.get("detections"):
                    detections = []
                    for item in payload["detections"]:
                        detections.append(AerialDetection(
                            str(item.get("id", item.get("detection_id", "swarm-pass"))),
                            GeoPoint(float(item["lat"]), float(item["lon"]),
                                     float(item.get("altitude", item.get("altitude_m", 0)))),
                            str(item.get("label", "candidate")),
                            float(item.get("confidence", .7)),
                            DetectionBand(str(item.get("band", "rgb")).lower()),
                            str(item.get("source_frame", "swarm-pass")),
                            str(item.get("evidence", "simulated pass")),
                        ))
                    SWARM_MISSION.ingest_detections(detections)
                SWARM_MISSION.replan()
                self._json(swarm_scenario())
                return
            if self.path == "/api/swarm/reset":
                SWARM_MISSION = None
                self._json(swarm_scenario())
                return
            if self.path == "/api/mission/telemetry":
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length))
                MISSION.update_telemetry(
                    GeoPoint(float(payload["lat"]), float(payload["lon"]), float(payload.get("altitude_m", 0))),
                    float(payload.get("elapsed_minutes", 0)), set(payload.get("completed_cell_ids", [])),
                    payload.get("battery_percent"), payload.get("coverage_fraction"),
                    set(payload.get("investigated_detection_ids", [])),
                )
                self._json(scenario())
                return
            if self.path == "/api/mission/detections/investigate":
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length))
                MISSION.complete_investigation(str(payload["detection_id"]))
                self._json(scenario())
                return
            form = parse_multipart(self)
            if self.path == "/api/mission/reset":
                MISSION = build_mission()
                if DEM is not None:
                    MISSION.configure_terrain(DEM)
                if SATELLITE_CONTEXT is not None:
                    MISSION.configure_satellite_context(SATELLITE_CONTEXT)
                self._json(scenario())
                return
            if self.path == "/api/mission/detections/clear":
                MISSION.clear_detections()
                self._json(scenario())
                return
            if self.path == "/api/imagery/ingest":
                upload = form["image"]
                frame = ingest_frame(upload.file.read(), upload.filename or "", form.getfirst("metadata", "{}"))
                self._json({"frame": metadata_json(frame.metadata, frame.bounds),
                            "source": frame.source, "shape": list(frame.array.shape),
                            "message": frame.message})
                return
            if self.path == "/api/terrain/load":
                upload = form["dem"]
                DEM = load_dem_upload(upload.file.read(), upload.filename or "uploaded.tif")
                PLANNER_MODE = "dem_voxel_astar"
                MISSION.configure_terrain(DEM)
                MISSION.replan()
                self._json({"terrain": terrain_payload(), "mission": scenario(),
                            "message": f"{DEM.source.upper()} DEM loaded for terrain-aware planning"})
                return
            if self.path == "/api/satellite/context":
                upload = form["image"]
                frame = ingest_frame(upload.file.read(), upload.filename or "", form.getfirst("metadata", "{}"))
                if frame.metadata.source_type != "satellite":
                    raise ValueError("satellite context requires a GeoTIFF/COG or source_type=satellite")
                SATELLITE_CONTEXT = build_satellite_context(frame.array, frame.metadata, frame.bounds)
                MISSION.configure_satellite_context(SATELLITE_CONTEXT)
                MISSION.replan()
                self._json({"context": SATELLITE_CONTEXT.summary(), "mission": scenario()})
                return
            if self.path == "/api/frames/detect":
                sensor = form.getfirst("sensor", "rgb")
                upload = form["image"]
                frame = ingest_frame(upload.file.read(), upload.filename or "", form.getfirst("metadata", "{}"))
                image, meta = frame.array, frame.metadata
                if sensor == "thermal":
                    detections = detect_thermal(image, meta)
                elif sensor == "multispectral":
                    detections = detect_multispectral(image, meta)
                else:
                    detections = detect_rgb_frame(image, meta)
                MISSION.update_telemetry(meta.location)
                MISSION.replan(detections)
                self._json(scenario())
                return
            if self.path == "/api/change-detection":
                previous_upload = form["previous"]
                current_upload = form["current"]
                metadata = form.getfirst("metadata", "{}")
                previous_frame = ingest_frame(previous_upload.file.read(), previous_upload.filename or "", metadata)
                current_frame = ingest_frame(current_upload.file.read(), current_upload.filename or "", metadata)
                detections, registration = detect_change_with_report(previous_frame.array, current_frame.array, current_frame.metadata)
                MISSION.replan(detections)
                result = scenario()
                result["change_registration"] = {"method": registration.method,
                                                  "matches": registration.matches,
                                                  "inliers": registration.inliers,
                                                  "score": registration.score,
                                                  "registered": registration.registered,
                                                  "message": registration.message}
                self._json(result)
                return
            if self.path == "/api/dataset/analyze":
                archive = form["dataset"].file.read()
                with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                    manifest = json.loads(bundle.read("manifest.json"))
                    all_detections = []
                    for item in manifest.get("frames", []):
                        raw = bundle.read(item["path"])
                        frame = ingest_frame(raw, item["path"], item)
                        image, meta = frame.array, frame.metadata
                        sensor = item.get("sensor", "rgb")
                        if sensor == "thermal":
                            all_detections.extend(detect_thermal(image, meta))
                        elif sensor == "multispectral":
                            all_detections.extend(detect_multispectral(image, meta))
                        else:
                            all_detections.extend(detect_rgb_frame(image, meta))
                    MISSION.replan(all_detections)
                self._json(scenario())
                return
            self._json({"error": "unknown endpoint"}, 404)
        except (KeyError, ValueError, OSError, RuntimeError, TypeError, zipfile.BadZipFile) as error:
            if self.path.startswith("/api/swarm"):
                self._json({"error": str(error), "offline": True, "mode": "swarm",
                            "simulation_only": True, "planning_only": True,
                            "framing": SIMULATION_PLANNING_FRAMING}, 400)
            else:
                self._json({"error": str(error)}, 400)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def detection_json(detection) -> dict:
    return {"id": detection.detection_id, "lat": detection.location.lat, "lon": detection.location.lon,
            "altitude": detection.location.altitude_m,
            "confidence": round(detection.confidence, 3), "label": detection.label,
            "band": detection.band.value, "evidence": detection.evidence, "source_frame": detection.source_frame,
            "track_id": detection.track_id, "investigated": detection.investigated}


def main() -> None:
    global YOLO_DETECTOR, DEM, PLANNER_MODE, DETECTOR_MODE, DETECTOR_ERROR, OPENSKY
    parser = argparse.ArgumentParser(description="Run the offline KairoDrishti mission viewer")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--yolo-model")
    parser.add_argument("--detector", choices=("auto", "yolo", "opencv"), default="auto")
    parser.add_argument("--dem")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--opensky", action="store_true",
                        help="enable optional cached OpenSky ADS-B airspace context")
    parser.add_argument("--opensky-cache", default="runs/cache/opensky.json")
    args = parser.parse_args()
    if args.detector != "opencv":
        candidates = [Path(args.yolo_model)] if args.yolo_model else [
            ROOT.parent / "models" / "kairodristi-terrain.pt",
            ROOT.parent / "models" / "kairodristi-llvip-person.pt",
            ROOT.parent / "models" / "yolo11n.pt",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                YOLO_DETECTOR = YoloRgbDetector(candidate, device=args.device)
                YOLO_DETECTOR.load()
                DETECTOR_MODE = "yolo_explicit" if args.yolo_model else "yolo"
                break
            except Exception as error:
                DETECTOR_ERROR = f"YOLO model unavailable: {error}"
                YOLO_DETECTOR = None
                if args.detector == "yolo" or args.yolo_model:
                    raise
        if YOLO_DETECTOR is None and args.detector == "yolo":
            raise RuntimeError(DETECTOR_ERROR or "no YOLO model found")
    if args.dem:
        DEM = load_dem(args.dem)
        PLANNER_MODE = "dem_voxel_astar"
        MISSION.configure_terrain(DEM)
    OPENSKY = OpenSkyCache(enabled=args.opensky, cache_path=args.opensky_cache)
    handler = lambda *a, **kw: Handler(*a, directory=str(STATIC_ROOT), **kw)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"KairoDrishti dashboard: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

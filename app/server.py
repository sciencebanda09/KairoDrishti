"""Serve the deterministic KairoDrishti 3D mission dashboard offline."""
from __future__ import annotations

import argparse
import cgi
import io
import json
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from engine.search_rescue import (FrameMetadata, GeoPoint, SearchAndRescueMission,
                                  SearchCell, SearchMission, detect_change,
                                  detect_multispectral, detect_rgb, detect_thermal)

ROOT = Path(__file__).parent
STATIC_ROOT = ROOT / "dist" if (ROOT / "dist").exists() else ROOT / "static"


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
        "cells": [{"id": c.cell_id, "lat": c.center.lat, "lon": c.center.lon,
                   "altitude": c.center.altitude_m, "priority": c.priority,
                   "likelihood": c.likelihood, "terrain": c.terrain_score,
                   "visibility": c.visibility_score, "movement": c.movement_score}
                  for c in cells],
        "waypoints": [
            {**point, "altitude": next(
                (cell.center.altitude_m for cell in cells if cell.cell_id == point.get("cell_id")),
                home.altitude_m,
            )}
            for point in packet["waypoints"]
        ],
        "no_fly": [{"id": "NFZ-A", "lat": 30.3665, "lon": 78.0835, "radius": 0.0007, "height": 70}],
        "truth": {"lat": 30.3668, "lon": 78.0852, "label": "POSSIBLE PERSON", "confidence": .84},
        "coordinate_system": "WGS84 → local equirectangular metres → scene metres",
    }


class Handler(SimpleHTTPRequestHandler):
    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/mission":
            self._json(scenario())
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        global MISSION
        try:
            if self.path == "/api/mission/telemetry":
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length))
                MISSION.update_telemetry(
                    GeoPoint(float(payload["lat"]), float(payload["lon"]), float(payload.get("altitude_m", 0))),
                    float(payload.get("elapsed_minutes", 0)), set(payload.get("completed_cell_ids", [])),
                    payload.get("battery_percent"), payload.get("coverage_fraction"),
                )
                self._json(scenario())
                return
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                    environ={"REQUEST_METHOD": "POST",
                                             "CONTENT_TYPE": self.headers.get("content-type", ""),
                                             "CONTENT_LENGTH": self.headers.get("content-length", "0")})
            if self.path == "/api/mission/reset":
                MISSION = build_mission()
                self._json(scenario())
                return
            if self.path == "/api/frames/detect":
                metadata = json.loads(form.getfirst("metadata", "{}"))
                sensor = form.getfirst("sensor", "rgb")
                upload = form["image"]
                raw = upload.file.read()
                image = np.load(io.BytesIO(raw)) if upload.filename.lower().endswith(".npy") else np.asarray(Image.open(io.BytesIO(raw)))
                meta = FrameMetadata(metadata.get("frame_id", "live-0001"),
                                     GeoPoint(float(metadata["lat"]), float(metadata["lon"]), float(metadata.get("altitude_m", 0))),
                                     float(metadata.get("heading_deg", 0)), float(metadata.get("ground_sample_distance_m", .25)))
                if sensor == "thermal":
                    detections = detect_thermal(image, meta)
                elif sensor == "multispectral":
                    detections = detect_multispectral(image, meta)
                else:
                    detections = detect_rgb(image, meta)
                MISSION.update_telemetry(meta.location)
                MISSION.replan(detections)
                self._json({"detections": [detection_json(d) for d in detections], **scenario()})
                return
            if self.path == "/api/change-detection":
                metadata = json.loads(form.getfirst("metadata", "{}"))
                previous = np.load(io.BytesIO(form["previous"].file.read()))
                current = np.load(io.BytesIO(form["current"].file.read()))
                meta = FrameMetadata(metadata.get("frame_id", "change-0001"), GeoPoint(float(metadata["lat"]), float(metadata["lon"]), float(metadata.get("altitude_m", 0))), float(metadata.get("heading_deg", 0)), float(metadata.get("ground_sample_distance_m", .25)))
                detections = detect_change(previous, current, meta)
                MISSION.replan(detections)
                self._json({"detections": [detection_json(d) for d in detections], **scenario()})
                return
            if self.path == "/api/dataset/analyze":
                archive = form["dataset"].file.read()
                with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                    manifest = json.loads(bundle.read("manifest.json"))
                    all_detections = []
                    for item in manifest.get("frames", []):
                        raw = bundle.read(item["path"])
                        image = np.load(io.BytesIO(raw)) if item["path"].lower().endswith(".npy") else np.asarray(Image.open(io.BytesIO(raw)))
                        meta = FrameMetadata(item["frame_id"], GeoPoint(item["lat"], item["lon"], item.get("altitude_m", 0)), item.get("heading_deg", 0), item.get("ground_sample_distance_m", .25))
                        sensor = item.get("sensor", "rgb")
                        if sensor == "thermal":
                            all_detections.extend(detect_thermal(image, meta))
                        elif sensor == "multispectral":
                            all_detections.extend(detect_multispectral(image, meta))
                        else:
                            all_detections.extend(detect_rgb(image, meta))
                    MISSION.replan(all_detections)
                self._json({"detections": [detection_json(d) for d in all_detections], **scenario()})
                return
            self._json({"error": "unknown endpoint"}, 404)
        except (KeyError, ValueError, OSError, zipfile.BadZipFile) as error:
            self._json({"error": str(error)}, 400)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def detection_json(detection) -> dict:
    return {"id": detection.detection_id, "lat": detection.location.lat, "lon": detection.location.lon,
            "confidence": round(detection.confidence, 3), "label": detection.label,
            "band": detection.band.value, "evidence": detection.evidence, "source_frame": detection.source_frame}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline KairoDrishti mission viewer")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
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

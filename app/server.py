"""Serve the deterministic KairoDrishti 3D mission dashboard offline."""
from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from engine.search_rescue import GeoPoint, SearchAndRescueMission, SearchCell, SearchMission

ROOT = Path(__file__).parent


def scenario() -> dict:
    home = GeoPoint(30.362, 78.082, 1200)
    cells = [
        SearchCell("RIDGE-01", GeoPoint(30.363, 78.083, 1180), 25_000, .80, .55, movement_score=.92),
        SearchCell("STREAM-02", GeoPoint(30.366, 78.086, 1120), 25_000, .92, .35, movement_score=.72),
        SearchCell("CLEARING-03", GeoPoint(30.371, 78.091, 1250), 25_000, .30, .90, movement_score=.38),
        SearchCell("TRAIL-04", GeoPoint(30.368, 78.080, 1140), 25_000, .62, .62, movement_score=.86),
    ]
    mission = SearchAndRescueMission(SearchMission("SAR-DEMO-001", home, cells, battery_minutes=18))
    packet = mission.field_packet()
    return {
        "mission_id": mission.mission.mission_id,
        "home": {"lat": home.lat, "lon": home.lon, "altitude": home.altitude_m},
        "cells": [{"id": c.cell_id, "lat": c.center.lat, "lon": c.center.lon,
                   "altitude": c.center.altitude_m, "priority": c.priority,
                   "likelihood": c.likelihood, "terrain": c.terrain_score,
                   "visibility": c.visibility_score, "movement": c.movement_score}
                  for c in cells],
        "waypoints": packet["waypoints"],
        "no_fly": [{"id": "NFZ-A", "lat": 30.3665, "lon": 78.0835, "radius": 0.0007, "height": 70}],
        "truth": {"lat": 30.3668, "lon": 78.0852, "label": "POSSIBLE PERSON", "confidence": .84},
        "coordinate_system": "WGS84 → local equirectangular metres → scene metres",
    }


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/mission":
            body = json.dumps(scenario()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline KairoDrishti mission viewer")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = lambda *a, **kw: Handler(*a, directory=str(ROOT / "static"), **kw)
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

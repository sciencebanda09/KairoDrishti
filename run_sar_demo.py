"""Run a deterministic, connectivity-free PS #8 mission demonstration."""
from __future__ import annotations

import json

from engine.search_rescue import GeoPoint, SearchAndRescueMission, SearchCell, SearchMission


def main() -> None:
    home = GeoPoint(30.362, 78.082, 1200)
    cells = [
        SearchCell("RIDGE-01", GeoPoint(30.363, 78.083, 1180), 25_000,
                   terrain_score=.80, visibility_score=.55, movement_score=.92),
        SearchCell("STREAM-02", GeoPoint(30.366, 78.086, 1120), 25_000,
                   terrain_score=.92, visibility_score=.35, movement_score=.72),
        SearchCell("CLEARING-03", GeoPoint(30.371, 78.091, 1250), 25_000,
                   terrain_score=.30, visibility_score=.90, movement_score=.38),
    ]
    mission = SearchAndRescueMission(SearchMission("SAR-DEMO-001", home, cells, battery_minutes=18))
    print(json.dumps(mission.field_packet(), indent=2))


if __name__ == "__main__":
    main()

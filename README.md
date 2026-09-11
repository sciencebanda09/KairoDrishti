# KairoDrishti — Search & Rescue Aerial Intelligence

KairoDrishti is an offline-first decision-support system for PS #8. It helps rescue teams search difficult terrain by converting aerial observations into ranked sectors, corroborated detections, and geolocated flight waypoints.

It is field-oriented: the primary output is an actionable search packet, not a gallery of images. The system runs locally without connectivity, cloud inference, or a database.

## Capabilities

- Person and human-made-sign detections with geolocation, confidence, label, source frame, and sensor band.
- Cross-band fusion for RGB, thermal, and multispectral detections.
- Search-area prioritisation using last-known position, movement radius, terrain, visibility, and prior coverage.
- Battery-aware coverage planning with a protected return-to-home reserve.
- Offline change detection for aligned successive-pass imagery.
- Compact field packets containing detections and ordered waypoints.

## Workflow

```text
RGB / thermal / multispectral frames
              ↓
     local detector adapters
              ↓
       cross-band fusion
              ↓
  terrain + LKP + movement scoring
              ↓
 battery-aware coverage planning
              ↓
 geolocated field packet + RTH waypoint
```

## Quick start

```bash
pip install -e ".[dev]"
python run_sar_demo.py
pytest tests/unit/test_search_rescue.py -q
```

The deterministic demo needs no network connection, model download, database, or cloud service. It prints ranked sectors followed by an explicit return-to-home waypoint.

## Python API

```python
from engine.search_rescue import (
    GeoPoint, SearchCell, SearchMission, SearchAndRescueMission,
)

home = GeoPoint(30.362, 78.082, 1200)
mission = SearchAndRescueMission(SearchMission(
    "mission-001", home,
    [SearchCell("sector-a", GeoPoint(30.363, 78.083), 20_000,
                terrain_score=.8, movement_score=.9)],
    battery_minutes=20,
))
field_packet = mission.field_packet()
```

`field_packet` contains the mission ID, offline status, fused detections, ordered search waypoints, priority rationales, and a final `return_home` waypoint. Local detector adapters can create `AerialDetection` records and call `mission.ingest_detections(...)` without changing the planner.

## Project layout

```text
engine/search_rescue/        models, fusion, prioritization, route planning, change detection
run_sar_demo.py              deterministic offline field-packet demo
tests/unit/test_search_rescue.py
                             focused SAR behavior tests
```

## Validation and safety boundary

```bash
py -3.11 -m pytest tests/unit/test_search_rescue.py -q
py -3.11 -m compileall -q engine/search_rescue run_sar_demo.py
```

The planner is decision support for a human-led rescue team. A detection is an investigation cue, not confirmation of a survivor. Production deployment still needs calibrated detectors, a proper local geodetic projection, terrain and airspace feeds, aircraft-specific flight constraints, and field validation.

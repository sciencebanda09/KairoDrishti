# KairoDrishti Search & Rescue Aerial Intelligence

KairoDrishti is an offline-first decision-support system for PS #8. It helps rescue teams search difficult terrain by converting aerial observations into ranked sectors, corroborated detections, and geolocated flight waypoints.

It is field-oriented: the primary output is an actionable search packet, not a gallery of images. The system runs locally without connectivity, cloud inference, or a database.

## Capabilities

- Person and human-made-sign detections with geolocation, confidence, label, source frame, and sensor band.
- Cross-band fusion for RGB, thermal, and multispectral detections.
- Search-area prioritisation using last-known position, movement radius, terrain, visibility, and prior coverage.
- Battery-aware coverage planning with a protected return-to-home reserve.
- Offline change detection for aligned successive-pass imagery.
- Compact field packets containing detections and ordered waypoints.
- Live RGB webcam ingestion and uploaded RGB/thermal/multispectral datasets.
- Detection-driven investigation waypoints, route replanning, airspace checks, and return-home reserve protection.

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
py -3.11 -m pytest -q
python run_sar_demo.py
```

The deterministic engine needs no model download, database, or cloud service. It prints ranked sectors followed by an explicit return-to-home waypoint.

## Live 3D dashboard

Install the frontend dependencies once, then run the local Three.js mission viewer. Start the API in one terminal and Vite in another:

```bash
npm install
py -3.11 -m app.server               # API: http://127.0.0.1:8765
npm run dev                          # UI: http://127.0.0.1:5173
```

For a single-server production preview:

```bash
npm run build
python -m app.server                 # serves the compiled UI at http://127.0.0.1:8765
```

The browser scene is deterministic and offline. Its real Three.js height-field terrain, authored drone and vegetation, route, search sectors, no-fly volume, drone altitude, camera frustum, battery, coverage, detection alert, investigation diversion, and return-home leg all use one simulation clock. Click **START MISSION**: the drone flies the route, sends a test frame through the local detector, diverts to investigate, resumes the updated route, and returns home. **TEST EVENT** triggers the same backend detector path immediately. **EXPORT PACKET** downloads the current route as GeoJSON and CSV.

### Real frame input

The field console also supports **START CAMERA** for local RGB webcam frames and **UPLOAD DATASET** for a ZIP containing a `manifest.json` plus image files. Each frame is decoded locally, passed through the RGB, thermal, or multispectral baseline detector, geolocated from its frame metadata, fused into the mission, and sent back to the Three.js scene as an authoritative route update.

Thermal frames may be grayscale PNG/JPEG or NumPy `.npy` arrays. A minimal dataset manifest is:

```json
{
  "frames": [
    {"frame_id": "frame-001", "path": "rgb/frame-001.jpg", "sensor": "rgb",
     "lat": 30.3668, "lon": 78.0852, "altitude_m": 120, "ground_sample_distance_m": 0.25},
    {"frame_id": "frame-001-ir", "path": "thermal/frame-001.npy", "sensor": "thermal",
     "lat": 30.3668, "lon": 78.0852, "altitude_m": 120, "ground_sample_distance_m": 0.25}
  ]
}
```

The first implementation is an offline classical pixel baseline, not a trained survivor-recognition model. It is intentionally replaceable by a learned detector adapter later. Detections at 65% confidence, or 55% when RGB and thermal evidence corroborate, insert an investigation waypoint and replan the remaining route while preserving the return reserve.

### Dataset manifest

```json
{
  "frames": [
    {"frame_id": "frame-001", "path": "rgb/frame-001.jpg", "sensor": "rgb",
     "lat": 30.3668, "lon": 78.0852, "altitude_m": 120,
     "ground_sample_distance_m": 0.25},
    {"frame_id": "frame-001-ir", "path": "thermal/frame-001.npy", "sensor": "thermal",
     "lat": 30.3668, "lon": 78.0852, "altitude_m": 120,
     "ground_sample_distance_m": 0.25}
  ]
}
```

Thermal input accepts grayscale PNG/JPEG or `.npy` arrays. Multispectral input accepts `.npy` arrays with at least two bands. Public pedestrian datasets often lack drone GPS, so add explicit simulated coordinates when replaying them in this demo.

### Local API

```text
GET  /api/mission
POST /api/mission/reset
POST /api/mission/telemetry
POST /api/frames/detect
POST /api/dataset/analyze
POST /api/change-detection
```

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
engine/search_rescue/        models, detectors, fusion, prioritization, coverage, replanning
app/server.py                offline API and dataset/frame ingestion server
frontend/                    Vite + Three.js operator console
run_sar_demo.py              deterministic offline field-packet demo
tests/unit/                  engine, detector, API, and replanning tests
```

## Validation and safety boundary

```bash
py -3.11 -m pytest -q
py -3.11 -m compileall -q engine app tests
npm run build
git diff --check
```

The planner is decision support for a human-led rescue team. A detection is an investigation cue, not confirmation of a survivor. The current detectors are deterministic classical baselines; production deployment still needs calibrated learned models, camera calibration, a proper local geodetic projection, terrain and airspace feeds, aircraft-specific flight constraints, and field validation.

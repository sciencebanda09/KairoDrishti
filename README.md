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
- Simulated drone replay and uploaded RGB/thermal/multispectral/geospatial raster datasets.
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
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -m pytest -q
python run_sar_demo.py
```

The deterministic engine needs no model download, database, or cloud service. It prints ranked sectors followed by an explicit return-to-home waypoint. Python 3.11 is the supported local runtime for the complete YOLO, OpenCV, and raster workflow.

## Offline field dashboard

Install the frontend dependencies once, then run the local operator console. Start the API in one terminal and Vite in another:

```bash
npm install
python -m app.server                  # API: http://127.0.0.1:8765
npm run dev                          # UI: http://127.0.0.1:5173
```

For a single-server production preview:

```bash
npm run build
python -m app.server                 # serves the compiled UI at http://127.0.0.1:8765
```

The browser console is offline-first and 3D-first. The primary view is the bounded Three.js terrain scene, with ranked sectors, route, configured no-fly area, simulated drone position, detections, coordinates, and terrain source. A compact 2D minimap keeps the whole search picture visible while the simulation camera follows the route. The header explicitly marks the aircraft as **SIMULATION ONLY**. The sensor workspace provides RGB, thermal, multispectral, and registered successive-pass change tabs. **EXPORT FIELD PACKET** supports GeoJSON, CSV, KML, and GPX.

### Real frame input

The field console supports the **DRONE CAMERA**, **LOAD FRAME**, **UPLOAD DATASET**, and **LOAD TWO PASSES** workflows. JPEG/PNG/NumPy frames can use explicit camera metadata, while GeoTIFF/COG imagery uses its raster transform and CRS when the optional `rasterio` dependency is installed. Images are decoded locally with OpenCV, passed through a terrain-trained YOLO model when available, or through the explicit OpenCV baseline fallback. Thermal and multispectral inputs remain separate workflows and are never mislabeled as RGB.

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

The offline baseline provides deterministic contrast, hot-region, band-ratio, and successive-pass candidate generation. These signals are fused with learned detections when a local YOLO checkpoint is available, and candidate confidence drives investigation waypoints and route replanning while preserving the return reserve.

### YOLO and DEM mode

Ultralytics is included in `requirements.txt`. Use the supplied trained checkpoint directly from the ignored local weights directory:

```powershell
python -m app.server `
  --detector yolo `
  --yolo-model .\datasets\weights\best.pt `
  --device cpu
```

The server and installation command must use the same interpreter. If you are not using the virtual environment, use `py -3.11 -m app.server` and install with `py -3.11 -m pip install -r requirements.txt`.

Prepare the supplied LLVIP archive without committing its 4 GB contents:

```bash
python tools/prepare_llvip.py C:/Users/kumar/Downloads/LLVIP.zip
pip install -e ".[yolo]"
python tools/train_yolo.py --data datasets/llvip_yolo/dataset.yaml \
  --base-model models/yolo11n.pt \
  --output models/kairodristi-llvip-person.pt --device cpu
python tools/fetch_dem.py --tile N30E078 --output data/dem
```

### Kaggle GPU training

The laptop can prepare the dataset, while Kaggle runs the expensive YOLO training on a GPU. The Kaggle CLI must be installed and authenticated first (`kaggle --version` and `kaggle config view`). The LLVIP-derived dataset is uploaded privately to the configured Kaggle account; the raw ZIP and generated images remain ignored by Git.

The wrapper creates a deterministic 85/15 train/validation split, rewrites the dataset YAML to use portable Kaggle paths, uploads or versions the private dataset, submits a GPU kernel, and optionally waits for and downloads the artifacts:

```powershell
py -3.11 tools\kaggle_train.py `
  --dataset datasets\llvip_yolo `
  --epochs 30 `
  --imgsz 640 `
  --batch 16 `
  --wait
```

Without `--wait`, monitor and download manually:

```powershell
kaggle kernels status dmechatronicx/kairodristi-llvip-training
kaggle kernels output dmechatronicx/kairodristi-llvip-training `
  -p runs\kaggle\kairodristi-llvip-training --force
```

When the run completes, the wrapper copies `best.pt` to `models/kairodristi-llvip-person.pt`, alongside the downloaded training plots and metrics under `runs/kaggle/`. Use that checkpoint with the local API:

```powershell
py -3.11 -m app.server `
  --yolo-model models\kairodristi-llvip-person.pt `
  --device cpu
```

Kaggle training uses the visible LLVIP frames for the RGB person detector. LLVIP does not provide the drone GPS metadata required for field geolocation, so the trained checkpoint is used as the detector while KairoDrishti's local frame metadata and planner continue to provide geolocation and route decisions. The server defaults to `--detector auto`: it tries the trained checkpoint, then the bundled `models/yolo11n.pt`, and then reports an explicit OpenCV fallback if YOLO cannot load. Use `--detector opencv` to force the classical path or `--detector yolo` to fail fast when a YOLO model is required.

Run advanced mode with a local checkpoint and cached SRTM DEM:

```bash
python -m app.server --yolo-model models/kairodristi-llvip-person.pt \
  --dem data/dem/N30E078.hgt --device cpu
```

The YOLO model detects RGB people; paired infrared evidence corroborates detections. The DEM-backed planner is separate from detection and validates terrain clearance, altitude, airspace volumes, battery, and return-home legs. If the model or DEM is missing, the server reports classical/planner fallback status rather than claiming advanced mode is active.

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
POST /api/imagery/ingest
POST /api/terrain/load
POST /api/satellite/context
GET  /api/mission/export?format=geojson|csv|kml|gpx
GET  /api/airspace
```

`/api/imagery/ingest` returns normalized frame metadata, CRS, raster bounds, image shape, and metadata quality without changing mission state. `/api/terrain/load` accepts local SRTM `.hgt` or GeoTIFF/COG DEM files and activates CRS-aware terrain routing. `/api/satellite/context` accepts a local Sentinel-style raster with B2/B3/B4/B8 metadata and returns RGB, false-colour, and NDVI context products. Satellite context adjusts visibility-based sector ranking only; it never creates survivor waypoints. `/api/change-detection` registers the previous pass to the current pass with ORB/RANSAC before computing the change mask and returns registration quality. `/api/airspace` is optional OpenSky ADS-B context; configured no-fly volumes remain authoritative and the endpoint falls back to a local cache or an explicit unavailable status.

### Local DEM and satellite context

The field console is local-first. Use **LOAD DEM** with an SRTM `.hgt` or a single-band GeoTIFF/COG elevation raster. The console reports source, CRS, resolution, valid-pixel statistics, and uses slope/elevation in terrain-aware routing and sector ranking. If no DEM is loaded, the 3D scene remains clearly labelled synthetic.

Use **LOAD SATELLITE CONTEXT** with a prepared Sentinel-2 GeoTIFF/COG. Provide band metadata as JSON when the raster is not a four-band B2/B3/B4/B8 stack:

```json
{"source_type":"satellite","bands":["B2","B3","B4","B8"],"timestamp":"2025-01-15T05:20:00Z","cloud_percent":12.4}
```

Sentinel-2 is an overview source, not a person detector. The product keeps drone RGB/thermal imagery as the primary detection path and does not download imagery or store Copernicus credentials.

### Terrain detector training and evaluation

The required terrain-search taxonomy is `person`, `clothing`, `shelter`, and `track`. Prepare a hybrid dataset using public aerial/person data, generated terrain fixtures, and later field annotations:

```powershell
py -3.11 tools\build_terrain_fixture.py --output datasets\terrain_fixture
py -3.11 tools\train_yolo.py --data datasets\terrain_fixture\dataset.yaml `
  --base-model models\yolo11n.pt --output models\kairodristi-terrain.pt
py -3.11 tools\evaluate_detector.py --model models\kairodristi-terrain.pt `
  --data datasets\terrain_fixture\dataset.yaml
```

The generated fixture is only a pipeline smoke test, not field validation. The evaluation report records precision, recall, mAP50, mAP50-95, and required-class coverage. A checkpoint is considered terrain-ready only when all four class names are present.

### Field packet schema and formats

Every route or candidate record uses WGS84 (`EPSG:4326`) and includes mission ID, record type, sequence, kind, latitude, longitude, altitude, dwell time, cell ID, sensor, source frame, confidence, rationale, timestamp, and CRS. The dashboard can export the same packet as GeoJSON, CSV, KML, or GPX for ground-team tools.

### Airspace, telemetry, and coordinated search

The current aircraft is a deterministic simulator and is labelled **SIMULATION ONLY**. KairoDrishti plans routes, sectors, and field waypoints for human-operated aircraft; it does not send commands to or autonomously launch a real aircraft. Optional OpenSky integration adds cached ADS-B context for situational awareness when the server is started with `--opensky`. OpenSheet is intentionally not part of the offline core.

The coordinated-search architecture is designed to extend from one simulated aircraft to a multi-drone planning view: sector assignment, non-overlapping coverage, battery reserves, candidate handoff, and route recovery can be represented without changing the field-packet contract.

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

The planner is decision support for a human-led rescue team. A detection is an investigation cue, not confirmation of a survivor. Terrain-trained YOLO and the OpenCV anomaly path are explicit, inspectable detector modes; the field packet always preserves source frame, sensor, confidence, rationale, and geolocation for review.

## Future plans

1. Coordinated multi-drone search planning with sector assignment, coverage deconfliction, battery reserves, and candidate handoff.
2. Persistent multi-team mission state with offline synchronization and conflict-safe field packet merging.
3. Expanded terrain-specific training and evaluation for people, clothing, shelters, and tracks across RGB, thermal, multispectral, and change-detection imagery.

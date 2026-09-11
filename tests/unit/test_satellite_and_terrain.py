from __future__ import annotations

import numpy as np
import pytest

from engine.search_rescue.detectors import FrameMetadata
from engine.search_rescue.models import GeoPoint, SearchCell, SearchMission
from engine.search_rescue.mission import SearchAndRescueMission
from engine.search_rescue.satellite import build_satellite_context
from engine.search_rescue.terrain import ElevationGrid, load_geotiff


def test_sentinel_context_builds_products_and_ndvi():
    red = np.full((16, 20), 0.2, dtype=np.float32)
    nir = np.full((16, 20), 0.8, dtype=np.float32)
    image = np.stack([np.full_like(red, 0.1), np.full_like(red, 0.15), red, nir], axis=-1)
    metadata = FrameMetadata("sentinel", GeoPoint(30.0, 78.0), image_width=20, image_height=16,
                             source_type="satellite", band_names=("B2", "B3", "B4", "B8"),
                             metadata_quality="georeferenced")
    context = build_satellite_context(image, metadata)
    assert set(context.products) == {"rgb", "false_colour", "ndvi"}
    assert float(context.ndvi.mean()) == pytest.approx(.6)
    assert context.rgb_preview and context.false_colour_preview and context.ndvi_preview


def test_satellite_context_updates_visibility_without_creating_detections():
    image = np.stack([
        np.full((20, 20), .1, dtype=np.float32),
        np.full((20, 20), .15, dtype=np.float32),
        np.full((20, 20), .2, dtype=np.float32),
        np.full((20, 20), .8, dtype=np.float32),
    ], axis=-1)
    metadata = FrameMetadata("sentinel", GeoPoint(30.0, 78.0), image_width=20, image_height=20,
                             crs="EPSG:4326", geo_transform=(.001, 0, 77.99, 0, -.001, 30.01),
                             source_type="satellite", band_names=("B2", "B3", "B4", "B8"),
                             metadata_quality="georeferenced")
    context = build_satellite_context(image, metadata)
    home = GeoPoint(30.0, 78.0)
    mission = SearchAndRescueMission(SearchMission("sat", home, [SearchCell("A", home, 1000)]))
    mission.configure_satellite_context(context)
    mission.replan()
    assert mission.mission.cells[0].vegetation_index == pytest.approx(.6)
    assert mission.mission.cells[0].satellite_context_score == pytest.approx(.2)
    assert not mission.detections


def test_elevation_grid_sample_and_slope_are_finite():
    values = np.tile(np.arange(20, dtype=np.float32), (20, 1))
    dem = ElevationGrid(values, 30.0, 77.0, 36)
    assert dem.sample(30.1, 77.1) == pytest.approx(10)
    assert dem.slope_degrees(30.1, 77.1) > 0
    assert dem.statistics["valid_pixels"] == 400


def test_mission_keeps_baseline_route_when_dem_does_not_cover_a_cell():
    dem = ElevationGrid(np.full((20, 20), 100, dtype=np.float32), 10.0, 20.0, 36)
    home = GeoPoint(30.0, 78.0)
    mission = SearchAndRescueMission(SearchMission("outside", home,
                                                   [SearchCell("A", GeoPoint(30.001, 78.001), 1000)]))
    mission.configure_terrain(dem)
    route = mission.replan()
    assert route[-1].kind.value == "return_home"
    assert "DEM coverage unavailable" in route[0].rationale


def test_geotiff_dem_loader_when_rasterio_is_available(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", width=10, height=10, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(78, 30.1, .01, .01),
                       nodata=-9999) as dataset:
        dataset.write(np.full((10, 10), 100, dtype=np.float32), 1)
    dem = load_geotiff(path)
    assert dem.source == "geotiff"
    assert dem.crs == "EPSG:4326"
    assert dem.sample(30.05, 77.95) == pytest.approx(100)

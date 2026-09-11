from __future__ import annotations

import json

import numpy as np
import pytest

from app.airspace import OpenSkyCache
from app.server import build_mission, readiness
from engine.search_rescue.change import register_previous
from engine.search_rescue.detectors import FrameMetadata
from engine.search_rescue.exports import export_packet
from engine.search_rescue.geolocation import geolocate_pixel
from engine.search_rescue.models import GeoPoint


def test_geospatial_affine_metadata_is_used_for_detection_coordinates():
    metadata = FrameMetadata("sat-1", GeoPoint(30.0, 78.0), image_width=100, image_height=100,
                             crs="EPSG:4326", geo_transform=(.001, 0, 78.0, 0, -.001, 30.1),
                             source_type="satellite", metadata_quality="georeferenced")
    point = geolocate_pixel(50, 50, 100, 100, metadata)
    assert point.lon == pytest.approx(78.0505)
    assert point.lat == pytest.approx(30.0495)


def test_registration_reports_transform_for_textured_passes():
    try:
        import cv2
    except ImportError:
        return
    before = np.zeros((180, 220), dtype=np.uint8)
    for x, y, radius in ((30, 30, 12), (80, 120, 18), (160, 60, 15), (190, 145, 11)):
        cv2.circle(before, (x, y), radius, 180, -1)
    cv2.rectangle(before, (105, 20), (135, 50), 240, -1)
    matrix = np.float32([[1, 0, 5], [0, 1, 3]])
    after = cv2.warpAffine(before, matrix, (before.shape[1], before.shape[0]))
    _, report = register_previous(before, after)
    assert report.registered
    assert report.matches >= 4
    assert report.method in {"orb-ransac-homography", "identity"}


def test_field_packet_exports_all_supported_formats():
    packet = build_mission().field_packet()
    for format_name, marker in (("geojson", "FeatureCollection"), ("csv", "record_type"),
                                ("kml", "<kml"), ("gpx", "<gpx")):
        content, content_type, extension = export_packet(packet, format_name)
        assert content
        assert marker in content
        assert extension == format_name
        assert content_type
    json.loads(export_packet(packet, "geojson")[0])


def test_readiness_and_opensky_are_explicitly_optional():
    status = readiness()
    assert status["simulation_only"] is True
    disabled = OpenSkyCache(enabled=False).snapshot()
    assert disabled["source"] == "disabled"
    assert disabled["available"] is False

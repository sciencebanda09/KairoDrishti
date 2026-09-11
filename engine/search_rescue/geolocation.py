"""Pixel-to-ground helpers for nadir image frames."""
from __future__ import annotations

from .detectors import FrameMetadata, _detection
from .models import DetectionBand


def geolocate_pixel(x: float, y: float, width: int, height: int, metadata: FrameMetadata):
    """Return a GPS point for a pixel centroid using the frame footprint."""
    # Reuse the detector's projection with a one-pixel candidate box.
    detection = _detection(metadata, (int(x), int(y), int(x), int(y), 1),
                           band=DetectionBand.RGB,
                           confidence=0.0, label="pixel", evidence="geolocation")
    return detection.location

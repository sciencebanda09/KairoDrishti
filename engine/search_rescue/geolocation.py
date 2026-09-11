"""Pixel-to-ground helpers for raster and aerial frames."""
from __future__ import annotations

from typing import Any

from .models import GeoPoint


def _transform_pixel(x: float, y: float, transform: tuple[float, float, float, float, float, float]) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    return a * x + b * y + c, d * x + e * y + f


def _project_crs(x: float, y: float, crs: str) -> tuple[float, float]:
    if not crs or crs.upper() in {"EPSG:4326", "OGC:CRS84", "WGS84"}:
        return x, y
    try:
        from rasterio.warp import transform
        lon, lat = transform(crs, "EPSG:4326", [x], [y])
        return lon[0], lat[0]
    except ImportError as exc:
        raise RuntimeError("rasterio is required to reproject non-WGS84 imagery") from exc


def geolocate_pixel(x: float, y: float, width: int, height: int, metadata: Any) -> GeoPoint:
    """Return a WGS84 ground point using a raster transform or aerial fallback."""
    if metadata.geo_transform is not None:
        east, north = _transform_pixel(x + .5, y + .5, metadata.geo_transform)
        lon, lat = _project_crs(east, north, metadata.crs)
        return GeoPoint(lat, lon, metadata.location.altitude_m)

    east = (x - width / 2) * metadata.ground_sample_distance_m
    north = (height / 2 - y) * metadata.ground_sample_distance_m
    import math
    angle = math.radians(metadata.heading_deg)
    rotated_east = east * math.cos(angle) + north * math.sin(angle)
    rotated_north = -east * math.sin(angle) + north * math.cos(angle)
    return GeoPoint(
        metadata.location.lat + rotated_north / 111_000,
        metadata.location.lon + rotated_east / (111_000 * max(math.cos(math.radians(metadata.location.lat)), .1)),
        metadata.location.altitude_m,
    )

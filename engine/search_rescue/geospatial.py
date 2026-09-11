"""Offline geospatial image ingestion and metadata normalization."""
from __future__ import annotations

import json
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .detectors import FrameMetadata
from .models import GeoPoint
from .vision import decode_array


@dataclass(frozen=True)
class IngestedFrame:
    array: np.ndarray
    metadata: FrameMetadata
    bounds: dict[str, float] | None
    source: str
    message: str


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metadata_dict(value: str | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("frame metadata must be a JSON object")
    return parsed


def _source_type(filename: str, metadata: dict[str, Any]) -> str:
    source = str(metadata.get("source_type", "")).lower().strip()
    if source:
        return source
    name = filename.lower()
    if "thermal" in name or "infrared" in name or name.endswith(".ir.npy"):
        return "thermal"
    if "multi" in name or "spectral" in name:
        return "multispectral"
    if "sat" in name or name.endswith((".tif", ".tiff")):
        return "satellite"
    return "drone"


def _normalise_band_name(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "")
    if text.startswith("B") and text[1:].isdigit():
        return f"B{int(text[1:])}"
    if text.startswith("B") and text[1:].startswith("0") and text[2:].isdigit():
        return f"B{int(text[1:])}"
    return text


def _band_names(values: dict[str, Any], dataset: Any = None) -> tuple[str, ...]:
    supplied = values.get("bands", values.get("band_names", ()))
    if isinstance(supplied, dict):
        supplied = [name for name, _ in sorted(supplied.items(), key=lambda item: int(item[1]))]
    if isinstance(supplied, str):
        supplied = [item.strip() for item in supplied.split(",") if item.strip()]
    supplied_names = [_normalise_band_name(item) for item in (supplied or ())]
    if dataset is not None:
        descriptions = list(dataset.descriptions or ())
        tags = [dataset.tags(index + 1) for index in range(dataset.count)]
        names = []
        for index in range(dataset.count):
            candidate = supplied_names[index] if index < len(supplied_names) else ""
            candidate = candidate or _normalise_band_name(descriptions[index] if index < len(descriptions) else "")
            candidate = candidate or _normalise_band_name(tags[index].get("band_name", tags[index].get("name", "")))
            names.append(candidate or f"BAND_{index + 1}")
        return tuple(names)
    return tuple(supplied_names)


def _point_from_metadata(metadata: dict[str, Any]) -> GeoPoint:
    location = metadata.get("location") or {}
    lat = metadata.get("lat", location.get("lat"))
    lon = metadata.get("lon", location.get("lon"))
    if lat is None or lon is None:
        raise ValueError("geospatial frame metadata requires lat and lon")
    return GeoPoint(_as_float(lat), _as_float(lon), _as_float(metadata.get("altitude_m", metadata.get("altitude", location.get("altitude_m", 0)))))


def _exif_metadata(raw: bytes) -> dict[str, Any]:
    """Read GPS EXIF values from JPEG/PNG when no sidecar overrides them."""
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(raw))
        gps = image.getexif().get(34853, {})
    except (ImportError, OSError):
        return {}
    if not gps:
        return {}

    def rational(value: Any) -> float:
        try:
            return float(value.numerator) / float(value.denominator)
        except AttributeError:
            return float(value)

    def coordinate(value: Any) -> float:
        degrees, minutes, seconds = (rational(item) for item in value)
        return degrees + minutes / 60 + seconds / 3600

    try:
        latitude = coordinate(gps[2])
        longitude = coordinate(gps[4])
        if str(gps.get(1, "N")).upper() == "S":
            latitude *= -1
        if str(gps.get(3, "E")).upper() == "W":
            longitude *= -1
        result: dict[str, Any] = {"lat": latitude, "lon": longitude}
        if 6 in gps:
            result["altitude_m"] = rational(gps[6]) * (-1 if gps.get(5, 0) else 1)
        if 17 in gps:
            result["heading_deg"] = rational(gps[17])
        return result
    except (KeyError, TypeError, ValueError):
        return {}


def _build_metadata(frame_id: str, location: GeoPoint, width: int, height: int,
                    values: dict[str, Any], *, crs: str = "EPSG:4326",
                    geo_transform: tuple[float, float, float, float, float, float] | None = None,
                    source_type: str = "drone", quality: str = "approximate",
                    band_names: tuple[str, ...] = (),
                    metadata_warnings: tuple[str, ...] = ()) -> FrameMetadata:
    cloud_value = values.get("cloud_percent", values.get("cloud_coverage", values.get("CLOUDY_PIXEL_PERCENTAGE")))
    try:
        cloud_percent = float(cloud_value) if cloud_value is not None else None
    except (TypeError, ValueError):
        cloud_percent = None
    return FrameMetadata(
        frame_id,
        location,
        _as_float(values.get("heading_deg", values.get("yaw_deg", 0))),
        _as_float(values.get("ground_sample_distance_m", values.get("gsd_m", .25)), .25),
        int(width), int(height),
        str(values.get("crs", crs)), geo_transform,
        _as_float(values.get("pitch_deg", values.get("camera_pitch_deg", -90)), -90),
        _as_float(values.get("roll_deg", values.get("camera_roll_deg", 0))),
        _as_float(values.get("focal_length_px", 0)),
        source_type,
        str(values.get("timestamp", "")),
        quality,
        tuple(band_names), tuple(metadata_warnings), cloud_percent,
        str(values.get("processing_level", values.get("product_level", ""))),
        )


def _rasterio_frame(raw: bytes, filename: str, values: dict[str, Any]) -> IngestedFrame:
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.warp import transform, transform_bounds
    except ImportError as exc:
        raise RuntimeError("rasterio is required for GeoTIFF/COG ingestion; install requirements.txt") from exc

    with MemoryFile(raw) as memory:
        with memory.open() as dataset:
            bands = dataset.read()
            array = bands[0] if bands.shape[0] == 1 else np.moveaxis(bands, 0, -1)
            dataset_values = {**dataset.tags(), **values}
            source_type = _source_type(filename, dataset_values)
            supplied_band_names = bool(values.get("bands") or values.get("band_names"))
            names = _band_names(dataset_values, dataset)
            warnings: list[str] = []
            if source_type == "satellite" and not supplied_band_names and bands.shape[0] == 4:
                names = ("B2", "B3", "B4", "B8")
                warnings.append("band order assumed B2,B3,B4,B8; provide metadata to remove this warning")
            elif source_type == "satellite" and not supplied_band_names:
                warnings.append("Sentinel band names are missing; provide a bands metadata list")
            transform_obj = dataset.transform
            transform_values = (transform_obj.a, transform_obj.b, transform_obj.c,
                                transform_obj.d, transform_obj.e, transform_obj.f)
            crs = str(dataset.crs or "EPSG:4326")
            center_x = (dataset.bounds.left + dataset.bounds.right) / 2
            center_y = (dataset.bounds.bottom + dataset.bounds.top) / 2
            if crs.upper() not in {"EPSG:4326", "OGC:CRS84", "WGS84"}:
                center_x, center_y = transform(crs, "EPSG:4326", [center_x], [center_y])
                center_lon, center_lat = center_x[0], center_y[0]
            else:
                center_lon, center_lat = center_x, center_y
            if "ground_sample_distance_m" not in dataset_values and "gsd_m" not in dataset_values:
                dataset_values["ground_sample_distance_m"] = (
                    abs(float(dataset.res[0])) if dataset.crs.is_projected else
                    abs(float(dataset.res[0])) * 111_000 * max(float(np.cos(np.radians(center_lat))), .1)
                )
            location = GeoPoint(center_lat, center_lon, _as_float(values.get("altitude_m", 0)))
            meta = _build_metadata(Path(filename).stem, location, array.shape[1], array.shape[0], dataset_values,
                                   crs=crs, geo_transform=transform_values,
                                   source_type=source_type,
                                   quality="georeferenced" if not warnings else "georeferenced-with-warnings",
                                   band_names=names, metadata_warnings=tuple(warnings))
            if crs.upper() in {"EPSG:4326", "OGC:CRS84", "WGS84"}:
                bounds = {"west": float(dataset.bounds.left), "south": float(dataset.bounds.bottom),
                          "east": float(dataset.bounds.right), "north": float(dataset.bounds.top), "crs": crs}
            else:
                west, south, east, north = transform_bounds(crs, "EPSG:4326", *dataset.bounds, densify_pts=21)
                bounds = {"west": float(west), "south": float(south),
                          "east": float(east), "north": float(north), "crs": "EPSG:4326"}
    message = "GeoTIFF geotransform loaded"
    if warnings:
        message += "; " + " ".join(warnings)
    return IngestedFrame(np.asarray(array), meta, bounds, "geotiff", message)


def ingest_frame(raw: bytes, filename: str = "", metadata: str | dict[str, Any] | None = None) -> IngestedFrame:
    """Decode a raster upload and return one normalized, geolocatable frame."""
    values = _metadata_dict(metadata)
    suffix = Path(filename).suffix.lower()
    if suffix in {".tif", ".tiff", ".cog"}:
        return _rasterio_frame(raw, filename, values)
    array = decode_array(raw, filename)
    exif = _exif_metadata(raw) if suffix in {".jpg", ".jpeg", ".png"} else {}
    values = {**exif, **values}
    location = _point_from_metadata(values)
    source_type = _source_type(filename, values)
    quality = "exif-gps" if "lat" in exif and "lon" in exif else "camera-metadata" if values.get("heading_deg") is not None else "approximate"
    meta = _build_metadata(Path(filename).stem or "frame-0001", location,
                           array.shape[1], array.shape[0], values,
                           source_type=source_type,
                           quality=quality,
                           band_names=_band_names(values))
    return IngestedFrame(np.asarray(array), meta, None, "image", "Image decoded with supplied camera metadata")


def metadata_json(metadata: FrameMetadata, bounds: dict[str, float] | None = None) -> dict[str, Any]:
    result = {
        "frame_id": metadata.frame_id,
        "lat": metadata.location.lat,
        "lon": metadata.location.lon,
        "altitude_m": metadata.location.altitude_m,
        "heading_deg": metadata.heading_deg,
        "ground_sample_distance_m": metadata.ground_sample_distance_m,
        "image_width": metadata.image_width,
        "image_height": metadata.image_height,
        "crs": metadata.crs,
        "source_type": metadata.source_type,
        "timestamp": metadata.timestamp,
        "metadata_quality": metadata.metadata_quality,
        "band_names": list(metadata.band_names),
        "metadata_warnings": list(metadata.metadata_warnings),
        "cloud_percent": metadata.cloud_percent,
        "processing_level": metadata.processing_level,
    }
    if metadata.geo_transform is not None:
        result["geo_transform"] = list(metadata.geo_transform)
    if bounds is not None:
        result["bounds"] = bounds
    return result

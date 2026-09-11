"""Local Sentinel-2 context products for prioritisation, not human detection."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np

from .detectors import FrameMetadata


SENTINEL_CONTEXT_BANDS = ("B2", "B3", "B4", "B8")


def _normalise_name(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "")
    if text.startswith("B") and text[1:].isdigit():
        return f"B{int(text[1:])}"
    return text


def _channels(image: Any) -> np.ndarray:
    array = np.asarray(image).astype(np.float32)
    if array.ndim != 3:
        raise ValueError("satellite context expects an HxWxB raster")
    # NPY workflows sometimes use band-first arrays. GeoTIFF ingestion is
    # already band-last, so only transpose the unambiguous small-first case.
    if array.shape[0] <= 16 and array.shape[2] > 16:
        array = np.moveaxis(array, 0, -1)
    if array.shape[2] < 2:
        raise ValueError("satellite context requires at least two bands")
    return array


def _band_map(array: np.ndarray, metadata: FrameMetadata) -> tuple[dict[str, int], list[str]]:
    names = [_normalise_name(name) for name in metadata.band_names]
    warnings = list(metadata.metadata_warnings)
    if not names and array.shape[2] == 4:
        names = list(SENTINEL_CONTEXT_BANDS)
        warnings.append("band order assumed B2,B3,B4,B8")
    if len(names) != array.shape[2]:
        raise ValueError(f"band metadata has {len(names)} names for {array.shape[2]} raster bands")
    mapping = {name: index for index, name in enumerate(names) if name}
    if not mapping or all(name.startswith("BAND_") for name in mapping):
        raise ValueError("Sentinel band names are required; provide metadata bands such as B2,B3,B4,B8")
    return mapping, warnings


def _finite_band(array: np.ndarray, index: int) -> np.ndarray:
    band = np.asarray(array[..., index], dtype=np.float32)
    return np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)


def _stretch(stack: np.ndarray) -> np.ndarray:
    result = np.zeros_like(stack, dtype=np.float32)
    for index in range(stack.shape[2]):
        band = stack[..., index]
        valid = band[np.isfinite(band)]
        if not valid.size:
            continue
        low, high = np.percentile(valid, [2, 98])
        result[..., index] = np.clip((band - low) / max(float(high - low), 1e-6), 0, 1)
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def _preview_data_url(image: np.ndarray, *, ndvi: bool = False) -> str:
    from PIL import Image

    if ndvi:
        index = np.clip(np.nan_to_num(image, nan=0.0), -1, 1)
        preview = np.zeros((*index.shape, 3), dtype=np.uint8)
        preview[..., 0] = np.clip((-index + .15) * 210, 0, 255).astype(np.uint8)
        preview[..., 1] = np.clip((index + .15) * 210, 0, 255).astype(np.uint8)
        preview[..., 2] = 35
    else:
        preview = image
    height, width = preview.shape[:2]
    scale = min(1.0, 320 / max(width, height))
    if scale < 1:
        preview = np.asarray(Image.fromarray(preview).resize(
            (max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR))
    output = io.BytesIO()
    Image.fromarray(preview).save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _pixel_for_location(metadata: FrameMetadata, lat: float, lon: float) -> tuple[float, float] | None:
    if metadata.geo_transform is None:
        return None
    x, y = lon, lat
    if metadata.crs and metadata.crs.upper() not in {"EPSG:4326", "OGC:CRS84", "WGS84"}:
        try:
            from rasterio.warp import transform
            xs, ys = transform("EPSG:4326", metadata.crs, [lon], [lat])
            x, y = xs[0], ys[0]
        except ImportError as exc:
            raise RuntimeError("rasterio is required for projected satellite context") from exc
    a, b, c, d, e, f = metadata.geo_transform
    determinant = a * e - b * d
    if abs(determinant) < 1e-12:
        return None
    dx, dy = x - c, y - f
    return ((e * dx - b * dy) / determinant,
            (-d * dx + a * dy) / determinant)


@dataclass(frozen=True)
class SatelliteContext:
    image: np.ndarray
    metadata: FrameMetadata
    bounds: dict[str, float] | None
    band_map: dict[str, int]
    warnings: tuple[str, ...]
    rgb_preview: str | None
    false_colour_preview: str | None
    ndvi_preview: str | None
    ndvi: np.ndarray | None
    cloud_percent: float | None

    @property
    def products(self) -> tuple[str, ...]:
        products: list[str] = []
        if {"B2", "B3", "B4"}.issubset(self.band_map):
            products.append("rgb")
        if {"B3", "B4", "B8"}.issubset(self.band_map):
            products.append("false_colour")
        if {"B4", "B8"}.issubset(self.band_map):
            products.append("ndvi")
        return tuple(products)

    @property
    def quality(self) -> str:
        return "ready" if not self.warnings else "warning"

    def summary(self, *, include_previews: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source": "sentinel-2-context",
            "frame_id": self.metadata.frame_id,
            "products": list(self.products),
            "band_map": {name: index + 1 for name, index in self.band_map.items()},
            "band_names": list(self.metadata.band_names),
            "warnings": list(self.warnings),
            "quality": self.quality,
            "cloud_percent": self.cloud_percent,
            "timestamp": self.metadata.timestamp,
            "resolution_m": self.metadata.ground_sample_distance_m,
            "crs": self.metadata.crs,
            "bounds": self.bounds,
            "metadata_quality": self.metadata.metadata_quality,
        }
        if include_previews:
            result.update({"rgb_preview": self.rgb_preview,
                           "false_colour_preview": self.false_colour_preview,
                           "ndvi_preview": self.ndvi_preview})
        return result

    def cell_features(self, lat: float, lon: float) -> dict[str, float] | None:
        if self.ndvi is None:
            return None
        pixel = _pixel_for_location(self.metadata, lat, lon)
        if pixel is None:
            return None
        x, y = pixel
        if x < 0 or y < 0 or x >= self.ndvi.shape[1] or y >= self.ndvi.shape[0]:
            return None
        value = float(self.ndvi[int(round(y)), int(round(x))])
        if not np.isfinite(value):
            return None
        normalized = float(np.clip((value + 1) / 2, 0, 1))
        # Dense vegetation is a visibility penalty, not evidence of a person.
        return {"ndvi": round(value, 4), "visibility_score": round(1 - normalized, 4)}


def build_satellite_context(image: Any, metadata: FrameMetadata,
                            bounds: dict[str, float] | None = None) -> SatelliteContext:
    array = _channels(image)
    mapping, warnings = _band_map(array, metadata)
    rgb = false_colour = ndvi = None
    rgb_preview = false_colour_preview = ndvi_preview = None
    if {"B2", "B3", "B4"}.issubset(mapping):
        rgb = _stretch(np.stack([_finite_band(array, mapping["B4"]),
                                 _finite_band(array, mapping["B3"]),
                                 _finite_band(array, mapping["B2"])], axis=-1))
        rgb_preview = _preview_data_url(rgb)
    if {"B3", "B4", "B8"}.issubset(mapping):
        false_colour = _stretch(np.stack([_finite_band(array, mapping["B8"]),
                                          _finite_band(array, mapping["B4"]),
                                          _finite_band(array, mapping["B3"])], axis=-1))
        false_colour_preview = _preview_data_url(false_colour)
    if {"B4", "B8"}.issubset(mapping):
        red = _finite_band(array, mapping["B4"])
        nir = _finite_band(array, mapping["B8"])
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
        ndvi = np.clip(ndvi, -1, 1)
        ndvi_preview = _preview_data_url(ndvi, ndvi=True)
    if not (rgb_preview or false_colour_preview or ndvi_preview):
        raise ValueError("satellite raster must include B2/B3/B4/B8 for a context product")
    return SatelliteContext(np.asarray(array), metadata, bounds, mapping, tuple(warnings),
                            rgb_preview, false_colour_preview, ndvi_preview, ndvi,
                            metadata.cloud_percent)

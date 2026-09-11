"""Small offline image detectors used by the mission demo.

These are deliberately conservative pixel baselines. They provide a real local
RGB/thermal path while keeping the detector interface replaceable by a learned
model adapter later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .models import AerialDetection, DetectionBand, GeoPoint
from .change import detect_changes


@dataclass(frozen=True)
class FrameMetadata:
    frame_id: str
    location: GeoPoint
    heading_deg: float = 0.0
    ground_sample_distance_m: float = 0.25


def _components(mask: np.ndarray, minimum: int = 12) -> list[tuple[int, int, int, int, int]]:
    labels, count = ndimage.label(mask)
    result = []
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        if len(xs) < minimum:
            continue
        result.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), len(xs)))
    return result


def _detection(meta: FrameMetadata, box: tuple[int, int, int, int, int], band: DetectionBand,
               confidence: float, label: str, evidence: str) -> AerialDetection:
    x0, y0, x1, y1, _ = box
    height = max(y1 - y0 + 1, 1)
    width = max(x1 - x0 + 1, 1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # Camera image coordinates: x right/east, y down/south. Rotate by heading.
    east = (cx - width / 2) * meta.ground_sample_distance_m
    north = (height / 2 - cy) * meta.ground_sample_distance_m
    angle = np.deg2rad(meta.heading_deg)
    rotated_east = east * np.cos(angle) + north * np.sin(angle)
    rotated_north = -east * np.sin(angle) + north * np.cos(angle)
    location = GeoPoint(
        meta.location.lat + rotated_north / 111_000,
        meta.location.lon + rotated_east / (111_000 * max(np.cos(np.deg2rad(meta.location.lat)), .1)),
        meta.location.altitude_m,
    )
    return AerialDetection(f"{meta.frame_id}-{band.value}-{x0}-{y0}", location, label,
                           min(1.0, max(0.0, confidence)), band, meta.frame_id, evidence)


def detect_rgb(image: Any, meta: FrameMetadata) -> list[AerialDetection]:
    array = np.asarray(image).astype(np.float32)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("RGB detector expects an HxWx3 image")
    rgb = array[..., :3] / (255.0 if array.max() > 1.5 else 1.0)
    luminance = rgb.mean(axis=2)
    background = ndimage.gaussian_filter(luminance, sigma=9)
    contrast = np.abs(luminance - background)
    threshold = max(float(np.quantile(contrast, .985)), .12)
    boxes = _components(contrast > threshold, minimum=max(10, array.shape[0] * array.shape[1] // 1800))
    results = []
    for box in boxes[:8]:
        score = float(np.clip(.55 + contrast[box[1]:box[3] + 1, box[0]:box[2] + 1].mean(), .55, .94))
        results.append(_detection(meta, box, DetectionBand.RGB, score, "possible person", "RGB contrast candidate"))
    return results


def detect_thermal(image: Any, meta: FrameMetadata) -> list[AerialDetection]:
    array = np.asarray(image).astype(np.float32)
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.ndim != 2:
        raise ValueError("thermal detector expects a 2D grayscale or temperature array")
    low, high = float(np.quantile(array, .70)), float(np.quantile(array, .995))
    spread = max(high - low, 1e-6)
    mask = array >= low + spread * .72
    boxes = _components(mask, minimum=max(6, array.size // 3000))
    return [_detection(meta, box, DetectionBand.THERMAL,
                       float(np.clip(.55 + (array[box[1]:box[3] + 1, box[0]:box[2] + 1].mean() - low) / spread * .35, .55, .98)),
                       "possible heat source", "thermal hot-region candidate") for box in boxes[:8]]


def detect_multispectral(image: Any, meta: FrameMetadata) -> list[AerialDetection]:
    array = np.asarray(image).astype(np.float32)
    if array.ndim != 3 or array.shape[2] < 2:
        raise ValueError("multispectral detector expects an HxWxB array with at least 2 bands")
    # A normalized band-ratio anomaly is useful for vegetation/clothing/sign cues.
    numerator, denominator = array[..., -1], np.maximum(array[..., 0], 1e-6)
    anomaly = np.abs((numerator - denominator) / (numerator + denominator + 1e-6))
    threshold = max(float(np.quantile(anomaly, .985)), .25)
    boxes = _components(anomaly >= threshold, minimum=max(8, array.shape[0] * array.shape[1] // 2200))
    return [_detection(meta, box, DetectionBand.MULTISPECTRAL, .62,
                       "human-made sign", "multispectral band-ratio anomaly") for box in boxes[:8]]


def detect_change(previous: Any, current: Any, meta: FrameMetadata) -> list[AerialDetection]:
    mask = detect_changes(np.asarray(previous), np.asarray(current), threshold=.18)
    boxes = _components(mask, minimum=max(6, mask.size // 3000))
    return [_detection(meta, box, DetectionBand.RGB, .68,
                       "new or moved human sign", "successive-pass change") for box in boxes[:8]]

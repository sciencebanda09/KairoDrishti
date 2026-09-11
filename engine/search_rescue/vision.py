"""Shared offline image decoding and OpenCV normalization helpers."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - dependency is declared, fallback keeps imports clear
    cv2 = None


def opencv_available() -> bool:
    return cv2 is not None


def decode_array(raw: bytes, filename: str = "") -> np.ndarray:
    """Decode .npy and common raster uploads into numpy arrays."""
    if Path(filename).suffix.lower() == ".npy":
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if cv2 is None:
        raise RuntimeError("OpenCV is required to decode raster uploads")
    encoded = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError(f"could not decode image upload: {filename or 'unnamed file'}")
    if decoded.ndim == 3 and decoded.shape[2] == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    elif decoded.ndim == 3 and decoded.shape[2] == 4:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
    return np.asarray(decoded)


def prepare_rgb(image: Any) -> np.ndarray:
    """Return a contiguous uint8 RGB image suitable for YOLO inference."""
    array = np.asarray(image)
    if array.ndim == 2:
        if cv2 is not None:
            array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
        else:
            low, high = float(array.min()), float(array.max())
            array = (array - low) / max(high - low, 1e-6) * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
        if cv2 is not None:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        else:
            array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3:
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.shape[2] >= 3:
            array = array[..., :3]
        else:
            raise ValueError("RGB input must have one or at least three channels")
    else:
        raise ValueError("RGB input must be a 2-D or 3-D image")

    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if float(np.nanmax(array)) <= 1.5:
            array *= 255.0
        else:
            low, high = np.nanpercentile(array, [1, 99])
            array = (array - low) / max(float(high - low), 1e-6) * 255.0
        array = np.clip(np.nan_to_num(array), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def refine_change_mask(mask: np.ndarray) -> np.ndarray:
    """Remove isolated pixels and close small gaps in a binary change mask."""
    binary = (np.asarray(mask).astype(np.uint8) * 255)
    if cv2 is None:
        return binary > 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned > 0

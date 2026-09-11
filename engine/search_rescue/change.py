from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is a declared runtime dependency
    cv2 = None


def _normalize_pair(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize common uint/float image ranges before differencing.

    Integer imagery is normalized against its storage range so 8-bit 0..255
    and 16-bit thermal 0..65535 data remain comparable. Float imagery already
    in 0..1 is preserved; other float ranges use a shared robust range so a
    global exposure change is not erased by independently scaling each pass.
    """
    before = np.asarray(previous, dtype=np.float32)
    after = np.asarray(current, dtype=np.float32)
    if np.issubdtype(np.asarray(previous).dtype, np.integer) or np.issubdtype(np.asarray(current).dtype, np.integer):
        maximum = max(float(before.max()), float(after.max()))
        if maximum <= 255:
            return np.clip(before / 255.0, 0.0, 1.0), np.clip(after / 255.0, 0.0, 1.0)
    maximum = max(float(np.nanmax(before)), float(np.nanmax(after)))
    if maximum <= 1.0:
        return np.clip(before, 0.0, 1.0), np.clip(after, 0.0, 1.0)
    finite = np.concatenate([before[np.isfinite(before)], after[np.isfinite(after)]])
    low, high = np.percentile(finite, [1, 99])
    scale = max(float(high - low), 1e-6)
    return np.clip((before - low) / scale, 0.0, 1.0), np.clip((after - low) / scale, 0.0, 1.0)


def detect_changes(previous: np.ndarray, current: np.ndarray, *, threshold: float = 0.18) -> np.ndarray:
    """Return a boolean change mask for aligned RGB/thermal/multispectral tiles."""
    before = np.asarray(previous)
    after = np.asarray(current)
    if before.shape != after.shape:
        raise ValueError("successive-pass imagery must have identical shapes")
    if before.ndim not in (2, 3):
        raise ValueError("imagery must be a 2-D band or 3-D band stack")
    before, after = _normalize_pair(before, after)
    delta = np.abs(after - before)
    if delta.ndim == 3:
        delta = delta.mean(axis=2)
    return delta >= threshold


@dataclass(frozen=True)
class RegistrationReport:
    method: str
    matches: int
    inliers: int
    score: float
    matrix: tuple[float, ...] | None
    registered: bool
    message: str


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array.mean(axis=2)
    array = array.astype(np.float32)
    low, high = np.nanpercentile(array, [1, 99])
    array = np.clip((array - low) / max(float(high - low), 1e-6) * 255, 0, 255)
    return np.nan_to_num(array).astype(np.uint8)


def register_previous(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, RegistrationReport]:
    """Align the previous pass to the current pass using ORB/RANSAC.

    An unreliable transform returns the original previous image and a report
    with ``registered=False`` so callers can surface the degraded result
    instead of presenting an unregistered comparison as ground truth.
    """
    before = np.asarray(previous)
    after = np.asarray(current)
    if before.ndim not in (2, 3) or after.ndim not in (2, 3):
        raise ValueError("successive-pass imagery must be 2-D or 3-D")
    if cv2 is None:
        return before, RegistrationReport("none", 0, 0, 0.0, None, False, "OpenCV is unavailable")
    if before.shape[:2] != after.shape[:2]:
        before = cv2.resize(before, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_LINEAR)
    before_gray, after_gray = _gray_uint8(before), _gray_uint8(after)
    orb = cv2.ORB_create(nfeatures=1200, fastThreshold=8)
    key_before, desc_before = orb.detectAndCompute(before_gray, None)
    key_after, desc_after = orb.detectAndCompute(after_gray, None)
    if desc_before is None or desc_after is None or len(key_before) < 4 or len(key_after) < 4:
        return before, RegistrationReport("identity", 0, 0, 0.0, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), False,
                                          "no stable features; identity alignment used")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_before, desc_after, k=2)
    good = [first for first, second in pairs if first.distance < .78 * second.distance]
    if len(good) < 4:
        return before, RegistrationReport("identity", len(good), 0, 0.0, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), False,
                                          "not enough reliable matches; identity alignment used")
    source = np.float32([key_before[item.queryIdx].pt for item in good])
    target = np.float32([key_after[item.trainIdx].pt for item in good])
    matrix, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    score = inliers / max(len(good), 1)
    if matrix is None or inliers < 4 or score < .28:
        return before, RegistrationReport("identity", len(good), inliers, score,
                                          (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                                          False, "homography confidence below threshold; identity alignment used")
    height, width = after.shape[:2]
    registered = cv2.warpPerspective(before, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    return registered, RegistrationReport("orb-ransac-homography", len(good), inliers, score,
                                          tuple(float(item) for item in matrix.flatten()),
                                          True, "previous pass registered to current pass")


def detect_registered_changes(previous: np.ndarray, current: np.ndarray, *, threshold: float = .18) -> tuple[np.ndarray, RegistrationReport]:
    """Register two passes before producing a change mask."""
    registered, report = register_previous(previous, current)
    return detect_changes(registered, current, threshold=threshold), report

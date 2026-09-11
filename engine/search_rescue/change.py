from __future__ import annotations

import numpy as np


def detect_changes(previous: np.ndarray, current: np.ndarray, *, threshold: float = 0.18) -> np.ndarray:
    """Return a boolean change mask for aligned RGB/thermal/multispectral tiles."""
    before = np.asarray(previous, dtype=np.float32)
    after = np.asarray(current, dtype=np.float32)
    if before.shape != after.shape:
        raise ValueError("successive-pass imagery must have identical shapes")
    if before.ndim not in (2, 3):
        raise ValueError("imagery must be a 2-D band or 3-D band stack")
    delta = np.abs(after - before)
    if delta.ndim == 3:
        delta = delta.mean(axis=2)
    return delta >= threshold

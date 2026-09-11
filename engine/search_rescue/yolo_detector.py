"""Optional local Ultralytics YOLO adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .detectors import FrameMetadata, _detection
from .detector_labels import TERRAIN_CLASSES
from .models import AerialDetection, DetectionBand
from .vision import prepare_rgb


class YoloRgbDetector:
    """Run a locally supplied YOLO checkpoint on RGB frames.

    Ultralytics is imported lazily so classical/offline mode still works when
    the optional package or checkpoint is unavailable.
    """

    def __init__(self, model_path: str | Path, *, confidence: float = .35,
                 iou: float = .5, device: str = "auto",
                 allowed_classes: tuple[str, ...] = TERRAIN_CLASSES) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.allowed_classes = tuple(allowed_classes)
        self.model: Any = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def class_names(self) -> tuple[str, ...]:
        names = getattr(self.model, "names", {}) if self.model is not None else {}
        values = tuple(names.values()) if isinstance(names, dict) else tuple(names) if names else ()
        return tuple(str(value).strip().lower() for value in values)

    @property
    def terrain_ready(self) -> bool:
        return all(label in self.class_names for label in self.allowed_classes)

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {self.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("install ultralytics to enable YOLO mode") from exc
        self.model = YOLO(str(self.model_path))

    def detect(self, image: np.ndarray, metadata: FrameMetadata) -> list[AerialDetection]:
        if self.model is None:
            self.load()
        array = prepare_rgb(image)
        results = self.model.predict(source=array[..., :3], conf=self.confidence,
                                     iou=self.iou, device=None if self.device == "auto" else self.device,
                                     verbose=False)
        detections: list[AerialDetection] = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
            confidences = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
            class_ids = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls)
            for box, confidence, class_id in zip(xyxy.tolist(), confidences.tolist(), class_ids.tolist()):
                label = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else names[int(class_id)]
                normalized_label = str(label).strip().lower()
                if normalized_label not in self.allowed_classes or float(confidence) < self.confidence:
                    continue
                x0, y0, x1, y1 = (int(round(value)) for value in box)
                x0 = max(0, min(x0, array.shape[1] - 1)); x1 = max(0, min(x1, array.shape[1] - 1))
                y0 = max(0, min(y0, array.shape[0] - 1)); y1 = max(0, min(y1, array.shape[0] - 1))
                if x1 <= x0 or y1 <= y0:
                    continue
                detections.append(_detection(
                    metadata, (x0, y0, x1, y1, max(1, (x1 - x0) * (y1 - y0))),
                    DetectionBand.RGB, float(confidence), normalized_label,
                    f"YOLO {normalized_label}; model={self.model_path.name}; device={self.device}",
                ))
        return detections

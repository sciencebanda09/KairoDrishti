from __future__ import annotations

from collections import defaultdict
from math import hypot

from .models import AerialDetection, DetectionBand


def fuse_detections(
    detections: list[AerialDetection], *, radius_m: float = 35.0
) -> list[AerialDetection]:
    """Fuse nearby cross-band detections without requiring a cloud service.

    Thermal and multispectral evidence receives a small corroboration bonus;
    labels are retained so ground teams can understand why a point was raised.
    Coordinates are treated as a local metre-like projection for clustering.
    """
    groups: list[list[AerialDetection]] = []
    for detection in detections:
        for group in groups:
            anchor = group[0].location
            # Good enough for the short distances covered by one image tile.
            if hypot((detection.location.lat - anchor.lat) * 111_000,
                     (detection.location.lon - anchor.lon) * 111_000) <= radius_m:
                group.append(detection)
                break
        else:
            groups.append([detection])

    fused: list[AerialDetection] = []
    for index, group in enumerate(groups, 1):
        bands = {item.band for item in group}
        confidence = max(item.confidence for item in group)
        confidence = min(1.0, confidence + 0.12 * max(0, len(bands) - 1))
        labels = ", ".join(sorted({item.label for item in group}))
        evidence = " + ".join(sorted({item.band.value for item in group}))
        fused.append(AerialDetection(
            detection_id=f"FUSED-{index:03d}",
            location=group[0].location,
            label=labels,
            confidence=confidence,
            band=DetectionBand.THERMAL if DetectionBand.THERMAL in bands else group[0].band,
            source_frame=", ".join(item.source_frame for item in group if item.source_frame),
            evidence=f"corroborated by {evidence}",
        ))
    return sorted(fused, key=lambda item: item.confidence, reverse=True)

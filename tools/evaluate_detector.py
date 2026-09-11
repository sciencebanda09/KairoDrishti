"""Evaluate a terrain-search YOLO checkpoint and write reproducible metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.search_rescue.detector_labels import TERRAIN_CLASSES, validate_class_names


def evaluate(model_path: Path, data_path: Path, output: Path, device: str = "cpu") -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = getattr(model, "names", {})
    classes = validate_class_names(names)
    metrics = model.val(data=str(data_path), split="test", device=device, verbose=False)
    box = getattr(metrics, "box", metrics)
    report = {
        "model": str(model_path.resolve()),
        "data": str(data_path.resolve()),
        "classes": list(classes),
        "required_classes": list(TERRAIN_CLASSES),
        "metrics": {
            "map50": float(getattr(box, "map50", 0.0)),
            "map50_95": float(getattr(box, "map", 0.0)),
            "precision": float(getattr(box, "mp", 0.0)),
            "recall": float(getattr(box, "mr", 0.0)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation/terrain-detector.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model, args.data, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()

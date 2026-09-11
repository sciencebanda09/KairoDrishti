"""Train a local terrain-search YOLO checkpoint from a prepared dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.search_rescue.detector_labels import TERRAIN_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", default="models/kairodristi-llvip-person.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--required-classes", nargs="+", default=list(TERRAIN_CLASSES),
                        choices=list(TERRAIN_CLASSES))
    args = parser.parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"YOLO dataset config not found: {data_path}")
    config = data_path.read_text(encoding="utf-8")
    if "val: images/val" not in config:
        raise ValueError("YOLO dataset config must define val: images/val; run split_yolo_dataset.py first")
    val_dir = data_path.parent / "images" / "val"
    if not val_dir.is_dir() or not any(val_dir.iterdir()):
        raise FileNotFoundError(f"validation split is missing or empty: {val_dir}")
    names = {line.split(":", 1)[1].strip() for line in config.splitlines()
             if ":" in line and line.strip().split(":", 1)[0].isdigit()}
    missing = [label for label in args.required_classes if label not in names]
    if missing:
        raise ValueError(f"dataset is missing terrain classes: {', '.join(missing)}")
    from ultralytics import YOLO
    model = YOLO(args.base_model)
    run_name = Path(args.output).stem
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=str(Path(args.output).parent), name=run_name,
                exist_ok=True)
    best = Path(args.output).parent / run_name / "weights" / "best.pt"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        best.replace(args.output)
    print(f"saved {args.output}; classes={','.join(args.required_classes)}")


if __name__ == "__main__":
    main()

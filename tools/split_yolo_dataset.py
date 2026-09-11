"""Create a portable YOLO dataset with a deterministic validation split."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.search_rescue.detector_labels import TERRAIN_CLASSES


def split_dataset(source: Path, output: Path, val_fraction: float = 0.15, seed: int = 42,
                  classes: tuple[str, ...] = TERRAIN_CLASSES) -> dict:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    source_images = source / "images" / "train"
    source_labels = source / "labels" / "train"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError("source must contain images/train and labels/train")

    images = sorted(source_images.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no JPG images found in {source_images}")

    pairs = []
    missing_labels = []
    for image in images:
        label = source_labels / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
        else:
            missing_labels.append(image.name)

    if not pairs:
        raise FileNotFoundError("no image/label pairs found")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))
    validation = set(shuffled[:val_count])

    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for image, label in pairs:
        split = "val" if (image, label) in validation else "train"
        shutil.copy2(image, output / "images" / split / image.name)
        shutil.copy2(label, output / "labels" / split / label.name)
        counts[split] += 1

    (output / "dataset.yaml").write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(classes)),
        encoding="utf-8",
    )
    report = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "seed": seed,
        "val_fraction": val_fraction,
        "images": counts,
        "missing_labels": missing_labels,
        "classes": list(classes),
    }
    (output / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", nargs="+", default=list(TERRAIN_CLASSES),
                        choices=list(TERRAIN_CLASSES))
    args = parser.parse_args()
    print(json.dumps(split_dataset(args.source, args.output, args.val_fraction, args.seed, tuple(args.classes)), indent=2))


if __name__ == "__main__":
    main()

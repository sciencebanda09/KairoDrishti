"""Build a tiny deterministic terrain-sign fixture for detector smoke tests.

This is not a substitute for field data. It makes the train/evaluate pipeline
reproducible in CI and provides positive examples for all required classes.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from engine.search_rescue.detector_labels import TERRAIN_CLASSES


def _box(draw: ImageDraw.ImageDraw, image: Image.Image, class_id: int, rng: random.Random) -> tuple[float, float, float, float]:
    width, height = image.size
    cx, cy = rng.randint(40, width - 40), rng.randint(40, height - 40)
    bw, bh = rng.randint(18, 56), rng.randint(22, 72)
    x0, y0, x1, y1 = cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2
    colors = [(235, 235, 235), (226, 72, 52), (220, 185, 45), (80, 55, 38)]
    if class_id == 0:
        draw.ellipse((cx - 7, y0, cx + 7, y0 + 14), fill=colors[class_id])
        draw.rectangle((cx - 10, y0 + 12, cx + 10, y1), fill=colors[class_id])
    elif class_id == 1:
        draw.rectangle((x0, y0, x1, y1), fill=colors[class_id])
    elif class_id == 2:
        draw.polygon(((x0, y1), (cx, y0), (x1, y1)), fill=colors[class_id])
    else:
        for index in range(4):
            px = cx - bw // 2 + index * max(4, bw // 3)
            draw.ellipse((px, cy - 4, px + 8, cy + 4), fill=colors[class_id])
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)


def build(output: Path, count: int = 60, seed: int = 42) -> dict:
    rng = random.Random(seed)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    for index in range(count):
        split = "test" if index % 10 == 0 else "val" if index % 5 == 0 else "train"
        pixels = np.zeros((384, 512, 3), dtype=np.uint8)
        pixels[..., 0] = rng.randint(35, 75)
        pixels[..., 1] = rng.randint(65, 125)
        pixels[..., 2] = rng.randint(35, 70)
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        labels = []
        for class_id in range(len(TERRAIN_CLASSES)):
            labels.append((class_id, *_box(draw, image, class_id, rng)))
        stem = f"terrain-{index:04d}"
        image.save(output / "images" / split / f"{stem}.jpg", quality=90)
        (output / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}" for class_id, cx, cy, width, height in labels) + "\n",
            encoding="utf-8",
        )
        counts[split] += 1
    dataset_root = output.resolve().as_posix()
    (output / "dataset.yaml").write_text(
        f"path: {dataset_root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" +
        "".join(f"  {index}: {name}\n" for index, name in enumerate(TERRAIN_CLASSES)),
        encoding="utf-8",
    )
    return {"output": str(output.resolve()), "classes": list(TERRAIN_CLASSES), "images": counts, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/terrain_fixture"))
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import json
    print(json.dumps(build(args.output, args.count, args.seed), indent=2))


if __name__ == "__main__":
    main()

"""GPU training entrypoint executed by Kaggle Kernels."""
from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO


DATASET_YAML = next(Path("/kaggle/input").rglob("dataset.yaml"))
RUN_ROOT = Path("/kaggle/working/runs")
EPOCHS = 30
IMGSZ = 640
BATCH = 16

model = YOLO("yolo11n.pt")
model.train(
    data=str(DATASET_YAML),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=0,
    workers=4,
    cache=False,
    project=str(RUN_ROOT),
    name="kairodristi-llvip",
)

run_dir = RUN_ROOT / "kairodristi-llvip"
for filename in ("best.pt", "last.pt"):
    source = run_dir / "weights" / filename
    if source.exists():
        shutil.copy2(source, Path("/kaggle/working") / filename)

print(f"Training artifacts written to {run_dir}")

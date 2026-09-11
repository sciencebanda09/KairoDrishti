"""Upload the prepared dataset, submit a Kaggle GPU kernel, and fetch its artifacts."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from split_yolo_dataset import split_dataset


OWNER = "dmechatronicx"
DATASET_SLUG = "kairodristi-llvip-yolo"
KERNEL_SLUG = "kairodristi-llvip-training"


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command))
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def write_dataset_metadata(folder: Path) -> None:
    metadata = {
        "title": "KairoDrishti LLVIP YOLO Dataset",
        "id": f"{OWNER}/{DATASET_SLUG}",
        "subtitle": "Private YOLO person-detection training data prepared from LLVIP",
        "description": "Visible LLVIP frames and YOLO person annotations for KairoDrishti training.",
        "licenses": [{"name": "other"}],
    }
    (folder / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_kernel_metadata(folder: Path) -> None:
    metadata = {
        "id": f"{OWNER}/{KERNEL_SLUG}",
        "title": "KairoDrishti LLVIP YOLO Training",
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{OWNER}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def dataset_exists() -> bool:
    result = subprocess.run(
        ["kaggle", "datasets", "list", "--user", OWNER, "--search", DATASET_SLUG, "--max-size", "100"],
        text=True,
        capture_output=True,
    )
    return result.returncode == 0 and f"{OWNER}/{DATASET_SLUG}" in result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/llvip_yolo"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wait", action="store_true", help="wait for the kernel and download output")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    runtime = root / ".kaggle_runtime"
    staged_dataset = runtime / "dataset"
    kernel = runtime / "kernel"
    if runtime.exists():
        shutil.rmtree(runtime)
    kernel.mkdir(parents=True)

    report = split_dataset(args.dataset, staged_dataset, args.val_fraction, args.seed)
    write_dataset_metadata(staged_dataset)
    print(json.dumps(report, indent=2))

    if dataset_exists():
        dataset_command = [
            "kaggle", "datasets", "version", "-p", str(staged_dataset),
            "--dir-mode", "zip", "-m", "Refresh LLVIP YOLO split",
        ]
    else:
        dataset_command = ["kaggle", "datasets", "create", "-p", str(staged_dataset), "--dir-mode", "zip"]
    run(dataset_command)

    template = (root / "kaggle" / "train.py").read_text(encoding="utf-8")
    template = template.replace("EPOCHS = 30", f"EPOCHS = {args.epochs}")
    template = template.replace("IMGSZ = 640", f"IMGSZ = {args.imgsz}")
    template = template.replace("BATCH = 16", f"BATCH = {args.batch}")
    (kernel / "train.py").write_text(template, encoding="utf-8")
    write_kernel_metadata(kernel)
    run(["kaggle", "kernels", "push", "-p", str(kernel), "--accelerator", "P100"])

    kernel_id = f"{OWNER}/{KERNEL_SLUG}"
    if not args.wait:
        print(f"Submitted {kernel_id}. Run with --wait to poll and download artifacts.")
        return

    while True:
        status = run(["kaggle", "kernels", "status", kernel_id], capture=True)
        print(status.stdout.strip())
        text = status.stdout.lower()
        if "complete" in text or "error" in text or "failed" in text:
            break
        time.sleep(30)

    output = root / "runs" / "kaggle" / KERNEL_SLUG
    output.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", kernel_id, "-p", str(output), "--force"])
    best = next(output.rglob("best.pt"), None)
    if best is None:
        raise RuntimeError(f"Kaggle completed without best.pt; inspect {output}")
    destination = root / "models" / "kairodristi-llvip-person.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    print(f"Downloaded trained model to {destination}")


if __name__ == "__main__":
    main()

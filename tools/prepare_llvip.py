"""Convert an LLVIP ZIP archive with Pascal VOC XML into YOLO layout."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree


def convert(archive: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    report = {"archive": str(archive), "images": 0, "labels": 0, "missing_pairs": 0}
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        annotations = [name for name in names if name.startswith("LLVIP/Annotations/") and name.endswith(".xml")]
        for annotation_name in annotations:
            root = ElementTree.fromstring(bundle.read(annotation_name))
            frame_id = Path(annotation_name).stem
            split = "train" if any(f"/visible/train/{frame_id}.jpg" in name for name in names) else "val"
            image_name = f"LLVIP/visible/{split}/{frame_id}.jpg"
            if image_name not in names:
                report["missing_pairs"] += 1
                continue
            image_out = output / "images" / split / f"{frame_id}.jpg"
            label_out = output / "labels" / split / f"{frame_id}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            image_out.write_bytes(bundle.read(image_name))
            size = root.find("size")
            width, height = int(size.findtext("width")), int(size.findtext("height"))
            labels = []
            for obj in root.findall("object"):
                if obj.findtext("name", "").lower() != "person":
                    continue
                box = obj.find("bndbox")
                x0, y0 = float(box.findtext("xmin")), float(box.findtext("ymin"))
                x1, y1 = float(box.findtext("xmax")), float(box.findtext("ymax"))
                labels.append(f"0 {((x0+x1)/2)/width:.6f} {((y0+y1)/2)/height:.6f} {(x1-x0)/width:.6f} {(y1-y0)/height:.6f}")
            label_out.write_text("\n".join(labels), encoding="utf-8")
            report["images"] += 1
            report["labels"] += len(labels)
        (output / "dataset.yaml").write_text(
            f"path: {output.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n",
            encoding="utf-8",
        )
        (output / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/llvip_yolo"))
    args = parser.parse_args()
    print(json.dumps(convert(args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()

"""M1.1: verify our YOLO -> pixel conversion against the organisers' COCO annotations.

For every frame of the mini dataset:
  - YOLO labels converted to pixel boxes must match the COCO boxes (iscrowd=0) per class,
  - ignore/*.txt regions must match the COCO iscrowd=1 regions,
  - image size read from the .jpg must match width/height in the COCO json.

Usage (from code/):  python scripts/check_format.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make src importable
from src.config import CODE_DIR, get_data_root, load_config
from src.data.convert import (coco_to_xyxy, iou_matrix, label_paths,
                              read_ignore, read_yolo_labels, yolo_to_xyxy)

IOU_OK = 0.99  # boxes must be near-identical (only rounding differences allowed)


def boxes_match(a, b) -> bool:
    """Same number of boxes and every box in `a` has a partner in `b` with IoU >= IOU_OK."""
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    return bool((iou_matrix(a, b).max(axis=1) >= IOU_OK).all())


def classes_match(cls_a, box_a, cls_b, box_b) -> bool:
    """Compare boxes class by class."""
    if sorted(cls_a.tolist()) != sorted(cls_b.tolist()):
        return False
    return all(boxes_match(box_a[cls_a == c], box_b[cls_b == c]) for c in np.unique(cls_a))


def load_coco_index(path: Path) -> dict:
    """COCO json -> {(clip, frame_stem): {"w", "h", "cls", "boxes", "crowd"}} with pixel xyxy boxes."""
    with open(path) as f:
        coco = json.load(f)
    print(f"[{path.parent.name}] sample image entry: {coco['images'][0]}")

    by_id = {}
    for img in coco["images"]:
        fn = Path(img["file_name"])
        by_id[img["id"]] = {"key": (fn.parent.name, fn.stem), "w": img["width"], "h": img["height"],
                            "cls": [], "boxes": [], "crowd": []}
    
    for a in coco["annotations"]:
        entry = by_id[a["image_id"]]
        if a.get("iscrowd", 0):
            entry["crowd"].append(tuple(a["bbox"]))  # each region is repeated once per class
        else:
            entry["boxes"].append(a["bbox"])
            entry["cls"].append(a["category_id"] - 1)  # COCO is 1-based, we use 0-based

    index = {}
    for e in by_id.values():
        unique_crowd = sorted(set(e["crowd"]))
        index[e["key"]] = {
            "w": e["w"], "h": e["h"],
            "cls": np.array(e["cls"], dtype=int),
            "boxes": coco_to_xyxy(e["boxes"]),
            "crowd": coco_to_xyxy(unique_crowd),
            # every unique ignore region must appear exactly once for each of the 8 classes
            "crowd_dup_ok": len(e["crowd"]) == 8 * len(unique_crowd),
        }
    return index


def main():
    cfg = load_config("configs/protocol.yaml")
    mini = get_data_root(cfg)
    ann_root = CODE_DIR / cfg["data_dirs"]["annotations"]

    checked = bad = missing = 0
    for split in ("train", "val"):
        coco = load_coco_index(ann_root / split / "annotations_coco.json")
        for img_path in sorted((mini / split / "images").rglob("*.jpg")):
            key = (img_path.parent.name, img_path.stem)
            if key not in coco:
                missing += 1
                continue
            ref = coco[key]

            h, w = cv2.imread(str(img_path)).shape[:2]
            lbl_path, ign_path = label_paths(img_path)
            cls, yolo = read_yolo_labels(lbl_path)
            ours = yolo_to_xyxy(yolo, w, h)
            ours_ign = yolo_to_xyxy(read_ignore(ign_path), w, h)

            ok_size = (w, h) == (ref["w"], ref["h"])
            ok_boxes = classes_match(cls, ours, ref["cls"], ref["boxes"])
            ok_ign = boxes_match(ours_ign, ref["crowd"]) and ref["crowd_dup_ok"]
            checked += 1

            if not (ok_size and ok_boxes and ok_ign):
                bad += 1
                if bad <= 5:  # print only the first few problems
                    print(f"MISMATCH {split}/{key}: size={ok_size} boxes={ok_boxes} ignore={ok_ign} "
                          f"(ours {len(ours)}/{len(ours_ign)} vs coco {len(ref['boxes'])}/{len(ref['crowd'])})")

    print(f"\nchecked frames: {checked} | mismatches: {bad} | not found in COCO: {missing}")
    print("FORMAT OK" if checked > 0 and bad == 0 and missing == 0 else "FORMAT PROBLEM")


if __name__ == "__main__":
    main()

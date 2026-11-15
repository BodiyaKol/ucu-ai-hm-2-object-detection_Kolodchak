"""M1.5: build the training subset and ground-truth files used by all models.

Train (every `frame_step`-th frame per clip):
  - ignored regions painted gray (`ignore_fill`) in the images,
  - boxes covered > 50% by ignored regions are dropped (they become invisible after masking),
  - YOLO labels next to the images (Ultralytics layout) + train.txt,
  - COCO json with pixel boxes (for torchvision models).
Val (all frames, images untouched):
  - COCO json used as evaluation ground truth (ignored regions as iscrowd=1, once per class).

Usage (from code/):
  python scripts/prepare_data.py                                  # local: data/mini -> outputs/prepared
  python scripts/prepare_data.py --out /kaggle/working/prepared   # Kaggle: full competition data
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make src importable
from src.config import get_data_root, get_output_dir, is_kaggle, load_config, set_seed
from src.data.convert import build_coco, label_paths, read_ignore, read_yolo_labels, yolo_to_xyxy
from src.data.ignore import ignore_coverage, mask_ignore_regions
from src.data.yolo_yaml import write_data_yaml

MAX_COVERAGE = 0.5  # drop boxes covered more than this by ignored regions (train only)


def read_frame_labels(img_path: Path, w: int, h: int):
    """Labels of one frame: classes, normalized YOLO boxes, pixel xyxy boxes, pixel xyxy ignored regions."""
    lbl_path, ign_path = label_paths(img_path)
    cls, yolo = read_yolo_labels(lbl_path)
    return cls, yolo, yolo_to_xyxy(yolo, w, h), yolo_to_xyxy(read_ignore(ign_path), w, h)


def frame_entry(img_path: Path, split_dir: Path, w: int, h: int, cls, boxes, ignore) -> dict:
    """Entry for build_coco; file_name is relative to the split folder (images/<clip>/<frame>.jpg)."""
    return {"file_name": img_path.relative_to(split_dir).as_posix(), "width": w, "height": h,
            "video": img_path.parent.name, "frame_id": int(img_path.stem),
            "classes": cls, "boxes": boxes, "ignore": ignore}


def prepare_train(src: Path, prepared: Path, step: int, fill: int):
    """Subsample, mask and write the training split. Returns (coco entries, train.txt lines, stats)."""
    out = prepared / "train"
    entries, lines = [], []
    stats = {"train_frames": 0, "frames_masked": 0, "boxes_kept": 0, "boxes_dropped_in_ignore": 0}

    for clip_dir in tqdm(sorted((src / "images").iterdir()), desc="train clips"):
        for img_path in sorted(clip_dir.glob("*.jpg"))[::step]:  # frames 1, 1+step, 1+2*step, ...
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            cls, yolo, boxes, ignore = read_frame_labels(img_path, w, h)
            keep = ignore_coverage(boxes, ignore) <= MAX_COVERAGE

            # Image: masked copy if the frame has ignored regions, otherwise a plain copy (no re-encoding)
            out_img = out / "images" / clip_dir.name / img_path.name
            out_img.parent.mkdir(parents=True, exist_ok=True)
            if len(ignore):
                cv2.imwrite(str(out_img), mask_ignore_regions(img, ignore, fill), [cv2.IMWRITE_JPEG_QUALITY, 95])
                stats["frames_masked"] += 1
            else:
                shutil.copy2(img_path, out_img)

            # YOLO labels: kept boxes only, same normalized format as the input
            out_lbl = out / "labels" / clip_dir.name / f"{img_path.stem}.txt"
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.write_text("".join(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n"
                                       for c, (x, y, bw, bh) in zip(cls[keep], yolo[keep])))

            lines.append("./" + out_img.relative_to(prepared).as_posix())
            entries.append(frame_entry(out_img, out, w, h, cls[keep], boxes[keep], ignore))
            stats["train_frames"] += 1
            stats["boxes_kept"] += int(keep.sum())
            stats["boxes_dropped_in_ignore"] += int((~keep).sum())
    return entries, lines, stats


def prepare_val(src: Path, step: int) -> list[dict]:
    """COCO entries for val frames (images are not copied or modified)."""
    entries = []
    for clip_dir in tqdm(sorted((src / "images").iterdir()), desc="val clips"):
        for img_path in sorted(clip_dir.glob("*.jpg"))[::step]:
            with Image.open(img_path) as im:  # reads only the header, fast
                w, h = im.size
            cls, _, boxes, ignore = read_frame_labels(img_path, w, h)
            entries.append(frame_entry(img_path, src, w, h, cls, boxes, ignore))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output folder (default: outputs/prepared)")
    ap.add_argument("--frame-step", type=int, default=None, help="override protocol frame_step")
    args = ap.parse_args()

    cfg = load_config("configs/protocol.yaml")
    set_seed(cfg["seed"])
    root = get_data_root(cfg)
    prepared = Path(args.out) if args.out else get_output_dir("prepared")
    step = args.frame_step or cfg["frame_step"]
    if (prepared / "train").exists():
        shutil.rmtree(prepared / "train")  # start clean
    print(f"data root: {root}\noutput: {prepared}\nframe_step: {step}")

    train_entries, lines, stats = prepare_train(root / cfg["train_split"], prepared, step, cfg["ignore_fill"])
    (prepared / "train.txt").write_text("\n".join(lines) + "\n")
    val_entries = prepare_val(root / cfg["val_split"], cfg["val_frame_step"])

    ann_dir = prepared / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for name, entries in (("train", train_entries), ("val", val_entries)):
        coco = build_coco(entries, cfg["class_names"])
        with open(ann_dir / f"{name}_coco.json", "w") as f:
            json.dump(coco, f)
        stats[f"{name}_coco_objects"] = sum(a["iscrowd"] == 0 for a in coco["annotations"])
        stats[f"{name}_coco_crowd"] = sum(a["iscrowd"] == 1 for a in coco["annotations"])

    stats.update({"frame_step": step, "val_frames": len(val_entries), "data_root": str(root)})
    (prepared / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    if not is_kaggle():  # local smoke test of the Ultralytics config; kernels write their own
        print("data.yaml:", write_data_yaml(prepared, root, prepared, cfg["class_names"], cfg["val_frame_step"]))


if __name__ == "__main__":
    main()

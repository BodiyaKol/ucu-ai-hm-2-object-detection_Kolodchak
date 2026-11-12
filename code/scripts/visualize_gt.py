"""M1.1: draw ground-truth boxes and ignored regions on sample mini frames -> images/ (repo root).

One middle frame per clip and split. Gray = ignored region, colour = class.
Usage (from code/):  python scripts/visualize_gt.py
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make src importable
from src.config import CODE_DIR, get_data_root, load_config
from src.data.convert import label_paths, read_ignore, read_yolo_labels, yolo_to_xyxy
from src.viz import add_legend, draw_boxes

OUT_DIR = CODE_DIR.parent / "images"  # images/ folder used by report.md


def main():
    cfg = load_config("configs/protocol.yaml")
    root = get_data_root(cfg)
    OUT_DIR.mkdir(exist_ok=True)

    for split in ("train", "val"):
        for clip_dir in sorted((root / split / "images").iterdir()):
            frames = sorted(clip_dir.glob("*.jpg"))
            img_path = frames[len(frames) // 2]  # middle frame of the clip

            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            lbl_path, ign_path = label_paths(img_path)
            cls, yolo = read_yolo_labels(lbl_path)
            ignore = yolo_to_xyxy(read_ignore(ign_path), w, h)

            vis = draw_boxes(img, yolo_to_xyxy(yolo, w, h), cls, ignore_xyxy=ignore)
            add_legend(vis)

            out = OUT_DIR / f"gt_{split}_{clip_dir.name}.jpg"
            cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"{out.name}: {w}x{h}, {len(cls)} objects, {len(ignore)} ignored regions")


if __name__ == "__main__":
    main()

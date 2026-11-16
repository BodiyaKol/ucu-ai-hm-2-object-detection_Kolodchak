"""Visualization helpers: draw GT / predicted boxes, ignored regions, failure cases."""
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
# One BGR colour per class (OpenCV uses BGR)
COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 255, 0),
          (255, 0, 0), (255, 0, 255), (0, 255, 255), (128, 0, 255)]


def draw_boxes(image, boxes_xyxy, classes=None, scores=None, ignore_xyxy=None, thickness=1, color=(0, 255, 0)):
    """Return a copy of a BGR image with class-coloured boxes (or `color` if classes is None) and gray ignored regions."""
    img = image.copy()
    if ignore_xyxy is not None and len(ignore_xyxy):
        overlay = img.copy()
        for x1, y1, x2, y2 in np.asarray(ignore_xyxy).astype(int):
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 128, 128), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes_xyxy).astype(int)):
        c = COLORS[int(classes[i])] if classes is not None else color
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thickness)
        if scores is not None:
            cv2.putText(img, f"{scores[i]:.2f}", (x1, max(y1 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
    return img


def add_legend(image):
    """Draw the class-colour legend in the top-left corner (in place) and return the image."""
    for i, name in enumerate(CLASS_NAMES):
        y = 20 + i * 20
        cv2.rectangle(image, (8, y - 12), (22, y + 2), COLORS[i], -1)
        cv2.putText(image, name, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, name, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i], 1)
    return image


def save_failure_cases(gt: dict, preds: list, images_root, out_dir, n: int = 6,
                       score_thresh: float = 0.3, min_gt: int = 20) -> list:
    """Save the n val frames with the lowest recall@IoU0.5 (preds with score >= score_thresh).

    GT = white boxes, predictions = class-coloured boxes with score, ignore regions = gray.
    """
    from src.data.convert import coco_to_xyxy, iou_matrix
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g, crowd, p = defaultdict(list), defaultdict(set), defaultdict(list)
    for a in gt["annotations"]:
        (crowd[a["image_id"]].add(tuple(a["bbox"])) if a["iscrowd"] else g[a["image_id"]].append(a["bbox"]))
    for d in preds:
        if d["score"] >= score_thresh:
            p[d["image_id"]].append(d)
    rows = []
    for im in gt["images"]:
        if len(g[im["id"]]) < min_gt:
            continue
        gb = coco_to_xyxy(g[im["id"]])
        pb = coco_to_xyxy([d["bbox"] for d in p[im["id"]]]) if p[im["id"]] else np.zeros((0, 4))
        recall = float((iou_matrix(gb, pb).max(1) >= 0.5).mean()) if len(pb) else 0.0
        rows.append((recall, im))
    rows.sort(key=lambda r: r[0])
    saved = []
    for recall, im in rows[:n]:
        img = cv2.imread(str(Path(images_root) / im["file_name"]))
        ign = coco_to_xyxy(list(crowd[im["id"]])) if crowd[im["id"]] else None
        img = draw_boxes(img, coco_to_xyxy(g[im["id"]]), ignore_xyxy=ign, color=(255, 255, 255))
        dets = p[im["id"]]
        if dets:
            img = draw_boxes(img, coco_to_xyxy([d["bbox"] for d in dets]),
                             classes=[d["category_id"] - 1 for d in dets], scores=[d["score"] for d in dets])
        cv2.putText(img, f"{im['file_name']}  recall@0.5={recall:.2f}  (score>={score_thresh})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        name = f"{Path(im['file_name']).parent.name}_{Path(im['file_name']).stem}_r{recall:.2f}.jpg"
        cv2.imwrite(str(out_dir / name), add_legend(img), [cv2.IMWRITE_JPEG_QUALITY, 80])
        saved.append(name)
    return saved

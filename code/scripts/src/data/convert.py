"""Box format conversions and label-file readers.

Internal convention in this project: pixel xyxy boxes (x1, y1, x2, y2), class ids 0-7.
Other formats are produced only at the boundaries:
  YOLO labels   : normalized (cx, cy, w, h)
  COCO / submit : pixel top-left (x, y, w, h)
"""
from pathlib import Path

import numpy as np


def subsample_frames(frame_ids: list[int], step: int) -> list[int]:
    """Keep every `step`-th frame of a clip."""
    return sorted(frame_ids)[::step]


def xywh_to_yolo(box, img_w: int, img_h: int):
    """Single box: top-left (x, y, w, h) in pixels -> YOLO normalized (cx, cy, w, h)."""
    x, y, w, h = box
    return ((x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h)


def yolo_to_xywh(box, img_w: int, img_h: int):
    """Single box: YOLO normalized (cx, cy, w, h) -> top-left (x, y, w, h) in pixels."""
    cx, cy, w, h = box
    w_px, h_px = w * img_w, h * img_h
    return (cx * img_w - w_px / 2, cy * img_h - h_px / 2, w_px, h_px)


def yolo_to_xyxy(boxes, img_w: int, img_h: int) -> np.ndarray:
    """Array (N, 4) of normalized (cx, cy, w, h) -> pixel (x1, y1, x2, y2)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy = b[:, 0] * img_w, b[:, 1] * img_h
    w, h = b[:, 2] * img_w, b[:, 3] * img_h
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def xyxy_to_coco(boxes) -> np.ndarray:
    """Pixel (x1, y1, x2, y2) -> COCO / submission top-left (x, y, w, h)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def coco_to_xyxy(boxes) -> np.ndarray:
    """COCO top-left (x, y, w, h) -> pixel (x1, y1, x2, y2)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def iou_matrix(a, b) -> np.ndarray:
    """Pairwise IoU between pixel xyxy boxes a (N, 4) and b (M, 4) -> (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def label_paths(img_path) -> tuple[Path, Path]:
    """<split>/images/<clip>/<frame>.jpg -> (<split>/labels/.../<frame>.txt, <split>/ignore/.../<frame>.txt)."""
    img_path = Path(img_path)
    split_dir = img_path.parents[2]
    rel = img_path.relative_to(split_dir / "images").with_suffix(".txt")
    return split_dir / "labels" / rel, split_dir / "ignore" / rel


def read_yolo_labels(path) -> tuple[np.ndarray, np.ndarray]:
    """labels/*.txt -> (classes (N,) int, normalized boxes (N, 4)). Missing or empty file -> empty arrays."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 4))
    data = np.loadtxt(path, ndmin=2)
    return data[:, 0].astype(int), data[:, 1:5]


def read_ignore(path) -> np.ndarray:
    """ignore/*.txt -> normalized boxes (N, 4). Missing or empty file -> empty array."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 4))
    return np.loadtxt(path, ndmin=2)[:, :4]

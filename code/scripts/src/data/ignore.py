"""Ignored regions: masking in training images and how much of a box they cover."""
import numpy as np


def mask_ignore_regions(image: np.ndarray, ignore_xyxy, fill_value: int = 114) -> np.ndarray:
    """Paint ignored regions (pixel xyxy) with a constant gray value. Returns a copy."""
    image = image.copy()
    h, w = image.shape[:2]
    for x1, y1, x2, y2 in np.asarray(ignore_xyxy, dtype=np.float64).reshape(-1, 4):
        x1, y1 = max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0)
        x2, y2 = min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
        image[y1:y2, x1:x2] = fill_value
    return image


def ignore_coverage(boxes_xyxy, ignore_xyxy) -> np.ndarray:
    """Fraction of each box area covered by ignored regions, clipped to [0, 1].

    Overlapping ignored regions are counted twice (rare; the clip keeps the value valid).
    """
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(ignore_xyxy, dtype=np.float64).reshape(-1, 4)
    if len(b) == 0 or len(g) == 0:
        return np.zeros(len(b))
    iw = np.clip(np.minimum(b[:, None, 2], g[None, :, 2]) - np.maximum(b[:, None, 0], g[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(b[:, None, 3], g[None, :, 3]) - np.maximum(b[:, None, 1], g[None, :, 1]), 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return np.clip((iw * ih).sum(axis=1) / np.maximum(area, 1e-9), 0, 1)

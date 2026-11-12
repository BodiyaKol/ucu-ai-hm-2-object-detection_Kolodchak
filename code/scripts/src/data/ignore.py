"""Ignore-region handling (VisDrone category 0 / 'ignored regions')."""
import numpy as np


def mask_ignore_regions(image: np.ndarray, boxes, fill_value: int = 114) -> np.ndarray:
    """Paint ignore boxes (x, y, w, h) with a constant colour. TODO: decide fill (mean / gray)."""
    image = image.copy()
    for x, y, w, h in boxes:
        x, y, w, h = map(int, (x, y, w, h))
        image[y : y + h, x : x + w] = fill_value
    return image

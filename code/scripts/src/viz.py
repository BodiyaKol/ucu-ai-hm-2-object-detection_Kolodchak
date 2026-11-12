"""Visualization helpers: GT / prediction boxes, failure-case grids (M1, M3+)."""


def draw_boxes(image, boxes, labels=None, scores=None, color=(0, 255, 0)):
    """TODO: draw xywh boxes on a copy of image."""
    raise NotImplementedError


def failure_grid(samples, out_path, ncols: int = 3):
    """TODO: grid of (image, GT, pred) for failure cases -> ../images/."""
    raise NotImplementedError

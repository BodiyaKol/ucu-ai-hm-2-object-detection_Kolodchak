"""Visualization helpers: draw GT / predicted boxes and ignored regions."""
import cv2
import numpy as np

CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
# One BGR colour per class (OpenCV uses BGR)
COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 255, 0),
          (255, 0, 0), (255, 0, 255), (0, 255, 255), (128, 0, 255)]


def draw_boxes(image, boxes_xyxy, classes=None, scores=None, ignore_xyxy=None, thickness=1):
    """Return a copy of a BGR image with class-coloured boxes and gray ignored regions."""
    img = image.copy()

    # Ignored regions: semi-transparent gray fill
    if ignore_xyxy is not None and len(ignore_xyxy):
        overlay = img.copy()
        for x1, y1, x2, y2 in np.asarray(ignore_xyxy).astype(int):
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 128, 128), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    # Object boxes: colour = class, optional score text
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes_xyxy).astype(int)):
        color = COLORS[int(classes[i])] if classes is not None else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if scores is not None:
            cv2.putText(img, f"{scores[i]:.2f}", (x1, max(y1 - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return img


def add_legend(image):
    """Draw the class-colour legend in the top-left corner (in place) and return the image."""
    for i, name in enumerate(CLASS_NAMES):
        y = 20 + i * 20
        cv2.rectangle(image, (8, y - 12), (22, y + 2), COLORS[i], -1)
        cv2.putText(image, name, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, name, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i], 1)
    return image


def failure_grid(samples, out_path, ncols: int = 3):
    """(M3+) Grid of (image, GT, pred) for failure cases. TODO in M3."""
    raise NotImplementedError

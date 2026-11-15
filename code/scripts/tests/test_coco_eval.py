import numpy as np
import pytest

from src.data.convert import build_coco
from src.eval.coco_eval import coco_evaluate, to_coco_results

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
GT_BOXES = [[10, 10, 110, 60], [200, 200, 230, 260]]  # one car, one person (pixel xyxy)
IGNORE = [[400, 300, 600, 450]]


def _gt(ignore):
    entry = {"file_name": "images/clip/0000001.jpg", "width": 640, "height": 480,
             "video": "clip", "frame_id": 1,
             "classes": np.array([2, 0]), "boxes": np.array(GT_BOXES, float),
             "ignore": np.array(ignore, float).reshape(-1, 4)}
    return build_coco([entry], CLASSES)


def test_perfect_predictions():
    preds = to_coco_results(1, GT_BOXES, [0.9, 0.8], [2, 0])
    m = coco_evaluate(_gt([]), preds)
    assert m["mAP"] == pytest.approx(1.0)
    assert m["per_class_AP"]["car"] == pytest.approx(1.0)
    assert m["per_class_AP"]["bus"] is None  # no GT of this class


def test_detection_in_ignore_region_is_not_fp():
    # perfect boxes + a confident "car" inside the ignore region (ranked first)
    preds = to_coco_results(1, GT_BOXES + [[420, 320, 480, 380]], [0.9, 0.8, 0.99], [2, 0, 2])
    with_ignore = coco_evaluate(_gt(IGNORE), preds)
    without_ignore = coco_evaluate(_gt([]), preds)
    assert with_ignore["mAP"] == pytest.approx(1.0)
    # without the region it becomes a top-ranked FP: car precision 0.5 at full recall
    assert without_ignore["per_class_AP"]["car"] == pytest.approx(0.5, abs=0.01)


def test_empty_predictions():
    assert coco_evaluate(_gt([]), [])["mAP"] == 0.0


def test_map_uses_500_dets_not_100():
    # 150 objects in one frame (dense like VisDrone); with top-100 only, mAP would be ~0.67
    boxes = [[40 * i, 40 * j, 40 * i + 20, 40 * j + 20] for i in range(15) for j in range(10)]
    entry = {"file_name": "images/clip/0000001.jpg", "width": 640, "height": 480,
             "video": "clip", "frame_id": 1, "classes": np.full(150, 2),
             "boxes": np.array(boxes, float), "ignore": np.zeros((0, 4))}
    gt = build_coco([entry], CLASSES)
    preds = to_coco_results(1, boxes, np.linspace(0.9, 0.5, 150), [2] * 150)
    assert coco_evaluate(gt, preds)["mAP"] == pytest.approx(1.0)

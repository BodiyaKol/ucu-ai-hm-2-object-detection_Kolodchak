import numpy as np

from src.data.convert import build_coco
from src.data.ignore import ignore_coverage, mask_ignore_regions


def test_mask_paints_only_region():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    out = mask_ignore_regions(img, [[2, 2, 5, 5]], fill_value=114)
    assert (out[2:5, 2:5] == 114).all() and out[0, 0, 0] == 0
    assert img.sum() == 0  # original untouched


def test_coverage():
    boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
    cov = ignore_coverage(boxes, [[0, 0, 5, 10]])
    assert np.allclose(cov, [0.5, 0.0])
    assert len(ignore_coverage(boxes, np.zeros((0, 4)))) == 2


def test_build_coco_crowd_per_class():
    entry = {"file_name": "images/c/0000001.jpg", "width": 100, "height": 50, "video": "c", "frame_id": 1,
             "classes": np.array([2]), "boxes": np.array([[10, 10, 20, 30]]), "ignore": np.array([[0, 0, 5, 5]])}
    coco = build_coco([entry], [f"c{i}" for i in range(8)])
    objs = [a for a in coco["annotations"] if a["iscrowd"] == 0]
    crowd = [a for a in coco["annotations"] if a["iscrowd"] == 1]
    assert objs[0]["category_id"] == 3 and objs[0]["bbox"] == [10, 10, 10, 20]
    assert len(crowd) == 8 and {a["category_id"] for a in crowd} == set(range(1, 9))

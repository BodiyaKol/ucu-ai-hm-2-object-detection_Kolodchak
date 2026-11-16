import json
import cv2
import numpy as np
import torch

from src.data.dataset import VisDroneDetection, collate_fn


def test_dataset_and_flip(tmp_path):
    (tmp_path / "images/c").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images/c/1.jpg"), np.zeros((40, 100, 3), np.uint8))
    coco = {"images": [{"id": 1, "file_name": "images/c/1.jpg", "width": 100, "height": 40}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 5, 20, 10], "iscrowd": 0},
                            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 20], "iscrowd": 1},
                            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [60, 5, 0.5, 10], "iscrowd": 0}],
            "categories": []}
    (tmp_path / "t.json").write_text(json.dumps(coco))
    ds = VisDroneDetection(tmp_path / "t.json", tmp_path, train=False)
    img, t = ds[0]
    assert img.shape == (3, 40, 100) and t["labels"].tolist() == [3]  # crowd + degenerate box dropped
    assert t["boxes"].tolist() == [[10, 5, 30, 15]]
    ds.train = True
    torch.manual_seed(0)
    flipped = [ds[0][1]["boxes"].tolist()[0] for _ in range(20)]
    assert [70, 5, 90, 15] in flipped and [10, 5, 30, 15] in flipped
    imgs, targets = collate_fn([ds[0], ds[0]])
    assert len(imgs) == 2 and len(targets) == 2

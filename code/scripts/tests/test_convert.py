import pytest
import numpy as np

from src.data.convert import subsample_frames, xywh_to_yolo, yolo_to_xywh, coco_to_xyxy, read_yolo_labels, xyxy_to_coco, yolo_to_xyxy


def test_yolo_round_trip():
    box = (10.0, 20.0, 30.0, 40.0)
    back = yolo_to_xywh(xywh_to_yolo(box, 1920, 1080), 1920, 1080)
    assert back == pytest.approx(box)


def test_subsample():
    assert subsample_frames(list(range(1, 11)), 5) == [1, 6]


def test_yolo_to_xyxy_to_coco():
    xyxy = yolo_to_xyxy([[0.5, 0.5, 0.1, 0.2]], 1000, 500)
    assert np.allclose(xyxy, [[450, 200, 550, 300]])
    coco = xyxy_to_coco(xyxy)
    assert np.allclose(coco, [[450, 200, 100, 100]])
    assert np.allclose(coco_to_xyxy(coco), xyxy)


def test_read_yolo_labels(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("2 0.5 0.5 0.1 0.2\n0 0.1 0.1 0.05 0.05\n")
    cls, boxes = read_yolo_labels(f)
    assert cls.tolist() == [2, 0] and boxes.shape == (2, 4)

    empty = tmp_path / "empty.txt"
    empty.write_text("")
    cls, boxes = read_yolo_labels(empty)
    assert len(cls) == 0 and boxes.shape == (0, 4)
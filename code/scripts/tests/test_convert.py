import pytest

from src.data.convert import subsample_frames, xywh_to_yolo, yolo_to_xywh


def test_yolo_round_trip():
    box = (10.0, 20.0, 30.0, 40.0)
    back = yolo_to_xywh(xywh_to_yolo(box, 1920, 1080), 1920, 1080)
    assert back == pytest.approx(box)


def test_subsample():
    assert subsample_frames(list(range(1, 11)), 5) == [1, 6]

import csv

from src.eval.coco_eval import to_coco_results
from src.submission import MAX_DETS_PER_IMAGE, check_submission, predictions_to_csv


def test_max_dets():
    assert MAX_DETS_PER_IMAGE == 500


def _test_csv(tmp_path):
    f = tmp_path / "test_images.csv"
    f.write_text("image_id,file_name,clip,frame,width,height\n"
                 "c_0000001,test/images/c/0000001.jpg,c,1,640,480\n"
                 "c_0000002,test/images/c/0000002.jpg,c,2,640,480\n"
                 "c_0000003,test/images/c/0000003.jpg,c,3,640,480\n")
    return f


def test_csv_format(tmp_path):
    test_csv, out = _test_csv(tmp_path), tmp_path / "sub.csv"
    # image 1: one car (class 2), pixel xyxy (10, 20, 40, 60) -> top-left xywh (10, 20, 30, 40)
    preds = to_coco_results("c_0000001", [[10, 20, 40, 60]], [0.9], [2])
    # image 2: 600 detections with increasing score -> only the best 500 must survive
    preds += to_coco_results("c_0000002", [[0, 0, 5, 5]] * 600, [i / 1000 for i in range(600)], [0] * 600)
    # image 3: no detections
    assert predictions_to_csv(preds, test_csv, out) == 3

    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["image_id"] for r in rows] == ["c_0000001", "c_0000002", "c_0000003"]
    assert rows[0]["PredictionString"] == "2 0.9000 10.0 20.0 30.0 40.0"  # 0-based class, top-left xywh
    tokens = rows[1]["PredictionString"].split()
    assert len(tokens) == 6 * 500 and float(tokens[1]) == 0.599  # sorted, highest score first
    assert min(float(s) for s in tokens[1::6]) == 0.1  # scores 0.000-0.099 dropped
    assert rows[2]["PredictionString"] == ""

    stats = check_submission(out, test_csv)
    assert stats == {"rows": 3, "detections": 501, "empty_rows": 1}


def test_unknown_image_id_raises(tmp_path):
    import pytest
    preds = to_coco_results("not_a_test_image", [[0, 0, 5, 5]], [0.5], [0])
    with pytest.raises(ValueError):
        predictions_to_csv(preds, _test_csv(tmp_path), tmp_path / "sub.csv")

import pytest

from src.submission import MAX_DETS_PER_IMAGE


def test_max_dets():
    assert MAX_DETS_PER_IMAGE == 500


@pytest.mark.skip(reason="TODO: implement once predictions_to_csv exists")
def test_csv_format():
    # TODO: fake predictions -> csv; check columns, top-left xywh, 0-based ids, <=500 per image
    pass

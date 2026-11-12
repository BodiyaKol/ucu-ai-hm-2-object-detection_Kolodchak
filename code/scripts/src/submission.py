"""Predictions -> Kaggle CSV.

TODO: check exact required columns in the Kaggle competition description.
Assumed: boxes in top-left x, y, w, h (pixels), 0-based class ids, at most 500 boxes per image.
"""
MAX_DETS_PER_IMAGE = 500


def predictions_to_csv(pred_json: str, out_csv: str) -> None:
    raise NotImplementedError

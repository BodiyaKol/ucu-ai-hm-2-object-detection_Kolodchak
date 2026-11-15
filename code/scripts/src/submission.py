"""COCO-style results -> Kaggle submission CSV.

Format (competition page): header "image_id,PredictionString", one row per test image.
PredictionString = space-separated groups "class score x y w h":
class 0-7, (x, y) = top-left corner, pixels of the original image.
Empty string if no detections. At most 500 detections per image are scored.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

MAX_DETS_PER_IMAGE = 500
NUM_CLASSES = 8


def _read_image_ids(test_csv) -> list[str]:
    with open(test_csv, newline="") as fh:
        return [row["image_id"] for row in csv.DictReader(fh)]


def format_prediction_string(dets: list[dict]) -> str:
    """COCO results of one image -> 'class score x y w h ...' (top-500 by score)."""
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)[:MAX_DETS_PER_IMAGE]
    parts = []
    for d in dets:
        x, y, w, h = d["bbox"]
        parts.append(f'{d["category_id"] - 1} {d["score"]:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}')
    return " ".join(parts)


def predictions_to_csv(preds, test_csv, out_csv) -> int:
    """Write the submission. preds: COCO results list (or json path) with image_id = test image_id string.

    Every image of test_csv gets a row (empty if no detections). Returns the number of rows.
    """
    if isinstance(preds, (str, Path)):
        with open(preds) as fh:
            preds = json.load(fh)
    by_image = defaultdict(list)
    for d in preds:
        by_image[d["image_id"]].append(d)

    image_ids = _read_image_ids(test_csv)
    unknown = set(by_image) - set(image_ids)
    if unknown:
        raise ValueError(f"{len(unknown)} predicted image_ids are not in {test_csv}, e.g. {sorted(unknown)[:3]}")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "PredictionString"])
        for image_id in image_ids:
            writer.writerow([image_id, format_prediction_string(by_image.get(image_id, []))])
    return len(image_ids)


def check_submission(sub_csv, test_csv) -> dict:
    """Validate a submission file; raise ValueError on the first problem. Returns small stats."""
    with open(sub_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    if header != ["image_id", "PredictionString"]:
        raise ValueError(f"bad header: {header}")
    if [r[0] for r in rows] != _read_image_ids(test_csv):
        raise ValueError("image_ids differ from test_images.csv (missing, extra or reordered rows)")

    n_dets, n_empty = 0, 0
    for image_id, pred in rows:
        tokens = pred.split()
        if len(tokens) % 6:
            raise ValueError(f"{image_id}: number of values is not a multiple of 6")
        k = len(tokens) // 6
        if k > MAX_DETS_PER_IMAGE:
            raise ValueError(f"{image_id}: {k} detections > {MAX_DETS_PER_IMAGE}")
        for i in range(0, len(tokens), 6):
            if not 0 <= int(tokens[i]) < NUM_CLASSES:
                raise ValueError(f"{image_id}: bad class {tokens[i]}")
        n_dets += k
        n_empty += k == 0
    return {"rows": len(rows), "detections": n_dets, "empty_rows": n_empty}

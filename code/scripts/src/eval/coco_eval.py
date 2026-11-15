"""COCO-style evaluation with pycocotools.

Ignore regions must be in the GT as iscrowd=1 (once per class, see build_coco):
pycocotools matches detections inside them to the crowd box and drops them,
so they are counted neither as TP nor as FP.
"""
import contextlib
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.data.convert import xyxy_to_coco

# Order of the first 6 values of COCOeval.stats (all computed at maxDets[2]).
STAT_NAMES = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]


def _load(obj):
    """Accept a path to a json file or an already loaded dict/list."""
    if isinstance(obj, (str, Path)):
        with open(obj) as fh:
            return json.load(fh)
    return obj


def to_coco_results(image_id, boxes_xyxy, scores, classes) -> list[dict]:
    """Detections of one image in our convention (pixel xyxy, class 0-7) -> COCO results list."""
    # int ids for val (COCO GT), string ids (e.g. "clip_001_0000001") for the test submission
    image_id = image_id if isinstance(image_id, str) else int(image_id)
    return [{"image_id": image_id, "category_id": int(c) + 1,
             "bbox": [round(float(v), 2) for v in box], "score": round(float(s), 5)}
            for box, s, c in zip(xyxy_to_coco(boxes_xyxy), scores, classes)]


def coco_evaluate(gt, preds, max_dets: int = 500, verbose: bool = False) -> dict:
    """Return mAP, AP50, AP75, AP_small/medium/large and per_class_AP.

    gt: COCO GT dict or json path. preds: COCO results list or json path.
    A value of -1 means "no GT in this size range" (COCO convention);
    per_class_AP is None for classes without GT.
    """
    gt, preds = _load(gt), _load(preds)
    names = {c["id"]: c["name"] for c in gt["categories"]}
    if len(preds) == 0:  # loadRes fails on an empty list
        return {**{k: 0.0 for k in STAT_NAMES}, "per_class_AP": {n: 0.0 for n in names.values()}}

    log = io.StringIO()
    with contextlib.redirect_stdout(log):  # pycocotools prints a lot
        coco_gt = COCO()
        coco_gt.dataset = {"info": {}, "licenses": [], **gt}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.maxDets = [1, 100, max_dets]  # default is 100: too few for ~46 objects/frame (max 180)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    if verbose:
        print(log.getvalue())

    metrics = {k: float(v) for k, v in zip(STAT_NAMES, ev.stats[:6])}
    # precision shape: [IoU thr, recall, class, area range, maxDets]; area 0 = all, maxDets -1 = max_dets
    prec = ev.eval["precision"][:, :, :, 0, -1]
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        p = prec[:, :, k]
        p = p[p > -1]
        per_class[names[cat_id]] = float(p.mean()) if p.size else None
    metrics["per_class_AP"] = per_class
    return metrics

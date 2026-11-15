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

STAT_NAMES = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]


def _load(obj):
    """Accept a path to a json file or an already loaded dict/list."""
    if isinstance(obj, (str, Path)):
        with open(obj) as fh:
            return json.load(fh)
    return obj


def _mean(p) -> float:
    """Mean over valid entries (-1 = no GT), same rule as COCOeval.summarize; -1 if nothing valid."""
    p = p[p > -1]
    return float(p.mean()) if p.size else -1.0


def to_coco_results(image_id, boxes_xyxy, scores, classes) -> list[dict]:
    """Detections of one image in our convention (pixel xyxy, class 0-7) -> COCO results list."""
    # int ids for val (COCO GT), string ids (e.g. "clip_001_0000001") for the test submission
    image_id = image_id if isinstance(image_id, str) else int(image_id)
    return [{"image_id": image_id, "category_id": int(c) + 1,
             "bbox": [round(float(v), 2) for v in box], "score": round(float(s), 5)}
            for box, s, c in zip(xyxy_to_coco(boxes_xyxy), scores, classes)]


def _load_results(coco_gt: COCO, preds: list[dict]) -> COCO:
    """Light replacement of COCO.loadRes for bbox results.

    loadRes also adds a polygon per detection (never used for bbox eval), which roughly doubles
    RAM for ~1.4M val detections and killed the first Kaggle run.
    """
    img_ids = set(coco_gt.getImgIds())
    anns = []
    for i, d in enumerate(preds, start=1):
        if d["image_id"] not in img_ids:
            raise ValueError(f"prediction for unknown image_id {d['image_id']}")
        w, h = d["bbox"][2], d["bbox"][3]
        anns.append({"id": i, "image_id": d["image_id"], "category_id": d["category_id"],
                     "bbox": d["bbox"], "score": d["score"], "area": w * h, "iscrowd": 0})
    coco_dt = COCO()
    coco_dt.dataset = {"images": coco_gt.dataset["images"], "categories": coco_gt.dataset["categories"],
                       "annotations": anns}
    coco_dt.createIndex()
    return coco_dt


def coco_evaluate(gt, preds, max_dets: int = 500, verbose: bool = False) -> dict:
    """Return mAP, AP50, AP75, AP_small/medium/large and per_class_AP, all at `max_dets` per image.

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
        coco_dt = _load_results(coco_gt, preds)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.maxDets = [1, 100, max_dets]  # default is 100: too few for ~46 objects/frame (max 180)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    if verbose:
        print(log.getvalue())

    # NOTE: COCOeval.stats[0] is hard-coded to maxDets=100, so all numbers are computed here
    # from the precision array at maxDets[-1] = max_dets.
    # precision shape: [IoU thr (0.50:0.95), recall, class, area (all, small, medium, large), maxDets]
    p = ev.eval["precision"][..., -1]
    metrics = {"mAP": _mean(p[:, :, :, 0]), "AP50": _mean(p[0, :, :, 0]), "AP75": _mean(p[5, :, :, 0]),
               "AP_small": _mean(p[:, :, :, 1]), "AP_medium": _mean(p[:, :, :, 2]),
               "AP_large": _mean(p[:, :, :, 3])}
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        ap = _mean(p[:, :, k, 0])
        per_class[names[cat_id]] = ap if ap > -1 else None
    metrics["per_class_AP"] = per_class
    return metrics

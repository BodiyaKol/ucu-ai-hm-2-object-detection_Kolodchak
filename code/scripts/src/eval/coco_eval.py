"""pycocotools wrapper. Ignore regions must be in GT as iscrowd=1 so detections on them are not FP."""


def coco_evaluate(gt_json: str, pred_json: str, max_dets: int = 500) -> dict:
    """Return {"mAP": .., "AP50": .., "AP75": .., "AP_small": .., "AP_medium": .., "AP_large": ..}.

    TODO: COCO(gt) / loadRes(pred) / COCOeval; set params.maxDets = [1, 100, max_dets].
    """
    raise NotImplementedError

"""Format conversions and frame subsampling.

VisDrone annotation row (TODO: verify): frame, target_id, x, y, w, h, score, category, truncation, occlusion
"""


def subsample_frames(frame_ids: list[int], step: int) -> list[int]:
    """Keep every `step`-th frame of a clip."""
    return sorted(frame_ids)[::step]


def xywh_to_yolo(box, img_w: int, img_h: int):
    """Top-left (x, y, w, h) in pixels -> YOLO normalized (cx, cy, w, h)."""
    x, y, w, h = box
    return ((x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h)


def yolo_to_xywh(box, img_w: int, img_h: int):
    """YOLO normalized (cx, cy, w, h) -> top-left (x, y, w, h) in pixels."""
    cx, cy, w, h = box
    w_px, h_px = w * img_w, h * img_h
    return (cx * img_w - w_px / 2, cy * img_h - h_px / 2, w_px, h_px)


def visdrone_to_yolo(raw_dir, out_dir, frame_step: int):
    """TODO: write images/ + labels/ in Ultralytics format."""
    raise NotImplementedError


def visdrone_to_coco(raw_dir, out_json, frame_step: int):
    """TODO: write COCO json (ignore regions as iscrowd=1)."""
    raise NotImplementedError

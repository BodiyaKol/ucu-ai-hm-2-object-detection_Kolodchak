"""Builders for torchvision Faster R-CNN (M3) and RetinaNet (M4) from a config dict."""


def build_faster_rcnn(cfg: dict, num_classes: int):
    """TODO: fasterrcnn_resnet50_fpn_v2 + replace box predictor, apply RPN/RoI params from cfg."""
    raise NotImplementedError


def build_retinanet(cfg: dict, num_classes: int):
    """TODO: retinanet_resnet50_fpn_v2 + custom AnchorGenerator (k-means), focal alpha/gamma, IoU thresholds."""
    raise NotImplementedError

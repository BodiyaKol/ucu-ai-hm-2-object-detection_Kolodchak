"""Builders for torchvision Faster R-CNN (M3) and RetinaNet (M4) from the `model:` config block.

Both: COCO-pretrained ResNet-50 FPN v2 backbone, images resized so the longest side = imgsz,
configurable anchor sizes (one base size per FPN level P2..P6, ratios 0.5/1/2),
up to `max_dets` detections per image with score >= `score_thresh` (competition protocol).
"""
from functools import partial

import torch
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights, RetinaNet_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def build_model(mcfg: dict, num_classes: int, imgsz: int, max_dets: int = 500, score_thresh: float = 0.001):
    sizes = mcfg["anchor_sizes"]
    ratios = ((0.5, 1.0, 2.0),) * len(sizes)
    common = dict(min_size=imgsz, max_size=imgsz)  # longest side -> imgsz, aspect ratio kept

    if mcfg["name"] == "faster_rcnn":
        n = mcfg["rpn_post_nms_top_n"]
        m = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, **common,
            rpn_pre_nms_top_n_train=2 * n, rpn_post_nms_top_n_train=n,
            rpn_pre_nms_top_n_test=2 * n, rpn_post_nms_top_n_test=n,
            box_batch_size_per_image=mcfg["box_batch_size_per_image"],
            box_detections_per_img=max_dets, box_score_thresh=score_thresh)
        m.rpn.anchor_generator = AnchorGenerator(tuple((s,) for s in sizes), ratios)  # 3 anchors / location
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes + 1)  # + background
    elif mcfg["name"] == "retinanet":
        m = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1, **common,
            score_thresh=score_thresh, detections_per_img=max_dets, topk_candidates=mcfg["topk_candidates"])
        # 3 scales x 3 ratios = 9 anchors / location, same as the default (regression head is kept)
        m.anchor_generator = AnchorGenerator(tuple((s, s * 2 ** (1 / 3), s * 2 ** (2 / 3)) for s in sizes), ratios)
        na = m.anchor_generator.num_anchors_per_location()[0]
        m.head.classification_head = RetinaNetClassificationHead(
            m.backbone.out_channels, na, num_classes + 1, norm_layer=partial(torch.nn.GroupNorm, 32))
    else:
        raise ValueError(mcfg["name"])
    return m

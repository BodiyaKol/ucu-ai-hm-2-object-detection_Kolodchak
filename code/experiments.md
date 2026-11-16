# Experiments

## Experiment log

All val numbers: our COCO evaluator (`scripts/src/eval/coco_eval.py`) on all 2,846 val frames, ignore regions as iscrowd,
top-500 detections, score ≥ 0.001, batch-1 inference, Kaggle T4. Full metrics per run: `results/<id>.json`.
LB = public / private leaderboard. Runs marked *late* were submitted after the deadline (for the report only).

| id | milestone | model / change | config | imgsz | training | val mAP | AP50 | AP_small | bus AP | img/s | LB public / private | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m2_yolo_base | M2 | YOLO11s (COCO), Ultralytics defaults, close_mosaic 2 | `yolo/base.yaml` | 1024 | 10 ep, 15 min | 0.186 | 0.376 | 0.103 | 0.011 | 22.6 | 0.212 / 0.195 | first valid submission; below baseline.csv (0.2216). Ultralytics val (multi-label NMS, no ignore) 0.198 |
| m3_frcnn_base | M3 | Faster R-CNN R50-FPN v2, torchvision defaults | `faster_rcnn/base.yaml` | 1024 | 5 ep, 25 min cap | **0.212** | 0.427 | 0.092 | 0.042 | 6.5 | 0.202 / 0.187 *late* | best AP75 / AP_large among CNNs |
| m3_frcnn_small | M3 | + k-means anchors 8..128 | `faster_rcnn/small_anchors.yaml` | 1024 | 5 ep, 25 min cap | 0.204 | 0.430 | 0.090 | 0.053 | 6.8 | 0.178 / 0.171 *late* | **failed**: small anchors did not help |
| m3_frcnn_rpn2000 | M3 | + small anchors + 2000 proposals | `faster_rcnn/rpn_2000.yaml` | 1024 | 5 ep, 25 min cap | 0.198 | 0.424 | 0.099 | 0.031 | 6.0 | 0.179 / 0.163 *late* | **failed**: AP_small up, mAP down |
| m4_retina_small | M4 | RetinaNet R50-FPN v2, k-means anchors 8..128 | `retinanet/base.yaml` | 1024 | 7 ep, 25 min cap | 0.056 | 0.174 | 0.031 | 0.001 | 7.5 | 0.063 / 0.048 *late* | **failed**: new head not trained in 25 min |
| m7_m1024 | M5 / M7 B | YOLO11m (bigger model) | `yolo/final_m1024.yaml` | 1024 | 32 ep, 1.5 h cap | 0.177 | 0.350 | 0.108 | 0.045 | 38.6 | 0.211 / 0.189 | finished after the deadline (Kaggle queue) |
| m7_s1280 | M5 / M7 A | YOLO11s, imgsz 1280, close_mosaic 5 | `yolo/final_s1280.yaml` | 1280 | 37 ep, 1.5 h cap | 0.185 | 0.374 | 0.102 | 0.008 | 37.6 | 0.287 / 0.192 | **submitted final**. Public gain did not hold on private |
| m6_rtdetr | M6 | RT-DETR-L (COCO), 300 queries | `rtdetr/base.yaml` | 1024 | 8 ep, 45 min cap | **0.252** | **0.493** | **0.139** | **0.172** | 16.0 | **0.340 / 0.270** *late* | **final model of the report**: best on val and both LB parts |

## Conclusions

- **Model family > hyper-parameters:** RT-DETR wins every slice (small objects, large objects, AP75, rare class bus).
  Anchors, proposals, YOLO model size and longer training did not move val within 25-90 min budgets.
- **Val vs LB:** the 1280 YOLO jumped on public (0.212 → 0.287) with no val change; private (0.195 vs 0.192) confirmed
  val was right and the public test clips were atypical (lower resolution, see report section 3, point 6).
- **Small objects** are the main weakness of every model (AP_small ≈ half of mAP); **bus** is the worst class everywhere.

## Protocol decisions (M1.4)

| Decision | Value | Evidence |
|---|---|---|
| Split | 56 train clips / 7 val clips (by sequence) | frames of one clip are highly correlated |
| Frame step (train) | 10 (2,446 frames) | `results/eda_choices.json` → temporal: 0.45% new objects at step 1, 4.2% at step 10; `results/prepare_stats.json` |
| Frame step (val) | 1 (all 2,846 frames) | test is scored on every frame |
| Input size | 1024 (longest side) | `results/eda.json` → resize: 20% of objects < 16 px at 1024 vs 54% at 640 |
| Ignored regions | masked in training, iscrowd=1 × 8 classes in eval GT | M1.1 format check; organisers' COCO uses the same scheme |
| Eval | COCO mAP, max 500 dets, score ≥ 0.001 | competition rules |
| Anchors (M3/M4) | default vs k-means (k=9) | avg best IoU 0.547 vs 0.727; 69% of boxes < 32 px |

Known risk from M1: test resolution is lower than train/val (median longest side 1360 vs 1904, clips down to 680 px).
Checked: on public LB the 1280 model gained +0.075 with no val change, on private it did not (see Conclusions).

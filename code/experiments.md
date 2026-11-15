# Experiments

## Experiment log

All val numbers: our COCO evaluator (`scripts/src/eval/coco_eval.py`) on all 2,846 val frames, ignore regions as iscrowd,
top-500 detections, score ≥ 0.001, batch-1 inference. Full metrics per run: `results/<id>.json`.

| id | model / change | imgsz | epochs | val mAP | AP50 | AP_small | public LB | GPU time (T4) | notes |
|---|---|---|---|---|---|---|---|---|---|
| m2_yolo_base | YOLO11s (COCO-pretrained), Ultralytics defaults, close_mosaic 2 | 1024 | 10 | 0.186 | 0.376 | 0.103 | 0.2121 | 15 min train + 8 min infer | Organisers' baseline.csv on LB ≈ 0.2216 (not beaten yet). Ultralytics val (multi-label NMS, no ignore) 0.198. Bus AP 0.011 (Ultralytics 0.063). LB > val: test frames are downscaled less at 1024 (median 1360 vs 1904 px). |

## Protocol decisions (M1.4)

| Decision | Value | Evidence |
|---|---|---|
| Split | train clips / 7 val clips (by sequence) | frames of one clip are highly correlated |
| Frame step (train) | 10 (2,446 frames) | `results/eda_choices.json` → temporal: 0.45% new objects at step 1, 4.2% at step 10; `results/prepare_stats.json` |
| Frame step (val) | 1 (all 2,846 frames) | test is scored on every frame |
| Input size | 1024 (longest side) | `results/eda.json` → resize: 20% of objects < 16 px at 1024 vs 54% at 640 |
| Ignored regions | masked in training, iscrowd=1 × 8 classes in eval GT | M1.1 format check; organisers' COCO uses the same scheme |
| Eval | COCO mAP, max 500 dets, score ≥ 0.001 | competition rules |
| Anchors (M4) | default vs k-means (k=9) | avg best IoU 0.547 vs 0.727; 69% of boxes < 32 px |

Known risk: test resolution is lower than train/val (median longest side 1360 vs 1904, clips down to 680 px),
so val mAP may overestimate the leaderboard score.
**Checked in M2 (m2_yolo_base):** the opposite happened, LB 0.212 > val 0.186. At a fixed imgsz of 1024, test frames are
shrunk less (×0.75 vs ×0.54), so objects keep more pixels. A low-resolution val check is still planned for M7.

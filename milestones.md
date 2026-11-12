# HW2 Milestone Plan (VisDrone-VID detection)

Code + report live in the SAME template repo. Never touch index.html, assets/, .github/.

```
repo/
├── report.md, images/
├── requirements.txt
├── src/            # shared: data, eval, submission
├── notebooks/      # one clean notebook per milestone/model
└── experiments.md  # log of every experiment (incl. failures)
```

| # | Milestone | Points | Key outputs |
|---|-----------|--------|-------------|
| M0 | Setup & fixed protocol | – | repo skeleton, Kaggle env, W&B, split by clip (val/ = holdout), subsampling, imgsz, budget, seed |
| M1 | Data understanding & preprocessing | 1.0 | class/size/density EDA per clip, k-means anchors, ignore-region masking, GT visualization, decisions |
| M2 | Eval + submission tooling, first valid submit | – | pycocotools eval (mAP, AP50, AP_small, FPS, iscrowd ignores), CSV writer, YOLO baseline, local vs LB check |
| M3 | Two-stage: Faster R-CNN (torchvision) | 1.0 | 2+ params (RPN proposals, RoI sampling, GIoU loss), metrics, 3+ failure cases, RPN vs 2nd stage role |
| M4 | One-stage anchor-based: RetinaNet | 1.0 | anchors from k-means, focal α/γ, IoU thresholds, compare vs Faster R-CNN, no-proposal issues |
| M5 | Anchor-free: YOLO (Ultralytics) | 1.0 | imgsz, P2 head, loss/assigner; search range; by-size results; anchor-free assignment explanation |
| M6 | Query-based: RT-DETR | 0.75 | num queries vs density, backbone LR, mAP-vs-epoch plot vs YOLO, matching vs NMS |
| M7 | Your pipeline | 1.25 | data/inference (SAHI, TTA, Soft-NMS) separated from model changes (bigger model, WBF); summary table; pick 2 finals by val |
| M8 | Bonus: temporality | +1 | tracker / Seq-NMS rescoring, val gain |
| M9 | Bonus: visualization | +1 | W&B project, Grad-CAM or attention maps + interpretation |
| M10 | Report & reproducibility | 2.0 | ~1500 words, own text, run instructions + seeds, clean notebooks, citations, push before deadline |

GPU budget (~60h): M2 2h · M3 8h · M4 8h · M5 10h · M6 8h · M7 15h · bonus 5h · reserve 4h.

Report note: AI-written reports are not tolerated — Claude provides facts/pointers per milestone, the user writes the text.
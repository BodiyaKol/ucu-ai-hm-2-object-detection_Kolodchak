## Protocol decisions (M1.4)

| Decision | Value | Evidence |
|---|---|---|
| Split | train clips / 7 val clips (by sequence) | frames of one clip are highly correlated |
| Frame step (train) | 10 (~2,420 frames) | `results/eda_choices.json` → temporal: 0.45% new objects at step 1, 4.2% at step 10 |
| Frame step (val) | 1 (all 2,846 frames) | test is scored on every frame |
| Input size | 1024 (longest side) | `results/eda.json` → resize: 20% of objects < 16 px at 1024 vs 54% at 640 |
| Ignored regions | masked in training, iscrowd=1 × 8 classes in eval GT | M1.1 format check; organisers' COCO uses the same scheme |
| Eval | COCO mAP, max 500 dets, score ≥ 0.001 | competition rules |
| Anchors (M4) | default vs k-means (k=9) | avg best IoU 0.547 vs 0.727; 69% of boxes < 32 px |

Known risk: test resolution is lower than train/val (median longest side 1360 vs 1904, clips down to 680 px),
so val mAP may overestimate the leaderboard score.


# HW2 — Object Detection on VisDrone-VID: code

> TODO: fill in before submission (M10). Everything below must be enough to reproduce report numbers.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

TODO: python version, CUDA version used on Kaggle.

## 2. Data

TODO: where to download the dataset, Kaggle dataset slug.

Expected local layout (gitignored):

```
code/data/
├── raw/          # original VisDrone-VID (sequences/ + annotations/)
└── processed/    # output of scripts/prepare_data.py (YOLO / COCO format)
```

On Kaggle the paths are resolved automatically in `scripts/src/config.py` (TODO: check).

## 3. Prepare data

```bash
cd scripts
python prepare_data.py --protocol ../configs/protocol.yaml
python run_eda.py      --protocol ../configs/protocol.yaml   # plots -> ../../images/
```

## 4. Train

Training runs on Kaggle GPU via `notebooks/0X_*.ipynb`. Each notebook takes one config from `configs/`.

| Milestone | Notebook | Config |
|-----------|----------|--------|
| M2 | 02_baseline_submission.ipynb | configs/yolo/base.yaml |
| M3 | 03_faster_rcnn.ipynb | configs/faster_rcnn/*.yaml |
| M4 | 04_retinanet.ipynb | configs/retinanet/*.yaml |
| M5 | 05_yolo_anchor_free.ipynb | configs/yolo/*.yaml |
| M6 | 06_rtdetr.ipynb | configs/rtdetr/*.yaml |
| M7 | 07_final_pipeline.ipynb | TODO |

## 5. Evaluate & submit

```bash
cd scripts
python evaluate.py        --gt <coco_gt.json> --pred <preds.json> --out ../results/<exp_id>.json
python make_submission.py --pred <preds.json> --out ../outputs/submission.csv
```

## 6. Seeds & reproducibility

- Global seed: `seed` in `configs/protocol.yaml` (TODO: value).
- Final submissions: TODO (exp id, config, weights link).

## 7. Tests

```bash
cd scripts && pytest tests -q
```

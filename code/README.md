# HW2: Object Detection on VisDrone-VID (code)

Code for the report in [`../report.md`](../report.md). **Final model: RT-DETR-L** (`configs/rtdetr/base.yaml`,
imgsz 1024, 300 queries, 45 min on a Kaggle T4): val mAP 0.252, LB 0.340 public / 0.270 private.
The submission chosen before the deadline was YOLO11s @1280 (`configs/yolo/final_s1280.yaml`, LB 0.287 / 0.192),
because the RT-DETR run finished afterwards. All runs: [`experiments.md`](experiments.md).

## 1. How the repo works

One idea everywhere: **one protocol, one evaluator, one prediction format for every model.**

- Every script is run **from `code/`** and imports the shared library `scripts/src/`.
- Every script works in two places without changes:
  - **locally (CPU)** on the mini dataset `data/mini` (2 clips x 20 frames per split, same layout as the competition), with `--mini` settings from the config;
  - **in a Kaggle kernel (GPU T4)** on the full data. `src/config.py` detects Kaggle (non-empty `/kaggle/input`) and finds the competition data and the prepared data there.
- Heavy runs are Kaggle **script kernels** in `notebooks/<NN_name>/` (`run.py` + `kernel-metadata.json`). `run.py` clones this repo from GitHub and calls the same scripts, so **commit + push before `kaggle kernels push`**.
- Every run writes `outputs/<exp>/metrics.json` (gitignored). The small JSON with the numbers is copied to `results/<exp>.json` (committed) and gets a row in `experiments.md`.

```
code/
├── configs/
│   ├── protocol.yaml           # shared: seed 42, frame_step 10, imgsz 1024, 500 dets, score 0.001, class names
│   ├── rtdetr/base.yaml        # FINAL model: rtdetr-l.pt, imgsz 1024, 0.75 h time cap
│   ├── yolo/                   # base (m2), final_s1280 (m7, deadline submission), final_m1024 (m7 B)
│   ├── faster_rcnn/            # base (defaults) -> small_anchors -> rpn_2000 (configs inherit with `base:`)
│   └── retinanet/base.yaml     # k-means anchors 8..128
├── scripts/                    # CLI entry points (run from code/)
│   ├── prepare_data.py         # M1.5: train subset (every 10th frame, ignore regions painted), YOLO + COCO files
│   ├── train_yolo.py           # YOLO and RT-DETR (Ultralytics): train -> val COCO eval -> submission.csv
│   ├── train_torchvision.py    # Faster R-CNN / RetinaNet: time-capped train -> val eval -> failure cases
│   ├── evaluate.py             # predictions json -> COCO metrics json
│   ├── make_submission.py      # predictions json -> Kaggle CSV (+ format check)
│   ├── run_eda.py, analyze_choices.py, visualize_gt.py, check_format.py   # M1: EDA plots -> ../images/
│   ├── src/                    # importable library (below)
│   └── tests/                  # pytest on tiny fake data
├── notebooks/
│   ├── final_rtdetr_walkthrough.ipynb   # step-by-step run of the final model on mini (Colab / local)
│   └── NN_*/                   # Kaggle kernels: 00_setup, 01_prepare, 02_baseline, 02b_infer, 03_torchvision,
│                               # 06_rtdetr, 07a/07b_final, 08a/08b_late (test inference from saved weights)
├── results/                    # committed: metrics per experiment, EDA numbers, kernel logs
├── experiments.md              # experiment log (one row per run)
├── data/                       # gitignored: data/mini, data/annotations
└── outputs/                    # gitignored: weights, predictions, submission.csv
```

### The library `scripts/src/`

| module | what to use it for |
|---|---|
| `config.py` | `load_config(path)` (YAML with `base:` inheritance), `set_seed`, `get_data_root(cfg)`, `get_output_dir(exp)`, `is_kaggle()` |
| `data/convert.py` | box conversions (`yolo_to_xyxy`, `xyxy_to_coco`, `coco_to_xyxy`), `iou_matrix`, `build_coco` (GT with ignore regions as `iscrowd=1`) |
| `data/ignore.py` | painting ignored regions in training images |
| `data/yolo_yaml.py` | `data.yaml` for Ultralytics from the prepared folder |
| `data/dataset.py` | `VisDroneDetection` torch Dataset + `collate_fn` for torchvision models |
| `models/torchvision_models.py` | `build_model(model_cfg, ...)`: Faster R-CNN / RetinaNet with custom anchors |
| `eval/coco_eval.py` | `coco_evaluate(gt, preds)`: mAP, AP50, AP75, AP_small/medium/large, per-class AP, **all at 500 detections** |
| `submission.py` | `predictions_to_csv`, `check_submission` (top-left xywh, 0-based class ids, ≤ 500 per image) |
| `viz.py` | `draw_boxes`, `save_failure_cases` (lowest-recall val frames) |

Prediction format inside the code: COCO results `{"image_id", "category_id" (1..8), "bbox" [x, y, w, h], "score"}`.
`to_coco_results` in `eval/coco_eval.py` converts model outputs (pixel xyxy, classes 0..7) into it.

## 2. Install

Python 3.14 locally (Kaggle: 3.13 with its preinstalled CUDA torch, the kernels only `pip install ultralytics==8.4.214 pycocotools`).

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # CPU torch 2.15, ultralytics 8.4.214, pycocotools, kaggle CLI
pytest scripts/tests -q                # 20 passed
```

Kaggle CLI: put the API token into `~/.kaggle/` (`chmod 600`).

## 3. Data

- **Mini (local / Colab):** produced by the `hw2-00-setup` kernel (`notebooks/00_setup/`), unpack into `code/data/mini/`
  (`train/`, `val/`, `test/`, `test_images.csv`). The walkthrough notebook downloads it from Google Drive.
- **Full (Kaggle):** attach the competition `multiple-object-video-detection-hw-2-ucu-ai-2026` (already set in every `kernel-metadata.json`).
- **Annotations for EDA:** `code/data/annotations/` (from the setup kernel), used by `run_eda.py` and `analyze_choices.py`.

## 4. Reproduce the final model (RT-DETR-L)

**Locally on mini (CPU, ~5 min)**, the same chain as on Kaggle:
```bash
python scripts/prepare_data.py                                               # data/mini -> outputs/prepared
python scripts/train_yolo.py --exp nb_mini --mini --config configs/rtdetr/base.yaml
cat outputs/nb_mini/metrics.json                                             # val metrics + submission check
```
`train_yolo.py` picks `RTDETR` or `YOLO` from the weights name in the config. Useful flags: `--train-only`,
`--weights <last.pt>` (inference only), `--skip-test` (val only).
Or open `notebooks/final_rtdetr_walkthrough.ipynb` (Colab or Jupyter) and run it top to bottom.

**Full run on Kaggle GPU:**
```bash
git push                                                          # kernels clone the repo from GitHub
kaggle kernels push -p notebooks/01_prepare                       # once: prepared train subset (hw2-01-prepare)
kaggle kernels push -p notebooks/06_rtdetr --accelerator NvidiaTeslaT4      # train (0.75 h) + val eval
kaggle kernels push -p notebooks/08b_late_rtdetr --accelerator NvidiaTeslaT4  # val + test from saved weights
kaggle kernels status bohdankolodchak/hw2-08b-late-rtdetr
kaggle kernels output bohdankolodchak/hw2-08b-late-rtdetr -p outputs/08b
kaggle competitions submit -c multiple-object-video-detection-hw-2-ucu-ai-2026 \
    -f outputs/08b/m6_rtdetr/submission.csv -m "m6_rtdetr"
cp outputs/08b/m6_rtdetr/metrics.json results/m6_rtdetr_test.json
```

## 5. Other models

| report section | command / kernel | configs |
|---|---|---|
| M2 YOLO baseline | `python scripts/train_yolo.py --exp m2_yolo_base` · kernels `02_baseline`, `02b_infer` | `yolo/base.yaml` |
| M3 Faster R-CNN, M4 RetinaNet | `python scripts/train_torchvision.py --exp <id> --config <cfg> [--test] [--skip-val] [--weights last.pt]` · kernels `03_torchvision`, `08a_late_torchvision` | `faster_rcnn/*.yaml`, `retinanet/base.yaml` |
| M5/M7 YOLO 1280 / YOLO11m | kernels `07a_final_s1280`, `07b_final_m1024` | `yolo/final_s1280.yaml`, `yolo/final_m1024.yaml` |
| M1 EDA | `python scripts/run_eda.py`, `analyze_choices.py`, `visualize_gt.py`, `check_format.py` | `protocol.yaml` |

Evaluate or convert any saved predictions:
```bash
python scripts/evaluate.py --gt outputs/prepared/annotations/val_coco.json --pred outputs/<exp>/val_preds.json --out results/<exp>.json
python scripts/make_submission.py --pred <test_preds.json> --out outputs/<exp>/submission.csv   # COCO results with test image_id strings
```
(`train_yolo.py` and `train_torchvision.py --test` already write `submission.csv` themselves.)

To add an experiment: new YAML in `configs/<family>/` (with `base: base.yaml` if it changes only a few values) →
mini run locally → commit + push → Kaggle kernel → copy `metrics.json` to `results/<exp>.json` → row in `experiments.md`.

## 6. Reproducibility

- Seed 42 (`configs/protocol.yaml`) for python / numpy / torch and Ultralytics (`seed=`, deterministic by default).
- Fixed protocol for every model: train every 10th frame with ignored regions painted, val = all 2,846 frames of the 7 val clips,
  longest side 1024 (1280 for `final_s1280`), batch-1 inference, score ≥ 0.001, ≤ 500 detections, one COCO evaluator.
- Training is **time-capped** (Ultralytics `time`, torchvision `max_minutes`): the LR schedule is fitted to the budget, so a rerun on another GPU gives a different number of epochs.
- Pinned versions: `requirements.txt` (local) and `ultralytics==8.4.214` in the kernels.

"""M2.3: YOLO baseline (Ultralytics): train -> val COCO eval -> test submission.

The same script runs locally (CPU, data/mini, --mini) and in the Kaggle kernel (GPU, full data).

Usage (from code/):
  python scripts/prepare_data.py                          # once: data/mini -> outputs/prepared
  python scripts/train_yolo.py --exp m2_mini --mini       # CPU dry-run of the whole chain
  python scripts/train_yolo.py --exp m2_yolo_base         # Kaggle (prepared data found automatically)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make src importable
from src.config import CODE_DIR, KAGGLE_INPUT, get_data_root, get_output_dir, is_kaggle, load_config, set_seed
from src.data.yolo_yaml import write_data_yaml
from src.eval.coco_eval import coco_evaluate, to_coco_results
from src.submission import check_submission, predictions_to_csv


def find_prepared(arg) -> Path:
    """Folder with train.txt + annotations/ written by prepare_data.py."""
    if arg:
        return Path(arg)
    if is_kaggle():  # output of the hw2-01-prepare kernel, attached via kernel_sources
        for pattern in ("*/prepared/train.txt", "*/*/prepared/train.txt", "*/*/*/prepared/train.txt"):
            hits = sorted(KAGGLE_INPUT.glob(pattern))
            if hits:
                return hits[0].parent
        raise FileNotFoundError("prepared/train.txt not found under /kaggle/input (attach hw2-01-prepare)")
    prepared = CODE_DIR / "outputs/prepared"
    if not (prepared / "train.txt").exists():
        raise FileNotFoundError("run `python scripts/prepare_data.py` first")
    return prepared


def predict(model, paths, ids, **kw):
    """Run the detector on images; return COCO results (ids[i] belongs to paths[i]) and images/s."""
    preds, t0 = [], time.time()
    results = model.predict([str(p) for p in paths], stream=True, verbose=False, **kw)
    for path, image_id, r in zip(paths, ids, results):
        assert Path(r.path).name == Path(path).name, "prediction order mismatch"
        b = r.boxes
        preds += to_coco_results(image_id, b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                 b.cls.cpu().numpy().astype(int))
    return preds, len(paths) / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment id, e.g. m2_yolo_base")
    ap.add_argument("--config", default="configs/yolo/base.yaml")
    ap.add_argument("--mini", action="store_true", help="local CPU dry-run settings")
    ap.add_argument("--prepared", default=None)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    cfg, ycfg = load_config("configs/protocol.yaml"), load_config(args.config)
    set_seed(cfg["seed"])
    tr = {**ycfg["train"], **(ycfg["mini"] if args.mini else {})}
    device = 0 if torch.cuda.is_available() and not args.mini else "cpu"
    root, prepared, out = get_data_root(cfg), find_prepared(args.prepared), get_output_dir(args.exp)
    print(f"data root: {root}\nprepared: {prepared}\noutput: {out}\ndevice: {device}\nsettings: {tr}")

    # ---- train (no per-epoch validation to save GPU; Ultralytics still validates after the last epoch)
    data_yaml = write_data_yaml(prepared, root, out, cfg["class_names"], cfg["val_frame_step"])
    model = YOLO(ycfg["model"]["weights"])
    t0 = time.time()
    res = model.train(data=str(data_yaml), imgsz=tr["imgsz"], epochs=tr["epochs"], batch=tr["batch"],
                      workers=tr["workers"], close_mosaic=tr["close_mosaic"], device=device,
                      seed=cfg["seed"], val=False,
                      project=str(out), name="train", exist_ok=True)
    train_min = (time.time() - t0) / 60
    ultra_map = float(res.box.map) if res is not None else None  # Ultralytics' own mAP50-95 (no ignore regions)

    model = YOLO(out / "train/weights/last.pt")
    kw = dict(imgsz=tr["imgsz"], conf=cfg["eval"]["score_thresh"], max_det=cfg["eval"]["max_dets"],
              device=device)
    if device != "cpu":
        kw["quantize"] = 16  # FP16 inference on GPU (Ultralytics 8.4 name for the old `half`)

    # ---- val: our COCO evaluator (ignore regions = iscrowd)
    gt = json.load(open(prepared / "annotations/val_coco.json"))
    val_paths = [root / cfg["val_split"] / img["file_name"] for img in gt["images"]]
    val_preds, val_fps = predict(model, val_paths, [img["id"] for img in gt["images"]], **kw)
    json.dump(val_preds, open(out / "val_preds.json", "w"))  # kept for failure analysis later
    metrics = coco_evaluate(gt, val_preds, max_dets=cfg["eval"]["max_dets"], verbose=True)

    # ---- test: submission for every image of test_images.csv
    test_csv = root / "test_images.csv"
    rows = list(csv.DictReader(open(test_csv)))
    rows = [r for r in rows if (root / r["file_name"]).exists()]  # mini has only a few test clips
    if is_kaggle():
        assert len(rows) == 3811, f"expected 3811 test images, found {len(rows)}"
    test_preds, _ = predict(model, [root / r["file_name"] for r in rows], [r["image_id"] for r in rows], **kw)
    predictions_to_csv(test_preds, test_csv, out / "submission.csv")
    sub_stats = check_submission(out / "submission.csv", test_csv)

    summary = {"exp": args.exp, "weights": ycfg["model"]["weights"], "mini": args.mini, **tr,
               "train_time_min": round(train_min, 1), "ultralytics_val_mAP50_95": ultra_map,
               "val": metrics, "val_images_per_s": round(val_fps, 1),
               "n_val_images": len(val_paths), "n_test_images_predicted": len(rows), "submission": sub_stats}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""M2.3: YOLO baseline (Ultralytics): train -> val COCO eval -> test submission.

The same script runs locally (CPU, data/mini, --mini) and in Kaggle kernels (GPU, full data).
Training and inference run in SEPARATE processes on Kaggle: the training process keeps a lot of RAM
(trainer, dataloader workers), and inference in the same process was killed for out-of-memory.

Usage (from code/):
  python scripts/prepare_data.py                                       # once: data/mini -> outputs/prepared
  python scripts/train_yolo.py --exp m2_mini --mini                    # CPU dry-run: train + inference
  python scripts/train_yolo.py --exp m2_mini --mini --train-only       # train only
  python scripts/train_yolo.py --exp m2_mini --mini --weights outputs/m2_mini/train/weights/last.pt  # inference only
"""
import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make src importable
from src.config import CODE_DIR, KAGGLE_INPUT, get_data_root, get_output_dir, is_kaggle, load_config, set_seed
from src.data.yolo_yaml import write_data_yaml
from src.eval.coco_eval import coco_evaluate, to_coco_results
from src.submission import check_submission, predictions_to_csv


def log_mem(stage: str) -> None:
    """Print RAM of this process and of the machine (to see how close we are to the limit)."""
    vm = psutil.virtual_memory()
    print(f"[mem] {stage}: process {psutil.Process().memory_info().rss / 1e9:.1f} GB, "
          f"available {vm.available / 1e9:.1f} / {vm.total / 1e9:.1f} GB", flush=True)


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


def predict(model, paths, ids, list_file: Path, **kw):
    """Run the detector; return COCO results (ids[i] belongs to paths[i]) and images/s.

    Paths are passed through a .txt file: Ultralytics then reads the images lazily, one per batch
    (LoadImagesAndVideos). A Python LIST of paths is instead decoded completely into RAM and run as
    ONE batch (check_source -> autocast_list -> LoadPilAndNumpy). That caused the RAM kill (all val
    frames at once), the CUDA OOM (250-image batch) and predictions that depended on the list length
    (letterbox to a common shape when image sizes in the batch differ).
    """
    list_file.write_text("\n".join(str(Path(p).absolute()) for p in paths) + "\n")
    raw, t0 = [], time.time()
    results = model.predict(str(list_file), stream=True, verbose=False, batch=1, **kw)
    for i, (path, r) in enumerate(zip(paths, results)):
        assert Path(r.path).name == Path(path).name, "prediction order mismatch"
        b = r.boxes
        raw.append((b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)))
        if (i + 1) % 500 == 0:
            log_mem(f"predict {i + 1}/{len(paths)} images")
    fps = len(paths) / (time.time() - t0)
    preds = []  # COCO dicts are built only after all images are done
    for image_id, (boxes, scores, classes) in zip(ids, raw):
        preds += to_coco_results(image_id, boxes, scores, classes)
    return preds, fps



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment id, e.g. m2_yolo_base")
    ap.add_argument("--config", default="configs/yolo/base.yaml")
    ap.add_argument("--mini", action="store_true", help="local CPU dry-run settings")
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--train-only", action="store_true", help="train and stop (inference in a fresh process)")
    ap.add_argument("--weights", default=None, help="skip training; run inference with these weights")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    cfg, ycfg = load_config("configs/protocol.yaml"), load_config(args.config)
    set_seed(cfg["seed"])
    tr = {**ycfg["train"], **(ycfg["mini"] if args.mini else {})}
    device = 0 if torch.cuda.is_available() and not args.mini else "cpu"
    root, prepared, out = get_data_root(cfg), find_prepared(args.prepared), get_output_dir(args.exp)
    print(f"data root: {root}\nprepared: {prepared}\noutput: {out}\ndevice: {device}\nsettings: {tr}", flush=True)
    log_mem("start")

    # ---- train (no per-epoch validation to save GPU; Ultralytics still validates after the last epoch)
    train_info_path = out / "train_info.json"
    if args.weights is None:
        data_yaml = write_data_yaml(prepared, root, out, cfg["class_names"], cfg["val_frame_step"])
        model = YOLO(ycfg["model"]["weights"])
        t0 = time.time()
        res = model.train(data=str(data_yaml), imgsz=tr["imgsz"], epochs=tr["epochs"], batch=tr["batch"],
                          workers=tr["workers"], close_mosaic=tr["close_mosaic"], time=tr.get("time"),device=device,
                          seed=cfg["seed"], val=False, project=str(out), name="train", exist_ok=True)
        train_info = {"train_time_min": round((time.time() - t0) / 60, 1),
                      "ultralytics_val_mAP50_95": float(res.box.map) if res is not None else None}
        train_info_path.write_text(json.dumps(train_info, indent=2))
        log_mem("after train")
        if args.train_only:
            return
        del model, res
        gc.collect()
        weights = out / "train/weights/last.pt"
    else:
        weights = Path(args.weights)
    train_info = json.loads(train_info_path.read_text()) if train_info_path.exists() else {}

    model = YOLO(weights)
    kw = dict(imgsz=tr["imgsz"], conf=cfg["eval"]["score_thresh"], max_det=cfg["eval"]["max_dets"],
              device=device)
    if device != "cpu":
        kw["quantize"] = 16  # FP16 inference on GPU (Ultralytics 8.4 name for the old `half`)

    # ---- val: our COCO evaluator (ignore regions = iscrowd)
    gt = json.load(open(prepared / "annotations/val_coco.json"))
    val_paths = [root / cfg["val_split"] / img["file_name"] for img in gt["images"]]
    val_preds, val_fps = predict(model, val_paths, [img["id"] for img in gt["images"]], out / "val_list.txt", **kw)
    log_mem(f"after val predict ({len(val_preds)} detections)")
    json.dump(val_preds, open(out / "val_preds.json", "w"))  # kept for failure analysis later
    metrics = coco_evaluate(gt, val_preds, max_dets=cfg["eval"]["max_dets"], verbose=True)
    log_mem("after COCO eval")
    del val_preds, gt
    gc.collect()

    # ---- test: submission for every image of test_images.csv
    test_csv = root / "test_images.csv"
    rows = list(csv.DictReader(open(test_csv)))
    rows = [r for r in rows if (root / r["file_name"]).exists()]  # mini has only a few test clips
    if is_kaggle():
        assert len(rows) == 3811, f"expected 3811 test images, found {len(rows)}"
    test_preds, _ = predict(model, [root / r["file_name"] for r in rows], [r["image_id"] for r in rows],
                            out / "test_list.txt", **kw)
    log_mem(f"after test predict ({len(test_preds)} detections)")
    predictions_to_csv(test_preds, test_csv, out / "submission.csv")
    sub_stats = check_submission(out / "submission.csv", test_csv)

    summary = {"exp": args.exp, "weights": ycfg["model"]["weights"], "checkpoint": str(weights),
               "mini": args.mini, **tr, **train_info,
               "val": metrics, "val_images_per_s": round(val_fps, 1),
               "n_val_images": len(val_paths), "n_test_images_predicted": len(rows), "submission": sub_stats}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

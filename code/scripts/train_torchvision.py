"""M3/M4: torchvision Faster R-CNN / RetinaNet: time-capped training -> val COCO eval -> failure cases.

Same protocol and evaluator as the YOLO runs (imgsz = longest side, top-500, score >= 0.001, batch-1 inference).
Usage (from code/):
  python scripts/train_torchvision.py --exp m3_mini --config configs/faster_rcnn/base.yaml --mini
  python scripts/train_torchvision.py --exp m3_frcnn_base --config configs/faster_rcnn/base.yaml [--test]
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make src importable
from src.config import get_data_root, get_output_dir, is_kaggle, load_config, set_seed
from src.data.dataset import VisDroneDetection, collate_fn, load_image
from src.eval.coco_eval import coco_evaluate, to_coco_results
from src.models.torchvision_models import build_model
from src.submission import check_submission, predictions_to_csv
from src.viz import save_failure_cases
from train_yolo import find_prepared, log_mem


def train(model, loader, device, lr, max_minutes, max_epochs, out):
    """SGD + warm-up + cosine decay over the time budget; AMP on GPU. Saves out/last.pt each epoch."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    t0, it, warm, log = time.time(), 0, 100, []
    budget = max_minutes * 60 if max_minutes else None
    for epoch in range(max_epochs):
        model.train()
        running, n = 0.0, 0
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            frac = (time.time() - t0) / budget if budget else it / (max_epochs * len(loader))
            cur_lr = lr * min(1.0, (it + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
            for g in opt.param_groups:
                g["lr"] = cur_lr
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = sum(model(images, targets).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            running, n, it = running + loss.item(), n + 1, it + 1
            if it % 50 == 0:
                print(f"epoch {epoch + 1} iter {it} loss {running / n:.3f} lr {cur_lr:.5f} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if budget and time.time() - t0 > budget:
                break
        log.append({"epoch": epoch + 1, "iters": it, "loss": round(running / max(n, 1), 4),
                    "minutes": round((time.time() - t0) / 60, 1)})
        print(log[-1], flush=True)
        torch.save(model.state_dict(), out / "last.pt")
        if budget and time.time() - t0 > budget:
            break
    return log


@torch.no_grad()
def predict(model, paths, ids, device):
    """Batch-1 inference; returns COCO results (labels 1-8 -> classes 0-7) and images/s."""
    model.eval()
    raw, t0 = [], time.time()
    for i, p in enumerate(paths):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            r = model([load_image(p).to(device)])[0]
        keep = r["labels"] >= 1  # label 0 = background (never predicted by Faster R-CNN; possible in RetinaNet)
        raw.append((r["boxes"][keep].float().cpu().numpy(), r["scores"][keep].float().cpu().numpy(),
                    (r["labels"][keep] - 1).cpu().numpy()))
        if (i + 1) % 500 == 0:
            log_mem(f"predict {i + 1}/{len(paths)} images")
    fps = len(paths) / (time.time() - t0)
    preds = []
    for image_id, (b, s, c) in zip(ids, raw):
        preds += to_coco_results(image_id, b, s, c)
    return preds, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--test", action="store_true", help="also predict test and write submission.csv")
    ap.add_argument("--skip-val", action="store_true", help="no val eval (test inference only)")
    ap.add_argument("--weights", default=None, help="skip training; load this state_dict")
    ap.add_argument("--prepared", default=None)
    args = ap.parse_args()

    cfg, mcfg = load_config("configs/protocol.yaml"), load_config(args.config)
    set_seed(cfg["seed"])
    tr = {**mcfg["train"], **(mcfg["mini"] if args.mini else {})}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.mini else "cpu")
    root, prepared, out = get_data_root(cfg), find_prepared(args.prepared), get_output_dir(args.exp)
    print(f"data root: {root}\nprepared: {prepared}\noutput: {out}\ndevice: {device}\n"
          f"model: {mcfg['model']}\nsettings: {tr}", flush=True)
    log_mem("start")

    model = build_model(mcfg["model"], len(cfg["class_names"]), tr["imgsz"],
                        cfg["eval"]["max_dets"], cfg["eval"]["score_thresh"]).to(device)
    train_log = []
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        ds = VisDroneDetection(prepared / "annotations/train_coco.json", prepared / "train", train=True)
        loader = DataLoader(ds, batch_size=tr["batch"], shuffle=True, num_workers=tr["workers"],
                            collate_fn=collate_fn)
        print(f"train images: {len(ds)}", flush=True)
        train_log = train(model, loader, device, tr["lr"], tr["max_minutes"], tr["epochs"], out)
        log_mem("after train")

    # ---- val: our COCO evaluator + failure cases
    gt = json.load(open(prepared / "annotations/val_coco.json"))
    val_paths = [root / cfg["val_split"] / im["file_name"] for im in gt["images"]]
    metrics, val_fps, failures = None, None, []
    if not args.skip_val:
        val_preds, val_fps = predict(model, val_paths, [im["id"] for im in gt["images"]], device)
        json.dump(val_preds, open(out / "val_preds.json", "w"))
        metrics = coco_evaluate(gt, val_preds, max_dets=cfg["eval"]["max_dets"], verbose=True)
        failures = save_failure_cases(gt, val_preds, root / cfg["val_split"], out / "failures")

    sub_stats = None
    if args.test:
        test_csv = root / "test_images.csv"
        rows = [r for r in csv.DictReader(open(test_csv)) if (root / r["file_name"]).exists()]
        test_preds, _ = predict(model, [root / r["file_name"] for r in rows], [r["image_id"] for r in rows], device)
        predictions_to_csv(test_preds, test_csv, out / "submission.csv")
        sub_stats = check_submission(out / "submission.csv", test_csv)

    summary = {"exp": args.exp, "config": args.config, "model": mcfg["model"], "mini": args.mini, **tr,
               "train_log": train_log, "val": metrics, "val_images_per_s": val_fps and round(val_fps, 1),
               "n_val_images": len(val_paths), "failure_cases": failures, "submission": sub_stats}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "train_log"}, indent=2))


if __name__ == "__main__":
    main()

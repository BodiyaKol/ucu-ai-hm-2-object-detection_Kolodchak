"""M1.3: statistics behind design choices.

1. Anchor shapes: k-means with IoU distance (YOLOv2) on box sizes at training scale,
   compared with the default RetinaNet anchors.
2. Temporal redundancy: how similar frames t and t+k are (same-track IoU, share of new objects)
   -> justification for frame_step.
3. Resolution gap: image sizes per clip in train / val / test.

Usage (from code/):  python scripts/analyze_choices.py
Outputs: images/eda_anchors.png, images/eda_temporal.png, results/eda_choices.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make src and run_eda importable
from run_eda import IMG_DIR, RES_DIR, load_split, pct, save, style
from src.config import CODE_DIR, load_config

K_VALUES = [3, 6, 9, 12]          # number of anchor clusters to compare
STEPS = [1, 2, 3, 5, 10, 20]      # frame distances for redundancy analysis
SAMPLE = 100_000                  # boxes used for k-means (speed)


# ---------------------------------------------------------------- 1. anchors

def wh_iou(wh, anchors) -> np.ndarray:
    """IoU between boxes and anchors when both are centred at the same point: (N, 2) x (K, 2) -> (N, K)."""
    inter = (np.minimum(wh[:, None, 0], anchors[None, :, 0]) *
             np.minimum(wh[:, None, 1], anchors[None, :, 1]))
    union = (wh[:, 0] * wh[:, 1])[:, None] + (anchors[:, 0] * anchors[:, 1])[None, :] - inter
    return inter / union


def kmeans_iou(wh, k: int, seed: int, iters: int = 100) -> np.ndarray:
    """k-means with distance 1 - IoU and median update (YOLOv2). Returns (k, 2) sorted by area."""
    rng = np.random.default_rng(seed)
    centers = wh[rng.choice(len(wh), k, replace=False)]
    for _ in range(iters):
        assign = wh_iou(wh, centers).argmax(axis=1)
        new = np.array([np.median(wh[assign == j], axis=0) if np.any(assign == j) else centers[j]
                        for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return centers[np.argsort(centers.prod(axis=1))]


def retinanet_default_anchors() -> np.ndarray:
    """torchvision RetinaNet defaults: 5 sizes x 3 scales x 3 aspect ratios (h/w) = 45 (w, h) shapes."""
    shapes = []
    for size in (32, 64, 128, 256, 512):
        for scale in (1.0, 2 ** (1 / 3), 2 ** (2 / 3)):
            for ratio in (0.5, 1.0, 2.0):
                s = size * scale
                shapes.append((s / np.sqrt(ratio), s * np.sqrt(ratio)))
    return np.array(shapes)


def anchor_analysis(objs, imgsz: int, seed: int):
    """Box shapes rescaled to training size (longest side = imgsz) -> anchor fit statistics."""
    scale = imgsz / np.maximum(objs["width"], objs["height"]).to_numpy()
    wh = np.stack([objs["w"].to_numpy() * scale, objs["h"].to_numpy() * scale], axis=1)
    wh = wh[(wh > 0).all(axis=1)]
    rng = np.random.default_rng(seed)
    wh = wh[rng.choice(len(wh), min(SAMPLE, len(wh)), replace=False)]

    result = {"imgsz": imgsz, "avg_best_iou": {}}
    centers9 = None
    for k in K_VALUES:
        centers = kmeans_iou(wh, k, seed)
        result["avg_best_iou"][f"kmeans_k{k}"] = round(float(wh_iou(wh, centers).max(axis=1).mean()), 3)
        if k == 9:
            centers9 = centers
    default = retinanet_default_anchors()
    result["avg_best_iou"]["retinanet_default_45"] = round(float(wh_iou(wh, default).max(axis=1).mean()), 3)

    ratio = wh[:, 1] / wh[:, 0]
    result["kmeans_k9_wh"] = centers9.round(1).tolist()
    result["kmeans_k9_sqrt_area"] = np.sqrt(centers9.prod(axis=1)).round(1).tolist()
    result["aspect_h_over_w"] = dict(zip(["q10", "q50", "q90"], np.quantile(ratio, [0.1, 0.5, 0.9]).round(2).tolist()))
    result["boxes_below_32px_%"] = pct(np.sqrt(wh.prod(axis=1)) < 32)
    return result, wh, centers9, default


def plot_anchors(wh, centers, default, imgsz):
    fig, ax = plt.subplots(figsize=(6, 5))
    idx = np.random.default_rng(0).choice(len(wh), min(20_000, len(wh)), replace=False)
    ax.scatter(wh[idx, 0], wh[idx, 1], s=2, alpha=0.08, color="#2a78d6", label="boxes (sample)")
    ax.scatter(default[:, 0], default[:, 1], s=40, facecolors="none", edgecolors="#0b0b0b",
               marker="s", label="RetinaNet default anchors")
    ax.scatter(centers[:, 0], centers[:, 1], s=90, color="#eb6834", marker="X", label="k-means anchors (k=9)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, f"Box shapes at imgsz={imgsz} vs anchors", "width, px (log)", "height, px (log)")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    save(fig, "eda_anchors.png")


# ---------------------------------------------------------------- 2. temporal redundancy

def paired_iou(a, b) -> np.ndarray:
    """Row-wise IoU between two (N, 4) arrays of top-left (x, y, w, h) boxes."""
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-9)


def temporal_analysis(objs) -> pd.DataFrame:
    """For each step k: IoU of the same track between frames t and t+k, and share of new objects at t+k."""
    o = objs[["video", "frame_id", "track_id", "x", "y", "w", "h"]]
    first_frame = o.groupby("video")["frame_id"].transform("min")
    rows = []
    for k in STEPS:
        later = o.assign(frame_id=o["frame_id"] - k)  # object at t+k, keyed by t
        m = o.merge(later, on=["video", "frame_id", "track_id"], suffixes=("", "_k"))
        iou = paired_iou(m[["x", "y", "w", "h"]].to_numpy(), m[["x_k", "y_k", "w_k", "h_k"]].to_numpy())
        n_later = int((o["frame_id"] - k >= first_frame).sum())  # objects whose frame t exists
        rows.append({"step": k,
                     "mean_iou_same_object": round(float(iou.mean()), 3),
                     "iou_above_0.5_%": pct(iou > 0.5),
                     "new_objects_%": round(100 * (1 - len(m) / n_later), 2)})
    return pd.DataFrame(rows)


def plot_temporal(t):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(t["step"], t["mean_iou_same_object"], marker="o", linewidth=2, color="#2a78d6")
    style(a1, "Same object: box IoU between t and t+k", "frame step k", "mean IoU")
    a2.plot(t["step"], t["new_objects_%"], marker="o", linewidth=2, color="#2a78d6")
    style(a2, "Objects at t+k not present at t", "frame step k", "% of objects")
    for ax in (a1, a2):
        ax.set_xticks(t["step"])
    save(fig, "eda_temporal.png")


# ---------------------------------------------------------------- 3. resolution gap

def resolution_gap(imgs, test_csv: Path) -> dict:
    """Per split: clip count and longest image side (one resolution per clip)."""
    test = pd.read_csv(test_csv).rename(columns={"clip": "video"})
    test["split"] = "test"
    cols = ["split", "video", "width", "height"]
    clips = pd.concat([imgs[cols], test[cols]]).drop_duplicates(["split", "video"]).copy()
    clips["longest"] = np.maximum(clips["width"], clips["height"])

    out = {}
    for split, g in clips.groupby("split"):
        out[split] = {"clips": len(g),
                      "longest_side_median": float(g["longest"].median()),
                      "longest_side_min": int(g["longest"].min()),
                      "longest_side_max": int(g["longest"].max()),
                      "resolutions": (g["width"].astype(str) + "x" + g["height"].astype(str))
                                     .value_counts().to_dict()}
    return out


# ---------------------------------------------------------------- main

def main():
    cfg = load_config("configs/protocol.yaml")
    ann_root = CODE_DIR / cfg["data_dirs"]["annotations"]

    train_objs, train_imgs = load_split(ann_root / "train" / "annotations_coco.json", "train")
    _, val_imgs = load_split(ann_root / "val" / "annotations_coco.json", "val")

    anchors, wh, centers9, default = anchor_analysis(train_objs, cfg["imgsz"], cfg["seed"])
    temporal = temporal_analysis(train_objs)
    resolution = resolution_gap(pd.concat([train_imgs, val_imgs]), ann_root / "test_images.csv")

    out = {"anchors": anchors, "temporal": temporal.to_dict(orient="records"), "resolution": resolution}
    with open(RES_DIR / "eda_choices.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    print("===== anchors =====\n", json.dumps(anchors, indent=1))
    print("\n===== temporal redundancy (train) =====\n", temporal.to_string(index=False))
    print("\n===== resolution per split =====\n", json.dumps(resolution, indent=1))

    plot_anchors(wh, centers9, default, cfg["imgsz"])
    plot_temporal(temporal)


if __name__ == "__main__":
    main()

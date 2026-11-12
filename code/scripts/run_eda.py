"""M1.2: dataset statistics + plots for the report.

Reads the full COCO annotation files (data/annotations/{train,val}/annotations_coco.json).
Writes plots to images/eda_*.png (repo root), numbers to results/eda.json and results/eda_per_clip.csv.

Usage (from code/):  python scripts/run_eda.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make src importable
from src.config import CODE_DIR, load_config
from src.viz import CLASS_NAMES

IMG_DIR = CODE_DIR.parent / "images"
RES_DIR = CODE_DIR / "results"
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834"}
SMALL, MEDIUM = 32, 96                       # COCO size thresholds on sqrt(area), px
IMGSZ_CANDIDATES = [640, 960, 1024, 1280, 1536]


def pct(mask) -> float:
    """Percentage of True values."""
    return round(100 * float(np.mean(mask)), 2)


# ---------------------------------------------------------------- loading

def load_split(path: Path, split: str):
    """COCO json -> (objects df, images df). Crowd regions are de-duplicated (stored once per class)."""
    with open(path) as f:
        coco = json.load(f)

    imgs = pd.DataFrame(coco["images"])[["id", "video", "frame_id", "width", "height"]]
    imgs = imgs.rename(columns={"id": "image_id"})
    imgs["split"] = split

    anns = pd.DataFrame(coco["annotations"])
    del coco  # free memory

    # Objects (iscrowd = 0)
    extra = [c for c in ("occlusion", "truncation", "track_id") if c in anns.columns]
    objs = anns.loc[anns["iscrowd"] == 0, ["image_id", "category_id", "bbox"] + extra].copy()
    bbox = np.array(objs["bbox"].tolist(), dtype=float)
    objs["x"], objs["y"], objs["w"], objs["h"] = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    objs["sqrt_area"] = np.sqrt(objs["w"] * objs["h"])
    objs["cls"] = objs["category_id"] - 1
    objs = objs.drop(columns=["bbox", "category_id"]).merge(imgs, on="image_id")

    # Per-image counts and ignored area (unique crowd regions only)
    imgs["n_objects"] = imgs["image_id"].map(objs.groupby("image_id").size()).fillna(0).astype(int)
    crowd = anns.loc[anns["iscrowd"] == 1, ["image_id", "bbox"]].copy()
    crowd["bbox"] = crowd["bbox"].apply(tuple)
    crowd = crowd.drop_duplicates()
    crowd["area"] = crowd["bbox"].apply(lambda b: b[2] * b[3])
    imgs["n_ignore"] = imgs["image_id"].map(crowd.groupby("image_id").size()).fillna(0).astype(int)
    ign_area = imgs["image_id"].map(crowd.groupby("image_id")["area"].sum()).fillna(0)
    imgs["ignore_frac"] = (ign_area / (imgs["width"] * imgs["height"])).clip(upper=1)  # overlaps ignored
    return objs, imgs


# ---------------------------------------------------------------- statistics

def split_summary(objs, imgs) -> dict:
    """Key numbers for one split."""
    clips = imgs.drop_duplicates("video")
    s = {
        "clips": int(imgs["video"].nunique()),
        "frames": len(imgs),
        "objects": len(objs),
        "objects_per_frame": {"mean": round(imgs["n_objects"].mean(), 1),
                              "median": float(imgs["n_objects"].median()),
                              "max": int(imgs["n_objects"].max())},
        "size_px": {"median": round(float(objs["sqrt_area"].median()), 1),
                    "small_%": pct(objs["sqrt_area"] < SMALL),
                    "medium_%": pct((objs["sqrt_area"] >= SMALL) & (objs["sqrt_area"] < MEDIUM)),
                    "large_%": pct(objs["sqrt_area"] >= MEDIUM)},
        "class_share_%": {CLASS_NAMES[c]: pct(objs["cls"] == c) for c in range(8)},
        "resolutions_by_clip": (clips["width"].astype(str) + "x" + clips["height"].astype(str))
                               .value_counts().to_dict(),
        "frames_with_ignore_%": pct(imgs["n_ignore"] > 0),
        "ignored_area_mean_%": round(100 * imgs["ignore_frac"].mean(), 2),
    }
    if "occlusion" in objs.columns:
        occ = objs["occlusion"].value_counts(normalize=True).sort_index() * 100
        s["occlusion_%"] = {str(k): round(v, 1) for k, v in occ.items()}
    return s


def per_class_table(objs) -> pd.DataFrame:
    """Count, share and size statistics per class."""
    t = objs.groupby("cls").agg(count=("sqrt_area", "size"), median_px=("sqrt_area", "median"),
                                median_w=("w", "median"), median_h=("h", "median"))
    t["share_%"] = 100 * t["count"] / t["count"].sum()
    t["small_%"] = 100 * (objs["sqrt_area"] < SMALL).groupby(objs["cls"]).mean()
    t.index = [CLASS_NAMES[i] for i in t.index]
    return t.round(1)


def resize_table(objs) -> pd.DataFrame:
    """Share of objects below 8 / 16 px after resizing the longest image side to imgsz."""
    longest = np.maximum(objs["width"], objs["height"])
    rows = []
    for s in IMGSZ_CANDIDATES:
        px = objs["sqrt_area"] * s / longest
        rows.append({"imgsz": s, "below_8px_%": pct(px < 8), "below_16px_%": pct(px < 16),
                     "median_px": round(float(px.median()), 1)})
    return pd.DataFrame(rows)


def per_clip_table(objs, imgs) -> pd.DataFrame:
    """One row per clip: resolution, density, size, dominant class, ignored area."""
    fr = imgs.groupby(["split", "video"]).agg(
        frames=("image_id", "size"), width=("width", "first"), height=("height", "first"),
        objects_per_frame=("n_objects", "mean"),
        ignored_area_pct=("ignore_frac", lambda x: 100 * x.mean()))
    ob = objs.groupby(["split", "video"]).agg(
        median_px=("sqrt_area", "median"),
        small_pct=("sqrt_area", lambda x: 100 * (x < SMALL).mean()),
        top_class=("cls", lambda x: CLASS_NAMES[int(x.mode()[0])]))
    return fr.join(ob).round(1).reset_index()


# ---------------------------------------------------------------- plots

def style(ax, title, xlabel, ylabel):
    """Shared minimal style: left title, light y-grid, no top/right spines."""
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG_DIR / name, dpi=150)
    plt.close(fig)
    print("saved", name)


def plot_classes(objs):
    fig, ax = plt.subplots(figsize=(8, 4))
    x, w = np.arange(8), 0.38
    for i, split in enumerate(("train", "val")):
        share = objs.loc[objs["split"] == split, "cls"].value_counts(normalize=True)
        ax.bar(x + (i - 0.5) * w, share.reindex(range(8), fill_value=0) * 100, w,
               color=SPLIT_COLORS[split], label=split)
    ax.set_xticks(x, CLASS_NAMES)
    style(ax, "Class distribution", "", "% of objects")
    ax.legend(frameon=False)
    save(fig, "eda_classes.png")


def plot_sizes(objs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.logspace(0, 3, 60)
    for split in ("train", "val"):
        a1.hist(objs.loc[objs["split"] == split, "sqrt_area"], bins=bins, density=True,
                histtype="step", linewidth=2, color=SPLIT_COLORS[split], label=split)
    a1.set_xscale("log")
    for t, name in ((SMALL, "small | medium"), (MEDIUM, "medium | large")):
        a1.axvline(t, color="#52514e", linestyle="--", linewidth=1)
        a1.text(t, a1.get_ylim()[1] * 0.92, f" {name}", fontsize=8, color="#52514e")
    style(a1, "Object size", "sqrt(box area), px (log scale)", "density")
    a1.legend(frameon=False, loc="upper left")

    train = objs[objs["split"] == "train"]
    a2.boxplot([train.loc[train["cls"] == c, "sqrt_area"] for c in range(8)], showfliers=False,
               patch_artist=True, boxprops=dict(facecolor="#cde2fb", color="#2a78d6"),
               medianprops=dict(color="#0b0b0b"), whiskerprops=dict(color="#2a78d6"),
               capprops=dict(color="#2a78d6"))
    a2.set_xticks(range(1, 9), CLASS_NAMES, rotation=30)
    a2.set_yscale("log")
    a2.axhline(SMALL, color="#52514e", linestyle="--", linewidth=1)
    style(a2, "Object size per class (train)", "", "sqrt(box area), px (log scale)")
    save(fig, "eda_sizes.png")


def plot_density(imgs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(0, imgs["n_objects"].max() + 10, 10)
    for split in ("train", "val"):
        a1.hist(imgs.loc[imgs["split"] == split, "n_objects"], bins=bins, density=True,
                histtype="step", linewidth=2, color=SPLIT_COLORS[split], label=split)
    style(a1, "Objects per frame", "objects in a frame", "density")
    a1.legend(frameon=False)

    clip = imgs.groupby(["split", "video"])["n_objects"].mean().sort_values().reset_index()
    a2.bar(range(len(clip)), clip["n_objects"], color=clip["split"].map(SPLIT_COLORS), width=0.8)
    a2.set_xticks([])
    style(a2, "Mean objects per frame, per clip (sorted)", "clip", "objects / frame")
    a2.legend(handles=[Patch(color=c, label=s) for s, c in SPLIT_COLORS.items()], frameon=False)
    save(fig, "eda_density.png")


def plot_resize(table):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["imgsz"], table["below_8px_%"], marker="o", linewidth=2, color="#2a78d6", label="< 8 px")
    ax.plot(table["imgsz"], table["below_16px_%"], marker="o", linewidth=2, color="#eb6834", label="< 16 px")
    ax.set_xticks(table["imgsz"])
    style(ax, "Objects that become tiny after resizing", "input size (longest side), px", "% of objects")
    ax.legend(frameon=False)
    save(fig, "eda_resize.png")


# ---------------------------------------------------------------- main

def main():
    cfg = load_config("configs/protocol.yaml")
    ann_root = CODE_DIR / cfg["data_dirs"]["annotations"]
    IMG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)

    parts = [load_split(ann_root / s / "annotations_coco.json", s) for s in ("train", "val")]
    objs = pd.concat([p[0] for p in parts], ignore_index=True)
    imgs = pd.concat([p[1] for p in parts], ignore_index=True)

    summary = {s: split_summary(objs[objs["split"] == s], imgs[imgs["split"] == s]) for s in ("train", "val")}
    cls_t = per_class_table(objs[objs["split"] == "train"])
    res_t = resize_table(objs)
    clip_t = per_clip_table(objs, imgs)
    summary["per_class_train"] = cls_t.to_dict(orient="index")
    summary["resize"] = res_t.to_dict(orient="records")

    with open(RES_DIR / "eda.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    clip_t.to_csv(RES_DIR / "eda_per_clip.csv", index=False)

    # Console report
    for s in ("train", "val"):
        print(f"\n===== {s} =====")
        print(json.dumps({k: v for k, v in summary[s].items() if k != "class_share_%"}, indent=1, default=float))
    print("\n===== per class (train) =====\n", cls_t.to_string())
    print("\n===== after resizing =====\n", res_t.to_string(index=False))
    cols = ["split", "video", "width", "height", "objects_per_frame", "median_px", "small_pct", "top_class"]
    srt = clip_t.sort_values("objects_per_frame")
    print("\n===== least / most dense clips =====\n", pd.concat([srt.head(5), srt.tail(5)])[cols].to_string(index=False))

    plot_classes(objs)
    plot_sizes(objs)
    plot_density(imgs)
    plot_resize(res_t)


if __name__ == "__main__":
    main()

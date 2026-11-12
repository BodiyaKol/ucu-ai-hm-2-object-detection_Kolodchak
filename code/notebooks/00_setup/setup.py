"""M0 setup kernel (runs on Kaggle CPU, no internet needed).

1. Saves environment info (python, torch, GPU, image counts) -> env.json
2. Builds a mini dataset (first N frames of first K clips per split) -> mini.zip
3. Packs the COCO annotation files of train/ and val/ -> annotations.zip (for local EDA)

Download results locally with:
    kaggle kernels output <user>/hw2-00-setup -p data/
"""
import json
import platform
import shutil
from pathlib import Path

import pandas as pd

N_CLIPS, N_FRAMES = 2, 20
WORK = Path("/kaggle/working")
TMP = Path("/tmp/hw2")


def find_root() -> Path:
    """Competition data is mounted somewhere under /kaggle/input."""
    for pattern in ("*/test_images.csv", "*/*/test_images.csv"):
        hits = sorted(Path("/kaggle/input").glob(pattern))
        if hits:
            return hits[0].parent
    raise FileNotFoundError("Competition data not found under /kaggle/input")


def env_info(root: Path) -> dict:
    """Python/torch/GPU versions and number of images per split."""
    info = {"python": platform.python_version(), "data_root": str(root)}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if info["cuda"] else None
    except ImportError:
        info["torch"] = None
    for split in ("train", "val", "test"):
        info[f"{split}_images"] = sum(1 for _ in (root / split / "images").rglob("*.jpg"))
    return info


def copy_split(src: Path, dst: Path) -> int:
    """Copy images + matching labels/ignore files for the first clips of a split."""
    clips = sorted(p.name for p in (src / "images").iterdir() if p.is_dir())[:N_CLIPS]
    copied = 0
    for clip in clips:
        for img in sorted((src / "images" / clip).glob("*.jpg"))[:N_FRAMES]:
            for sub, ext in (("images", ".jpg"), ("labels", ".txt"), ("ignore", ".txt")):
                f = src / sub / clip / (img.stem + ext)
                if f.exists():  # ignore/ files exist only for some frames
                    out = dst / sub / clip / f.name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, out)
            copied += 1
    return copied


def build_mini(root: Path, dst: Path) -> dict:
    """Mini copy of train/val/test with the original folder layout."""
    counts = {s: copy_split(root / s, dst / s) for s in ("train", "val")}

    test = pd.read_csv(root / "test_images.csv")
    keep = sorted(test["clip"].unique())[:N_CLIPS]
    test = test[test["clip"].isin(keep)].groupby("clip").head(N_FRAMES)
    for fn in test["file_name"]:
        out = dst / fn
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / fn, out)
    test.to_csv(dst / "test_images.csv", index=False)
    counts["test"] = len(test)
    return counts


def pack_annotations(root: Path, dst: Path) -> None:
    """Copy full-dataset annotation files (no images) for local EDA."""
    for split in ("train", "val"):
        (dst / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / split / "annotations_coco.json", dst / split)
    shutil.copy2(root / "test_images.csv", dst)


def main():
    root = find_root()

    info = env_info(root)
    print(json.dumps(info, indent=2))
    (WORK / "env.json").write_text(json.dumps(info, indent=2))

    counts = build_mini(root, TMP / "mini")
    print("mini frames:", counts)
    shutil.make_archive(str(WORK / "mini"), "zip", TMP / "mini")

    pack_annotations(root, TMP / "annotations")
    shutil.make_archive(str(WORK / "annotations"), "zip", TMP / "annotations")
    print("done")


if __name__ == "__main__":
    main()

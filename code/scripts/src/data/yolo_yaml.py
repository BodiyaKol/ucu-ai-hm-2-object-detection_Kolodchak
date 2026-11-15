"""Ultralytics dataset config (data.yaml), written at training time because paths differ per environment."""
from pathlib import Path

import yaml


def write_data_yaml(prepared_root: Path, data_root: Path, out_dir: Path,
                    class_names: list[str], val_step: int = 1) -> Path:
    """Write val.txt + data.yaml into out_dir (must be writable) and return the data.yaml path.

    train: prepared_root/train.txt (lines "./train/images/...", resolved by Ultralytics
           relative to the txt location) -> masked, subsampled training images.
    val:   absolute paths to the ORIGINAL val images (every `val_step`-th frame per clip);
           Ultralytics finds labels by replacing /images/ with /labels/ in the path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_paths = []
    for clip_dir in sorted((Path(data_root) / "val" / "images").iterdir()):
        val_paths += sorted(clip_dir.glob("*.jpg"))[::val_step]
    val_txt = out_dir / "val.txt"
    val_txt.write_text("\n".join(str(p.resolve()) for p in val_paths) + "\n")

    cfg = {"path": str(Path(prepared_root).resolve()),
           "train": "train.txt",
           "val": str(val_txt.resolve()),
           "names": {i: n for i, n in enumerate(class_names)}}
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return data_yaml

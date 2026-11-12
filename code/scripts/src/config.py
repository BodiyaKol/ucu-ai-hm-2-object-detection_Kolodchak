"""Shared helpers: config loading, seeding and path resolution.

Works the same locally (CPU, mini dataset) and inside a Kaggle kernel (GPU, full dataset).
"""
import os
import random
from pathlib import Path

import numpy as np
import yaml

# This file is code/scripts/src/config.py -> parents[2] is code/
CODE_DIR = Path(__file__).resolve().parents[2]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def is_kaggle() -> bool:
    """True when running inside a Kaggle kernel."""
    return KAGGLE_INPUT.exists()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base: nested dicts are merged, other values replaced."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path) -> dict:
    """Load a YAML config.

    If the file contains `base: other.yaml` (path relative to this file), the base
    is loaded first and this file's values override it (deep merge).
    """
    path = Path(path)
    if not path.is_absolute():
        path = CODE_DIR / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base = cfg.pop("base", None)
    if base is not None:
        cfg = _deep_merge(load_config(path.parent / base), cfg)
    return cfg


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_data_root(cfg: dict) -> Path:
    """Folder that contains train/, val/, test/ and test_images.csv."""
    if is_kaggle():
        # Competition data is mounted somewhere under /kaggle/input
        for pattern in ("*/test_images.csv", "*/*/test_images.csv"):
            hits = sorted(KAGGLE_INPUT.glob(pattern))
            if hits:
                return hits[0].parent
        raise FileNotFoundError("Competition data not found under /kaggle/input")

    root = CODE_DIR / cfg["data_dirs"][cfg["local_dataset"]]
    if not root.exists():
        raise FileNotFoundError(f"Local dataset not found: {root}")
    return root


def get_output_dir(name: str) -> Path:
    """Folder for weights/predictions of one experiment (gitignored locally)."""
    base = KAGGLE_WORKING if is_kaggle() else CODE_DIR / "outputs"
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out

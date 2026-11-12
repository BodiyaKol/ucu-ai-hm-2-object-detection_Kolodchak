"""Unit tests for src/config.py (run from code/: pytest scripts/tests -q)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add code/scripts to path
from src.config import load_config, set_seed


def test_protocol_loads():
    cfg = load_config("configs/protocol.yaml")
    assert cfg["num_classes"] == len(cfg["class_names"]) == 8


def test_base_merge(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb: 2\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\nb: 3\n")
    assert load_config(tmp_path / "child.yaml") == {"a": 1, "b": 3}


def test_seed_reproducible():
    set_seed(0)
    a = np.random.rand(3)
    set_seed(0)
    b = np.random.rand(3)
    assert np.allclose(a, b)

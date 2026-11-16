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

def test_nested_merge(tmp_path):
    (tmp_path / "base.yaml").write_text("model:\n  name: frcnn\n  lr: 0.01\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\nmodel:\n  lr: 0.02\n")
    assert load_config(tmp_path / "child.yaml") == {"model": {"name": "frcnn", "lr": 0.02}}


def test_experiment_configs_inherit_base():
    # every child config must keep the model fields of its base.yaml
    for child in ("faster_rcnn/small_anchors.yaml", "faster_rcnn/rpn_2000.yaml", "yolo/img1280.yaml"):
        cfg = load_config(f"configs/{child}")
        assert "name" in cfg["model"] or "weights" in cfg["model"], child
    # two-level chain: rpn_2000 -> small_anchors -> base
    cfg = load_config("configs/faster_rcnn/rpn_2000.yaml")
    assert cfg["model"]["anchor_sizes"] == [8, 16, 32, 64, 128] and cfg["model"]["rpn_post_nms_top_n"] == 2000
    assert cfg["train"]["max_minutes"] == 25
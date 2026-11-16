"""Late submissions (after deadline, for the report): test inference with the weights trained in hw2-03-torchvision.

Weights come from that kernel's output (attached via kernel_sources): /kaggle/input/.../<exp>/last.pt
Outputs: /kaggle/working/<exp>/{submission.csv, metrics.json}
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/BodiyaKol/ucu-ai-hm-2-object-detection_Kolodchak.git"
CLONE = Path("/tmp/repo")
EXPS = [("m3_frcnn_base", "configs/faster_rcnn/base.yaml"),
        ("m3_frcnn_small", "configs/faster_rcnn/small_anchors.yaml"),
        ("m3_frcnn_rpn2000", "configs/faster_rcnn/rpn_2000.yaml"),
        ("m4_retina_small", "configs/retinanet/base.yaml")]


def find(exp):
    for depth in range(1, 6):
        hits = sorted(Path("/kaggle/input").glob("*/" * depth + f"{exp}/last.pt"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{exp}/last.pt not found under /kaggle/input")


subprocess.run(["git", "clone", "--depth", "1", REPO, str(CLONE)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"], check=True)
for exp, cfg in EXPS:
    w = find(exp)
    print("weights:", w, flush=True)
    r = subprocess.run([sys.executable, "scripts/train_torchvision.py", "--exp", exp, "--config", cfg,
                        "--weights", str(w), "--skip-val", "--test"], cwd=CLONE / "code")
    print("RESULT", exp, "ok" if r.returncode == 0 else f"FAILED ({r.returncode})", flush=True)
shutil.rmtree(CLONE, ignore_errors=True)

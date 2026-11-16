"""Late submission (after deadline, for the report): RT-DETR val + test inference with weights from hw2-06-rtdetr."""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/BodiyaKol/ucu-ai-hm-2-object-detection_Kolodchak.git"
CLONE = Path("/tmp/repo")
EXP, CONFIG = "m6_rtdetr", "configs/rtdetr/base.yaml"

hits = []
for depth in range(1, 6):
    hits += sorted(Path("/kaggle/input").glob("*/" * depth + f"{EXP}/train/weights/last.pt"))
assert hits, "last.pt not found: attach hw2-06-rtdetr"
print("weights:", hits[0], flush=True)
subprocess.run(["git", "clone", "--depth", "1", REPO, str(CLONE)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.214", "pycocotools"], check=True)
subprocess.run([sys.executable, "scripts/train_yolo.py", "--exp", EXP, "--config", CONFIG, "--weights", str(hits[0])],
               cwd=CLONE / "code", check=True)
shutil.rmtree(CLONE, ignore_errors=True)

"""M7 final B (GPU T4): YOLO11m at 1024, time-capped training, then inference in a fresh process.

Outputs: /kaggle/working/<EXP>/{metrics.json, submission.csv, val_preds.json, train/weights/last.pt}
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/BodiyaKol/ucu-ai-hm-2-object-detection_Kolodchak.git"
CLONE = Path("/tmp/repo")
EXP, CONFIG = "m7_m1024", "configs/yolo/final_m1024.yaml"

subprocess.run(["git", "clone", "--depth", "1", REPO, str(CLONE)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.214", "pycocotools"], check=True)
code = CLONE / "code"
subprocess.run([sys.executable, "scripts/train_yolo.py", "--exp", EXP, "--config", CONFIG, "--train-only"],
               cwd=code, check=True)
subprocess.run([sys.executable, "scripts/train_yolo.py", "--exp", EXP, "--config", CONFIG,
                "--weights", f"/kaggle/working/{EXP}/train/weights/last.pt"], cwd=code, check=True)
shutil.rmtree(CLONE, ignore_errors=True)

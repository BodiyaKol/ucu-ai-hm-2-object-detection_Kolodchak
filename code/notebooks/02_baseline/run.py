"""M2.3 Kaggle kernel (GPU T4): YOLO baseline -> val COCO metrics + submission.csv.

Clones the public repo, installs Ultralytics, runs scripts/train_yolo.py on the full data.
Prepared training data comes from the hw2-01-prepare kernel (kernel_sources).
Outputs: /kaggle/working/m2_yolo_base/{metrics.json, submission.csv, val_preds.json, train/...}
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/BodiyaKol/ucu-ai-hm-2-object-detection_Kolodchak.git"
CLONE = Path("/tmp/repo")  # outside /kaggle/working so it is not saved as output

subprocess.run(["git", "clone", "--depth", "1", REPO, str(CLONE)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pycocotools"], check=True)
subprocess.run([sys.executable, "-m", "pip", "show", "ultralytics"], check=True)
subprocess.run([sys.executable, "-c", "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"],
               check=True)
subprocess.run([sys.executable, "scripts/train_yolo.py", "--exp", "m2_yolo_base"], cwd=CLONE / "code", check=True)
shutil.rmtree(CLONE, ignore_errors=True)

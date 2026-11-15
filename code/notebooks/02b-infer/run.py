"""M2.3 Kaggle kernel (GPU T4): inference only with the weights trained by hw2-02-baseline.

The first hw2-02-baseline run finished training (10 epochs) but was killed for out-of-memory during
inference; its /kaggle/working output (incl. last.pt) is attached here via kernel_sources.
Outputs: /kaggle/working/m2_yolo_base/{metrics.json, submission.csv, val_preds.json}
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/BodiyaKol/ucu-ai-hm-2-object-detection_Kolodchak.git"
CLONE = Path("/tmp/repo")
EXP = "m2_yolo_base"

hits = []
for pattern in (f"*/{EXP}/train/weights/last.pt", f"*/*/{EXP}/train/weights/last.pt",
                f"*/*/*/{EXP}/train/weights/last.pt"):
    hits += sorted(Path("/kaggle/input").glob(pattern))
assert hits, "last.pt not found: attach the hw2-02-baseline output"
print("weights:", hits[0])

subprocess.run(["git", "clone", "--depth", "1", REPO, str(CLONE)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.214", "pycocotools"], check=True)
subprocess.run([sys.executable, "scripts/train_yolo.py", "--exp", EXP, "--weights", str(hits[0])],
               cwd=CLONE / "code", check=True)
shutil.rmtree(CLONE, ignore_errors=True)

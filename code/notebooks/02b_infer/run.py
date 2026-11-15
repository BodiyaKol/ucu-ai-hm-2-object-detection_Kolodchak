"""M2.3 Kaggle kernel (GPU T4): inference only with the weights trained by hw2-02-baseline.

The first hw2-02-baseline run finished training (10 epochs) but was killed for out-of-memory during
inference. Kaggle does not allow attaching the output of a failed run, so last.pt was downloaded and
uploaded as the private dataset bohdankolodchak/hw2-weights (file m2_yolo_base_last.pt).
Outputs: /kaggle/working/m2_yolo_base/{metrics.json, submission.csv, val_preds.json}
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/BodiyaKol/ucu-ai-hm-2-object-detection_Kolodchak.git"
CLONE = Path("/tmp/repo")
EXP = "m2_yolo_base"
WEIGHTS = "m2_yolo_base_last.pt"

hits = []
for depth in range(1, 5):  # dataset mount path differs between Kaggle versions
    hits += sorted(Path("/kaggle/input").glob("*/" * depth + WEIGHTS))
assert hits, f"{WEIGHTS} not found: attach the bohdankolodchak/hw2-weights dataset"
print("weights:", hits[0])

subprocess.run(["git", "clone", "--depth", "1", REPO, str(CLONE)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.214", "pycocotools"], check=True)
subprocess.run([sys.executable, "scripts/train_yolo.py", "--exp", EXP, "--weights", str(hits[0])],
               cwd=CLONE / "code", check=True)
shutil.rmtree(CLONE, ignore_errors=True)

"""M3 + M4 Kaggle kernel (GPU T4): Faster R-CNN (3 settings) and RetinaNet, each time-capped, val-only.

Runs sequentially; a failed experiment does not stop the next one.
Outputs: /kaggle/working/<exp>/{metrics.json, val_preds.json, failures/*.jpg, last.pt}
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/BodiyaKol/ucu-ai-hm-2-object-detection_Kolodchak.git"
CLONE = Path("/tmp/repo")
EXPS = [("m3_frcnn_base", "configs/faster_rcnn/base.yaml"),
        ("m3_frcnn_small", "configs/faster_rcnn/small_anchors.yaml"),
        ("m4_retina_small", "configs/retinanet/base.yaml"),
        ("m3_frcnn_rpn2000", "configs/faster_rcnn/rpn_2000.yaml")]

subprocess.run(["git", "clone", "--depth", "1", REPO, str(CLONE)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"], check=True)
results = {}
for exp, cfg in EXPS:
    r = subprocess.run([sys.executable, "scripts/train_torchvision.py", "--exp", exp, "--config", cfg],
                       cwd=CLONE / "code")
    results[exp] = r.returncode
    print("RESULT", exp, "ok" if r.returncode == 0 else f"FAILED ({r.returncode})", flush=True)
print(results)
shutil.rmtree(CLONE, ignore_errors=True)

"""M1.5 Kaggle kernel (CPU): run prepare_data.py on the full competition data.

Clones the public repo, writes /kaggle/working/prepared (attached later by GPU kernels),
and checks val GT counts against the organisers' COCO file.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/BodiyaKol/ucu-ai-hm-2-object-detection_Kolodchak.git"
WORK = Path("/kaggle/working")
CLONE = Path("/tmp/repo")  # outside /kaggle/working so it is not saved as output

subprocess.run(["git", "clone", "--depth", "1", REPO, str(CLONE)], check=True)
subprocess.run([sys.executable, str(CLONE / "code/scripts/prepare_data.py"), "--out", str(WORK / "prepared")],
               check=True)

# Integrity check on full val: our GT vs organisers' GT (object and unique-crowd counts)
ours = json.load(open(WORK / "prepared/annotations/val_coco.json"))
ref_path = next(Path("/kaggle/input").glob("**/val/annotations_coco.json"))
ref = json.load(open(ref_path))
count = lambda c, crowd: sum(a.get("iscrowd", 0) == crowd for a in c["annotations"])
print("val objects  ours / organisers:", count(ours, 0), "/", count(ref, 0))
print("val crowd    ours / organisers:", count(ours, 1), "/", count(ref, 1))
print("val images   ours / organisers:", len(ours["images"]), "/", len(ref["images"]))
shutil.rmtree(CLONE, ignore_errors=True)

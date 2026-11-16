"""torch Dataset for torchvision detectors (M3 Faster R-CNN, M4 RetinaNet).

Reads the COCO json written by prepare_data.py (train: masked images in prepared/train).
Target format expected by torchvision: {"boxes": float xyxy (N, 4), "labels": int64 (N,)},
labels 1-8 (0 is reserved for background in torchvision).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.convert import coco_to_xyxy


def load_image(path) -> torch.Tensor:
    """JPEG -> float RGB tensor (3, H, W) in [0, 1]."""
    im = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(im).permute(2, 0, 1).float() / 255.0


class VisDroneDetection(Dataset):
    def __init__(self, coco_json, images_dir, train: bool = True):
        self.images_dir, self.train = Path(images_dir), train
        coco = json.load(open(coco_json))
        anns = defaultdict(list)
        for a in coco["annotations"]:
            if not a.get("iscrowd", 0):  # ignore regions are already masked in the training images
                anns[a["image_id"]].append(a)
        self.items = []
        for im in coco["images"]:
            aa = anns[im["id"]]
            boxes = coco_to_xyxy([a["bbox"] for a in aa]) if aa else np.zeros((0, 4))
            labels = np.array([a["category_id"] for a in aa], dtype=np.int64)
            ok = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)  # torchvision needs w, h > 0
            self.items.append((im["file_name"], boxes[ok].astype(np.float32), labels[ok]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, boxes, labels = self.items[idx]
        img = load_image(self.images_dir / name)
        boxes, labels = torch.from_numpy(boxes.copy()), torch.from_numpy(labels.copy())
        if self.train and random.random() < 0.5:  # horizontal flip (the only augmentation)
            img, w = img.flip(-1), img.shape[-1]
            if len(boxes):
                boxes = torch.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1)
        return img, {"boxes": boxes.reshape(-1, 4), "labels": labels}


def collate_fn(batch):
    """Images have different sizes: keep lists (torchvision detection models accept lists)."""
    return tuple(zip(*batch))

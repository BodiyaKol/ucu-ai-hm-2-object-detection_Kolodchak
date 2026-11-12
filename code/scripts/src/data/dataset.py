"""torch Dataset for torchvision detectors (M3 Faster R-CNN, M4 RetinaNet)."""
from torch.utils.data import Dataset


class VisDroneDetection(Dataset):
    """Returns (image_tensor, target) where target = {"boxes": xyxy FloatTensor, "labels": Int64Tensor}.

    TODO: read COCO json produced by prepare_data.py, apply transforms/augmentations.
    """

    def __init__(self, coco_json, images_dir, transforms=None):
        self.transforms = transforms
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


def collate_fn(batch):
    return tuple(zip(*batch))

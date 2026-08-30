"""
PyTorch Dataset + transforms for the processed Cats vs Dogs folders.

Expected layout (produced by data/prepare_data.py):
    data/processed/train/cat/*.jpg
    data/processed/train/dog/*.jpg
    data/processed/val/cat/*.jpg
    ...

Transforms:
  - SimpleCNN        : uses [-1, 1] normalization (NORM_MEAN/STD = 0.5)
  - MobileNetV3Small : uses ImageNet normalization (IMAGENET_MEAN/STD)
    Use `get_transforms(model_name)` to get the right pair.
"""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224
CLASS_TO_IDX = {"cat": 0, "dog": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Simple normalization (training SimpleCNN from scratch, no pretrained backbone).
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

# ImageNet normalization required by pretrained MobileNetV3 backbone.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Default transforms (SimpleCNN, [-1,1] normalization) ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# --- ImageNet transforms (MobileNetV3Small) ---
imagenet_train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

imagenet_eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_transforms(model_name: str):
    """
    Return (train_transform, eval_transform) appropriate for the given model.

    Args:
        model_name: 'SimpleCNN' or 'MobileNetV3Small'
    Returns:
        Tuple[transform, transform]
    """
    if model_name == "MobileNetV3Small":
        return imagenet_train_transform, imagenet_eval_transform
    # Default: SimpleCNN and any future scratch-trained models
    return train_transform, eval_transform


class CatsDogsDataset(Dataset):
    def __init__(self, root_dir, split: str, transform=None):
        self.split_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            cls_dir = self.split_dir / cls_name
            if not cls_dir.exists():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((f, cls_idx))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.split_dir}. "
                f"Did you run `python data/prepare_data.py` first?"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.transform:
                im = self.transform(im)
        return im, label

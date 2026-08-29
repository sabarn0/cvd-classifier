"""
PyTorch Dataset + transforms for the processed Cats vs Dogs folders.

Expected layout (produced by data/prepare_data.py):
    data/processed/train/cat/*.jpg
    data/processed/train/dog/*.jpg
    data/processed/val/cat/*.jpg
    ...
"""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224
CLASS_TO_IDX = {"cat": 0, "dog": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Simple normalization (training from scratch, no pretrained backbone).
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

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

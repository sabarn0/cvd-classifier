"""
Data acquisition & preprocessing for the Cats vs Dogs classification task.

- Downloads the dataset via kagglehub
- Locates the two class folders (cat / dog) regardless of the exact
  folder-naming convention used by the dataset
- Filters out corrupt / unreadable images (a known issue with this dataset)
- Resizes to 224x224 RGB
- Splits into train/val/test (80/10/10 by default), stratified per class
- Writes the result to data/processed/{train,val,test}/{cat,dog}/*.jpg

Run:
    python data/prepare_data.py --output-dir data/processed
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMG_SIZE = (224, 224)
CLASS_NAMES = ["cat", "dog"]


def download_dataset() -> Path:
    """Downloads the dataset via kagglehub and returns the local path."""
    import kagglehub

    path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
    print(f"Path to dataset files: {path}")
    return Path(path)


def find_class_dirs(root: Path) -> dict:
    """
    Walk the downloaded dataset and find the directory containing images
    for each class ('cat' / 'dog'), regardless of exact naming
    (e.g. 'Cat', 'PetImages/Cat', 'cats', ...).
    """
    found = {}
    for class_name in CLASS_NAMES:
        candidates = []
        for p in root.rglob("*"):
            if p.is_dir() and class_name in p.name.lower():
                # count image files directly inside this dir
                n_images = sum(
                    1 for f in p.iterdir()
                    if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                if n_images > 0:
                    candidates.append((n_images, p))
        if not candidates:
            raise FileNotFoundError(
                f"Could not locate an image folder for class '{class_name}' under {root}. "
                f"Inspect the downloaded dataset structure and adjust find_class_dirs()."
            )
        # pick the directory with the most images matching this class name
        candidates.sort(key=lambda x: x[0], reverse=True)
        found[class_name] = candidates[0][1]
    return found


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def process_and_save(src: Path, dst: Path) -> bool:
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im = im.resize(IMG_SIZE)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="JPEG", quality=90)
        return True
    except (UnidentifiedImageError, OSError):
        return False


def split_files(files, train_frac, val_frac, seed):
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Cats vs Dogs dataset")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap on images per class, useful for a fast local smoke test.",
    )
    args = parser.parse_args()

    raw_root = download_dataset()
    class_dirs = find_class_dirs(raw_root)
    print("Found class directories:")
    for cls, d in class_dirs.items():
        print(f"  {cls}: {d}")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    summary = {}
    for cls, class_dir in class_dirs.items():
        all_files = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"[{cls}] found {len(all_files)} raw files, validating...")

        valid_files = [f for f in all_files if is_readable_image(f)]
        skipped = len(all_files) - len(valid_files)
        print(f"[{cls}] {len(valid_files)} valid, {skipped} corrupt/skipped")

        if args.max_per_class:
            random.Random(args.seed).shuffle(valid_files)
            valid_files = valid_files[: args.max_per_class]

        splits = split_files(valid_files, args.train_frac, args.val_frac, args.seed)

        summary[cls] = {}
        for split_name, files in splits.items():
            n_ok = 0
            for f in files:
                dst = output_dir / split_name / cls / f.name
                if process_and_save(f, dst):
                    n_ok += 1
            summary[cls][split_name] = n_ok
            print(f"[{cls}/{split_name}] wrote {n_ok} images -> {output_dir / split_name / cls}")

    print("\n=== Summary ===")
    for cls, splits in summary.items():
        print(cls, splits)
    print(f"\nProcessed dataset available at: {output_dir.resolve()}")
    print("Next: `dvc add data/processed` (or Git-LFS) to version this data.")


if __name__ == "__main__":
    main()

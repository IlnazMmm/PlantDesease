"""
Split dataset in ImageFolder format into train/valid folders.
Usage:
  python split_dataset.py --src "./New Plant Diseases Dataset(Augmented)/valid" --dst "./New Plant Diseases Dataset(Augmented)" --train_ratio 0.8
"""

import argparse
from pathlib import Path
import shutil
import random

def split_dataset(src, dst, train_ratio=0.8):
    src = Path(src)
    dst = Path(dst)

    train_dir = dst / "train"
    valid_dir = dst / "valid"

    # создаём папки
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    classes = [d for d in src.iterdir() if d.is_dir()]

    for cls in classes:
        images = list(cls.glob("*"))
        random.shuffle(images)

        n_train = int(len(images) * train_ratio)
        train_imgs = images[:n_train]
        valid_imgs = images[n_train:]

        # создаём поддиректории для каждого класса
        (train_dir / cls.name).mkdir(parents=True, exist_ok=True)
        (valid_dir / cls.name).mkdir(parents=True, exist_ok=True)

        # копируем
        for img in train_imgs:
            shutil.copy(img, train_dir / cls.name / img.name)
        for img in valid_imgs:
            shutil.copy(img, valid_dir / cls.name / img.name)

        print(f"{cls.name}: {len(train_imgs)} train, {len(valid_imgs)} valid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Путь до исходной папки (например valid/)")
    parser.add_argument("--dst", required=True, help="Куда сохранить train/ и valid/")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Доля train (default=0.8)")
    args = parser.parse_args()

    split_dataset(args.src, args.dst, args.train_ratio)

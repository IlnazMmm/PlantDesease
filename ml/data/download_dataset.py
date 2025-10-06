"""
Script to download PlantVillage dataset (instructions).
PlantVillage isn't a single direct download in this script; typically use:
- Kaggle dataset: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
You can download manually via kaggle CLI.

This script is a helper to organize files after download.
"""
import argparse
from pathlib import Path
import shutil

def organize(src_dir, dest_dir):
    src = Path(src_dir)
    dest = Path(dest_dir)
    if not src.exists():
        raise RuntimeError("Source not found. Download dataset manually (Kaggle) and point --src")
    dest.mkdir(parents=True, exist_ok=True)
    # assume src has subdirs per class (ImageFolder-style)
    for p in src.iterdir():
        if p.is_dir():
            shutil.copytree(p, dest / p.name, dirs_exist_ok=True)
    print("Organized dataset to", dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dest", default="ml/data/plantvillage")
    args = parser.parse_args()
    organize(args.src, args.dest)

"""Utilities for preparing datasets and dataloaders for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class DataLoaders:
    """Container for train and validation dataloaders."""

    train: DataLoader
    valid: DataLoader
    class_names: Tuple[str, ...]
    train_size: int
    valid_size: int


def _build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    valid_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transforms, valid_transforms


def _split_dataset(dataset: Dataset, valid_fraction: float) -> Tuple[Dataset, Dataset]:
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1 (exclusive).")

    valid_size = int(len(dataset) * valid_fraction)
    train_size = len(dataset) - valid_size
    if valid_size == 0 or train_size == 0:
        raise ValueError("Validation split is empty; adjust valid_fraction or dataset size.")
    train_subset, valid_subset = random_split(dataset, [train_size, valid_size])
    return train_subset, valid_subset


def create_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    valid_fraction: float | None = None,
) -> DataLoaders:
    """Create train/validation dataloaders for an ImageFolder dataset."""

    train_dir = data_root / "train"
    valid_dir = data_root / "valid"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    train_tfms, valid_tfms = _build_transforms(image_size)

    if valid_dir.exists():
        train_dataset = datasets.ImageFolder(train_dir, transform=train_tfms)
        valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_tfms)
        class_names = tuple(train_dataset.classes)
    else:
        full_dataset = datasets.ImageFolder(train_dir, transform=train_tfms)
        class_names = tuple(full_dataset.classes)
        if valid_fraction is None:
            raise FileNotFoundError(
                "Validation directory not found and valid_fraction is None. "
                "Provide --valid-fraction to create a split from the training data."
            )
        train_subset, valid_subset = _split_dataset(full_dataset, valid_fraction)
        # apply validation transforms lazily to avoid reloading data
        valid_subset.dataset = datasets.ImageFolder(train_dir, transform=valid_tfms)
        train_dataset = train_subset
        valid_dataset = valid_subset

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_size = len(train_dataset)
    valid_size = len(valid_dataset)

    return DataLoaders(
        train=train_loader,
        valid=valid_loader,
        class_names=class_names,
        train_size=train_size,
        valid_size=valid_size,
    )


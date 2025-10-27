"""Command-line training entrypoint for the plant disease classifier."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_utils import DataLoaders, create_dataloaders
from model_utils import build_efficientnet_b0, trainable_parameters
from trainer import Trainer


@dataclass
class TrainConfig:
    data_dir: Path
    epochs: int
    out_path: Path
    device: str
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_workers: int = 4
    image_size: int = 224
    use_pretrained: bool = True
    freeze_backbone: bool = False
    valid_fraction: Optional[float] = None
    scheduler_patience: int = 2


def _prepare_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Choose --device cpu instead.")
    return torch.device(device_name)


def _create_scheduler(optimizer: optim.Optimizer, patience: int) -> Optional[ReduceLROnPlateau]:
    if patience <= 0:
        return None
    return ReduceLROnPlateau(optimizer, mode="max", patience=patience, factor=0.5)


def _save_checkpoint(out_path: Path, state_dict: dict, class_names: tuple[str, ...], history: list[dict], config: TrainConfig) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {key: value.cpu() for key, value in state_dict.items()}
    torch.save(
        {
            "model_state": cpu_state,
            "classes": class_names,
            "history": history,
            "config": asdict(config),
        },
        out_path,
    )
    print(f"Saved best model to {out_path}")


def train_model(config: TrainConfig) -> None:
    device = _prepare_device(config.device)
    print(f"Using device: {device}")

    dataloaders: DataLoaders = create_dataloaders(
        data_root=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        valid_fraction=config.valid_fraction,
    )

    model = build_efficientnet_b0(
        num_classes=len(dataloaders.class_names),
        pretrained=config.use_pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        trainable_parameters(model),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = _create_scheduler(optimizer, config.scheduler_patience)

    trainer = Trainer(model, criterion, optimizer, device, scheduler=scheduler)
    best_state, history = trainer.fit(
        epochs=config.epochs,
        train_loader=dataloaders.train,
        valid_loader=dataloaders.valid,
    )

    _save_checkpoint(
        out_path=config.out_path,
        state_dict=best_state,
        class_names=dataloaders.class_names,
        history=history.to_serializable(),
        config=config,
    )


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train an EfficientNet-based plant disease classifier.")
    parser.add_argument("--data", required=True, help="Path to dataset root containing train/ and valid/ folders")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--out", required=True, help="Output path for the trained model checkpoint")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run training on")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze EfficientNet feature extractor")
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=None,
        help="If provided, create a validation split from train/ when valid/ directory is absent",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=2,
        help="Patience for ReduceLROnPlateau scheduler; set 0 to disable",
    )

    args = parser.parse_args()
    return TrainConfig(
        data_dir=Path(args.data),
        epochs=args.epochs,
        out_path=Path(args.out),
        device=args.device,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        valid_fraction=args.valid_fraction,
        scheduler_patience=args.scheduler_patience,
    )


if __name__ == "__main__":
    train_model(_parse_args())


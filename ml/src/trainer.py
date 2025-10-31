"""Training utilities for supervised classification."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass
class TrainingHistory:
    train: List[EpochMetrics] = field(default_factory=list)
    valid: List[EpochMetrics] = field(default_factory=list)

    def append(self, train_metrics: EpochMetrics, valid_metrics: EpochMetrics) -> None:
        self.train.append(train_metrics)
        self.valid.append(valid_metrics)

    def to_serializable(self) -> List[Dict[str, float]]:
        history = []
        for idx, (train_metrics, valid_metrics) in enumerate(zip(self.train, self.valid), start=1):
            history.append(
                {
                    "epoch": idx,
                    "train_loss": train_metrics.loss,
                    "train_accuracy": train_metrics.accuracy,
                    "valid_loss": valid_metrics.loss,
                    "valid_accuracy": valid_metrics.accuracy,
                }
            )
        return history


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: Optional[ReduceLROnPlateau] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def _run_epoch(self, dataloader: DataLoader, train: bool) -> EpochMetrics:
        if train:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total
        return EpochMetrics(loss=avg_loss, accuracy=accuracy)

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> tuple[Dict[str, torch.Tensor], TrainingHistory]:
        history = TrainingHistory()
        best_state = copy.deepcopy(self.model.state_dict())
        best_accuracy = 0.0

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            valid_metrics = self._run_epoch(valid_loader, train=False)
            history.append(train_metrics, valid_metrics)

            if self.scheduler is not None:
                self.scheduler.step(valid_metrics.accuracy)

            if valid_metrics.accuracy > best_accuracy:
                best_accuracy = valid_metrics.accuracy
                best_state = copy.deepcopy(self.model.state_dict())

            print(
                f"Epoch {epoch}/{epochs} "
                f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.3f} "
                f"val_loss={valid_metrics.loss:.4f} val_acc={valid_metrics.accuracy:.3f}"
            )

        return best_state, history


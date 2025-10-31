"""Model builders used by the training script."""

from __future__ import annotations

from typing import Iterable

from torch import nn
from torchvision import models


def _get_efficientnet_weights(pretrained: bool):
    try:  # torchvision>=0.13
        weights_enum = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return weights_enum
    except AttributeError:  # fallback for older versions
        return pretrained


def build_efficientnet_b0(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Construct an EfficientNet-B0 classifier adapted for the dataset."""

    weights = _get_efficientnet_weights(pretrained)
    try:
        model = models.efficientnet_b0(weights=weights)
    except TypeError:  # torchvision<0.13
        model = models.efficientnet_b0(pretrained=bool(weights))

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Return parameters that require gradients."""

    return (p for p in model.parameters() if p.requires_grad)


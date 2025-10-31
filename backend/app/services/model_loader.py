"""Utilities for loading and caching the inference model bundle."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image
from src.infer import load_model, predict
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder

DEFAULT_WEIGHTS_PATH = "./ml/models/model_v5.pth" #Path(__file__).resolve().parents[3] / "ml" / "models" / "model_v3.pth"


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    labels: Tuple[str, ...]


# convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
# resnet architecture
class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        # self.conv5 = ConvBlock(256, 256, pool=True)
        # self.conv6 = ConvBlock(256, 512, pool=True)
        # self.conv7 = ConvBlock(512, 512, pool=True)

        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, x):  # x is the loaded batch
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        # out = self.conv5(out)
        # out = self.conv6(out)
        # out = self.conv7(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

@lru_cache(maxsize=2)
def _load_bundle(weights_path: str) -> ModelBundle:
    data = torch.load(weights_path, map_location='cpu')
    classes = data["classes"]
    model = CNN_NeuralNet(3, len(data["classes"]))
    # model.classifier[1] = torch.nn.Linear(512, len(classes))
    # print(data["model_state"])
    print(data["classes"])
    model.load_state_dict(data["model_state"])
    model.to('cpu').eval()
    return ModelBundle(model=model, labels=tuple(classes))
    # model, classes = load_model(weights_path)
    # return ModelBundle(model=model, labels=tuple(classes))


class ModelPredictor:
    """Wraps the training package helpers with caching and simple helpers."""

    def __init__(self, weights_path: Path | str | None = None) -> None:
        self._weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH

    @property
    def weights_path(self) -> Path:
        return self._weights_path

    @property
    def bundle(self) -> ModelBundle:
        return _load_bundle(str(self._weights_path))

    def predict(self, image_path: str) -> Dict[str, float]:
        bundle = self.bundle
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        im = Image.open(image_path).convert("RGB")
        x = transform(im).unsqueeze(0).to('cpu')
        with torch.no_grad():
            out = self.bundle.model(x)
            probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return {"class":  self.bundle.labels[idx], "confidence": float(probs[idx]), "probs": probs.tolist()}
        # return predict(bundle.model, bundle.labels, image_path)

    def model(self) -> Any:
        return self.bundle.model

    def labels(self) -> Tuple[str, ...]:
        return self.bundle.labels

    def label_index(self, label: str) -> int | None:
        try:
            return self.labels().index(label)
        except ValueError:
            return None

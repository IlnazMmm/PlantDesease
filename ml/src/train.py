"""
Minimal PyTorch training script for ImageFolder dataset.
Usage:
  python train.py --data ./ml/data/plantvillage_resized --epochs 3 --out ./ml/models/model_v1.pth
"""

import argparse
from pathlib import Path
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

def train(data_dir, epochs, out_path, device="cpu"):
    device = torch.device(device)

    # Трансформации для train и valid
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    valid_tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Поддиректории
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"

    # Датасеты
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    valid_ds = datasets.ImageFolder(valid_dir, transform=valid_tfms)

    classes = train_ds.classes
    print("Classes:", classes)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=4)

    # Модель
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(device)

    # Оптимизатор и лосс
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Обучение
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(train_ds)

        # Валидация
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = out.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total

        print(f"Epoch {epoch+1}/{epochs} "
              f"loss={epoch_loss:.4f} "
              f"val_acc={acc:.3f} "
              f"time={(time.time()-start):.1f}s")

    # Сохранение
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "classes": classes}, out_path)
    print("Saved model to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)  # путь до "New Plant Diseases Dataset(Augmented)"
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.data, args.epochs, args.out, device=args.device)

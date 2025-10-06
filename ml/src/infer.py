"""
Simple inference script to load saved model and predict on a single image.
"""
import torch
from torchvision import transforms, models
from PIL import Image
import argparse
from pathlib import Path
import numpy as np

def load_model(path, device="cpu"):
    data = torch.load(path, map_location=device)
    classes = data["classes"]
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(data["model_state"])
    model.to(device).eval()
    return model, classes

def predict(model, classes, image_path, device="cpu"):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    im = Image.open(image_path).convert("RGB")
    x = transform(im).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return {"class": classes[idx], "confidence": float(probs[idx]), "probs": probs.tolist()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--img", required=True)
    args = parser.parse_args()
    model, classes = load_model(args.model)
    r = predict(model, classes, args.img)
    print(r)

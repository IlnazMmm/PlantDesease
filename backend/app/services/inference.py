"""
Simple inference service:
 - loads a lightweight model (or dummy)
 - returns dict: plant, disease, confidence, description, treatment, gradcam_image (PIL)
This is a minimal example. Replace model loading with real PyTorch/ONNX inference.
"""
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from src.infer import load_model, predict

# Dummy label map (should correspond to trained model)
LABELS = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Healthy"
]

DESCRIPTIONS = {
    "Tomato___Early_blight": "Грибковое заболевание, проявляется коричневыми пятнами...",
    "Tomato___Late_blight": "Поздняя гниль томатов, быстро распространяется при влажности...",
    "Tomato___Septoria_leaf_spot": "Бактериальное пятно листьев...",
    "Tomato___Healthy": "Растение здорово."
}

TREATMENTS = {
    "Tomato___Early_blight": "Применять фунгициды, удалять поражённые листья.",
    "Tomato___Late_blight": "Карантин, обработка специализированными препаратами.",
    "Tomato___Septoria_leaf_spot": "Соблюдать севооборот, обработка медью.",
    "Tomato___Healthy": "Нет рекомендаций."
}

@lru_cache(maxsize=1)
def _load_model_bundle(path_to_weights: str = "./ml/models/model_v3.pth") -> Tuple[object, Tuple[str, ...]]:
    """Load and cache the model/label bundle so repeated predictions are cheap."""
    model, classes = load_model(path_to_weights)
    return model, tuple(classes)


def _real_predict(path_to_image: str, path_to_weights: str = "./ml/models/model_v3.pth") -> Dict[str, float]:
    model, classes = _load_model_bundle(path_to_weights)
    return predict(model, classes, path_to_image)

def _fallback_overlay(image_path: str, label: str) -> Image.Image:
    """Return a simple translucent red overlay if Grad-CAM fails."""
    im = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (255, 0, 0, 80))
    combined = Image.alpha_composite(im, overlay).convert("RGB")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()
    ImageDraw.Draw(combined).text((10, 10), label, fill=(255, 255, 255), font=font)
    return combined


def _make_gradcam_overlay(image_path: str, label: str) -> Image.Image:
    """Generate a Grad-CAM heatmap overlay for the given prediction label."""

    try:
        model, classes = _load_model_bundle()
        model.eval()
        device = next(model.parameters()).device

        try:
            target_index = classes.index(label)
        except ValueError:
            target_index = None

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        activations: Dict[str, torch.Tensor] = {}
        gradients: Dict[str, torch.Tensor] = {}

        target_layer = model.features[-1]

        def forward_hook(_, __, output):
            activations["value"] = output.detach()

        def backward_hook(_, __, output):
            gradients["value"] = output[0].detach()

        handle_forward = target_layer.register_forward_hook(forward_hook)
        if hasattr(target_layer, "register_full_backward_hook"):
            handle_backward = target_layer.register_full_backward_hook(backward_hook)
        else:
            handle_backward = target_layer.register_backward_hook(backward_hook)

        try:
            with torch.enable_grad():
                scores = model(input_tensor)
            if target_index is None:
                target_index = int(scores.argmax(dim=1).item())

            score = scores[:, target_index]
            if hasattr(model, "zero_grad"):
                try:
                    model.zero_grad(set_to_none=True)
                except TypeError:
                    model.zero_grad()
            score.backward()

            if "value" not in activations or "value" not in gradients:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

            activations_value = activations["value"]
            gradients_value = gradients["value"]

            weights = gradients_value.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * activations_value).sum(dim=1)).squeeze(0)

            cam_np = cam.cpu().numpy()
            if not np.any(cam_np):
                raise RuntimeError("Grad-CAM produced an empty heatmap")

            cam_np -= cam_np.min()
            cam_np /= cam_np.max() + 1e-8

            heatmap = Image.fromarray(np.uint8(cam_np * 255), mode="L")
            heatmap = heatmap.resize(image.size, resample=Image.BILINEAR)

            red_channel = heatmap
            zero_channel = Image.new("L", heatmap.size, 0)
            heatmap_rgba = Image.merge(
                "RGBA", (red_channel, zero_channel, zero_channel, heatmap)
            )

            combined = Image.alpha_composite(image.convert("RGBA"), heatmap_rgba).convert(
                "RGB"
            )

        finally:
            handle_forward.remove()
            handle_backward.remove()

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=18)
        except Exception:
            font = ImageFont.load_default()
        ImageDraw.Draw(combined).text((10, 10), label, fill=(255, 255, 255), font=font)
        return combined

    except Exception:
        return _fallback_overlay(image_path, label)

def predict_image(image_path: str):
    raw_prediction = _real_predict(image_path)
    label = raw_prediction.get("class", "Unknown")
    confidence = float(raw_prediction.get("confidence", 0.0))

    if "___" in label:
        plant, disease = label.split("___", maxsplit=1)
    else:
        plant, disease = "Unknown", label

    gradcam = _make_gradcam_overlay(image_path, label)

    return {
        "plant": plant,
        "disease": disease,
        "confidence": confidence,
        "description": DESCRIPTIONS.get(label, ""),
        "treatment": TREATMENTS.get(label, ""),
        "gradcam_image": gradcam,
    }

"""Grad-CAM generation utilities."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision import transforms

from .model_loader import ModelPredictor

logger = logging.getLogger(__name__)


class GradCamGenerator:
    """Creates Grad-CAM overlays for the configured model."""

    def __init__(self, predictor: ModelPredictor) -> None:
        self._predictor = predictor

    def generate(self, image_path: str, label: str) -> Image.Image:
        try:
            model = self._predictor.model()
            model.eval()
            device = next(model.parameters()).device

            target_index = self._predictor.label_index(label)

            preprocess = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            image = Image.open(image_path).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0).to(device)
            input_tensor.requires_grad_(True)

            activations: Dict[str, torch.Tensor] = {}
            gradients: Dict[str, torch.Tensor] = {}

            target_layer = self._resolve_target_layer(model)

            def forward_hook(_, __, output):
                activations["value"] = output.detach()

            def backward_hook(_, __, grad_output):
                gradients["value"] = grad_output[0].detach()

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

                heatmap = self._build_heatmap(cam, image.size)
                combined = Image.alpha_composite(image.convert("RGBA"), heatmap).convert("RGB")

            finally:
                handle_forward.remove()
                handle_backward.remove()

            self._annotate_image(combined, label)
            return combined
        except Exception as exc:  # pragma: no cover - Grad-CAM should never break predictions
            logger.warning("Falling back to translucent overlay for Grad-CAM: %s", exc)
            return self._fallback_overlay(image_path, label)

    def _build_heatmap(self, cam: torch.Tensor, image_size: tuple[int, int]) -> Image.Image:
        cam_np = cam.detach().cpu().numpy()
        if not np.any(cam_np):
            raise RuntimeError("Grad-CAM produced an empty heatmap")

        cam_np -= cam_np.min()
        cam_np /= cam_np.max() + 1e-8

        threshold = float(np.quantile(cam_np, 0.6))
        cam_np = np.clip(cam_np - threshold, 0.0, None)
        if np.max(cam_np) > 0:
            cam_np /= np.max(cam_np)

        heatmap = Image.fromarray(np.uint8(cam_np * 255), mode="L")
        heatmap = heatmap.resize(image_size, resample=Image.BILINEAR)

        zero_channel = Image.new("L", heatmap.size, 0)
        green_channel = Image.fromarray(np.uint8(np.array(heatmap) * 0.6), mode="L")
        return Image.merge(
            "RGBA",
            (
                heatmap,
                green_channel,
                zero_channel,
                heatmap,
            ),
        )

    def _resolve_target_layer(self, model: torch.nn.Module) -> torch.nn.Module:
        cached = getattr(model, "_gradcam_target_layer", None)
        if cached is not None:
            return cached

        conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        if not conv_layers:
            raise RuntimeError("Model has no convolutional layers for Grad-CAM")

        spatial_layers = [module for module in conv_layers if module.kernel_size != (1, 1)]
        target_layer = spatial_layers[-1] if spatial_layers else conv_layers[-1]

        model._gradcam_target_layer = target_layer  # type: ignore[attr-defined]
        return target_layer

    def _fallback_overlay(self, image_path: str, label: str) -> Image.Image:
        image = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (255, 0, 0, 80))
        combined = Image.alpha_composite(image, overlay).convert("RGB")
        self._annotate_image(combined, label)
        return combined

    def _annotate_image(self, image: Image.Image, label: str) -> None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=18)
        except Exception:  # pragma: no cover - fallback to default font when DejaVu недоступен
            font = ImageFont.load_default()
        ImageDraw.Draw(image).text((10, 10), label, fill=(255, 255, 255), font=font)

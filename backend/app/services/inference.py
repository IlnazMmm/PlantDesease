"""
Simple inference service:
 - loads a lightweight model (or dummy)
 - returns dict: plant, disease, confidence, description, treatment, gradcam_image (PIL)
This is a minimal example. Replace model loading with real PyTorch/ONNX inference.
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

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

def _dummy_predict(image_path: str):
    # random pick to simulate model
    label = random.choices(LABELS, weights=[0.2,0.1,0.1,0.6], k=1)[0]
    conf = round(random.uniform(0.6, 0.99), 3)
    return label, conf

def _real_predict(path_to_image: str, path_to_weights: str = './ml/models/model_v3.pth'):
    model, classes = load_model(path_to_weights)
    return predict(model, classes, path_to_image)

def _make_gradcam_overlay(image_path: str, label: str):
    # create a fake heatmap overlay using blur & red ellipse
    im = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (255,0,0,0))
    draw = ImageDraw.Draw(overlay)
    w,h = im.size
    # draw an ellipse in center as "hotspot"
    draw.ellipse((w*0.2, h*0.2, w*0.8, h*0.8), fill=(255,0,0,80))
    combined = Image.alpha_composite(im, overlay).convert("RGB")
    # add small label text
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except:
        font = ImageFont.load_default()
    draw2 = ImageDraw.Draw(combined)
    draw2.text((10,10), label, fill=(255,255,255), font=font)
    return combined

def predict_image(image_path: str):
    label, conf = _real_predict(image_path)
    plant = label.split("___")[0] if "___" in label else "Unknown"
    disease = label.split("___")[1] if "___" in label else label
    gradcam = _make_gradcam_overlay(image_path, disease)
    return {
        "plant": plant,
        "disease": disease,
        "confidence": conf,
        "description": DESCRIPTIONS.get(label, ""),
        "treatment": TREATMENTS.get(label, ""),
        "gradcam_image": gradcam
    }

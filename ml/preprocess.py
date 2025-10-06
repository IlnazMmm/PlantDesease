"""
Simple preprocessing pipeline: resize images to 224x224, remove tiny images.
"""
from PIL import Image
from pathlib import Path
import argparse

def resize_folder(src, dst, size=(224,224)):
    src = Path(src)
    dst = Path(dst)
    for imgp in src.rglob("*"):
        if not imgp.is_file():
            continue
        try:
            im = Image.open(imgp).convert("RGB")
            im = im.resize(size)

            # вычисляем относительный путь (сохраняем структуру поддиректорий)
            rel_path = imgp.relative_to(src)
            out_path = dst / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Картинка {rel_path} обработано")

            im.save(out_path, quality=90)
        except Exception as e:
            print("skip", imgp, e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    resize_folder(args.src, args.dst)

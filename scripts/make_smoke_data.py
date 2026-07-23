"""
Generate a tiny SYNTHETIC classification dataset to smoke-test the pipeline
(train_cnn -> evaluate -> predict) WITHOUT downloading the real CarDD data.

Each fake "class" gets a distinct base colour + shape so the model can actually
learn something in a couple of epochs. This only proves the code runs; it says
nothing about real accuracy. Delete data/classification afterwards and run the
real download_data.py + prepare_classification.py for the actual project.

Usage:
    .venv\\Scripts\\python scripts\\make_smoke_data.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config

CLASSES = {
    "smoke_dent": (200, 60, 60),
    "smoke_scratch": (60, 160, 200),
    "smoke_crack": (90, 200, 90),
}
PER_SPLIT = {"train": 24, "val": 8, "test": 8}
SIZE = 96


def make_image(base_rgb, shape_idx):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (SIZE, SIZE), base_rgb)
    draw = ImageDraw.Draw(img)
    # add noise so images are not identical
    for _ in range(120):
        x, y = random.randint(0, SIZE - 1), random.randint(0, SIZE - 1)
        jitter = tuple(max(0, min(255, c + random.randint(-40, 40))) for c in base_rgb)
        draw.point((x, y), fill=jitter)
    # class-specific shape
    other = tuple(255 - c for c in base_rgb)
    if shape_idx == 0:
        draw.ellipse([20, 20, 76, 76], outline=other, width=4)
    elif shape_idx == 1:
        draw.line([10, 10, 86, 86], fill=other, width=5)
    else:
        draw.rectangle([25, 25, 71, 71], outline=other, width=4)
    return img


def main():
    random.seed(config.SEED)
    root = config.CLASSIFICATION_DIR
    print(f"Writing synthetic smoke-test data to {root}")
    for split, n in PER_SPLIT.items():
        for shape_idx, (cls, rgb) in enumerate(CLASSES.items()):
            out = root / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                make_image(rgb, shape_idx).save(out / f"{split}_{cls}_{i}.jpg")
    total = sum(PER_SPLIT.values()) * len(CLASSES)
    print(f"Done. {total} synthetic images across {len(CLASSES)} classes.")


if __name__ == "__main__":
    main()

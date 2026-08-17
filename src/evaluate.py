"""
Step 4 - Evaluate the trained CNN on the held-out test set.

Produces the numbers the assignment rubric asks for: accuracy, precision,
recall and F1 (per class + macro/weighted), plus a confusion-matrix image.

Usage:
    .venv\\Scripts\\python src\\evaluate.py
    .venv\\Scripts\\python src\\evaluate.py --model models\\cnn_damage_classifier.pth
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import utils
from dataset import build_dataloaders
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.MODEL_PATH))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    import torch

    device = utils.get_device()
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", config.IMG_SIZE)
    backbone = ckpt.get("backbone", config.BACKBONE)
    print(f"[model] {args.model}  backbone={backbone}  classes={class_names}")

    net = build_model(len(class_names), backbone=backbone, pretrained=False).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    loaders, _ = build_dataloaders(img_size, config.BATCH_SIZE)
    loader = loaders[args.split]

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = net(images).argmax(1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    # Same reporting path as every other algorithm, so src/compare.py can read all
    # of them uniformly. Writes results/cnn/ (or results/cnn_val/ for --split val).
    name = "cnn" if args.split == "test" else f"cnn_{args.split}"
    metrics = utils.save_run_report(
        name, y_true, y_pred, class_names,
        title=f"CNN ({backbone}) - {args.split} set")
    metrics["backbone"] = backbone
    utils.save_json(metrics, config.RESULTS_DIR / name / "metrics.json")

    print(f"\nSaved report, metrics and confusion matrix to results/{name}/")
    print("Compare against the other algorithms with:  python src/compare.py")


if __name__ == "__main__":
    main()

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
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, precision_recall_fscore_support)

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

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    print(f"\n=== {args.split.upper()} SET RESULTS ===")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision (macro): {p:.4f}")
    print(f"Recall    (macro): {r:.4f}")
    print(f"F1        (macro): {f1:.4f}\n")
    print(report)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "classification_report.txt").write_text(
        f"Accuracy: {acc:.4f}\nPrecision(macro): {p:.4f}\n"
        f"Recall(macro): {r:.4f}\nF1(macro): {f1:.4f}\n\n{report}",
        encoding="utf-8")
    utils.save_json(
        {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1},
        config.RESULTS_DIR / "cnn_metrics.json")
    utils.plot_confusion_matrix(
        cm, class_names, config.RESULTS_DIR / "confusion_matrix.png",
        title=f"CNN ({backbone}) - {args.split} set")
    print("\nSaved report, metrics JSON and confusion matrix to results/")


if __name__ == "__main__":
    main()

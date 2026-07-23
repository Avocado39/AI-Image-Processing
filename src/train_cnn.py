"""
Step 3 - Train the CNN damage-type classifier (Member A).

Transfer learning with a torchvision backbone (default ResNet18). Saves the best
model (by validation macro-F1) plus a training-history plot and JSON log.

Usage:
    .venv\\Scripts\\python src\\train_cnn.py
    .venv\\Scripts\\python src\\train_cnn.py --epochs 20 --backbone mobilenet_v2
    .venv\\Scripts\\python src\\train_cnn.py --no-pretrained --epochs 2   # quick test
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import utils
from dataset import build_dataloaders
from model import build_model


def run_epoch(net, loader, criterion, device, optimizer=None):
    """One pass over loader. If optimizer is given -> train, else -> eval."""
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    training = optimizer is not None
    net.train() if training else net.eval()

    total_loss, y_true, y_pred = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(outputs.argmax(1).cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return avg_loss, acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--backbone", default=config.BACKBONE)
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch")
    ap.add_argument("--freeze-backbone", action="store_true", help="train only the head")
    ap.add_argument("--out", default=str(config.MODEL_PATH))
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"[device] {device}")

    loaders, class_names = build_dataloaders(args.img_size, args.batch_size)
    print(f"[data] classes = {class_names}")
    for split in ("train", "val", "test"):
        print(f"[data] {split}: {len(loaders[split].dataset)} images")

    net = build_model(len(class_names), backbone=args.backbone,
                      pretrained=not args.no_pretrained,
                      freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=config.WEIGHT_DECAY)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(net, loaders["train"], criterion, device, optimizer)
        va_loss, va_acc, va_f1 = run_epoch(net, loaders["val"], criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        flag = ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "state_dict": net.state_dict(),
                "class_names": class_names,
                "backbone": args.backbone,
                "img_size": args.img_size,
            }, args.out)
            flag = "  <- best (saved)"
        print(f"epoch {epoch:2d}/{args.epochs}  "
              f"train_loss={tr_loss:.3f} acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.3f} acc={va_acc:.3f} f1={va_f1:.3f}  "
              f"({time.time() - t0:.1f}s){flag}")

    utils.save_json(history, config.RESULTS_DIR / "cnn_training_history.json")
    utils.plot_history(history, config.RESULTS_DIR / "cnn_training_history.png")
    print(f"\nBest val macro-F1: {best_f1:.3f}")
    print(f"Model saved to: {args.out}")
    print("Next run:  python src\\evaluate.py")


if __name__ == "__main__":
    main()

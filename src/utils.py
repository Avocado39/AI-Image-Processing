"""Small shared helpers: reproducibility, device selection, saving, plotting."""
import json
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Make runs reproducible across random / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return CUDA device if a GPU is available, otherwise CPU."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] {path}")


def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix") -> None:
    """Save a confusion-matrix heatmap to out_path."""
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt
    import seaborn as sns

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.4 * len(class_names) + 2, 1.2 * len(class_names) + 2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_history(history, out_path) -> None:
    """Plot train/val loss and val accuracy over epochs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train loss")
    ax1.plot(epochs, history["val_loss"], label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.set_title("Loss")
    ax2.plot(epochs, history["val_acc"], label="val accuracy", color="green")
    ax2.plot(epochs, history["val_f1"], label="val macro-F1", color="orange")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("score"); ax2.legend(); ax2.set_title("Validation metrics")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")

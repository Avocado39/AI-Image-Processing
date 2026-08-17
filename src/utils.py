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


def save_run_report(name, y_true, y_pred, class_names, title=None) -> dict:
    """Score one classification run and write its results to results/<name>/.

    Every classifier calls this, so the three-way comparison reads one shape of
    file instead of three. Writes metrics.json, classification_report.txt and
    confusion_matrix.png, and returns the metrics dict.

    `name` labels the run and becomes the folder, e.g. "cnn" or "svm_hog".
    """
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, precision_recall_fscore_support)
    import config

    out_dir = config.RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels)
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=labels)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   digits=4, zero_division=0, labels=labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "run": name,
        "task": "classification",
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "per_class": {
            cls: {"precision": float(per_p[i]), "recall": float(per_r[i]),
                  "f1": float(per_f1[i]), "support": int(support[i])}
            for i, cls in enumerate(class_names)
        },
    }

    save_json(metrics, out_dir / "metrics.json")
    (out_dir / "classification_report.txt").write_text(
        f"{name}\n\nAccuracy: {acc:.4f}\nPrecision(macro): {p:.4f}\n"
        f"Recall(macro): {r:.4f}\nF1(macro): {f1:.4f}\n\n{report}",
        encoding="utf-8")
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png",
                          title=title or f"{name} - test set")

    print(f"\n=== {name} ===")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision (macro): {p:.4f}")
    print(f"Recall    (macro): {r:.4f}")
    print(f"F1        (macro): {f1:.4f}\n")
    print(report)
    return metrics


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

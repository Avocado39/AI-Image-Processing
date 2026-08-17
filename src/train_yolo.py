"""
YOLO object detection - locate AND classify each damage region.

Solves a strictly harder problem than train_cnn.py and train_svm.py, which
classify a crop they are handed. YOLO draws a bounding box around every damaged
area and labels it, so it trains directly on the ORIGINAL CarDD YOLO labels and
needs no cropping - which is why data/raw/ must not be deleted.

Because of that, its scores are NOT directly comparable: detection is measured
with mAP and IoU, since a prediction only counts if the box overlaps the true
region. Do not put mAP in the same column as the classifiers' accuracy.

Setup:
    .venv\\Scripts\\python -m pip install -r requirements-yolo.txt

The Ultralytics trainer needs a data.yaml describing the dataset. CarDD's
Kaggle download already contains one - find it with the path printed by
download_data.py (look for 'data.yaml'). If the paths inside it are relative,
either edit them to absolute paths or pass the folder as `data=`.

Run:
    .venv\\Scripts\\python src\\train_yolo.py --data "C:\\path\\to\\data.yaml" --epochs 50
    .venv\\Scripts\\python src\\train_yolo.py --predict car.jpg   # after training
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import utils


def resolve_data_yaml(path):
    """Return a data.yaml whose split paths exist on this machine.

    CarDD's shipped data.yaml hardcodes Kaggle paths (/kaggle/input/...), which do
    not exist anywhere else. Rather than making every user hand-edit it, rewrite a
    local copy next to the original with paths pointing at the actual folders.
    """
    import yaml

    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent

    if all((root / split).exists() or Path(spec.get(split, "")).exists()
           for split in ("train", "val")):
        if Path(spec.get("train", "")).exists():
            return str(path)                       # already valid, use as-is

    fixed = dict(spec)
    for split in ("train", "val", "test"):
        candidate = root / split / "images"
        if candidate.exists():
            fixed[split] = str(candidate.resolve())

    out = path.with_name("data_local.yaml")
    out.write_text(yaml.safe_dump(fixed, sort_keys=False), encoding="utf-8")
    print(f"[data] rewrote split paths for this machine -> {out}")
    return str(out)


def train(data_yaml, epochs, img_size, weights, fraction=1.0):
    from ultralytics import YOLO

    data_yaml = resolve_data_yaml(data_yaml)
    model = YOLO(weights)  # e.g. yolov8n.pt (nano) - small & fast to start
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        fraction=fraction,
        project=str(config.RESULTS_DIR / "yolo"),
        name="cardd",
        exist_ok=True,
        seed=config.SEED,
    )

    # Metrics come from an explicit validation pass, not from train()'s return
    # value: model.val() reliably returns a DetMetrics object with a stable .box
    # interface, whereas what train() hands back varies between versions.
    # Ultralytics also writes its own plots + confusion matrix into
    # results/yolo/cardd/.
    # project/name are passed explicitly, or Ultralytics drops the validation plots
    # into a stray runs/detect/val/ at the current working directory instead of
    # alongside the training output.
    save_metrics(model.val(
        data=data_yaml,
        imgsz=img_size,
        project=str(config.RESULTS_DIR / "yolo"),
        name="val",
        exist_ok=True,
    ))


def save_metrics(results):
    """Extract Ultralytics' box metrics into results/yolo/metrics.json."""
    def get(attr, key):
        v = getattr(getattr(results, "box", object()), attr, None)
        if v is None:
            v = getattr(results, "results_dict", {}).get(key)
        return float(v) if v is not None else None

    metrics = {
        "run": "yolo",
        "task": "detection",          # NOT comparable with classification accuracy
        "precision": get("mp", "metrics/precision(B)"),
        "recall": get("mr", "metrics/recall(B)"),
        "mAP50": get("map50", "metrics/mAP50(B)"),
        "mAP50_95": get("map", "metrics/mAP50-95(B)"),
        "note": "Detection is scored with mAP/IoU because a prediction only counts "
                "if the box overlaps the true region. Report separately from the "
                "classifiers rather than as another row in the accuracy table.",
    }
    utils.save_json(metrics, config.RESULTS_DIR / "yolo" / "metrics.json")

    print("\n=== yolo (detection) ===")
    for k in ("precision", "recall", "mAP50", "mAP50_95"):
        v = metrics[k]
        print(f"{k:12s}: {v:.4f}" if v is not None else f"{k:12s}: n/a")


def predict(image_path, weights):
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.predict(image_path, save=True)
    for r in results:
        print(r.boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to CarDD data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of the training set to use; <1 for a fast "
                         "smoke run on CPU")
    ap.add_argument("--predict", help="run detection on a single image (needs trained weights)")
    args = ap.parse_args()

    if args.predict:
        predict(args.predict, args.weights)
    elif args.data:
        train(args.data, args.epochs, args.img_size, args.weights, args.fraction)
    else:
        ap.error("Pass --data <data.yaml> to train, or --predict <image> to detect.")


if __name__ == "__main__":
    main()

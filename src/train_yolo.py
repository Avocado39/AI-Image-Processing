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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import utils


def train(data_yaml, epochs, img_size, weights):
    from ultralytics import YOLO

    model = YOLO(weights)  # e.g. yolov8n.pt (nano) - small & fast to start
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        project=str(config.RESULTS_DIR / "yolo"),
        name="cardd",
        seed=config.SEED,
    )
    # Ultralytics writes its own plots + confusion matrix into results/yolo/cardd/.
    # Pull the headline numbers out into the same metrics.json shape the classifiers
    # use, so src/compare.py can pick them up - marked task="detection" so nothing
    # accidentally lists mAP alongside classification accuracy.
    save_metrics(results)


def save_metrics(results):
    """Extract Ultralytics' box metrics into results/yolo/metrics.json."""
    import sys

    box = getattr(getattr(results, "box", None), "__dict__", None)
    if box is None and not hasattr(results, "results_dict"):
        print("Could not read metrics off the results object - check the run "
              "folder manually.", file=sys.stderr)
        return

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
    ap.add_argument("--predict", help="run detection on a single image (needs trained weights)")
    args = ap.parse_args()

    if args.predict:
        predict(args.predict, args.weights)
    elif args.data:
        train(args.data, args.epochs, args.img_size, args.weights)
    else:
        ap.error("Pass --data <data.yaml> to train, or --predict <image> to detect.")


if __name__ == "__main__":
    main()

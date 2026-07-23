"""
MEMBER C - YOLO object detection (locate AND classify each damage region).

Unlike Members A & B (which classify a whole crop), YOLO draws a bounding box
around every damaged area and labels it. It trains directly on the ORIGINAL
CarDD YOLO labels - no cropping needed.

Setup:
    .venv\\Scripts\\python -m pip install -r requirements-yolo.txt

The Ultralytics trainer needs a data.yaml describing the dataset. CarDD's
Kaggle download already contains one - find it with the path printed by
download_data.py (look for 'data.yaml'). If the paths inside it are relative,
either edit them to absolute paths or pass the folder as `data=`.

Run:
    .venv\\Scripts\\python src\\members\\member_c_yolo.py --data "C:\\path\\to\\data.yaml" --epochs 50
    .venv\\Scripts\\python src\\members\\member_c_yolo.py --predict car.jpg   # after training
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: F401  (available for Member C to reuse paths/seed)


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
    # Ultralytics automatically reports precision, recall, mAP@50, mAP@50-95
    # and writes plots + a confusion matrix into results/yolo/cardd/.
    print(results)
    # TODO (Member C): report mAP/precision/recall and compare detection vs the
    # classification approach of Members A & B in the documentation.


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

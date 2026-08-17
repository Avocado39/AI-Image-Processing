"""
Export the trained YOLO detector to ONNX for the web demo.

Exported WITHOUT built-in NMS, so the output is the raw prediction tensor
(1, 4 + nc, N). The web app decodes it and runs non-maximum suppression in numpy,
which keeps the deployment free of torch and of onnxruntime-extensions.

Also writes yolo_meta.json with the class names in the model's OWN order. That
order comes from CarDD's data.yaml (dent, scratch, crack, ...) and is NOT the
alphabetical order the classification model uses - mixing them up would mislabel
every box.

Usage:
    .venv\\Scripts\\python src\\export_yolo.py
    .venv\\Scripts\\python src\\export_yolo.py --weights path\\to\\best.pt --img-size 416
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

DEFAULT_WEIGHTS = config.RESULTS_DIR / "yolo" / "cardd" / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--img-size", type=int, default=416,
                    help="must match how the model was trained")
    ap.add_argument("--out", default=str(config.MODELS_DIR),
                    help="where to write yolo_damage_detector.onnx + yolo_meta.json")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"{weights} not found. Train first with src/train_yolo.py.")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names                       # {0: 'dent', 1: 'scratch', ...}
    class_names = [names[i] for i in sorted(names)]
    print(f"[model] {weights.name}  classes={class_names}")

    exported = Path(model.export(format="onnx", imgsz=args.img_size, opset=17))
    print(f"[export] {exported}  ({exported.stat().st_size / 1e6:.1f} MB)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "yolo_damage_detector.onnx"
    shutil.move(str(exported), onnx_path)

    (out_dir / "yolo_meta.json").write_text(json.dumps({
        "class_names": class_names,
        "img_size": args.img_size,
        "source": f"ultralytics YOLO, {weights.name}",
        "preprocessing": "letterbox to img_size, RGB, scaled to [0,1] - NO ImageNet "
                         "normalisation (unlike the classifier)",
        "output": "raw (1, 4+nc, N) tensor; the app decodes it and runs NMS",
    }, indent=2), encoding="utf-8")

    print(f"[saved] {onnx_path}")
    print(f"[saved] {out_dir / 'yolo_meta.json'}")
    print("\nCopy both into car-damage-demo/models/ and redeploy.")


if __name__ == "__main__":
    main()

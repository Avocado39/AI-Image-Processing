"""
Step 5 - Predict the damage type of a single image (demo / inference).

Usage:
    .venv\\Scripts\\python src\\predict.py --image path\\to\\car.jpg
    .venv\\Scripts\\python src\\predict.py --image car.jpg --topk 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import utils
from dataset import build_transforms
from model import build_model


def predict_image(image_path, model_path=str(config.MODEL_PATH), topk=3):
    import torch
    from PIL import Image

    device = utils.get_device()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", config.IMG_SIZE)

    net = build_model(len(class_names), backbone=ckpt.get("backbone", config.BACKBONE),
                      pretrained=False).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    _, eval_tf = build_transforms(img_size)
    img = Image.open(image_path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0].cpu()

    k = min(topk, len(class_names))
    top_p, top_i = probs.topk(k)
    return [(class_names[i], float(p)) for p, i in zip(top_p, top_i)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default=str(config.MODEL_PATH))
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    results = predict_image(args.image, args.model, args.topk)
    print(f"\nPrediction for {args.image}:")
    for rank, (name, prob) in enumerate(results, 1):
        print(f"  {rank}. {name:15s} {prob * 100:5.1f}%")
    print(f"\n=> Most likely damage type: {results[0][0]}")


if __name__ == "__main__":
    main()

"""
Step 2 - Turn CarDD's YOLO detection labels into a classification dataset.

CarDD ships as an object-detection dataset: each image has a .txt label file
with one line per damaged region:  `class_id  cx  cy  w  h`  (all normalised).
For a *classification* model (Member A's CNN) we want one image = one damage
type. So we crop every labelled box out of its image and drop the crop into a
folder named after its damage class:

    data/classification/
        train/  dent/ scratch/ crack/ glass_shatter/ lamp_broken/ tire_flat/
        val/    ...
        test/   ...

If the raw dataset already has train/val/test folders we respect them; otherwise
we make a reproducible random split at the image level (so crops from one image
never leak across splits).

Usage:
    .venv\\Scripts\\python src\\prepare_classification.py
    .venv\\Scripts\\python src\\prepare_classification.py --raw "C:\\path\\to\\dataset"
"""
import argparse
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------------------------------------------------------------------------
# Locating the raw dataset and its class names
# ---------------------------------------------------------------------------
def resolve_raw_path(cli_raw: str | None) -> Path:
    if cli_raw:
        return Path(cli_raw)
    if config.RAW_PATH_FILE.exists():
        return Path(config.RAW_PATH_FILE.read_text(encoding="utf-8").strip())
    raise SystemExit(
        "Could not find the dataset. Run 'python src\\download_data.py' first, "
        "or pass --raw <path>."
    )


def load_class_names(raw_root: Path) -> list[str]:
    """Prefer the dataset's own data.yaml so ids always match the labels."""
    try:
        import yaml
    except ImportError:
        yaml = None

    if yaml is not None:
        for yml in raw_root.rglob("*.yaml"):
            try:
                data = yaml.safe_load(yml.read_text(encoding="utf-8"))
            except Exception:
                continue
            names = data.get("names") if isinstance(data, dict) else None
            if isinstance(names, dict):  # {0: 'dent', 1: 'scratch', ...}
                names = [names[k] for k in sorted(names)]
            if names:
                print(f"[classes] loaded {len(names)} names from {yml.name}: {names}")
                return [str(n).strip().replace(" ", "_") for n in names]

    print(f"[classes] no data.yaml found - using config.CLASS_NAMES: {config.CLASS_NAMES}")
    return list(config.CLASS_NAMES)


# ---------------------------------------------------------------------------
# Pairing images with their YOLO label files
# ---------------------------------------------------------------------------
def build_image_index(raw_root: Path) -> dict[str, Path]:
    """Map every image's stem -> path (for fallback lookup)."""
    index: dict[str, Path] = {}
    for p in raw_root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            index.setdefault(p.stem, p)
    return index


def find_image_for_label(label: Path, stem_index: dict[str, Path]) -> Path | None:
    # 1) standard YOLO layout: .../labels/x.txt  ->  .../images/x.jpg
    s = str(label)
    if os.sep + "labels" + os.sep in s:
        base = s.replace(os.sep + "labels" + os.sep, os.sep + "images" + os.sep)
        for ext in IMAGE_EXTS:
            cand = Path(base).with_suffix(ext)
            if cand.exists():
                return cand
    # 2) image sitting next to the label
    for ext in IMAGE_EXTS:
        cand = label.with_suffix(ext)
        if cand.exists():
            return cand
    # 3) anywhere in the tree with the same stem
    return stem_index.get(label.stem)


def detect_split(path: Path) -> str | None:
    parts = {p.lower() for p in path.parts}
    if "test" in parts:
        return "test"
    if parts & {"val", "valid", "validation"}:
        return "val"
    if "train" in parts:
        return "train"
    return None


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------
def parse_boxes(label_file: Path):
    """Yield (class_id, cx, cy, w, h) for each valid line."""
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        yield cid, cx, cy, w, h


def crop_and_save(image_path: Path, boxes, class_names, out_dir: Path, saved_counts):
    from PIL import Image

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[warn] cannot open {image_path}: {e}")
        return 0
    W, H = img.size
    n_saved = 0
    for i, (cid, cx, cy, w, h) in enumerate(boxes):
        if cid < 0 or cid >= len(class_names):
            continue
        # normalised -> pixel box, with padding
        bw, bh = w * W, h * H
        px, py = cx * W, cy * H
        pad_w, pad_h = bw * config.BOX_PADDING, bh * config.BOX_PADDING
        x1 = max(0, int(px - bw / 2 - pad_w))
        y1 = max(0, int(py - bh / 2 - pad_h))
        x2 = min(W, int(px + bw / 2 + pad_w))
        y2 = min(H, int(py + bh / 2 + pad_h))
        if (x2 - x1) < config.MIN_BOX_SIZE or (y2 - y1) < config.MIN_BOX_SIZE:
            continue
        cls = class_names[cid]
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        crop = img.crop((x1, y1, x2, y2))
        crop.save(cls_dir / f"{image_path.stem}_{i}.jpg", quality=92)
        saved_counts[cls] = saved_counts.get(cls, 0) + 1
        n_saved += 1
    return n_saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="Path to the downloaded dataset (default: data/raw_path.txt)")
    args = ap.parse_args()

    rng = random.Random(config.SEED)
    raw_root = resolve_raw_path(args.raw)
    if not raw_root.exists():
        raise SystemExit(f"Raw path does not exist: {raw_root}")
    print(f"[raw] {raw_root}")

    class_names = load_class_names(raw_root)
    stem_index = build_image_index(raw_root)
    print(f"[index] found {len(stem_index)} images")

    # Gather (image_path, boxes, split_hint) records at the image level.
    records = []
    label_files = [p for p in raw_root.rglob("*.txt")
                   if p.name.lower() not in {"classes.txt", "readme.txt"}]
    print(f"[labels] scanning {len(label_files)} .txt files")
    missing = 0
    for lf in label_files:
        boxes = list(parse_boxes(lf))
        if not boxes:
            continue
        img = find_image_for_label(lf, stem_index)
        if img is None:
            missing += 1
            continue
        records.append((img, boxes, detect_split(lf)))
    if missing:
        print(f"[labels] {missing} label files had no matching image (skipped)")
    if not records:
        raise SystemExit("No image/label pairs found - check the dataset layout printed by download_data.py")

    # Decide the split for each record.
    hinted = [r for r in records if r[2] is not None]
    if len(hinted) >= 0.5 * len(records):
        print("[split] using the dataset's own train/val/test folders")
        by_split = {"train": [], "val": [], "test": []}
        for img, boxes, hint in records:
            by_split[hint or "train"].append((img, boxes))
        # Guarantee val/test exist even if the raw set only had some of them.
        for need in ("val", "test"):
            if not by_split[need] and by_split["train"]:
                rng.shuffle(by_split["train"])
                k = max(1, int(0.15 * len(by_split["train"])))
                by_split[need] = by_split["train"][:k]
                by_split["train"] = by_split["train"][k:]
    else:
        print("[split] no split folders detected - making a reproducible random split")
        imgs = [(img, boxes) for img, boxes, _ in records]
        rng.shuffle(imgs)
        n = len(imgs)
        n_test = int(config.TEST_SPLIT * n)
        n_val = int(config.VAL_SPLIT * n)
        by_split = {
            "test": imgs[:n_test],
            "val": imgs[n_test:n_test + n_val],
            "train": imgs[n_test + n_val:],
        }

    # Fresh output folder.
    if config.CLASSIFICATION_DIR.exists():
        print(f"[clean] removing existing {config.CLASSIFICATION_DIR}")
        shutil.rmtree(config.CLASSIFICATION_DIR)

    grand_total = 0
    for split, items in by_split.items():
        out_dir = config.CLASSIFICATION_DIR / split
        counts: dict[str, int] = {}
        for img, boxes in items:
            grand_total += crop_and_save(img, boxes, class_names, out_dir, counts)
        pretty = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"
        print(f"[{split:5s}] images={len(items):4d}  crops -> {pretty}")

    print(f"\nDone. {grand_total} crops written to {config.CLASSIFICATION_DIR}")
    print("Next run:  python src\\train_cnn.py")


if __name__ == "__main__":
    main()

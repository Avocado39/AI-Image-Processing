"""
Build the zips the Colab notebooks expect in Google Drive.

    cardd_src.zip             source tree, ~45 KB  - rebuild whenever src/ changes
    cardd_classification.zip  cropped dataset, ~360 MB - only if the data changes

Run this before every Colab session that follows a change under `src/`. A stale
`cardd_src.zip` is a nasty failure: the notebook unpacks it, reports success, and
then dies on a missing file - which is what happens if the bundle predates a rename.
So this script also checks the archive contains every script the notebooks invoke.

Usage:
    .venv\\Scripts\\python scripts\\make_colab_bundle.py           # source only (fast)
    .venv\\Scripts\\python scripts\\make_colab_bundle.py --data    # both
"""
import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ITEMS = ["src", "docs", "requirements.txt", "requirements-yolo.txt",
                "README.md"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}

# Scripts the notebooks call with `!python ...`. Missing any of these means a run
# fails partway through rather than up front, so check before uploading.
REQUIRED = [
    "src/config.py", "src/utils.py",
    "src/train_cnn.py", "src/evaluate.py",        # train_colab.ipynb
    "src/model.py", "src/dataset.py",
    "src/download_data.py",                       # train_yolo_colab.ipynb
    "src/train_yolo.py", "src/export_yolo.py",
]


def build_source(out: Path) -> None:
    written = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in SOURCE_ITEMS:
            p = PROJECT_ROOT / item
            if not p.exists():
                print(f"  (skipping missing {item})")
                continue
            files = [p] if p.is_file() else [f for f in sorted(p.rglob("*"))
                                             if f.is_file()]
            for f in files:
                if SKIP_PARTS & set(f.parts):
                    continue
                rel = f.relative_to(PROJECT_ROOT)
                zf.write(f, rel)
                written.append(rel.as_posix())

    missing = [r for r in REQUIRED if r not in written]
    if missing:
        out.unlink(missing_ok=True)
        sys.exit("Bundle is incomplete - the notebooks would fail partway through.\n"
                 "Missing:\n  " + "\n  ".join(missing))

    print(f"{out.name}: {len(written)} files, {out.stat().st_size / 1e3:.0f} KB "
          f"(all {len(REQUIRED)} notebook entry points present)")


def build_data(out: Path) -> None:
    src = PROJECT_ROOT / "data" / "classification"
    if not src.exists():
        sys.exit(f"{src} not found - run src/prepare_classification.py first.")

    # JPEGs are already compressed, so STORED is far faster at the same size.
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, Path("classification") / f.relative_to(src))
                n += 1
    print(f"{out.name}: {n} files, {out.stat().st_size / 1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true",
                    help="also rebuild the ~360 MB dataset zip (slow)")
    args = ap.parse_args()

    build_source(PROJECT_ROOT / "cardd_src.zip")
    if args.data:
        build_data(PROJECT_ROOT / "cardd_classification.zip")

    print("\nUpload to Google Drive at MyDrive/cardd/, REPLACING what is there.")
    print("Then in Colab: Runtime > Restart session before re-running, or the old "
          "files stay unpacked.")


if __name__ == "__main__":
    main()

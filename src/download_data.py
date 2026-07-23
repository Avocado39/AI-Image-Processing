"""
Step 1 - Download the CarDD dataset from Kaggle.

Usage:
    .venv\\Scripts\\python src\\download_data.py

Requires Kaggle credentials. Create an API token at
https://www.kaggle.com/settings  ->  "Create New API Token" and save the file to
C:\\Users\\<you>\\.kaggle\\kaggle.json  (or set KAGGLE_KEY in the environment).

This downloads the dataset zip via Kaggle's public REST API (works with both the
classic username+key and the newer single-token format), extracts it under
data/raw/ and records the location in data/raw_path.txt for the next step.
"""
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path

import config

DATASET_SLUG = config.KAGGLE_DATASET
API_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"


def get_token() -> str:
    """Read the Kaggle key from env or ~/.kaggle/kaggle.json."""
    key = os.environ.get("KAGGLE_KEY")
    if key:
        return key
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        if data.get("key"):
            return data["key"]
    raise SystemExit(
        "No Kaggle credentials found. Create an API token at "
        "https://www.kaggle.com/settings and save kaggle.json to "
        f"{Path.home() / '.kaggle'}, or set KAGGLE_KEY."
    )


def download_zip(token: str, dest_zip: Path, max_retries: int = 12) -> None:
    """Stream the dataset zip with resume + retry.

    The Kaggle endpoint redirects to a Google Cloud Storage signed URL which
    supports HTTP Range requests, so a dropped connection can be resumed from
    the number of bytes already on disk instead of starting over.
    """
    import time

    import requests

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_SLUG} ...")
    base_headers = {"Authorization": f"Bearer {token}"}
    resume = dest_zip.stat().st_size if dest_zip.exists() else 0
    if resume:
        print(f"  resuming from {resume/1e6:.0f} MB already on disk")
    total = None
    next_mark = 0
    attempt = 0

    while True:
        headers = dict(base_headers)
        if resume:
            headers["Range"] = f"bytes={resume}-"
        try:
            with requests.get(API_URL, headers=headers, stream=True,
                              timeout=(30, 120), allow_redirects=True) as r:
                if r.status_code == 416:                      # range beyond EOF = already complete
                    cr = r.headers.get("Content-Range", "")   # bytes */total
                    if "/" in cr:
                        total = int(cr.split("/")[-1])
                    print("  file already fully downloaded")
                    break
                if r.status_code == 206:                      # partial content - good
                    mode = "ab"
                    cr = r.headers.get("Content-Range", "")   # bytes start-end/total
                    if "/" in cr:
                        total = int(cr.split("/")[-1])
                elif r.status_code == 200:                    # server sent whole file
                    mode = "wb"
                    resume = 0
                    total = int(r.headers.get("Content-Length", 0)) or total
                else:
                    r.raise_for_status()
                    mode = "ab" if resume else "wb"

                with open(dest_zip, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                        if not chunk:
                            continue
                        f.write(chunk)
                        resume += len(chunk)
                        if total:
                            pct = resume * 100 // total
                            if pct >= next_mark:
                                print(f"  {pct:3d}%  ({resume/1e6:.0f}/{total/1e6:.0f} MB)")
                                next_mark = pct + 5

            if total is None or resume >= total:
                break  # finished (or total unknown but stream ended cleanly)

        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                OSError) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            resume = dest_zip.stat().st_size if dest_zip.exists() else 0
            wait = min(30, 3 * attempt)
            print(f"  [retry {attempt}/{max_retries}] connection dropped at "
                  f"{resume/1e6:.0f} MB ({e.__class__.__name__}); waiting {wait}s...")
            time.sleep(wait)

    final = dest_zip.stat().st_size
    if total and final != total:
        raise SystemExit(f"Download incomplete: {final} of {total} bytes. Re-run to resume.")
    print(f"Saved zip: {dest_zip}  ({final/1e6:.0f} MB)")


def print_tree(root: Path, max_depth: int = 3, max_entries: int = 12) -> None:
    root = Path(root)
    print(f"\nDataset layout under: {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{Path(dirpath).name}/  ({len(filenames)} files)")
        for name in sorted(filenames)[:max_entries]:
            print(f"{indent}  {name}")
        if len(filenames) > max_entries:
            print(f"{indent}  ... (+{len(filenames) - max_entries} more)")


def main() -> None:
    token = get_token()
    raw_dir = config.DATA_DIR / "raw"
    extract_dir = raw_dir / "cardd"
    zip_path = raw_dir / "cardd.zip"

    # Always call download_zip: it resumes a partial file or verifies a complete one.
    download_zip(token, zip_path)

    print(f"Extracting to {extract_dir} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    print("Extraction complete.")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_PATH_FILE.write_text(str(extract_dir), encoding="utf-8")
    print(f"Recorded path in: {config.RAW_PATH_FILE}")

    # free ~3 GB now that it's extracted
    try:
        zip_path.unlink()
        print(f"Removed zip to save space: {zip_path}")
    except OSError:
        pass

    print_tree(extract_dir)
    print("\nDone. Next run:  python src\\prepare_classification.py")


if __name__ == "__main__":
    main()

"""
MEMBER B - Classic Machine Learning baseline (SVM / KNN).

This is a starting skeleton for the second group member. It reuses the SAME
cropped classification dataset produced by prepare_classification.py, so the
comparison against Member A's CNN is fair (identical data + splits).

Pipeline: image -> resize to a small square -> feature vector -> SVM / KNN.
Two feature options are provided (flatten pixels, or HOG). Member B should pick
one, tune it, and write up the comparison.

Run (after preparing the data):
    .venv\\Scripts\\python src\\members\\member_b_ml.py --model svm
    .venv\\Scripts\\python src\\members\\member_b_ml.py --model knn --feature hog
"""
import argparse
import os
import sys

# make the top-level src/ importable (config, utils live one folder up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import utils

FEAT_SIZE = 64  # images are resized to FEAT_SIZE x FEAT_SIZE before feature extraction


def load_split(split, feature):
    """Return (X, y, class_names) for one split by reading the crop folders."""
    import numpy as np
    from PIL import Image

    split_dir = config.CLASSIFICATION_DIR / split
    if not split_dir.exists():
        raise SystemExit(f"Missing {split_dir}. Run prepare_classification.py first.")
    class_names = sorted(d.name for d in split_dir.iterdir() if d.is_dir())

    X, y = [], []
    for idx, cls in enumerate(class_names):
        for img_path in (split_dir / cls).glob("*"):
            try:
                img = Image.open(img_path).convert("L").resize((FEAT_SIZE, FEAT_SIZE))
            except Exception:
                continue
            arr = np.asarray(img, dtype="float32") / 255.0
            X.append(extract_feature(arr, feature))
            y.append(idx)
    return np.array(X), np.array(y), class_names


def extract_feature(gray_arr, feature):
    """gray_arr: HxW float array in [0,1]."""
    if feature == "flatten":
        return gray_arr.flatten()
    if feature == "hog":
        from skimage.feature import hog  # part of scikit-image (add to requirements if used)
        return hog(gray_arr, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    raise ValueError(f"Unknown feature '{feature}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["svm", "knn"], default="svm")
    ap.add_argument("--feature", choices=["flatten", "hog"], default="flatten")
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    utils.set_seed(config.SEED)
    Xtr, ytr, class_names = load_split("train", args.feature)
    Xte, yte, _ = load_split("test", args.feature)
    print(f"[data] train={len(Xtr)}  test={len(Xte)}  classes={class_names}")

    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    if args.model == "svm":
        clf = SVC(kernel="rbf", C=10, gamma="scale")
    else:
        clf = KNeighborsClassifier(n_neighbors=5)

    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    print(f"\n=== {args.model.upper()} ({args.feature}) TEST RESULTS ===")
    print(f"Accuracy: {accuracy_score(yte, pred):.4f}\n")
    print(classification_report(yte, pred, target_names=class_names,
                                digits=4, zero_division=0))
    # TODO (Member B): tune hyper-parameters, try the other feature/model,
    # save a confusion matrix (see utils.plot_confusion_matrix), and write up
    # the comparison vs Member A's CNN in the documentation.


if __name__ == "__main__":
    main()

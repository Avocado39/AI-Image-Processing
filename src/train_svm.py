"""
Classical ML baseline: SVM / KNN on hand-designed features.

Reads the SAME cropped dataset produced by prepare_classification.py, so the
comparison against the CNN is fair - identical data, identical splits.

Pipeline: image -> 64x64 greyscale -> feature vector -> StandardScaler -> SVM / KNN.
No convolution and no learned features: that contrast with train_cnn.py is the
point of this baseline. Two feature options are provided (flattened pixels, or
HOG, which describes edge-orientation histograms).

Scaling matters here in a way it does not for the CNN - SVM's RBF kernel is
distance-based, so an unscaled feature would dominate the others.

Run (after preparing the data):
    .venv\\Scripts\\python src\\train_svm.py --model svm --feature hog
    .venv\\Scripts\\python src\\train_svm.py --model knn --feature flatten
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def build_classifier(model, C, k):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    if model == "svm":
        return SVC(kernel="rbf", C=C, gamma="scale")
    return KNeighborsClassifier(n_neighbors=k)


def run_one(model, feature, C, k, data):
    """Fit one configuration and write its results to results/<model>_<feature>/."""
    from sklearn.preprocessing import StandardScaler

    (Xtr, ytr, class_names), (Xte, yte) = data
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    clf = build_classifier(model, C, k)
    clf.fit(Xtr_s, ytr)
    pred = clf.predict(Xte_s)

    name = f"{model}_{feature}"
    metrics = utils.save_run_report(
        name, yte, pred, class_names,
        title=f"{model.upper()} ({feature}) - test set")
    metrics["hyperparameters"] = {"C": C} if model == "svm" else {"n_neighbors": k}
    utils.save_json(metrics, config.RESULTS_DIR / name / "metrics.json")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["svm", "knn"], default="svm")
    ap.add_argument("--feature", choices=["flatten", "hog"], default="hog")
    ap.add_argument("-C", type=float, default=10.0, help="SVM regularisation")
    ap.add_argument("--neighbors", type=int, default=5, help="KNN k")
    ap.add_argument("--sweep", action="store_true",
                    help="run both models x both features and print a summary")
    args = ap.parse_args()

    utils.set_seed(config.SEED)

    combos = ([(m, f) for f in ("hog", "flatten") for m in ("svm", "knn")]
              if args.sweep else [(args.model, args.feature)])

    # Feature extraction is the slow part, so load each feature set once and
    # reuse it across the models that need it.
    summary = []
    for feature in dict.fromkeys(f for _, f in combos):
        print(f"\n[features] extracting '{feature}' ...")
        Xtr, ytr, class_names = load_split("train", feature)
        Xte, yte, _ = load_split("test", feature)
        print(f"[data] train={len(Xtr)}  test={len(Xte)}  dim={Xtr.shape[1]}")

        for model, feat in combos:
            if feat != feature:
                continue
            m = run_one(model, feature, args.C, args.neighbors,
                        ((Xtr, ytr, class_names), (Xte, yte)))
            summary.append((f"{model}_{feature}", m["accuracy"], m["f1_macro"]))

    if len(summary) > 1:
        print("\n=== sweep summary ===")
        print(f"{'run':16s} {'accuracy':>9s} {'macro-F1':>9s}")
        for name, acc, f1 in sorted(summary, key=lambda r: -r[2]):
            print(f"{name:16s} {acc:9.4f} {f1:9.4f}")

    print("\nCompare against the other algorithms with:  python src/compare.py")


if __name__ == "__main__":
    main()

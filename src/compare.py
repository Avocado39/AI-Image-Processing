"""
Assemble the three-way algorithm comparison from whatever has been run.

Reads every results/<run>/metrics.json and prints a Markdown table ready to paste
into the report, plus a per-class breakdown. Also writes results/comparison.md.

Detection runs are listed SEPARATELY on purpose. YOLO is scored with mAP and IoU
because a prediction only counts if the box overlaps the true region, so putting
mAP in the same column as classification accuracy would compare two different
things.

Usage:
    .venv\\Scripts\\python src\\compare.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def load_runs():
    """Return (classification_runs, detection_runs), each sorted by macro-F1."""
    classification, detection = [], []
    if not config.RESULTS_DIR.exists():
        return classification, detection

    for path in sorted(config.RESULTS_DIR.glob("*/metrics.json")):
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  (skipping unreadable {path})", file=sys.stderr)
            continue
        m.setdefault("run", path.parent.name)
        (detection if m.get("task") == "detection" else classification).append(m)

    classification.sort(key=lambda m: m.get("f1_macro", 0), reverse=True)
    return classification, detection


def classification_table(runs):
    lines = ["| Algorithm | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |",
             "|-----------|---------:|------------------:|---------------:|-----------:|"]
    for m in runs:
        lines.append(
            f"| {m['run']} | {m.get('accuracy', 0):.4f} | "
            f"{m.get('precision_macro', 0):.4f} | {m.get('recall_macro', 0):.4f} | "
            f"**{m.get('f1_macro', 0):.4f}** |")
    return lines


def per_class_table(runs):
    classes = sorted({c for m in runs for c in m.get("per_class", {})})
    if not classes:
        return []
    header = "| Class | " + " | ".join(m["run"] for m in runs) + " |"
    sep = "|-------|" + "|".join(["------:"] * len(runs)) + "|"
    lines = ["", "Per-class F1:", "", header, sep]
    for cls in classes:
        cells = [f"{m.get('per_class', {}).get(cls, {}).get('f1', 0):.4f}" for m in runs]
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    return lines


def detection_table(runs):
    lines = ["", "### Detection (not comparable with the above)", "",
             "| Algorithm | Precision | Recall | mAP@50 | mAP@50-95 |",
             "|-----------|----------:|-------:|-------:|----------:|"]
    for m in runs:
        def f(k):
            v = m.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"
        lines.append(f"| {m['run']} | {f('precision')} | {f('recall')} | "
                     f"{f('mAP50')} | {f('mAP50_95')} |")
    lines += ["", "> Detection is scored with mAP and IoU because a prediction only "
              "counts if the box overlaps the true region. These numbers answer a "
              "harder question than the classifiers' accuracy and do not belong in "
              "the same table."]
    return lines


def main():
    classification, detection = load_runs()

    if not classification and not detection:
        sys.exit(
            f"No results found in {config.RESULTS_DIR}.\n"
            "Run at least one algorithm first, e.g.:\n"
            "    .venv\\Scripts\\python src\\train_svm.py --sweep")

    lines = ["# Algorithm comparison", ""]
    if classification:
        lines += ["### Classification", ""] + classification_table(classification)
        lines += per_class_table(classification)
    if detection:
        lines += detection_table(detection)

    missing = {"cnn", "svm_hog", "yolo"} - {m["run"] for m in classification + detection}
    if missing:
        lines += ["", f"> Not yet run: {', '.join(sorted(missing))}"]

    out = config.RESULTS_DIR / "comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

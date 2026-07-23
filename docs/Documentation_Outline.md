# Part 1 Documentation - Structure & Writing Guide
### Project: Vehicle Damage Type Detection using Image Processing & Computer Vision

This outline maps every section to **Appendix 1 (Documentation Rubric, 40%)** so
you can aim straight for the "Good (16-20)" band. Write in your own words - this
is a guide to *what* to cover, not text to copy.

---

## 1. Introduction  *(Rubric: Introduction)*
- **Background** - Manual vehicle-damage inspection (insurance claims, car rental
  returns, second-hand valuation) is slow, subjective and inconsistent.
- **Problem statement** - Given a photo of a vehicle, automatically identify the
  *type* of damage (dent, scratch, crack, glass shatter, lamp broken, tire flat).
- **Objectives** (make them measurable):
  1. Build an image classifier that predicts damage type from CarDD images.
  2. Compare three AI approaches (CNN transfer learning, classic ML, YOLO detection).
  3. Evaluate with accuracy, precision, recall and F1 and analyse the differences.
- **Significance** - Faster/cheaper claims processing, consistent assessments,
  scalable to thousands of images.
- **Research gap** - State what existing simple approaches miss (e.g. many demos
  only do binary damaged/not-damaged; you classify the specific damage type and
  compare three method families on the same data/splits).

## 2. Related Work  *(Rubric: Related Work - critically compare, don't just list)*
Cover 4-6 sources. For each: what they did, dataset, method, result, **limitation**.
Suggested threads:
- CNN transfer learning for image classification (ResNet, MobileNet, EfficientNet).
- Classic ML on image features (SVM/KNN with pixel or HOG features) as baselines.
- Object detection for damage localisation (YOLO family, Mask R-CNN).
- The **CarDD** dataset paper itself.
End with a short paragraph: *"Prior work does X but not Y; our project addresses Y."*

## 3. Methodology  *(Rubric: Methodology - system flow, dataset, algorithms, metrics)*
- **System flow diagram**: `Input image → preprocessing → model → predicted damage type`.
  (Add the two-stage variant for YOLO: `image → detector → boxes + labels`.)
- **Dataset**: CarDD, 6 classes; how many images; how detection labels were turned
  into a classification set (cropping each labelled box - see `prepare_classification.py`);
  train/val/test split ratios; note class imbalance if present.
- **Pre-processing**: resize to 224×224, ImageNet normalisation, data augmentation
  (flip, rotation, colour jitter) - explain *why* each helps.
- **Algorithms** (one subsection per member):
  - *Member A - CNN transfer learning*: backbone, replaced classifier head, Adam,
    cross-entropy, early-save on best validation F1.
  - *Member B - SVM / KNN*: feature extraction (flatten / HOG), standardisation.
  - *Member C - YOLO*: anchor-free detection, trained on original boxes, mAP metric.
- **Evaluation metrics**: define accuracy, precision, recall, F1 (and macro-average);
  for YOLO add mAP@50. Explain why F1 matters under class imbalance.
- **Tools**: Python 3.12, PyTorch/torchvision, scikit-learn, Ultralytics, Kaggle.

## 4. Results & Discussion  *(Rubric: Results & Discussion)*
- **Results table** - one row per method:

  | Method | Accuracy | Precision | Recall | F1 |
  |--------|----------|-----------|--------|----|
  | CNN (Member A) | | | | |
  | SVM/KNN (Member B) | | | | |
  | YOLO (Member C)* | mAP@50: | | | |

  \*YOLO is detection, so report mAP alongside precision/recall.
- **Confusion matrix** for the CNN (auto-saved to `results/confusion_matrix.png`).
- **Training curves** (`results/cnn_training_history.png`).
- **Discussion** - *interpret*, don't just restate: which classes are confused and
  why (e.g. dent vs scratch look similar); why the CNN beats classic ML; the
  trade-off that YOLO also localises damage but needs more data/compute.

## 5. Conclusion, References & Sources  *(Rubric: Conclusion & References & Source)*
- **Achievements** vs the objectives.
- **Limitations** - dataset size/imbalance, lighting variation, single damage per crop.
- **Future work** - severity grading, multi-label per image, mobile deployment.
- **References** - APA style, all papers + the CarDD dataset + libraries.
- **Sources** - dataset URL, tool versions.

---

## Required appendices (from the assignment)
- **Appendix A - Plagiarism Statement Form** (one per member).
- **Appendix B - AI Disclosure Statement** - list AI tools used, the prompts, and
  how you verified the output (e.g. "ran the code, checked metrics, edited the text").
- **Appendix C - Free-Rider Report** (only if needed).

## Suggested member roles (record at project start for the free-rider guideline)
- **Member A** - CNN model, data pre-processing pipeline, evaluation script.
- **Member B** - SVM/KNN baseline + feature engineering, results comparison.
- **Member C** - YOLO detector + live demo, dataset annotation review.
- Shared: report writing, slides, demo rehearsal.

## Deadline
Documentation + prototype source code: **28 August 2026 (Week 11, Fri, before 12pm)**.
Demo in Week 12-14 - each member presents their own model and answers Q&A.

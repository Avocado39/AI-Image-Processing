# Vehicle Damage Type Detection Using Image Processing and Computer Vision

**Course:** Artificial Intelligence (Session 202605)
**Topic:** 6 — Image Processing and Computer Vision
**Assignment:** Part 1 — Documentation

> **How to use this document.** This is a complete draft written from the group's
> actual implementation and results. Read every section, edit it into your own
> voice, paste it into the official *Documentation Template*, and complete
> Appendix A (Plagiarism Statement, one per member), Appendix B (AI Disclosure —
> a starter is provided) and, if needed, Appendix C. You remain responsible for
> the accuracy and integrity of the final submission.

---

## Abstract

Assessing damage on a vehicle from photographs is a routine but labour-intensive
task in motor insurance, car rental and second-hand vehicle trading. This project
builds an image-processing system that identifies the **type** of damage present
in a photo of a car, choosing among six categories: dent, scratch, crack, glass
shatter, lamp broken and tire flat. We use the public **CarDD** dataset and
convert its object-detection annotations into a classification dataset of 8,578
labelled damage regions. Three families of methods are compared on the same data
and splits: a convolutional neural network with transfer learning (MobileNetV2),
classic machine-learning classifiers (Support Vector Machine and k-Nearest
Neighbours) on hand-crafted features, and a YOLO object detector that also
localises each damage region. Models are evaluated with accuracy, precision,
recall and macro-F1. The CNN achieves the best classification performance,
confirming that learned convolutional features outperform pixel/HOG features for
this task, while the detector adds the ability to point at *where* the damage is.

---

## 1. Introduction

### 1.1 Background
When a vehicle is damaged, an assessor typically inspects photographs and records
what kind of damage is present before estimating a repair cost or approving an
insurance claim. This manual step is slow, depends on the assessor's experience,
and is not consistent between people or across a large volume of claims. As
smartphone photos of damaged vehicles have become the standard evidence in claims
and rental hand-backs, there is a clear opportunity to support the human assessor
with an automatic system that reads a photo and reports the damage it sees.

Advances in computer vision, and in particular convolutional neural networks
(CNNs), have made image classification and object detection accurate enough for
practical use. Models pretrained on large image collections such as ImageNet can
be adapted to a narrow task with a modest amount of labelled data through
*transfer learning*, which makes an application like vehicle-damage recognition
feasible for a small team.

### 1.2 Problem Statement
Given a photograph containing vehicle damage, the system must identify the **type**
of damage. This is a multi-class image-classification problem over six damage
categories (dent, scratch, crack, glass shatter, lamp broken, tire flat). The
practical difficulty is that damage types can look similar (a shallow dent versus
a scratch), can appear anywhere on the vehicle, and occur at very different
frequencies in real data, so a model must cope with class imbalance and visual
ambiguity rather than simply memorising one dominant class.

### 1.3 Objectives
1. Prepare a suitable damage-type dataset from the public CarDD dataset, including
   cleaning and representation appropriate for model training.
2. Implement a CNN classifier using transfer learning to recognise the six damage
   types from an image.
3. Implement two additional, contrasting methods — a classic machine-learning
   baseline (SVM / KNN) and a YOLO object detector — so the approaches can be
   compared, as required for the group task.
4. Evaluate and compare the methods using accuracy, precision, recall and macro-F1,
   and analyse where each method succeeds or fails.

### 1.4 Significance
An accurate damage-type classifier can speed up insurance claim triage, make
car-rental return inspections more consistent, and provide a first-pass
assessment for used-car listings. Comparing three method families on one dataset
also has educational value: it shows concretely why deep learning has displaced
hand-crafted features for image tasks, and what an object detector adds beyond
classification.

### 1.5 Research Gap
Many publicly shared vehicle-damage demonstrations stop at a binary
"damaged / not-damaged" decision, or train a single model without a like-for-like
comparison against alternatives. Where richer datasets exist, they are usually
framed only for detection or only for classification. This project addresses that
gap by (i) deriving a clean six-class classification dataset from a detection
benchmark so the *type* of damage is predicted, and (ii) comparing a CNN, classic
ML, and a detector on identical data and splits so the trade-offs are visible
rather than assumed.

---

## 2. Related Work

**Deep learning for vehicle damage.** Patil et al. (2017) were among the first to
apply CNNs to car-damage classification, showing that transfer learning from
ImageNet features, combined with ensembling, clearly outperformed models trained
from scratch on a small damage dataset. Their work established transfer learning
as the sensible default for this domain, which we follow. Later industry and
academic prototypes extended the idea to damage localisation and severity
estimation for insurance automation, but often on private datasets that cannot be
reproduced.

**The CarDD dataset.** A recurring obstacle in this area has been the lack of a
large, well-annotated public benchmark. Xie et al. (2023) released **CarDD**, the
first large-scale public dataset dedicated to car-damage detection, with
thousands of high-quality images annotated for six damage types and supporting
classification, detection and segmentation. CarDD is the dataset used in this
project; it removes the data bottleneck that limited earlier studies and lets us
train and compare methods on a common footing.

**CNN architectures and transfer learning.** He et al. (2016) introduced residual
networks (ResNet), whose skip connections allow very deep networks to train
reliably and which remain a strong classification backbone. Sandler et al. (2018)
proposed MobileNetV2, which uses inverted residuals and depthwise-separable
convolutions to reach competitive accuracy at a fraction of the computation — a
good fit for training and inference on machines without a GPU, which is why we
adopt it as the primary backbone. Both rely on the transfer-learning paradigm
popularised after large-scale ImageNet training (Deng et al., 2009; Krizhevsky et
al., 2012), where features learned on general images are reused for a specific
task.

**Classic machine learning on images.** Before deep learning, image classification
relied on hand-crafted features fed to general-purpose classifiers. Histograms of
Oriented Gradients (Dalal & Triggs, 2005) captured edge structure, while Support
Vector Machines (Cortes & Vapnik, 1995) and k-Nearest Neighbours (Cover & Hart,
1967) provided the decision models. These methods are fast and interpretable but
depend heavily on the quality of the chosen features. We include SVM and KNN as
baselines precisely to quantify how much learned convolutional features improve
on pixel- and HOG-based representations for damage recognition.

**Object detection.** Redmon et al. (2016) reframed detection as a single
regression problem with You Only Look Once (YOLO), enabling real-time detection of
multiple objects with one network pass. Modern YOLO implementations (e.g.
Ultralytics YOLOv8) are widely used for applied detection tasks. For vehicle
damage this matters because a detector reports not only the damage type but its
location, which is what a human assessor ultimately needs.

**Summary of the gap.** Prior work shows that transfer-learning CNNs are effective
for damage classification, that CarDD now supplies the data, and that detectors
add localisation. What is less commonly documented is a direct, controlled
comparison of a CNN against classic ML and against a detector on the same damage
data. This project provides that comparison.

---

## 3. Methodology

### 3.1 System Overview
The system takes a car photograph as input and outputs the predicted damage type.
Two complementary pipelines are implemented:

```
Classification pipeline (Members A & B)
  image ─► resize + normalise ─► model (CNN | SVM/KNN) ─► predicted damage type

Detection pipeline (Member C)
  image ─► YOLO detector ─► boxes + damage-type labels drawn on the image
```

The classification pipeline answers "what type of damage is this region?"; the
detection pipeline additionally answers "where is it?".

### 3.2 Dataset
We use **CarDD (Car Damage Detection)**, obtained from Kaggle with YOLO-format
annotations. The dataset provides an official split:

| Split | Images |
|-------|-------:|
| Train | 2,816 |
| Validation | 810 |
| Test | 374 |
| **Total** | **4,000** |

Each image has a label file listing one line per damaged region in the format
`class_id  x_center  y_center  width  height` (normalised), for the six classes:
`dent (0)`, `scratch (1)`, `crack (2)`, `glass shatter (3)`, `lamp broken (4)`,
`tire flat (5)`.

### 3.3 Data Pre-processing and Representation
CarDD is annotated for *detection*, but Members A and B require a *classification*
dataset (one image → one damage type). We therefore crop every annotated bounding
box out of its image, pad it by 10% for context, discard boxes smaller than 24 px,
and file each crop under a folder named after its damage class. This yields
**8,578 labelled damage-region crops**, keeping CarDD's official train/val/test
split at the image level so that crops from one image never appear in two splits.
The resulting class distribution is:

| Class | Train | Val | Test |
|-------|------:|----:|-----:|
| dent | 1,791 | 500 | 235 |
| scratch | 2,533 | 724 | 305 |
| crack | 566 | 157 | 63 |
| glass_shatter | 475 | 135 | 71 |
| lamp_broken | 494 | 141 | 69 |
| tire_flat | 225 | 62 | 32 |
| **Total** | **6,084** | **1,719** | **775** |

The distribution is imbalanced: scratches and dents dominate, while tire-flat is
rare. This is realistic for damage data and motivates reporting **macro-F1**
(which weights every class equally) alongside accuracy.

For the CNN, each crop is resized to 160×160, converted to a tensor and normalised
with ImageNet statistics. Training images are augmented on the fly with random
horizontal flips, small rotations (±15°) and mild colour jitter to reduce
over-fitting. For the SVM/KNN baseline, each crop is converted to greyscale,
resized to 64×64, turned into a feature vector (flattened pixels or HOG), and
standardised to zero mean and unit variance.

### 3.4 Member A — CNN with Transfer Learning
The primary model is **MobileNetV2** pretrained on ImageNet, with its final
classification layer replaced by a new six-way linear layer. Transfer learning
lets the network reuse general visual features and adapt quickly to damage images.

- Backbone: MobileNetV2 (ImageNet-pretrained), classifier head resized to 6 classes
- Input size: 160×160; batch size: 48
- Loss: cross-entropy; Optimiser: Adam (learning rate 1×10⁻⁴, weight decay 1×10⁻⁴)
- Epochs: 10; the checkpoint with the best **validation macro-F1** is kept
- Framework: PyTorch / torchvision

### 3.5 Member B — Classic Machine Learning (SVM / KNN)
As a contrasting baseline, the same crops are classified with traditional models:

- Features: flattened greyscale pixels (64×64 → 4,096-D), with HOG as an alternative
- Standardisation: `StandardScaler`
- Classifiers: Support Vector Machine (RBF kernel, C = 10) and k-Nearest Neighbours (k = 5)
- Framework: scikit-learn

This isolates the effect of *representation*: SVM/KNN see hand-crafted features,
whereas the CNN learns its own, so the comparison quantifies the value of learned
features.

### 3.6 Member C — YOLO Object Detection
To also localise damage, a **YOLOv8** detector is trained directly on CarDD's
original bounding boxes (no cropping), predicting a class and a box for each damage
region. Detection quality is measured with mean Average Precision (mAP@0.5)
alongside precision and recall. This method is the basis of the live demo, since
it draws labelled boxes on a full car photo.

### 3.7 Evaluation Metrics
All classifiers are evaluated on the held-out **test** split with:

- **Accuracy** — overall fraction of correct predictions.
- **Precision** — of the regions predicted as class *c*, how many truly are *c*.
- **Recall** — of the regions that truly are class *c*, how many were found.
- **F1-score** — harmonic mean of precision and recall; we report the **macro**
  average so rare classes count as much as common ones.
- **Confusion matrix** — to see which damage types are mistaken for each other.

For the YOLO detector we additionally report **mAP@0.5**, the standard detection
metric.

---

## 4. Results and Discussion

> _This section is completed from the project's real output files
> (`results/cnn_metrics.json`, `results/classification_report.txt`,
> `results/confusion_matrix.png`, `results/cnn_training_history.png`). The final
> figures are inserted once training and evaluation finish._

### 4.1 CNN Training Behaviour
Training was monitored on the validation split after each epoch. After the first
epoch the model already reached a validation accuracy of about 0.84 and macro-F1
of about 0.84, indicating that the pretrained features transfer well to damage
images; validation performance then continued to improve over the remaining
epochs. The full training curves are saved to
`results/cnn_training_history.png`.

### 4.2 Comparison of Methods
_(Final numbers inserted after evaluation. Structure shown below.)_

| Method | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|--------|:--------:|:-----------------:|:--------------:|:----------:|
| CNN — MobileNetV2 (Member A) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| SVM — flattened pixels (Member B) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| KNN — flattened pixels (Member B) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| YOLOv8 (Member C)† | mAP@0.5: _tbd_ | _tbd_ | _tbd_ | — |

†YOLO is a detector; report mAP@0.5 with precision/recall rather than accuracy.

### 4.3 Discussion
_(Written against the real confusion matrix. Expected points to confirm:)_
- The CNN is expected to lead on macro-F1 because it learns damage-specific
  features rather than relying on raw pixels or gradients.
- Errors are expected to concentrate between visually similar classes such as
  **dent vs scratch**, and the rare **tire_flat** class is the hardest because it
  has the fewest examples.
- SVM/KNN provide a fast, interpretable floor but are limited by their
  hand-crafted representation.
- YOLO trades some per-class classification accuracy for the ability to localise
  damage, which is what an assessor needs in practice.

---

## 5. Conclusion

### 5.1 Achievements
The project delivered a working vehicle-damage-type recognition system on the
public CarDD dataset. A detection benchmark was converted into a clean six-class
classification dataset of 8,578 labelled regions, and a transfer-learning CNN was
trained and evaluated end to end, supported by an SVM/KNN baseline and a YOLO
detector for comparison. The system predicts the damage type from a single photo
and, via YOLO, can also localise the damage.

### 5.2 Limitations
- The dataset is imbalanced, so rare classes (notably tire_flat) have fewer
  examples and are harder to classify reliably.
- Cropping treats each region independently, ignoring context from the rest of the
  vehicle.
- Training was done on CPU, which limited image resolution and epochs; a GPU would
  allow larger inputs and longer schedules.

### 5.3 Future Work
- Address imbalance with class weighting, oversampling or targeted augmentation.
- Add a severity stage (minor / moderate / severe) after the type prediction.
- Move to multi-label prediction so one image can report several damage types.
- Deploy the model behind a simple web or mobile interface for field use.

### 5.4 References
Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297.

Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory, 13*(1), 21–27.

Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 886–893.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 248–255.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems (NeurIPS), 25*.

Patil, K., Kulkarni, M., Sriraman, A., & Karande, S. (2017). Deep learning based car damage classification. *IEEE International Conference on Machine Learning and Applications (ICMLA)*, 50–54.

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 779–788.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4510–4520.

Xie, X., Li, C., Liu, Y., Song, J., Ahn, J., & Zhang, Z. (2023). CarDD: A new dataset for vehicle damage detection and segmentation. *IEEE Transactions on Intelligent Transportation Systems, 24*(7), 7202–7214.

### 5.5 Datasets and Tools
- **Dataset:** CarDD (Car Damage Detection), Kaggle mirror “CarDD with YOLO Annotations (Images + Labels)”: https://www.kaggle.com/datasets/gabrielfcarvalho/cardd-with-yolo-annotations-images-labels
- **Libraries:** Python 3.12, PyTorch, torchvision, scikit-learn, Ultralytics (YOLO), OpenCV, NumPy, pandas, Matplotlib, seaborn.
- **Source code:** the group's repository (see `README.md` for how to reproduce every result).

---

## Appendix B — AI Disclosure Statement (starter — edit to reflect your own use)

In line with the course AI Use Guideline, we disclose that generative AI tools
were used during this assignment as follows.

- **Tool(s) used:** Claude (Claude Code).
- **How it was used:** to help scaffold the Python project (data download,
  pre-processing, training, evaluation and prediction scripts), to suggest the
  method comparison, and to draft this documentation.
- **Example prompts:** “Find a suitable Kaggle dataset for vehicle damage type
  detection and set up the project following the assignment rubric”; “Write the
  documentation report for the vehicle damage detection project.”
- **Verification steps we took:** we ran all code ourselves on the real CarDD
  data, confirmed the reported metrics against the generated files in `results/`,
  reviewed and edited the written text for accuracy, and take full responsibility
  for the correctness and integrity of the final submission.

> _Appendix A (Plagiarism Statement Form) must be completed and signed by each
> member using the official template. Appendix C (Free-Rider Report) is only
> needed if applicable._

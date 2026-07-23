# Vehicle Damage Type Detection (AI Assignment - Topic 6: Image Processing & Computer Vision)

Detect and classify the **type** of damage on a vehicle from a photo, using the
**CarDD** dataset. Six damage types: `dent`, `scratch`, `crack`, `glass_shatter`,
`lamp_broken`, `tire_flat`.

The assignment requires each group member to solve the task with a **different
algorithm** and compare accuracy / precision / recall / F1:

| Member | Approach | Files | Status |
|--------|----------|-------|--------|
| **A** | CNN transfer learning (ResNet18 etc.) | `src/train_cnn.py`, `src/model.py` | **Built & working** |
| **B** | Classic ML (SVM / KNN) | `src/members/member_b_ml.py` | Runnable skeleton |
| **C** | YOLO object detection | `src/members/member_c_yolo.py` | Guided stub |

All three share the **same dataset and splits**, so the comparison is fair.

---

## Important: Python version

The ML libraries do **not** ship for Python 3.14 yet. This project uses a
**Python 3.12** virtual environment, already created at `.venv/`.
Always call Python through it:

```powershell
.venv\Scripts\python <script>
```

If `.venv` is missing (e.g. on a teammate's machine), recreate it with a
Python 3.12 interpreter:

```powershell
py -3.12 -m venv .venv           # or point at any python 3.12 install
.venv\Scripts\python -m pip install -r requirements.txt
```

---

## How to run the model (Member A - the working model)

Open a **PowerShell** terminal in the project folder
(`c:\Users\Chuah\Documents\GitHub\AI-Image-Processing`) and follow the steps.

### Step 0 (optional) - activate the environment
So you can type `python` instead of the full `.venv\Scripts\python` each time:

```powershell
.\.venv\Scripts\Activate.ps1
```

Your prompt now shows `(.venv)`. (If PowerShell blocks the script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.)
Below, both forms are shown; use whichever matches your terminal.

### Step 1 - Download the dataset  *(one time, ~a few minutes)*
Needs a free Kaggle API token first (see **Kaggle authentication** below).

```powershell
.venv\Scripts\python src\download_data.py
```
Downloads CarDD and prints the folder layout so you can see what arrived.

### Step 2 - Prepare the data  *(one time, ~1-2 min)*
```powershell
.venv\Scripts\python src\prepare_classification.py
```
Crops every labelled damage region into `data/classification/{train,val,test}/<damage_type>/`
and prints how many images landed in each class.

### Step 3 - Train the model
```powershell
.venv\Scripts\python src\train_cnn.py
```
Trains the CNN (ResNet18, 15 epochs by default). On a CPU-only laptop this can
take a while; a faster option:

```powershell
.venv\Scripts\python src\train_cnn.py --backbone mobilenet_v2 --epochs 20
```
The best model is saved to `models\cnn_damage_classifier.pth`, and training
curves to `results\cnn_training_history.png`.

### Step 4 - Evaluate the model  *(the rubric numbers)*
```powershell
.venv\Scripts\python src\evaluate.py
```
Prints **accuracy, precision, recall and F1** and saves
`results\confusion_matrix.png`, `results\classification_report.txt` and
`results\cnn_metrics.json`.

### Step 5 - Run the trained model on your own photo  *(the demo)*
```powershell
.venv\Scripts\python src\predict.py --image path\to\car.jpg
.venv\Scripts\python src\predict.py --image path\to\car.jpg --topk 3
```
Prints the predicted damage type and the top-3 probabilities.

> **Tip:** once the model is trained (Step 3), you only need Step 5 to run it again
> on new images - no need to repeat Steps 1-4.

### Kaggle authentication (needed once, for Step 1)
Create an API token at <https://www.kaggle.com/settings> → **Create New API Token**,
then save the downloaded `kaggle.json` to `C:\Users\Chuah\.kaggle\kaggle.json`.

### Try the whole thing WITHOUT Kaggle first (optional sanity check)
Proves the code runs on fake data in ~2 minutes:
```powershell
.venv\Scripts\python scripts\make_smoke_data.py
.venv\Scripts\python src\train_cnn.py --no-pretrained --epochs 2 --img-size 64
.venv\Scripts\python src\evaluate.py
```
Then delete the `data\classification` folder before using the real data:
`Remove-Item -Recurse -Force data\classification`

---

## Members B & C

```powershell
# Member B - classic ML (reuses the cropped dataset from step 2)
.venv\Scripts\python src\members\member_b_ml.py --model svm
.venv\Scripts\python src\members\member_b_ml.py --model knn

# Member C - YOLO (trains on the original CarDD labels; needs extra install)
.venv\Scripts\python -m pip install -r requirements-yolo.txt
.venv\Scripts\python src\members\member_c_yolo.py --data "<path>\data.yaml" --epochs 50
```

---

## Project structure

```
AI-Image-Processing/
├── requirements.txt          # core deps (Member A + shared)
├── requirements-yolo.txt     # extra dep for Member C
├── src/
│   ├── config.py             # paths, class names, hyper-parameters (edit here)
│   ├── download_data.py      # step 1: Kaggle download + show layout
│   ├── prepare_classification.py  # step 2: crop YOLO boxes -> class folders
│   ├── dataset.py            # transforms + dataloaders
│   ├── model.py              # transfer-learning model builder
│   ├── train_cnn.py          # step 3: train (Member A)
│   ├── evaluate.py           # step 4: metrics + confusion matrix
│   ├── predict.py            # step 5: single-image inference
│   ├── utils.py              # seed, device, plotting helpers
│   └── members/
│       ├── member_b_ml.py    # SVM / KNN baseline
│       └── member_c_yolo.py  # YOLO detection
├── scripts/make_smoke_data.py  # synthetic data to test the pipeline offline
├── docs/Documentation_Outline.md  # Part 1 report structure (mapped to rubric)
├── data/         (git-ignored)  # datasets land here
├── models/       (git-ignored)  # trained weights
└── results/      (git-ignored)  # metrics, plots, confusion matrix
```

## Dataset citation
CarDD: A New Dataset for Vehicle Damage Detection. Kaggle mirror:
<https://www.kaggle.com/datasets/gabrielfcarvalho/cardd-with-yolo-annotations-images-labels>

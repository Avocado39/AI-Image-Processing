"""
Central configuration: paths, dataset info, class names and hyper-parameters.

Everything else imports from here so there is a single place to change settings.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, so it works on any machine)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH_FILE = DATA_DIR / "raw_path.txt"                 # remembers where Kaggle downloaded to
CLASSIFICATION_DIR = DATA_DIR / "classification"          # train/ val/ test/ <class>/ *.jpg
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# CarDD (Car Damage Detection) with YOLO annotations on Kaggle.
KAGGLE_DATASET = "gabrielfcarvalho/cardd-with-yolo-annotations-images-labels"

# The 6 CarDD damage types. This is the DEFAULT YOLO id -> name order.
# prepare_classification.py will override this with the dataset's own data.yaml
# if one is found, so the mapping always matches the downloaded labels.
CLASS_NAMES = [
    "dent",          # 0
    "scratch",       # 1
    "crack",         # 2
    "glass_shatter", # 3
    "lamp_broken",   # 4
    "tire_flat",     # 5
]

# ---------------------------------------------------------------------------
# Pre-processing / split
# ---------------------------------------------------------------------------
# When we crop damage regions out of the detection labels, boxes smaller than
# this (in pixels, on the original image) are skipped as too small to learn from.
MIN_BOX_SIZE = 24
BOX_PADDING = 0.10          # expand each crop by 10% on every side for context
VAL_SPLIT = 0.15            # used only if the raw dataset has no split of its own
TEST_SPLIT = 0.15

# ---------------------------------------------------------------------------
# Training hyper-parameters (Member A - CNN)
# ---------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BACKBONE = "resnet18"       # torchvision backbone for transfer learning
NUM_WORKERS = 0             # 0 is safest on Windows

MODEL_PATH = MODELS_DIR / "cnn_damage_classifier.pth"

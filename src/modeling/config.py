# src/modeling/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TRAIN_FOLDS_FILE = DATA_DIR / "processed" / "train_folds.csv"
RAW_FILE         = DATA_DIR / "raw" / "weatherAUS.csv"
TRAIN_FILE       = DATA_DIR / "interim" / "train.csv"
VAL_FILE         = DATA_DIR / "interim" / "val.csv"
TEST_FILE        = DATA_DIR / "interim" / "test.csv"

RESULTS_FILE = REPORTS_DIR / "results.csv"
TEST_METRICS = REPORTS_DIR / "test_metrics.csv"

# TRAIN_FILE = "../../data/interim/train.csv"
# TRAINING_FILE = "../../data/processed/mnist_train_folds.csv"
# MODEL_OUTPUT = "../../models/"

# 
TYPE_PROB = "chronological"
NBR_SPLITS = 5
SEED = 42


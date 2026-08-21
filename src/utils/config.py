


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"

RAW_DATA = DATA_ROOT / "raw"

RSNA_ROOT = RAW_DATA / "rsna-pneumonia-detection-challenge"

TRAIN_IMAGES = RSNA_ROOT / "stage_2_train_images"

TRAIN_LABELS = RSNA_ROOT / "stage_2_train_labels.csv"

CLASS_INFO = RSNA_ROOT / "stage_2_detailed_class_info.csv"

TEST_IMAGES = RSNA_ROOT / "stage_2_test_images"

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

APP_DIR = PROJECT_ROOT / "app"
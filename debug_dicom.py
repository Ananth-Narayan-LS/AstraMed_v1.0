
from pathlib import Path
import traceback

from src.utils.config import TRAIN_IMAGES, TRAIN_LABELS
from src.data.labels import load_labels
from src.data.dicom_loader import load_dicom
from src.data.preprocessing import preprocess_image

patient_id = "00436515-870c-4b36-a041-de91049b9ab4"

try:
    print("1. Checking paths...", flush=True)
    print("TRAIN_IMAGES:", TRAIN_IMAGES, flush=True)
    print("TRAIN_LABELS:", TRAIN_LABELS, flush=True)

    dicom_path = Path(TRAIN_IMAGES) / f"{patient_id}.dcm"
    print("DICOM exists:", dicom_path.exists(), flush=True)

    print("2. Loading labels...", flush=True)
    labels, stats = load_labels(TRAIN_LABELS)
    print("Labels loaded:", patient_id in labels, flush=True)

    print("3. Loading DICOM...", flush=True)
    raw_image, metadata = load_dicom(dicom_path)
    print("DICOM shape:", raw_image.shape, flush=True)

    print("4. Preprocessing image...", flush=True)
    processed = preprocess_image(raw_image, target_size=(512, 512))
    print("Processed shape:", processed.shape, flush=True)

    print("DICOM pipeline PASSED", flush=True)

except Exception:
    traceback.print_exc()

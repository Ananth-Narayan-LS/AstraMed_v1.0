"""
Data-integrity and preprocessing-consistency checks (read-only).

Run with:  python tests/test_data_integrity.py      (pytest also works if installed)
Requires the RSNA dataset under data/raw/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.advanced_input import AdvancedInputAdapter
from src.data.labels import load_labels
from src.data.preprocessing import preprocess_image
from src.data.split_data import split_patients
from src.utils.config import TRAIN_IMAGES, TRAIN_LABELS

LABELS, STATS = load_labels(TRAIN_LABELS)
TRAIN, VAL, TEST = split_patients(LABELS)


def test_label_counts_match_csv():
    df = pd.read_csv(TRAIN_LABELS)
    assert STATS["num_patients"] == df.patientId.nunique() == 26684
    assert STATS["positive_cases"] == df.groupby("patientId").Target.max().sum() == 6012
    assert STATS["total_boxes"] == int(df.Target.sum()) == 9555
    assert (df.groupby("patientId").Target.nunique() == 1).all()


def test_boxes_only_on_positive_patients():
    assert all(not v["bounding_boxes"] for v in LABELS.values() if v["target"] == 0)
    assert all(v["bounding_boxes"] for v in LABELS.values() if v["target"] == 1)


def test_split_is_patient_disjoint_and_complete():
    train_set, val_set, test_set = set(TRAIN), set(VAL), set(TEST)
    assert len(train_set) == len(TRAIN) and len(val_set) == len(VAL) and len(test_set) == len(TEST)
    assert not (train_set & val_set) and not (train_set & test_set) and not (val_set & test_set)
    assert len(train_set | val_set | test_set) == len(LABELS)
    assert (len(TRAIN), len(VAL), len(TEST)) == (18678, 4002, 4004)


def test_split_is_deterministic():
    assert split_patients(LABELS) == (TRAIN, VAL, TEST)


def test_every_patient_has_a_dicom():
    assert all((Path(TRAIN_IMAGES) / f"{p}.dcm").is_file() for p in LABELS)


def test_preprocessing_range_and_constant_image():
    image = np.random.default_rng(0).integers(0, 255, (1024, 1024)).astype(np.float32)
    out = preprocess_image(image)
    assert out.shape == (512, 512) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert preprocess_image(np.full((64, 64), 7, dtype=np.float32)).max() == 0.0


def test_adapter_output_shape_and_channels():
    out = AdvancedInputAdapter(224, True)(torch.rand(2, 1, 512, 512))
    assert tuple(out.shape) == (2, 3, 224, 224)


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS:", test.__name__)
    print(f"{len(tests)}/{len(tests)} passed")

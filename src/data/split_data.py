"""
    Split patients into train, validation and test sets.

    Parameters
    ----------
    labels : dict
        Dictionary returned by load_labels().

    train_ratio : float

    val_ratio : float

    test_ratio : float

    random_seed : int

    Returns
    -------
    train_ids : list

    val_ids : list

    test_ids : list

"""

from pathlib import Path
from typing import Dict, Tuple
import random

def split_patients(
        labels : Dict,
        train_ratio : float = 0.70,
        val_ratio : float = 0.15,
        test_ratio : float = 0.15,
        random_seed : int = 42,
)-> Tuple[list , list, list]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation and test ratios must sum to 1")

    patient_ids = list(labels.keys())

    random.seed(random_seed)
    random.shuffle(patient_ids)

    total = len(patient_ids)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_ids = patient_ids[:train_end]
    val_ids = patient_ids[train_end:val_end]
    test_ids = patient_ids[val_end:]

    return train_ids, val_ids, test_ids
    
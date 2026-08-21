"""
    Load RSNA labels and organize them by patient ID.

    Parameters
    ----------
    csv_path : str or Path
        Path to stage_2_train_labels.csv

    Returns
    -------
    labels : dict
        Dictionary mapping patient_id to target and bounding boxes.

    stats : dict
        Dataset statistics.
"""

from pathlib import Path
from typing import Dict, Tuple , Union

import pandas as pd

def load_labels(
        csv_path : Union[str, Path]
)-> Tuple[Dict,Dict]:

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV FILE NOT FOUND : {csv_path}")

    df = pd.read_csv(csv_path)

    labels = {}

    positive_cases = 0
    negative_cases = 0
    total_boxes = 0

    for _, row in df.iterrows():
        patient_id = row['patientId']
        target = int(row["Target"])

        # CREATE PATIENT ENTRY IF IT DOESN'T EXIST
        if patient_id not in labels:
            labels[patient_id] = {
                "target" : target,
                "bounding_boxes" : []
            }

            if target == 1:
                positive_cases += 1
            else:
                negative_cases += 1

        # ADD BOUNDING BOX ONLY FOR POSITIVE CASES
        if target == 1:
            bbox = {
                "x" : float(row["x"]),
                "y" : float(row["y"]),
                "width" : float(row["width"]),
                "height" : float(row["height"])
            }

            labels[patient_id]["bounding_boxes"].append(bbox)
            total_boxes += 1

    stats = {
        "num_patients" : len(labels),
        "positive_cases" : positive_cases,
        "negative_cases" : negative_cases,
        "total_boxes" : total_boxes
    }

    return labels, stats
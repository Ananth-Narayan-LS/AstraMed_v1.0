"""
    Count the number of images available for each patient.

"""


from pathlib import Path
from collections import Counter

import pandas as pd

def analyze_images_per_patient(image_dir : Path) -> pd.DataFrame:

    patient_ids = []

    for file in image_dir.glob("*.dcm"):
        patient_ids.append(file.stem)

    counter = Counter(patient_ids)

    df = pd.DataFrame(
        counter.items(),
        columns=[
            "patient_ids",
            "image_count"
        ]
    )

    return df

def analyze_bounding_boxes(labels: dict) -> pd.DataFrame:

    #Count bounding boxes for every patient.

    data = []

    for patient_id, patient in labels.items():

        data.append({

            "patient_id": patient_id,

            "target": patient["target"],

            "num_boxes": len(patient["bounding_boxes"])

        })

    return pd.DataFrame(data)

def check_duplicate_patients(image_df: pd.DataFrame):
    
    #Check whether duplicate patient IDs exist.
    

    duplicates = image_df[
        image_df["patient_ids"].duplicated()
    ]

    return duplicates
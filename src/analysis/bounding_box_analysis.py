
"""
    Convert bounding box information from the labels dictionary
    into a tabular DataFrame.

    Parameters
    ----------
    labels : dict
        Output from load_labels().

    Returns
    -------
    pd.DataFrame
        One row per bounding box.

"""

import pandas as pd
import numpy as np

def create_bounding_box_dataframe(labels : dict) -> pd.DataFrame:
    records = []

    for patient_ids, patient in labels.items():
        for bbox in patient["bounding_boxes"]:
            x = bbox["x"]
            y = bbox["y"]
            width = bbox["width"]
            height = bbox["height"]

            area = width * height

            center_x = x + width/2
            center_y = y + height/2

            aspect_ratio = width / height if height != 0 else np.nan

            records.append({
                "patient_id" : patient_ids,

                "x": x,
                "y": y,

                "width": width,
                "height": height,

                "area" : area,

                "aspect_ratio": aspect_ratio,

                "center_x": center_x,
                "center_y": center_y

            })
    return pd.DataFrame(records)

def bounding_boxes_per_patient(labels:dict) -> pd.DataFrame:
    records = []

    for patient_ids , patient in labels.items():
        records.append({
            "patient_ids" : patient_ids,

            "target" : patient["target"],

            "num_boxes": len(
                patient["bounding_boxes"]
            )
        })

    return pd.DataFrame(records)
"""
    Generate overall dataset statistics.

    Parameters
    ----------
    labels : dict
        Output of load_labels().

    Returns
    -------
    dict
        Dictionary containing dataset statistics.

"""
from collections import Counter
from typing import Dict
import pandas as pd

def generate_dataset_statistics(labels: Dict)->Dict:

    total_patients = len(labels)

    positive_cases = 0
    negative_cases = 0

    total_bounding_boxes = 0
    multiple_bbox_images = 0

    images_per_patients = []

    for patient in labels.values():
        target = patient["target"]
        boxes = patient["bounding_boxes"]

        if target == 1:
            positive_cases += 1
        else:
            negative_cases += 1

        total_bounding_boxes += len(boxes)

        if len(boxes) > 1:
            multiple_bbox_images += 1

        images_per_patients.append(1)

    stats = {
        "Total Patients" : total_patients,

        "Total Images" : len(images_per_patients),

        "Positive Cases" : positive_cases,

        "Negative cases" : negative_cases,

        "Average Images Per Patient" : round(sum(images_per_patients)/total_patients , 2),

        "Total Bounding Boxes" : total_bounding_boxes,

        "Images with Multiple Bounding Boxes" : multiple_bbox_images,
    }

    # return stats

    # Tomorrow we'll generate 100+ statistics.Instead, I'd return a pandas DataFrame.

    return pd.DataFrame(
        stats.items(),
        columns=['Statistic','Value']
    )
"""
    Analyze positive and negative class distribution.

    Parameters
    ----------
    labels : dict
        Output from load_labels().

    Returns
    -------
    pd.DataFrame
        Class counts and percentages.

"""
from typing import Dict
import pandas as pd

def analyze_class_distribution(labels: Dict)->pd.DataFrame:
    total = len(labels)

    positive = sum(
        patient["target"]
        for patient in labels.values()
    )

    negative = total - positive

    df = pd.DataFrame({
        "Class" : [
            "Positive" ,
            "Negative"
        ],

        "Count" : [
            positive,
            negative
         ]
    })

    df["Percentage"] = (
        df["Count"] / total * 100
    ).round(2)

    return df

def recommend_class_weighting(class_df: pd.DataFrame) -> str:
    """
    Recommend whether class weighting is needed.
    """

    positive_percentage = class_df.loc[
        class_df["Class"] == "Positive",
        "Percentage"
    ].iloc[0]

    if positive_percentage < 40:
        return (
            "Class imbalance detected. "
            "Consider weighted loss or focal loss."
        )

    return (
        "Class distribution is reasonably balanced."
    )
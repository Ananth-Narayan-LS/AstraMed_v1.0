"""
metrics.py

Evaluation metrics for AstraMed binary pneumonia classification.
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

def calculate_metrics(
        targets,
        probabilities,
        threshold = 0.5,
):
    """
    Calculate classification metrics.

    Parameters
    ----------
    targets : array-like
        Ground-truth binary labels.

    probabilities : array-like
        Predicted probabilities for pneumonia.

    threshold : float
        Probability threshold used to convert
        probabilities into binary predictions.

    Returns
    -------
    dict
        Dictionary containing classification metrics
        and confusion matrix.
    """
    # convert inputs to numpy arrays
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)

    # convert probabalititoes into binary predictions
    predictions = (
        probabilities >= threshold
    ).astype(int)

    # classification metrics
    accuracy = accuracy_score(
        targets,
        predictions
    )

    precision = precision_score(
        targets,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0
    )

    # ROC-AUC uses probabilities,
    # NOT thresholded predictions
    try:
        roc_auc = roc_auc_score(
            targets,
            probabilities
        )
    except ValueError:
        roc_auc = float("nan")

    # confusion matrix 
    cm = confusion_matrix(
        targets,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }
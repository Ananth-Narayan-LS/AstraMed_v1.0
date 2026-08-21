

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from src.data.dicom_loader import load_dicom

# Detect unusually small or large image dimensions.

def detect_dimension_outliers(
        image_df : pd.DataFrame
) -> dict:
    results = {}

    for column in ["width","height"]:
        q1 = image_df[column].quantile(0.25)
        q3 = image_df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = image_df[
            (image_df[column] < lower ) |
            (image_df[column] > upper ) 
        ]

        results[column] = {
            "lower_bound" : lower,
            "upper_bound" : upper,
            "outliers" : outliers,
        }

    return results

# Detect unusually large bounding boxes using IQR.
# looking for unusually large annotations, because extremely small boxes are not necessarily errors.

def detect_bbox_outliers(
        bbox_df : pd.DataFrame
) -> pd.DataFrame:

    q1 = bbox_df["area"].quantile(0.25)
    q3 = bbox_df["area"].quantile(0.75)

    iqr = q3 - q1

    upper = q3 + 1.5 * iqr

    outliers = bbox_df[
        bbox_df["area"] > upper 
    ].copy()

    return outliers

# Calculate missing metadata counts and percentages.

def detect_missing_metadata(
        metadata_df : pd.DataFrame
) -> pd.DataFrame:

    result = pd.DataFrame({
        "missing_count" : metadata_df.isna().sum(),

        "missing_percentage" : (
            metadata_df.isna().mean() * 100
        ).round(2)
    })

    return result.sort_values(
        "missing_percentage", ascending=False
    )

# Check whether sampled DICOM files can be decoded successfully.

def check_dicom_integrity(
        image_dir : Path,
        sample_size : int = 1000,
        random_state : int = 42
) -> pd.DataFrame:

    image_paths = list(image_dir.glob("*.dcm"))

    rng = np.random.default_rng(random_state)

    if len(image_paths) > sample_size:
        indices = rng.choice(
            len(image_paths),
            size = sample_size,
            replace=False
        )

        image_paths = [ image_paths[i] for i in indices ]

    records = []

    for image_path in image_paths:
        try:
            image, _ = load_dicom(image_path)
            records.append({
                "patient_id" : image_path.stem,
                "path" : str(image_path),
                "valid" : True,
                "error" : None
            })
        except Exception as e:

            records.append({
                "patient_id" : image_path.stem,
                "path" : str(image_path),
                "valid" : False,
                "error" : str(e)
            })
    return pd.DataFrame(records)

# Detect images with extremely low pixel variation.

def detect_blank_images(
        image_dir : Path,
        sample_size : int = 1000,
        random_state : int = 42,
        std_threshold : float = 5.0
) -> pd.DataFrame:
    image_paths = list(image_dir.glob("*.dcm"))

    rng = np.random.default_rng(random_state)

    if len(image_paths) > sample_size:
        indices = rng.choice(
            len(image_paths),
            size = sample_size,
            replace=False
        )

        image_paths = [ image_paths[i] for i in indices ]

    records = []

    for image_path in image_paths:
        try:
            image , _ = load_dicom(image_path)

            std = float(np.std(image))

            records.append({
                "patient_id" : image_path.stem,

                "std_intensity" : std,

                "is_blank"  :std < std_threshold
            })

        except Exception:
            records.append({
                "patient_id" : image_path.stem,

                "std_intensity" : np.nan,

                "is_blank" : False
            })

    return pd.DataFrame(records)
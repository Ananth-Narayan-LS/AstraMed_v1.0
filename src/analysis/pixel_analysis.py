
"""
    Compute pixel intensity statistics for a sample of DICOM images.

    Parameters
    ----------
    image_dir : Path
        Directory containing DICOM images.

    sample_size : int
        Number of images to analyze.

    Returns
    -------
    pd.DataFrame
        Pixel intensity statistics.

"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dicom_loader import load_dicom

def analyze_pixel_statistics(
    image_dir: Path,
    sample_size: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:

    image_paths = list(image_dir.glob("*.dcm"))

    rng = np.random.default_rng(random_state)

    if len(image_paths) > sample_size:

        selected_indices = rng.choice(
            len(image_paths),
            size=sample_size,
            replace=False
        )

        image_paths = [
            image_paths[i]
            for i in selected_indices
        ]

    records = []

    for image_path in image_paths:

        image, _ = load_dicom(image_path)

        records.append({

            "patient_id": image_path.stem,

            "mean_intensity": np.mean(image),

            "median_intensity": np.median(image),

            "std_intensity": np.std(image),

            "min_intensity": np.min(image),

            "max_intensity": np.max(image),

            "dynamic_range": np.ptp(image)
        })

    return pd.DataFrame(records)

def find_extreme_images(pixel_df : pd.DataFrame):
    # return the darkest and brightest images

    darkest = pixel_df.nsmallest(
        5,
        "mean_intensity"
    )

    brighest = pixel_df.nlargest(
        5,
        "mean_intensity"
    )

    return darkest, brighest
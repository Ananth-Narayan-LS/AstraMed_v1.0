
"""
    Extract useful DICOM metadata from a sample of images.

    Parameters
    ----------
    image_dir : Path
        Directory containing DICOM files.

    sample_size : int
        Number of images to analyze.

    random_state : int
        Seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        Extracted metadata.

"""

from pathlib import Path

import pandas as pd
import pydicom

def extract_metadata(
        image_dir : Path,
        sample_size : int = 1000,
        random_state : int = 42
)-> pd.DataFrame:
    
    image_paths = list(image_dir.glob("*.dcm"))

    # Reproducible random sampling
    if len(image_paths) > sample_size:
        sample = pd.Series(image_paths).sample(
            n = sample_size,
            random_state = random_state
        )
        image_paths = sample.tolist()

    records = []

    for image_path in image_paths:
        dicom = pydicom.dcmread(
            image_path,
            stop_before_pixels=True
        )

        pixel_spacing = getattr(
            dicom,
            "PixelSpacing",
            [None, None] 
        )

        records.append({
            "patient_id" : image_path.stem,

            "age": convert_age_to_years(
                getattr(
                    dicom,
                    "PatientAge",
                    None
                )
            ),

            "sex" : getattr(
                dicom,
                "PatientSex",
                "Unkown"
            ),

            "view_position" : getattr(
                dicom,
                "ViewPosition",
                "Unkown"
            ),

            "pixel_spacing_x" : pixel_spacing[0],

            "pixel_spacing_y" : pixel_spacing[1],

            "manufacturer" : getattr(
                dicom,
                "Manufacturer",
                "Unkown"
            ),

            "modality" : getattr(
                dicom,
                "Modality",
                "Unkown"
            )
        })

    return pd.DataFrame(records)

def convert_age_to_years(age):

    if pd.isna(age):
        return None

    age = str(age).strip().upper()

    if len(age) < 4:
        return None

    value = age[:-1]

    unit = age[-1]

    try:

        value = int(value)

    except ValueError:

        return None

    if unit == "Y":
        return value

    elif unit == "M":
        return round(value / 12, 1)

    elif unit == "W":
        return round(value / 52, 1)

    elif unit == "D":
        return round(value / 365, 1)

    return None
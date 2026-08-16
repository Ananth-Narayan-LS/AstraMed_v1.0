

"""
    Extract image properties from all DICOM files.

"""

from pathlib import Path

import pandas as pd
import pydicom

def analyze_image_properties(image_dir : Path) -> pd.DataFrame:
    records = []

    for file in image_dir.glob("*.dcm"):

        dicom = pydicom.dcmread(file , stop_before_pixels=True)

        rows = dicom.Rows
        cols = dicom.Columns

        aspect_ratio = round(cols / rows , 3)

        pixel_spacing = getattr(
            dicom,
            "PixelSpacing",
            [None, None]
        )

        view_position = getattr(
            dicom,
            "ViewPosition",
            "Unkown"
        )

        records.append({
            "patient_id" : file.stem,

            "height" : rows,

            "width" : cols,

            "aspect_ratio" : aspect_ratio,

            "pixel_spacing_x" : pixel_spacing[0],

            "pixel_spacing_y" : pixel_spacing[1],

            "view_position" : view_position
        })

    return pd.DataFrame(records)
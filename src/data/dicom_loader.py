
'''
    Read a DICOM file and return its image data and metadata.
'''

from pathlib import Path
from typing import Dict, Tuple , Union

import numpy as np
import pydicom

def load_dicom(
        path : Union[str, Path],
        return_metadata : bool = True
)-> Union[np.ndarray , Tuple[np.ndarray, Dict]]:
    """
    Load a DICOM image and optionally return metadata.

    Parameters
    ----------
    path : str or Path
        Path to the DICOM (.dcm) file.

    return_metadata : bool, default=True
        If True, returns both image and metadata.
        If False, returns only the image.

    Returns
    -------
    image : numpy.ndarray
        Pixel array extracted from the DICOM image.

    metadata : dict (optional)
        Dictionary containing useful DICOM metadata.

    Raises
    ------
    FileNotFoundError
        If the DICOM file does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"DICOM FILE NOT FOUND : {path}")

    # READ DICOM FILE
    dicom = pydicom.dcmread(path)

    # EXTRACT IMAGES AS NUMPY ARRAY
    image = dicom.pixel_array.astype(np.float32)

    if not return_metadata:
        return image

    #EXTRACT USEFUL METADATA
    metadata = {
        "patient_id" : getattr(dicom,"PatientID",None),
        "modality" : getattr(dicom,"Modality",None),
        "rows" : getattr(dicom,"Rows",None),
        "columns" : getattr(dicom,"Rows",None),
        "photometric_interpretation" : getattr(
            dicom,
            "PhotometricInterpretation",
            None,
        ),
        "pixel_spacing": getattr(dicom, "PixelSpacing", None),
        "bits_stored": getattr(dicom, "BitsStored", None),
        "bits_allocated": getattr(dicom, "BitsAllocated", None),
    }

    return image, metadata
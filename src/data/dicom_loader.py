
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
        "PatientID" : getattr(dicom,"PatientID",None),
        "Modality" : getattr(dicom,"Modality",None),
        "Rows" : getattr(dicom,"Rows",None),
        "Columns" : getattr(dicom,"Columns",None),
        "PhotometricInterpretation" : getattr(
            dicom,
            "PhotometricInterpretation",
            None,
        ),
        "PixelSpacing": getattr(dicom, "PixelSpacing", None),
        "BitsStored": getattr(dicom, "BitsStored", None),
        "BitsAllocated": getattr(dicom, "BitsAllocated", None),
        "image_shape":image.shape,
        "dtype":str(image.dtype),
    }

    return image, metadata
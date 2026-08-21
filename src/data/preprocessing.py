"""
    Preprocess a chest X-ray image.

    Steps
    -----
    1. Resize
    2. Normalize pixel values
    3. Return float32 NumPy array

    Parameters
    ----------
    image : np.ndarray
        Input image.

    target_size : tuple
        Desired output size.

    Returns
    -------
    np.ndarray
        Preprocessed image.
    """


from typing import Tuple

import numpy as np
from PIL import Image

def preprocess_image(
        image: np.ndarray,
        target_size: Tuple[int, int] = (512, 512)
)-> np.ndarray:

    # Ensure image is grayscale
    if image.ndim != 2:
        raise ValueError(
            f"Expected a grayscale image with shape (H, W), got {image.shape}"
        )
    
    #Resize
    image = Image.fromarray(image)

    image = image.resize(
        target_size,
        Image.Resampling.BILINEAR 
    ) 

    image = np.array(image)

    #Normalise
    image = image.astype(np.float32)

    """
    image / 255 , Why?
    Because DICOM images are not 8-bit JPEGs.
    Some images have pixel ranges like 0 → 4095
    Others -1024 → 3071
    So, Min-Max normalization
    (image-min)/(max-min)
    works regardless of the original range.

    """
    # to handle division by zero i.e if an image has min == max
    if image.max() > image.min():

        image = (
            image - image.min()
        ) / (
            image.max() - image.min()
        )

    else:

        image = np.zeros_like(image)
        
    return image
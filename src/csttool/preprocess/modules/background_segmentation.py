"""
background_segmentation.py

Estimate brain mask with median Otsu.
https://docs.dipy.org/dev/examples_built/segmentation/brain_extraction_dwi_1.html#sphx-glr-examples-built-segmentation-brain-extraction-dwi-1-py
"""

import numpy as np
from dipy.segment.mask import median_otsu


def background_segmentation(
    data: np.ndarray,
    median_radius: int = 2,
    numpass: int = 1,
    autocrop: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate brain mask with median Otsu.

    Parameters
    ----------
    data : np.ndarray
        4D DWI data array.
    median_radius : int, optional
        Radius of the median filter. Default is 2.
    numpass : int, optional
        Number of passes for the median filter. Default is 1.
    autocrop : bool, optional
        Whether to autocrop the data. Default is True.

    Returns
    -------
    masked_data : np.ndarray
        4D masked DWI data array.
    mask : np.ndarray
        3D binary brain mask array.
    """
    masked_data, mask = median_otsu(
        data,
        median_radius=median_radius,
        numpass=numpass,
        autocrop=autocrop
    )

    return masked_data, mask
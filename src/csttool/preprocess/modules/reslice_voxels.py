"""
reslice_voxels.py

Reslice DWI volumes to a common voxel size.
https://docs.dipy.org/dev/examples_built/preprocessing/reslice_datasets.html
"""

import numpy as np
from dipy.align.reslice import reslice


def reslice_voxels(
    data: np.ndarray,
    affine: np.ndarray,
    voxel_size: tuple[float, float, float],
    new_voxel_size: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reslice DWI volumes to a common voxel size.

    Parameters
    ----------
    data : np.ndarray
        3D or 4D DWI data array.
    affine : np.ndarray
        4x4 affine transformation matrix.
    voxel_size : tuple[float, float, float]
        Current voxel size (zooms) in mm.
    new_voxel_size : tuple[float, float, float]
        Target voxel size (zooms) in mm.

    Returns
    -------
    resliced_data : np.ndarray
        Resliced data array.
    resliced_affine : np.ndarray
        Updated 4x4 affine transformation matrix.
    """
    resliced_data, resliced_affine = reslice(
        data,
        affine,
        voxel_size,
        new_voxel_size
    )

    return resliced_data, resliced_affine
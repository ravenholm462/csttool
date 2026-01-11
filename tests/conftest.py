import pytest
import numpy as np
import nibabel as nib
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import create_nifti_header

@pytest.fixture
def synthetic_affine():
    """Returns a simple identity affine for testing."""
    return np.eye(4)

@pytest.fixture
def synthetic_image_data():
    """Returns a small 10x10x10 synthetic 3D image."""
    data = np.zeros((10, 10, 10), dtype=np.float32)
    # Create some "features"
    data[2:8, 2:8, 2:8] = 1.0  # Cube in the center
    return data

@pytest.fixture
def synthetic_nifti(synthetic_image_data, synthetic_affine):
    """Returns a synthetic nibabel Nifti1Image."""
    return nib.Nifti1Image(synthetic_image_data, synthetic_affine)

@pytest.fixture
def synthetic_tractogram(synthetic_nifti, synthetic_affine):
    """Returns a simple synthetic StatefulTractogram."""
    # Create a few straight streamlines
    # Streamline 1: straight line along Z axis
    sl1 = np.array([[5, 5, 2], [5, 5, 3], [5, 5, 4], [5, 5, 5], [5, 5, 6], [5, 5, 7], [5, 5, 8]], dtype=np.float32)
    # Streamline 2: slightly offset
    sl2 = np.array([[4, 4, 2], [4, 4, 3], [4, 4, 4], [4, 4, 5], [4, 4, 6], [4, 4, 7], [4, 4, 8]], dtype=np.float32)
    
    streamlines = [sl1, sl2]
    
    sft = StatefulTractogram(streamlines, synthetic_nifti, Space.RASMM)
    return sft

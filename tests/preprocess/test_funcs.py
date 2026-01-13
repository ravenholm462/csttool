import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
from csttool.preprocess.funcs import (
    is_dicom_dir,
    convert_to_nifti,
    denoise_nlmeans,
    background_segmentation,
    suppress_gibbs_oscillations,
)

def test_is_dicom_dir(tmp_path):
    """Test DICOM directory detection."""
    # Case 1: Empty directory
    assert is_dicom_dir(tmp_path) is False
    
    # Case 2: Directory with non-dicom file
    (tmp_path / "test.txt").touch()
    assert is_dicom_dir(tmp_path) is False
    
    # Case 3: Directory with DICOM file
    (tmp_path / "image.dcm").touch()
    assert is_dicom_dir(tmp_path) is True
    
    # Case 4: Non-existent path
    assert is_dicom_dir(tmp_path / "nonexistent") is False

@patch('csttool.preprocess.funcs.convert_dicom')
def test_convert_to_nifti(mock_convert, tmp_path):
    """Test DICOM to NIfTI conversion wrapper."""
    # Setup mocks
    out_dir = tmp_path / "out"
    dicom_dir = tmp_path / "dicom"
    
    # Mock return from dicom2nifti
    mock_convert.dicom_series_to_nifti.return_value = {
        "NII_FILE": str(out_dir / "nifti" / "dicom.nii.gz"),
        "BVAL_FILE": str(out_dir / "nifti" / "dicom.bval"),
        "BVEC_FILE": str(out_dir / "nifti" / "dicom.bvec")
    }
    
    # Mock os.makedirs to prevent FileNotFoundError
    with patch('os.path.exists', return_value=True), \
         patch('os.makedirs'):
        nii, bval, bvec = convert_to_nifti(dicom_dir, out_dir)
    
    assert nii.endswith("dicom.nii.gz")
    assert bval.endswith("dicom.bval")
    assert bvec.endswith("dicom.bvec")
    mock_convert.dicom_series_to_nifti.assert_called_once()

def test_denoise_nlmeans(synthetic_image_data):
    """Test NLMEANS denoising wrapper."""
    # Create 4D data (add time dimension)
    data_4d = np.repeat(synthetic_image_data[..., np.newaxis], 2, axis=3)
    
    # We mock piesno to avoid long computation and return a known 'sigma'
    with patch('csttool.preprocess.funcs.piesno') as mock_piesno:
        # mock piesno return: (sigma_map, mask)
        mock_mask = np.zeros(data_4d.shape[:3], dtype=bool) # all noise??
        # Let's say center is signal, outside is noise.
        # synthetic_image_data has cube in center.
        # Let's return a mask where noise is outside.
        # piesno returns 'noise_mask' (True where noise is).
        mock_piesno.return_value = (np.ones_like(synthetic_image_data)*0.1, ~synthetic_image_data.astype(bool))
        
        denoised, mask = denoise_nlmeans(data_4d, N=1, visualize=False)
        
        assert denoised.shape == data_4d.shape
        assert mask.shape == data_4d.shape[:3]
        # Check that it returns numpy array
        assert isinstance(denoised, np.ndarray)

def test_background_segmentation(synthetic_image_data, synthetic_gtab):
    """Test median_otsu wrapper."""
    # Create 4D data matching gtab length (1 b0 + 6 dwis = 7)
    data_4d = np.zeros(synthetic_image_data.shape + (7,))
    # Fill with some data
    data_4d[..., 0] = synthetic_image_data # b0
    data_4d[..., 1:] = synthetic_image_data[..., np.newaxis] * 0.5 # dwi
    
    masked_data, mask = background_segmentation(
        data_4d, 
        synthetic_gtab,
        median_radius=1,
        numpass=1,
        visualize=False
    )
    
    assert masked_data.shape == data_4d.shape
    assert mask.shape == data_4d.shape[:3]
    assert mask.dtype == bool or mask.dtype == np.uint8


def test_suppress_gibbs_oscillations_calls_dipy():
    """Test Gibbs unringing wrapper calls dipy with slice_axis."""
    data_4d = np.zeros((6, 6, 6, 2), dtype=np.float32)

    with patch('dipy.denoise.gibbs.gibbs_removal') as mock_gibbs:
        mock_gibbs.return_value = data_4d
        result = suppress_gibbs_oscillations(data_4d, slice_axis=1)

    mock_gibbs.assert_called_once_with(data_4d, slice_axis=1, num_processes=-1)
    assert result is data_4d

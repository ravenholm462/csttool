import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from csttool.extract.modules.registration import (
    compute_affine_registration,
    compute_syn_registration,
    load_mni_template
)

def test_load_mni_template():
    """Test MNI template loading (mocked)."""
    # Fix: Patch nilearn.datasets.load_mni152_template instead of non-existent local functions
    with patch('csttool.extract.modules.registration.datasets.load_mni152_template') as mock_load:
        # Mock what dipy returns (img, data, affine) ?? or just img?
        # csttool implementation: mni_img = load_mni152_template()
        # then mni_data = mni_img.get_fdata(), mni_affine = mni_img.affine
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.zeros((10,10,10))
        mock_img.affine = np.eye(4)
        mock_load.return_value = mock_img
        
        img, data, affine = load_mni_template(contrast="T1")
        
        assert data.shape == (10,10,10)
        assert affine.shape == (4,4)

def test_compute_affine_registration(synthetic_image_data, synthetic_affine):
    """Test affine registration wrapper."""
    # We mock the internal AffineRegistration to avoid compute
    with patch('csttool.extract.modules.registration.AffineRegistration') as mock_reg_class:
        mock_reg_inst = mock_reg_class.return_value
        # Mock optimize method
        mock_map = MagicMock()
        mock_map.affine = np.eye(4)
        mock_reg_inst.optimize.return_value = mock_map
        
        static = synthetic_image_data
        moving = synthetic_image_data # Same image, should register perfectly
        
        result = compute_affine_registration(
            static, synthetic_affine,
            moving, synthetic_affine,
            level_iters=[10, 10, 5]
        )
        
        assert result.affine.shape == (4,4)
        mock_reg_inst.optimize.assert_called()

def test_compute_syn_registration(synthetic_image_data, synthetic_affine):
    """Test SyN registration wrapper."""
    with patch('csttool.extract.modules.registration.SymmetricDiffeomorphicRegistration') as mock_syn_class:
        mock_syn_inst = mock_syn_class.return_value
        
        # Mock optimize output (mapping)
        mock_mapping = MagicMock()
        mock_syn_inst.optimize.return_value = mock_mapping
        
        result = compute_syn_registration(
            synthetic_image_data, synthetic_affine,
            synthetic_image_data, synthetic_affine,
            level_iters=[1, 1, 1],
            prealign=np.eye(4)
        )
        
        mock_syn_inst.optimize.assert_called()
        assert result == mock_mapping

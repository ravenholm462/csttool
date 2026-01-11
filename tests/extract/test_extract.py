import pytest
import numpy as np
import nibabel as nib
from csttool.extract.modules.create_roi_masks import create_cst_roi_masks

# Mocking csttool.extract.modules.create_roi_masks.CST_ROI_CONFIG locally if needed, 
# but usually we can test the function by passing a custom config if the function supports it.
# looking at usage in test_extract_modules.py:
# masks = create_cst_roi_masks(..., roi_config=atlas_result['roi_config'], ...)
# So we can pass a custom roi_config.

def test_create_cst_roi_masks_simple(synthetic_affine):
    """Test mask creation with synthetic boolean arrays."""
    shape = (10, 10, 10)
    
    # Create fake warped atlases
    # Subcortical: Label 16 is brainstem
    subcortical = np.zeros(shape, dtype=np.int32)
    subcortical[2:5, 2:5, 2:5] = 16 
    
    # Cortical: Label 7 is motor (precentral)
    cortical = np.zeros(shape, dtype=np.int32)
    cortical[6:9, 6:9, 6:9] = 7
    
    # Fake ROI config matching the labels we used
    roi_config = {
        'brainstem': {'atlas': 'subcortical', 'label': 16},
        'motor_left': {'atlas': 'cortical', 'label': 7}, 
        # For this test we can pretend motor_right is same label or different, 
        # usually separation happens by split.
        'motor_right': {'atlas': 'cortical', 'label': 7} 
    }
    
    # We need to mock separate_hemispheres to avoid complexity? 
    # Or just let it run. separate_hemispheres usually splits by x=0 (relative to midline).
    # If our data is 10x10x10. Midline could be index 5.
    
    # Let's mock separate_hemispheres inside the module to just return the mask as is for simplify testing
    # Or we can just ensure our synthetic data spans left/right.
    # We'll rely on functional mocking if possible, or simple inputs.
    
    masks = create_cst_roi_masks(
        warped_cortical=cortical,
        warped_subcortical=subcortical,
        subject_affine=synthetic_affine,
        roi_config=roi_config,
        save_masks=False,
        # We might need to mock internal calls if they do heavy IO or such.
        # create_cst_roi_masks does not save unless save_masks=True.
    )
    
    assert 'brainstem' in masks
    assert 'motor_left' in masks
    assert 'motor_right' in masks
    
    # Check that masks are boolean/binary
    assert masks['brainstem'].dtype == bool or masks['brainstem'].dtype == np.uint8
    assert np.sum(masks['brainstem']) > 0

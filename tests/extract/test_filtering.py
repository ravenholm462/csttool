import pytest
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.tracking.streamline import Streamlines
from csttool.extract.modules.endpoint_filtering import filter_streamlines_by_endpoints
from csttool.extract.modules.passthrough_filtering import extract_cst_passthrough

def test_extract_cst_passthrough_logic(synthetic_affine):
    """Test passthrough filtering logic."""
    # Create synthetic streamlines
    # SL1: passes through ROI A and ROI B
    sl1 = np.array([[10, 10, 0], [10, 10, 5], [10, 10, 10]]) 
    
    # SL2: passes through ROI A only
    sl2 = np.array([[10, 10, 0], [10, 10, 5], [20, 20, 20]]) 
    
    streamlines = Streamlines([sl1, sl2])
    
    shape = (30, 30, 30)
    # Brainstem at [10,10,0]
    brainstem = np.zeros(shape, dtype=np.uint8)
    brainstem[10, 10, 0] = 1 
    
    # Motor Left at [10,10,10]
    motor_left = np.zeros(shape, dtype=np.uint8)
    motor_left[10, 10, 10] = 1 
    
    # Motor Right (unused here)
    motor_right = np.zeros(shape, dtype=np.uint8)
    
    masks = {
        'brainstem': brainstem,
        'motor_left': motor_left,
        'motor_right': motor_right
    }
    
    # Extract
    result = extract_cst_passthrough(
        streamlines,
        masks,
        synthetic_affine,
        min_length=0, # disable length filtering for this short test
        max_length=1000,
        verbose=False
    )
    
    # SL1 hits brainstem AND motor_left -> Left CST
    # SL2 hits brainstem only -> dropped
    
    assert len(result['cst_left']) == 1
    assert np.allclose(result['cst_left'][0], sl1)
    assert len(result['cst_right']) == 0

def test_filter_streamlines_by_endpoints_logic(synthetic_affine):
    """Test endpoint filtering (stricter)."""
    # SL1: starts in A, ends in B
    sl1 = np.array([[10, 10, 0], [10, 10, 5], [10, 10, 10]])
    
    # SL2: Passes through A and B but continues
    sl2 = np.array([[10, 10, -5], [10, 10, 0], [10, 10, 10], [10, 10, 15]])
    
    streamlines = Streamlines([sl1, sl2])
    
    shape = (30, 30, 30)
    roi_a_mask = np.zeros(shape, dtype=np.uint8)
    roi_a_mask[10, 10, 0] = 1
    
    roi_b_mask = np.zeros(shape, dtype=np.uint8)
    roi_b_mask[10, 10, 10] = 1
    
    # filter_streamlines_by_endpoints signature: streamlines, roi_a, roi_b, affine
    filtered, indices = filter_streamlines_by_endpoints(
        streamlines,
        roi_a_mask,
        roi_b_mask,
        synthetic_affine,
        verbose=False
    )
    
    # SL1 starts/ends exactly in ROIs -> Keep
    # SL2 endpoints are [-5] and [15], neither in ROI -> Drop
    
    assert len(filtered) == 1
    assert np.allclose(filtered[0], sl1)

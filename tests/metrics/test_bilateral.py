import pytest
import numpy as np
from csttool.metrics.modules.bilateral_analysis import compare_bilateral_cst

def test_compare_bilateral_cst_asymmetry(synthetic_affine):
    """Test asymmetry index calculation."""
    # Create mock metric dictionaries
    left_metrics = {
        'morphology': {
            'n_streamlines': 10,
            'tract_volume': 100.0,
            'mean_length': 10.0
        },
        'fa': {'mean': 0.5, 'std': 0.05},
        'md': {'mean': 0.0007, 'std': 0.0001}
    }
    
    right_metrics = {
        'morphology': {
            'n_streamlines': 5,
            'tract_volume': 50.0,
            'mean_length': 10.0
        },
        'fa': {'mean': 0.5, 'std': 0.05},
        'md': {'mean': 0.0007, 'std': 0.0001}
    }
    
    comparison = compare_bilateral_cst(left_metrics, right_metrics)
    
    assert 'asymmetry' in comparison
    assert 'volume' in comparison['asymmetry']
    
    # Check value roughly
    ai = comparison['asymmetry']['volume']['laterality_index']
    # AI = (100-50)/(100+50) = 50/150 = 0.333
    assert 0.3 < ai < 0.4

def test_compare_bilateral_cst_empty(synthetic_affine):
    """Test bilateral comparison with empty sides."""
    left_metrics = {
        'morphology': {
            'n_streamlines': 0, 'tract_volume': 0.0, 'mean_length': 0.0
        },
        'fa': {'mean': 0.0, 'std': 0.0}, 'md': {'mean': 0.0, 'std': 0.0}
    }
    right_metrics = {
        'morphology': {
            'n_streamlines': 0, 'tract_volume': 0.0, 'mean_length': 0.0
        },
        'fa': {'mean': 0.0, 'std': 0.0}, 'md': {'mean': 0.0, 'std': 0.0}
    }
    
    comparison = compare_bilateral_cst(left_metrics, right_metrics)
    
    # Should handle it without crashing (NaN or 0)
    assert comparison['asymmetry']['volume']['laterality_index'] == 0.0

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


def test_compare_bilateral_cst_localized_metrics(synthetic_affine):
    """Test localized metrics LI computation."""
    # Create mock metric dictionaries with localized metrics
    left_metrics = {
        'morphology': {
            'n_streamlines': 10,
            'tract_volume': 100.0,
            'mean_length': 10.0
        },
        'fa': {
            'mean': 0.5, 'std': 0.05,
            'pontine': 0.45, 'plic': 0.50, 'precentral': 0.55
        },
        'md': {
            'mean': 0.0007, 'std': 0.0001,
            'pontine': 0.0006, 'plic': 0.0007, 'precentral': 0.0008
        }
    }

    right_metrics = {
        'morphology': {
            'n_streamlines': 10,
            'tract_volume': 100.0,
            'mean_length': 10.0
        },
        'fa': {
            'mean': 0.5, 'std': 0.05,
            'pontine': 0.40, 'plic': 0.50, 'precentral': 0.55
        },
        'md': {
            'mean': 0.0007, 'std': 0.0001,
            'pontine': 0.0006, 'plic': 0.0007, 'precentral': 0.0008
        }
    }

    comparison = compare_bilateral_cst(left_metrics, right_metrics)

    # Check localized LI keys exist
    assert 'fa_pontine' in comparison['asymmetry']
    assert 'fa_plic' in comparison['asymmetry']
    assert 'fa_precentral' in comparison['asymmetry']
    assert 'md_pontine' in comparison['asymmetry']
    assert 'md_plic' in comparison['asymmetry']
    assert 'md_precentral' in comparison['asymmetry']

    # Check FA pontine LI: (0.45-0.40)/(0.45+0.40) = 0.05/0.85 ≈ 0.059
    fa_pont_li = comparison['asymmetry']['fa_pontine']['laterality_index']
    assert 0.05 < fa_pont_li < 0.07

    # FA plic and precentral should be symmetric (LI ≈ 0)
    assert abs(comparison['asymmetry']['fa_plic']['laterality_index']) < 0.01
    assert abs(comparison['asymmetry']['fa_precentral']['laterality_index']) < 0.01

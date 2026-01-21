
import pytest
import numpy as np
from csttool.ingest.modules.assess_quality import assess_acquisition_quality

def test_assess_quality_good_data():
    """Test with optimal acquisition parameters."""
    # 60 directions, b=1000/2000, 2mm isotropic
    bvals = np.concatenate([np.zeros(5), np.ones(30)*1000, np.ones(30)*2000])
    
    # Generate random unit vectors
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(60, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    bvecs = np.concatenate([np.zeros((5, 3)), vecs])
    
    voxel_size = (2.0, 2.0, 2.0)
    
    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert len(warnings) == 0

def test_assess_quality_low_directions():
    """Test with insufficient directions."""
    # 10 directions
    bvals = np.concatenate([np.zeros(1), np.ones(10)*1000])
    
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(10, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    bvecs = np.concatenate([np.zeros((1, 3)), vecs])
    
    voxel_size = (2.0, 2.0, 2.0)
    
    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert any(w[0] == "CRITICAL" for w in warnings)
    assert any("Only 10 gradient directions" in w[1] for w in warnings)

def test_assess_quality_high_bvalue():
    """Test with very high b-value."""
    bvals = np.array([3500.0] * 30)
    bvecs = np.zeros((30, 3)) # Dummy bvecs
    bvecs[:, 0] = 1
    
    voxel_size = (2.0, 2.0, 2.0)
    
    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert any(w[0] == "WARNING" and "High b-value" in w[1] for w in warnings)

def test_assess_quality_large_voxels():
    """Test with large voxels."""
    bvals = np.ones(30) * 1000
    bvecs = np.zeros((30, 3))
    bvecs[:, 0] = 1 # Dummy, unique doesn't matter for this check
    
    voxel_size = (3.0, 3.0, 3.0)
    
    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert any(w[0] == "WARNING" and "Large voxel size" in w[1] for w in warnings)

def test_assess_quality_anisotropic():
    """Test with anisotropic voxels."""
    bvals = np.ones(30) * 1000
    bvecs = np.zeros((30, 3))
    
    voxel_size = (1.0, 1.0, 5.0) # Ratio 5.0
    
    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert any(w[0] == "WARNING" and "Anisotropic voxels" in w[1] for w in warnings)

def test_assess_quality_json():
    """Test JSON-derived warnings."""
    bvals = np.ones(30) * 1000
    bvecs = np.zeros((30, 3))
    voxel_size = (2.0, 2.0, 2.0)
    
    json_data = {
        "EchoTime": 0.15, # 150ms
        "MultibandAccelerationFactor": 5
    }
    
    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size, bids_json=json_data)
    
    assert any("Long echo time" in w[1] for w in warnings)
    assert any("High multiband factor" in w[1] for w in warnings)


# Tests for extract_acquisition_metadata
from csttool.ingest.modules.assess_quality import extract_acquisition_metadata

def test_extract_acquisition_metadata_basic():
    """Test extraction of acquisition metadata from arrays."""
    bvals = np.array([0, 0, 1000, 1000, 2000, 2000])
    bvecs = np.array([
        [0, 0, 0], [0, 0, 0],
        [1, 0, 0], [0, 1, 0],
        [0, 0, 1], [1, 1, 0]
    ])
    voxel_size = (2.0, 2.0, 2.0)
    
    acq = extract_acquisition_metadata(bvecs, bvals, voxel_size)
    
    assert acq['b_values'] == [0, 1000, 2000]
    assert acq['n_volumes'] == 6
    assert acq['resolution_mm'] == [2.0, 2.0, 2.0]
    assert acq['field_strength_T'] is None  # No JSON provided
    assert acq['echo_time_ms'] is None


def test_extract_acquisition_metadata_with_json():
    """Test extraction with BIDS JSON sidecar."""
    bvals = np.array([0, 1000, 1000])
    bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    voxel_size = (2.0, 2.0, 2.0)
    
    bids_json = {
        'MagneticFieldStrength': 3.0,
        'EchoTime': 0.089  # 89ms in seconds
    }
    
    acq = extract_acquisition_metadata(bvecs, bvals, voxel_size, bids_json=bids_json)
    
    assert acq['field_strength_T'] == 3.0
    assert acq['echo_time_ms'] == 89.0


def test_extract_acquisition_metadata_cli_overrides():
    """Test that CLI overrides take precedence over JSON."""
    bvals = np.array([0, 1000])
    bvecs = np.array([[0, 0, 0], [1, 0, 0]])
    voxel_size = (2.0, 2.0, 2.0)
    
    bids_json = {
        'MagneticFieldStrength': 3.0,
        'EchoTime': 0.089
    }
    
    overrides = {
        'field_strength_T': 7.0,  # Override from CLI
        'echo_time_ms': 75.0
    }
    
    acq = extract_acquisition_metadata(bvecs, bvals, voxel_size, bids_json=bids_json, overrides=overrides)
    
    # CLI overrides should win
    assert acq['field_strength_T'] == 7.0
    assert acq['echo_time_ms'] == 75.0

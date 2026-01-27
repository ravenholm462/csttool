
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
    # With multi-shell detection, should only have INFO messages
    assert all(sev in ["INFO"] for sev, _ in warnings)
    # No CRITICAL or WARNING messages
    assert not any(sev in ["CRITICAL", "WARNING"] for sev, _ in warnings)

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
    # With ratio 5.0 (> 2.0), should be CRITICAL
    assert any(w[0] == "CRITICAL" and "anisotropic" in w[1].lower() for w in warnings)

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


# Tests for enhanced functionality
def test_input_validation_mismatched_counts():
    """Test that mismatched bvals/bvecs counts are caught."""
    bvals = np.array([0, 1000, 1000])
    bvecs = np.array([[1, 0, 0], [0, 1, 0]])  # Only 2 vectors
    voxel_size = (2.0, 2.0, 2.0)

    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert any("count mismatch" in msg for _, msg in warnings)
    assert warnings[0][0] == "CRITICAL"


def test_input_validation_wrong_bvecs_shape():
    """Test that incorrect bvecs shape is caught."""
    bvals = np.array([0, 1000, 1000, 1000])
    bvecs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])  # Wrong shape: (4, 2) instead of (4, 3)
    voxel_size = (2.0, 2.0, 2.0)

    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert any("bvecs must have shape" in msg for _, msg in warnings)
    assert warnings[0][0] == "CRITICAL"


def test_negative_bvalues():
    """Test detection of negative b-values."""
    bvals = np.array([0, 1000, -500])  # Negative b-value
    bvecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    voxel_size = (2.0, 2.0, 2.0)

    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert any("Negative b-values" in msg for _, msg in warnings)
    assert any(sev == "CRITICAL" for sev, _ in warnings)


def test_extremely_high_bvalue():
    """Test detection of extremely high b-values."""
    bvals = np.array([0, 1000, 12000])  # Very high b-value
    bvecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    voxel_size = (2.0, 2.0, 2.0)

    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert any("Extremely high b-value" in msg for _, msg in warnings)


def test_b0_counting_and_distribution():
    """Test b=0 volume counting and gap detection."""
    # Create acquisition with sparse b=0s (indices 0, 50, 100)
    bvals = np.concatenate([[0], np.ones(49)*1000, [0], np.ones(49)*1000, [0]])
    rng = np.random.default_rng(42)
    bvecs = rng.normal(size=(len(bvals), 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    voxel_size = (2.0, 2.0, 2.0)

    warnings, metadata = assess_acquisition_quality(
        bvecs, bvals, voxel_size, return_metadata=True
    )

    assert metadata['b0_distribution']['n_volumes'] == 3
    assert metadata['b0_distribution']['max_gap'] == 50
    assert any("gap between b=0" in msg for _, msg in warnings)


def test_low_b0_count():
    """Test warning for insufficient b=0 volumes."""
    # Only 1 b=0 volume
    bvals = np.concatenate([[0], np.ones(30)*1000])
    rng = np.random.default_rng(42)
    bvecs = rng.normal(size=(len(bvals), 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    voxel_size = (2.0, 2.0, 2.0)

    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert any("Only 1 b=0" in msg and "Recommend" in msg for _, msg in warnings)


def test_no_b0_volumes():
    """Test critical warning when no b=0 volumes."""
    bvals = np.ones(30) * 1000
    bvecs = np.random.randn(30, 3)
    voxel_size = (2.0, 2.0, 2.0)

    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert any("No b=0 volumes" in msg for _, msg in warnings)
    assert any(sev == "CRITICAL" for sev, _ in warnings)


def test_near_zero_gradients():
    """Test detection of near-zero gradient vectors in DWI."""
    bvals = np.array([0, 1000, 1000, 1000])
    bvecs = np.array([
        [1, 0, 0],
        [0.99, 0, 0.01],
        [0.001, 0.001, 0.001],  # Near-zero gradient
        [0, 1, 0]
    ])
    voxel_size = (2.0, 2.0, 2.0)

    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert any("near-zero gradient" in msg.lower() for _, msg in warnings)
    assert any(sev == "CRITICAL" for sev, _ in warnings)


def test_shell_detection_single_shell():
    """Test single-shell detection."""
    bvals = np.concatenate([[0, 0], np.ones(30)*1000])
    rng = np.random.default_rng(42)
    bvecs = rng.normal(size=(len(bvals), 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    voxel_size = (2.0, 2.0, 2.0)

    warnings, metadata = assess_acquisition_quality(
        bvecs, bvals, voxel_size, return_metadata=True
    )

    assert len(metadata['shells']) == 1
    assert metadata['shells'][0]['bval'] == 1000


def test_shell_detection_multi_shell():
    """Test multi-shell detection with per-shell warnings."""
    # 2 b=0, 5 directions at b=1000, 30 directions at b=2000
    bvals = np.concatenate([[0, 0], np.ones(5)*1000, np.ones(30)*2000])
    rng = np.random.default_rng(42)
    # Generate unique vectors
    bvecs = rng.normal(size=(len(bvals), 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    voxel_size = (2.0, 2.0, 2.0)

    warnings, metadata = assess_acquisition_quality(
        bvecs, bvals, voxel_size, return_metadata=True, b0_threshold=50
    )

    assert len(metadata['shells']) == 2
    assert any(s['bval'] == 1000 for s in metadata['shells'])
    assert any(s['bval'] == 2000 for s in metadata['shells'])
    # Should warn about low direction count in b=1000 shell
    assert any("Shell b" in msg and "only" in msg for _, msg in warnings)
    # Should have INFO message about detecting shells
    assert any("Detected" in msg and "shell" in msg for _, msg in warnings)


def test_metadata_return():
    """Test that metadata is returned when requested."""
    bvals = np.array([0, 0, 1000, 1000, 1000])
    bvecs = np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,1]])
    bvecs = bvecs / np.linalg.norm(bvecs, axis=1, keepdims=True)
    voxel_size = (2.0, 2.0, 2.0)

    result = assess_acquisition_quality(bvecs, bvals, voxel_size, return_metadata=True)
    assert isinstance(result, tuple)
    warnings, metadata = result

    assert 'n_b0' in metadata
    assert 'n_dwi' in metadata
    assert 'n_directions' in metadata
    assert 'shells' in metadata
    assert 'b0_distribution' in metadata
    assert metadata['n_b0'] == 2
    assert metadata['n_dwi'] == 3


def test_metadata_not_returned_by_default():
    """Test that metadata is not returned by default."""
    bvals = np.array([0, 1000, 1000])
    bvecs = np.array([[1,0,0], [0,1,0], [0,0,1]])
    voxel_size = (2.0, 2.0, 2.0)

    result = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert isinstance(result, list)  # Should be list, not tuple


def test_bids_field_validation():
    """Test BIDS field presence tracking."""
    bvals = np.array([0, 1000, 1000])
    bvecs = np.array([[1,0,0], [0,1,0], [0,0,1]])
    voxel_size = (2.0, 2.0, 2.0)

    bids_json = {
        'PhaseEncodingDirection': 'j',
        'EchoTime': 0.08,
        # Missing TotalReadoutTime
    }

    warnings, metadata = assess_acquisition_quality(
        bvecs, bvals, voxel_size, bids_json=bids_json, return_metadata=True
    )

    assert metadata['bids_fields_present']['PhaseEncodingDirection'] == True
    assert metadata['bids_fields_present']['TotalReadoutTime'] == False
    assert any("Missing BIDS fields" in msg for _, msg in warnings)


def test_unusual_phase_encoding_direction():
    """Test warning for unusual phase encoding direction."""
    bvals = np.array([0, 1000, 1000])
    bvecs = np.array([[1,0,0], [0,1,0], [0,0,1]])
    voxel_size = (2.0, 2.0, 2.0)

    bids_json = {
        'PhaseEncodingDirection': 'xyz',  # Invalid
        'TotalReadoutTime': 0.05
    }

    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size, bids_json=bids_json)
    assert any("Unusual phase encoding" in msg for _, msg in warnings)


def test_voxel_volume_check():
    """Test large voxel volume warning."""
    bvals = np.ones(30) * 1000
    bvecs = np.random.randn(30, 3)
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    voxel_size = (3.0, 3.0, 3.0)  # Volume = 27 mm³

    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    assert any("voxel volume" in msg.lower() for _, msg in warnings)


def test_highly_anisotropic_voxels_critical():
    """Test critical warning for highly anisotropic voxels."""
    bvals = np.ones(30) * 1000
    bvecs = np.random.randn(30, 3)
    voxel_size = (1.0, 1.0, 3.0)  # Ratio = 3.0

    warnings = assess_acquisition_quality(bvecs, bvals, voxel_size)
    # Should have CRITICAL for ratio > 2.0
    assert any(sev == "CRITICAL" and "anisotropic" in msg.lower() for sev, msg in warnings)


def test_custom_b0_threshold():
    """Test using custom b0 threshold."""
    # Create data with b-values at 0, 75, 1000
    bvals = np.array([0, 75, 1000, 1000])
    bvecs = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0]])
    bvecs = bvecs / np.linalg.norm(bvecs, axis=1, keepdims=True)
    voxel_size = (2.0, 2.0, 2.0)

    # With threshold of 50, b=75 should be counted as DWI
    warnings1, metadata1 = assess_acquisition_quality(
        bvecs, bvals, voxel_size, b0_threshold=50.0, return_metadata=True
    )
    assert metadata1['n_b0'] == 1
    assert metadata1['n_dwi'] == 3

    # With threshold of 100, b=75 should be counted as b=0
    warnings2, metadata2 = assess_acquisition_quality(
        bvecs, bvals, voxel_size, b0_threshold=100.0, return_metadata=True
    )
    assert metadata2['n_b0'] == 2
    assert metadata2['n_dwi'] == 2

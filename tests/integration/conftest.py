import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
import struct
import shutil

@pytest.fixture
def integration_data_dir(tmp_path):
    """Create a directory for integration test data."""
    data_dir = tmp_path / "integration_data"
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def synthetic_dicom_dir(integration_data_dir):
    """
    Create a directory with minimal valid synthetic DICOM files.
    Constructs files with just enough structure to pass is_dicom_file checks
    and potentially be readable if we mocked the reader, 
    but for integration tests using the 'mocked' ingest pipeline, 
    we mainly need files that look like DICOMs on disk.
    
    If we were testing actual pydicom reading, we'd need a binary dicom generator,
    but since we are mocking pydicom in unit tests and might mock it here or 
    depend on the 'scan_only' behavior which just checks magic bytes for speed,
    we'll create files with the DICM magic.
    """
    dicom_dir = integration_data_dir / "study" / "series1"
    dicom_dir.mkdir(parents=True)
    
    # Create 5 dummy DICOM files
    for i in range(5):
        dcm_path = dicom_dir / f"img_{i:04d}.dcm"
        with open(dcm_path, "wb") as f:
            # Seek to 128 bytes (preamble)
            f.seek(128)
            # Write magic
            f.write(b"DICM")
            # Write some dummy content
            f.write(b"\x00" * 100)
            
    return dicom_dir

@pytest.fixture
def synthetic_nifti_data(integration_data_dir):
    """
    Create a valid synthetic NIfTI dataset (nii + bval + bvec) on disk.
    """
    nii_path = integration_data_dir / "test_dwi.nii.gz"
    
    # 10x10x10 volume with 6 timepoints (1 b0 + 5 dwi)
    shape = (10, 10, 10, 6)
    data = np.random.random(shape).astype(np.float32) * 100
    
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, nii_path)
    
    # Creating bvals (one b0, five b1000)
    bvals = np.array([0, 1000, 1000, 1000, 1000, 1000])
    bval_path = integration_data_dir / "test_dwi.bval"
    np.savetxt(bval_path, bvals, fmt='%d', newline=' ')
    
    # Creating bvecs
    # Must be unit vectors. Use simple axis-aligned vectors to avoid precision issues.
    bvecs = np.array([
        [0, 0, 0],       # b0
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],       # Repeat directions
        [0, 1, 0]
    ]).T # 3xN
    bvec_path = integration_data_dir / "test_dwi.bvec"
    np.savetxt(bvec_path, bvecs, fmt='%.8f')
    
    return nii_path

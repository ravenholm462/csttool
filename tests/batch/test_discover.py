import pytest
import os
from pathlib import Path
from csttool.batch.modules.discover import (
    discover_subjects, 
    detect_input_type, 
    sanitize_subject_id,
    find_bval_bvec
)

def test_sanitize_subject_id():
    assert sanitize_subject_id("sub-01") == "sub-01"
    assert sanitize_subject_id("Subject_123") == "Subject_123"
    with pytest.raises(ValueError):
        sanitize_subject_id("sub/01")
    with pytest.raises(ValueError):
        sanitize_subject_id("../evil")
    with pytest.raises(ValueError):
        sanitize_subject_id("")

def test_discover_bids_structure(tmp_path):
    # Case 1: Multi-session
    (tmp_path / "sub-01" / "ses-01" / "dwi").mkdir(parents=True)
    (tmp_path / "sub-01" / "ses-02" / "dwi").mkdir(parents=True)
    # Case 2: No session
    (tmp_path / "sub-02" / "dwi").mkdir(parents=True)
    # Case 3: Random directory (should be ignored if not matching pattern)
    (tmp_path / "other_dir").mkdir()
    
    subjects = discover_subjects(tmp_path)
    assert len(subjects) == 3
    
    sub01_ses01 = next(s for s in subjects if s['id'] == "sub-01" and s['session'] == "ses-01")
    assert sub01_ses01['dir'].name == "ses-01"
    
    sub02 = next(s for s in subjects if s['id'] == "sub-02")
    assert sub02['session'] is None

def test_detect_input_nifti(tmp_path):
    dwi_dir = tmp_path / "sub-01" / "dwi"
    dwi_dir.mkdir(parents=True)
    nifti_file = dwi_dir / "sub-01_dwi.nii.gz"
    nifti_file.touch()
    bval_file = dwi_dir / "sub-01_dwi.bval"
    bval_file.touch()
    
    itype, path = detect_input_type(tmp_path / "sub-01")
    assert itype == "nifti"
    assert path == nifti_file
    
    found_bval, found_bvec = find_bval_bvec(path)
    assert found_bval == bval_file
    assert found_bvec is None

def test_detect_input_dicom(tmp_path):
    dwi_dir = tmp_path / "sub-01" / "dwi"
    dwi_dir.mkdir(parents=True)
    (dwi_dir / "file1.dcm").touch()
    
    itype, path = detect_input_type(tmp_path / "sub-01")
    assert itype == "dicom"
    assert path == dwi_dir

def test_discovery_filters(tmp_path):
    (tmp_path / "sub-01").mkdir()
    (tmp_path / "sub-02").mkdir()
    (tmp_path / "sub-03").mkdir()
    
    # Include only 01 and 02
    subs = discover_subjects(tmp_path, include_subjects=["sub-01", "sub-02"])
    assert len(subs) == 2
    assert {s['id'] for s in subs} == {"sub-01", "sub-02"}
    
    # Exclude 01
    subs = discover_subjects(tmp_path, exclude_subjects=["sub-01"])
    assert len(subs) == 2
    assert "sub-01" not in {s['id'] for s in subs}

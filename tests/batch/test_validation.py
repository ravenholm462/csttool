import pytest
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from csttool.batch.modules.validation import validate_batch_preflight

@dataclass
class MockSubject:
    subject_id: str
    input_path: Path
    input_type: str = "nifti"
    session_id: Optional[str] = None
    bval_path: Optional[Path] = None
    bvec_path: Optional[Path] = None

@dataclass
class MockConfig:
    out: Path

def test_validate_missing_inputs(tmp_path):
    out_dir = tmp_path / "out"
    config = MockConfig(out=out_dir)
    
    subjects = [
        MockSubject(subject_id="sub-01", input_path=tmp_path / "missing.nii.gz")
    ]
    
    errors = validate_batch_preflight(subjects, config)
    # 1. missing nifti
    # 2. possibly environment check fails if dependencies not in test env (but we focus on INPUT)
    input_errors = [e for e in errors if e.category == "INPUT"]
    assert len(input_errors) > 0
    assert "NIfTI file not found" in input_errors[0].message

def test_validate_duplicate_subjects(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    config = MockConfig(out=out_dir)
    
    # Create real nifti to pass existence check
    nifti = tmp_path / "data.nii.gz"
    nifti.touch()
    
    subjects = [
        MockSubject(subject_id="sub-01", input_path=nifti),
        MockSubject(subject_id="sub-01", input_path=nifti) # Duplicate
    ]
    
    errors = validate_batch_preflight(subjects, config)
    config_errors = [e for e in errors if e.category == "CONFIG"]
    assert len(config_errors) > 0
    assert "Duplicate subject/session" in config_errors[0].message

def test_validate_invalid_names(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    config = MockConfig(out=out_dir)
    
    nifti = tmp_path / "data.nii.gz"
    nifti.touch()
    
    subjects = [
        MockSubject(subject_id="sub/evil", input_path=nifti)
    ]
    
    errors = validate_batch_preflight(subjects, config)
    input_errors = [e for e in errors if e.category == "INPUT"]
    assert any("Invalid subject ID" in e.message for e in input_errors)

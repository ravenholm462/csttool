import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
# Assuming we can import from csttool.ingest if it has a proper init?
# Let's check imports
from csttool.ingest import run_ingest_pipeline, scan_study

def test_scan_study_empty(tmp_path):
    """Test scanning empty directory."""
    series = scan_study(tmp_path)
    assert series == []

def test_scan_study_with_files(tmp_path):
    """Test scanning directory with mock DICOMs."""
    # We would need actual pydicom files to test properly or mock pydicom.dcmread
    # mocking is safer here.
    
    # Create a dummy file with magic bytes
    dcm_file = tmp_path / "test.dcm"
    with open(dcm_file, 'wb') as f:
        f.seek(128)
        f.write(b'DICM')
    
    series = scan_study(tmp_path)
    
    # scan_study as defined in scan_study.py only returns path, name, n_files
    assert len(series) == 1
    assert series[0]['n_files'] == 1

@patch('csttool.ingest.modules.convert_series.convert_dicom')
def test_run_ingest_pipeline(mock_convert, tmp_path):
    """Test ingestion pipeline execution."""
    # Mock conversion
    mock_convert.dicom_series_to_nifti.return_value = {
        'NII_FILE': str(tmp_path / "out.nii.gz")
    }
    
    # Create the dummy output file that the pipeline expects to exist
    (tmp_path / "out.nii.gz").touch()
    
    with patch('csttool.ingest.scan_study') as mock_scan:
        # Mock scan finding one series
        mock_scan.return_value = [{
            'series_number': 1,
            'description': 'DWI',
            'path': tmp_path,
            'files': []
        }]
        
        # Create output dir
        out_dir = tmp_path / "out"
        
        result = run_ingest_pipeline(
            tmp_path,
            out_dir,
            series_index=1
        )
        
        assert result['nifti_path'] is not None
        mock_convert.dicom_series_to_nifti.assert_called()

"""
Tests for report generation functions with metadata support.
"""
import pytest
import json
import tempfile
from pathlib import Path

from csttool.metrics.modules.reports import save_json_report


@pytest.fixture
def sample_comparison():
    """Create a minimal comparison dict for testing."""
    return {
        'left': {
            'morphology': {
                'n_streamlines': 100,
                'mean_length': 85.0,
                'tract_volume': 12000.0
            },
            'fa': {'mean': 0.45, 'std': 0.08}
        },
        'right': {
            'morphology': {
                'n_streamlines': 110,
                'mean_length': 88.0,
                'tract_volume': 13000.0
            },
            'fa': {'mean': 0.47, 'std': 0.07}
        },
        'asymmetry': {
            'volume': {'laterality_index': -0.04},
            'streamline_count': {'laterality_index': -0.05},
            'mean_length': {'laterality_index': -0.02},
            'fa': {'laterality_index': -0.02}
        }
    }


class TestSaveJsonReport:
    """Tests for save_json_report function."""
    
    def test_json_report_includes_version(self, sample_comparison):
        """Test that JSON report includes csttool version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_json_report(
                sample_comparison, 
                tmpdir, 
                "test_subject"
            )
            
            with open(json_path) as f:
                report = json.load(f)
            
            assert 'csttool_version' in report
            assert report['csttool_version'] is not None
    
    def test_json_report_includes_acquisition_metadata(self, sample_comparison):
        """Test that JSON report includes acquisition metadata when provided."""
        metadata = {
            'acquisition': {
                'protocol': 'Multi-shell',
                'b_values': [0, 1000, 2000],
                'n_directions': 64,
                'resolution': [2.0, 2.0, 2.0]
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_json_report(
                sample_comparison, 
                tmpdir, 
                "test_subject",
                metadata=metadata
            )
            
            with open(json_path) as f:
                report = json.load(f)
            
            assert 'acquisition' in report
            assert report['acquisition']['protocol'] == 'Multi-shell'
            assert report['acquisition']['b_values'] == [0, 1000, 2000]
            assert report['acquisition']['n_directions'] == 64
    
    def test_json_report_includes_processing_metadata(self, sample_comparison):
        """Test that JSON report includes processing metadata when provided."""
        metadata = {
            'processing': {
                'denoising_method': 'patch2self',
                'gibbs_correction': True,
                'motion_correction': False,
                'tracking_method': 'Deterministic (DTI)',
                'roi_approach': 'Atlas-to-Subject (HO)',
                'whole_brain_streamlines': 500000,
                'extraction_method': 'passthrough'
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_json_report(
                sample_comparison, 
                tmpdir, 
                "test_subject",
                metadata=metadata
            )
            
            with open(json_path) as f:
                report = json.load(f)
            
            assert 'processing' in report
            assert report['processing']['denoising_method'] == 'patch2self'
            assert report['processing']['whole_brain_streamlines'] == 500000
    
    def test_json_report_includes_qc_thresholds(self, sample_comparison):
        """Test that JSON report includes QC thresholds when provided."""
        metadata = {
            'qc_thresholds': {
                'fa_threshold': 0.15,
                'min_length': 30.0,
                'max_length': 200.0
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_json_report(
                sample_comparison, 
                tmpdir, 
                "test_subject",
                metadata=metadata
            )
            
            with open(json_path) as f:
                report = json.load(f)
            
            assert 'qc_thresholds' in report
            assert report['qc_thresholds']['fa_threshold'] == 0.15
            assert report['qc_thresholds']['min_length'] == 30.0
    
    def test_json_report_backward_compatible(self, sample_comparison):
        """Test that JSON report works without metadata (backward compatible)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_json_report(
                sample_comparison, 
                tmpdir, 
                "test_subject"
            )
            
            with open(json_path) as f:
                report = json.load(f)
            
            # Should have empty dicts for metadata sections
            assert 'acquisition' in report
            assert 'processing' in report
            assert 'qc_thresholds' in report
            assert report['acquisition'] == {}
            assert report['processing'] == {}
            assert report['qc_thresholds'] == {}
            
            # Core fields should still be present
            assert 'subject_id' in report
            assert 'processing_date' in report
            assert 'metrics' in report

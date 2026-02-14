"""
Tests for csttool.data.loader module.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import nibabel as nib
import numpy as np

from csttool.data.loader import (
    load_mni152_template,
    get_fmrib58_fa_path,
    get_fmrib58_fa_skeleton_path,
    get_harvard_oxford_path,
    get_user_data_dir,
    is_data_installed,
    DataNotInstalledError,
    _validate_if_needed,
    _load_validation_stamp,
    _save_validation_stamp,
)
from csttool.data.manifest import DATA_MANIFEST, verify_checksum, get_manifest_entry


class TestBundledData:
    """Tests for Tier 1 (bundled) data loading."""

    def test_load_mni152_template(self):
        """Test that MNI152 template loads successfully."""
        img, data, affine = load_mni152_template()

        # Check return types
        assert isinstance(img, nib.Nifti1Image)
        assert isinstance(data, np.ndarray)
        assert isinstance(affine, np.ndarray)

        # Check shapes
        assert data.ndim == 3
        assert affine.shape == (4, 4)

        # Check data is not empty
        assert data.size > 0
        assert np.any(data != 0)

    def test_mni152_checksum_matches_manifest(self):
        """Test that bundled MNI152 file matches manifest checksum."""
        from importlib.resources import files, as_file

        ref = files("csttool.data").joinpath("mni152", "MNI152_T1_1mm.nii.gz")
        with as_file(ref) as path:
            manifest_entry = get_manifest_entry("mni152/MNI152_T1_1mm.nii.gz")
            expected_sha256 = manifest_entry["sha256"]

            assert verify_checksum(path, expected_sha256), \
                "MNI152 bundled file checksum does not match manifest"


class TestUserFetchedData:
    """Tests for Tier 2 (user-fetched) data loading."""

    def test_get_fmrib58_fa_path_raises_when_missing(self):
        """Test that get_fmrib58_fa_path raises DataNotInstalledError when file missing."""
        with patch('csttool.data.loader._USER_DATA_DIR', Path("/nonexistent/path")):
            with pytest.raises(DataNotInstalledError) as exc_info:
                get_fmrib58_fa_path()

            assert "FMRIB58_FA template not found" in str(exc_info.value)
            assert "csttool fetch-data --accept-fsl-license" in str(exc_info.value)

    def test_get_fmrib58_fa_skeleton_path_raises_when_missing(self):
        """Test that get_fmrib58_fa_skeleton_path raises when file missing."""
        with patch('csttool.data.loader._USER_DATA_DIR', Path("/nonexistent/path")):
            with pytest.raises(DataNotInstalledError):
                get_fmrib58_fa_skeleton_path()

    def test_get_harvard_oxford_path_validates_atlas_name(self):
        """Test that get_harvard_oxford_path validates atlas_name parameter."""
        with pytest.raises(ValueError) as exc_info:
            get_harvard_oxford_path("invalid_name", "1mm")

        assert "cortical" in str(exc_info.value)
        assert "subcortical" in str(exc_info.value)

    def test_get_harvard_oxford_path_validates_resolution(self):
        """Test that get_harvard_oxford_path validates resolution parameter."""
        with pytest.raises(ValueError) as exc_info:
            get_harvard_oxford_path("cortical", "3mm")

        assert "1mm" in str(exc_info.value)
        assert "2mm" in str(exc_info.value)

    def test_get_harvard_oxford_path_raises_when_missing(self):
        """Test that get_harvard_oxford_path raises when atlas not found."""
        with patch('csttool.data.loader._USER_DATA_DIR', Path("/nonexistent/path")):
            with pytest.raises(DataNotInstalledError) as exc_info:
                get_harvard_oxford_path("cortical", "1mm")

            assert "Harvard-Oxford" in str(exc_info.value)
            assert "cortical" in str(exc_info.value)

    def test_get_user_data_dir(self):
        """Test that get_user_data_dir returns a valid path."""
        user_dir = get_user_data_dir()

        assert isinstance(user_dir, Path)
        assert "csttool" in str(user_dir)

    def test_is_data_installed_returns_false_when_missing(self):
        """Test that is_data_installed returns False when data missing."""
        with patch('csttool.data.loader._USER_DATA_DIR', Path("/nonexistent/path")):
            assert is_data_installed() is False


class TestValidation:
    """Tests for checksum validation and stamp logic."""

    def test_validate_if_needed_with_valid_file(self):
        """Test validation succeeds with valid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a temporary file
            test_file = tmpdir / "test.nii.gz"
            test_file.write_bytes(b"test data")

            # Mock manifest entry with correct SHA256 for "test data"
            manifest_entry = {"sha256": "916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9"}

            with patch('csttool.data.loader.get_manifest_entry', return_value=manifest_entry):
                with patch('csttool.data.loader._USER_DATA_DIR', tmpdir):
                    with patch('csttool.data.loader._VALIDATION_STAMP_FILE', tmpdir / ".validated"):
                        # Should not raise
                        _validate_if_needed(test_file, "test/test.nii.gz")

    def test_validate_if_needed_raises_on_checksum_mismatch(self):
        """Test validation fails with wrong checksum."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a temporary file
            test_file = tmpdir / "test.nii.gz"
            test_file.write_bytes(b"test data")

            # Mock manifest entry with wrong checksum
            manifest_entry = {"sha256": "wrongchecksum123"}

            with patch('csttool.data.loader.get_manifest_entry', return_value=manifest_entry):
                with patch('csttool.data.loader._USER_DATA_DIR', tmpdir):
                    with patch('csttool.data.loader._VALIDATION_STAMP_FILE', tmpdir / ".validated"):
                        with pytest.raises(DataNotInstalledError) as exc_info:
                            _validate_if_needed(test_file, "test/test.nii.gz")

                        assert "Checksum verification failed" in str(exc_info.value)

    def test_validation_stamp_saves_and_loads(self):
        """Test that validation stamp can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            stamp_data = {
                "/path/to/file.nii.gz": {
                    "size": 12345,
                    "mtime": 1234567890.0,
                    "sha256_ok": True,
                }
            }

            with patch('csttool.data.loader._USER_DATA_DIR', tmpdir):
                with patch('csttool.data.loader._VALIDATION_STAMP_FILE', tmpdir / ".validated"):
                    _save_validation_stamp(stamp_data)

                    loaded_data = _load_validation_stamp()
                    assert loaded_data == stamp_data


class TestManifest:
    """Tests for manifest.py functions."""

    def test_data_manifest_structure(self):
        """Test that DATA_MANIFEST has expected structure."""
        assert isinstance(DATA_MANIFEST, dict)
        assert len(DATA_MANIFEST) > 0

        # Check MNI152 entry exists (Tier 1)
        assert "mni152/MNI152_T1_1mm.nii.gz" in DATA_MANIFEST

        # Check FMRIB58_FA entries exist (Tier 2)
        assert "fmrib58_fa/FMRIB58_FA_1mm.nii.gz" in DATA_MANIFEST

        # Check Harvard-Oxford entries exist (Tier 2)
        assert "harvard_oxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz" in DATA_MANIFEST

    def test_manifest_entries_have_required_fields(self):
        """Test that all manifest entries have required fields."""
        required_fields = {"sha256", "source_url", "license", "version", "fsl_tag", "size_bytes"}

        for key, entry in DATA_MANIFEST.items():
            assert set(entry.keys()) == required_fields, \
                f"Entry {key} missing required fields"

            # Check SHA256 format (64 hex chars)
            assert len(entry["sha256"]) == 64, f"Invalid SHA256 for {key}"
            assert all(c in "0123456789abcdef" for c in entry["sha256"]), \
                f"Invalid SHA256 format for {key}"

    def test_get_manifest_entry_success(self):
        """Test get_manifest_entry returns correct entry."""
        entry = get_manifest_entry("mni152/MNI152_T1_1mm.nii.gz")

        assert isinstance(entry, dict)
        assert "sha256" in entry
        assert "source_url" in entry

    def test_get_manifest_entry_raises_on_invalid_key(self):
        """Test get_manifest_entry raises KeyError for invalid key."""
        with pytest.raises(KeyError) as exc_info:
            get_manifest_entry("invalid/key.nii.gz")

        assert "not found in manifest" in str(exc_info.value)

    def test_verify_checksum_with_valid_file(self):
        """Test verify_checksum returns True for valid file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test data")
            tmp_path = Path(tmp.name)

        try:
            # Correct SHA256 checksum for "test data"
            expected = "916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9"
            assert verify_checksum(tmp_path, expected) is True
        finally:
            tmp_path.unlink()

    def test_verify_checksum_with_invalid_checksum(self):
        """Test verify_checksum returns False for wrong checksum."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test data")
            tmp_path = Path(tmp.name)

        try:
            wrong_checksum = "0" * 64
            assert verify_checksum(tmp_path, wrong_checksum) is False
        finally:
            tmp_path.unlink()

    def test_verify_checksum_with_missing_file(self):
        """Test verify_checksum returns False for missing file."""
        assert verify_checksum(Path("/nonexistent/file.nii.gz"), "abc123") is False


class TestLicensing:
    """Tests for license file presence."""

    def test_mni152_license_exists(self):
        """Test that MNI152 license file exists."""
        from importlib.resources import files, as_file

        ref = files("csttool.data").joinpath("LICENSES", "MNI152_LICENSE.txt")
        with as_file(ref) as path:
            assert path.exists()
            content = path.read_text()
            assert "McGill" in content or "MNI" in content

    def test_fsl_license_exists(self):
        """Test that FSL license file exists."""
        from importlib.resources import files, as_file

        ref = files("csttool.data").joinpath("LICENSES", "FSL_LICENSE.txt")
        with as_file(ref) as path:
            assert path.exists()
            content = path.read_text()
            assert "FSL" in content or "Oxford" in content

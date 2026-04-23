"""Unit tests for csttool.bids.output helper functions and QC naming."""

import pytest
from csttool.bids.output import sanitize_bids_label, parse_dicom_age
from csttool.cli.commands.run import _resolve_qc_name, _QC_NAMES


# ---------------------------------------------------------------------------
# sanitize_bids_label
# ---------------------------------------------------------------------------

class TestSanitizeBidsLabel:
    def test_spaces_replaced(self):
        assert " " not in sanitize_bids_label("my label")

    def test_slashes_replaced(self):
        result = sanitize_bids_label("path/to/series")
        assert "/" not in result

    def test_numbers_first_prepended(self):
        result = sanitize_bids_label("3Tesla")
        assert result[0].isalpha()

    def test_consecutive_separators_collapsed(self):
        result = sanitize_bids_label("a  b--c")
        assert "--" not in result
        assert "  " not in result

    def test_max_len_respected(self):
        result = sanitize_bids_label("a" * 50)
        assert len(result) <= 20

    def test_max_len_custom(self):
        result = sanitize_bids_label("abcdefghij", max_len=5)
        assert len(result) <= 5

    def test_empty_string_returns_x(self):
        result = sanitize_bids_label("")
        assert result == "x"

    def test_all_symbols_returns_x(self):
        result = sanitize_bids_label("---")
        assert result[0].isalpha()

    def test_alphanumeric_unchanged(self):
        assert sanitize_bids_label("DWI01") == "DWI01"

    def test_leading_trailing_hyphens_stripped(self):
        result = sanitize_bids_label("!hello!")
        assert not result.startswith("-")
        assert not result.endswith("-")


# ---------------------------------------------------------------------------
# parse_dicom_age
# ---------------------------------------------------------------------------

class TestParseDicomAge:
    def test_years(self):
        assert parse_dicom_age("034Y") == 34.0

    def test_months(self):
        assert parse_dicom_age("018M") == pytest.approx(1.5, rel=1e-3)

    def test_days(self):
        result = parse_dicom_age("002D")
        assert result is not None
        assert result < 0.01

    def test_weeks(self):
        result = parse_dicom_age("004W")
        assert result is not None
        assert result < 0.1

    def test_empty_string_returns_none(self):
        assert parse_dicom_age("") is None

    def test_none_returns_none(self):
        assert parse_dicom_age(None) is None

    def test_invalid_format_returns_none(self):
        assert parse_dicom_age("notanage") is None

    def test_lowercase_unit(self):
        assert parse_dicom_age("025y") == 25.0

    def test_zero_years(self):
        assert parse_dicom_age("000Y") == 0.0


# ---------------------------------------------------------------------------
# _resolve_qc_name
# ---------------------------------------------------------------------------

class TestResolveQcName:
    def test_known_preproc_suffix(self):
        stage, label = _resolve_qc_name("sub001_brain_mask_qc.png")
        assert stage == "preproc"
        assert label == "brainmask"

    def test_known_tracking_suffix(self):
        stage, label = _resolve_qc_name("subject_tensor_maps.png")
        assert stage == "tracking"
        assert label == "tensormaps"

    def test_known_extraction_suffix(self):
        stage, label = _resolve_qc_name("sub001_registration_qc.png")
        assert stage == "extraction"
        assert label == "registration"

    def test_known_metrics_suffix(self):
        stage, label = _resolve_qc_name("sub001_tractogram_qc_sagittal.png")
        assert stage == "metrics"
        assert label == "tractogram-sagittal"

    def test_unknown_falls_back_to_misc(self):
        stage, label = _resolve_qc_name("something_unexpected.png")
        assert stage == "misc"
        assert "something_unexpected" in label

    def test_all_known_suffixes_resolve(self):
        for suffix in _QC_NAMES:
            stage, label = _resolve_qc_name(f"sub001{suffix}")
            assert stage != "misc", f"suffix {suffix!r} unexpectedly fell through to misc"

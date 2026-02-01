# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-01-28

### Added
- **Coordinate validation system** to prevent silent failures from tractogram/FA coordinate mismatches
  - Automatic validation checks bounding box overlap, detects voxel vs world space, and verifies orientation
  - New `--skip-coordinate-validation` flag to bypass validation (not recommended)
- **Hemisphere separation QC visualization** showing left/right CST bundles separately with midline reference
  - Displays cross-hemisphere contamination metrics
  - Color-coded QC status (green for good separation, red for warnings)
- **--quiet flag** for `extract`, `run`, and `batch` commands to suppress progress messages in batch pipelines
- Comprehensive documentation updates:
  - Expanded [limitations.md](docs/explanation/limitations.md) documenting coordinate validation as main technical risk
  - Updated [data-requirements.md](docs/getting-started/data-requirements.md) with coordinate space requirements
  - Added coordinate validation and new visualization info to CLI documentation

### Changed
- Extract command now validates tractogram coordinates against FA space before processing
- All QC visualization outputs now include hemisphere separation view by default when using `--save-visualizations`

## [0.3.1] - 2026-01-25

### Changed
- Unified batch analysis metrics with the single-subject report format.
- `batch_metrics.csv` now includes:
    - All diffusivity scalars (MD, RD, AD) in addition to FA.
    - Localized metrics for pontine, PLIC, and precentral regions.
    - Consistent column naming conventions (e.g., `left_n_streamlines` instead of `cst_l_streamline_count`).

## [0.3.0] - 2026-01-22

### Added
- Implemented robust batch processing system (`csttool batch`).
- Added manifest-based and BIDS-directory based batch execution.
- Added comprehensive pre-flight validation for batch inputs.
- Added consolidated CSV and JSON reporting for batch runs.
- Added parallel processing capabilities with timeout handling.

## [0.2.1] - 2026-01-20

### Changed
- Refactored CLI into a modular package structure (`src/csttool/cli/`).
- Moved CLI entry point commands to separate modules for better maintainability.

### Fixed
- Updated `dipy.core.gradients.gradient_table` call to use `bvecs` keyword argument, fixing future warnings/errors.
- Enabled `copy_header=True` in `image.resample_to_img` to preserve header information during atlas resampling.

## [0.2.0] - 2026-01-20

### Changed
- Refined CST extraction logic:
    - Added mutual exclusivity filter for motor cortices (streamlines cannot touch both).
    - Added midline rejection filter (streamlines cannot cross X=0 sagittal plane).
- Major refactor of PDF report generation:
    - Replaced inline report generation with Jinja2 templating.
    - Switched to WeasyPrint for robust HTML-to-PDF conversion.
    - Improved layout handling for A4 pages.

### Fixed
- Resolved ROI placement asymmetry caused by orientation mismatch (LAS vs RAS).
- Fixed `csttool run` failure when using NIfTI input (`--nifti`) instead of DICOM.
- Re-enabled SyN non-linear registration for improved ROI symmetry.
- Adjusted affine handling to respect original subject orientation.

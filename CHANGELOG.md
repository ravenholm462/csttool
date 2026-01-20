# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

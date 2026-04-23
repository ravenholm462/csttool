# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-04-23

### Added

- **BIDS-native output layout** — `csttool run` now writes a BIDS derivatives tree by
  default, with no extra flags required. All outputs are moved (not copied) into the
  subject directory; stage working directories are removed unconditionally afterwards.
  - `sub-<id>/[ses-<id>/]dwi/` — preprocessed NIfTI, bval/bvec, scalar maps
    (`space-orig_model-DTI_param-{FA,MD,RD,AD}_dwimap.nii.gz`), derivative JSON sidecars
  - `sub-<id>/[ses-<id>/]dwi/tractography/` — whole-brain, CST left, CST right, and
    bilateral combined tractograms
  - `sub-<id>/[ses-<id>/]figures/` — QC images renamed with stage and label entities
    (`stage-{preproc,tracking,extraction,metrics}_qc-{label}.png`)
  - `sub-<id>/[ses-<id>/]reports/` — HTML/PDF reports, metrics JSON/CSV, pipeline logs
  - `dataset_description.json`, `participants.tsv`, and `participants.json` at dataset root
  - `SourceDatasets: [{"URL": "bids::"}]` when derivatives are nested under a raw BIDS root
- **`--bids-out`** flag on `csttool run` — overrides the derivatives root (default: `--out`)
- **`--session-id`** flag on `csttool run` — sets the BIDS session label
- **`--bids-out`** flag on `csttool batch` — writes `dataset_description.json` and
  `participants.tsv` at the dataset root after the batch completes
- **Raw BIDS import** via `csttool import --dicom <dir> --raw-bids <out>` — produces a
  fully BIDS-compliant raw dataset (`DatasetType: raw`)
  - Subject label derived from SHA-256 hash of `PatientID` by default (anonymised)
  - Session label derived from `StudyDate`
  - `--keep-phi` flag to use `PatientID` directly (prints PHI warning)
  - `--subject-id` and `--session-id` to override auto-derivation
- **`dcm2niix` promoted to primary DICOM converter** — handles Siemens, GE, Philips, and
  Hitachi; generates BIDS JSON sidecars automatically. Falls back to `dicom2nifti` with a
  `fallback_used` flag and a vendor-specific warning for known-unreliable vendors.
- **`pydicom`** added as a required dependency
- **`bids/output.py`** module: `write_dataset_description`, `update_participants_tsv`,
  `write_participants_json`, `bids_filename`, `write_derivative_sidecar`,
  `sanitize_bids_label`, `parse_dicom_age`, `hash_patient_id`
- **`manufacturer`** field added to import report JSON and series info JSON
- 25 unit tests for BIDS output helpers and QC image naming (`tests/bids/`)

### Changed

- Output layout is BIDS derivatives by default — the flat stage-directory structure
  (`tracking/`, `extraction/`, `metrics/`, `preprocessing/`) is internal working state
  only, removed after finalization
- QC images explicitly routed to `figures/` with systematic names regardless of flags;
  output contract is stable across all flag combinations
- Reports (HTML, PDF) and tabular outputs (metrics JSON, CSV) routed to `reports/`,
  distinct from QC images in `figures/`

## [0.4.0] - 2026-01-28

### Added

- **Coordinate validation system** to prevent silent failures from tractogram/FA
  coordinate mismatches
  - Automatic validation checks bounding box overlap, detects voxel vs world space,
    and verifies orientation
  - New `--skip-coordinate-validation` flag to bypass validation (not recommended)
- **Hemisphere separation QC visualization** showing left/right CST bundles separately
  with midline reference
  - Displays cross-hemisphere contamination metrics
  - Color-coded QC status (green for good separation, red for warnings)
- **`--quiet` flag** for `extract`, `run`, and `batch` commands
- Documentation updates:
  - Expanded [limitations.md](docs/explanation/limitations.md)
  - Updated [data-requirements.md](docs/getting-started/data-requirements.md) with
    coordinate space requirements

### Changed

- Extract command now validates tractogram coordinates against FA space before processing
- All QC visualization outputs now include hemisphere separation view by default when
  using `--save-visualizations`

## [0.3.1] - 2026-01-25

### Changed

- Unified batch analysis metrics with the single-subject report format
- `batch_metrics.csv` now includes:
  - All diffusivity scalars (MD, RD, AD) in addition to FA
  - Localized metrics for pontine, PLIC, and precentral regions
  - Consistent column naming conventions

## [0.3.0] - 2026-01-22

### Added

- Robust batch processing system (`csttool batch`)
- Manifest-based and BIDS-directory based batch execution
- Comprehensive pre-flight validation for batch inputs
- Consolidated CSV and JSON reporting for batch runs
- Parallel processing with timeout handling

## [0.2.1] - 2026-01-20

### Changed

- Refactored CLI into a modular package structure (`src/csttool/cli/`)
- Moved CLI entry point commands to separate modules

### Fixed

- Updated `dipy.core.gradients.gradient_table` call to use `bvecs` keyword argument
- Enabled `copy_header=True` in `image.resample_to_img` to preserve header information

## [0.2.0] - 2026-01-20

### Changed

- Refined CST extraction logic:
  - Added mutual exclusivity filter for motor cortices
  - Added midline rejection filter (streamlines cannot cross X=0 sagittal plane)
- Major refactor of PDF report generation:
  - Replaced inline generation with Jinja2 templating
  - Switched to WeasyPrint for HTML-to-PDF conversion

### Fixed

- Resolved ROI placement asymmetry caused by orientation mismatch (LAS vs RAS)
- Fixed `csttool run` failure when using NIfTI input (`--nifti`) instead of DICOM
- Re-enabled SyN non-linear registration for improved ROI symmetry
- Adjusted affine handling to respect original subject orientation

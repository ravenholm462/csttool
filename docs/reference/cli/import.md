# Ingest Module Walkthrough

The `ingest` module handles the import and standardization of diffusion MRI data, converting raw DICOM images into analysis-ready NIfTI format.

## Core Capability: `csttool import`

This command interface wraps the ingest module to provide robust DICOM-to-NIfTI conversion, metadata extraction, and basic quality assessment.

### Key Features

- **DICOM Scanning**: Recursively scans directories to identify available series.
- **Robust Conversion**: Uses `dcm2niix` for reliable conversion.
- **Metadata Extraction**: Parses acquisition parameters (TE, TR, Field Strength, b-values) crucial for downstream processing.
- **Series Selection**: Allows selecting specific series when multiple are present.
- **BIDS Compliance**: Generates JSON sidecars compatible with BIDS.

### Usage

#### 1. Scan-Only Mode
Preview available series in a directory without converting.

```bash
csttool import --dicom /path/to/dicoms --scan-only
```

#### 2. Import Series
Convert a specific series to NIfTI.

```bash
csttool import \
    --dicom /path/to/dicoms \
    --out /path/to/output \
    --series 2 \
    --subject-id sub-001
```

#### 3. Standardize NIfTI
If you already have a NIfTI file, `import` acts as a standardization step, validating the file and extracting metadata.

```bash
csttool import \
    --nifti raw_dwi.nii.gz \
    --out /path/to/output
```

### Module Components

- **`scan_study.py`**: Analyzes DICOM headers to group files into series.
- **`convert_series.py`**: wrapper around `dcm2niix` execution.
- **`assess_quality.py`**: Checks if the acquisition parameters (e.g., number of directions, b-value shells) are suitable for CSD tracking.
- **`extract_acquisition_metadata`**: Homogenizes metadata from BIDS JSON and NIfTI headers for the final report.

### Quality Checks (`csttool check-dataset`)

The module also powers the `check-dataset` command, which provides a standalone quality report for an input dataset.

```bash
csttool check-dataset --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec
```

## Example Output

```text
Scanning directory: /raw/sub-001/dicoms
Found 5 series:
  1. localizer (3 images)
  2. t1_mprage (192 images)
  3. dwi_dir64_AP (65 images) [SELECTED]
  4. dwi_dir64_PA (1 image)
  5. t2_flair (192 images)

Converting series 3...
Reference: /raw/sub-001/dicoms/IM-0003-0001.dcm
Output: /processed/sub-001/dwi.nii.gz

✓ Import complete.
```

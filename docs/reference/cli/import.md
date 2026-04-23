# `csttool import`

Import DICOM data or standardise an existing NIfTI file. Supports two output modes:
a **raw BIDS dataset** (scanner-to-NIfTI with full BIDS organisation) and a
**flat output directory** for ad-hoc use.

---

## Modes

### 1. Raw BIDS import (recommended)

Converts a DICOM dump and organises the result as a BIDS-compliant raw dataset.

```bash
csttool import \
    --dicom /path/to/dicoms \
    --raw-bids /data/bids \
    --subject-id 001
```

Output:

```text
/data/bids/
├── dataset_description.json   (DatasetType: "raw")
├── participants.tsv
├── participants.json
└── sub-001/
    └── ses-20240315/          (derived from StudyDate, or use --session-id)
        └── dwi/
            ├── sub-001_ses-20240315_dwi.nii.gz
            ├── sub-001_ses-20240315_dwi.bval
            ├── sub-001_ses-20240315_dwi.bvec
            └── sub-001_ses-20240315_dwi.json  (BIDS sidecar from dcm2niix)
```

!!! warning "Anonymisation is on by default"
    Without `--subject-id`, the subject label is derived from a SHA-256 hash of
    the DICOM `PatientID` tag (e.g. `sub-a3f2b1c4`). Pass `--keep-phi` to use
    `PatientID` directly — a PHI warning is printed and the output dataset will
    **not** be de-identified.

### 2. Flat output (legacy / ad-hoc)

Converts a DICOM series to NIfTI in a plain output directory.

```bash
csttool import \
    --dicom /path/to/dicoms \
    --out /path/to/output \
    --subject-id sub-001
```

### 3. Scan-only

Preview available series without converting.

```bash
csttool import --dicom /path/to/dicoms --scan-only
```

### 4. Standardise existing NIfTI

Validate an existing NIfTI file and extract acquisition metadata.

```bash
csttool import --nifti raw_dwi.nii.gz --out /path/to/output
```

---

## DICOM converter

`csttool import` tries converters in this order:

1. **`dcm2niix`** (primary) — handles Siemens, GE, Philips, and Hitachi; generates
   BIDS JSON sidecars automatically. Recommended.
2. **`dicom2nifti`** (fallback) — used when `dcm2niix` is absent or fails on a
   specific series. Less reliable for non-Siemens data; does not generate BIDS
   JSON sidecars. A `fallback_used: true` flag is written to the import report,
   with the error reason.

The detected scanner vendor (`Manufacturer` DICOM tag) is logged in the import
report (`*_import_report.json`). If a known-problematic vendor (GE, Philips,
Hitachi) is detected and `dcm2niix` is unavailable, an explicit warning is printed.

Install `dcm2niix`:

```bash
brew install dcm2niix        # macOS
sudo apt install dcm2niix    # Debian / Ubuntu
conda install -c conda-forge dcm2niix
```

---

## Options

| Flag | Description |
| --- | --- |
| `--dicom <dir>` | Path to DICOM directory (study root or single series) |
| `--nifti <file>` | Path to existing NIfTI (skip conversion) |
| `--out <dir>` | Output directory for flat mode (required without `--raw-bids`) |
| `--raw-bids <dir>` | Organise output as a raw BIDS dataset at this path |
| `--subject-id <id>` | Subject label (without `sub-` prefix). Auto-derived from `PatientID` if omitted. |
| `--session-id <id>` | Session label (without `ses-` prefix). Overrides `StudyDate` derivation. |
| `--keep-phi` | Disable anonymisation: use `PatientID` directly as subject label. Prints PHI warning. |
| `--series <n>` | Series number to convert (1-indexed, as shown in `--scan-only`) |
| `--scan-only` | List available series and exit without converting |
| `--field-strength <T>` | Override field strength (Tesla) |
| `--echo-time <ms>` | Override echo time (ms) |
| `--verbose` | Print detailed processing information |

---

## Quality checks

The import module powers the standalone quality assessment command:

```bash
csttool check-dataset --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --verbose
```

This checks b-value shells, number of directions, and acquisition parameters without
running the full pipeline.

---

## Related

- [Output formats](../output-formats.md) — full description of the BIDS output layout
- [`csttool run`](run.md) — single-subject pipeline (uses raw BIDS output as input via `--nifti`)
- [`csttool batch`](batch.md) — multi-subject processing with BIDS auto-discovery

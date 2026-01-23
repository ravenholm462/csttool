# Run Command Walkthrough

The `run` command (`src/csttool/cli/commands/run.py`) is the primary interface for processing a single subject. It acts as a pipeline orchestrator, calling individual module functions in sequence and managing data flow.

## Pipeline Steps

1.  **Environment Check**: Ensures all tools are available.
2.  **Import**: Standardizes input data to NIfTI.
3.  **Preprocess**: Applies `patch2self` (or `nlmeans`), Gibbs unringing, and motion correction.
4.  **Track**: Generates a whole-brain tractogram.
5.  **Extract**: Filters the tractogram to isolate the Corticospinal Tract (CST).
6.  **Metrics**: Calculates FA/MD profiles and generates clinical reports.

## Usage Example

```bash
csttool run \
    --dicom /path/to/dicom/series \
    --out /path/to/output/sub-001 \
    --subject-id sub-001 \
    --denoise-method patch2self \
    --generate-pdf
```

## Features

-   **Pipeline Tracking**: Records execution time for each step.
-   **Error Handling**: Can stop on first error or continue with partial processing (`--continue-on-error`).
-   **Metadata**: Aggregates acquisition and processing metadata for the final report.
-   **Reporting**: Generates a comprehensive pipeline report in the output directory.

## Example Output

```text
CSTTOOL - COMPLETE CST ANALYSIS PIPELINE
======================================================================
Subject ID:     sub-001
Output:         /processed/sub-001
Started:        2025-01-23 10:00:00
======================================================================

▶▶▶ STEP 1/6: ENVIRONMENT CHECK ◀◀◀
✓ Environment Check Passed.

▶▶▶ STEP 2/6: IMPORT DATA ◀◀◀
Using existing NIfTI: /raw/sub-001/dwi.nii.gz

▶▶▶ STEP 3/6: PREPROCESSING ◀◀◀
✓ Denoising (Patch2Self) complete.
✓ Gibbs unringing complete.

▶▶▶ STEP 4/6: TRACTOGRAPHY ◀◀◀
✓ Whole-brain tracking complete: 450,000 streamlines.

▶▶▶ STEP 5/6: CST EXTRACTION ◀◀◀
✓ Left CST: 2,450 streamlines
✓ Right CST: 2,130 streamlines

▶▶▶ STEP 6/6: METRICS & REPORTS ◀◀◀
✓ HTML report generated: /processed/sub-001/metrics/report.html
✓ PDF report generated: /processed/sub-001/metrics/report.pdf

======================================================================
PIPELINE COMPLETE
======================================================================
Total time:     15.2 minutes
Step timing:
  ✓ check       : 0.5s
  ✓ import      : 1.2s
  ✓ preprocess  : 420.5s
  ✓ track       : 350.0s
  ✓ extract     : 120.5s
  ✓ metrics     : 20.1s

✓ All steps completed successfully!
```

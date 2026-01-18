# Quick Start

Run the entire CST assessment pipeline in one command.

---

## The `csttool run` Command

=== "NIfTI Input"

    ```bash
    csttool run --nifti /path/to/dwi.nii.gz --out /path/to/output \
        --subject-id sub-01 --generate-pdf --save-visualizations
    ```

=== "DICOM Input"

    ```bash
    csttool run --dicom /path/to/study/ --out /path/to/output \
        --subject-id sub-01 --generate-pdf --save-visualizations
    ```

!!! tip "Need sample data?"
    See [recommended datasets](data-requirements.md#recommended-datasets) for freely available diffusion MRI data.

!!! note "Runtime & disk space"
    Typical runtime is **2–10 minutes** depending on data size, hardware and chosen parameters. Notably, `patch2self` denoising and the `--perform-motion-correction` flag will increase runtime (in the latter case quite substantially). The pipeline generates up to **500 MB** of output files (tractograms, scalar maps, reports) per pipeline run.

---

## What Each Step Does

The pipeline runs 6 steps automatically:

| Step | Name | Description |
|------|------|-------------|
| 1 | **Check** | Validates your environment and dependencies |
| 2 | **Import** | Loads NIfTI/DICOM data and extracts gradient information |
| 3 | **Preprocess** | Denoises, corrects motion, and segments the brain |
| 4 | **Track** | Generates a whole-brain tractogram using CSD-based tractography |
| 5 | **Extract** | Isolates left and right CST using atlas-based ROI filtering |
| 6 | **Metrics** | Computes FA/MD/RD/AD along tracts and generates reports |

---

## Example Output

A successful run looks like this:

```
======================================================================
PIPELINE COMPLETE
======================================================================
Subject ID:     sub_sca201_ses01
Total time:     3.3 minutes (201 seconds)

Step timing:
  ✓ check       : 0.0s
  ✓ import      : 0.0s
  ✓ preprocess  : 7.7s
  ✓ track       : 72.2s
  ✓ extract     : 110.9s
  ✓ metrics     : 10.1s

✓ All steps completed successfully!

Outputs:
  Pipeline report: output/sub_sca201_ses01_pipeline_report.json
  Metrics:         output/metrics
  CST tractograms: output/extraction
======================================================================
```

---

## Output Structure

```
output/
├── sub-01_pipeline_report.json    # Full pipeline log
├── nifti/                         # Converted NIfTI (if DICOM input)
├── preprocessing/                 # Denoised, corrected data
│   └── visualizations/            # QC images (if --save-visualizations)
├── tracking/                      # Whole-brain tractogram
│   └── scalar_maps/               # FA, MD, RD, AD maps
├── extraction/                    # CST tractograms
│   └── trk/                       # Left/right CST .trk files
└── metrics/
    ├── report.pdf                 # Visual summary report
    ├── report.html                # Interactive HTML report
    ├── metrics_summary.csv        # Tabular metrics
    └── visualizations/            # Tract profiles, QC images
```

---

## Key Options

| Flag | Description |
|------|-------------|
| `--nifti` / `--dicom` | Input format (provide path to data) |
| `--out` | Output directory |
| `--denoise-method` | `patch2self` (default) or `nlmeans` |
| `--generate-pdf` | Generate PDF report |
| `--save-visualizations` | Save QC visualizations at each step |
| `--subject-id` | Subject ID |
| `--verbose` | Verbose output |

Run `csttool run --help` for all options.

---

## What's Next?

- **[Data Requirements](data-requirements.md)** — Input format specifications and recommended datasets
- **[Troubleshooting](../how-to/troubleshooting.md)** — Common issues and fixes
- **[CLI Reference](../reference/cli/run.md)** — Full `csttool run` documentation

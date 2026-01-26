# Metrics Module Walkthrough

The `metrics` module is responsible for the quantitative analysis of extracted tracts. It computes macro-structural and micro-structural properties and generates comprehensive clinical reports.

## Core Capability: `csttool metrics`

This command analyzes the Left and Right CST tractograms, samples diffusion scalar maps (FA, MD, RD, AD), and produces a set of reports and visualizations.

### Usage

```bash
csttool metrics \
    --cst-left extracted/cst_left.trk \
    --cst-right extracted/cst_right.trk \
    --fa dti_FA.nii.gz \
    --md dti_MD.nii.gz \
    --out metrics_results \
    --generate-pdf \
    --subject-id sub-001
```

### Analysis Pipeline

1.  **Hemispheric Analysis** (`analyze_cst_hemisphere`):
    -   **Morphology**: Calculates tract volume (mm³) and streamline count.
    -   **Scalar Statistics**: Computes mean and standard deviation for all provided maps (FA, MD, etc.).
    -   **Tract Profiling**: Resamples streamlines to 100 equidistant points and computes the average scalar value at each point, creating a "tract profile" representing the bundle from brainstem to cortex.

2.  **Bilateral Comparison** (`compare_bilateral_cst`):
    -   **Asymmetry**: Calculates the Laterality Index (LI) for key metrics:
        $$LI = \frac{Left - Right}{Left + Right}$$
    -   **Profile Correlation**: Measures the similarity between left and right tract profiles.

3.  **Reporting** (`modules.reports`):
    -   **JSON**: Full hierarchical data structure including all metadata.
    -   **CSV**: Flat summary row, ideal for aggregating multiple subjects.
    -   **PDF/HTML**: Rich visual report containing:
        -   Subject and acquisition metadata.
        -   Summary tables of FA/MD and volumes.
        -   Stacked profile plots (visualizing asymmetry along the tract).
        -   QC snapshots of the tracts overlaid on the FA map.

### Visualizations

-   **Tract Profiles**: Line plots showing FA variation along the z-axis (Superior-Inferior).
-   **Lateral-Directional Color Coding**: Visualizes tract orientation.
-   **QC Preview**: Axial, Sagittal, and Coronal views of the tractograms.

## Example Output

```text
Loading left CST: extraction/cst_left.trk
  Loaded 2,450 streamlines
Loading right CST: extraction/cst_right.trk
  Loaded 2,130 streamlines
Loading FA map: tracking/dti_FA.nii.gz

Analyzing LEFT CST
============================================================
Mean FA: 0.582 ± 0.120
Volume:  15.24 cm³

Analyzing RIGHT CST
============================================================
Mean FA: 0.575 ± 0.115
Volume:  14.80 cm³

Computing bilateral comparison
============================================================
Laterality Index (FA): 0.006 (Symmetric)
Profile Correlation:   0.92 (High Similarity)

Generating reports
============================================================
✓ JSON report: metrics/report.json
✓ CSV summary: metrics/summary.csv
✓ Tract profiles: metrics/visualizations/tract_profiles.png
✓ Bilateral comparison: metrics/visualizations/bilateral_comparison.png
✓ HTML report: metrics/report.html
✓ PDF report: metrics/report.pdf

METRICS COMPLETE
```

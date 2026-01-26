# Preprocess Module Walkthrough

The `preprocess` module prepares raw diffusion data for tractography by reducing noise and artifacts.

## Core Capability: `csttool preprocess`

This command runs a sequential pipeline of image correction steps.

### Usage

```bash
csttool preprocess \
    --nifti raw_dwi.nii.gz \
    --out preprocess_results \
    --denoise-method patch2self \
    --unring \
    --perform-motion-correction \
    --save-visualizations
```

### Pipeline Steps

1.  **Denoising**:
    -   **Patch2Self** (Default): Self-supervised learning method that uses information from other volumes to denoise the current volume. Highly recommended for modern acquisitions.
    -   **NLMeans**: Non-local means denoising (classic approach).
2.  **Gibbs Unringing** (Optional):
    -   Corrects for Gibbs ringing artifacts near sharp contrast boundaries (e.g., skull/cortex).
3.  **Motion Correction** (Optional):
    -   Registers all volumes to the first b=0 image using an affine transformation to correct for subject movement.
4.  **Reslicing** (Optional):
    -   Resamples the data to a target isotropic voxel size (e.g., 2mm iso).
5.  **Brain Masking**:
    -   Computes a binary brain mask using median_otsu to strip the skull.

### Output

-   `dti_preproc.nii.gz`: The fully corrected 4D DWI series.
-   `dti_preproc_mask.nii.gz`: The computed brain mask.
-   `visualizations/`:
    -   `denoising_residuals.png`: Visual check of what was removed (should be noise, not anatomy).
    -   `motion_correction_plot.png`: Estimated translation/rotation parameters over time.

## Example Output

```text
Starting preprocessing for sub-001...
Input: 96x96x60 (65 volumes)

1. Gibbs Unringing
   - Axis: 2 (z-axis)
   - Splits: 3

2. Motion Correction
   - Reference: Volume 0 (b=0)
   - Corrected 64 volumes
   - Max translation: 1.2mm (Vol 42)
   - Max rotation: 0.8 deg (Vol 42)

3. Denoising (Patch2Self)
   - Model: OLS
   - Shift: True
   - Sigma estimation: Local PCA

✓ Preprocessing complete: /processed/sub-001/preprocessing/dti_preproc.nii.gz
```

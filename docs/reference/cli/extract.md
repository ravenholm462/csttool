# Extract Module Walkthrough

The `extract` module allows for the isolation of the Corticospinal Tract (CST) from whole-brain tractograms or directly from DWI data.

## Core Capability: `csttool extract`

This command filters streamlines that connect the brainstem and the motor cortex. It uses the Harvard-Oxford atlas, registered to the subject's native space, to define these Regions of Interest (ROIs).

### Usage

```bash
csttool extract \
    --tractogram whole_brain.trk \
    --fa dti_FA.nii.gz \
    --out extraction_results \
    --extraction-method passthrough \
    --save-visualizations
```

### Methods

1.  **Passthrough (Default)**:
    -   **Input**: Whole-brain tractogram (.trk).
    -   **Logic**: Keeps streamlines that traverse *both* the Brainstem and Primary Motor Cortex ROIs.
    -   **Use Case**: Standard post-hoc filtering of deterministic tractography.

2.  **Endpoint**:
    -   **Input**: Whole-brain tractogram (.trk).
    -   **Logic**: Keeps streamlines that start/end specifically within the ROIs.
    -   **Use Case**: Stricter anatomical constraints.

3.  **ROI-Seeded**:
    -   **Input**: Preprocessed DWI + FA map (via `csttool run`).
    -   **Logic**: Performs tractography by seeding valid paths directly from the Motor Cortex and filtering for Brainstem connectivity.
    -   **Use Case**: Dense reconstruction of specific bundles; bypasses whole-brain tracking.

### Algorithm Steps

1.  **Registration**:
    -   Registers the MNI template to the subject's FA map using affine + SyN (non-linear) transformation.
2.  **Atlas Warping**:
    -   Transforms the Harvard-Oxford Cortical and Subcortical atlases into subject space.
3.  **ROI Creation**:
    -   **Brainstem**: Extracted from Subcortical atlas.
    -   **Motor Cortex**: Precentral Gyrus extracted from Cortical atlas.
    -   **Dilation**: ROIs are dilated (default: Brainstem 2 iter, Motor 1 iter) to ensure overlap with white matter.
4.  **Extraction**:
    -   Applies the selected filtering/tracking logic.
    -   Separates streamlines into Left and Right CST.
5.  **Output**:
    -   Saves `cst_left.trk`, `cst_right.trk`, and `cst_combined.trk`.
    -   Generates QC visualizations if requested.

## Example Output

```text
Step 1: Registering MNI template to subject space
  - Affine registration... Converged.
  - SyN registration... Converged.

Step 2: Warping Harvard-Oxford atlases to subject space
  - Warping Cortical Atlas... Done.
  - Warping Subcortical Atlas... Done.

Step 3: Creating CST ROI masks
  - Brainstem: 2840 voxels (after dilation=2)
  - Motor Cortex (Left): 1240 voxels (after dilation=1)
  - Motor Cortex (Right): 1180 voxels (after dilation=1)

Step 4: Extracting bilateral CST (method: passthrough)
  - Filtering 452,180 streamlines...

EXTRACTION COMPLETE
Subject: sub-001
Left CST:  2,450 streamlines
Right CST: 2,130 streamlines
Total:     4,580 streamlines
Extraction rate: 1.01%
```

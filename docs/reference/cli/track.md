# Tracking Module Walkthrough

The `tracking` module performs deterministic whole-brain tractography to reconstruct white matter pathways.

## Core Capability: `csttool track`

This command generates a whole-brain tractogram from preprocessed diffusion data using the Constant Solid Angle (CSA) ODF model.

### Usage

```bash
csttool track \
    --nifti dti_preproc.nii.gz \
    --out tracking_results \
    --fa-thr 0.2 \
    --sh-order 6 \
    --seed-density 2
```

### Algorithm Steps

1.  **Brain Masking**:
    -   Refines the brain mask using median_otsu to ensure tracking is confined to the brain.
2.  **Tensor Fitting (DTI)**:
    -   Fits the classical Tensor model to compute scalar maps:
        -   **FA**: Fractional Anisotropy (microstructural integrity).
        -   **MD**: Mean Diffusivity (overall diffusion magnitude).
        -   **RD/AD**: Radial/Axial Diffusivity.
3.  **Direction Estimation (CSA)**:
    -   Fits the Constant Solid Angle (CSA) model to the high-angular resolution data (HARDI).
    -   Computes Orientation Distribution Functions (ODFs) to resolve crossing fibers.
4.  **Seeding**:
    -   Generates seeds in all voxels where `FA > fa_thr` (approx. White Matter).
    -   Seeds are placed randomly within voxels based on `seed_density`.
5.  **Tracking**:
    -   Uses **EuDX** (Euler Integration) for deterministic tracking.
    -   **Stop Criteria**: Tracking stops if FA drops below threshold or angle exceeds limit (45°).
6.  **Output**:
    -   `tractogram.trk`: The resulting whole-brain streamlines.
    -   `dti_FA.nii.gz`: FA map (reference for downstream registration).
    -   `dti_MD.nii.gz`: MD map.

## Example Output

```text
Subject: sub-001
Loading preprocessed data from dti_preproc.nii.gz...

Step 1: Brain masking with median Otsu
Step 2: Tensor fit and scalar measures
  - FA/MD/RD/AD maps created.

Step 3: Direction field estimation (CSA ODF model)
  - SH order: 6
  - Peaks found.

Step 4: Stopping criterion and seed generation
  - FA threshold: 0.2
  - Seed density: 2
  - Generated 1,240,500 seeds within white matter mask.

Step 5: Deterministic tracking
  - Algorithm: EuDX
  - Step size: 0.5mm
  - Angle threshold: 45°

Step 6: Saving outputs
  - Tractogram: tracking/tractogram.trk
  - FA map: tracking/dti_FA.nii.gz

TRACKING COMPLETE
Whole-brain streamlines: 452,180
```

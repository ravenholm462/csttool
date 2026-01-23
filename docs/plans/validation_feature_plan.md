# CST Validation Feature Implementation Plan

## Goal
Add validation capability to compare csttool-extracted CST streamlines against reference tractograms (e.g., TractoInferno PYT bundles) with strict spatial compliance and robust metrics.

---

## 1. Safety & Architecture (High Priority)

### Spatial Consistency Contract
To prevent silent failures (e.g., getting Dice=0.0 due to MNI vs Scanner space mismatch), validation **must** enforce a spatial contract.

- **Requirement**: Validation takes a reference NIfTI (e.g., FA map) which defines the "Validation Grid" (Dimensions + Affine).
- **Check**: When loading a `.trk` file, the tool must validate:
    1. Tractogram `affine_to_rasmm` matches the reference space affine within tolerance.
       - **Tolerances**: Translation < 1.0mm, Rotation/Scale < 1e-3.
    2. OR, the tractogram explicitly declares a reference file that matches the provided reference space.
- **Resolution**: If spaces differ beyond tolerance, the tool must error with a clear message showing both affines.

### Robustness & Error Handling
- **Empty/Sparse Bundles**: 
    - **Differentiation**: Distinguish `EMPTY_CAND` (candidate has 0 streamlines) from `EMPTY_REF` (bad input).
    - **Metrics**: 
        - Dice: 0.0 if candidate empty, NaN if reference empty.
        - MDF: NaN if either is empty.
    - **Flags**: Always emit QC flags: `EMPTY_CAND`, `EMPTY_REF`, `BOTH_EMPTY`.
    - **Sparse**: Warn if candidate has < 10 streamlines.
- **Hemisphere Check**: 
    - Use the reference space orientation to define the separation plane.
    - Warn (do NOT error) if the bundle centroid appears to be in the wrong hemisphere.
    - Allow disabling via `--disable-hemisphere-check`.

---

## 2. CLI Design (High Priority)

Positional arguments are error-prone. Explicit flags are mandatory.

### Proposed Interface
```bash
csttool validate \
    --cand-left output/sub-01/cst_left.trk \
    --cand-right output/sub-01/cst_right.trk \
    --ref-left  derivatives/sub-01/PYT_L.trk \
    --ref-right derivatives/sub-01/PYT_R.trk \
    --ref-space derivatives/sub-01/dti_FA.nii.gz \
    --output-dir output/sub-01/validation \
    --visualize \
    --disable-hemisphere-check
```

---

## 3. Metrics Specification

### Core Metrics (v1)

1. **Dice Coefficient (Overlap)**
   - **Definition**: Binary occupancy overlap on the "Validation Grid".
   - **Method**: Streamlines -> Density Map -> Threshold > 0 -> Dice calculation.
   - **Parameters**: 
     - **Grid**: Defined by `--ref-space`.
     - **Dilation**: Optional dilation applied to *both* candidate and reference masks before Dice.

2. **Coverage (False Negatives)**
   - **Definition**: Fraction of valid reference volume NOT covered by Candidate. `1 - (Intersection / Ref_Volume)`.
   - **Method**: Computed on **non-dilated** occupancy maps by default to be strict.

3. **Overreach (False Positives)**
   - **Definition**: Fraction of Candidate volume that falls outside Reference volume. `(Cand_Volume - Intersection) / Cand_Volume`.
   - **Method**: Computed on **non-dilated** occupancy maps by default.

4. **Mean Closest Distance (MDF)**
   - **Definition**: Symmetric Mean of Closest Points. `(mean(min_dist(A, B)) + mean(min_dist(B, A))) / 2`.
   - **Resampling**: **Must** resample streamlines to a fixed step size (e.g., 2mm) to prevent bias from point density. Fixed point count (20) is deprecated.

5. **Streamline Count Ratio (Sanity)**
   - **Definition**: `Count(Candidate) / Count(Reference)`.

---

## 4. Reporting & Artifacts

### JSON Report (Source of Truth)
Must capture full reproducibility context:
- **Paths & Hashes**: Reference space, input bundles.
- **Configuration**: Resampling step size, dilation settings, tolerance used.
- **QC Flags**: `EMPTY_CAND`, `SPATIAL_MISMATCH_WARNED`, `HEMI_SWAP_WARNED`.
- **Metrics**: Full breakdown per side.
- **Version**: csttool version and git commit.

### Visualization Artifacts (High Priority)
When `--visualize` is active, generate and save:
- `cand_occ.nii.gz`, `ref_occ.nii.gz`, `overlap_map.nii.gz` (in reference space).
- **Snapshots**: Axial, Coronal, Sagittal PNGs using the reference space image as background and the overlap map as overlay (Red=Cand, Blue=Ref, Purple=Overlap).

---

## 5. Implementation Checklist & Priorities

### 🔴 High Priority: Safety, Accuracy & Visualization
- [ ] **Spatial Consistency Check**: Implement strict affine/grid validation with tolerances.
- [ ] **Explicit CLI Flags**: Implement `--cand-left`, `--ref-left`, `--ref-space`.
- [ ] **Robust Metric Signatures**: Update metrics to use grid from `--ref-space` and handle empty inputs correctly (distinct NaNs/Zeros).
- [ ] **MDF Resampling**: Implement fixed-step (mm) resampling for distance calculation.
- [ ] **Visualization**: Implement NIfTI and PNG generation for debugging.
- [ ] **Coverage Metric**: Implement reference coverage.
- [ ] **Unit Tests**: Affine mismatch, zero-streamline bundles, warnings verification.

### 🟡 Medium Priority: Usability & Insight
- [ ] **Streamline Count Ratio**: Implement simple count ratio metric.
- [ ] **Hemisphere Sanity Check**: Implement centroid-based warning with `--disable-hemisphere-check`.
- [ ] **Overreach**: Implement non-dilated overreach.
- [ ] **JSON Metadata**: Ensure all configuration parameters and versions are recorded.

### 🟢 Low Priority: Extensions
- [ ] **HTML Reports**: Generate pretty HTML summary.
- [ ] **Batch Command**: `csttool validate-batch`.
- [ ] **External Preprocessing**: Flags in `run` to skip preprocessing.

---

## 6. Verification Plan

### Automated Tests
Run `pytest tests/validation/`:
- `test_spatial.py`: Verify mismatches > 1mm raise error.
- `test_robustness.py`: Verify empty candidate returns Dice=0, empty reference returns Dice=NaN.
- `test_metrics_accuracy.py`: Verify synthetic bundles give expected Dice/MDF.

### Manual Verification
1. Run `csttool validate` on `sub-1282` against `PYT` bundles.
2. Verify visual snapshots align with expectation.
3. Verify JSON contains all metadata and reasonable metrics (Dice significantly > 0).

# ROI Asymmetry Investigation

**Date**: 2026-01-19  
**Status**: Fix Implemented (Pending Verification)  
**Issue**: Motor cortex ROIs are asymmetric even on healthy control datasets


## Problem Description

When visualizing the left and right motor cortex ROIs overlaid on subject FA maps, the ROIs appear visibly asymmetric:
- Left motor cortex extends further laterally
- Right motor cortex appears smaller and shifted
- This occurs even on healthy control datasets where symmetry is expected

**Observed in**: `/home/alem/data/thesis/out/ds003900/derivatives/trainset/sub-1282/dwi/extraction/visualizations/`

## Investigation Summary

### Step 1: Quantify the Asymmetry

Analyzed the saved ROI masks for sub-1282:

| Mask | Voxel Count | World X Range | Mean X |
|------|-------------|---------------|--------|
| Motor Left | 35,594 | [-62.5, 0.5] | -31.8 |
| Motor Right | 29,355 | [-0.5, 57.5] | 27.9 |

**L/R Ratio**: 1.21 (21% more voxels on left)

### Step 2: Check Original Atlas Symmetry

Analyzed the Harvard-Oxford cortical atlas in MNI space (before warping):

| Metric | Value |
|--------|-------|
| Precentral Gyrus (label 7) | 70,031 voxels |
| Left (X < 0) | 35,272 voxels |
| Right (X >= 0) | 34,759 voxels |
| **L/R Ratio** | **1.01** (nearly symmetric) |

**Conclusion**: The original atlas is symmetric. Asymmetry is introduced during warping.

### Step 3: Check Warped Atlas Before Hemisphere Splitting

Analyzed the warped cortical atlas in subject space:

| Metric | Value |
|--------|-------|
| Warped Precentral (label 7) | 50,044 voxels |
| Left (X < 0) | 27,650 voxels |
| Right (X >= 0) | 22,394 voxels |
| **L/R Ratio** | **1.23** (23% asymmetry) |
| **Mean X** | -5.1 (shifted left from midline) |

**Conclusion**: The affine registration is introducing asymmetry.

### Step 4: Test Alternative Atlas (Juelich)

Tested the Juelich histological atlas (BA4a + BA4p combined) as an alternative:

| Atlas | Original L/R Ratio | Warped L/R Ratio | Asymmetry |
|-------|-------------------|------------------|-----------|
| Harvard-Oxford (Precentral) | 1.01 | 1.24 | 23.5% |
| Juelich (BA4a+BA4p) | 1.03 | 1.71 | **70.7%** |

**Conclusion**: Juelich is MORE asymmetric after warping because its smaller ROI is more sensitive to registration errors.

## Root Cause Analysis

The asymmetry is introduced by the **affine registration process** (MNI T1 → Subject FA):

1. **Multi-modal registration**: T1-weighted template to FA map is challenging
2. **No symmetry constraint**: Standard affine registration doesn't enforce bilateral symmetry
3. **Local deformations**: Even small registration errors cause visible asymmetry in cortical ROIs
4. **Smaller ROIs are more affected**: Juelich's localized BA4 regions are more sensitive than Harvard-Oxford's larger precentral gyrus

## Potential Solutions

### Option A: Improve Registration Quality
- Add symmetric regularization to the affine registration
- Use more iterations or different optimization parameters
- Test with **FA-based template** (JHU-MNI-FA) instead of T1 template for better modal matching

### Option B: Post-hoc Symmetrization
- After warping, enforce symmetry by mirroring one hemisphere to match the other
- Average left and right halves
- **Cons**: Artificial, may not match actual anatomy

### Option C: Use Separate Hemisphere Registrations
- Register each hemisphere separately with symmetric constraints
- **Cons**: Complex implementation

## Investigation Results

### Test 1: Alternative Atlas (Juelich)
Tested Juelich histological atlas (BA4a + BA4p) as alternative to Harvard-Oxford.

| Atlas | Original L/R Ratio | Warped L/R Ratio | Asymmetry |
|-------|-------------------|------------------|-----------|
| Harvard-Oxford | 1.01 | 1.24 | 23.5% |
| Juelich (BA4a+BA4p) | 1.03 | 1.71 | **70.7%** |

**Result**: Juelich is MORE asymmetric after warping because its smaller ROI is more sensitive to registration errors. **Keep Harvard-Oxford**.

### Test 2: FA-to-FA Registration
Tested registration using DIPY's HCP FA template (`~/.dipy/bundle_fa_hcp/hcp_bundle_fa.nii.gz`).

**Result**: Failed due to grid mismatch between HCP FA template (145×174×145) and Harvard-Oxford atlas (182×218×182).

### Test 3: Re-enabling SyN Non-linear Registration ✓
Tested enabling SyN on top of affine registration (currently disabled in pipeline).

| Registration Method | L/R Ratio | Asymmetry |
|---------------------|-----------|-----------|
| Affine-only (current) | 1.235 | 23.5% |
| Affine + SyN | 1.130 | **13.0%** |

**Result**: ✓ **SyN improved symmetry by 10.5%!**

## Solution Implementation (2026-01-19)

### FA-to-FA Registration with Template Resampling
Addressed the grid mismatch issue found in **Test 2**.

-   **Problem**: HCP FA template (145×174×145) dims differed from Harvard-Oxford Atlas (MNI T1 grid 182×218×182).
-   **Solution**: Implemented `load_hcp_fa_template` in `registration.py`.
    -   Loads HCP FA template.
    -   Resamples it to the MNI T1 grid using `dipy.align.resample`.
    -   Uses this resampled FA template for registration against subject FA.
    -   The resulting mapping (MNI Grid -> Subject) is then valid for warping the Harvard-Oxford atlas.
-   **Status**: Code implemented. **Pending Testing** to quantify symmetry improvement.


## Recommendation

**Status**: ✅ SyN re-enabled and tested

### Results
- **Affine-only**: 23.5% asymmetry
- **Affine + SyN**: 13.3% asymmetry
- **Improvement**: 10.2%

### Assessment
While SyN significantly improves symmetry, **13.3% asymmetry is still visually noticeable** and may not be acceptable for healthy control datasets where near-perfect bilateral symmetry is expected.

### Next Steps

Since registration-based approaches have reached their limit, we should implement **post-hoc symmetrization**:

1. **Option A: Mirror averaging** - Average the left and right ROIs after warping
2. **Option B: Use larger ROI** - Mirror the larger hemisphere to the smaller side
3. **Option C: Anatomical constraints** - Use anatomical priors to enforce symmetry during hemisphere splitting

**Recommended**: Option A (mirror averaging) as it's the most conservative and preserves anatomical information from both hemispheres.

## Related Files

- `src/csttool/extract/modules/registration.py` - Registration implementation (✅ SyN re-enabled)
- `src/csttool/extract/modules/warp_atlas_to_subject.py` - Atlas warping
- `src/csttool/extract/modules/create_roi_masks.py` - ROI mask creation with hemisphere splitting (needs symmetrization)

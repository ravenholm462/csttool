# ROI Asymmetry Investigation - Part 2

**Date**: 2026-01-30
**Status**: Diagnostic Phase
**Issue**: Motor cortex ROIs appear visually asymmetric despite similar voxel counts

## Problem Summary

Motor cortex ROIs appear visually asymmetric despite having similar voxel counts (~6% difference). The right ROI extends inferiorly into anatomically incorrect regions, suggesting poor regional registration quality or spatial fragmentation.

## Evidence Collected

### Voxel Count Analysis

| Stage | Left | Right | L/R Ratio | Notes |
|-------|------|-------|-----------|-------|
| MNI split (before warp) | 35,272 | 34,759 | 1.015 | Nearly symmetric |
| Post-warp | 33,550 | 35,890 | 0.935 | Ratio flipped |
| Post-dilation (1 iter) | 42,289 | 44,822 | 0.943 | ~6% difference |

### Key Observations

1. **MNI hemisphere split is symmetric** - The X=0 boundary split produces nearly equal voxel counts (1.5% difference)

2. **Ratio flips during warping** - In MNI space, left is slightly larger. After warping to subject space, right becomes larger. This indicates asymmetric deformation.

3. **Visual vs numerical mismatch** - The 6% voxel count difference cannot explain the dramatic visual asymmetry where right appears 2-3x larger than left.

4. **Right ROI extends inferiorly** - In coronal view, the right motor ROI extends far inferior, almost to brainstem level. This does not match precentral gyrus anatomy.

## Hypotheses

### Primary Hypothesis: Asymmetric Registration Quality

The SyN diffeomorphic registration may be fitting one hemisphere better than the other, causing:
- Different local deformation magnitudes per hemisphere
- Labels warping to incorrect anatomical locations
- Spatial fragmentation that appears larger on 2D slices

### Secondary Hypothesis: Spatial Distribution Issue

Even with similar total voxel counts, the ROIs may have very different spatial distributions:
- One ROI could be compact/spherical
- The other could be elongated/fragmented
- Middle slices used for visualization may not be representative

## Diagnostic Plan

### 1. Jacobian Determinant Analysis

Compute the Jacobian determinant of the deformation field per hemisphere:
- J > 1: local expansion
- J < 1: local compression
- J = 1: no volume change

Compare statistics (mean, std, negative %) between left and right hemispheres to quantify registration asymmetry.

### 2. ROI Centroid Reporting

Report world coordinates of each motor ROI centroid after warping:
- Left motor should have X < 0
- Right motor should have X > 0
- Both should have similar Z coordinates (superior position)

### 3. Jacobian Visualization

Create a Jacobian map overlaid on FA showing:
- Red regions: local expansion
- Blue regions: local compression
- Asymmetric patterns indicate hemisphere-specific registration issues

## Implementation

See code changes in:
- `src/csttool/extract/modules/registration.py` - Jacobian analysis function
- `src/csttool/extract/modules/visualizations.py` - Jacobian visualization
- `src/csttool/extract/modules/warp_atlas_to_subject.py` - ROI centroid diagnostics

## Next Steps

Based on diagnostic findings:

1. **If Jacobian is asymmetric** → Tune SyN parameters or use different registration approach
2. **If centroids are misplaced** → Investigate atlas/orientation issues
3. **If both look reasonable** → Implement connected component filtering to remove scattered fragments

## Related Documentation

- [roi_asymmetry_investigation.md](roi_asymmetry_investigation.md) - Original fix (MNI space hemisphere splitting)
- [extract.md](../reference/cli/extract.md) - CLI documentation
- [extract.md](../reference/api/extract.md) - API documentation

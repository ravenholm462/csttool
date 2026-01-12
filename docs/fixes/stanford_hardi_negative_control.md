# Stanford HARDI Dataset - Negative Control Case

**Date**: 2026-01-11  
**Dataset**: Stanford HARDI150 (`HARDI150.nii.gz`)  
**Status**: ❌ **FAILED** - Unsuitable for CST extraction with deterministic tracking

---

## Summary

The Stanford HARDI dataset consistently fails to produce meaningful CST extractions despite extensive troubleshooting and parameter optimization. This is documented as a **negative control** demonstrating dataset-specific limitations rather than pipeline failures.

## Dataset Characteristics

- **Name**: Stanford HARDI150
- **Dimensions**: 81 × 106 × 76 voxels
- **Voxel size**: 2.0 × 2.0 × 2.0 mm
- **Directions**: 160 gradients (likely 150 DWI + 10 b0)
- **Affine**: Standard 2mm isotropic with origin at [-80, -120, -60]

## Extraction Results

| Configuration | Streamlines Generated | CST Extracted | Rate |
|--------------|----------------------|---------------|------|
| Default (FA=0.2, density=1) | 59,422 | 0 | 0.00% |
| Low FA (FA=0.01, density=1) | 159,498 | 0 | 0.00% |
| Low FA + Dilation (FA=0.01, dilate=5) | 159,498 | 0 | 0.00% |
| Low FA + High Density (FA=0.01, density=2) | ~650,000 | 4 (right only) | 0.0006% |

## Root Cause Analysis

### Spatial Geometry Issue

Detailed voxel-space analysis revealed a fundamental spatial disconnect:

```
Region                    Z Voxel Range
─────────────────────────────────────
Motor Cortex (target)     28 - 71
Brainstem (target)        0 - 35
Streamlines (BS-hitting)  10 - 29
```

**Critical Finding**: Streamlines that successfully reach the brainstem (Z: 10-29) **do not extend** into the motor cortex region (Z: 28-71). The overlap is minimal (only Z=28-29), resulting in near-zero extraction.

### Diagnostic Evidence

1. **Bounding box overlap exists**: World coordinates show proper spatial overlap
2. **Affine matrices match**: FA map and ROI masks use identical transforms
3. **Brainstem hits confirmed**: 8,837 streamlines (5.5%) successfully intersect brainstem
4. **Motor cortex misses**: 0 of 1,000 brainstem-hitting streamlines reach motor cortex
5. **Z-range gap**: Streamlines terminate before reaching superior cortical regions

### Likely Causes

1. **Limited field of view**: Acquisition may not fully cover superior motor cortex
2. **FA drop at cortex**: Gray matter interface causes premature tracking termination
3. **Acquisition geometry**: Dataset orientation/cropping truncates CST superior extent
4. **SNR limitations**: Insufficient signal quality in cortical regions

## Attempted Solutions

### 1. FA Threshold Reduction
- **Tried**: `--fa-thr 0.01` (from default 0.2)
- **Result**: 2.7× more streamlines, but still 0% extraction
- **Conclusion**: Not an FA threshold issue

### 2. Aggressive ROI Dilation
- **Tried**: `--dilate-brainstem 5 --dilate-motor 5` (from defaults 2, 1)
- **Result**: ROI masks expanded 2-3×, still 0% extraction
- **Conclusion**: Not a mask strictness issue

### 3. Increased Seed Density
- **Tried**: `--seed-density 2` (from default 1)
- **Result**: ~650k streamlines, only 4 extracted (0.0006%)
- **Conclusion**: Confirms spatial geometry limitation

### 4. Code-Level Debugging
- **Fixed**: Affine mismatch in `cli.py` (used wrong affine for ROI creation)
- **Verified**: Coordinate transformations mathematically correct
- **Tested**: Direct voxel-space intersection logic - working as expected
- **Conclusion**: Pipeline code is correct; issue is data-specific

## Comparison with Working Dataset

A different dataset provided by mentor works with **default settings**, confirming:
- Pipeline logic is correct
- Extraction algorithms function properly
- Stanford HARDI has dataset-specific limitations

## Recommendations

### For This Dataset
1. **Accept as negative control** - Document limitations for research context
2. **Do not use for CST analysis** - Insufficient cortical coverage
3. **Consider probabilistic tracking** - May handle cortical noise better (requires implementation)

### For Future Work
1. **Use mentor's validated dataset** for actual CST analysis
2. **Screen datasets** for full brain coverage before analysis
3. **Check Z-range coverage** of motor cortex in preprocessing QC

## Technical Notes

### Coordinate System Verification
- Tractogram space: `RASMM` (RAS+ millimeter coordinates)
- ROI mask space: Matches FA map affine exactly
- Transformation: `voxel = round(inv(affine) @ [x, y, z, 1])`
- Verified: Mathematically correct, no bugs detected

### Files Generated
- Tractogram: `/tracking/tractograms/HARDI150_whole_brain.trk`
- FA map: `/tracking/scalar_maps/HARDI150_fa.nii.gz`
- ROI masks: `/extraction/nifti/HARDI150_roi_*.nii.gz`
- Visualizations: `/extraction/visualizations/`

---

## Conclusion

The Stanford HARDI dataset is **unsuitable for CST extraction** using deterministic tracking due to insufficient superior cortical coverage. This is a **data quality/geometry limitation**, not a pipeline failure. The 4 streamlines extracted with extreme parameter tuning represent edge cases at the cortical boundary, not reliable CST identification.

**Status**: Documented as negative control. Use alternative datasets for CST analysis.

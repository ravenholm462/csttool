# ROI Asymmetry Investigation - Part 3

**Date**: 2026-01-30
**Status**: Diagnostic Phase Complete - Root Cause Identified
**Issue**: CST streamline counts show 18% asymmetry (L/R = 0.82) despite similar motor ROI voxel counts

## Summary

Added per-stage diagnostic output to `passthrough_filtering.py` to trace where asymmetry is introduced. The diagnostics reveal that:

1. **Brainstem check causes the asymmetry flip**, not motor cortex
2. **Motor ROI conditional yields are equal** (~15% both sides)
3. **Right motor ROI is mispositioned** - extends 9.5mm into left hemisphere

## Diagnostic Output (sub-1280)

```
Per-Stage Asymmetry Analysis:
    Stage                           Left      Right     L/R Ratio
    ------------------------------------------------------------
    Input (post-length filter)   138,977   111,538     1.246
    Pass through brainstem        18,181    23,152     0.785
    Pass through motor cortex      2,762     3,366     0.821
    Final (after exclusions)       2,762     3,366     0.821

    Conditional motor yields (P(motor | brainstem+hemisphere)):
      Left:   15.2% (2,762 / 18,181)
      Right:  14.5% (3,366 / 23,152)
      Yield ratio (L/R): 1.045

    Motor ROI Geometry:
    motor_left:
        COM (mm): X=-35.5, Y=9.9, Z=37.9
        X extent: [-61.5, -4.5] mm
        Distance from midline: 35.5 mm
        Voxel count: 42,289
    motor_right:
        COM (mm): X=25.3, Y=9.3, Z=42.8
        X extent: [-9.5, 54.5] mm
        Distance from midline: 25.3 mm
        Voxel count: 44,822
    → right motor ROI is 10.2 mm closer to midline
```

## Key Findings

### 1. Brainstem Check Causes the Flip

| Stage | L/R Ratio | Change |
|-------|-----------|--------|
| Input | 1.246 | - |
| Brainstem | 0.785 | Δ = -0.461 (FLIP) |
| Motor | 0.821 | Δ = +0.036 |
| Final | 0.821 | Δ = 0 |

The brainstem gate transforms a left-heavy input (25% more left) into a right-heavy pool (27% more right). This is the dominant asymmetry injector.

### 2. Motor ROI Yields Are Equal

```
Left yield:  15.2%
Right yield: 14.5%
Ratio: 1.045 ≈ 1.0
```

The motor cortex ROIs catch streamlines at nearly identical rates. Motor ROI geometry is NOT causing the asymmetry in final counts.

### 3. Right Motor ROI Crosses Midline

```
motor_left:  X extent [-61.5, -4.5] mm  → stays in left hemisphere
motor_right: X extent [-9.5, 54.5] mm  → extends INTO left hemisphere
```

The right motor ROI:
- COM is 10.2 mm closer to midline than left (25.3 vs 35.5 mm)
- Extends 9.5 mm into the left hemisphere (X = -9.5 mm)
- Has 6% more voxels (44,822 vs 42,289)

This indicates a registration/warping issue or atlas hemisphere split problem.

### 4. Brainstem Pass-Through Rates Differ

| Hemisphere | Input | Pass Brainstem | Rate |
|------------|-------|----------------|------|
| Left | 138,977 | 18,181 | 13.1% |
| Right | 111,538 | 23,152 | 20.8% |

Right hemisphere streamlines pass through brainstem at 1.6x the rate of left. Possible causes:
- Brainstem ROI asymmetry
- Streamline trajectory differences
- Centroid-based hemisphere classification mismatch

## Interpretation

### What We Learned

1. **Motor ROI geometry asymmetry exists but doesn't affect CST yield**
   - The equal conditional yields (15.2% vs 14.5%) prove this
   - The mispositioned right ROI still catches the same proportion

2. **Brainstem is the asymmetry source**
   - Either the brainstem ROI is asymmetric
   - Or streamlines have genuinely different trajectories per hemisphere
   - Or the centroid-based hemisphere classification differs from actual anatomy

3. **Input tractogram is left-heavy, but output is right-heavy**
   - The pipeline reverses the bias
   - This is unexpected and indicates a systematic issue

### What Remains Unknown

1. **Brainstem ROI geometry** - We haven't analyzed its COM, extent, symmetry
2. **Why right motor ROI crosses midline** - Registration issue? Atlas boundary issue?
3. **Whether centroid-based classification is consistent** - May differ from anatomical midline

## Next Steps

### Immediate (Tomorrow)

1. **Add brainstem ROI geometry logging**
   - Print COM, X/Y/Z extent for brainstem ROI
   - Check if brainstem is positioned asymmetrically

2. **Investigate right motor ROI crossing midline**
   - Check `split_atlas_hemispheres_mni()` in `warp_atlas_to_subject.py`
   - Verify the X=0 split is applied correctly
   - Check if the issue is pre-warp (MNI space) or post-warp (subject space)

3. **Compare hemisphere classification methods**
   - Current: centroid X < 0 = left
   - Alternative: majority of points in left/right
   - Check if mismatch exists

### If Brainstem is Asymmetric

- Review brainstem ROI extraction (label 49 from subcortical atlas)
- Check if dilation is symmetric
- Consider separate L/R brainstem ROIs

### If Motor ROI Positioning is the Issue

- Review the MNI hemisphere split logic
- Check registration quality for motor cortex region specifically
- Consider adding connected component filtering to remove fragments

## Code Changes Made

### File: `src/csttool/extract/modules/passthrough_filtering.py`

Added Phase 1 and Phase 2 diagnostics:

1. **Input hemisphere distribution** - Centroid-based L/R classification
2. **Brainstem hemisphere split** - `left_bs`, `right_bs`, `lr_bs`
3. **Conditional motor yields** - P(motor | brainstem + hemisphere)
4. **ROI geometry logging** - COM, X extent, midline distance
5. **Asymmetry change detection** - Highlights which stage introduces asymmetry

New helper functions:
- `get_roi_geometry(mask, affine, name)` - Computes ROI spatial metrics
- `print_roi_geometry(geom)` - Formatted output

New stats fields:
- `left_bs`, `right_bs`, `lr_bs` - Brainstem hemisphere split
- `left_motor_yield`, `right_motor_yield` - Conditional yields

## Related Documentation

- [roi_asymmetry_investigation.md](roi_asymmetry_investigation.md) - Part 1: MNI space hemisphere splitting fix
- [roi_asymmetry_investigation_2.md](roi_asymmetry_investigation_2.md) - Part 2: Jacobian analysis (symmetric)
- [extract.md](../reference/cli/extract.md) - CLI documentation

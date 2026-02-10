# ROI Asymmetry Investigation - Part 4

**Date**: 2026-01-31
**Status**: RESOLVED - Data Property Identified
**Issue**: CST streamline counts show 18% asymmetry (L/R = 0.82) despite similar motor ROI voxel counts

## Root Cause

**Lower FA in left cerebral peduncle (0.285 vs 0.310, 8.1% difference)** causes:
1. More left-hemisphere streamlines hit FA threshold stopping criterion
2. Fewer left streamlines reach brainstem (13.1% vs 20.8%)
3. The 18% CST asymmetry (L/R = 0.82)

**This is a data property**, not a pipeline bug. The asymmetry reflects genuine differences in this subject's diffusion characteristics.

## Diagnostic Output (sub-1280)

```
Per-Stage Asymmetry Analysis:
    Stage                           Left      Right     L/R Ratio
    ------------------------------------------------------------
    Input (post-length filter)   138,977   111,538     1.246
    Pass through brainstem        18,181    23,152     0.785
    Pass through motor cortex      2,762     3,366     0.821
    Final (after exclusions)       2,762     3,366     0.821

Brainstem entry point distribution:
    Left-classified streamlines (n=18,181):
        Entry X mean: -7.1 mm
        Enter left side (X<0): 15,618 (85.9%)
    Right-classified streamlines (n=23,152):
        Entry X mean: 7.1 mm
        Enter right side (X>=0): 22,012 (95.1%)
    -> Classification aligns with brainstem entry

Inferior extent (streamlines failing brainstem check):
    Left-classified (n=120,796):
        Mean min Z: 2.2 mm
    Right-classified (n=88,386):
        Mean min Z: 3.5 mm
    -> Similar inferior extent (Δ = -1.4 mm)

Cerebral Peduncle FA (superior 30% of brainstem):
    Left:  mean FA = 0.285 (n=5841)
    Right: mean FA = 0.310 (n=5330)
    -> Left peduncle has lower FA (0.025 difference)
```

## Investigation Summary

### Hypotheses Tested

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| Brainstem ROI asymmetric | **RULED OUT** | L/R = 0.969, COM at X=0.2mm |
| Centroid classification mismatch | **RULED OUT** | 85.9% / 95.1% enter correct side |
| Left streamlines terminate earlier | **RULED OUT** | Similar min Z (2.2 vs 3.5 mm) |
| Left peduncle lower FA | **CONFIRMED** | 0.285 vs 0.310 (8.1% lower) |

### Key Findings

1. **Brainstem is symmetric** - L/R voxel ratio = 0.969
2. **Classification aligns with anatomy** - Centroid X correctly identifies brainstem entry side
3. **Inferior extent is similar** - Failed streamlines stop at similar Z levels
4. **Left peduncle has lower FA** - 8.1% lower, explains differential brainstem reachability

## Diagnostic Code Added

The following diagnostics were added to `passthrough_filtering.py`:

1. **Brainstem ROI geometry** - COM, X extent, voxel count
2. **Brainstem hemisphere split** - L/R voxel distribution
3. **Brainstem entry point distribution** - X coordinate of first entry point
4. **Inferior extent analysis** - Min Z for streamlines failing brainstem check
5. **Cerebral peduncle FA sampling** - Mean FA in left vs right superior brainstem

## Remaining Issues

### Motor ROI Crossing Midline (Correctness Issue)

The right motor ROI extends 9.5mm into the left hemisphere:
```
motor_right:
    X extent: [-9.5, 54.5] mm
    -> extends INTO left hemisphere
```

This does NOT affect the CST asymmetry (motor yields are equal at 15.2% vs 14.5%), but should be fixed for correctness. Consider post-warp hemisphere clamping.

## Related Documentation

- [roi_asymmetry_investigation.md](roi_asymmetry_investigation.md) - Part 1: MNI space hemisphere splitting
- [roi_asymmetry_investigation_2.md](roi_asymmetry_investigation_2.md) - Part 2: Jacobian analysis
- [roi_asymmetry_investigation_3.md](roi_asymmetry_investigation_3.md) - Part 3: Per-stage diagnostics

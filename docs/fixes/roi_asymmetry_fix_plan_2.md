# Plan: Diagnose CST Streamline Asymmetry Root Cause

## Problem Statement

Motor cortex ROI voxel counts are similar (~6% difference), but resulting CST streamline counts show ~18% asymmetry (L/R ratio 0.82). The asymmetry is being **amplified during tracking**, not at ROI definition.

**Evidence:**
- Jacobian analysis: symmetric (mean L=0.999, R=1.004)
- ROI voxel counts: ~6% difference (expected)
- Streamline counts: 2,762 (L) vs 3,366 (R) = 18% difference

## Diagnostic Results (sub-1280)

```
Per-Stage Asymmetry Analysis:
    Stage                           Left      Right     L/R Ratio
    ------------------------------------------------------------
    Input (post-length filter)   138,977   111,538     1.246
    Pass through brainstem        41,333       -           -
    Pass through motor cortex      2,762     3,366     0.821
    Final (after exclusions)       2,762     3,366     0.821
```

### Key Finding: Motor ROI OVERRIDES Input Bias

- **Input tractogram**: LEFT-biased (L/R = 1.246, 25% more left)
- **After motor cortex**: RIGHT-biased (L/R = 0.821, 18% more right)
- **Change**: Δ = -0.425 (sign flip)

### Corrected Interpretation

The motor ROI is not "amplifying" tracking bias - it's **overriding** it in the opposite direction. This strongly suggests motor ROI interaction with streamlines is the dominant factor, not the tractogram's global hemisphere count.

### Missing Diagnostic: Brainstem Hemisphere Split

The table shows brainstem total (41,333) but NOT the L/R breakdown. We cannot determine if:
- The brainstem gate already reshapes the pool
- Or if motor ROIs alone are responsible

### What's Needed

1. **Brainstem hemisphere split**: Left_bs, Right_bs, L/R_bs
2. **Conditional motor pass rates**:
   - P(pass motor_left | pass brainstem AND left hemisphere)
   - P(pass motor_right | pass brainstem AND right hemisphere)
3. **Hemisphere assignment consistency**: Verify same method used at all stages (centroid X vs midsagittal plane)

### Root Cause Hypotheses

1. **ROI geometry mismatch**: Right motor ROI COM closer to midline or better aligned with streamline trajectories
2. **Overlap with WM**: One ROI intersects more white matter due to gyral placement
3. **Internal capsule proximity**: One ROI shifted toward the typical streamline density region

## Already Implemented (Phase 1)

Added to `passthrough_filtering.py`:
- Input hemisphere distribution (centroid X)
- Motor cortex pass counts per hemisphere
- Basic asymmetry change detection

This produced the diagnostic results above, revealing the motor ROI flip.

## Next Implementation: Refined Diagnostics (Phase 2)

### 1. Add Brainstem Hemisphere Split
**File:** `src/csttool/extract/modules/passthrough_filtering.py`

After brainstem check, classify by hemisphere using centroid X:
```python
# Track brainstem passes by hemisphere
left_bs_count = 0
right_bs_count = 0

for sl in streamlines_filtered:
    if streamline_passes_through(sl, brainstem, affine):
        if np.mean(sl[:, 0]) < 0:
            left_bs_count += 1
        else:
            right_bs_count += 1
```

Output:
```
Pass through brainstem       Left_bs    Right_bs    L/R_bs
```

### 2. Add Conditional Motor Pass Rates
**File:** `src/csttool/extract/modules/passthrough_filtering.py`

Calculate yield per hemisphere:
```python
# P(pass motor_left | pass brainstem AND left hemisphere)
left_yield = passes_left_motor_count / left_bs_count if left_bs_count > 0 else 0

# P(pass motor_right | pass brainstem AND right hemisphere)
right_yield = passes_right_motor_count / right_bs_count if right_bs_count > 0 else 0
```

Output:
```
Conditional motor yields:
    Left:  X.X% (N/M pass motor_left | brainstem+left)
    Right: X.X% (N/M pass motor_right | brainstem+right)
```

### 3. Add ROI Geometry Logging
**File:** `src/csttool/extract/modules/passthrough_filtering.py` or `create_roi_masks.py`

For each motor ROI, print:
- COM (center of mass) in world mm
- COM X distance from midline
- X extent (min_x, max_x)

```python
from scipy.ndimage import center_of_mass

def get_roi_geometry(mask, affine, name):
    coords = np.argwhere(mask > 0)
    com_vox = center_of_mass(mask)
    com_world = nib.affines.apply_affine(affine, com_vox)

    world_coords = nib.affines.apply_affine(affine, coords)
    x_min, x_max = world_coords[:, 0].min(), world_coords[:, 0].max()

    print(f"    {name}:")
    print(f"        COM (mm): X={com_world[0]:.1f}, Y={com_world[1]:.1f}, Z={com_world[2]:.1f}")
    print(f"        X extent: [{x_min:.1f}, {x_max:.1f}] mm")
    print(f"        Distance from midline: {abs(com_world[0]):.1f} mm")
```

## Expected Output After Implementation

```
Per-Stage Asymmetry Analysis:
    Stage                           Left      Right     L/R Ratio
    ------------------------------------------------------------
    Input (post-length filter)   138,977   111,538     1.246
    Pass through brainstem        25,000    16,333     1.531    ← NEW
    Pass through motor cortex      2,762     3,366     0.821
    Final (after exclusions)       2,762     3,366     0.821

Conditional motor yields:
    Left:  11.0% (2,762 / 25,000)                                ← NEW
    Right: 20.6% (3,366 / 16,333)                                ← NEW

Motor ROI Geometry:                                               ← NEW
    motor_left:
        COM (mm): X=-35.2, Y=-18.4, Z=62.1
        X extent: [-52.0, -18.0] mm
        Distance from midline: 35.2 mm
    motor_right:
        COM (mm): X=28.5, Y=-17.9, Z=61.8
        X extent: [12.0, 45.0] mm
        Distance from midline: 28.5 mm            ← Closer to midline!
```

This reveals:
- Whether brainstem already reshapes the hemisphere pool
- Whether motor yield differs significantly per hemisphere
- Whether ROI geometry explains the yield difference

## Files to Modify

| File | Change |
|------|--------|
| `src/csttool/extract/modules/passthrough_filtering.py` | Add brainstem L/R split, conditional yields, ROI geometry |

## Verification

Re-run on sub-1280 and check:
1. Does brainstem gate change the L/R ratio significantly?
2. Is the conditional motor yield asymmetric (right >> left)?
3. Is the right motor ROI COM closer to midline?

# Plan: Diagnose CST Streamline Asymmetry Root Cause

## Problem Statement

Motor cortex ROI voxel counts are similar (~6% difference), but resulting CST streamline counts show ~18% asymmetry (L/R ratio 0.82). The asymmetry is being **amplified during tracking**, not at ROI definition.

**Evidence:**
- Jacobian analysis: symmetric (mean L=0.999, R=1.004)
- ROI voxel counts: ~6% difference (expected)
- Streamline counts: 2,762 (L) vs 3,366 (R) = 18% difference

## Key Finding: Diagnostics Already Exist

The `roi_seeded_tracking.py` module (lines 429-475) already collects per-hemisphere stats:
- `left_seeds` / `right_seeds` - seed counts
- `left_raw` / `right_raw` - streamlines before length filter
- `left_after_length` / `right_after_length` - after length filter
- `cst_left_count` / `cst_right_count` - final counts
- `left_yield` / `right_yield` - success percentages

**However**, the verbose output only prints final counts, not intermediate stages.

## Implementation Plan (Passthrough Method)

Since you're using the **passthrough filtering** method, the diagnostic focus is different. Passthrough filtering takes a whole-brain tractogram and filters by ROI traversal.

### Current Flow (passthrough_filtering.py)
```
Input: whole-brain tractogram → length filter → brainstem check → motor ROI check → output
```

### 1. Add Intermediate Per-Hemisphere Counts
**File:** `src/csttool/extract/modules/passthrough_filtering.py` (lines 88-127)

Track counts at each decision point:

```python
# New counters (add near line 90)
passes_brainstem_count = 0
passes_left_motor_count = 0
passes_right_motor_count = 0

for i, sl in enumerate(streamlines_filtered):
    passes_bs = streamline_passes_through(sl, brainstem, affine)

    if passes_bs:
        passes_brainstem_count += 1  # NEW: count brainstem hits

        passes_left = streamline_passes_through(sl, motor_left, affine)
        passes_right = streamline_passes_through(sl, motor_right, affine)

        # NEW: count before exclusion logic
        if passes_left:
            passes_left_motor_count += 1
        if passes_right:
            passes_right_motor_count += 1

        # ... rest of exclusion logic
```

### 2. Add Stats and Verbose Output
**File:** `src/csttool/extract/modules/passthrough_filtering.py` (lines 129-150)

Update stats dict and verbose output:

```python
stats = {
    'total_input': len(streamlines),
    'after_length_filter': len(streamlines_filtered),
    # NEW: intermediate counts
    'passes_brainstem': passes_brainstem_count,
    'passes_left_motor': passes_left_motor_count,  # Before exclusion
    'passes_right_motor': passes_right_motor_count,  # Before exclusion
    # Existing
    'cst_left_count': len(cst_left),
    'cst_right_count': len(cst_right),
    'bilateral_excluded': bilateral_excluded_count,
    'midline_excluded': midline_excluded_count,
    ...
}

# NEW: Diagnostic summary
if verbose:
    lr_motor = passes_left_motor_count / passes_right_motor_count if passes_right_motor_count > 0 else 0
    lr_final = len(cst_left) / len(cst_right) if len(cst_right) > 0 else 0

    print("\nPer-Stage Asymmetry Analysis:")
    print("    Stage                           Left      Right     L/R Ratio")
    print("    " + "-" * 60)
    print(f"    Pass through motor cortex      {passes_left_motor_count:>8,}  {passes_right_motor_count:>8,}     {lr_motor:.3f}")
    print(f"    Final (after exclusions)       {len(cst_left):>8,}  {len(cst_right):>8,}     {lr_final:.3f}")
```

### 3. Add Input Tractogram Hemisphere Analysis
**File:** `src/csttool/extract/modules/passthrough_filtering.py` or `src/csttool/cli/commands/extract.py`

The whole-brain tractogram is loaded from `args.tractogram` (user-provided). Before filtering, analyze hemisphere distribution:

```python
# Count input streamlines by hemisphere (centroid X coordinate)
left_input = sum(1 for sl in streamlines_filtered if np.mean(sl[:, 0]) < 0)
right_input = sum(1 for sl in streamlines_filtered if np.mean(sl[:, 0]) >= 0)

print(f"Input tractogram hemisphere distribution:")
print(f"    Left (X<0):  {left_input:,}")
print(f"    Right (X≥0): {right_input:,}")
print(f"    L/R Ratio:   {left_input/right_input:.3f}")
```

This reveals if asymmetry already exists in the input tractogram.

## Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `src/csttool/extract/modules/passthrough_filtering.py` | 88-150 | Add intermediate counts + diagnostic output |

## Expected Output After Implementation

```
PASS-THROUGH CST EXTRACTION
============================================================

Input: 250,000 streamlines

[Step 1/2] Length filtering (20-200 mm)...
    250,000 → 180,000 streamlines

Input tractogram hemisphere distribution:
    Left (X<0):  88,000
    Right (X≥0): 92,000
    L/R Ratio:   0.957

[Step 2/2] Pass-through filtering...

Per-Stage Asymmetry Analysis:
    Stage                           Left      Right     L/R Ratio
    ------------------------------------------------------------
    Input (post-length filter)     88,000    92,000     0.957    ← Small input bias
    Pass through brainstem          5,500     6,200     0.887    ← Brainstem geometry
    Pass through motor cortex       3,200     4,100     0.780    ← Motor ROI asymmetry
    Final (after exclusions)        2,762     3,366     0.821

    Asymmetry introduced at:
    → Brainstem check (changed from 0.957 to 0.887)
    → Motor cortex check (changed from 0.887 to 0.780)

    Rejection breakdown:
    Bilateral excluded: 150
    Midline excluded: 380
```

This pinpoints exactly where the L/R ratio changes occur.

## Verification

1. Re-run pipeline on sub-1280 with verbose output:
   ```bash
   csttool extract --fa /path/to/sub-1280_fa.nii.gz \
                   --tractogram /path/to/sub-1280_whole_brain.trk \
                   --extraction-method passthrough \
                   --out /path/to/output
   ```

2. Review the new diagnostic output to identify where L/R ratio diverges

3. Compare the stage-by-stage ratios to identify the primary source of asymmetry

## Interpretation Guide (After Running)

| If asymmetry appears at... | Root cause | Next action |
|---------------------------|------------|-------------|
| Input tractogram | Whole-brain tracking has hemisphere bias | Check seeding mask, FA map for asymmetry |
| Brainstem check | Brainstem ROI positioning/size differs | Inspect brainstem mask positioning |
| Motor cortex check | Motor ROI geometry/positioning differs | Compare L/R motor ROI shapes, centroids |
| After exclusions | Exclusion logic biased | Review bilateral/midline thresholds |

## Next Steps (After Diagnosis)

Based on findings:
- **Input asymmetric**: The whole-brain tracking needs investigation (separate from ROI issues)
- **Brainstem stage**: Check brainstem ROI centering and dilation
- **Motor stage**: Compare motor ROI centroids, bounding boxes, and shapes
- **Exclusion stage**: Consider adjusting `MIDLINE_TOLERANCE_MM` or bilateral logic

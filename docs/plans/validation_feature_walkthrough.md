# Validation Feature Walkthrough

The [validate](file:///home/alemnalo/csttool/src/csttool/cli/commands/validate.py#100-224) command has been added to `csttool` to robustly compare extracted CST tractograms against reference bundles (e.g., from TractoInferno).

## New Capability: `csttool validate`

This command ensures that your tracking results match the spatial reference grid and computes clinically relevant metrics.

### Key Features
- **Strict Spatial Checks**: Automatically verifies that candidate bundles match the reference anatomy (affine/dimensions). Aborts if they differ > 1mm to prevent silent failures.
- **Robust Metrics**: Handles empty bundles, partial overlaps, and density differences correctly.
- **Visualization**: Generates NIfTI overlays and PNG snapshots (Axial/Coronal/Sagittal) for visual inspection.
- **Safety**: Warns if L/R bundles appear swapped based on hemisphere alignment.

### Usage

```bash
csttool validate \
    --cand-left output/sub-1282/cst_left.trk \
    --cand-right output/sub-1282/cst_right.trk \
    --ref-left  derivatives/sub-1282/PYT_L.trk \
    --ref-right derivatives/sub-1282/PYT_R.trk \
    --ref-space derivatives/sub-1282/dti_FA.nii.gz \
    --output-dir validation_results \
    --visualize
```

### Metrics Explained

| Metric | Description | Target |
|--------|-------------|--------|
| **Dice** | Spatial overlap of binary density masks. | > 0.7 (Ideal) |
| **MDF** | Mean of closest distances (symmetric). | < 3-4 mm |
| **Overreach** | Fraction of candidate outside reference boundaries. | < 0.2 |
| **Coverage** | Fraction of reference volume covered by candidate. | > 0.8 |
| **Count Ratio** | Ratio of streamlines (Candidate / Reference). | ~ 1.0 |

### Output

The command produces:
1. `validation_report.json`: Full metrics and metadata.
2. `visualizations/`: Directory containing:
   - `val_candidate_occ.nii.gz`, `val_reference_occ.nii.gz`, `val_overlap_occ.nii.gz`
   - `val_snapshot_{x,y,z}.png` (Red=Candidate, Blue=Reference)

### Validation Status

- **Unit Tests**: [tests/validation/test_bundle_comparison.py](file:///home/alemnalo/csttool/tests/validation/test_bundle_comparison.py) (Passing)
- **Integration Tests**: [tests/validation/test_cli_validate.py](file:///home/alemnalo/csttool/tests/validation/test_cli_validate.py) (Passing)
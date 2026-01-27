# Extract Module Walkthrough

The `extract` module allows for the isolation of the Corticospinal Tract (CST) from whole-brain tractograms or directly from DWI data.

## Core Capability: `csttool extract`

This command filters streamlines that connect the brainstem and the motor cortex. It uses the Harvard-Oxford atlas, registered to the subject's native space, to define these Regions of Interest (ROIs).

### Basic Usage

```bash
csttool extract \
    --tractogram whole_brain.trk \
    --fa dti_FA.nii.gz \
    --out extraction_results \
    --subject-id sub-001
```

### Advanced Usage

```bash
csttool extract \
    --tractogram whole_brain.trk \
    --fa dti_FA.nii.gz \
    --out extraction_results \
    --subject-id sub-001 \
    --extraction-method passthrough \
    --min-length 30 \
    --max-length 200 \
    --dilate-brainstem 2 \
    --dilate-motor 1 \
    --fast-registration \
    --save-visualizations
```

### Parameters

#### Required Parameters
- `--tractogram PATH`: Path to whole-brain tractogram file (.trk format)
- `--fa PATH`: Path to FA (Fractional Anisotropy) map (.nii or .nii.gz)
- `--out PATH`: Output directory for results

#### Optional Parameters
- `--subject-id STR`: Subject identifier for output naming (default: derived from input files)
- `--extraction-method {passthrough,endpoint}`: Extraction method (default: `passthrough`)
  - Note: `roi-seeded` method requires raw DWI and is only available via `csttool run`
- `--min-length FLOAT`: Minimum streamline length in mm (default: `30.0`)
- `--max-length FLOAT`: Maximum streamline length in mm (default: `200.0`)
- `--dilate-brainstem INT`: Brainstem ROI dilation iterations (default: `2`)
- `--dilate-motor INT`: Motor cortex ROI dilation iterations (default: `1`)
- `--fast-registration`: Use faster registration (less accurate, useful for testing)
- `--save-visualizations`: Generate QC visualizations of extracted tracts
- `--verbose`: Print detailed progress output (default: enabled)

### Extraction Methods

#### 1. Passthrough (Default)
```bash
csttool extract --tractogram whole_brain.trk --fa fa.nii.gz --out results --extraction-method passthrough
```
- **Input**: Whole-brain tractogram (.trk)
- **Logic**: Keeps streamlines that pass through *both* the Brainstem and Primary Motor Cortex ROIs
- **Use Case**: Standard post-hoc filtering; more permissive than endpoint method
- **Typical yield**: 1-2% of whole-brain streamlines

#### 2. Endpoint
```bash
csttool extract --tractogram whole_brain.trk --fa fa.nii.gz --out results --extraction-method endpoint
```
- **Input**: Whole-brain tractogram (.trk)
- **Logic**: Keeps only streamlines where endpoints (first/last points) fall within the ROIs
- **Use Case**: Stricter anatomical constraints; more selective filtering
- **Typical yield**: Lower than passthrough method

#### 3. ROI-Seeded
```bash
csttool run --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --out results --extraction-method roi-seeded
```
- **Input**: Preprocessed DWI + FA map (requires `csttool run`, not `csttool extract`)
- **Logic**: Seeds tractography directly from Motor Cortex ROIs; filters by Brainstem traversal
- **Parameters**:
  - `--seed-fa-threshold`: FA threshold for valid seed points (default: `0.15`)
  - `--seed-density`: Seeds per voxel (default: `2`)
- **Use Case**: Dense CST reconstruction without whole-brain tracking overhead
- **Note**: This method is NOT available in `csttool extract` - use `csttool run` instead

### Algorithm Pipeline

#### Step 1: Registration
Registers the MNI152 template to the subject's FA map using:
- **Affine registration**: Coarse alignment (12 DOF)
- **SyN registration**: Non-linear deformation field
- **Control**: Use `--fast-registration` for reduced iterations (testing only)
  - Standard: `[10000, 1000, 100]` affine, `[10, 10, 5]` SyN
  - Fast: `[1000, 100, 10]` affine, `[5, 5, 3]` SyN

#### Step 2: Atlas Warping
Applies computed transforms to Harvard-Oxford atlases:
- **Cortical Atlas**: 48 cortical regions
- **Subcortical Atlas**: 21 subcortical structures
- Both atlases are warped to match the subject's FA space

#### Step 3: ROI Creation
Extracts and prepares anatomical ROIs:
- **Brainstem**: Extracted from Subcortical atlas
  - Dilated by `--dilate-brainstem` iterations (default: `2`)
- **Motor Cortex (Left & Right)**: Precentral Gyrus from Cortical atlas
  - Dilated by `--dilate-motor` iterations (default: `1`)
- **Purpose**: Dilation ensures ROIs overlap with white matter streamlines

#### Step 4: Streamline Filtering
Applies extraction logic based on selected method:
- **Passthrough**: Streamlines traversing both ROIs
- **Endpoint**: Streamlines with endpoints in ROIs
- Length filtering: `--min-length` to `--max-length` (default: 30-200mm)
- Bilateral separation: Automatically splits into left/right CST

#### Step 5: Output Generation
Saves extracted tractograms and metadata:
- **Tractograms**: `{subject_id}_cst_left.trk`, `{subject_id}_cst_right.trk`, `{subject_id}_cst_combined.trk`
- **Report**: JSON file with extraction statistics
- **Visualizations**: Optional PNG/HTML plots (`--save-visualizations`)
- **ROI Masks**: Saved as NIfTI files for QC

## Output Files

After successful extraction, the output directory contains:

```
extraction_results/
├── {subject_id}_cst_left.trk          # Left hemisphere CST
├── {subject_id}_cst_right.trk         # Right hemisphere CST
├── {subject_id}_cst_combined.trk      # Bilateral CST
├── {subject_id}_extraction_report.json # Statistics and metadata
├── {subject_id}_motor_left_roi.nii.gz  # Left motor cortex mask
├── {subject_id}_motor_right_roi.nii.gz # Right motor cortex mask
├── {subject_id}_brainstem_roi.nii.gz   # Brainstem mask
└── mni_to_subject_warped.nii.gz       # Warped MNI template (QC)
```

If `--save-visualizations` is used, additional files are generated for quality control.

## Example Output

```text
Loading tractogram: whole_brain.trk
  Loaded 452,180 streamlines
Loading FA map: dti_FA.nii.gz

============================================================
Step 1: Registering MNI template to subject space
============================================================
  Starting Affine registration...
  Affine registration converged
  Starting SyN registration...
  SyN registration converged

============================================================
Step 2: Warping Harvard-Oxford atlases to subject space
============================================================
  Warping Cortical Atlas...
  Warping Subcortical Atlas...

============================================================
Step 3: Creating CST ROI masks
============================================================
  Brainstem: 2,840 voxels (after dilation=2)
  Motor Cortex (Left): 1,240 voxels (after dilation=1)
  Motor Cortex (Right): 1,180 voxels (after dilation=1)

============================================================
Step 4: Extracting bilateral CST (method: passthrough)
============================================================
  Filtering streamlines...
  Left CST: 2,450 streamlines
  Right CST: 2,130 streamlines

============================================================
Step 5: Saving extracted tractograms
============================================================
  Saved: sub-001_cst_left.trk
  Saved: sub-001_cst_right.trk
  Saved: sub-001_cst_combined.trk

============================================================
EXTRACTION COMPLETE
============================================================
Subject: sub-001
Left CST:  2,450 streamlines
Right CST: 2,130 streamlines
Total:     4,580 streamlines
Extraction rate: 1.01%
============================================================
```

## Troubleshooting

### "roi-seeded method requires raw DWI data"
The ROI-seeded extraction method is not available in `csttool extract`. Use `csttool run` instead:
```bash
csttool run --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --extraction-method roi-seeded
```

### Low extraction rate (< 0.5%)
- Increase ROI dilation: `--dilate-brainstem 3 --dilate-motor 2`
- Try passthrough method instead of endpoint
- Check that tractogram and FA are in the same space
- Verify whole-brain tractogram quality

### Very high extraction rate (> 5%)
- Decrease ROI dilation: `--dilate-brainstem 1 --dilate-motor 0`
- Try endpoint method for stricter filtering
- Adjust length constraints: `--min-length 40 --max-length 180`

### Registration failures
- Ensure FA map has good contrast and quality
- Try `--fast-registration` first to verify pipeline, then run full registration
- Check that FA values are in expected range (0-1)

## Tips and Best Practices

1. **Subject ID**: Always specify `--subject-id` for consistent file naming across subjects
2. **Method Selection**: Start with `passthrough` (default), then try `endpoint` if too many false positives
3. **ROI Dilation**: Default values (brainstem=2, motor=1) work well for most cases
4. **Length Filtering**: Adjust based on your data - smaller voxels may need lower thresholds
5. **QC**: Use `--save-visualizations` to verify ROI placement and extracted streamlines
6. **Registration Speed**: Use `--fast-registration` for initial testing, then full registration for final results

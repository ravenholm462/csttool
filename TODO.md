# TODO

## General

- [ ] Write project documentation.
  - For troubleshooting documentation: patch2self denoising may produce streamlines that are too short -> extraction fails. Switching to NLMeans may help.
- [ ] Review and update documentation after module integration

## CLI

- [x] Implement a `run` command that will run all the steps in the following order: `check`, `import`, `preprocess`, `track`, `extract`, `metrics`. 
- [x] Add "--visualizations" flag to save the figs.

## Import

- [x] Input DICOMS of one subject are represented as multiple series. The tool does not account for this currently. Resolve proper data import.

## Preprocessing

### Architecture & Integration
- [X] **Replace `funcs.py` with modular architecture** - Integrate all new modules in `preprocess/modules/`:
  - `load_dataset.py` - Load NIfTI/DICOM and build gradient table
  - `denoise.py` - NLMeans and Patch2Self denoising
  - `gibbs_unringing.py` - Gibbs oscillation correction
  - `background_segmentation.py` - Brain mask computation
  - `perform_motion_correction.py` - Between-volume motion correction
  - `save_preprocessed.py` - Save outputs with organized structure
  - `visualizations.py` - QC visualizations (already implemented)
- [X] Update `preprocess/__init__.py` to export all new module functions
- [X] Update `cli.py` to use the new modular pipeline
- [x] Deprecate/remove old functions in `funcs.py` after migration

### Validation & Research
- [x] Correction for Gibbs' oscillations missing. Implement.
- [ ] Validate the Gibbs' correction. 
- [X] Is NLMEANS denoising good enough? Resolved: Patch2Self is now default.

### Testing
- [x] Add unit tests for all new preprocessing modules:
  - **`denoise.py`**:
    - [x] Test NLMeans denoising
    - [x] Test Patch2Self denoising
    - [x] Test brain mask handling (None vs provided)
    - [x] Test invalid method validation
  - **`load_dataset.py`**:
    - [x] Test NIfTI loading
    - [x] Test DICOM conversion
    - [x] Test gradient file detection (.bval/.bvec vs .bvals/.bvecs)
  - **`background_segmentation.py`**:
    - [x] Test brain mask generation
  - **`gibbs_unringing.py`**:
    - [x] Test Gibbs correction
  - **`perform_motion_correction.py`**:
    - [x] Test motion correction
  - **`save_preprocessed.py`**:
    - [x] Test output structure and file naming

### Future Enhancements
- [ ] Add timing information to processing functions (optional)
- [ ] Ensure visualization.py handles all QC visualizations

## Tracking

- [x] Visualization of full brain tractogram is too messy. Fix.
- [ ] Add heatmap legends for MD, directions etc.

## Extraction

- [x] Output architecture too messy. Fix.
- [x] Visualizations of LFS/RHS CST too messy, same as for full brain tractogram. Fix.
- [x] Extraction fails for alternate datasets (e.g. OpenNeuro, Stanford-HARDI). Full-brain tractogram generated, but all CST streamlines are filtered out.

## Metrics

- [x] Revise PDF report (single-page layout).
  - [x] **Header Block** (2-3 lines):
    - [x] Subject/Session ID, Date, csttool version
    - [x] Bold line: "Metrics Extracted In: Native Space" (explicit space declaration)
  - [x] **Metrics Table** (compact):
    - [x] Fix MD superscript formatting (×10⁻³)
    - [x] Add radial diffusivity (RD) and axial diffusivity (AD) columns
    - [x] Color-code Laterality Index values
  - [x] **Visualization Row** (side-by-side):
    - [x] Left (60% width): Stacked FA/MD profile plots
      - X-axis: "Pontine Level (0%)" → "PLIC (50%)" → "Precentral Gyrus (100%)"
      - FA Y-axis: 0 to ~0.6
      - MD Y-axis: ×10⁻³ mm²/s (e.g., 0.7 to 1.1)
    - [x] Right (40% width): 3D tractogram QC preview
      - Left/Right CST in different colors
      - Overlay on mid-sagittal or axial T1 slice at internal capsule level
  - [x] **Footer Note** (tiny font, optional):
    - [x] Brief method note (e.g., "Probabilistic tractography, DTI model")
  - [x] Minimize margins for single-page fit

- [x] Implement RD/AD pipeline integration:
  - [x] Compute RD and AD maps in tracking module (AD=λ₁, RD=(λ₂+λ₃)/2)
  - [x] Add --rd and --ad CLI arguments to `csttool metrics`
  - [x] Pass rd_map and ad_map to `analyze_cst_hemisphere()` in `cmd_metrics()`

- [ ] Implement metrics space conversion (Native ↔ MNI152 Template Space)
- [ ] Update the .csv and .json report files to include the new metrics.

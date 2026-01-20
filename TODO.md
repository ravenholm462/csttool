# TODO

## General

- [ ] Write project documentation.
  - See `docs/DOCUMENTATION_CHECKLIST.md` for detailed status.
- [/] Trim down the codebase.
  - [x] Remove unused code
  - [x] Remove unused dependencies
  - [x] Remove unused files
  - [ ] Remove unused directories
  - [x] Fix warning print statements
  - [/] Format all print statements throughout the codebase so they are unified and minimal.

## CLI
- [x] `cli.py` is an absolute monster of a script. Needs refactoring.
  - [x] Break down `cli.py` into submodules (e.g., `cli/commands/*.py`).

## Batch processing
- [ ] **See:** `src/csttool/batch/IMPLEMENTATION_PLAN.md` for detailed tracking.
- [ ] Implement batch processing workflow.
  - [ ] Create unified reports of processed batch with statistics (Batch Report).

## Preprocessing
- [x] Pipeline should skip preprocessing by default, as it is inefficient, slow, and the available datasets are often preprocessed using superior software like FSL. If not skipped, the default pipeline should include skip Gibbs' correction and motion correctio, and keep everything else.
- [x] Update reports to reflect preprocessing status (Skipped vs Executed).

## Validation & Research
- [ ] Find standard values for healthy controls, add to metrics.
- [ ] Use batch processing to process entire TractoInferno training set.

### Testing
- [ ] Update all unit tests if necessary.
- [ ] Ensure edge case coverage.
- [ ] Add unit tests for metrics consistency.

## Tracking
- [ ] Make angle threshold adjustible, keep 45 degrees default.
  - [ ] Add CLI option.

## Extraction


## Metrics

- [x] Figure out if metrics calculated in native or MNI space first. (Answer: Native)
- [ ] Implement metrics space conversion (Native ↔ MNI152 Template Space)
- [ ] Acquisiton and processing parameters are hardcoded, fix.
- [ ] Decide which acquistion and processing parameters to report.
  - [ ] Update the pipeline to report the selected parameters:
    - [ ] Acquisition parameters (autopopulate during `import` from JSON sidecar or derive)
      - [ ] Field strength
      - [ ] Echo time (TE) / Repetition time (TR)
      - [ ] b-value in s/mm^2
      - [ ] Voxel resolution (e.g. 2.0 x 2.0 x 2.0 mm^3)
      - [ ] Number of directions.
      - [ ] Number of volumes.
    - [ ] Processing parameters (get from csttool)
      - [ ] Denoising method (either find in JSON sidecar, write "External" if not found or add applied csttool method if `--preprocess` used)
      - [ ] Motion / Eddy current correction (+ method), if applied (either find in JSON sidecar, write "External" if not found or add applied csttool method if `--preprocess` used)
      - [ ] Tracking thresholds (FA threshold, angle threshold, step size)
      - [ ] ROI atlas & registration method
      - [ ] Tractography algorithm
      - [ ] Seeding (density, region)
    
 


# TODO

## General

- [/] Write project documentation.
  - See `docs/DOCUMENTATION_CHECKLIST.md` for detailed status.
- [x] Trim down the codebase.
  - [x] Remove unused code
  - [x] Remove unused dependencies
  - [x] Remove unused files
  - [x] Remove unused directories
  - [x] Fix warning print statements
  - [x] Format all print statements throughout the codebase so they are unified and minimal.

- [x] Make all atlases csttool is using available in the repo directly, following the example of FMRIB58_FA template

- [ ] Align the thesis narrative found in docs/seminar/scientific-narrative.md with the actual implementation.

  ## 1. Determinism & Reproducibility

  - [ ] Make fixed random seed the default behavior
  - [ ] Allow optional random seeding via explicit CLI flag
  - [ ] Verify identical `.trk` files across repeated runs with fixed seed
  - [ ] Compare derived metrics (FA, MD, RD, AD, LI) across repeated runs
  - [ ] Quantify numeric deviation across runs
  - [ ] Define acceptable tolerance thresholds for metric stability
  - [ ] Document reproducibility guarantees clearly in thesis

  ## 2. Parameter Transparency

  - [ ] Centralize all tractography parameters in one configuration module
  - [ ] Audit for undocumented DIPY defaults
  - [ ] Log all thresholds and tracking parameters in CLI output
  - [ ] Ensure full parameter set is written to a run log file
  - [ ] Include seed value in run output
  - [ ] Include ROI definitions and affine information in logs

  ## 3. Version Locking & Environment Control

  - [ ] Pin exact library versions in requirements.txt / pyproject.toml
  - [ ] Log library versions at runtime
  - [ ] Log Python version at runtime
  - [ ] Document atlas version and template source explicitly
  - [ ] Ensure nilearn fetch behavior is stable and reproducible
  - [ ] Consider storing atlas checksum or local copy

  ## 4. Input Assumptions & Validation

  - [ ] Explicitly document required preprocessing assumptions
    - [ ] Eddy correction
    - [ ] Motion correction
    - [ ] Tensor fitting
    - [ ] Skull stripping
  - [ ] Add input validation checks where feasible
  - [ ] Fail early if critical assumptions are violated
  - [ ] Clearly define supported input formats (DICOM vs NIfTI)

  ## 5. Validation Framework Rigor

  - [ ] Define precisely how Dice score is computed
    - [ ] Voxel mask based?
    - [ ] Density map based?
    - [ ] Thresholding applied?
  - [ ] Ensure bundles are in identical coordinate space before comparison
  - [ ] Validate affine alignment prior to overlap computation
  - [ ] Document reference pipeline parameters (e.g. FSL)
  - [ ] Explicitly state validation as comparative, not anatomical ground truth

  ## 6. Metric Stability & Sensitivity

  - [ ] Test sensitivity of FA/MD/RD/AD to streamline count variation
  - [ ] Test LI stability under small perturbations
  - [ ] Remove small percentage of streamlines and observe metric change
  - [ ] Evaluate metric stability across repeated deterministic runs
  - [ ] Document findings quantitatively

  ## 7. Runtime & Performance

  - [ ] Measure runtime on standard dataset
  - [ ] Record memory usage during extraction
  - [ ] Confirm lightweight execution claim
  - [ ] Remove unnecessary intermediate files
  - [ ] Ensure pipeline runs end-to-end without manual intervention

  ## 8. Edge Case Handling

  - [ ] Define behavior if no CST found in one hemisphere
  - [ ] Define behavior if FA map missing
  - [ ] Define behavior if gradients malformed
  - [ ] Ensure clear, explicit error messages
  - [ ] Avoid silent failures

  ## 9. Output Completeness & Run Provenance

  - [ ] Include run ID in output
  - [ ] Include timestamp
  - [ ] Include Git commit hash
  - [ ] Include full parameter configuration
  - [ ] Include seed value used
  - [ ] Include library versions
  - [ ] Store structured run metadata file (JSON/YAML)

  ## 10. Narrative Alignment Check

  - [ ] Verify every scientific claim in thesis maps to measurable behavior
  - [ ] Remove claims not backed by implementation
  - [ ] Revise implementation if it contradicts thesis philosophy
  - [ ] Confirm that reproducibility claims are demonstrably true
  - [ ] Ensure validation criteria are explicitly defined and testable

## CLI

- [x] `cli.py` is an absolute monster of a script. Needs refactoring.
  - [x] Break down `cli.py` into submodules (e.g., `cli/commands/*.py`).

## Batch processing

- [x] **See:** `src/csttool/batch/IMPLEMENTATION_PLAN.md` for detailed tracking.
- [x] Implement batch processing workflow.
  - [x] Create unified reports of processed batch with statistics (Batch Report).

## Preprocessing

- [x] Pipeline should skip preprocessing by default, as it is inefficient, slow, and the available datasets are often preprocessed using superior software like FSL. If not skipped, the default pipeline should include skip Gibbs' correction and motion correctio, and keep everything else.
- [x] Update reports to reflect preprocessing status (Skipped vs Executed).

## Validation & Research

- [x] Add CST validation against reference bundles (TractoInferno PYT)
  - [x] `csttool validate` command with Dice, overreach, MDF metrics
  - [x] Unit tests for validation module
- [x] Use batch processing to process entire TractoInferno training set.

### Testing

- [x] Update all unit tests if necessary.
- [x] Ensure edge case coverage.
- [x] Add unit tests for metrics consistency.

## Tracking

- [ ] If ROI seeded extraction chosen, whole brain tractography is a waste of time. Refactor tracking and extraction to account for this.

## Extraction

- [ ] See above regarding ROI seeded extraction of CST.

## Metrics

- [x] Figure out if metrics calculated in native or MNI space first. (Answer: Native)
- [ ] Implement metrics space conversion (Native ↔ MNI152 Template Space)
- [x] Acquisition and processing parameters are hardcoded, fix.
- [x] Decide which acquisition and processing parameters to report.
  - [x] Update the pipeline to report the selected parameters:
    - [x] Acquisition parameters (autopopulate during `import` from JSON sidecar or derive)
      - [x] Field strength (with `--field-strength` CLI override)
      - [x] Echo time (TE) (with `--echo-time` CLI override)
      - [x] b-value in s/mm^2 (auto-derived)
      - [x] Voxel resolution (auto-derived)
      - [x] Number of directions (auto-derived)
      - [x] Number of volumes (auto-derived)
    - [x] Processing parameters (get from csttool)
      - [x] Denoising method
      - [x] Motion / Eddy current correction
      - [x] Tracking thresholds (FA threshold, angle threshold, step size)
      - [x] ROI atlas & registration method
      - [x] Tractography algorithm
      - [x] Seeding (density)

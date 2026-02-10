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
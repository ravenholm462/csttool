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
- [ ] Pipeline should skip preprocessing by default, as it is inefficient, slow, and the available datasets are often preprocessed using superior software like FSL. If not skipped, the default pipeline should include skip Gibbs' correction and motion correctio, and keep everything else.

## Validation & Research
- [ ] Find standard values for healthy controls, add to metrics.
- [ ] Use batch processing to process entire TractoInferno training set.

### Testing
- [ ] Update all unit tests if necessary.
- [ ] Ensure edge case coverage.
- [ ] Add unit tests for metrics consistency.

## Tracking


## Extraction


## Metrics

- [x] Figure out if metrics calculated in native or MNI space first. (Answer: Native)
- [ ] Implement metrics space conversion (Native ↔ MNI152 Template Space)


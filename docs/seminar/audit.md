# Narrative–Implementation Audit

Comparison of `docs/seminar/scientific-narrative.md` claims against the unticked checkboxes in `TODO.md` (General, sections 1–10). Each item is marked:

- **YES** — already addressed in the codebase
- **PARTIAL** — framework exists but incomplete
- **NO** — not yet implemented

---

## 1. Determinism & Reproducibility

The narrative claims "reproducibility" and "deterministic tractography." The codebase supports optional seeding but does not enforce it by default.

| Checkbox | Status | Evidence |
|----------|--------|----------|
| Make fixed random seed the default behavior | **NO** | `--rng-seed` defaults to `None` (`cli/__init__.py:247`) |
| Allow optional random seeding via explicit CLI flag | **YES** | `--rng-seed` argument implemented (`cli/__init__.py:246-251`); passed to DIPY's `LocalTracking` (`tracking/modules/run_tractography.py:1-38`) |
| Verify identical `.trk` files across repeated runs with fixed seed | **NO** | No automated test exists |
| Compare derived metrics across repeated runs | **NO** | No stability test suite |
| Quantify numeric deviation across runs | **NO** | Not implemented |
| Define acceptable tolerance thresholds for metric stability | **NO** | Not defined |
| Document reproducibility guarantees clearly in thesis | **NO** | `scientific-narrative.md` makes the claim but no quantitative evidence backs it yet |

**Note:** Visualization code uses unseeded `np.random.choice()` in multiple modules (`extract/modules/visualizations.py`, `preprocess/modules/visualizations.py`). This does not affect analysis outputs but prevents fully deterministic report generation.

---

## 2. Parameter Transparency

The narrative emphasizes "direct control over seeding, stopping criteria, and ROI-based filtering" and "transparency of the extraction logic."

| Checkbox | Status | Evidence |
|----------|--------|----------|
| Centralize all tractography parameters in one configuration module | **PARTIAL** | Parameters collected in `tracking_params` dict (`cli/commands/track.py:135-146`) but no dedicated config module exists |
| Audit for undocumented DIPY defaults | **NO** | `relative_peak_threshold=0.8` and `min_separation_angle=45` are hardcoded in the dict but not surfaced to the user or documented as DIPY defaults |
| Log all thresholds and tracking parameters in CLI output | **PARTIAL** | Verbose mode prints FA threshold and seed count (`tracking/modules/seed_and_stop.py:36-68`); not all parameters printed unless verbose |
| Ensure full parameter set is written to a run log file | **YES** | JSON report includes step_size, fa_threshold, seed_density, sh_order, stopping_criterion, relative_peak_threshold, min_separation_angle, random_seed (`tracking/modules/save_tracking_outputs.py:134-165`) |
| Include seed value in run output | **YES** | `tracking_params['random_seed']` included in JSON report (`tracking/modules/save_tracking_outputs.py:144`) |
| Include ROI definitions and affine information in logs | **YES** | ROI labels and warp quality logged during extraction (`extract/modules/warp_atlas_to_subject.py`); coordinate validation reports affine info (`extract/modules/coordinate_validation.py`) |

---

## 3. Version Locking & Environment Control

The narrative's reproducibility claim requires stable, pinned environments.

| Checkbox | Status | Evidence |
|----------|--------|----------|
| Pin exact library versions in pyproject.toml | **NO** | All dependencies unpinned — `"numpy"`, `"dipy"`, etc. with no version constraints (`pyproject.toml:13-25`) |
| Log library versions at runtime | **NO** | Only `csttool` version logged (`cli/utils.py:260`, `metrics/modules/reports.py:92`); numpy/scipy/dipy versions not captured |
| Log Python version at runtime | **NO** | `requires-python = ">=3.10"` in pyproject.toml but not logged at runtime |
| Document atlas version and template source explicitly | **YES** | `data/manifest.py:10-75` — complete manifest with FSL tags, source URLs, and versions |
| Ensure nilearn fetch behavior is stable and reproducible | **PARTIAL** | csttool bundles its own atlases instead of relying on nilearn fetch; `data/loader.py` loads from local data directory with checksum verification |
| Consider storing atlas checksum or local copy | **YES** | SHA256 checksums in `data/manifest.py`; local copies bundled; checksums verified on load (`data/loader.py:64-117`) |

---

## 4. Input Assumptions & Validation

The narrative states csttool "accepts diffusion-weighted MRI data" and runs with "minimal user interaction."

| Checkbox | Status | Evidence |
|----------|--------|----------|
| Document required preprocessing assumptions | **PARTIAL** | Documented in code docstrings (`preprocess/preprocess.py:37-81`) but no standalone specification. Eddy/motion correction handled optionally; skull stripping via Otsu thresholding implicit in brain masking |
| Add input validation checks where feasible | **YES** | Coordinate space validation (`extract/modules/coordinate_validation.py:16-210`), directory/file existence checks, bval/bvec format validation (`preprocess/modules/load_dataset.py`) |
| Fail early if critical assumptions are violated | **YES** | `ValueError` in strict coordinate validation mode; `FileNotFoundError` with helpful messages; broken symlink detection (`cli/commands/run.py:108-118`) |
| Clearly define supported input formats (DICOM vs NIfTI) | **YES** | Automatic DICOM detection and conversion (`preprocess/modules/load_dataset.py:44-75`); supports `.bval`/`.bvals` and `.bvec`/`.bvecs` variants |

---

## 5. Validation Framework Rigor

The narrative states validation evaluates "technical correctness and methodological equivalence relative to established pipelines."

| Checkbox | Status | Evidence |
|----------|--------|----------|
| Define precisely how Dice score is computed | **YES** | Binary density maps from streamlines, threshold > 0, Dice = 2*intersection / (cand_vol + ref_vol) (`validation/bundle_comparison.py:141-186`) |
| Ensure bundles are in identical coordinate space before comparison | **YES** | `check_spatial_compatibility()` validates affines with translation (1mm) and rotation/scale (1e-3) tolerances; raises `SpatialMismatchError` on mismatch (`validation/bundle_comparison.py:51-94`) |
| Validate affine alignment prior to overlap computation | **YES** | Affine validation in coordinate_validation.py; orientation code checks; bounds verification against reference volume |
| Document reference pipeline parameters (e.g. FSL) | **PARTIAL** | Uses bundled MNI152 and FMRIB58_FA templates (`extract/modules/registration.py:39-100`); FSL atlas tags in manifest; but no formal comparison table documenting FSL pipeline parameters side-by-side |
| Explicitly state validation as comparative, not anatomical ground truth | **YES** | `scientific-narrative.md:33-47` — "evaluates technical correctness and methodological equivalence relative to established pipelines" |

---

## 6. Metric Stability & Sensitivity

The narrative claims "stable and interpretable tract-level diffusion metrics." This section has the largest gap.

| Checkbox | Status | Evidence |
|----------|--------|----------|
| Test sensitivity of FA/MD/RD/AD to streamline count variation | **NO** | No perturbation tests exist. FA/MD/RD/AD computation is implemented (`tracking/modules/fit_tensors.py:20-40`) but never stress-tested |
| Test LI stability under small perturbations | **NO** | LI computation works (`metrics/modules/bilateral_analysis.py:48-140`) but no perturbation or stability tests |
| Remove small percentage of streamlines and observe metric change | **NO** | Not implemented |
| Evaluate metric stability across repeated deterministic runs | **NO** | No automated test exists |
| Document findings quantitatively | **PARTIAL** | LI interpretation thresholds defined (`bilateral_analysis.py:180-192`); `streamline_count_ratio()` exists (`validation/bundle_comparison.py:257-279`); but no quantitative sensitivity report |

---

## 7. Runtime & Performance

The narrative claims csttool is "lightweight" and "computationally efficient."

| Checkbox | Status | Evidence |
|----------|--------|----------|
| Measure runtime on standard dataset | **PARTIAL** | Step-level timing tracked in `run.py` (`step_times` dict, line 54); batch processing records `duration_seconds` per subject (`batch/batch.py:67`); but no benchmark results documented |
| Record memory usage during extraction | **NO** | Not implemented |
| Confirm lightweight execution claim | **PARTIAL** | No intermediate files stored in output; optional visualization saving; but no formal benchmarks |
| Remove unnecessary intermediate files | **YES** | Batch workflow has `keep_work` flag (`batch/batch.py:53`); `promote_outputs()` manages cleanup |
| Ensure pipeline runs end-to-end without manual intervention | **YES** | `run.py` orchestrates all 6 steps (check → import → preprocess → track → extract → metrics) fully automated; `continue_on_error` flag for graceful degradation |

---

## 8. Edge Case Handling

The narrative claims "minimal user interaction" and automation. Edge case handling is well covered.

| Checkbox | Status | Evidence |
|----------|--------|----------|
| Define behavior if no CST found in one hemisphere | **YES** | Empty mask returns `None` (`extract/modules/passthrough_filtering.py:57-59`); LI handles zero denominators (`metrics/modules/bilateral_analysis.py:166-172`); asymmetry warnings printed |
| Define behavior if FA map missing | **YES** | Falls back to MNI T1 template with warning (`extract/modules/registration.py:68-73`); explicit file check with early return in CLI (`cli/commands/extract.py:40-44`) |
| Define behavior if gradients malformed | **YES** | Shape correction, zero-direction handling, graceful BIDS JSON fallback (`ingest/modules/assess_quality.py:6-75`) |
| Ensure clear, explicit error messages | **YES** | Multi-line diagnostic errors with root cause and suggestions (`extract/modules/coordinate_validation.py:203-208`); `SpatialMismatchError` with expected vs actual values; `FileNotFoundError` with paths |
| Avoid silent failures | **YES** | All errors printed and tracked; `failed_steps` list in pipeline; batch error categories (INPUT_MISSING, VALIDATION, PIPELINE_FAILED, TIMEOUT, SYSTEM) |

---

## 9. Output Completeness & Run Provenance

| Checkbox | Status | Evidence |
|----------|--------|----------|
| Include run ID in output | **NO** | No UUID or unique run identifier generated |
| Include timestamp | **YES** | Console output (`cli/commands/run.py:65`), pipeline report JSON (`cli/utils.py:262-263`), metrics report (`metrics/modules/reports.py:91`) |
| Include Git commit hash | **NO** | Not captured anywhere |
| Include full parameter configuration | **YES** | Comprehensive JSON with all tracking parameters (`tracking/modules/save_tracking_outputs.py:134-165`) |
| Include seed value used | **YES** | In JSON report (`tracking/modules/save_tracking_outputs.py:144`) |
| Include library versions | **PARTIAL** | csttool version logged (`cli/utils.py:260`, `__init__.py:2`); dependency versions (numpy, dipy, etc.) not captured |
| Store structured run metadata file (JSON/YAML) | **YES** | `{stem}_tracking_report.json`, `{subject_id}_pipeline_report.json`, `{subject_id}_bilateral_metrics.json`, JSON Lines batch logs |

---

## 10. Narrative Alignment Check

These are meta-tasks that require manual review once the above items are resolved.

| Checkbox | Status | Evidence |
|----------|--------|----------|
| Verify every scientific claim maps to measurable behavior | **NO** | Pending — this audit is a step toward that |
| Remove claims not backed by implementation | **NO** | Pending |
| Revise implementation if it contradicts thesis philosophy | **NO** | Pending |
| Confirm reproducibility claims are demonstrably true | **NO** | Requires determinism (section 1) and metric stability (section 6) to be addressed first |
| Ensure validation criteria are explicitly defined and testable | **NO** | Dice computation is implemented, but tolerance thresholds and pass/fail criteria not formally defined |

---

## Summary

| Section | Done | Partial | Missing | Coverage |
|---------|:----:|:-------:|:-------:|:--------:|
| 1. Determinism & Reproducibility | 1 | 0 | 6 | Low |
| 2. Parameter Transparency | 3 | 2 | 1 | Medium |
| 3. Version Locking & Environment Control | 2 | 1 | 3 | Medium |
| 4. Input Assumptions & Validation | 3 | 1 | 0 | High |
| 5. Validation Framework Rigor | 3 | 1 | 1 | High |
| 6. Metric Stability & Sensitivity | 0 | 1 | 4 | Low |
| 7. Runtime & Performance | 2 | 2 | 1 | Medium |
| 8. Edge Case Handling | 5 | 0 | 0 | **Complete** |
| 9. Output Completeness & Provenance | 4 | 1 | 2 | High |
| 10. Narrative Alignment | 0 | 0 | 5 | None |

**Strongest areas:** Edge case handling (8), input validation (4), validation framework (5).  
**Largest gaps:** Determinism enforcement (1), metric stability testing (6), narrative alignment (10).

The narrative's central claim — reproducibility — depends on resolving sections 1, 3, and 6 first. Section 10 is a meta-review that can only be completed after the others are addressed.
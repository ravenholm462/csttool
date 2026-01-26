# csttool Documentation Checklist

**Last Updated**: 2026-01-26

This checklist tracks documentation progress for all pages defined in `mkdocs.yml`.

---

## 📊 Progress Summary

| Section | Complete | Total | Status |
|---------|----------|-------|--------|
| Getting Started | 3 | 3 | � Completed |
| Tutorials | 0 | 2 | ⬜ Not Started |
| How-To Guides | 0 | 3 | ⬜ Not Started |
| CLI Reference | 0 | 11 | ⬜ Not Started |
| API Reference | 0 | 7 | ⬜ Not Started |
| Explanation | 0 | 5 | ⬜ Not Started |
| Contributing | 0 | 3 | ⬜ Not Started |

---

## 🚀 Getting Started (Priority: HIGH)

These are the first pages new users encounter.

- [x] **installation.md** — Dependencies, pip install, system deps, development setup
- [x] **quickstart.md** — End-to-end example from data to PDF report
  - [x] Show `csttool run` command with all flags
  - [x] Explain expected outputs (tractograms, reports)
  - [x] Link to sample data (HCP or Brainlife) - This is done in recommended_datasets.md
- [x] **data-requirements.md** — Input format specifications
  - [x] NIfTI + bval/bvec requirements
  - [x] DICOM directory structure
  - [x] FOV and resolution requirements
  - [x] Gradient file naming conventions (.bval/.bvec vs .bvals/.bvecs)

---

## 📚 Tutorials (Priority: MEDIUM)

Step-by-step learning-oriented guides.

- [ ] **first-analysis.md** — Complete walkthrough with HCP subject
  - [ ] Download sample data
  - [ ] Run each pipeline step
  - [ ] Interpret the PDF report
  - [ ] Understanding the metrics
- [ ] **batch-processing.md** — Processing multiple subjects
  - [ ] Shell scripting approach
  - [ ] Parallel processing tips
  - [ ] Output organization

---

## 🔧 How-To Guides (Priority: MEDIUM)

Task-oriented recipes for specific goals.

- [ ] **multiple-subjects.md** — Batch processing setup
- [ ] **data-formats.md** — Converting between formats
  - [ ] DICOM to NIfTI
  - [ ] Handling different gradient file conventions
- [ ] **troubleshooting.md** — Common issues and fixes
  - [ ] Patch2Self produces short streamlines → use NLMeans
  - [ ] CST extraction fails → check FOV coverage
  - [ ] WeasyPrint installation errors
  - [ ] Missing system dependencies

---

## 📖 CLI Reference (Priority: HIGH)

Complete command documentation. Can be auto-generated from `--help` + examples.

- [ ] **overview.md** — CLI overview page
  - [ ] List all commands with brief descriptions
  - [ ] Common patterns and examples
- [ ] **check.md** — `csttool check`
  - [ ] Purpose and usage
  - [ ] Expected output
- [ ] **check_dataset.md** — `csttool check-dataset`
  - [ ] Acquisition quality assessment
  - [ ] Input options (--dwi, --bval, --bvec, --json)
- [ ] **import.md** — `csttool import`
  - [ ] All flags (--dicom, --nifti, --out, etc.)
  - [ ] Examples for DICOM and NIfTI
- [ ] **preprocess.md** — `csttool preprocess`
  - [ ] --denoise-method flag (nlmeans, patch2self)
  - [ ] --perform-motion-correction flag
  - [ ] --save-visualizations flag
- [ ] **track.md** — `csttool track`
  - [ ] Tracking parameters
  - [ ] Output files
- [ ] **extract.md** — `csttool extract`
  - [ ] Extraction methods (endpoint, passthrough, roi-seeded)
  - [ ] ROI dilation parameters
  - [ ] Atlas registration
- [ ] **metrics.md** — `csttool metrics`
  - [ ] Report formats (CSV, JSON, HTML, PDF)
  - [ ] Metric definitions
- [ ] **validate.md** — `csttool validate`
  - [ ] Bundle comparison against reference tractograms
  - [ ] Metrics: overlap, coverage, distance
- [ ] **run.md** — `csttool run`
  - [ ] Full pipeline execution
  - [ ] All combined flags
- [ ] **batch.md** — `csttool batch`
  - [ ] Manifest JSON schema
  - [ ] BIDS auto-discovery
  - [ ] Parallel processing options

---

## 🔌 API Reference (Priority: LOW)

For developers using csttool as a library. Can be auto-generated with mkdocstrings.

- [ ] **preprocess.md** — `csttool.preprocess` module
- [ ] **tracking.md** — `csttool.tracking` module
- [ ] **extract.md** — `csttool.extract` module
- [ ] **metrics.md** — `csttool.metrics` module
- [ ] **validation.md** — `csttool.validation` module (not in mkdocs.yml yet)
- [ ] **batch.md** — `csttool.batch` module (not in mkdocs.yml yet)
- [ ] **ingest.md** — `csttool.ingest` module (not in mkdocs.yml yet)

---

## 💡 Explanation (Priority: LOW)

Background knowledge and design rationale.

- [ ] **diffusion-mri.md** — dMRI basics for non-experts
- [ ] **tractography.md** — How tractography works
- [ ] **cst-anatomy.md** — CST anatomical landmarks
- [ ] **design-decisions.md** — Why csttool is built this way
- [ ] **limitations.md** — Known limitations and caveats

---

## 👥 Contributing (Priority: LOW)

For potential contributors.

- [ ] **development-setup.md** — Clone, venv, testing
- [ ] **code-style.md** — Formatting, linting, conventions
- [ ] **architecture.md** — Codebase structure and design

---

## 📝 Writing Guidelines

When writing documentation:

1. **Keep it concise** — Users want answers, not essays
2. **Show, don't tell** — Use code examples liberally
3. **Test your examples** — Every command should work
4. **Link related pages** — Help users navigate
5. **Use admonitions** — `!!! note`, `!!! warning`, `!!! tip`

---

## 🎯 Recommended Order

1. ✅ `installation.md` (DONE)
2. ✅ `quickstart.md` (DONE)
3. ✅ `data-requirements.md` (DONE)
4. CLI Reference (all 11 pages)
5. `troubleshooting.md`
6. `first-analysis.md`
7. Everything else

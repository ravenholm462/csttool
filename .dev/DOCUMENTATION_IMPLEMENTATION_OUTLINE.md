# csttool Documentation Implementation Outline

**Purpose:** Comprehensive guide for documenting csttool - a Python-based neuroimaging analysis tool for automated CST assessment using diffusion MRI.

**Documentation Framework:** [Diátaxis](https://diataxis.fr/) (Tutorial → How-To → Reference → Explanation)

**Timeline:** 8-9 weeks for full documentation | 3-4 weeks for thesis-critical subset

---

## Phase 1: Foundation (Week 1-2)
*Goal: Establish documentation infrastructure and core user-facing content*

### Milestone 1.1: Documentation Infrastructure ⭐
**Priority: CRITICAL**

- [X] Set up MkDocs Material (see setup guide below)
- [X] Configure `mkdocs.yml` with proper navigation structure
- [ ] Set up automatic API documentation generation
- [X] Configure Read the Docs (or GitHub Pages) for hosting
- [ ] Create documentation contribution guidelines

**Directory Structure:**
```
docs/
├── index.md                    # Landing page
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── data-requirements.md
├── tutorials/
│   ├── first-analysis.md
│   ├── batch-processing.md
│   └── custom-parameters.md
├── how-to/
│   ├── multiple-subjects.md
│   ├── data-formats.md
│   ├── customize-extraction.md
│   └── troubleshooting.md
├── reference/
│   ├── cli/
│   │   ├── check.md
│   │   ├── import.md
│   │   ├── preprocess.md
│   │   ├── track.md
│   │   ├── extract.md
│   │   ├── metrics.md
│   │   └── run.md
│   ├── api/
│   │   ├── preprocess.md
│   │   ├── tracking.md
│   │   ├── extract.md
│   │   └── metrics.md
│   ├── output-formats.md
│   └── parameters.md
├── explanation/
│   ├── diffusion-mri.md
│   ├── tractography.md
│   ├── cst-anatomy.md
│   ├── design-decisions.md
│   └── limitations.md
└── contributing/
    ├── development-setup.md
    ├── code-style.md
    ├── testing.md
    └── architecture.md
```

**Best Practice Resources:**
- [MkDocs Material Documentation](https://squidfunk.github.io/mkdocs-material/)
- [Write the Docs - Documentation Guide](https://www.writethedocs.org/guide/)
- [Scientific Python Development Guide](https://learn.scientific-python.org/development/)
- [DIPY Documentation Structure](https://docs.dipy.org/) (neuroimaging example)

---

### Milestone 1.2: README Enhancement ⭐
**Priority: CRITICAL**

Transform the current minimal README into a compelling entry point.

**Components to Add:**

**Header with Badges:**
```markdown
# csttool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/csttool/badge/?version=latest)](https://csttool.readthedocs.io/)

*Simple, modular CST assessment tool for diffusion MRI*
```

**Quick Installation:**
```markdown
## Installation

```bash
pip install git+https://github.com/ravenholm462/csttool.git
```

Or for development:
```bash
git clone https://github.com/ravenholm462/csttool.git
cd csttool
pip install -e .
```
```

**30-Second Quickstart:**
```markdown
## Quickstart

```bash
# Run complete pipeline
csttool run --dicom /path/to/dicom --out results --subject-id sub-01

# Or step by step
csttool check
csttool import --dicom /path/to/dicom --out data
csttool preprocess --nifti data/nifti/sub-01.nii.gz --out results
csttool track --nifti results/preprocessed/sub-01_preproc.nii.gz --out results
csttool extract --tractogram results/tracking/tractograms/sub-01.trk --fa results/tracking/scalar_maps/sub-01_fa.nii.gz --out results
csttool metrics --cst-left results/extraction/trk/sub-01_cst_left.trk --cst-right results/extraction/trk/sub-01_cst_right.trk --fa results/tracking/scalar_maps/sub-01_fa.nii.gz --out results
```
```

**Key Features with Visuals:**
- End-to-end pipeline from DICOM to metrics
- Bilateral CST extraction with atlas-based ROIs
- Quality control visualizations
- Comprehensive JSON/CSV/PDF reports
- Modular architecture for easy extension

**Documentation Link:**
```markdown
## Documentation

📚 **[Full Documentation](https://csttool.readthedocs.io/)**

- [Installation Guide](link)
- [First Analysis Tutorial](link)
- [CLI Reference](link)
- [API Documentation](link)
```

**Citation:**
```markdown
## Citation

If you use csttool in your research, please cite:

```bibtex
@mastersthesis{nalo2025csttool,
  title={Automated Corticospinal Tract Assessment Using Diffusion MRI},
  author={Nalo, Alem},
  year={2025},
  school={University Name}
}
```
```

**Example References:**
- [FSL README](https://git.fmrib.ox.ac.uk/fsl/fsl)
- [DIPY README](https://github.com/dipy/dipy)
- [ANTs README](https://github.com/ANTsX/ANTs)

---

## Phase 2: User Documentation (Week 2-4)
*Goal: Enable users to successfully use csttool*

### Milestone 2.1: Getting Started Guide ⭐
**Priority: CRITICAL**

#### `docs/getting-started/installation.md`

**Content Structure:**
```markdown
# Installation

## System Requirements

- **Operating System:** Linux, macOS, or Windows (WSL2)
- **Python:** 3.10 or higher
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 10GB for software + space for data

## Dependencies

csttool requires the following core packages:
- DIPY (diffusion imaging)
- nibabel (neuroimaging I/O)
- NumPy, SciPy (numerical computing)
- matplotlib (visualization)
- nilearn (atlas handling)

## Installation Methods

### Method 1: Install from GitHub (Recommended)
[Step-by-step instructions]

### Method 2: Development Installation
[Instructions for contributors]

## Verify Installation

```bash
csttool check
```

## Troubleshooting

### Common Issues:
1. **Import errors...**
2. **DIPY installation fails...**
3. **Command not found...**
```

#### `docs/getting-started/data-requirements.md`

**Content:**
- DICOM requirements (series, directions, b-values)
- NIfTI format specifications
- bvals/bvecs format
- Recommended datasets (link to existing `recommended_datasets.md`)
- BIDS compliance notes
- Quality criteria for tractography

**Best Practice Resources:**
- [PyPA Packaging Tutorial](https://packaging.python.org/en/latest/tutorials/)
- [BIDS Specification](https://bids-specification.readthedocs.io/)

---

### Milestone 2.2: First Tutorial - End-to-End Analysis ⭐
**Priority: CRITICAL**

#### `docs/tutorials/first-analysis.md`

**Learning Objectives:**
- Install csttool
- Obtain sample data
- Run complete pipeline
- Interpret outputs
- Troubleshoot common issues

**Tutorial Structure:**
```markdown
# Your First CST Analysis

**⏱️ Time Required:** 15-20 minutes  
**📋 Prerequisites:** csttool installed, basic command-line familiarity

## Overview

This tutorial walks you through analyzing a single subject's diffusion MRI data to extract and characterize the corticospinal tract.

## Step 1: Get Sample Data (5 min)

We'll use preprocessed data from OpenNeuro dataset ds004910.

```bash
# Download subject sub-01
wget https://openneuro.org/crn/datasets/ds004910/snapshots/1.0.0/files/sub-01:dwi:sub-01_dwi.nii.gz
wget https://openneuro.org/crn/datasets/ds004910/snapshots/1.0.0/files/sub-01:dwi:sub-01_dwi.bval
wget https://openneuro.org/crn/datasets/ds004910/snapshots/1.0.0/files/sub-01:dwi:sub-01_dwi.bvec
```

## Step 2: Environment Check (1 min)

Verify csttool installation:
```bash
csttool check
```

✅ Expected output: [screenshot]

## Step 3: Run Complete Pipeline (10 min)

```bash
csttool run \
  --nifti sub-01_dwi.nii.gz \
  --out results_sub-01 \
  --subject-id sub-01 \
  --save-visualizations
```

⏳ This will take approximately 10 minutes.

### What's Happening?

The pipeline runs six stages:
1. **Check:** Verify environment
2. **Import:** Validate input data (skipped - already NIfTI)
3. **Preprocess:** Denoise and skull strip
4. **Track:** Generate whole-brain tractogram
5. **Extract:** Isolate bilateral CST
6. **Metrics:** Compute microstructural measures

## Step 4: Explore Results (5 min)

Navigate to your output directory:
```bash
cd results_sub-01
tree -L 2
```

Expected structure:
[Directory tree visualization]

### Key Outputs:

**Tractograms:**
- `extraction/trk/sub-01_cst_left.trk` - Left hemisphere CST
- `extraction/trk/sub-01_cst_right.trk` - Right hemisphere CST

**Metrics:**
- `metrics/reports/sub-01_metrics.json` - All computed metrics
- `metrics/reports/sub-01_metrics.csv` - Statistical summary
- `metrics/reports/sub-01_clinical_report.pdf` - Visual report

**Visualizations:**
- `extraction/visualizations/sub-01_extraction_summary.png`
- `metrics/visualizations/sub-01_tract_profiles.png`

### Interpreting Results

**Check your extraction rate:**
```bash
cat extraction/logs/sub-01_extraction_report.json | grep extraction_rate
```

✅ **Good:** 5-15% for HCP-quality data  
⚠️ **Low:** <2% may indicate data quality issues  
❌ **None:** Check registration and ROI overlap

**View FA along tract:**
Open `metrics/visualizations/sub-01_tract_profiles.png`

[Annotated example image showing:
- Expected FA range: 0.4-0.6
- Motor cortex (low FA) → Internal capsule (high FA) → Brainstem (moderate FA)
- Bilateral symmetry]

## Step 5: Common Issues

### Issue: "No streamlines pass through ROIs"

**Causes:**
- Registration failure
- Low-quality diffusion data
- Restrictive ROI dilation

**Solutions:**
1. Check registration: `extraction/visualizations/*_registration_qa.png`
2. Try looser parameters:
   ```bash
   csttool extract [...] --roi-dilation 5 --extraction-method passthrough
   ```

### Issue: "Preprocessing fails - cannot estimate noise"

**Cause:** Insufficient signal-to-noise ratio

**Solution:** Skip denoising (not recommended, but functional):
```bash
csttool preprocess --nifti data.nii.gz --out results --skip-denoising
```

## Next Steps

- **Multiple Subjects:** Learn batch processing in [Batch Processing Tutorial](link)
- **Custom Parameters:** See [Parameter Tuning Guide](link)
- **Understand Outputs:** Read [Output Format Reference](link)
- **Troubleshooting:** Full guide at [Troubleshooting](link)

## Summary

You've successfully:
✅ Obtained diffusion MRI data  
✅ Ran the complete csttool pipeline  
✅ Generated CST tractograms and metrics  
✅ Interpreted quality control outputs

**Typical Results for HCP-Quality Data:**
- Processing time: 8-15 minutes
- CST extraction rate: 5-15%
- Mean FA: 0.45-0.55
- Laterality indices: -0.1 to +0.1
```

**Best Practice Resources:**
- [Carpentries Lesson Template](https://carpentries.github.io/lesson-example/)
- [Real Python Tutorials](https://realpython.com/)
- [DIPY Tutorials](https://dipy.org/tutorials/)

---

### Milestone 2.3: Command Reference ⭐
**Priority: HIGH**

Create detailed reference pages for each CLI command.

#### Template for Each Command (`docs/reference/cli/*.md`):

```markdown
# csttool [command]

## Purpose

[One-sentence description]

## When to Use

[Scenario descriptions]

## Syntax

```bash
csttool [command] [REQUIRED_ARGS] [OPTIONS]
```

## Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--input` | Path | Description |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--verbose` | flag | False | Enable verbose output |

## Examples

### Basic Usage
```bash
[command with minimal args]
```

### Advanced Usage
```bash
[command with multiple options]
```

### Common Workflows
1. [Specific use case 1]
2. [Specific use case 2]

## Output Structure

```
output_dir/
├── [subdirectory]/
│   └── [files]
```

## Output Files

| File | Format | Description |
|------|--------|-------------|
| `file.ext` | Format | Purpose |

## Quality Control

- **What to check:** [QC steps]
- **Expected values:** [Ranges]
- **Warning signs:** [Red flags]

## Troubleshooting

### Common Errors

**Error:** "Message"
- **Cause:** Explanation
- **Solution:** Fix

## Related Commands

- [`csttool other-command`](link) - Related functionality

## See Also

- [Tutorial: First Analysis](link)
- [API Reference: Module](link)
```

**Commands to Document:**
1. `check` - Environment verification
2. `import` - DICOM conversion (with series selection)
3. `preprocess` - Denoising, masking, motion correction
4. `track` - Whole-brain tractography
5. `extract` - CST extraction (all three methods)
6. `metrics` - Bilateral analysis
7. `run` - Complete pipeline

**Best Practice Resources:**
- [Click Documentation Style](https://click.palletsprojects.com/)
- [Git Command Reference](https://git-scm.com/docs)
- [FSL Command Line Tools](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslCommands)

---

## Phase 3: How-To Guides (Week 4-5)
*Goal: Task-oriented recipes for specific use cases*

### Milestone 3.1: Common Workflows
**Priority: MEDIUM**

#### `docs/how-to/multiple-subjects.md`

**Content:**
```markdown
# Process Multiple Subjects

## Scenario

You have a cohort of 20 subjects and need to process them all with consistent parameters.

## Prerequisites

- csttool installed
- Data organized in BIDS-like structure
- Sufficient disk space (~5GB per subject)

## Strategy 1: Simple Bash Loop

```bash
#!/bin/bash

DATA_DIR="/path/to/data"
OUT_DIR="/path/to/output"

for subject in sub-01 sub-02 sub-03; do
  echo "Processing $subject..."
  
  csttool run \
    --nifti "$DATA_DIR/${subject}/dwi/${subject}_dwi.nii.gz" \
    --out "$OUT_DIR/$subject" \
    --subject-id "$subject" \
    --save-visualizations \
    --continue-on-error
    
  echo "Completed $subject"
done
```

## Strategy 2: Parallel Processing

[GNU parallel example]

## Strategy 3: Python Script

[Python with multiprocessing]

## Monitoring Progress

[Tips for tracking long-running processes]

## Aggregating Results

```bash
# Collect all metrics
python aggregate_results.py --results-dir /output --output summary.csv
```

[Example aggregation script]
```

#### Other How-To Guides:

- `data-formats.md` - Working with DICOM multi-series, pre-processed NIfTI
- `customize-extraction.md` - Adjusting ROI parameters, filtering strategies
- `interpret-results.md` - Understanding JSON/CSV outputs, generating figures
- `quality-control.md` - Identifying failed subjects, registration QA

---

### Milestone 3.2: Troubleshooting Guide
**Priority: HIGH (for thesis support)**

#### `docs/how-to/troubleshooting.md`

**Structure:**
```markdown
# Troubleshooting Guide

## Quick Diagnostics

Run these commands first:
```bash
csttool check
csttool --version
python -c "import dipy; print(dipy.__version__)"
```

## Installation Issues

### Problem: Command not found
### Problem: Import errors
### Problem: DIPY compilation fails

## Data Issues

### Problem: DICOM import fails
### Problem: Multiple series detected
### Problem: bvals/bvecs mismatch

## Processing Failures

### Problem: Preprocessing crashes
### Problem: No tractogram generated
### Problem: All streamlines filtered out

## Low Extraction Rates (<2%)

**This is the most common issue.**

[Your detailed learnings about:
- Data quality requirements
- Registration validation
- ROI overlap checking
- Parameter adjustment strategies]

## Performance Issues

### Problem: Very slow processing
### Problem: Out of memory errors

## Getting Help

1. Check [FAQ](link)
2. Search [GitHub Issues](link)
3. Open new issue with:
   - csttool version
   - Python version
   - Full command
   - Error message
   - Data characteristics
```

---

## Phase 4: Reference Documentation (Week 5-6)
*Goal: Complete technical reference*

### Milestone 4.1: API Documentation
**Priority: MEDIUM**

**Setup automatic API documentation:**

Install tools:
```bash
pip install mkdocstrings[python] mkdocs-material
```

Configure in `mkdocs.yml`:
```yaml
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
          options:
            docstring_style: numpy
            show_source: true
```

Create module pages:

#### `docs/reference/api/preprocess.md`
```markdown
# Preprocessing API

::: csttool.preprocess.funcs
    options:
      show_root_heading: true
      show_source: true
```

**Best Practice Resources:**
- [mkdocstrings Documentation](https://mkdocstrings.github.io/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)

---

### Milestone 4.2: Output Format Specifications
**Priority: CRITICAL (for thesis)**

#### `docs/reference/output-formats.md`

**Content:**
```markdown
# Output Format Reference

## Tractogram Files (.trk)

csttool produces TrackVis format tractograms.

**Format:** Binary, TrackVis .trk format  
**Specification:** http://www.trackvis.org/docs/?subsect=fileformat

**Contents:**
- Streamline coordinates (native subject space)
- Header with voxel size, dimensions, voxel order
- Number of streamlines

**Loading in Python:**
```python
from nibabel.streamlines import load
tractogram = load('cst_left.trk')
streamlines = tractogram.streamlines
```

## JSON Reports

All pipeline stages produce JSON reports for reproducibility.

### Extraction Report Schema

```json
{
  "subject_id": "sub-01",
  "timestamp": "2025-01-12T10:30:00",
  "inputs": {
    "tractogram": "path/to/whole_brain.trk",
    "fa_map": "path/to/fa.nii.gz"
  },
  "parameters": {
    "extraction_method": "passthrough",
    "roi_dilation": 3,
    "fa_threshold": 0.2
  },
  "results": {
    "cst_left_count": 1234,
    "cst_right_count": 1198,
    "extraction_rate": 12.3
  }
}
```

## CSV Metrics

### Unilateral Metrics
[Column definitions]

### Bilateral Metrics
[Column definitions]

## PDF Reports

[Contents and interpretation]
```

---

### Milestone 4.3: Parameters Reference
**Priority: HIGH**

#### `docs/reference/parameters.md`

Document all tunable parameters with defaults, ranges, and effects.

---

## Phase 5: Explanation & Theory (Week 6-7)
*Goal: Deep understanding of concepts and decisions*

### Milestone 5.1: Scientific Background
**Priority: HIGH (thesis context)**

#### `docs/explanation/diffusion-mri.md`
- DTI fundamentals
- FA/MD interpretation
- Limitations

#### `docs/explanation/tractography.md`
- Deterministic vs probabilistic
- Streamline generation
- Stopping criteria

#### `docs/explanation/cst-anatomy.md`
- Anatomical course
- Clinical significance
- Expected metrics

**Best Practice Resources:**
- [DIPY Theory Documentation](https://docs.dipy.org/stable/theory/)
- Link to key papers (Basser 1994, etc.)

---

### Milestone 5.2: Design Decisions ⭐
**Priority: CRITICAL (thesis justification)**

#### `docs/explanation/design-decisions.md`

**Document your key learnings:**

```markdown
# Design Decisions

## Atlas-to-Subject vs Streamlines-to-MNI

### Decision
csttool transforms the MNI atlas to subject space rather than transforming streamlines to MNI space.

### Rationale

**Advantages:**
1. **Preserves streamline integrity** - No coordinate transforms
2. **Direct FA/MD sampling** - Metrics computed in native space
3. **Simpler pipeline** - Standard approach in neuroimaging
4. **Better reproducibility** - Fewer transformation steps

**Disadvantages:**
1. Registration quality critical
2. Cannot directly compare streamline coordinates across subjects

### Evidence
[Your testing results, examples]

### References
- Standard practice in DIPY workflows
- [FSL TBSS documentation](link)

## Pass-Through Filtering

[Your corona radiata learnings]
[Why this was necessary]
[When to use vs endpoint filtering]

## ROI Dilation Strategy

[Your pragmatic approach]
[Trade-offs documented]
```

---

### Milestone 5.3: Known Limitations ⭐
**Priority: CRITICAL (intellectual honesty)**

#### `docs/explanation/limitations.md`

```markdown
# Known Limitations

## Deterministic Tractography

### Crossing Fiber Issue

**Problem:** Streamlines terminate prematurely in regions with crossing fibers (e.g., corona radiata) due to reduced FA.

**Impact:** Creates unbridgeable gaps between anatomical regions that should be connected.

**Evidence:** [Your testing, figures showing termination patterns]

**Mitigation:**
- Pass-through filtering recovers some connections
- Generous ROI dilation
- Acceptance that deterministic methods have inherent limitations

**Alternative Approaches:**
- Probabilistic tractography (future work)
- Higher-order diffusion models (HARDI, DSI)

## Clinical vs Research Use

csttool is designed as a research pipeline, not a clinical diagnostic tool.

**Not suitable for:**
- Clinical diagnosis
- Treatment planning
- Individual patient decision-making

**Requires validation for:**
- Different scanners
- Different protocols
- Clinical populations

## Dataset Requirements

[Minimum quality standards]
[Situations where extraction may fail]
```

**Best Practice Resources:**
- [Scientific honesty in reporting](https://www.nature.com/articles/d41586-019-01715-4)
- Example: [FSL Known Issues](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FAQ)

---

## Phase 6: Developer Documentation (Week 7-8)
*Goal: Enable contributions and extensions*

### Milestone 6.1: Contributing Guide
**Priority: LOW (post-thesis)**

#### `docs/contributing/development-setup.md`

```markdown
# Development Setup

## Fork and Clone

```bash
git clone https://github.com/yourusername/csttool.git
cd csttool
```

## Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

## Install in Editable Mode

```bash
pip install -e ".[test]"
```

## Run Tests

```bash
pytest tests/
```

## Code Style

We follow PEP 8 with NumPy docstring conventions.

```bash
# Format code
black src/

# Check style
flake8 src/
```
```

---

### Milestone 6.2: Architecture Documentation
**Priority: LOW**

#### `docs/contributing/architecture.md`

- Module dependency diagram
- Data flow
- Extension points
- Performance considerations

---

## Phase 7: Final Polish (Week 8-9)
*Goal: Professional presentation*

### Milestone 7.1: Visual Assets

- Pipeline flowcharts (expand existing diagrams)
- Example outputs (screenshots)
- QC visualizations
- Optional: 5-minute video tutorial

**Tools:**
- Graphviz for diagrams
- Matplotlib for figures
- OBS Studio for recording

---

### Milestone 7.2: Citation and Publication

#### `CITATION.cff`
```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: Nalo
    given-names: Alem
title: "csttool: Automated CST Assessment Using Diffusion MRI"
version: 0.1.0
date-released: 2025-01-XX
url: "https://github.com/ravenholm462/csttool"
```

#### Zenodo DOI
- Archive repository on Zenodo
- Get permanent DOI
- Add badge to README

#### Optional: JOSS Paper
- [Journal of Open Source Software](https://joss.theoj.org/)
- Short paper describing software
- Peer-reviewed
- Citable publication

---

### Milestone 7.3: Documentation Review

**Checklist:**
- [ ] All links work
- [ ] Code examples run
- [ ] Screenshots current
- [ ] Spelling/grammar checked
- [ ] User testing completed
- [ ] Feedback incorporated

**User Testing:**
- Have 2-3 people unfamiliar with csttool try the tutorial
- Note where they get stuck
- Revise based on feedback

---

## Summary: Priority Tiers

### 🔴 CRITICAL (Do First - 3-4 weeks)

Essential for thesis and basic usability:

1. **MkDocs Setup** (Phase 1.1)
2. **README Enhancement** (Phase 1.2)
3. **Installation Guide** (Phase 2.1)
4. **First Tutorial** (Phase 2.2)
5. **Output Formats** (Phase 4.2)
6. **Design Decisions** (Phase 5.2)
7. **Known Limitations** (Phase 5.3)

### 🟡 HIGH (Thesis Value - 2 weeks)

Strengthens academic contribution:

8. **CLI Reference** (Phase 2.3)
9. **Scientific Background** (Phase 5.1)
10. **Troubleshooting** (Phase 3.2)
11. **Parameter Reference** (Phase 4.3)

### 🟢 MEDIUM (Good Practice - 2 weeks)

Improves usability:

12. **How-To Guides** (Phase 3.1)
13. **API Documentation** (Phase 4.1)
14. **Architecture** (Phase 6.2)

### 🔵 LOW (Nice to Have - 1-2 weeks)

Post-thesis improvements:

15. **Contributing Guide** (Phase 6.1)
16. **Video Tutorial** (Phase 7.1)
17. **JOSS Paper** (Phase 7.2)

---

## Best Practice Resources Summary

### Documentation Frameworks
- ⭐ [Diátaxis Framework](https://diataxis.fr/) - THE framework
- [Write the Docs Guide](https://www.writethedocs.org/guide/)
- [Google Style Guide](https://developers.google.com/style)

### Scientific Python
- ⭐ [Scientific Python Dev Guide](https://learn.scientific-python.org/development/)
- [NumPy Documentation Guide](https://numpy.org/devdocs/dev/howto-docs.html)
- [DIPY Documentation](https://docs.dipy.org/)

### Neuroimaging Examples
- ⭐ [FSL Documentation](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)
- [ANTs Wiki](https://github.com/ANTsX/ANTs/wiki)
- [MRtrix3 Docs](https://mrtrix.readthedocs.io/)

### Tools
- ⭐ [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [Read the Docs](https://readthedocs.org/)

---

## Timeline Recommendation

| Week | Phase | Hours/Week | Focus |
|------|-------|------------|-------|
| 1 | Setup + README | 8-10 | Infrastructure |
| 2-3 | Installation + Tutorial | 10-12 | User onboarding |
| 4 | CLI Reference | 8-10 | Command docs |
| 5 | Output + Parameters | 8-10 | Reference |
| 6 | Theory + Decisions | 10-12 | Explanation |
| 7 | Troubleshooting | 6-8 | Support |
| 8 | Review + Polish | 6-8 | Quality |

**Total: ~70 hours for critical path (Weeks 1-7)**

---

## Success Metrics

**Documentation is successful when:**
- [ ] A new user can run their first analysis in <30 minutes
- [ ] 80% of questions answered in docs (track via GitHub issues)
- [ ] Thesis reviewers find technical decisions well-justified
- [ ] Contributors can understand codebase from docs
- [ ] Future you can remember why you made certain choices

---

**Next Steps:**
1. Set up MkDocs Material (see setup guide)
2. Create enhanced README
3. Write first tutorial
4. Get feedback from test user
5. Iterate based on usage

Good luck with your documentation! 🚀

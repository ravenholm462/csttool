# CST Validation Feature for csttool

Add validation capability to compare csttool-extracted CST streamlines against reference tractograms (e.g., TractoInferno PYT bundles) and optionally use preprocessed data.

## User Review Required

> [!IMPORTANT]
> **Metric Selection**: The plan proposes bundle overlap and streamline distance metrics. Please confirm if you want all of these or prefer a subset.

> [!IMPORTANT]
> **Preprocessed Data Modes**: Should csttool support using external fODF/FA maps directly, or just provide validation against reference bundles?

---

## Background: TractoInferno Preprocessed Data

The dataset provides the following preprocessed data per subject:

| Directory | Contents | Potential Use |
|-----------|----------|---------------|
| `dti/` | FA, AD, MD, RD maps | Skip DTI fitting; use FA directly for registration |
| `fodf/` | fODF + peaks | Skip CSD; use fODF directly for tractography |
| `mask/` | WM, GM, CSF masks | Use WM mask for seeding/tracking |
| `tractography/` | **PYT_L, PYT_R** (CST) | **Ground truth for validation** |

---

## Proposed Changes

### Validation Module

#### [NEW] [\_\_init\_\_.py](file:///home/alem/csttool/src/csttool/validation/__init__.py)

Module init exposing validation functions.

---

#### [NEW] [bundle_comparison.py](file:///home/alem/csttool/src/csttool/validation/bundle_comparison.py)

Core validation metrics:

```python
def compute_bundle_overlap(candidate_trk, reference_trk, voxel_size):
    """Dice overlap on streamline density maps."""

def compute_overreach(candidate_trk, reference_trk, voxel_size):
    """Fraction of candidate outside reference envelope."""

def mean_closest_distance(candidate_trk, reference_trk):
    """MDF: average closest point distance between bundles."""

def generate_validation_report(metrics_dict, output_path):
    """Generate HTML/JSON validation report."""
```

---

### CLI Integration

#### [NEW] [validate.py](file:///home/alem/csttool/src/csttool/cli/commands/validate.py)

New CLI command:

```bash
csttool validate \
    --candidate /path/to/cst_left.trk /path/to/cst_right.trk \
    --reference /path/to/PYT_L.trk /path/to/PYT_R.trk \
    --output-dir /path/to/validation_output
```

**Arguments:**
- `--candidate`: csttool-extracted CST tractograms (L/R)
- `--reference`: Ground truth bundles (e.g., TractoInferno PYT)
- `--output-dir`: Output directory for reports
- `--metrics`: Optional list of metrics to compute (default: all)

---

#### [MODIFY] [\_\_init\_\_.py](file:///home/alem/csttool/src/csttool/cli/commands/__init__.py)

Register the new `validate` subcommand.

---

### Optional: Preprocessed Data Input Mode

#### [MODIFY] [run.py](file:///home/alem/csttool/src/csttool/cli/commands/run.py)

Add optional flags to skip internal processing and use external data:

```bash
csttool run --nifti input.nii.gz \
    --external-fa /path/to/fa.nii.gz \
    --external-fodf /path/to/fodf.nii.gz \
    --external-wm-mask /path/to/wm_mask.nii.gz
```

This allows direct use of TractoInferno preprocessed data for faster processing and fair comparison.

---

## Verification Plan

### Automated Tests

#### Unit Tests

Create `tests/validation/test_bundle_comparison.py`:

```bash
pytest tests/validation/test_bundle_comparison.py -v
```

Test cases:
1. **Identical bundles** → Dice = 1.0, MDF = 0.0
2. **Disjoint bundles** → Dice = 0.0
3. **Partial overlap** → 0 < Dice < 1

---

### Manual Verification

1. **Download TractoInferno reference bundles:**
   ```bash
   cd /home/alem/data/thesis/in/ds003900/derivatives/trainset/sub-1282/tractography
   datalad get sub-1282__PYT_L.trk sub-1282__PYT_R.trk
   ```

2. **Run csttool on same subject** (if not already processed)

3. **Run validation command:**
   ```bash
   csttool validate \
       --candidate output/sub-1282/cst_left.trk output/sub-1282/cst_right.trk \
       --reference /path/to/PYT_L.trk /path/to/PYT_R.trk \
       --output-dir output/sub-1282/validation
   ```

4. **Review output:** Check generated report for Dice, overreach, and MDF values.

---

## Implementation Order

1. Create `validation/` module with metrics
2. Add CLI `validate` command
3. Write unit tests
4. (Optional) Add external preprocessed data flags to `run`

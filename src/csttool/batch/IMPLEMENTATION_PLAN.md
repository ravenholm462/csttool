# Batch Processing for csttool

## Problem Statement

Currently, csttool processes one subject at a time via `csttool run`. For studies with multiple subjects, each with multiple sessions, users must manually loop through subjects in shell scripts. This lacks unified progress tracking, aggregate reporting, and error recovery.

---

## Design Decisions (Confirmed)

| Decision | Choice |
|----------|--------|
| **Execution model** | Sequential (v1), threading as future option (v2) |
| **Manifest format** | JSON |
| **Input structure** | BIDS-compliant (subject/session/datatype) |
| **Output structure** | BIDS-derivative-like |

---

## Proposed Changes

### Batch Module Structure

```
src/csttool/batch/
├── __init__.py              # Package exports
├── batch.py                 # Main orchestrator (run_batch)
├── IMPLEMENTATION_PLAN.md   # This file
├── TASK.md                  # Progress tracking
└── modules/
    ├── __init__.py
    ├── discover.py          # BIDS subject/session discovery
    ├── manifest.py          # JSON manifest loading/validation
    └── report.py            # Batch summary report generation
```

---

### BIDS Input Structure (Expected)

```
/data/study/
├── dataset_description.json        # Optional BIDS metadata
├── participants.tsv                # Optional participant list
├── sub-001/
│   ├── ses-01/
│   │   └── dwi/
│   │       ├── sub-001_ses-01_dwi.nii.gz
│   │       ├── sub-001_ses-01_dwi.bval
│   │       └── sub-001_ses-01_dwi.bvec
│   └── ses-02/
│       └── dwi/
│           └── ...
├── sub-002/
│   └── ses-01/
│       └── dwi/
│           └── ...
└── sub-003/
    └── ses-01/
        └── dwi/
            └── ...
```

Discovery logic:
1. Find directories matching `sub-*` pattern
2. Within each subject, find directories matching `ses-*` pattern
3. Within each session, look for `dwi/` subdirectory
4. Find `*_dwi.nii.gz` + matching `.bval`/`.bvec`

> [!NOTE]
> Sessions are optional in BIDS. Discovery should handle both:
> - `sub-001/ses-01/dwi/` (with sessions)
> - `sub-001/dwi/` (without sessions, session = None)

---

### BIDS-Derivative Output Structure

```
/results/derivatives/csttool/
├── dataset_description.json        # BIDS derivative metadata
├── batch_summary.json              # Execution log
├── batch_metrics.csv               # Aggregate metrics (all subjects/sessions)
├── sub-001/
│   ├── ses-01/
│   │   ├── dwi/
│   │   │   └── sub-001_ses-01_desc-preproc_dwi.nii.gz
│   │   ├── tractography/
│   │   │   ├── sub-001_ses-01_tractogram.trk
│   │   │   ├── sub-001_ses-01_desc-cstL_tractogram.trk
│   │   │   └── sub-001_ses-01_desc-cstR_tractogram.trk
│   │   └── metrics/
│   │       ├── sub-001_ses-01_metrics.json
│   │       ├── sub-001_ses-01_metrics.csv
│   │       └── sub-001_ses-01_report.pdf
│   └── ses-02/
│       └── ...
├── sub-002/
│   └── ses-01/
│       └── ...
└── logs/
    ├── sub-001_ses-01.log
    └── sub-003_ses-01_error.log
```

---

### JSON Manifest Schema

```json
{
  "name": "Study Name",
  "description": "Optional study description",
  "options": {
    "denoise_method": "patch2self",
    "generate_pdf": true,
    "extraction_method": "passthrough"
  },
  "subjects": [
    {
      "id": "sub-001",
      "session": "ses-01",
      "nifti": "/absolute/path/to/sub-001_ses-01_dwi.nii.gz"
    },
    {
      "id": "sub-001",
      "session": "ses-02",
      "nifti": "/absolute/path/to/sub-001_ses-02_dwi.nii.gz"
    },
    {
      "id": "sub-002",
      "session": "ses-01",
      "dicom": "/absolute/path/to/sub-002/ses-01/dicom/"
    },
    {
      "id": "sub-003",
      "session": "ses-01",
      "nifti": "/path/to/sub-003_ses-01_dwi.nii.gz",
      "options": {
        "denoise_method": "nlmeans"
      }
    }
  ]
}
```

- `session` field required for each subject entry
- Same subject can appear multiple times with different sessions
- `options` at root level: defaults for all subjects
- `options` per subject: overrides for that subject/session

---

### Module Specifications

#### [NEW] [batch.py](file:///home/alem/csttool/src/csttool/batch/batch.py)

```python
@dataclass
class SubjectSpec:
    subject_id: str
    session_id: str | None           # None if no session
    input_path: Path
    input_type: Literal["nifti", "dicom"]
    options: dict                    # Per-subject/session overrides

@dataclass
class BatchConfig:
    output_dir: Path
    denoise_method: str = "patch2self"
    generate_pdf: bool = False
    continue_on_error: bool = True
    # ... other shared options

@dataclass  
class SubjectResult:
    subject_id: str
    session_id: str | None
    success: bool
    duration_seconds: float
    error: str | None = None
    outputs: dict | None = None

def run_batch(
    subjects: list[SubjectSpec],
    config: BatchConfig,
    verbose: bool = False
) -> list[SubjectResult]:
    """Run csttool pipeline for each subject/session sequentially."""
```

---

#### [NEW] [discover.py](file:///home/alem/csttool/src/csttool/batch/modules/discover.py)

```python
def discover_subjects(
    bids_dir: Path,
    subject_pattern: str = "sub-*",
    session_pattern: str = "ses-*"
) -> list[SubjectSpec]:
    """
    Discover subjects/sessions from BIDS directory structure.
    
    Handles both:
    - {bids_dir}/sub-*/ses-*/dwi/*_dwi.nii.gz (with sessions)
    - {bids_dir}/sub-*/dwi/*_dwi.nii.gz (without sessions)
    """

def validate_subject_inputs(subjects: list[SubjectSpec]) -> list[str]:
    """Validate all subject inputs exist. Returns list of errors."""
```

---

#### [NEW] [manifest.py](file:///home/alem/csttool/src/csttool/batch/modules/manifest.py)

```python
def load_manifest(manifest_path: Path) -> tuple[list[SubjectSpec], BatchConfig]:
    """Load and validate JSON manifest file."""

def validate_manifest_schema(data: dict) -> list[str]:
    """Validate manifest against expected schema."""
```

---

#### [NEW] [report.py](file:///home/alem/csttool/src/csttool/batch/modules/report.py)

```python
def generate_batch_summary(
    results: list[SubjectResult],
    output_dir: Path,
    config: BatchConfig
) -> None:
    """Generate batch_summary.json and batch_metrics.csv."""

def create_bids_derivative_metadata(
    output_dir: Path,
    source_dataset: str | None = None
) -> None:
    """Create dataset_description.json for BIDS derivative."""
```

---

### CLI Addition

#### [MODIFY] [cli.py](file:///home/alem/csttool/src/csttool/cli.py)

```python
p_batch = subparsers.add_parser("batch", help="Process multiple subjects/sessions")

# Input (mutually exclusive)
input_group = p_batch.add_mutually_exclusive_group(required=True)
input_group.add_argument("--bids-dir", type=Path, help="BIDS directory to discover subjects")
input_group.add_argument("--manifest", type=Path, help="JSON manifest file")

# Discovery options
p_batch.add_argument("--subject-pattern", default="sub-*")
p_batch.add_argument("--session-pattern", default="ses-*")

# Output
p_batch.add_argument("--out", type=Path, required=True)

# Batch control
p_batch.add_argument("--continue-on-error", action="store_true", default=True)
p_batch.add_argument("--dry-run", action="store_true")
```

---

## Implementation Order

1. `modules/discover.py` - BIDS subject/session discovery
2. `modules/manifest.py` - JSON manifest loading
3. `batch.py` - Sequential batch orchestrator  
4. `modules/report.py` - Batch summary generation
5. CLI integration in `cli.py`
6. Tests

---

## Future: Parallel Processing (v2)

Threading/multiprocessing considerations:
- `concurrent.futures.ThreadPoolExecutor` for I/O-bound steps
- Memory usage (DWI data is large)
- Logging from parallel workers

# Batch Processing for csttool

## Problem Statement

Currently, csttool processes one subject at a time via `csttool run`. For studies with multiple subjects, each with multiple sessions, users must manually loop through subjects in shell scripts. This lacks unified progress tracking, aggregate reporting, and error recovery.

---

## Design Decisions (Confirmed)

| Decision | Choice |
|----------|--------|
| **Execution model** | Sequential, one child process per subject (v1); pool-based parallelism (v2) |
| **Process isolation** | Each subject runs in `multiprocessing.Process` for safe timeout and memory reclamation |
| **Manifest format** | JSON |
| **Input structure** | BIDS-compliant (subject/session/datatype) |
| **Output structure** | BIDS-derivative-like (not fully compliant) |
| **Resume behavior** | Resume by default; `_done.json` is source of truth |
| **Input types** | NIfTI and DICOM (with series disambiguation) |
| **DICOM conversion** | Use existing `ingest.convert_series`; explicit `series_uid` if ambiguous |
| **Pipeline integration** | Internal Python API called from child process |
| **Subject timeout** | 120 minutes default; enforced via `Process.join(timeout=...) + terminate()` |
| **Signal handling** | Graceful stop with 30s grace period, then SIGKILL |
| **Locking** | `fcntl.flock()` as sole authority; metadata informational only |
| **Output atomicity** | Atomic rename from `_work/`; `_done.json` written last |
| **Option precedence** | CLI > manifest per-subject > manifest root > code defaults |

---

## Error Handling Strategy

### Error Categories

```python
class ErrorCategory(Enum):
    INPUT_MISSING = "input_missing"      # Files not found
    VALIDATION = "validation"            # Invalid data format
    PIPELINE_FAILED = "pipeline_failed"  # Processing step failed
    TIMEOUT = "timeout"                  # Subject exceeded time limit
    SYSTEM = "system"                    # Disk/memory/permissions (batch-abort)
```

### Error Response Matrix

| Scenario | Category | Behavior |
|----------|----------|----------|
| Subject input file missing | `INPUT_MISSING` | Skip subject, continue batch |
| Invalid bval/bvec format | `VALIDATION` | Skip subject, continue batch |
| Pipeline step fails mid-run | `PIPELINE_FAILED` | Move work to failed_attempts, continue batch |
| Subject exceeds 120 min | `TIMEOUT` | Kill process, move work to failed_attempts, continue |
| Disk full / permissions error | `SYSTEM` | **Abort batch immediately** |
| Output dir locked | `SYSTEM` | **Abort batch immediately** |
| Subject already completed | — | Skip subject (resume behavior), log as "skipped" |
| `--force` flag provided | — | Re-process all subjects, ignoring prior completions |

---

## Work Directory Strategy

Intermediate files are written to `_work/` during processing, promoted or moved on completion:

```
sub-001/ses-01/
├── metrics/              # Final outputs (on success)
├── tractography/         # Final outputs (on success)
├── logs/                 # Per-attempt logs with timestamps
├── _work/                # Current processing attempt
├── _done.json            # Completion marker (on success)
└── _failed_attempts/
    ├── 2026-01-22T07-55-12_TIMEOUT/
    └── 2026-01-22T08-31-04_PIPELINE_FAILED/
```

### Behavior

| Event | Action |
|-------|--------|
| Processing starts | Create/clear `_work/`, write intermediates there |
| **Success** | Atomic promotion (see below), delete `_work/`, write `_done.json` last |
| **Failure** | Move `_work/` to `_failed_attempts/<timestamp>_<category>/`, log failure |
| **SIGINT/SIGTERM** | Graceful stop (30s grace), move `_work/` to failed, exit |
| `--keep-work` flag | Retain `_work/` even on success (for debugging) |

### Atomic Output Promotion

On success, outputs are promoted atomically:

```python
def promote_outputs(work_dir: Path, final_dir: Path, done_marker: dict) -> None:
    """
    Atomically promote outputs from _work/ to final location.
    
    Sequence:
    1. Verify all expected outputs exist in _work/
    2. For each output file:
       - os.rename(_work/file, final/file)  # Atomic on same filesystem
    3. Write _done.json atomically:
       - Write to _done.json.tmp
       - os.rename(_done.json.tmp, _done.json)
    4. Delete _work/ only after _done.json rename succeeds
    
    Invariant: If _done.json exists, all outputs listed in it exist.
    """
```

### Completion Marker (`_done.json`)

Written last on successful completion:

```json
{
  "csttool_version": "1.2.3",
  "config_hash": "sha256:abc123...",
  "completed_at": "2026-01-22T08:45:00Z",
  "duration_seconds": 847.3,
  "outputs": {
    "tractogram_left": "tractography/sub-001_ses-01_desc-cstL_tractogram.trk",
    "tractogram_right": "tractography/sub-001_ses-01_desc-cstR_tractogram.trk",
    "metrics": "metrics/sub-001_ses-01_metrics.json"
  }
}
```

> [!IMPORTANT]
> **Resume source of truth:** `_done.json` in each subject directory
> - Checked first; if exists and config_hash matches, subject is skipped
> - File-based, survives `batch_summary.json` deletion or corruption
>
> **Batch summary role:** `batch_summary.json` at output root
> - Execution history and summary counters
> - For reporting and debugging only
> - **NOT used for resume decisions**
> - Safe to delete without affecting resume behavior

---

## Process Isolation Architecture

Each subject runs in a separate child process for safe timeout enforcement and memory reclamation.

### Why Process Isolation?

| Approach | Problem |
|----------|---------|
| Single process + `signal.SIGALRM` | Unix-only; corrupts state mid-write |
| Single process + threading flag | Blocking C-extensions (numpy, nibabel) won't check flag |
| Thread exception injection | Unreliable, leaks resources |

**Solution:** `multiprocessing.Process` per subject.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Parent Process (Orchestrator)                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  run_batch()                                          │  │
│  │    for subject in subjects:                           │  │
│  │      result = run_subject_with_timeout(subject)       │  │
│  │      update_batch_summary(result)                     │  │
│  │      if result.status == "failed":                    │  │
│  │        move_work_to_failed_attempts()                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          │ spawn                             │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Child Process (per subject)                          │  │
│  │    setup_subject_logger()                             │  │
│  │    result = process_single_subject(subject, config)   │  │
│  │    result_queue.put(result)                           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Timeout Enforcement

```python
proc = Process(target=_run_subject_worker, args=(subject, config, queue))
proc.start()
proc.join(timeout=config.timeout_minutes * 60)

if proc.is_alive():
    proc.terminate()  # Sends SIGTERM
    proc.join()       # Wait for cleanup
    return SubjectResult(status="failed", error_category=ErrorCategory.TIMEOUT)
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **Reliable timeout** | OS-level SIGTERM; interrupts any blocking call |
| **Memory reclamation** | Child memory freed on exit; no accumulation across 50+ subjects |
| **State isolation** | Corrupted child cannot affect parent orchestrator |
| **Clean cleanup** | Parent moves `_work/` to `_failed_attempts/` after termination |
| **Future parallelism** | Naturally extends to `ProcessPoolExecutor` in v2 |

### Serialization Requirements

For `multiprocessing.Queue`:
- `SubjectSpec`, `BatchConfig`, `SubjectResult` must be picklable
- Use dataclasses with simple types (Path, str, int, dict)
- Avoid lambdas, open file handles, or logger objects

### Child Process Logging

Child process sets up its own logger (cannot share with parent):

```python
def _run_subject_worker(subject, config, log_path, result_queue):
    # Create fresh logger in child
    logger = setup_subject_logger(subject.subject_id, subject.session_id, log_path)
    # ...
```

### Signal Handling (SIGINT/SIGTERM)

Graceful interruption with 30-second grace period:

```python
import signal

GRACE_PERIOD_SECONDS = 30
current_child: Process | None = None

def handle_interrupt(signum, frame):
    """Handle SIGINT/SIGTERM in parent orchestrator."""
    global current_child
    
    # 1. Immediately mark batch as interrupted
    write_batch_summary(status="interrupted")
    
    # 2. Attempt graceful shutdown of current subject
    if current_child and current_child.is_alive():
        current_child.terminate()  # Send SIGTERM
        current_child.join(timeout=GRACE_PERIOD_SECONDS)
        
        # 3. Force kill if still running
        if current_child.is_alive():
            current_child.kill()  # Send SIGKILL
            current_child.join()
    
    # 4. Move any _work/ to _failed_attempts/
    cleanup_interrupted_work()
    
    sys.exit(130 if signum == signal.SIGINT else 143)

signal.signal(signal.SIGINT, handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)
```

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
    ├── discover.py          # BIDS subject/session discovery (NIfTI + DICOM)
    ├── validation.py        # Preflight validation (environment, inputs, outputs)
    ├── manifest.py          # JSON manifest loading/validation
    ├── locking.py           # Directory locking for concurrent protection
    ├── logging_setup.py     # Per-subject log isolation
    └── report.py            # Batch summary report generation
```

---

### Locking Mechanism

Uses `fcntl.flock()` as the **sole authority** for lock ownership:

```python
# modules/locking.py
import fcntl
import os

@dataclass
class BatchLock:
    lock_file: Path
    fd: int

def acquire_batch_lock(output_dir: Path) -> BatchLock:
    """
    Acquire batch lock using fcntl.flock().
    
    The held flock IS the authority - not PID, not timestamp.
    Metadata in lock file is informational only (for debugging).
    
    Raises:
        LockError if lock cannot be acquired (another process holds it)
    """
    lock_path = output_dir / "batch.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise LockError(f"Batch lock held - see {lock_path} for details")
    
    # Write informational metadata (not used for lock decisions)
    lock_path.write_text(json.dumps({
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": datetime.now().isoformat()
    }))
    
    return BatchLock(lock_file=lock_path, fd=fd)

def acquire_subject_lock(subject_dir: Path) -> SubjectLock:
    """
    Create .lock file in subject output directory.
    
    Uses same flock pattern. Protects against parallel processing.
    """

def release_lock(lock: BatchLock | SubjectLock) -> None:
    """Release flock and close file descriptor. May delete lock file."""
    fcntl.flock(lock.fd, fcntl.LOCK_UN)
    os.close(lock.fd)
    lock.lock_file.unlink(missing_ok=True)
```

> [!NOTE]
> Locks use `fcntl.flock()` as the **sole authority**.
> Metadata (PID, hostname, timestamp) is informational only for debugging.
> Stale lock cleanup only attempted after flock acquisition succeeds.

---

### Logging Architecture

Per-subject log isolation with structured format:

```python
# modules/logging_setup.py

def setup_subject_logger(
    subject_id: str,
    session_id: str | None,
    log_dir: Path,
    level: str = "INFO"
) -> logging.Logger:
    """
    Create isolated logger for this subject/session.
    
    Log file: {log_dir}/{subject_id}_{session_id}_{timestamp}.log
    Format: JSON lines (for machine parsing) or text (for human reading)
    """
```

#### Log File Naming

```
logs/
├── sub-001_ses-01_2026-01-22T08-45-00.log
├── sub-001_ses-01_2026-01-22T09-12-33.log  # Retry attempt
└── sub-002_ses-01_2026-01-22T10-00-00.log
```

#### Log Format (JSON Lines)

```json
{"time": "2026-01-22T08:45:01Z", "level": "INFO", "step": "preprocess", "msg": "Starting denoising..."}
{"time": "2026-01-22T08:47:23Z", "level": "INFO", "step": "preprocess", "msg": "Denoising complete"}
{"time": "2026-01-22T08:47:24Z", "level": "INFO", "step": "track", "msg": "Starting tracking..."}
```

---

### BIDS Input Structure (Expected)

Supports both NIfTI and DICOM input:

#### NIfTI Input Structure
```
/data/study/
├── dataset_description.json        # Optional BIDS metadata
├── participants.tsv                # Optional participant list
├── sub-001/
│   ├── ses-01/
│   │   └── dwi/
│   │       ├── sub-001_ses-01_dwi.nii.gz
│   │       ├── sub-001_ses-01_dwi.bval
│   │       ├── sub-001_ses-01_dwi.bvec
│   │       └── sub-001_ses-01_dwi.json   # Optional sidecar
│   └── ses-02/
│       └── dwi/
│           └── ...
├── sub-002/
│   └── ses-01/
│       └── dwi/
│           └── ...
└── sub-003/
    └── dwi/                        # No session (session = None)
        └── sub-003_dwi.nii.gz
```

#### DICOM Input Structure
```
/data/study/
├── sub-001/
│   ├── ses-01/
│   │   └── dwi/
│   │       └── *.dcm              # DICOM files
│   └── ses-02/
│       └── dwi/
│           └── *.dcm
└── sub-002/
    └── dwi/                        # No session
        └── *.dcm
```

Discovery logic:
1. Find directories matching `sub-*` pattern
2. Within each subject, find directories matching `ses-*` pattern (if any)
3. Within each session (or subject if no session), look for `dwi/` subdirectory
4. Detect input type:
   - **NIfTI**: Find `*_dwi.nii.gz` + matching `.bval`/`.bvec` (+ optional `.json`)
   - **DICOM**: Find `*.dcm` files (no NIfTI present)

> [!NOTE]
> Sessions are optional in BIDS. Discovery handles both:
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
      "session": null,
      "dicom": "/absolute/path/to/sub-002/dicom/",
      "series_uid": "1.2.840.113619.2.55.3..."
    },
    {
      "id": "sub-003",
      "nifti": "/path/to/sub-003_dwi.nii.gz",
      "options": {
        "denoise_method": "nlmeans"
      }
    }
  ]
}
```

**Schema rules:**
- `session` is **optional** (omit or set to `null` for sessionless subjects)
- Exactly one of `nifti` or `dicom` must be specified per subject
- `series_uid` is **optional** for DICOM subjects (required if directory has multiple series)
- `id` must match BIDS pattern: alphanumeric, hyphen, underscore only
- Same subject can appear multiple times with different sessions
- `options` at root level: defaults for all subjects
- `options` per subject: overrides for that subject/session

---

### BIDS Derivative Metadata

Automatically generated `dataset_description.json`:

```json
{
  "Name": "csttool derivatives",
  "BIDSVersion": "1.9.0",
  "DatasetType": "derivative",
  "GeneratedBy": [
    {
      "Name": "csttool",
      "Version": "1.2.3",
      "CodeURL": "https://github.com/..."
    }
  ],
  "SourceDatasets": [
    {
      "URL": "file:///data/study/"
    }
  ]
}
```

### BIDS Derivative Scope

csttool produces **BIDS-derivative-like** output, not fully compliant BIDS derivatives.

**Included (v1):**
- Valid `dataset_description.json` with GeneratedBy and SourceDatasets
- BIDS-style file naming (`sub-*_ses-*_desc-*_suffix.ext`)
- BIDS directory structure (`sub-*/ses-*/datatype/`)

**Not included (v1):**
- `participants.tsv` propagation from source
- `scans.tsv` generation at session level
- Full provenance chain in sidecar JSON files
- BIDS Validator compliance

**Future (v2):**
- Optional `--bids-compliant` flag for stricter output
- `participants.tsv` propagation
- `scans.tsv` generation

---

### Module Specifications

#### [NEW] [batch.py](file:///home/alemnalo/csttool/src/csttool/batch/batch.py)

```python
@dataclass
class SubjectSpec:
    subject_id: str
    session_id: str | None           # None if no session
    input_path: Path
    input_type: Literal["nifti", "dicom"]
    bval_path: Path | None           # For NIfTI inputs
    bvec_path: Path | None           # For NIfTI inputs
    json_path: Path | None           # Optional sidecar
    series_uid: str | None           # For DICOM with multiple series
    options: dict                    # Per-subject/session overrides

@dataclass
class BatchConfig:
    output_dir: Path
    denoise_method: str = "patch2self"
    generate_pdf: bool = False
    continue_on_error: bool = True
    force: bool = False              # Ignore prior completions
    skip_preprocessing: bool = True
    keep_work: bool = False          # Retain _work/ on success
    timeout_minutes: int = 120       # Per-subject timeout
    # ... other shared options from cmd_run

@dataclass  
class SubjectResult:
    subject_id: str
    session_id: str | None
    status: Literal["success", "failed", "skipped"]
    error_category: ErrorCategory | None  # None if success/skipped
    duration_seconds: float
    error: str | None = None
    log_path: Path | None = None     # Path to subject log file
    outputs: dict | None = None      # Paths to outputs

def run_batch(
    subjects: list[SubjectSpec],
    config: BatchConfig,
    verbose: bool = False
) -> list[SubjectResult]:
    """
    Run csttool pipeline for each subject/session sequentially.
    
    Process isolation:
    - Each subject runs in a separate multiprocessing.Process
    - Parent enforces timeout via Process.join(timeout=...) + terminate()
    - Results returned via multiprocessing.Queue
    - Ensures clean timeout behavior and prevents memory accumulation
    
    Resume behavior:
    - Checks _done.json in subject output directory
    - Verifies config_hash matches current config
    - Skips subjects with valid _done.json unless config.force=True
    - Re-attempts subjects with status="failed"
    
    Signal handling:
    - SIGINT/SIGTERM: Wait for current child to complete, write summary, exit
    - Sets batch status to "interrupted" in summary
    """

def run_subject_with_timeout(
    subject: SubjectSpec,
    config: BatchConfig,
    log_path: Path
) -> SubjectResult:
    """
    Run single subject in isolated child process with timeout.
    
    Spawns multiprocessing.Process, enforces timeout, handles cleanup.
    
    Args:
        subject: Subject specification
        config: Batch configuration (includes timeout_minutes)
        log_path: Path where child process should write logs
    
    Returns:
        SubjectResult with status, duration, outputs or error
    
    Timeout behavior:
        - Process.join(timeout=config.timeout_minutes * 60)
        - If still alive: terminate(), return TIMEOUT error
        - Parent handles _work/ cleanup after termination
    """

def _run_subject_worker(
    subject: SubjectSpec,
    config: BatchConfig,
    log_path: Path,
    result_queue: Queue
) -> None:
    """
    Worker function executed in child process.
    
    Sets up logging, runs pipeline, puts result in queue.
    Must be picklable (no closures or bound methods).
    """

def compute_config_hash(config: BatchConfig) -> str:
    """Compute SHA256 hash of relevant config fields for cache invalidation."""

def is_subject_completed(
    subject: SubjectSpec,
    output_dir: Path,
    expected_config_hash: str
) -> bool:
    """
    Check if subject has already been successfully processed.
    
    Returns True only if:
    - _done.json exists in subject output directory
    - config_hash in _done.json matches expected_config_hash
    """
```

---

#### [NEW] [discover.py](file:///home/alemnalo/csttool/src/csttool/batch/modules/discover.py)

```python
def discover_subjects(
    bids_dir: Path,
    subject_pattern: str = "sub-*",
    session_pattern: str = "ses-*",
    include_subjects: list[str] | None = None,
    exclude_subjects: list[str] | None = None
) -> list[SubjectSpec]:
    """
    Discover subjects/sessions from BIDS directory structure.
    
    Handles both NIfTI and DICOM inputs:
    - {bids_dir}/sub-*/ses-*/dwi/*_dwi.nii.gz (NIfTI with sessions)
    - {bids_dir}/sub-*/dwi/*_dwi.nii.gz (NIfTI without sessions)
    - {bids_dir}/sub-*/ses-*/dwi/*.dcm (DICOM with sessions)
    - {bids_dir}/sub-*/dwi/*.dcm (DICOM without sessions)
    
    Args:
        bids_dir: Root BIDS directory
        subject_pattern: Glob pattern for subject directories
        session_pattern: Glob pattern for session directories
        include_subjects: Only include these subjects (fnmatch patterns)
        exclude_subjects: Exclude these subjects (fnmatch patterns)
    
    Returns:
        List of SubjectSpec objects with discovered inputs
    """

def detect_input_type(dwi_dir: Path) -> tuple[Literal["nifti", "dicom"], Path]:
    """
    Detect whether dwi directory contains NIfTI or DICOM.
    
    Priority: NIfTI > DICOM (if both exist, prefer NIfTI)
    
    Returns:
        Tuple of (input_type, primary_input_path)
    """

def find_bval_bvec(nifti_path: Path) -> tuple[Path | None, Path | None]:
    """Find .bval and .bvec files matching NIfTI naming convention."""

def find_json_sidecar(nifti_path: Path) -> Path | None:
    """Find .json sidecar file matching NIfTI naming convention."""

def validate_subject_inputs(subjects: list[SubjectSpec]) -> list[str]:
    """
    Validate all subject inputs exist.
    
    For NIfTI: Check NIfTI, bval, bvec files exist
    For DICOM: Check directory contains .dcm files
    
    Returns:
        List of error messages (empty if all valid)
    """

def sanitize_subject_id(subject_id: str) -> str:
    """
    Sanitize subject ID to prevent path traversal attacks.
    
    Only allows: alphanumeric, hyphen, underscore.
    Raises ValueError if ID contains invalid characters.
    """

def check_single_dwi_per_session(dwi_dir: Path) -> None:
    """
    Verify directory contains exactly one DWI dataset.
    
    Raises ValueError if multiple *_dwi.nii.gz files found.
    (v1 limitation: multiple runs not supported)
    """

def detect_dicom_series(dicom_dir: Path) -> list[dict]:
    """
    Scan DICOM directory and return unique series.
    
    Reads DICOM headers to identify distinct series.
    
    Returns:
        List of dicts with:
        - SeriesInstanceUID
        - SeriesDescription
        - SeriesNumber
        - file_count
    
    Example:
        [
            {"uid": "1.2.840...", "description": "DWI_b1000", "number": 4, "file_count": 65},
            {"uid": "1.2.840...", "description": "DWI_b0", "number": 3, "file_count": 12}
        ]
    """

def validate_single_series(
    dicom_dir: Path,
    series_uid: str | None = None
) -> str:
    """
    Ensure unambiguous DICOM series selection.
    
    If series_uid provided: validate it exists in directory
    If series_uid not provided and multiple series found: raise error
    
    Error message format:
        "Multiple DICOM series found in /path/to/dicom/:
         - 1.2.840.113619.2.55.3... (DWI_b1000, 65 files)
         - 1.2.840.113619.2.55.4... (DWI_b0, 12 files)
         Specify series_uid in manifest to select one."
    
    Returns:
        Selected SeriesInstanceUID
    """
```

---

#### [NEW] [validation.py](file:///home/alemnalo/csttool/src/csttool/batch/modules/validation.py)

```python
@dataclass
class ValidationResult:
    """Result of preflight validation."""
    valid: bool
    errors: list[str]
    warnings: list[str]
    
def validate_batch_preflight(
    subjects: list[SubjectSpec],
    config: BatchConfig,
    check_environment: bool = True
) -> ValidationResult:
    """
    Comprehensive preflight validation for batch processing.
    
    Runs all validation checks and returns aggregated results.
    Used by --validate-only and --dry-run flags.
    
    Checks:
    1. Environment (if check_environment=True)
    2. Input files existence and accessibility
    3. Output directory permissions and disk space
    4. No concurrent batch lock
    5. BIDS naming conventions
    6. Subject ID sanitization
    """

def check_environment() -> tuple[bool, list[str]]:
    """
    Verify required dependencies are installed.
    
    Reuses logic from cmd_check in cli/commands/check.py.
    
    Returns:
        Tuple of (all_ok, missing_dependencies)
    """

def check_input_files(subjects: list[SubjectSpec]) -> list[str]:
    """
    Validate all subject input files exist and are accessible.
    
    For NIfTI:
    - Check .nii.gz file exists and is readable
    - Check .bval file exists
    - Check .bvec file exists
    - Verify files are not empty
    
    For DICOM:
    - Check directory exists
    - Check directory contains .dcm files
    - Verify at least one DICOM file is readable
    
    Returns:
        List of error messages (empty if all valid)
    """

def check_output_directory(output_dir: Path) -> list[str]:
    """
    Validate output directory is writable and has sufficient space.
    
    Checks:
    - Directory exists or can be created
    - Write permissions
    - Estimated disk space requirement vs available space
    - No concurrent batch.lock file
    
    Returns:
        List of error messages (empty if valid)
    """

def estimate_disk_space_required(
    subjects: list[SubjectSpec],
    config: BatchConfig
) -> float:
    """
    Estimate total disk space required in GB.
    
    Heuristic based on:
    - Number of subjects
    - Average DWI size (~2-4 GB per subject)
    - Intermediate files multiplier
    - PDF generation overhead
    
    Returns:
        Estimated GB required
    """

def check_bids_naming(subjects: list[SubjectSpec]) -> list[str]:
    """
    Validate BIDS naming conventions.
    
    Checks:
    - Subject IDs match pattern: sub-[alphanumeric_-]+
    - Session IDs match pattern: ses-[alphanumeric_-]+
    - No path traversal characters (../, ./)
    
    Returns:
        List of warning messages (non-fatal)
    """

def check_no_concurrent_batch(output_dir: Path) -> tuple[bool, str | None]:
    """
    Check for existing batch.lock file.
    
    Returns:
        Tuple of (is_locked, lock_info)
        lock_info contains PID and hostname if locked
    """


---

#### [NEW] [manifest.py](file:///home/alemnalo/csttool/src/csttool/batch/modules/manifest.py)

```python
def load_manifest(manifest_path: Path) -> tuple[list[SubjectSpec], BatchConfig]:
    """
    Load and validate JSON manifest file.
    
    Manifest supports both NIfTI and DICOM inputs:
    - "nifti": path to .nii.gz file (auto-discovers bval/bvec/json)
    - "dicom": path to directory containing .dcm files
    - "bval"/"bvec": optional explicit paths (for non-standard naming)
    """

def validate_manifest_schema(data: dict) -> list[str]:
    """Validate manifest against expected schema. Returns list of errors."""

def resolve_relative_paths(manifest_path: Path, data: dict) -> dict:
    """Convert relative paths in manifest to absolute paths."""
```

---

#### [NEW] [report.py](file:///home/alemnalo/csttool/src/csttool/batch/modules/report.py)

```python
def generate_batch_summary(
    results: list[SubjectResult],
    output_dir: Path,
    config: BatchConfig
) -> None:
    """
    Generate batch_summary.json with execution log.
    
    Used for resume behavior and debugging.
    """

def aggregate_metrics(
    results: list[SubjectResult],
    output_dir: Path
) -> None:
    """
    Generate batch_metrics.csv by aggregating individual metrics.json files.
    
    See batch_metrics.csv schema below.
    """

def create_bids_derivative_metadata(
    output_dir: Path,
    source_dataset: str | None = None,
    csttool_version: str | None = None
) -> None:
    """Create dataset_description.json for BIDS derivative."""
```

---

### batch_metrics.csv Schema

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | str | Subject identifier (e.g., "sub-001") |
| `session_id` | str | Session identifier, empty if none |
| `status` | str | "success" / "failed" / "skipped" |
| `error_category` | str | Error category if failed, empty otherwise |
| `duration_seconds` | float | Processing time for this subject |
| `cst_l_streamline_count` | int | Left CST streamline count |
| `cst_r_streamline_count` | int | Right CST streamline count |
| `cst_l_volume_cm3` | float | Left CST volume |
| `cst_r_volume_cm3` | float | Right CST volume |
| `cst_l_mean_fa` | float | Left CST mean FA |
| `cst_r_mean_fa` | float | Right CST mean FA |
| `cst_l_mean_md` | float | Left CST mean MD |
| `cst_r_mean_md` | float | Right CST mean MD |
| `laterality_index` | float | Laterality index |
| `error` | str | Error message if failed, empty otherwise |

> [!NOTE]
> Additional metrics columns may be added as they become available in the per-subject reports.
>
> **NaN Policy:** For failed/skipped subjects, numeric metric columns are empty (NaN).
> - pandas reads empty cells as `np.nan` automatically
> - Consistent with statistical conventions
> - Always populated: `subject_id`, `session_id`, `status`, `error_category`, `duration_seconds`, `error`

**CSV example:**

```csv
subject_id,session_id,status,error_category,duration_seconds,cst_l_streamline_count,cst_l_mean_fa,error
sub-001,ses-01,success,,847.3,1523,0.45,
sub-002,,failed,TIMEOUT,7200.0,,,Processing exceeded 120 minute limit
sub-003,ses-01,success,,912.1,1489,0.47,
```

---

### batch_summary.json Schema

Versioned execution log written after each subject completes:

```json
{
  "schema_version": "1.0.0",
  "csttool_version": "1.2.3",
  "config_hash": "sha256:abc123...",
  "config": {
    "denoise_method": "patch2self",
    "skip_preprocessing": true,
    "timeout_minutes": 120
  },
  "batch_status": "completed",
  "started_at": "2026-01-22T08:00:00Z",
  "completed_at": "2026-01-22T12:45:00Z",
  "total_subjects": 50,
  "completed": 48,
  "failed": 2,
  "skipped": 0,
  "results": [
    {
      "subject_id": "sub-001",
      "session_id": "ses-01",
      "status": "success",
      "duration_seconds": 847.3,
      "log_path": "logs/sub-001_ses-01_2026-01-22T08-00-00.log"
    },
    {
      "subject_id": "sub-002",
      "session_id": null,
      "status": "failed",
      "error_category": "TIMEOUT",
      "error": "Processing exceeded 120 minute limit",
      "duration_seconds": 7200.0,
      "log_path": "logs/sub-002_2026-01-22T08-15-00.log"
    }
  ]
}
```

**batch_status values:**
- `"completed"` - All subjects processed (success or fail)
- `"interrupted"` - Batch stopped via SIGINT/SIGTERM
- `"aborted"` - Batch stopped due to SYSTEM error

---

### CLI Addition

#### [NEW] [batch.py](file:///home/alemnalo/csttool/src/csttool/cli/commands/batch.py)

```python
p_batch = subparsers.add_parser("batch", help="Process multiple subjects/sessions")

# Input (mutually exclusive)
input_group = p_batch.add_mutually_exclusive_group(required=True)
input_group.add_argument("--bids-dir", type=Path, 
    help="BIDS directory to auto-discover subjects")
input_group.add_argument("--manifest", type=Path, 
    help="JSON manifest file with explicit subject list")

# Discovery options (only used with --bids-dir)
p_batch.add_argument("--subject-pattern", default="sub-*",
    help="Glob pattern for subject directories (default: sub-*)")
p_batch.add_argument("--session-pattern", default="ses-*",
    help="Glob pattern for session directories (default: ses-*)")
p_batch.add_argument("--include-subjects", nargs="+",
    help="Only process subjects matching these patterns")
p_batch.add_argument("--exclude-subjects", nargs="+",
    help="Exclude subjects matching these patterns")

# Output
p_batch.add_argument("--out", type=Path, required=True,
    help="Output directory for BIDS derivatives")

# Batch control
p_batch.add_argument("--force", action="store_true",
    help="Re-process all subjects, ignoring prior completions")
p_batch.add_argument("--dry-run", action="store_true",
    help="Validate inputs and show execution plan without processing")
p_batch.add_argument("--validate-only", action="store_true",
    help="Validate all inputs exist without processing")
p_batch.add_argument("--keep-work", action="store_true",
    help="Retain _work/ directories even on success")
p_batch.add_argument("--timeout-minutes", type=int, default=120,
    help="Per-subject timeout in minutes (default: 120)")

# Processing options (same as cmd_run)
p_batch.add_argument("--denoise-method", choices=["patch2self", "nlmeans", "none"],
    default="patch2self")

# Preprocessing flag: use mutually exclusive group to avoid bug
preproc_group = p_batch.add_mutually_exclusive_group()
preproc_group.add_argument(
    "--preprocessing", dest="preprocessing", action="store_true", default=True,
    help="Run preprocessing steps (default)")
preproc_group.add_argument(
    "--no-preprocessing", dest="preprocessing", action="store_false",
    help="Skip preprocessing steps")

p_batch.add_argument("--generate-pdf", action="store_true")
# ... other options inherited from run command
```

#### Preflight Validation Behavior

**`--validate-only`:**
- Runs `validate_batch_preflight()` with all checks
- Prints validation report to stdout
- Exits with code 0 (success) or 1 (validation failed)
- Does NOT process any subjects

**Output format:**
```
Batch Preflight Validation
===========================

Environment Check:
  ✓ All required dependencies available

Input Files (50 subjects):
  ✓ All input files exist and are readable
  ⚠️  sub-003: No JSON sidecar found (optional)

Output Directory:
  ✓ /path/to/output is writable
  ✓ Estimated 200 GB required, 500 GB available
  ✓ No concurrent batch lock detected

BIDS Naming:
  ⚠️  sub-001_session-01: Non-standard session naming (expected ses-01)

Summary: 48 errors, 2 warnings
Status: ✓ VALIDATION PASSED
```

**`--dry-run`:**
- Runs `validate_batch_preflight()`
- Prints validation report (same as `--validate-only`)
- **Additionally** prints execution plan:
  - Subjects to process (with resume status)
  - Subjects to skip (already completed)
  - Estimated total runtime
  - Estimated disk usage
- Exits without processing

**Output format:**
```
[... validation report ...]

Execution Plan
==============

To Process (48 subjects):
  sub-001/ses-01  [new]
  sub-002         [new, no session]
  sub-003/ses-01  [retry, previously failed: TIMEOUT]
  ...

To Skip (2 subjects):
  sub-010/ses-01  [completed 2026-01-21, config matches]
  sub-011/ses-01  [completed 2026-01-21, config matches]

Estimated Runtime: 16-24 hours (48 subjects × 20-30 min avg)
Estimated Disk Usage: 200 GB
```

---

## Implementation Order

1. `modules/locking.py` - Directory locking for concurrent protection
2. `modules/logging_setup.py` - Per-subject log isolation
3. `modules/discover.py` - BIDS subject/session + DICOM discovery
4. `modules/validation.py` - Preflight validation (environment, inputs, outputs)
5. `modules/manifest.py` - JSON manifest loading
6. `batch.py` - Sequential batch orchestrator with resume logic
7. `modules/report.py` - Batch summary and metrics aggregation
8. CLI integration (`cli/commands/batch.py`)
9. Unit tests in `tests/batch/`
10. Integration test with multi-subject dataset

---

## Future: Parallel Processing (v2)

The process isolation architecture in v1 provides a natural foundation for parallelism.

### v2 Implementation

```python
from concurrent.futures import ProcessPoolExecutor

def run_batch_parallel(subjects, config, max_workers: int = 4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_subject_with_timeout, s, config, log_path): s
            for s in subjects
        }
        for future in as_completed(futures):
            result = future.result()
            update_batch_summary(result)
```

### Considerations

- `--jobs N` CLI argument for parallelism control
- Memory usage estimation: ~8-16 GB per worker for DWI processing
- Suggested default: `min(4, cpu_count() // 4)` to avoid memory exhaustion
- Per-subject locking already implemented (reused from v1)
- Logging already isolated per subject (reused from v1)


# Batch Processing Walkthrough

The `batch` module provides robust, parallel processing capabilities for large datasets, allowing `csttool` to handle hundreds of subjects efficiently.

## Core Capability: `csttool batch`

This command orchestrates the execution of the full pipeline (or subsets of it) across multiple subjects, handling scheduling, logging, and error recovery.

### Key Features

- **Parallel Execution**: Processes subjects concurrently using `multiprocessing`.
- **Fault Tolerance**: Each subject runs in an isolated process; crashes do not stop the entire batch.
- **State Management**: Tracks completed subjects to allow resuming interrupted runs (`_done.json` markers).
- **Auto-Discovery**: Can automatically find subjects in BIDS-like directory structures.
- **Flexible Input**: Supports both structured manifests and directory scanning.

### Usage

#### 1. Manifest Mode (Recommended for reproducibility)

Use a JSON manifest to explicitly define subjects and their files.

```bash
csttool batch \
    --manifest study_manifest.json \
    --out /path/to/output \
    --preprocessing \
    --denoise_method patch2self
```

**Manifest Example (`study_manifest.json`):**
```json
{
  "global_options": {
    "preprocessing": true
  },
  "subjects": [
    {
      "id": "sub-001",
      "nifti": "/raw/sub-001/dwi.nii.gz"
    },
    {
      "id": "sub-002",
      "dicom": "/raw/sub-002/dicoms/"
    }
  ]
}
```

#### 2. Auto-Discovery Mode

Automatically find subjects in a directory.

```bash
csttool batch \
    --bids-dir /path/to/raw_data \
    --out /path/to/output \
    --include "sub-0*" \
    --exclude "sub-099"
```

### Internal Architecture

1.  **Orchestrator (`run_batch`)**:
    -   Iterates through `SubjectSpec` objects.
    -   Checks for completion (skips if done).
    -   Acquires a file lock on the subject's output directory.
    -   Spawns a worker process.

2.  **Worker (`_run_subject_worker`)**:
    -   Sets up isolated file logging (`logs/sub-XXX.log`).
    -   Constructs an argument namespace mimicking `csttool run` CLI arguments.
    -   Executes the pipeline.
    -   Reports success/failure back to the parent via a queue.

3.  **Output Promotion**:
    -   Writes to a temporary `_work` directory.
    -   On success, atomically moves `_work` content to the final subject folder and writes `_done.json`.

### Output Structure

```
output_dir/
├── batch_metrics.csv       # Aggregate metrics for all subjects
├── batch_report.html       # Visual summary
├── sub-001/
│   ├── _done.json          # Completion marker
│   ├── logs/               # Execution logs
│   ├── dti_FA.nii.gz       # Pipeline outputs...
│   └── ...
├── sub-002/
└── ...
```

## Example Output

```text
Starting batch processing for 50 subjects
Output directory: /data/study_results
Config hash: a1b2c3d4

1. sub-001: Success in 15.2m
2. sub-002: Success in 14.8m
3. sub-003: Failed (TIMEOUT) - exceeded 120m limit
4. sub-004: Success in 15.5m
...

BATCH PROCESSING COMPLETE
========================================
Total:      50
Success:    48
Failed:     1 (sub-003)
Skipped:    0

Main metrics: /data/study_results/batch_metrics.csv
Batch Report: /data/study_results/batch_report.html
```

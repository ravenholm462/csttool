# Batch Processing

Run `csttool` over a cohort of subjects with a single command. The full per-command reference is in [`batch`](../reference/cli/batch.md); this page walks through the typical flow end-to-end.

## Two input modes

`csttool batch` accepts either a BIDS dataset or a JSON manifest.

=== "BIDS"

    ```bash
    csttool batch \
        --bids-dir /data/bids \
        --out ./derivatives \
        --bids-out \
        --generate-pdf
    ```

    All subjects with a `dwi/` directory are discovered automatically.

=== "Manifest"

    Create `cohort.json`:

    ```json
    {
      "subjects": [
        { "subject_id": "sub-001", "nifti": "/raw/sub-001/dwi.nii.gz" },
        { "subject_id": "sub-002", "nifti": "/raw/sub-002/dwi.nii.gz" },
        { "subject_id": "sub-003", "dicom": "/raw/sub-003/dicom" }
      ]
    }
    ```

    Then:

    ```bash
    csttool batch --manifest cohort.json --out ./derivatives --generate-pdf
    ```

## Step 1 — Dry-run first

Long batches deserve a sanity check. `--dry-run` lists planned subjects and inputs without touching disk:

```bash
csttool batch --bids-dir /data/bids --out ./derivatives --dry-run
```

Use `--validate-only` to additionally exit non-zero if any manifest entry or BIDS layout fails validation.

## Step 2 — Launch

When you are happy with the plan, drop `--dry-run`:

```bash
csttool batch --bids-dir /data/bids --out ./derivatives --bids-out --generate-pdf
```

Subjects are processed sequentially. Each subject prints the same `CHECK → IMPORT → PREPROCESS → TRACK → EXTRACT → METRICS` banners that you see in single-subject runs.

Use `--timeout-minutes 120` so a stuck subject does not block the entire cohort.

## Step 3 — Aggregate metrics

Each subject writes a `summary.csv` row under its metrics directory. Concatenate them:

```bash
find ./derivatives -name 'summary.csv' \
    | xargs -I{} sh -c 'tail -n +2 {}' \
    | (head -n 1 < "$(find ./derivatives -name 'summary.csv' | head -1)"; cat) \
    > ./derivatives/cohort_metrics.csv
```

For larger cohorts, prefer Python:

```python
import pandas as pd, glob
df = pd.concat(pd.read_csv(p) for p in glob.glob("derivatives/**/summary.csv", recursive=True))
df.to_csv("derivatives/cohort_metrics.csv", index=False)
```

## Resuming and re-running

`batch` skips subjects whose output directories already exist. To force re-processing:

```bash
csttool batch --bids-dir /data/bids --out ./derivatives --include sub-007 --force
```

## Parallelising across machines

`batch` runs subjects sequentially within a single process. For real parallelism, partition the cohort with `--include` and launch one `csttool batch` per host or scheduler slot. On SLURM:

```bash
sbatch --array=0-9 run_subjects.sh
```

with `run_subjects.sh` selecting its slice of subject IDs from `$SLURM_ARRAY_TASK_ID`.

## Related

- [Multiple subjects how-to](../how-to/multiple-subjects.md) — task-recipe focus.
- [`batch` CLI reference](../reference/cli/batch.md).
- [`run` CLI reference](../reference/cli/run.md) — single-subject pipeline that `batch` invokes per subject.

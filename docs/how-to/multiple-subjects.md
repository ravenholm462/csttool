# Process Multiple Subjects

Use [`csttool batch`](../reference/cli/batch.md) to run the full pipeline across many subjects. Two input modes are supported.

## Option A: BIDS auto-discovery

Point `--bids-dir` at a BIDS root and `csttool` will enumerate every `sub-*` (and optional `ses-*`) with a `dwi/` directory.

```bash
csttool batch \
    --bids-dir /data/bids \
    --out ./derivatives \
    --bids-out \
    --generate-pdf
```

The `--bids-out` flag writes outputs in BIDS-Derivatives layout so they can be consumed by downstream BIDS apps.

### Filtering subjects

```bash
csttool batch --bids-dir /data/bids --out ./derivatives \
    --include sub-001 sub-002 sub-003
```

```bash
csttool batch --bids-dir /data/bids --out ./derivatives \
    --exclude sub-pilot01 sub-pilot02
```

## Option B: JSON manifest

When inputs are not BIDS, describe each subject explicitly.

```json
{
  "subjects": [
    {
      "subject_id": "sub-001",
      "dicom": "/raw/sub-001/dicom",
      "session_id": "ses-01"
    },
    {
      "subject_id": "sub-002",
      "nifti": "/raw/sub-002/dwi.nii.gz",
      "bval":  "/raw/sub-002/dwi.bval",
      "bvec":  "/raw/sub-002/dwi.bvec"
    }
  ]
}
```

```bash
csttool batch --manifest cohort.json --out ./derivatives --generate-pdf
```

## Dry-run before launching

Always preview a long run first.

```bash
csttool batch --bids-dir /data/bids --out ./derivatives --dry-run
```

`--dry-run` lists the planned actions without touching disk. `--validate-only` goes a step further and exits non-zero on any manifest or BIDS discovery error.

## Re-running and resuming

`batch` skips subjects whose outputs already exist. To force re-processing of a specific subject, delete its output folder or pass `--force`.

## Performance tips

!!! tip "Parallelism"
    `batch` runs subjects sequentially within a single process. For real parallelism, partition subjects across hosts with `--include` and orchestrate at the shell or scheduler level (e.g. SLURM array jobs).

!!! tip "Time-box hung subjects"
    Pass `--timeout-minutes 120` so a stuck registration or denoising step does not block the whole cohort.

## Related

- [`batch` CLI reference](../reference/cli/batch.md)
- [Data formats](data-formats.md)
- [Troubleshooting](troubleshooting.md)

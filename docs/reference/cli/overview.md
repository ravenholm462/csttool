# CLI Overview

`csttool` is invoked as a single binary with a family of subcommands. Each subcommand corresponds to one stage of the corticospinal-tract analysis pipeline (or to orchestration of all stages). The entry point is defined in `src/csttool/cli/__init__.py` and dispatched via `argparse`.

```bash
csttool <command> [options]
csttool <command> --help
```

## Commands at a glance

The commands fall into four functional groups.

### Environment & data preparation

| Command | Purpose |
|---|---|
| [`check`](check.md) | Verify the Python environment and required dependencies. |
| [`check-dataset`](check_dataset.md) | Assess DWI acquisition quality and recommend processing options. |
| [`fetch-data`](fetch-data.md) | Download FSL-licensed atlases (FMRIB58_FA, Harvard-Oxford). |
| [`import`](import.md) | Convert DICOM to NIfTI or validate an existing NIfTI dataset. |

### Processing stages

| Command | Purpose |
|---|---|
| [`preprocess`](preprocess.md) | Denoise, unring, motion-correct and brain-mask the DWI series. |
| [`track`](track.md) | Whole-brain deterministic tractography with the CSA-ODF model. |
| [`extract`](extract.md) | Atlas-based ROI filtering to isolate the bilateral CST. |
| [`metrics`](metrics.md) | Compute scalar metrics, tract profiles and a PDF/HTML report. |
| [`validate`](validate.md) | Compare extracted bundles against reference tractograms. |

### Orchestration

| Command | Purpose |
|---|---|
| [`run`](run.md) | Execute the full pipeline (check → import → preprocess → track → extract → metrics) for one subject. |
| [`batch`](batch.md) | Run the pipeline over many subjects via manifest or BIDS auto-discovery. |

## Typical workflows

**Single subject, end-to-end**

```bash
csttool fetch-data --accept-fsl-license
csttool run --dicom /data/raw/sub-001 --out ./derivatives \
    --denoise-method patch2self --perform-motion-correction --generate-pdf
```

**Stage-by-stage (interactive debugging)**

```bash
csttool import     --dicom /data/raw/sub-001 --out ./work --subject-id sub-001
csttool preprocess --nifti ./work/sub-001/dwi/sub-001_dwi.nii.gz --out ./work/sub-001/preproc
csttool track      --nifti ./work/sub-001/preproc/dti_preproc.nii.gz --out ./work/sub-001/tracking
csttool extract    --tractogram ./work/sub-001/tracking/whole_brain.trk \
                   --fa ./work/sub-001/tracking/dti_FA.nii.gz \
                   --out ./work/sub-001/extract
csttool metrics    --cst-left ./work/sub-001/extract/cst_left.trk \
                   --cst-right ./work/sub-001/extract/cst_right.trk \
                   --fa ./work/sub-001/tracking/dti_FA.nii.gz \
                   --out ./work/sub-001/metrics --generate-pdf
```

**Batch over a BIDS dataset**

```bash
csttool batch --bids-dir /data/bids --out ./derivatives --bids-out --generate-pdf
```

## Global conventions

- **Paths**: every `--*` option that names a file accepts an absolute or relative path; output directories are created on demand.
- **Verbosity**: most commands accept `--verbose` and `--quiet` (the latter overrides).
- **Determinism**: tracking is seeded with `--rng-seed` (default `42`); pass `--random` to disable seeding.
- **BIDS output**: `run` and `batch` accept `--bids-out` to emit a BIDS-Derivatives layout.

See [Parameters](../parameters.md) for a single-page reference of every flag.

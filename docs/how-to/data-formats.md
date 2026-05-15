# Work With Different Data Formats

`csttool` accepts two input types: DICOM directories and 4D NIfTI files with separate gradient sidecars. This page covers conversion, naming conventions, and the FOV check that catches most "extraction failed" errors before they happen.

## DICOM → NIfTI

`csttool import` wraps `dcm2niix` and produces a BIDS-style output.

```bash
csttool import --dicom /raw/sub-001/dicom --out ./work --subject-id sub-001
```

If the DICOM directory contains multiple diffusion series, specify one:

```bash
csttool import --dicom /raw/sub-001/dicom --out ./work --subject-id sub-001 --series 4
```

Use `--scan-only` to see what would be converted without writing anything:

```bash
csttool import --dicom /raw/sub-001/dicom --out ./work --scan-only
```

## Gradient sidecar naming

`csttool` auto-discovers `.bval`/`.bvec` files that share the stem of the DWI NIfTI. Both BIDS-style and dcm2niix-style names are recognised:

| DWI file | Accepted gradient files |
|---|---|
| `sub-001_dwi.nii.gz` | `sub-001_dwi.bval` + `sub-001_dwi.bvec` |
| `sub-001_dwi.nii.gz` | `sub-001_dwi.bvals` + `sub-001_dwi.bvecs` |

If your sidecars live elsewhere, pass them explicitly to `check-dataset` (the other commands derive everything from `import`'s output):

```bash
csttool check-dataset --dwi raw_dwi.nii.gz --bval custom.bval --bvec custom.bvec
```

## FOV and resolution

Atlas-based CST extraction registers a template (FMRIB58_FA) into your subject's space, then projects motor-cortex and brainstem ROIs. If your acquisition does not include both endpoints in the field of view, extraction will succeed but produce empty or truncated bundles.

!!! warning "Whole-brain FOV is required"
    The DWI volume must cover from the **vertex** down to the **caudal medulla** (foramen magnum). Acquisitions clipped at the supratentorial level will silently produce bilateral CSTs that stop at the level of the cropping plane.

Use `csttool check-dataset --dwi raw_dwi.nii.gz` for a coverage diagnostic before launching the pipeline. See the [`check-dataset` reference](../reference/cli/check_dataset.md) for the full report.

## Voxel size

Default tracking parameters assume voxels in the 1.5–2.5 mm isotropic range. If your data is highly anisotropic, reslice during preprocessing:

```bash
csttool preprocess --nifti raw.nii.gz --out ./preproc --target-voxel-size 2 2 2
```

## Related

- [`import` reference](../reference/cli/import.md)
- [`check-dataset` reference](../reference/cli/check_dataset.md)
- [Data requirements](../getting-started/data-requirements.md)

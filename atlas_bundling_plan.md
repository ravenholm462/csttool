# Plan: Bundle Atlases/Templates for csttool (License-Aware)

## Context

csttool depends on nilearn to fetch MNI152 and Harvard-Oxford atlas files at runtime (network download to `~/.nilearn/data/`). The FMRIB58_FA template sits at the project root in `templates/fmrib58_fa/`, accessed via a fragile `Path(__file__).parents[4]` hack that breaks in installed packages. This change makes atlas/template data locally available while respecting distribution licenses.

## License Audit

| Asset | License | Source | Bundleable in wheel? |
|-------|---------|--------|---------------------|
| MNI152 T1 1mm (ICBM 2009) | Permissive BSD-like (free use/copy/redistribute, include copyright) | McGill/MNI | **Yes** |
| FMRIB58_FA 1mm | FSL non-commercial | Oxford/FMRIB | **No** |
| FMRIB58_FA skeleton 1mm | FSL non-commercial | Oxford/FMRIB | **No** |
| Harvard-Oxford cortical | FSL non-commercial | Oxford/FMRIB | **No** |
| Harvard-Oxford subcortical | FSL non-commercial | Oxford/FMRIB | **No** |

Sources:
- MNI152: https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/
- FSL license: https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html
- Harvard-Oxford: https://nilearn.github.io/dev/modules/description/harvard_oxford.html

## Architecture: Two-Tier Data Strategy

### Tier 1: Bundled in package (permissive license)

- MNI152 T1 1mm template -> `src/csttool/data/mni152/`
- Ships with the wheel, always available, no network needed
- Accessed via `importlib.resources.files()` + `as_file()` (Python 3.10+)

### Tier 2: User-fetched with license acknowledgment (FSL non-commercial)

- FMRIB58_FA templates -> `<platformdirs.user_data_dir>/csttool/fmrib58_fa/`
- Harvard-Oxford atlases -> `<platformdirs.user_data_dir>/csttool/harvard_oxford/`
- Downloaded via `csttool fetch-data --accept-fsl-license` from FSL GitLab (pinned tags)
- Pipeline raises `DataNotInstalledError` with actionable instructions if data missing
- No silent fallback. No nilearn intermediary at runtime.

## Directory Structure

### In-package (bundled, committed to git):

```text
src/csttool/data/
    __init__.py
    loader.py                           # Unified path resolution for both tiers
    manifest.py                         # SHA256 checksums, pinned source URLs, versions
    LICENSES/
        MNI152_LICENSE.txt              # ICBM copyright notice (verbatim)
        FSL_LICENSE.txt                 # FSL non-commercial terms (for display during fetch)
    mni152/
        MNI152_T1_1mm.nii.gz           # ~900 KB, permissive license
```

### User data directory (fetched on demand, cross-platform via `platformdirs`):

```text
<user_data_dir>/csttool/               # Linux: ~/.local/share/csttool
    fmrib58_fa/                         # macOS: ~/Library/Application Support/csttool
        FMRIB58_FA_1mm.nii.gz          # Windows: C:\Users\X\AppData\Local\csttool
        FMRIB58_FA-skeleton_1mm.nii.gz
    harvard_oxford/
        HarvardOxford-cort-maxprob-thr25-1mm.nii.gz
        HarvardOxford-sub-maxprob-thr25-1mm.nii.gz
        HarvardOxford-cort-maxprob-thr25-2mm.nii.gz
        HarvardOxford-sub-maxprob-thr25-2mm.nii.gz
    .metadata.json                      # Provenance record
    .validated                          # Stamp file: checksum verified (size + mtime)
```

## Pinned Source URLs (Verified)

All Tier 2 downloads pinned to specific FSL GitLab tags (not `master`):

```text
# data_standard tag: 2208.0
FMRIB58_FA_1mm:
  https://git.fmrib.ox.ac.uk/fsl/data_standard/-/raw/2208.0/FMRIB58_FA_1mm.nii.gz
FMRIB58_FA-skeleton_1mm:
  https://git.fmrib.ox.ac.uk/fsl/data_standard/-/raw/2208.0/FMRIB58_FA-skeleton_1mm.nii.gz

# data_atlases tag: 2103.0
HarvardOxford-cort-maxprob-thr25-1mm:
  https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/2103.0/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz
HarvardOxford-sub-maxprob-thr25-1mm:
  https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/2103.0/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz
HarvardOxford-cort-maxprob-thr25-2mm:
  https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/2103.0/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz
HarvardOxford-sub-maxprob-thr25-2mm:
  https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/2103.0/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz
```

URL path for `HarvardOxford-cort-maxprob-thr25-1mm.nii.gz` at tag `2103.0` confirmed to return binary NIfTI data (verified via raw fetch).

## Checksum Manifest: `src/csttool/data/manifest.py`

```python
DATA_MANIFEST = {
    # Tier 1: bundled
    "mni152/MNI152_T1_1mm.nii.gz": {
        "sha256": "<computed at commit time>",
        "source_url": "https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/",
        "license": "BSD-like (ICBM)",
        "version": "ICBM 2009c Nonlinear Symmetric",
        "fsl_tag": None,
    },
    # Tier 2: user-fetched, pinned to FSL GitLab tags
    "fmrib58_fa/FMRIB58_FA_1mm.nii.gz": {
        "sha256": "<computed from current file>",
        "source_url": "https://git.fmrib.ox.ac.uk/fsl/data_standard/-/raw/2208.0/FMRIB58_FA_1mm.nii.gz",
        "license": "FSL non-commercial",
        "version": "FSL data_standard",
        "fsl_tag": "2208.0",
    },
    # ... one entry per file, 7 total
}
```

Utility functions:
- `verify_checksum(path, expected_sha256) -> bool`
- `get_manifest_entry(relative_key) -> dict`

## Checksum Verification Strategy

- **At fetch time**: Always verify SHA256 immediately after download. Reject on mismatch.
- **At first use**: On first pipeline load after fetch, verify SHA256 once. Write `.validated` stamp file containing `{filename: {size, mtime, sha256_ok}}`.
- **On subsequent loads**: Check file size + mtime against `.validated` stamp. If unchanged, skip re-hashing. If file was modified, re-verify SHA256.
- **Never on every load**: Avoids checksumming multi-MB files on every pipeline invocation.

## Provenance Metadata: `.metadata.json`

Written by `csttool fetch-data` after successful download:

```json
{
    "fetched_at": "2026-02-12T14:30:00Z",
    "csttool_version": "0.4.0",
    "fsl_data_standard_tag": "2208.0",
    "fsl_data_atlases_tag": "2103.0",
    "files": {
        "fmrib58_fa/FMRIB58_FA_1mm.nii.gz": {
            "source_url": "https://git.fmrib.ox.ac.uk/fsl/data_standard/-/raw/2208.0/FMRIB58_FA_1mm.nii.gz",
            "sha256": "abc123...",
            "sha256_verified": true,
            "size_bytes": 2621440
        }
    },
    "license_accepted": "FSL non-commercial",
    "license_accepted_at": "2026-02-12T14:29:55Z"
}
```

## Implementation Steps

### Step 1: Add `platformdirs` dependency

**File:** `pyproject.toml`

Add `"platformdirs"` to `dependencies` list. Zero transitive deps, handles Linux/macOS/Windows user data paths.

### Step 2: Create in-package data directory and populate

- `mkdir -p src/csttool/data/{mni152,LICENSES}`
- Extract MNI152 template: one-time Python to save from nilearn to `src/csttool/data/mni152/MNI152_T1_1mm.nii.gz`
- Create `src/csttool/data/LICENSES/MNI152_LICENSE.txt` - verbatim ICBM copyright:
  > Copyright (C) 1993-2009 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University. Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby granted, provided that the above copyright notice appear in all copies.
- Create `src/csttool/data/LICENSES/FSL_LICENSE.txt` - FSL non-commercial terms from https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html
- Compute SHA256 of MNI152 file for the manifest

### Step 3: Create `src/csttool/data/manifest.py`

Hardcoded dict of all 7 data files with: `sha256`, `source_url` (pinned to tag), `license`, `version`, `fsl_tag`.

Utility: `verify_checksum(path, expected_sha256) -> bool`.

### Step 4: Create `src/csttool/data/__init__.py`

Minimal package marker that re-exports public API from `loader.py`.

### Step 5: Create `src/csttool/data/loader.py`

**Tier 1 access** via `importlib.resources`:

```python
from importlib.resources import files, as_file

def load_mni152_template():
    """Load bundled MNI152 T1 1mm. Always available."""
    ref = files("csttool.data").joinpath("mni152", "MNI152_T1_1mm.nii.gz")
    with as_file(ref) as path:
        img = nib.load(str(path))
        data = img.get_fdata().copy()   # Eager load, independent of file handle
        affine = img.affine.copy()
    return nib.Nifti1Image(data, affine), data, affine
```

**Tier 2 access** via `platformdirs`:

```python
from platformdirs import user_data_dir

_USER_DATA_DIR = Path(user_data_dir("csttool", ensure_exists=False))

def get_fmrib58_fa_path() -> Path:
    path = _USER_DATA_DIR / "fmrib58_fa" / "FMRIB58_FA_1mm.nii.gz"
    if not path.exists():
        raise DataNotInstalledError(
            "FMRIB58_FA template not found.\n"
            "Run 'csttool fetch-data --accept-fsl-license' to download "
            "FSL-licensed atlas data (non-commercial use only)."
        )
    _validate_if_needed(path, "fmrib58_fa/FMRIB58_FA_1mm.nii.gz")
    return path
```

**Validation stamp logic** (`_validate_if_needed`):
- Check `.validated` stamp for matching size + mtime
- If stale or missing: compute SHA256, verify against manifest, update stamp
- If mismatch: raise error

Functions:
- `load_mni152_template()` -> `(img, data, affine)` via `importlib.resources` + `as_file()`
- `get_fmrib58_fa_path()` -> Tier 2 `Path`; raises `DataNotInstalledError` if missing
- `get_harvard_oxford_path(atlas_name)` -> Tier 2 `Path`; same error behavior
- `get_user_data_dir()` -> Tier 2 root `Path` (cross-platform)
- `is_data_installed()` -> `bool`, checks all Tier 2 files present

Custom exception `DataNotInstalledError(FileNotFoundError)`.

### Step 6: Create `csttool fetch-data` CLI command

**New file:** `src/csttool/cli/commands/fetch_data.py`

Downloads directly from FSL GitLab at pinned tags via `urllib.request` (stdlib):

Command behavior:
1. Display FSL license summary prominently, including the broad definition of "commercial use"
2. **Require `--accept-fsl-license`** flag (no shorthand). Interactive y/n prompt if flag omitted.
3. Download each file from pinned tag URLs (see "Pinned Source URLs" section above)
4. Verify SHA256 against `manifest.py` - **reject and delete on mismatch**
5. Write `.metadata.json` with full provenance (csttool version, tag refs, timestamps, checksums)
6. Write `.validated` stamp
7. Report results: files fetched, sizes, checksums, storage location

Register in CLI via `subparsers.add_parser("fetch-data", ...)` in `cli/__init__.py`.

### Step 7: Update `registration.py`

**File:** `src/csttool/extract/modules/registration.py`

- **`load_mni_template()`** (lines 40-55): Replace `datasets.load_mni152_template()` with `from csttool.data.loader import load_mni152_template`
- **`load_fmrib58_fa_template()`** (lines 58-141): Remove entire cache/parents[4] logic (lines 67-103). Replace with `from csttool.data.loader import get_fmrib58_fa_path`. Keep resampling logic (lines 120-141) unchanged.
- Remove `from nilearn import datasets` (line 34) and `import shutil` (line 14)

### Step 8: Update `warp_atlas_to_subject.py`

**File:** `src/csttool/extract/modules/warp_atlas_to_subject.py`

- **`fetch_harvard_oxford()`** (lines 92-170): Replace `datasets.fetch_atlas_harvard_oxford()` with `get_harvard_oxford_path()` + `nib.load()`. No fallback - raise `DataNotInstalledError` if missing.
- The `cortical_labels` / `subcortical_labels` return keys are **never consumed** downstream (verified via grep). Drop them from the return dict.

### Step 9: Update `diagnostic.py`

**File:** `src/csttool/extract/modules/diagnostic.py`

Replace nilearn calls with `get_harvard_oxford_path()` + `nib.load()`. Print unique integer labels from the NIfTI data.

### Step 10: Handle existing `templates/` directory

- Delete `templates/` directory from project root
- FMRIB58 files are no longer shipped in the repo; they come via `csttool fetch-data`

### Step 11: Update tests

**File:** `tests/extract/test_registration.py`
- `test_load_mni_template()`: Patch target changes from `registration.datasets.load_mni152_template` to the new `csttool.data.loader` import.

**New file:** `tests/test_data_loader.py`
- Verify MNI152 bundled file exists and checksum matches manifest
- Test `DataNotInstalledError` raised when Tier 2 files missing
- Test `get_harvard_oxford_path()` validates atlas names
- Test `verify_checksum()` utility
- Test validation stamp logic (stale mtime triggers re-check)

### Step 12: Verify

- `pytest tests/` - all tests pass
- `hatch build && unzip -l dist/*.whl | grep nii` - only MNI152 in wheel
- `csttool fetch-data --accept-fsl-license` - downloads Tier 2 data, checksums pass
- `.metadata.json` written with correct provenance including pinned tags
- `.validated` stamp written
- Full pipeline run end-to-end with fetched data

## Critical Files

| File | Action |
|------|--------|
| `pyproject.toml` | **Modify** - add `platformdirs` dependency |
| `src/csttool/data/__init__.py` | **Create** - package marker + re-exports |
| `src/csttool/data/loader.py` | **Create** - two-tier path resolution (`importlib.resources` + `platformdirs`) |
| `src/csttool/data/manifest.py` | **Create** - SHA256 checksums, pinned URLs, versions |
| `src/csttool/data/LICENSES/MNI152_LICENSE.txt` | **Create** - verbatim ICBM copyright |
| `src/csttool/data/LICENSES/FSL_LICENSE.txt` | **Create** - FSL terms for display during fetch |
| `src/csttool/data/mni152/MNI152_T1_1mm.nii.gz` | **Create** - extracted from nilearn, committed |
| `src/csttool/cli/commands/fetch_data.py` | **Create** - `csttool fetch-data --accept-fsl-license` |
| `src/csttool/extract/modules/registration.py` | **Modify** - use loader for MNI152 + FMRIB58 |
| `src/csttool/extract/modules/warp_atlas_to_subject.py` | **Modify** - use loader for Harvard-Oxford |
| `src/csttool/extract/modules/diagnostic.py` | **Modify** - use loader |
| `src/csttool/cli/__init__.py` | **Modify** - register fetch-data subcommand |
| `tests/extract/test_registration.py` | **Modify** - update mock targets |
| `tests/test_data_loader.py` | **Create** - loader + manifest + validation tests |
| `templates/` | **Delete** - replaced by two-tier system |

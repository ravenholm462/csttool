# Gradient File Extension Bug Fix

**Date**: 2026-01-11  
**Issue**: Pipeline failed to load datasets with `.bvals` and `.bvecs` extensions (plural)  
**Status**: ✅ **FIXED**

---

## Problem

Some public datasets (e.g., from Brainlife.io) use `.bvals` and `.bvecs` as gradient file extensions instead of the standard `.bval` and `.bvec`. This caused the pipeline to fail at multiple stages:

1. **Preprocessing**: `load_dataset()` couldn't find gradient files
2. **Gradient file copying**: `copy_gradient_files()` couldn't locate source files  
3. **Tracking**: `get_gtab_for_preproc()` couldn't find copied gradient files

## Solution

Updated three functions to support **both** `.bval`/`.bvec` (singular) and `.bvals`/`.bvecs` (plural) extensions:

### 1. `src/csttool/cli.py`

**Added helper function:**
```python
def find_gradient_files(base_path: Path, stem: str) -> tuple[Path, Path]:
    """Find .bval/.bvec files with flexible extension support."""
    for bval_ext, bvec_ext in [('.bval', '.bvec'), ('.bvals', '.bvecs')]:
        bval = base_path / f"{stem}{bval_ext}"
        bvec = base_path / f"{stem}{bvec_ext}"
        if bval.exists() and bvec.exists():
            return bval, bvec
    raise FileNotFoundError(...)
```

**Updated `get_gtab_for_preproc()`:**
- Now uses `find_gradient_files()` helper
- Tries both extensions before raising error
- Provides clear error message listing both attempted extensions

### 2. `src/csttool/preprocess/funcs.py`

**Updated `load_dataset()`:**
```python
# Try both extensions
for bval_ext, bvec_ext in [('.bval', '.bvec'), ('.bvals', '.bvecs')]:
    test_bval = join(nifti_path, fname + bval_ext)
    test_bvec = join(nifti_path, fname + bvec_ext)
    if os.path.exists(test_bval) and os.path.exists(test_bvec):
        fbval = test_bval
        fbvec = test_bvec
        break
```

**Updated `copy_gradient_files()`:**
- Detects source files with either extension
- **Always saves as `.bval`/`.bvec`** (singular) for consistency
- This ensures downstream steps always find files in standard location

## Key Design Decision

**Normalization Strategy**: Input files can have either extension, but **output files are always normalized to `.bval`/`.bvec`** (singular). This ensures:
- Consistency across pipeline stages
- Backward compatibility with existing code
- Simpler downstream logic (only need to check one extension)

## Testing

Verified with Brainlife dataset:
```bash
csttool run --nifti /path/to/sub_sca201_ses02_dwi.nii.gz --out /output --verbose
```

**Input files:**
- `sub_sca201_ses02_dwi.bvals` ✓
- `sub_sca201_ses02_dwi.bvecs` ✓

**Output files (normalized):**
- `sub_sca201_ses02_dwi_dwi_preproc_nomc.bval` ✓
- `sub_sca201_ses02_dwi_dwi_preproc_nomc.bvec` ✓

**Result**: Pipeline completed successfully through all stages.

## Files Modified

1. `/home/alem/csttool/src/csttool/cli.py`
   - Added `find_gradient_files()` helper function
   - Updated `get_gtab_for_preproc()` to use helper

2. `/home/alem/csttool/src/csttool/preprocess/funcs.py`
   - Updated `load_dataset()` to try both extensions
   - Updated `copy_gradient_files()` to detect and normalize extensions

## Backward Compatibility

✅ **Fully backward compatible** - datasets with standard `.bval`/`.bvec` extensions continue to work exactly as before (checked first in the loop).

---

## Summary

The fix enables `csttool` to work with datasets from various sources (HCP, Brainlife, etc.) that may use different gradient file naming conventions, while maintaining internal consistency by normalizing to standard `.bval`/`.bvec` extensions.

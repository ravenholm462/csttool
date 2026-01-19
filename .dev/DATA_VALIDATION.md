# CST Tool Validation Guidance

## Overview

This document provides guidance for validating `csttool` on diffusion MRI datasets, including assessment of default parameters and identification of acquisition characteristics that may predict poor corticospinal tract (CST) extraction.

---

## 1. Default Parameter Assessment

### Summary Table

| Parameter | Default | Verdict | Notes |
|-----------|---------|---------|-------|
| `--sh-order` | 6 | ✅ Auto-validated | Automatically reduced if insufficient directions |
| `--fa-thr` | 0.2 | ✅ Reasonable | Standard in literature |
| `--seed-density` | 1 | ✅ Conservative | Adequate for clinical; research often uses 2–4 |
| `--min-length` | 20 | ⚠️ May be too permissive | CST typically 80–150 mm |
| `--max-length` | 200 | ✅ Appropriate | Covers full CST extent |
| `--step-size` | 0.5 | ⚠️ Context-dependent | Should be ~0.5× minimum voxel dimension |
| `--coil-count` | 4 | ⚠️ Often incorrect | Modern scanners use 20–64 channel coils |
| `--denoise-method` | patch2self | ✅ Good choice | Self-supervised, doesn't require coil count |

### Detailed Analysis

#### Spherical Harmonic Order (`--sh-order 6`)

The maximum SH order is constrained by the number of gradient directions:

$$l_{\max} \text{ requires } R = \frac{(l_{\max}+1)(l_{\max}+2)}{2} \text{ coefficients}$$

| SH Order | Minimum Directions Required |
|----------|----------------------------|
| 4 | 15 |
| 6 | 28 |
| 8 | 45 |

**Problem:** Many clinical protocols acquire only 20–30 directions. Using SH order 6 with insufficient directions leads to noisy, unreliable FOD estimates.

**Status**: ✅ **Implemented** in `estimate_directions.py`.

The `validate_sh_order()` function now automatically checks gradient direction count and reduces SH order if necessary, emitting a warning:

```python
# Example output when running with 20 directions and --sh-order 6:
# UserWarning: Requested SH order 6 requires ≥28 gradient directions, 
# but only 20 found. Reducing to SH order 4 for stable fitting.
```

#### Minimum Streamline Length (`--min-length 20`)

The CST spans from motor cortex through corona radiata, internal capsule, cerebral peduncle, pons, and medulla—typically 80–150 mm in adults.

A 20 mm minimum will capture many spurious short fragments, particularly in noisy data or regions with crossing fibers.

**Recommendation:** Consider increasing default to 40–50 mm, or implement length-based confidence weighting.

#### Step Size (`--step-size 0.5`)

The standard recommendation is step size ≈ 0.5× the smallest voxel dimension. For 2 mm isotropic data, 0.5 mm is appropriate. For 3 mm isotropic data, 0.5 mm may be unnecessarily fine (slower processing, no quality benefit).

**Recommendation:** Make step size adaptive:

```python
min_voxel_dim = min(nifti_header.get_zooms()[:3])
recommended_step = min_voxel_dim * 0.5
```

#### Coil Count (`--coil-count 4`)

This parameter is used by PIESNO for noise estimation. Modern scanners typically use:

- 20-channel head coils (older 3T)
- 32-channel head coils (standard 3T)
- 64-channel head coils (high-end 3T, 7T)

**Note:** If using `patch2self` (the default), PIESNO may not be invoked, making this parameter irrelevant. Verify whether this parameter affects the pipeline when `patch2self` is selected.

---

## 2. Acquisition Parameters Predicting Poor CST Extraction

### Critical Parameters

These can be extracted from BIDS JSON sidecar files or NIfTI headers:

#### Number of Gradient Directions

**Source:** Count unique non-zero vectors in `.bvec` file

| Directions | Assessment |
|------------|------------|
| < 15 | ❌ Insufficient for any SH-based method |
| 15–19 | ⚠️ Marginal; limit to SH order 4 |
| 20–27 | ⚠️ Acceptable; limit to SH order 4–6 |
| 28–44 | ✅ Good; supports SH order 6 |
| ≥ 45 | ✅ Excellent; supports SH order 8 |

**Literature:** Tournier et al. established that b-values up to 3000 s/mm² require at least 28 unique directions for SH order 6. However, these are theoretical lower bounds—practical acquisitions benefit from more directions to improve SNR.

#### b-Value

**Source:** `DiffusionBValue` in BIDS JSON, or maximum value in `.bval` file

| b-value (s/mm²) | Assessment |
|-----------------|------------|
| < 800 | ⚠️ Poor diffusion contrast; FA may be underestimated |
| 800–1500 | ✅ Optimal for single-shell DTI/CSD |
| 1500–2500 | ✅ Good for HARDI; adequate SNR |
| > 2500 | ⚠️ Significant SNR reduction; requires careful denoising |

**Literature:** Clinical DWI typically uses b = 1000 s/mm². For tractography, b = 1000–2000 s/mm² provides good balance between diffusion contrast and SNR.

#### Voxel Size

**Source:** NIfTI header (`get_zooms()`) or `SliceThickness` + `PixelSpacing` in BIDS JSON

| Voxel Size | Assessment |
|------------|------------|
| ≤ 2.0 mm isotropic | ✅ Excellent |
| 2.0–2.5 mm isotropic | ✅ Good |
| > 2.5 mm isotropic | ⚠️ Significant partial volume effects |
| Anisotropic (e.g., 2×2×4 mm) | ⚠️ May cause directional bias |

**Rationale:** The internal capsule and brainstem are compact structures where the CST is tightly packed. Large voxels increase partial volume averaging with adjacent structures, leading to inaccurate diffusion estimates and premature streamline termination.

#### Echo Time

**Source:** `EchoTime` in BIDS JSON (typically in seconds; multiply by 1000 for ms)

| Echo Time | Assessment |
|-----------|------------|
| < 80 ms | ✅ Good SNR |
| 80–100 ms | ✅ Acceptable |
| > 100 ms | ⚠️ Significant T2 decay; reduced SNR |

### Secondary Parameters

| BIDS Field | Threshold | Concern |
|------------|-----------|---------|
| `MultibandAccelerationFactor` | > 4 | Reduced SNR, potential slice aliasing |
| `ParallelReductionFactorInPlane` | > 3 | Reduced SNR from GRAPPA/SENSE acceleration |
| `PhaseEncodingDirection` | A>>P or P>>A | Worst susceptibility distortions near brainstem |
| `NumberOfAverages` | = 1 | No averaging; maximum noise |
| `PartialFourier` | < 0.75 | Reduced SNR and potential artifacts |

### CST-Specific Anatomical Challenges

The CST passes through several regions where tractography commonly fails:

1. **Corona Radiata**
   - Fanning fibers create complex multi-fiber orientations
   - Crossing with corpus callosum and superior longitudinal fasciculus

2. **Internal Capsule**
   - Highly compact fiber geometry
   - Partial volume with adjacent basal ganglia structures
   - Narrow posterior limb (~10 mm wide)

3. **Cerebral Peduncle**
   - Convergence of descending fibers
   - Crossing with pontocerebellar fibers

4. **Brainstem (Pons/Medulla)**
   - Susceptibility artifacts from air-tissue interfaces
   - Complex crossing fiber architecture
   - Compact pyramidal tract

---

## 3. Proposed Acquisition Quality Assessment

### Implementation Example

```python
from typing import List, Tuple
import numpy as np

def assess_acquisition_quality(
    bids_json: dict,
    bvecs: np.ndarray,
    bvals: np.ndarray,
    voxel_size: Tuple[float, float, float]
) -> List[Tuple[str, str]]:
    """
    Assess DWI acquisition quality for CST tractography.
    
    Returns list of (severity, message) tuples.
    Severity levels: "CRITICAL", "WARNING", "INFO"
    """
    warnings = []
    
    # Count gradient directions (exclude b=0)
    b0_threshold = 50  # s/mm²
    dwi_mask = bvals > b0_threshold
    n_directions = len(np.unique(bvecs[:, dwi_mask], axis=1))
    
    if n_directions < 15:
        warnings.append((
            "CRITICAL",
            f"Only {n_directions} gradient directions detected. "
            f"Minimum 15 required for basic tractography, 28+ recommended."
        ))
    elif n_directions < 28:
        warnings.append((
            "WARNING",
            f"{n_directions} gradient directions limits SH order to 4. "
            f"Consider acquiring ≥28 directions for SH order 6."
        ))
    
    # Check b-value
    max_bval = np.max(bvals)
    if max_bval < 800:
        warnings.append((
            "WARNING",
            f"Maximum b-value ({max_bval:.0f} s/mm²) may underestimate FA "
            f"and reduce diffusion contrast."
        ))
    elif max_bval > 3000:
        warnings.append((
            "WARNING",
            f"High b-value ({max_bval:.0f} s/mm²) may have SNR limitations. "
            f"Ensure adequate denoising."
        ))
    
    # Check voxel size
    max_voxel = max(voxel_size)
    if max_voxel > 2.5:
        warnings.append((
            "WARNING",
            f"Large voxel size ({voxel_size[0]:.1f}×{voxel_size[1]:.1f}×"
            f"{voxel_size[2]:.1f} mm) may cause partial volume effects "
            f"in internal capsule and brainstem."
        ))
    
    # Check for anisotropic voxels
    voxel_ratio = max(voxel_size) / min(voxel_size)
    if voxel_ratio > 1.5:
        warnings.append((
            "WARNING",
            f"Anisotropic voxels (ratio {voxel_ratio:.1f}) may cause "
            f"directional bias in tractography."
        ))
    
    # Check echo time (if available)
    echo_time_ms = bids_json.get("EchoTime", 0) * 1000
    if echo_time_ms > 100:
        warnings.append((
            "WARNING",
            f"Long echo time ({echo_time_ms:.0f} ms) reduces SNR due to T2 decay."
        ))
    
    # Check multiband factor (if available)
    mb_factor = bids_json.get("MultibandAccelerationFactor", 1)
    if mb_factor > 4:
        warnings.append((
            "INFO",
            f"High multiband factor ({mb_factor}) may reduce SNR. "
            f"Verify image quality."
        ))
    
    # Check parallel imaging factor (if available)
    parallel_factor = bids_json.get("ParallelReductionFactorInPlane", 1)
    if parallel_factor > 3:
        warnings.append((
            "INFO",
            f"High parallel imaging factor ({parallel_factor}) may reduce SNR."
        ))
    
    return warnings


def get_recommended_sh_order(n_directions: int) -> int:
    """Return maximum recommended SH order for given directions."""
    # Add 20% margin above theoretical minimum
    if n_directions >= 54:  # 45 * 1.2
        return 8
    elif n_directions >= 34:  # 28 * 1.2
        return 6
    elif n_directions >= 18:  # 15 * 1.2
        return 4
    else:
        return 2
```

### Quality Report Template

```
================================================================================
                        CST TOOL - ACQUISITION QUALITY REPORT
================================================================================

Subject ID: {subject_id}
Scan Date:  {scan_date}

ACQUISITION PARAMETERS
----------------------
Gradient directions:     {n_directions}
Maximum b-value:         {max_bval} s/mm²
Voxel size:              {vox_x:.2f} × {vox_y:.2f} × {vox_z:.2f} mm
Echo time:               {echo_time:.1f} ms
Multiband factor:        {mb_factor}

QUALITY ASSESSMENT
------------------
{quality_warnings}

RECOMMENDED SETTINGS
--------------------
Maximum SH order:        {recommended_sh_order}
Suggested step size:     {suggested_step_size:.2f} mm

================================================================================
```

---

## 4. Validation Datasets

### Recommended Test Data

| Dataset | Description | Use Case |
|---------|-------------|----------|
| **HCP Young Adult** | High-quality multi-shell (b=1000, 2000, 3000), 90 directions per shell, 1.25 mm isotropic | Gold standard reference |
| **HCP Test-Retest** | Same protocol, repeated scans | Reproducibility testing |
| **ISMRM 2015 Tractography Challenge** | Synthetic phantoms with ground truth | Algorithm validation |
| **TractoInferno** | Large-scale tractography benchmark | Comprehensive evaluation |

### Minimum Validation Protocol

1. **High-quality data** (HCP-like): Verify expected CST anatomy
2. **Clinical-quality data** (30 dirs, 2 mm, b=1000): Test robustness
3. **Edge cases**: Low directions (15–20), high b-value (>2500), large voxels (>2.5 mm)
4. **Pathological data**: If available, test with lesions affecting CST

---

## 5. References

1. Mukherjee P, et al. (2008). Diffusion Tensor MR Imaging and Fiber Tractography: Technical Considerations. *AJNR*, 29(5):843-852. https://www.ajnr.org/content/29/5/843

2. Tournier JD, et al. (2013). Determination of the appropriate b value and number of gradient directions for high-angular-resolution diffusion-weighted imaging. *NMR in Biomedicine*, 26(12):1775-1786.

3. Jones DK (2004). The effect of gradient sampling schemes on measures derived from diffusion tensor MRI: A Monte Carlo study. *Magnetic Resonance in Medicine*, 51(4):807-815.

4. Ni H, et al. (2006). Effects of number of diffusion gradient directions on derived diffusion tensor imaging indices in human brain. *AJNR*, 27(8):1776-1781.

5. MRtrix3 Documentation: Spherical Harmonics. https://mrtrix.readthedocs.io/en/latest/concepts/spherical_harmonics.html

6. DIPY Documentation: Constrained Spherical Deconvolution. https://dipy.org/documentation/latest/examples_built/reconst_csd/

---

## 6. Open Questions — Resolved

> **Status**: All questions investigated and resolved as of 2026-01-19.

### 1. SH Order Validation ✅ **Now Implemented**

**Finding**: csttool now validates that the specified SH order is achievable with the available gradient directions.

**Location**: [`src/csttool/tracking/modules/estimate_directions.py`](file:///home/alemnalo/csttool/src/csttool/tracking/modules/estimate_directions.py)

**Behavior**:
- `get_max_sh_order(n_directions)` returns the maximum safe SH order
- `validate_sh_order(gtab, sh_order)` checks and emits a `UserWarning` if the requested order is too high
- The order is automatically reduced to the maximum safe value

```python
# Example warning:
# UserWarning: Requested SH order 6 requires ≥28 gradient directions, 
# but only 20 found. Reducing to SH order 4 for stable fitting.
```

---

### 2. Coil Count with Patch2Self ✅ **Confirmed Irrelevant**

**Finding**: The `--coil-count` parameter is **only used with `nlmeans` denoising**. When `patch2self` is selected (the default), the coil count is completely ignored.

**Location**: [`src/csttool/preprocess/modules/denoise.py`](file:///home/alemnalo/csttool/src/csttool/preprocess/modules/denoise.py)

**Code path**:
```python
if denoise_method == "nlmeans":
    noise, _ = piesno(data, N=N, ...)  # ← N (coil_count) used
elif denoise_method == "patch2self":
    denoised_data = patch2self(data, bvals=bvals, ...)  # ← No N parameter
```

**CLI Update**: Help text now clarifies this:
> `--coil-count`: Only used with `--denoise-method nlmeans` (ignored for patch2self).

---

### 3. Extraction Method Comparison ⚠️ **No Benchmarks Yet**

**Finding**: Three extraction methods exist, but no systematic comparison has been performed.

| Method | Description | Availability |
|--------|-------------|--------------|
| `endpoint` | Filter by streamline endpoints in ROIs (strict) | `extract`, `run` |
| `passthrough` | Filter by streamlines passing through ROIs (permissive) | `extract`, `run` — **default** |
| `roi-seeded` | Seed tracking directly from ROIs (requires raw DWI) | `run` only |

**Known issues** (from TODO.md):
> "patch2self denoising may produce streamlines that are too short → extraction fails. Switching to NLMeans may help."

**Recommendation**: Future validation work should compare:
1. Streamline count and anatomical coverage per method
2. Reproducibility across test-retest data
3. Robustness to acquisition quality variations
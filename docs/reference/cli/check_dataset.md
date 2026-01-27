# Check Dataset Command

The `check-dataset` command performs comprehensive quality assessment on a diffusion MRI dataset (NIfTI + bval/bvec) without running the full pipeline. This command provides detailed diagnostics for both single-shell and multi-shell acquisitions.

## Usage

```bash
csttool check-dataset \
    --dwi data.nii.gz \
    --bval data.bval \
    --bvec data.bvec \
    --json sidecar.json \
    --b0-threshold 50.0 \
    --verbose
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dwi` | Path | Required | Path to DWI NIfTI file (.nii or .nii.gz) |
| `--bval` | Path | Optional | Path to bval file (auto-detected if not provided) |
| `--bvec` | Path | Optional | Path to bvec file (auto-detected if not provided) |
| `--json` | Path | Optional | Path to BIDS JSON sidecar for metadata validation |
| `--b0-threshold` | Float | 50.0 | B-value threshold for classifying b=0 volumes (s/mm²) |
| `--verbose` | Flag | False | Show detailed metrics and per-shell analysis |

## Quality Checks

### Input Validation
-   **Shape Validation**: Ensures bvecs have correct shape (N, 3)
-   **Count Matching**: Verifies bvals and bvecs have matching counts
-   **B-value Validation**: Detects negative or extremely high b-values (>10000 s/mm²)
-   **Gradient Quality**: Identifies near-zero gradient vectors in DWI volumes

### B=0 Volume Analysis
-   **Count Assessment**: Warns if < 3 b=0 volumes (recommends ≥3 for robust registration)
-   **Distribution Analysis**: Checks spacing between b=0 volumes for motion correction
-   **Gap Detection**: Flags large gaps (>30 volumes) between b=0 acquisitions

### Shell Detection & Analysis
-   **Automatic Shell Detection**: Uses adaptive tolerance (5%) to cluster b-values
-   **Per-Shell Direction Counting**: Validates sufficient directions per shell
-   **Multi-Shell Support**: Reports acquisition type (single vs. multi-shell)
-   **Per-Shell Quality**: Warns if shell has < 6 unique directions

### Direction & SH Order Validation
-   **Total Direction Count**: Validates minimum 15 directions for tractography
-   **SH Order Recommendations**: Suggests maximum safe SH order based on directions
-   **Per-Shell SH Orders**: (Verbose mode) Provides SH recommendations per shell

### Voxel Quality Assessment
-   **Voxel Volume**: Warns if voxel volume > 15 mm³ (SNR concerns)
-   **Voxel Size**: Checks for large dimensions > 2.5 mm (partial volume effects)
-   **Anisotropy**: WARNING at ratio > 1.5, CRITICAL at > 2.0 (directional bias risk)

### BIDS Metadata Validation
-   **Critical Fields**: Validates PhaseEncodingDirection and TotalReadoutTime
-   **Phase Encoding**: Checks for unusual phase encoding directions
-   **Acquisition Parameters**: Validates EchoTime, MultibandFactor, ParallelImaging
-   **SNR Factors**: Warns about high multiband (>4) or parallel imaging (>3) factors

## Example Outputs

### Single-Shell Dataset (Basic)

```bash
csttool check-dataset --dwi sub-01_dwi.nii.gz
```

```text
================================================================================
                        CST TOOL - ACQUISITION QUALITY REPORT
================================================================================

Subject/File: sub-01_dwi.nii.gz
Scan Date:    Unknown

B=0 VOLUMES
-----------
Count:                   3

ACQUISITION PARAMETERS
----------------------
Acquisition type:        Single-shell
B-value:                 1000 s/mm²
Gradient directions:     32
Total DWI volumes:       32
Voxel size:              2.00 x 2.00 x 2.00 mm

BIDS METADATA
-------------
✗ PhaseEncodingDirection (required for distortion correction)
✗ TotalReadoutTime (required for distortion correction)

QUALITY ASSESSMENT
------------------
⚠️  [WARNING] Missing BIDS fields for distortion correction: PhaseEncodingDirection, TotalReadoutTime

RECOMMENDED SETTINGS
--------------------
Maximum SH order:        6
Suggested step size:     1.00 mm

================================================================================
```

### Multi-Shell Dataset (Verbose)

```bash
csttool check-dataset --dwi sub-02_multishell.nii.gz --json sub-02.json --verbose
```

```text
================================================================================
                        CST TOOL - ACQUISITION QUALITY REPORT
================================================================================

Subject/File: sub-02_multishell.nii.gz
Scan Date:    2024-03-15T10:30:00

B=0 VOLUMES
-----------
Count:                   5
Maximum gap:             20 volumes
Indices:                 [0, 20, 40, 60, 80]

ACQUISITION PARAMETERS
----------------------
Acquisition type:        Multi-shell (2 shells)
  Shell 1: b=1000 s/mm² (32 directions, 32 volumes)
  Shell 2: b=2000 s/mm² (60 directions, 60 volumes)
Voxel size:              2.00 x 2.00 x 2.00 mm
Echo time:               80.0 ms
Multiband factor:        2

BIDS METADATA
-------------
✓ PhaseEncodingDirection (required for distortion correction)
✓ TotalReadoutTime (required for distortion correction)
✓ EchoTime
✓ MultibandAccelerationFactor

QUALITY ASSESSMENT
------------------
ℹ️  [INFO] Detected 2 b-value shell(s): [1000, 2000]

RECOMMENDED SETTINGS
--------------------
Maximum SH order:        8

Per-shell SH orders:
  b=1000: SH order 6
  b=2000: SH order 8

Suggested step size:     1.00 mm

DETAILED METRICS
----------------
Total volumes:           97
B=0 volumes:             5
DWI volumes:             92
Unique directions:       60
B0 threshold used:       50.0 s/mm²
Voxel volume:            8.00 mm³
Voxel anisotropy:        1.00:1

================================================================================
```

### Poor Quality Dataset

```bash
csttool check-dataset --dwi sub-03_poor.nii.gz --verbose
```

```text
================================================================================
                        CST TOOL - ACQUISITION QUALITY REPORT
================================================================================

Subject/File: sub-03_poor.nii.gz
Scan Date:    Unknown

B=0 VOLUMES
-----------
Count:                   1

ACQUISITION PARAMETERS
----------------------
Acquisition type:        Single-shell
B-value:                 700 s/mm²
Gradient directions:     10
Total DWI volumes:       10
Voxel size:              3.00 x 3.00 x 3.50 mm

BIDS METADATA
-------------
✗ PhaseEncodingDirection (required for distortion correction)
✗ TotalReadoutTime (required for distortion correction)

QUALITY ASSESSMENT
------------------
⚠️  [WARNING] Only 1 b=0 volume(s). Recommend ≥3 for robust registration
❌ [CRITICAL] Only 10 gradient directions detected. Minimum 15 required for basic tractography, 28+ recommended.
⚠️  [WARNING] Maximum b-value (700 s/mm²) may underestimate FA and reduce diffusion contrast.
⚠️  [WARNING] Large voxel size (3.0x3.0x3.5 mm) may cause partial volume effects in internal capsule and brainstem.
⚠️  [WARNING] Large voxel volume (31.5 mm³) may reduce SNR
⚠️  [WARNING] Anisotropic voxels (ratio 1.2) may cause directional bias in tractography.
⚠️  [WARNING] Missing BIDS fields for distortion correction: PhaseEncodingDirection, TotalReadoutTime

RECOMMENDED SETTINGS
--------------------
Maximum SH order:        2
Suggested step size:     1.50 mm

DETAILED METRICS
----------------
Total volumes:           11
B=0 volumes:             1
DWI volumes:             10
Unique directions:       10
B0 threshold used:       50.0 s/mm²
Voxel volume:            31.50 mm³
Voxel anisotropy:        1.17:1

================================================================================
```

## Severity Levels

Quality warnings are categorized by severity:

- **❌ CRITICAL**: Issues that will likely cause pipeline failure or severely compromised results
  - Insufficient gradient directions (< 15)
  - Mismatched bvals/bvecs counts
  - No b=0 volumes
  - Negative b-values
  - Near-zero gradients in DWI
  - Highly anisotropic voxels (ratio > 2.0)

- **⚠️ WARNING**: Issues that may degrade results or require attention
  - Low b=0 count (< 3)
  - Suboptimal direction count (15-27)
  - Extreme b-values (< 800 or > 3000 s/mm²)
  - Large voxels (> 2.5 mm or volume > 15 mm³)
  - Anisotropic voxels (ratio > 1.5)
  - Missing BIDS metadata
  - Long echo time (> 100 ms)

- **ℹ️ INFO**: Informational messages about acquisition details
  - Multi-shell detection
  - High multiband/parallel imaging factors
  - B=0 distribution patterns
  - Unusual phase encoding directions

## Use Cases

### Pre-Acquisition Protocol Validation
Verify acquisition protocol meets CST tractography requirements before scanning:
```bash
csttool check-dataset --dwi pilot_scan.nii.gz --verbose
```

### Quality Control
Check acquired data immediately after scanning:
```bash
csttool check-dataset --dwi sub-01_dwi.nii.gz --json sub-01_dwi.json
```

### Multi-Shell Optimization
Validate multi-shell protocols with custom b0 threshold:
```bash
csttool check-dataset --dwi multishell.nii.gz --b0-threshold 100 --verbose
```

### Batch Dataset Screening
Quick quality screening for large datasets:
```bash
for dwi in data/*.nii.gz; do
    echo "Checking $dwi"
    csttool check-dataset --dwi "$dwi"
done
```

## Notes

- The command auto-detects `.bval` and `.bvec` files if not explicitly provided
- Shell detection uses 5% relative tolerance for clustering b-values
- Direction counting uses 4-decimal rounding to handle floating-point precision
- SH order recommendations follow standard requirements:
  - SH order 2: 6 directions
  - SH order 4: 15 directions
  - SH order 6: 28 directions
  - SH order 8: 45 directions
- Verbose mode provides per-shell analysis for multi-shell acquisitions

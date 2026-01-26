# Check Dataset Command

The `check-dataset` command performs a quality assessment on a diffusion MRI dataset (NIfTI + bval/bvec) without running the full pipeline.

## Usage

```bash
csttool check-dataset \
    --dwi data.nii.gz \
    --bval data.bval \
    --bvec data.bvec \
    --json sidecar.json
```

## Quality Checks

-   **Dimension Verification**: Checks if image dimensions (4D) match gradient vector counts.
-   **Shell Detection**: Identifies b-value shells and counts directions per shell.
-   **CSD Suitability**: Verifies if the number of directions supports the requested Spherical Harmonic order (e.g., > 15 directions for SH order 4, > 28 for SH order 6, > 45 for SH order 8).
-   **Metadata Presence**: Checks for critical acquisition parameters like Echo Time (TE) and Field Strength.

## Example Output

```text
Dataset Quality Assessment
==========================
File: /data/dwi.nii.gz
Dimensions: (106, 106, 60, 65)

Shells:
  b=0: 1 volumes
  b=1000: 32 directions
  b=2000: 32 directions

CSD Suitability:
  Total gradient directions: 64
  Max SH order possible: 8 (requires 45 directions)
  Requested SH order: 6 (requires 28 directions) -> OK

Metadata:
  Phase Encoding: AP (derived from JSON)
  Readout Time: 0.05s
  Field Strength: 3.0T [PASS]
  TE / TR: 89ms / 7200ms
```

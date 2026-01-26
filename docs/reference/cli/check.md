# Check Command

The `check` command validates the system environment to ensure all dependencies are correctly installed and configured.

## Usage

```bash
csttool check
```

## Checks Performed

-   **Python Dependencies**: Verifies all required Python packages are installed.
-   **External Tools**: Checks for the presence of:
    -   `FSL` (FMRIB Software Library)
    -   `MRtrix3`
-   **Environment Variables**: Ensures `$FSLDIR` and other necessary variables are set.

## Example Output

```text
CSTTool Environment Check
=========================
Python version: 3.10.12
csttool version: 0.1.0

Dependencies:
  numpy: OK (1.24.3)
  nibabel: OK (5.1.0)
  dipy: OK (1.7.0)

External Tools:
  FSL: OK (/usr/local/fsl)
  MRtrix3: OK (/usr/bin/mrview)

Status: OK
```

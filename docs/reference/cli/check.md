# check Command

The `check` command validates the system environment to ensure all dependencies are correctly installed and configured.

## Usage

```bash
csttool check
```

## Checks Performed

The command automatically performs the following checks:

- **Python Version**: Displays the installed Python version
- **csttool Version**: Shows the installed csttool version
- **Python Dependencies**: Verifies all required packages from `pyproject.toml` are installed and displays their versions
- **csttool Modules**: Auto-discovers and tests that all csttool submodules can be imported:
  - `csttool.batch`
  - `csttool.extract`
  - `csttool.ingest`
  - `csttool.metrics`
  - `csttool.preprocess`
  - `csttool.validation`

## Example Output

```text
csttool environment check
Python: 3.13.11
Version: 0.3.1

✓ numpy: 1.26.4
✓ scipy: 1.16.3
✓ cython: 3.1.6
✓ dipy: 1.11.0
✓ matplotlib: 3.10.7
✓ nibabel: 5.3.2
✓ dicom2nifti: unknown
✓ nilearn: 0.12.1
✓ weasyprint: 67.0
✓ jinja2: 3.1.6

✓ csttool.batch: available
✓ csttool.extract: available
✓ csttool.ingest: available
✓ csttool.metrics: available
✓ csttool.preprocess: available
✓ csttool.validation: available

✓ All required dependencies and modules available
```

## Exit Status

- **Exit code 0**: All checks passed
- **Exit code 1**: One or more checks failed

If any required dependency is missing, the command will display:
```
✗ Some dependencies or modules missing - install with: pip install -e .
```

## See Also

- [Installation](../getting-started/installation.md) — How to install csttool and dependencies
- [Troubleshooting](../how-to/troubleshooting.md) — Common issues and solutions

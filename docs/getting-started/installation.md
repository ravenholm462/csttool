# Installation

## Requirements

- **Python 3.10+**
- **System libraries** for PDF generation (WeasyPrint)

---

## Quick Install

```bash
pip install git+https://github.com/ravenholm462/csttool.git
```

---

## System Dependencies

csttool uses [WeasyPrint](https://weasyprint.org/) for PDF report generation, which requires system-level libraries.

=== "Ubuntu/Debian"

    ```bash
    sudo apt install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev libcairo2
    ```

=== "Fedora/RHEL"

    ```bash
    sudo dnf install pango gdk-pixbuf2 cairo libffi-devel
    ```

=== "macOS"

    ```bash
    brew install pango gdk-pixbuf cairo libffi
    ```

=== "Windows"

    Install [GTK3 runtime](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer).

!!! warning "WeasyPrint installation issues"
    If you encounter errors during installation or PDF generation, see the 
    [WeasyPrint installation guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation).

---

## Development Install

For contributing or local development:

```bash
# Clone repository
git clone https://github.com/ravenholm462/csttool.git
cd csttool

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install in editable mode with test dependencies
pip install -e ".[test]"
```

---

## Verify Installation

```bash
# Check CLI is accessible
csttool --version

# Verify environment and dependencies
csttool check
```

Expected output:
```
csttool version 0.3.1
```

---

## Python Dependencies

csttool installs the following packages automatically:

| Package | Purpose |
|---------|---------|
| `dipy` | Diffusion MRI processing, tractography |
| `nibabel` | NIfTI file I/O |
| `nilearn` | Neuroimaging utilities, atlas handling |
| `dicom2nifti` | DICOM to NIfTI conversion |
| `numpy`, `scipy` | Numerical computing |
| `matplotlib` | Visualizations |
| `weasyprint`, `jinja2` | HTML/PDF report generation |

---

## Next Steps

- [Quick Start Guide](quickstart.md) — Run your first CST analysis
- [Data Requirements](data-requirements.md) — Prepare your input data

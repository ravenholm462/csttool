# csttool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://ravenholm462.github.io/csttool/)

Simple, modular CST assessment tool for diffusion MRI.

`csttool` is a Python-based command-line tool for automated assessment of the **corticospinal tract (CST)** using diffusion-weighted MRI (DW-MRI) data. It is designed to be simple, modular, and easily portable to different computing environments.

## Installation

```bash
pip install git+https://github.com/ravenholm462/csttool.git
```

Or for development:
```bash
git clone https://github.com/ravenholm462/csttool.git
cd csttool
pip install -e .
```

## Structure

v0.1.0

```
├── diagrams
├── LICENSE
├── pyproject.toml
├── README.md
├── src
│   └── csttool
│       ├── cli.py
│       ├── extract
│       ├── __init__.py
│       ├── metrics
│       ├── preprocess
│       └── tracking
├── tests
│   ├── extract
│   ├── ingest
│   ├── integration
│   ├── metrics
│   ├── preprocess
│   └── tracking
```

## Config file

The build will be defined via a `.toml` configuration file. TOML stands for Tom's Obvious, Minimal Language ([learn more](https://en.wikipedia.org/wiki/TOML)). The configuration file is needed to make the tool installable. The configuration file contains the following:

1. The build backend - using [hatchling](https://pypi.org/project/hatchling/).
2. Project metadata - name, version, author, license, dependencies
3. Console entrypoint - definition allowing the use of `csttool` from any shell.

## CLI skeleton

The skeleton consists of two files:

1. `src/csttool/__init__.py`
   - Identifies the package `csttool`
   - Contains the version number via `--version`
2. `src/csttool/cli.py`
   - Defines the main entry point for the command line interface
   - Python's built-in `argparse` module to handle commands and arguments
   - Uses subparsers to create modular commands within the main parser, each bound to its own Python function. Each command implements one workflow step (e.g. `check`, `import`, `preprocess`, `track`, `extract`, `metrics`).

## Preprocessing

The ![preprocessing pipeline](https://github.com/ravenholm462/csttool/tree/main/diagrams/png/preprocessing.png) performs the following steps:

1. Load NIfTI + bvals/bvecs and build a gradient table
2. Estimate noise and denoise with NLMEANS
3. Compute a brain mask with median Otsu on b0 volumes
4. Perform between volume motion correction (optional)
5. Save the preprocessed data, generate visualizations

## Tracking

The ![tracking pipeline](https://github.com/ravenholm462/csttool/tree/main/diagrams/png/tracking.png) performs the following steps:

1. Tensor fitting and scalar measures (FA, MD)
2. Direction field estimation with a CSA ODF model
3. FA based stopping criterion and seed generation
4. Deterministic local tracking
5. Save generated whole-brain tractogram, generate visualizations

## Extraction

The ![extraction pipeline](https://github.com/ravenholm462/csttool/tree/main/diagrams/png/extraction.png) performs the following steps:

1. Performs spatial registration of the moving image (subject) to a static image (MNI 152 template)
2. Warps Harvard-Oxford parcellation atlas to subject native space using registration mapping
3. Create ROI masks to isolate the corticospinal tract (brainsteam, motor left, motor right)
4. Filter all streamlines from the whole-brain tractogram not passing through the ROIs
5. Save extracted CST tractograms, generate visualizations

## Metrics

The ![metrics pipeline](https://github.com/ravenholm462/csttool/tree/main/diagrams/png/metrics.png) performs the following steps:

1. Performs unilateral analysis of left and right CST (morphology, FA, MD, tract profile)
2. Performs bilateral analysis of entire CST (assymetry metrics, volume laterlaity, FA/MD laterality)
3. Generates reports
4. Generates visualizations

## Testing
The `csttool` project includes a comprehensive suite of unit and integration tests.

To run the full test suite:
```bash
pytest tests/
```

To run only integration tests (end-to-end pipeline verification):
```bash
pytest tests/integration/
```

To run only unit tests:
```bash
pytest tests/ --ignore=tests/integration/
```

## Usage examples

```bash
csttool run --dicom /path/to/dicom --out /path/to/out --save-visualizations --generate-pdf
```




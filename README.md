## Purpose
`csttool` is a Python-based command-line tool for automated assessment of the **corticospinal tract (CST)** using diffusion-weighted MRI (DW-MRI) data. It is designed to be simple, modular, and easily portable to different computing environments.

## Core ideas

The core ideas behind `csttool` are:

1. Simplicity - achieved through using already existing Python libraries to create the tool. It should be installable anywhere, be easy to use, and its results should be easy to read and interpret.

2. Modularity - ability to expand the tool with more functionality as is necessary.

## Structure

v0.0.1

```
├── LICENSE
├── pyproject.toml
├── README.md
├── src
│   └── csttool
│       ├── cli.py
│       ├── __init__.py
│       ├── metrics
│       │   └── __init__.py
│       ├── preprocess
│       │   ├── funcs.py
│       │   ├── __init__.py
│       │   └── tests
│       │       └── pipeline_test.py
│       └── tracking
│           ├── funcs.py
│           ├── __init__.py
└── tests
    ├── cli_test.sh
    ├── __init__.py
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
   - Uses subparsers to create modular commands within the main parser, each bound to its own Python function. Each command implements one workflow step (e.g. `check`, `preprocess`, `track`, `metrics`).

## Preprocessing

The preprocessing pipeline performs the following steps:

1. Load NIfTI + bvals/bvecs and build a gradient table
2. Estimate noise and denoise with NLMEANS
3. Compute a brain mask with median Otsu on b0 volumes
4. Perform between volume motion correction
5. Save the preprocessed data to disk

## Tracking

The tracking pipeline performs the following steps:

1. Tensor fitting and scalar measures (FA, MD)
2. Direction field estimation with a CSA ODF model
3. FA based stopping criterion and seed generation
4. Deterministic local tracking
5. Tractogram saving helper

## Analysis

Coming soon.

## How to run

The following codeblock sets up and runs `csttool` in a Linux shell:

```
git clone git@github.com:ravenholm462/csttool.git
cd csttool
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .  # install in editable mode
csttool --help
csttool --version
csttool check
```

For Windows, replace `source .venv/bin/activate` with `.venv\Scripts\Activate.ps1`

For Debian/Ubuntu-based Linux systems, ensure that the `python3-venv` package is installed via `apt install python3.12-venv`.

## Testing

Each submodule will have its own `/test` folder. The root level `/test` folder houses the CLI `.sh` test script.

## Usage examples

### Preprocess data

```bash
csttool preprocess --nifti /data/sub01_dwi.nii.gz --out /data/output
```

### Run deterministic tracking

```
csttool track --nifti /data/output/sub01_dwi_preproc.nii.gz --out /data/output
```




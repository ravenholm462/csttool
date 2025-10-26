## Purpose
`csttool` is a Python-based command-line tool for automated assessment of the **corticospinal tract (CST)** using diffusion-weighted MRI (DW-MRI) data. It is designed to be simple, modular, and easily portable to different computing environments.

## Core ideas

The core ideas behind `csttool` are:

1. Simplicity
Simplicity is achieved through using already existing Python libraries to create the tool. It should be installable anywhere, be easy to use, and its results should be easy to read and interpret.

2. Modularity
Modularity implies the ability to expand the tool with more functionality as is necessary.

## Structure

v0.0.1

```
csttool/
├─ pyproject.toml
├─ README.md
├─ src/
│  └─ csttool/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ preprocess/          # dataset preparation
│     │  └─ __init__.py
│     ├─ tracking/            # tracking algorithms
│     │  └─ __init__.py
│     └─ metrics/             # extraction of desired metrics
└─ tests/
   └─ __init__.py
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

## How to run

The following codeblock sets up and runs `csttool` in a Linux shell:

```
git clone https://github.com/ravenholm462/csttool.git
cd csttool
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .  # install in editable mode
csttool --help
csttool --version
csttool check
```

For Windows, replace `source .venv/bin/activate` with `.venv\Scripts\Activate.ps1`

## Testing

The `/tests` directory is meant for unit and integration tests for each submodule. The goal is to ensure reproducibility and to support future deployment via e.g. Docker.





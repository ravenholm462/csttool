# CLI Overview

The `cli` module is the user-facing entry point for `csttool`. It uses `argparse` to structure commands into a suite of subtools, ranging from individual processing steps to full-pipeline orchestration.

## Entry Point: `csttool`

The main entry point is defined in `src/csttool/cli/__init__.py`. It sets up the global parser and registers subcommands.

### Available Commands

-   **`check`**: Validates the system environment (dependencies, FSL/MRtrix installations).
-   **`check-dataset`**: Assess acquisition quality of a DWI dataset.
-   **`fetch-data`**: Downloads FSL-licensed atlas data (FMRIB58_FA, Harvard-Oxford).
-   **`import`**: Converts DICOM to NIfTI or ingests existing NIfTI files.
-   **`preprocess`**: Runs denoising, unringing, and motion correction.
-   **`track`**: Performs whole-brain deterministic tractography.
-   **`extract`**: Extracts CST bundles using atlas-based ROIs.
-   **`metrics`**: Computes diffusion metrics and generates PDF/HTML reports.
-   **`validate`**: Compares extracted bundles against a ground truth.
-   **`run`**: Orchestrates the entire pipeline (Check → Import → Preprocess → Track → Extract → Metrics).
-   **`batch`**: Runs the pipeline in parallel across multiple subjects.

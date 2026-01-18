# Quick Start

## The `csttool run` Command

Run the entire CST assessment pipeline in one command:

```bash
csttool run /path/to/input /path/to/output --subject-id sub-01 --generate-pdf --save-visualizations
```

Given an input directory with DWI-MRI data in either DICOM or NIfTI format, and an output directory, it will run the following steps:

1. Import the data 
2. Preprocess the data
3. Generate a whole brain tractogram
4. Isolate the corticospinal tract using anatomical ROIs
5. Compute metrics
6. Generate a PDF, a JSON and a CSV report

The `--save-visualizations` flag will preserve visualizations of each step for quality control.

!!! tip "Need sample data?"
    See [recommended datasets](data-requirements.md#recommended-datasets#recommended-datasets) for freely available diffusion MRI data.

## Output structure

```
output/
├── sub-01_pipeline_report.json
├── nifti/
├── preprocessing/
├── tracking/
├── extraction/
└── metrics/
    ├── report.pdf
    ├── report.html
    └── metrics.csv
```

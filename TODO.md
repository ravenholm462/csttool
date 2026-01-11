# TO-DO

### General

- [ ] Write project documentation.

### CLI

- [x] Implement a `run` command that will run all the steps in the following order: `check`, `import`, `preprocess`, `track`, `extract`, `metrics`. 
- [X] Add "--visualizations" flag to save the figs.

### Import

- [X] Input DICOMS of one subject are represented as multiple series. The tool does not account for this currently. Resolve proper data import.

### Preprocessing

- [ ] Correction for Gibbs' oscillations missing. Implement.
- [ ] Is NLMEANS denoising good enough?
- [ ] Architecture of this module does not correspond to the others (still uses `funcs.py`). Revise.

### Tracking

- [X] Visualization of full brain tractogram is too messy. Fix.
- [ ] Add heatmap legends for MD, directions etc.

### Extraction

- [X] Output architecture too messy. Fix.
- [X] Visualizations of LFS/RHS CST too messy, same as for full brain tractogram. Fix.
- [ ] Extraction fails for alternate datasets (e.g. OpenNeuro, Stanford-HARDI). Full-brain tractogram generated, but all CST streamlines are filtered out. Fix.

### Metrics

- [ ] Revise PDF report (content, formatting, missing visuals).
- [ ] Fractional anisotropy PDF - add positional markings to the x axis (i.e. brainstem at the beginning, motor cortex at the end) to make orientation easier
- [X] Delete clinical interpretation text. It is not a clinical tool yet.
- [ ] Add QC visualizations to the report (e.g. the isolated CST, all 3 planes, see extraction visualizations)

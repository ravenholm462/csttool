# TO-DO

### General

[ ] Write project documentation.

### CLI

[ ] Implement a `run` command that will run all the steps in the following order: `check`, `import`, `preprocess`, `track`, `extract`, `metrics`. 

### Import

[ ] Input DICOMS of one subject are represented as multiple series. The tool does not account for this currently. Resolve proper data import.

### Preprocessing

[ ] Correction for Gibbs' oscillations missing. Implement.
[ ] Architecture of this module does not correspond to the others (still uses `funcs.py`). Revise.

### Tracking

[ ] Visualization of full brain tractogram is too messy. Fix.

### Extraction

[ ] Output architecture too messy. Fix.
[ ] Visualizations of LFS/RHS CST too messy, same as for full brain tractogram. Fix.

### Metrics

[ ] Revise PDF report (content, formatting, missing visuals).

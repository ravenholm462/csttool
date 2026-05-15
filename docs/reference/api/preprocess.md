# `csttool.preprocess`

The preprocessing module exposes the high-level functions that the CLI [`preprocess`](../cli/preprocess.md) command orchestrates. Use it directly when you want to script preprocessing from Python.

```python
from csttool.preprocess import preprocess

preprocess(
    nifti="raw_dwi.nii.gz",
    out="./preproc",
    denoise_method="patch2self",
    perform_motion_correction=True,
)
```

::: csttool.preprocess

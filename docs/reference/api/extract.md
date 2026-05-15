# `csttool.extract`

Atlas-based ROI filtering that isolates the bilateral CST from a whole-brain tractogram. The CLI counterpart is [`extract`](../cli/extract.md).

```python
from csttool.extract import extract_cst

extract_cst(
    tractogram="tracking/whole_brain.trk",
    fa="tracking/dti_FA.nii.gz",
    out="./extract",
    extraction_method="endpoint",
)
```

::: csttool.extract

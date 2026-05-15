# `csttool.metrics`

CST metric computation and PDF/HTML reporting. The CLI counterpart is [`metrics`](../cli/metrics.md).

```python
from csttool.metrics import compute_metrics

compute_metrics(
    cst_left="extract/cst_left.trk",
    cst_right="extract/cst_right.trk",
    fa="tracking/dti_FA.nii.gz",
    out="./metrics",
    generate_pdf=True,
)
```

::: csttool.metrics

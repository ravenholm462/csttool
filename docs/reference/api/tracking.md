# `csttool.tracking`

Whole-brain deterministic tractography. The CLI counterpart is [`track`](../cli/track.md).

```python
from csttool.tracking.modules import run_tracking

run_tracking(
    nifti="preproc/dti_preproc.nii.gz",
    out="./tracking",
    fa_thr=0.2,
    seed_density=1,
    rng_seed=42,
)
```

::: csttool.tracking

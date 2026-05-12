# Design Decisions

This page explains the reasoning behind non-obvious choices in csttool's architecture
and algorithms.

---

## Extraction methods: why four options exist

csttool offers four extraction methods, each trading off sensitivity, specificity, and
computational cost differently.

| Method | Input needed | L/R symmetric | Speed | Best for |
|--------|-------------|--------------|-------|----------|
| `passthrough` | Tractogram | Moderate (|LI| ≈ 0.1) | Fast | Cohort studies |
| `endpoint` | Tractogram | Moderate | Fast | Strict anatomical criterion |
| `roi-seeded` | Raw DWI | Poor (|LI| ≈ 0.35) | Moderate | Dense reconstruction |
| `bidirectional` | Raw DWI | Excellent (|LI| ≈ 0.002) | Slow | Single-subject symmetry |

---

## Bidirectional seeding: why not just use passthrough?

Passthrough filters a whole-brain tractogram for streamlines that traverse both the
brainstem and the motor cortex ROI. It is fast and works well for cohorts, but it
produces a modest streamline count asymmetry (|LI| ≈ 0.1 on typical data) because the
atlas-warped motor cortex ROI lands at a slightly different position relative to the
GM/WM boundary on each hemisphere.

A four-phase systematic audit on in-vivo 3T data confirmed:

1. The data quality is symmetric (no L/R signal imbalance across 71 DWI volumes)
2. Registration quality is symmetric (Jacobian determinant: L 1.000 ± 0.388, R 0.999 ± 0.330)
3. Motor ROI sizes and FA microstructure are symmetric (1,149 vs 1,135 FA > 0.2 voxels)
4. The underlying tract is symmetric — brainstem-seeded reverse tracking produces R/L = 0.987

The asymmetry is direction-dependent: passthrough gives R > L (LI = −0.128) while
roi-seeded gives L > R (LI = +0.347). An asymmetry that reverses sign with seeding
direction is the hallmark of a cortical interface placement artifact, not anatomy.

**Bidirectional seeding** eliminates this by:
- Running a forward pass (motor → brainstem) and a reverse pass (brainstem → motor)
- Retaining only forward streamlines whose count is bounded by the reverse count per side
- Enforcing the same bilateral target count (minimum across all four pass counts)
- Selecting from each forward bundle the streamlines with highest spatial overlap with
  the reverse density map

Result: LI = +0.002 — matches the brainstem-seeded ground truth (LI = +0.007).

Full technical write-up: [Bidirectional seeding — motivation and validation](../fixes/bidirectional_seeding.md)

---

## Why `bidirectional` is `run`-only, not available in `csttool extract`

`csttool extract` takes a pre-computed whole-brain tractogram as input. Bidirectional
seeding requires re-running tractography from two separate seed regions (motor cortex and
brainstem), which demands the raw DWI data. This is the same constraint as `roi-seeded`,
which has always been `run`-only.

Making it work with a pre-computed tractogram would require a different algorithm — for
example, filtering the existing tractogram by both endpoint regions and applying a
spatial overlap criterion. This is a valid future direction but would produce different
(and likely less accurate) results than the full bidirectional tracking approach.

---

## Why ODF parameters differ between passthrough and roi-seeded / bidirectional

The whole-brain tracking step (used by passthrough) uses stricter ODF parameters
(`relative_peak_threshold = 0.8`, `min_separation_angle = 45°`, `npeaks = 1`) to produce
a compact, high-quality whole-brain tractogram with minimal false connections.

The ROI-seeded and bidirectional methods use more permissive parameters
(`relative_peak_threshold = 0.5`, `min_separation_angle = 25°`) inherited from the
`roi_seeded_tracking` module. More permissive parameters allow the tracker to follow
complex crossing regions near the motor cortex and brainstem, which increases yield from
dense focal seeding.

This is an intentional asymmetry: whole-brain seeding needs conservative filtering to
keep tractogram size manageable; ROI seeding benefits from more flexibility because false
connections are later filtered by the ROI traversal criterion.

---

## Why `--extraction-method passthrough` is the default

Passthrough is the best balance of sensitivity, speed, and robustness for the common
case (cohort studies, first-time users). It works with a pre-computed tractogram (no
re-tracking needed), handles moderate motion and registration imperfection gracefully,
and has been validated on 167 TractoInferno subjects with 98.8% success rate.

Bidirectional is superior for single-subject bilateral symmetry analysis but is
approximately 3× slower (four tracking passes) and assumes symmetric anatomy — an
assumption that is invalid in stroke, tumour, or resection cases.

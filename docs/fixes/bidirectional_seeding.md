# Bidirectional Seeding — Motivation, Audit, and Implementation

**Added in:** csttool unreleased (post-0.5.0)  
**CLI flag:** `--extraction-method bidirectional` (available on `csttool run` only)  
**Module:** `src/csttool/extract/modules/bidirectional_filtering.py`

---

## 1. Problem statement

Atlas-based CST extraction seeds streamlines from the motor cortex (precentral gyrus,
Harvard-Oxford labels 7/107). The warped ROI centres do not land at identical positions
relative to the GM/WM boundary on each hemisphere. This causes the passthrough and
roi-seeded methods to yield asymmetric streamline counts even in healthy subjects.

On personal 3T in-vivo data (2026-04-24, Siemens Prisma_fit, NLMeans preprocessing):

| Method | Left CST | Right CST | R/L | LI |
|--------|---------|---------|-----|-----|
| Passthrough | 3,732 | 4,826 | 1.29 | −0.128 |
| ROI-seeded (forward) | 562 | 271 | 0.48 | +0.347 |

These two methods yield opposite asymmetry directions (right > left for passthrough,
left > right for roi-seeded), which is the hallmark of a **cortical interface placement
artifact**, not a structural difference.

---

## 2. Four-phase audit

A systematic investigation confirmed the cause before implementing the fix.

### Phase 1 — Raw data integrity

All 71 DWI volumes showed symmetric L/R signal balance (ratios 0.96–0.99, no flagged
volumes). Eddy outlier slices were corrected by `--repol`. No data quality cause.

### Phase 2 — Registration quality

Jacobian determinant: Left mean = 1.000 ± 0.388, Right mean = 0.999 ± 0.330, 0%
negative voxels on either side. Motor ROI sizes: Left 6,675 voxels, Right 6,499 voxels
(2.7% difference). ROI centres differ by 3.9 mm (Z) and 4.6 mm (Y) — the left ROI
sits marginally more superior and posterior. No registration failure.

### Phase 3 — ROI microstructure

| Test | Left | Right |
|------|------|-------|
| FA within motor ROI (mean) | 0.104 | 0.106 |
| FA > 0.2 seed-eligible voxels | 1,149 (17.2%) | 1,135 (17.5%) |
| Cerebral peduncle FA | 0.261 | 0.262 |
| Boundary voxel FA > 0.2 | 20% | 21% |

All symmetric. Seeding density cannot explain the asymmetry.

### Phase 4 — Seeding simulations

| Experiment | Seeds from | Left | Right | R/L |
|-----------|-----------|------|-------|-----|
| 4A: TDI whole-brain | — | 23.67 | 23.27 | **0.983** |
| 4B: ROI-seeded forward | Motor cortex | 562 | 271 | **0.48** |
| 4C: Brainstem-seeded reverse | Brainstem | 1,045 | 1,031 | **0.987** |

**Conclusion:** The whole-brain tractography is symmetric (4A). The tract itself is
symmetric (4C: brainstem seeds reach each motor cortex equally). The asymmetry appears
only when seeding from the motor cortex (4B) — and reverses direction between passthrough
and roi-seeded. This is a direction-dependent cortical interface placement artifact.

---

## 3. Algorithm

The bidirectional method retains only streamlines confirmed by both tracking directions.

```
Pass A (forward):
  seed motor_left_mask  → track → filter by brainstem_mask  → left_fwd
  seed motor_right_mask → track → filter by brainstem_mask  → right_fwd

Pass B (reverse):
  seed brainstem_mask → track → filter by motor_left_mask  → bs_to_left
  seed brainstem_mask → track → filter by motor_right_mask → bs_to_right

Voxelise:
  dens_L = density_map(bs_to_left,  affine, shape)
  dens_R = density_map(bs_to_right, affine, shape)

Per-side count cap (no bilateral cap):
  n_keep_L = min(len(left_fwd),  len(bs_to_left))
  n_keep_R = min(len(right_fwd), len(bs_to_right))

  cst_left  = top n_keep_L from left_fwd  ranked by overlap score with dens_L
  cst_right = top n_keep_R from right_fwd ranked by overlap score with dens_R
```

**Overlap score** for a streamline `s`: sum of density values `dens[vox]` at each voxel
the streamline passes through. Higher score = more spatial overlap with reverse bundle.

**Why count-bounded:** A naive `density > 0` intersection is a no-op — both forward and
reverse bundles traverse the same CST voxels. The asymmetry is in *how many* seeds
initiate valid CST trajectories, not *where* those trajectories go. Capping by the
reverse count removes excess forward streamlines that arose from the ROI placement
advantage.

**Why per-side only (no bilateral cap):** csttool is a clinical tool; a bilateral
symmetry cap would erase genuine unilateral CST reduction (stroke, tumour, Wallerian
degeneration). Each hemisphere is therefore bounded independently by its own reverse-pass
count, which removes the cortical-interface inflation per side without imposing a
symmetry prior. To help the user judge whether any residual L/R asymmetry is
methodological or structural, the method reports a per-side forward/reverse inflation
ratio and an `artifact_index` (see Section 6).

---

## 4. Result

On the same in-vivo data:

| Method | Left | Right | R/L | LI |
|--------|------|-------|-----|-----|
| Passthrough (NLMeans) | 3,732 | 4,826 | 1.29 | −0.128 |
| ROI-seeded forward | 562 | 271 | 0.48 | +0.347 |
| Bidirectional (per-side cap) | 441 | 271 | 0.61 | **+0.239** |
| Brainstem-seeded reference | 1,045 | 1,031 | 0.99 | +0.007 |

The per-side bidirectional method removes the cortical-interface inflation per side
(left forward 562 → 441 capped by reverse; right forward 271 → 271, no excess to remove)
without enforcing bilateral symmetry. The residual L/R asymmetry on this healthy subject
is flagged by the diagnostic: inflation_L = 562/441 = 1.27, inflation_R = 271/346 = 0.78,
artifact_index = 0.39 (> 0.20) — i.e. the method correctly reports that the residual
asymmetry is likely methodological rather than structural. All diffusion metrics (FA, MD,
RD, AD) remain symmetric regardless of method.

---

## 5. Implementation details

**New module:** `src/csttool/extract/modules/bidirectional_filtering.py`

Reuses the following utilities from `roi_seeded_tracking.py`:
- `generate_seeds_from_mask` — seed point generation
- `track_from_seeds` — LocalTracking wrapper
- `filter_by_target_roi` — ROI traversal filter
- `filter_by_length` — length filter
- `streamline_passes_through` — point-in-mask test (used for overlap scoring)

New helpers in `bidirectional_filtering.py`:
- `_voxelise(streamlines, affine, shape)` — returns density map (float32)
- `_overlap_score(streamline, density, affine)` — sum of density values along streamline
- `_select_top_n(forward_bundle, density, n_keep)` — selects top-N by overlap score

**ODF parameters** (shared across all four passes for consistent direction estimation):
- `relative_peak_threshold = 0.5`
- `min_separation_angle = 25°`
- `sh_order = 6`
- `fa_threshold = 0.15`

Peaks are computed once and reused for all four tracking passes, ensuring symmetric fiber
direction estimates.

**`bidirectional` is `run`-only**, like `roi-seeded`. It requires raw DWI data to run
tractography from both ends. `csttool extract` (which takes a precomputed tractogram)
prints a clear error message if `--extraction-method bidirectional` is passed.

---

## 6. Statistics reported

The extraction log (`sub-<id>_log-extraction.json`) includes, in addition to the standard
fields:

```json
{
  "statistics": {
    "left_seeds": 53400,
    "right_seeds": 51992,
    "bs_seeds": 59152,
    "left_forward_count": 562,
    "right_forward_count": 271,
    "bs_to_left_count": 441,
    "bs_to_right_count": 346,
    "cst_left_count": 441,
    "cst_right_count": 271,
    "left_intersection_rate": 78.5,
    "right_intersection_rate": 100.0,
    "forward_reverse_ratio_left": 1.27,
    "forward_reverse_ratio_right": 0.78,
    "artifact_index": 0.39,
    "method": "bidirectional"
  }
}
```

`left_intersection_rate` / `right_intersection_rate` give the fraction of each forward
bundle retained after the per-side cap.

`forward_reverse_ratio_left` / `forward_reverse_ratio_right` are per-side
forward-pass / reverse-pass ratios (the cortical-interface inflation).

`artifact_index = |inflation_L − inflation_R| / max(inflation_L, inflation_R)` indicates
whether residual L/R count asymmetry is methodological or structural:

- `artifact_index ≤ 0.20` → inflation symmetric on both sides; residual count asymmetry
  is likely structural (genuine biology or pathology).
- `artifact_index > 0.20` → one cortical ROI seeded more efficiently than the other;
  residual count asymmetry may be a methodological artifact and should be interpreted
  cautiously.

---

## 7. When to use bidirectional

**Use bidirectional when:**
- Single-subject analysis where you need to disentangle methodological from structural
  L/R count differences
- Passthrough laterality index |LI| > 0.15 and you want to confirm whether it is
  methodological (check `artifact_index`)
- You are comparing CST metrics between hemispheres
- Clinical populations with possible unilateral motor pathology — the method no longer
  enforces bilateral symmetry, so genuine asymmetry is preserved and the
  `artifact_index` helps distinguish it from interface artifacts

**Stick with passthrough when:**
- Large cohort studies (bidirectional is slower: 4 tracking passes + intersection)
- You only need group-level summaries and do not care about per-subject L/R counts

---

## 8. Related documents

- [Limitations — CST streamline count asymmetry](../explanation/limitations.md#cst-streamline-count-asymmetry)
- Processing report: `/mnt/neurodata/notes/processing-report-alem-20260424.md`
- Audit data: `/mnt/neurodata/out/alem/audit/`

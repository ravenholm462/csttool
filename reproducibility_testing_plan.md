# Reproducibility testing plan — in-house DWI dataset (2026-04-24)

## Context

A new in-house acquisition at `/mnt/neurodata/alem/TI_TEST1_ALEM_N_…/` provides **ten b=0 + 70×b1000 DWI runs on a single subject, same session** (Prisma_fit 3T, CMRR `ep_b0`, 2 mm iso, 72 slices, TR 2550, MB/iPAT off). It is the first real-subject dataset available for end-to-end csttool testing. The primary goal is to stress-test **reproducibility** along three independent axes — bit-level, input-swap, and preprocessing-sensitive — using the same subject so that between-run variance is dominated by acquisition/tool behaviour, not biology.

Findings from the metadata sweep (see [reports/bvec_diff_report.md](reports/bvec_diff_report.md)):

- **8 usable AP runs:** series 6, 8, 10, 12, 18 labelled AP *plus* 7, 9, 11, 13 labelled "PA" but actually AP (CSA `PhaseEncodingDirectionPositive=1`, same as AP; see report).
- **1 genuine PA run:** series 19 (`PhaseEncodingDirectionPositive=0`).
- Between-pair head-pose drift: slab pitch grows monotonically from ~2° (pair 1) to ~9° (pair 4); pair 5 (18/19) is acquired after a fresh localizer and is the "clean" reference.
- Every run has the **same 70-direction b-table** replayed byte-for-byte.

This gives us three things we could never get from synthetic DIPY data: (1) many repeats of the same subject/session, (2) a controlled motion/pose perturbation across repeats, (3) one real blip-reversed AP/PA pair.

## Goals

1. **Confirm bit-level determinism** on real clinical-shape data (not just DIPY `small_64D`).
2. **Quantify input-swap stability** — how much do CST metrics move when the same subject is re-scanned with a slightly different slab pose?
3. **Measure motion sensitivity** — do CST metrics drift monotonically with the known pose drift across pairs 1→4, and is that drift driven by slab pose or by within-run residual motion?
4. **Test the topup path end-to-end** on the one real AP/PA pair (18/19) and verify it improves (or at least doesn't degrade) metrics vs. the no-topup run of AP 0018 alone.

## Clinical-effect-size frame (what counts as "stable")

Reproducibility numbers are only meaningful against a clinically interpretable scale. Anchor the L2/L3 interpretation against published within-session CST metric variability:

- **Tract-mean FA:** within-session CV of ~1–3 % in healthy adults is the usual reproducibility bar (test–retest literature on DTI). Shifts >5 % warrant scrutiny.
- **Tract-mean MD/RD/AD:** CV ~1–2 % typical.
- **Volume / streamline count:** far noisier; CV 5–15 % is normal and dominated by seeding + masking.
- **FA laterality index (LI):** changes of **|ΔLI| > 0.10** are flagged as meaningful in ALS-CST literature; anything below ~0.03 is within noise.

These are the thresholds each metric table in L2/L3 will be compared against. Numbers deep inside the "noise" band are successes; numbers straddling the clinical threshold deserve investigation before claiming tool stability.

## Data layout for testing

Import once from DICOM with `csttool import` into a BIDS tree under `/home/alem/csttool/data/invivo-2026-04-24/`:

```
sub-01/ses-01/dwi/
  sub-01_ses-01_acq-ap_run-01_dwi.{nii.gz,bval,bvec,json}   # series 6
  sub-01_ses-01_acq-ap_run-02_dwi.{nii.gz,…}                # series 8
  sub-01_ses-01_acq-ap_run-03_dwi.{nii.gz,…}                # series 10
  sub-01_ses-01_acq-ap_run-04_dwi.{nii.gz,…}                # series 12
  sub-01_ses-01_acq-ap_run-05_dwi.{nii.gz,…}                # series 18 (post-relocalize)
  sub-01_ses-01_acq-ap_run-06_dwi.{nii.gz,…}                # series 7  (mislabelled PA → AP)
  sub-01_ses-01_acq-ap_run-07_dwi.{nii.gz,…}                # series 9
  sub-01_ses-01_acq-ap_run-08_dwi.{nii.gz,…}                # series 11
  sub-01_ses-01_acq-ap_run-09_dwi.{nii.gz,…}                # series 13
  sub-01_ses-01_dir-PA_run-01_epi.{nii.gz,…}                # series 19 (the real PA)
sub-01/ses-01/anat/
  sub-01_ses-01_T1w.{nii.gz,json}                            # series 5
```

Rationale: naming the ex-"PA" scans as AP (`acq-ap`) prevents downstream tools from treating them as blip-reversed. The one genuine PA is placed in the BIDS `fmap`-style slot as a `dir-PA_epi` sidecar referenced by pair-5 AP.

Critical files to reuse:
- [src/csttool/cli/commands/import_cmd.py](src/csttool/cli/commands/import_cmd.py) — DICOM → BIDS conversion (already uses dcm2niix)
- [src/csttool/bids/output.py](src/csttool/bids/output.py) — BIDS filename helpers
- [src/csttool/reproducibility/context.py](src/csttool/reproducibility/context.py) — per-run provenance capture
- [src/csttool/reproducibility/tolerance.py](src/csttool/reproducibility/tolerance.py) — numeric tolerances (reuse as-is for Level 1)

⚠ Manual override needed: for series 7/9/11/13 the dcm2niix JSON already writes `PhaseEncodingDirection: j-` (correct), but the **SeriesDescription contains "PA"** and any logic in csttool that infers direction from the filename or description must be checked. Audit [src/csttool/cli/commands/import_cmd.py](src/csttool/cli/commands/import_cmd.py) and [src/csttool/batch/modules/discover.py](src/csttool/batch/modules/discover.py) for such heuristics before the import; if any exist, pass `acq-ap` explicitly.

## Test levels

### Level 1 — Bit-level determinism (single real input, repeated) + provenance diff

**Question:** Does `csttool run` produce byte-identical tractograms and metrics across N repeats on real data, and does provenance differ only in the fields it's *supposed* to differ in?

**Procedure.** Pick one run (recommend **series 18 AP**, cleanest geometry). Run the full pipeline 3× with the same `--seed`, `--save-visualizations` off (faster), to separate output directories. Compare in order:

1. **Tractogram MD5** of the final `.trk` across the 3 runs. If identical, the output is provably bit-identical and no further coordinate arithmetic is needed.
2. If hashes differ: fall back to streamline fingerprint comparison via `compute_streamline_fingerprint` ([tests/reproducibility/test_determinism.py](tests/reproducibility/test_determinism.py)) + per-tract scalar summaries ([tests/reproducibility/test_metric_stability.py](tests/reproducibility/test_metric_stability.py)) against existing tolerances.
3. **Provenance diff** (absorbs the old Level 5): each run emits a blob via [get_provenance_dict](src/csttool/reproducibility/provenance.py). Assert that across the 3 repeats provenance differs **only** in `run_timestamp` and the pre-seed random-state snapshot. Any other difference (library version, platform, env var) is investigated, not silently accepted.

**Pass gate:** step 1 hash-identity is the strong gate. If it fails, step 2's tolerances (`TOLERANCE_COORDINATES_*`, `TOLERANCE_FA_MEAN_*`) are the fallback. A coordinate-level failure is a tool bug — likely an un-seeded library — and gets investigated (not just tolerated via loose thresholds).

**Artefact:** `reports/repro/L1_determinism.md` with hash table + provenance-diff summary.

### Level 2 — Input-swap stability (same subject, different acquisitions)

**Question:** How much do CST metrics move when the *acquisition* changes but the subject does not, separated into (a) seconds-apart repeats of the same pose and (b) across-pose drift?

**Critical design note.** The 8 usable AP runs are *not* 8 i.i.d. draws. Runs 6/7, 8/9, 10/11, 12/13 are **pairs** acquired ~3 minutes apart with (nearly) identical slab pose — each pair shares head position far more than any two runs from different pairs. Run 18 is a solo post-relocalize acquisition with no within-pair twin. Treating the 8 runs as an unblocked sample would inflate effective N and underestimate CV. The design must respect this blocking.

**Procedure.** Run the full pipeline on all **8 AP runs** (6, 7, 8, 9, 10, 11, 12, 13, 18) with identical seed and config via a single 9-entry `csttool batch` manifest ([src/csttool/batch/batch.py](src/csttool/batch/batch.py)). For each run record: left-CST volume, right-CST volume, mean FA / MD / RD / AD per side, LI (FA, MD, volume), streamline count.

**Analysis — two separate passes:**

1. **Within-pair (n=4 pairs):** compute |Δmetric| between the two members of each pair (6↔7, 8↔9, 10↔11, 12↔13). Four deltas per metric. This isolates *sequence + reconstruction + seeding* noise with pose held nearly constant — the tightest repeatability floor the tool can achieve on this scanner.
2. **Across-pair (n=5 pose levels):** take one representative per pair (or the pair mean) plus run-18, giving 5 points at 5 different slab poses. Quote CV across these 5 points, using a **pairwise / centroid-based distance** (mean pairwise |Δ|, or deviation from the 5-point centroid) rather than deltas against any single reference run — avoids the bias of privileging run-18.

Report both numbers side-by-side for every metric. The gap between (1) and (2) *is* the between-pose drift; if they're close, the tool is pose-insensitive at this scale.

**Pass gate:** qualitative, anchored by the clinical-effect-size frame above. FA/MD CVs well under 5 % (within-pair ideally <1–2 %) are a success. The numbers from (1) populate the "sensitivity tolerance" fields currently marked TODO in [src/csttool/reproducibility/tolerance.py:45-58](src/csttool/reproducibility/tolerance.py#L45-L58) — use within-pair p95 as the tolerance, since that represents the cleanest repeatability ceiling. The across-pair numbers document *drift*, not tolerance.

**Pilot.** Before the full 9-run batch, smoke one within-pair (6 + 7) end-to-end to flush out any config / filename / PE-direction bugs. Full batch only after pilot confirms sane output.

**Artefact:** `reports/repro/L2_input_swap.md` with: 9-row per-run metric table, within-pair delta table (4 rows × metrics), across-pair pairwise-distance summary, and proposed within-pair-p95 tolerance values.

### Level 3 — Motion sensitivity (two-axis analysis over Level 2 outputs)

**Question:** Do CST metrics drift with (a) the between-run slab-pose change, or (b) within-run residual motion, or both?

**Why two axes.** Slab pitch is a single static value per run (from CSA `sNormal`, known before preprocessing). Within-run residual motion is a per-volume quantity emitted by the between-volume motion-correction step of `csttool preprocess` ([src/csttool/preprocess/modules/](src/csttool/preprocess/modules/)) — typically stored as a per-volume rigid-transform log. The two are not redundant: a well-positioned slab can still have within-run motion, and vice versa. Only by plotting against both do we distinguish pose-setup sensitivity from subject-motion sensitivity.

**Procedure.** No new csttool runs — pure analysis over Level-2 outputs.

1. **Pose axis:** slab pitch angle per run (already computed: pair 1 = 3.0°, pair 2 = 1.8°, pair 3 = 3.8°, pair 4 = 8.8°, pair 5 = 10.7°; see [/home/alem/.claude/plans/data-acquisition-is-done-dapper-boot.md](/home/alem/.claude/plans/data-acquisition-is-done-dapper-boot.md)). 5 points (one per pair mean).
2. **Motion axis:** extract the mean (or 90th-percentile) RMS displacement across volumes from the motion-correction log of each of the 8 AP runs. 8 points.
3. For each metric (FA, MD, LI, volume…) compute Spearman ρ vs. both axes independently, and a partial correlation controlling for the other axis if N permits. Scatter each metric against both axes on the same page.

**Pass gate:** qualitative, interpreted against the clinical-effect-size frame. The pattern matters more than any single ρ:

- Low |ρ| on both axes → tool is robust to this dataset's motion range.
- High |ρ| on pose, low on motion → pose-setup is the dominant driver (flag for thesis discussion).
- High |ρ| on motion, low on pose → within-run motion dominates; argue for stronger motion-correction or rejection thresholds.
- High on both → the two are entangled and this dataset can't separate them — state that honestly rather than claim a specific mechanism.

**Artefact:** `reports/repro/L3_motion.md` + one two-panel PNG per metric (pose axis | motion axis).

### Level 4 — Topup / distortion-correction end-to-end (or L4′ distortion gap)

**Question:** Does running preprocessing with real blip-reversed topup on pair 5 (18/19) change CST metrics vs. running AP-only on series 18?

**Procedure — Level 4 proper (only if topup is wired in).**
- **Run A:** `csttool preprocess` on AP 0018 alone (no reverse-PE).
- **Run B:** `csttool preprocess` on AP 0018 with `--reverse-pe` pointing at PA 0019 b=0.
- Propagate both through `track` + `extract` + `metrics` with the same seed.
- Diff final metrics and overlay the two CST tractograms.

**Prerequisite.** Check whether the current preprocessing pipeline exposes a reverse-PE / topup path. From [src/csttool/preprocess/preprocess.py](src/csttool/preprocess/preprocess.py) and `modules/`, it looks like only NLMEANS denoising, median-Otsu masking, and between-volume motion correction are implemented — **no topup wrapper yet**. If that's confirmed, L4 becomes L4′ below.

**Level 4′ — distortion gap (confound-aware).** Running PA 0019 through the AP pipeline as-if-AP would mix two independent effects: (i) EPI distortion differing from the AP slab's, and (ii) the EPI→T1w registration step of `csttool extract` fighting the different distortion pattern, causing artefactual partial-volume and atlas-mapping shifts. A naive comparison between series 18 (AP) and series 19 (PA-as-AP) CSTs would attribute registration failure to distortion.

Mitigation — isolate distortion from registration via a shared-transform protocol:

1. Compute the T1w→MNI warp **once** from the T1-MPRAGE (series 5). Use it identically for both 18 and 19.
2. Derive the EPI→T1w rigid transform **from series 18 only** and apply it to both series 18 and series 19 derivatives. Do *not* re-estimate it from the distorted PA image — that's what would conflate distortion with registration.
3. Track both 18 and 19 through `csttool track` in their own native EPI space, then map streamlines into T1w space using the fixed transforms from steps 1–2.
4. Compare CSTs in T1w space. Residual differences now cleanly isolate the effect of EPI distortion on streamline geometry/metrics.

State explicitly in the writeup that L4′ demonstrates the *magnitude of uncorrected distortion's effect*, not a validated distortion correction — because there is no ground-truth CST to claim "correction" against.

**Artefact:** `reports/repro/L4_topup.md` if L4 ran, otherwise `reports/repro/L4_distortion_gap.md` with the shared-transform protocol, the resulting CST overlay, and metric deltas framed as a distortion-sensitivity estimate.

## Execution order

1. **Audit** `import_cmd.py` + `discover.py` for filename-based PE-direction inference (30 min, read-only).
2. **Import once** into the BIDS tree above (~5 min per series × 10 = 50 min).
3. **L2 pilot**: smoke one within-pair (series 6 + series 7) end-to-end before the full batch, to flush out import / PE-inference / filename bugs.
4. **Level 1** (1 series × 3 repeats ≈ 3× full pipeline runtime). Absorbs the provenance-diff check (old L5).
5. **Level 2** full batch (9 AP runs via `csttool batch`).
6. **Level 3** — pure analysis pass over Level-2 outputs and motion-correction logs, no new csttool runs.
7. **Level 4** or **L4′** depending on preprocessing-path audit.

Rough wallclock budget: 1 day import + scripting, 1–2 days compute (dominated by Level 2), 1 day analysis + writeup.

## Implementation skeleton

Drive the whole plan from a `Makefile` at the repo root (or `justfile` — same idea). Each level is a target; intermediate outputs are real files so targets are resumable and only what changed re-runs. The one heavy artefact is the L2 `csttool batch` manifest; everything else is thin analysis glue.

```make
# Makefile — reproducibility study
DATA      := data/invivo-2026-04-24
DICOM     := /mnt/neurodata/alem/TI_TEST1_ALEM_N_26_04_24-13_19_39-DST-1_3_12_2_1107_5_2_43_67043/HEAD_HEAD_STUDIEN_20260424_132122_391000
REPORTS   := reports/repro
SEED      := 42

# ---- one-off: DICOM -> BIDS ------------------------------------------------
$(DATA)/dataset_description.json:
	csttool import --dicom $(DICOM) --out $(DATA) --bids

import: $(DATA)/dataset_description.json

# ---- L2 pilot: one within-pair, full pipeline ------------------------------
pilot: import
	csttool run --bids $(DATA) --subject sub-01 --acq ap --run 01 \
		--out $(REPORTS)/pilot/run-01 --seed $(SEED)
	csttool run --bids $(DATA) --subject sub-01 --acq ap --run 06 \
		--out $(REPORTS)/pilot/run-06 --seed $(SEED)
	python scripts/repro/check_pilot.py $(REPORTS)/pilot

# ---- L1: bit-level determinism on series 18 --------------------------------
L1: import
	for i in 1 2 3; do \
		csttool run --bids $(DATA) --subject sub-01 --acq ap --run 05 \
			--out $(REPORTS)/L1/rep-$$i --seed $(SEED); \
	done
	python scripts/repro/l1_determinism.py \
		$(REPORTS)/L1/rep-1 $(REPORTS)/L1/rep-2 $(REPORTS)/L1/rep-3 \
		> $(REPORTS)/L1_determinism.md

# ---- L2: input-swap stability (9 runs, one batch) --------------------------
L2: import configs/l2_manifest.tsv
	csttool batch --manifest configs/l2_manifest.tsv \
		--out $(REPORTS)/L2/runs --seed $(SEED)
	python scripts/repro/l2_within_vs_across.py \
		$(REPORTS)/L2/runs > $(REPORTS)/L2_input_swap.md

# ---- L3: motion sensitivity (pure analysis, no new runs) -------------------
L3: L2
	python scripts/repro/l3_motion_axes.py \
		--runs $(REPORTS)/L2/runs \
		--pose-table configs/slab_pitch.tsv \
		> $(REPORTS)/L3_motion.md

# ---- L4 / L4': distortion (shared-warp protocol) ---------------------------
L4: L2
	python scripts/repro/l4_distortion.py \
		--ap  $(REPORTS)/L2/runs/sub-01_run-05 \
		--pa  $(DATA)/sub-01/ses-01/fmap/sub-01_ses-01_dir-PA_run-01_epi.nii.gz \
		--t1w $(DATA)/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz \
		> $(REPORTS)/L4_distortion_gap.md

all: L1 L2 L3 L4

.PHONY: import pilot L1 L2 L3 L4 all
```

The matching `configs/l2_manifest.tsv` is a 9-row TSV mapping each series to a subject-id / acq / run label — the format `csttool batch` already accepts (see [src/csttool/batch/modules/manifest.py](src/csttool/batch/modules/manifest.py)). `configs/slab_pitch.tsv` is a 2-column file `(run_label, pitch_deg)` with values from the metadata report.

Scripts under `scripts/repro/` are each ~50–100 lines:

- `l1_determinism.py` — MD5 tractograms + provenance diff
- `l2_within_vs_across.py` — load the 9 metrics tables, emit within-pair Δ and across-pair pairwise-distance summaries
- `l3_motion_axes.py` — extract motion RMS from `csttool preprocess` logs, scatter + Spearman vs. both axes
- `l4_distortion.py` — shared-warp distortion comparison
- `check_pilot.py` — smoke assertions on the pilot pair

Re-running one level costs only that level's compute: tweaking an L3 plot never touches L2's ~1–2 days.

## Deliverables

- One BIDS dataset at `data/invivo-2026-04-24/` (not checked in; path noted in `.gitignore`).
- Per-level markdown in `reports/repro/L{1,2,3,4}_*.md`.
- Filled-in sensitivity tolerances in [src/csttool/reproducibility/tolerance.py](src/csttool/reproducibility/tolerance.py) from Level 2 *within-pair p95*.
- Provenance-diff assertion folded into the L1 test harness under [tests/reproducibility/](tests/reproducibility/).
- A short summary section in the thesis results chapter (`thesis/results/`) once all four levels have reports.

## Verification (end-to-end smoke)

Before running the full suite, do a smoke test: `csttool run --dicom <series-18-dicom-dir> --out /tmp/smoke --subject-id smoke-test --seed 42`. Confirm it finishes cleanly, produces a CST `.trk`, and writes `provenance.json`. If that works the four levels above are just repeated invocations with different inputs and accounting scripts.

## Out of scope

- Cross-subject reproducibility (we have one subject).
- Cross-scanner reproducibility (one scanner).
- Longitudinal reproducibility (one session).
- Multi-shell behaviour (single b=1000 shell only).
- Eddy-current correction (no eddy-blipped variants acquired).

Each of these is a legitimate follow-up but is not answerable with this dataset.

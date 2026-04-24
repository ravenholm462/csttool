# AP/PA pair check — bvec diff + PE polarity

Date: 2026-04-24
Source: `/mnt/neurodata/alem/TI_TEST1_ALEM_N_26_04_24-…/HEAD_HEAD_STUDIEN_20260424_132122_391000/`
Converter: `dcm2niix` (v from project venv), outputs in `/tmp/bvec_check/`

## TL;DR

**Only pair 5 (series 18/19) is a genuine blip-reversed AP/PA pair.** Pairs 1–4 (series 6/7, 8/9, 10/11, 12/13) have PA runs whose bvecs match their AP exactly **and** whose CSA `PhaseEncodingDirectionPositive` flag is **identical to the AP**. They are not PE-reversed and are not usable as topup inputs. PA 19 is the only PA run whose PE polarity is actually flipped.

## What the numbers say

### 1. bvec tables are identical within every pair

For each of the five pairs I loaded the `.bvec` files (3×71 matrix) and compared the 70 directional volumes (b=1000) column-by-column. Result:

| Pair | PE(AP) in JSON | PE(PA) in JSON | bvecs identical (|Δ|<1e-3) | after flipping y of AP |
|---|---|---|---|---|
| 1 (6/7)   | `j-` | `j-` | **70/70** | 0/70 |
| 2 (8/9)   | `j-` | `j-` | **70/70** | 0/70 |
| 3 (10/11) | `j-` | `j-` | **70/70** | 0/70 |
| 4 (12/13) | `j-` | `j-` | **70/70** | 0/70 |
| 5 (18/19) | `j-` | `j`  | **70/70** | 0/70 |

So the diffusion gradient table is replayed byte-for-byte in every PA. Mean |AP·PA| = 1.0000 across all pairs.

### 2. But only pair 5 has the PE polarity actually reversed

`.bvec` identity is not enough — dcm2niix expresses bvecs in image space, so if the image is rotated 180° in-plane the bvecs come out unchanged even when the underlying gradient directions in patient space differ. The decisive field is **`PhaseEncodingDirectionPositive`** in the Siemens CSA image header (the scanner's own record of blip polarity):

| Series | Role | CSA `PhaseEncodingDirectionPositive` |
|---|---|---|
| 0006 | AP pair 1 | `1` |
| 0007 | PA pair 1 | **`1`** ← same as AP |
| 0008 | AP pair 2 | `1` |
| 0009 | PA pair 2 | **`1`** ← same as AP |
| 0010 | AP pair 3 | `1` |
| 0011 | PA pair 3 | **`1`** ← same as AP |
| 0012 | AP pair 4 | `1` |
| 0013 | PA pair 4 | **`1`** ← same as AP |
| 0018 | AP pair 5 | `1` |
| 0019 | PA pair 5 | **`0`** ← properly flipped |

`BandwidthPerPixelPhaseEncode = 13.355` in every run, so readout timing is identical.

This is corroborated by the dcm2niix sidecar `PhaseEncodingDirection`: `j-` in all APs, `j-` in PA 7/9/11/13 (same as AP), `j` only in PA 19.

## Interpretation

The `sWipMemBlock.alFree[0] = 256` WIP flag that PA 7/9/11/13 set (and PA 19 does NOT set) is **not** a blip-polarity reversal. It appears to be the 180° in-plane-rotation toggle of the CMRR diffusion sequence: the slab is rotated so the image looks like a PA acquisition, but the phase-encoding gradient is blipped in the same direction as the AP run. Net effect — those PA runs carry **the same EPI distortion as their paired AP**, so they cannot be used to estimate a b0 field with `topup`.

PA 19 does not set `alFree[0]`; instead its PE polarity is genuinely inverted at the scanner level (CSA flag = 0), giving an actual blip-reversed b=0 suitable for `topup`.

## Practical consequences

- **For topup/eddy:** use **only pair 5 (18/19)** as the AP/PA pair. The b=0 volumes from AP 0018 + PA 0019 are the only distortion-reversed pair available.
- **For everything else (tractography, tensor fitting, motion simulation studies):** pairs 1–4 are still usable as five independent AP DWI acquisitions (total 10 AP runs counting pair 5 AP) of the same 70-direction b=1000 table — just do **not** treat the "PA" members as blip-reversed. Label them PE = `j-` (same as AP) in any BIDS `dir-` entity.
- The dcm2niix JSON for pairs 1–4 already correctly reports `PhaseEncodingDirection: j-` for both AP and PA members, so downstream BIDS tools will not be misled as long as filenames are not blindly based on the series description ("PA" in the name is misleading for 7/9/11/13).

## Files produced

- Ten NIfTIs + sidecars at `/tmp/bvec_check/CMRR_MBEP2D_DIFF_*.nii` / `.bval` / `.bvec` / `.json` (temporary, safe to delete)
- This report: `reports/bvec_diff_report.md`

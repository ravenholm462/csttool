You’ve got a solid “user journey” doc. The gaps are not about prose, they’re about correctness contracts, reproducibility, and failure modes. Here’s a direct review that tests the logic and calls out where your current wording will mislead users or hide bugs.

## Assumptions you are taking for granted

1. **Everything is in the same space**

   * You imply the tractogram and FA map match, and that warped atlas masks match the tractogram’s coordinate system.
   * In practice, `.trk` can carry an affine, an “image space” header, or neither, and your FA defines another grid. Without explicit checks, you can silently extract garbage.

2. **Harvard-Oxford registration to FA is reliable “by default”**

   * You present “Affine + SyN” as if it will converge and be meaningful across subjects, artifacts, partial coverage, and anisotropic voxels.
   * Even “converged” can mean “optimizer stopped”, not “good alignment”.

3. **“Bilateral separation automatically splits into left/right CST” is well-defined**

   * How do you define left vs right? By motor ROI hemisphere only, by streamline midpoint x sign, by endpoint location, by majority overlap? These can disagree, especially near the midline.

4. **Length thresholds in mm are comparable across coordinate conventions**

   * Streamline length depends on step size, voxel sizes, and whether coordinates are in mm. If the tractogram is in voxel coordinates or has wrong scaling, your length filter becomes nonsense.

5. **ROI dilation “ensures ROIs overlap with white matter streamlines”**

   * This is a heuristic that can easily inflate false positives, especially if dilation crosses sulci or merges across CSF in low resolution data. It needs guardrails.

## Logic tests and potential flaws

### 1. Space contract is not enforced (this is the biggest gap)

Right now you only *suggest* “check that tractogram and FA are in the same space” in troubleshooting. That is too late. This should be a hard preflight check.

What should be explicitly documented and implemented:

* **Define the reference space**: FA NIfTI grid (shape + affine) is the extraction grid.
* **Validate tractogram space**:

  * If tractogram has an affine or a reference image, verify it matches FA within tolerance.
  * If it does not, error with a clear message and how to fix.
* **Validate ROI masks**: Confirm warped ROI masks are in FA grid and same affine.

If you do not do this, you will get “reasonable looking” outputs that are anatomically wrong. This is a reproducibility killer for your thesis pipeline .

### 2. Registration: you state parameter schedules as facts, but users cannot verify

You list:

* Standard affine iterations `[10000, 1000, 100]`, SyN `[10, 10, 5]`
* Fast affine `[1000, 100, 10]`, SyN `[5, 5, 3]`

Two issues:

* Those numbers are meaningless without saying **which library** and **which meaning** (multi-resolution levels, iteration per level, metric, sampling, shrink factors, smoothing sigmas, etc.).
* Your log example says “converged”, but you do not provide **any QC metric**.

Minimum documentation upgrade:

* Record in the JSON report: transform type, metric, number of levels, iteration schedule, and a scalar quality indicator (for example final metric value, or Dice of a coarse brain mask after warp).
* Save overlays by default (not only with `--save-visualizations`) at least for the ROI masks on FA.

### 3. “Passthrough” and “Endpoint” definitions need precision

You say:

* Passthrough: “streamlines that pass through both ROIs”
* Endpoint: “endpoints fall within the ROIs”

Ambiguities you must pin down:

* Does “pass through” mean **any point intersects ROI voxel mask**? Or “intersects after resampling to step size X”?
* For endpoint: which endpoint corresponds to which ROI? Do you require one endpoint in motor and the other in brainstem, or do you accept both endpoints in motor if the streamline also intersects brainstem?
* What happens if a streamline touches both left and right motor ROIs due to dilation or midline crossing?

Right now “automatically splits left/right” suggests you handle all these, but the doc doesn’t explain how.

### 4. Hemispheric labeling can be wrong without a midline rule

Your pipeline will encounter streamlines that:

* start in left motor ROI but cross to right side,
* intersect both motor ROIs due to dilation,
* terminate in cerebellum or other brainstem-adjacent areas.

You need a deterministic rule, documented, and stored in the report. Examples:

* Label by **motor ROI membership of the endpoint closest to cortex** (if endpoint method).
* Label by **largest overlap fraction with left vs right motor ROI** (for passthrough).
* Reject streamlines that match both hemispheres above a threshold.

### 5. ROI choice is too broad and will pull non-CST fibers

“Brainstem” from Harvard-Oxford subcortical atlas is a very permissive ROI for CST. Passthrough with dilation is likely to capture:

* corticobulbar fibers,
* frontopontine tracts,
* and other descending pathways that traverse brainstem.

That might be acceptable if your thesis goal is “CST-like descending tract”, but the doc claims CST. You should either:

* tighten the brainstem ROI definition (for example restrict to cerebral peduncle region if you have an atlas that supports it), or
* explicitly acknowledge the anatomical limitation and treat it as “candidate CST bundle” pending validation.

Given your thesis objective is automated CST assessment, the doc should clearly separate “extraction heuristic” from “validated CST” .

### 6. “Typical yield 1 to 2%” is not stable and can mislead

Yield depends heavily on:

* tractography algorithm and parameters used to create the whole-brain tractogram,
* number of streamlines,
* seeding strategy,
* step size, stopping criteria,
* data quality.

Putting a yield number makes users think something is “wrong” when it is not, or “fine” when it is wrong. Better:

* remove the percentage, or
* scope it: “observed in our pipeline with X tractography settings on dataset Y”.

### 7. `--verbose` “default enabled” conflicts with typical CLI expectations

If verbose is enabled by default, it is not “verbose”, it is “normal”. Users expect `--verbose` to increase logging, and default to concise output.
This is small, but it causes friction and makes batch logs noisy.

### 8. Outputs: what exactly is in the JSON report?

You mention “extraction statistics and metadata” but do not define fields. For reproducibility, it should include:

* input file paths and checksums (or at least filenames),
* FA image shape, voxel size, affine hash,
* tractogram header info (voxel to RAS mapping if present),
* registration settings + transform filenames,
* ROI voxel counts pre and post dilation,
* filtering counts at each stage (length filter before and after ROI filter),
* hemisphere split logic and counts,
* software version, commit hash, and random seeds (if any randomness exists elsewhere).

Right now, the doc reads like the JSON exists but does not guarantee it is useful.

## Concrete fixes to the documentation

If you change nothing else, add these sections:

1. **Spatial Consistency Contract (Preflight)**

   * FA defines reference grid.
   * Tractogram must be in FA world coordinates (mm) or must include a reference that matches FA.
   * ROI masks are warped into FA grid.
   * Tool errors out on mismatch.

2. **Hemisphere Assignment Rule**

   * One clear rule for passthrough and endpoint.
   * Behavior for ambiguous streamlines.

3. **Registration QC**

   * What QC artifacts are always saved.
   * What constitutes a warning vs hard failure.

4. **What “Brainstem” and “Motor” mean**

   * Exact atlas labels used (Harvard-Oxford label names and thresholds).
   * Any binarization thresholding.
   * A warning that these ROIs are atlas-derived approximations in subject space.

5. **Report schema**

   * A short table of JSON keys.

## Quick “red flag” checklist for your current pipeline behavior

If any of these are true, your extraction can be silently wrong:

* tractogram was produced in a different space than the FA map
* FA was skull-stripped or resampled but tractogram references original grid
* warped ROIs look plausible but are shifted by a few mm
* dilation causes left and right motor ROIs to touch near midline
* endpoints are not truly cortical due to streamline truncation at WM mask

If you want, paste the extraction report JSON schema (or a real report example) and the exact hemisphere-splitting logic. I’ll sanity check it for “can this ever mislabel or leak across hemispheres” and propose a minimal set of assertions that should hard-fail.

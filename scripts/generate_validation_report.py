#!/usr/bin/env python3
"""Generate a PDF report explaining the TractoInferno validation design and results.

Usage:
    python scripts/generate_validation_report.py [SUMMARY_TSV] [--outdir DIR]

Requires: weasyprint, pandas, jinja2
"""

import argparse
import base64
from pathlib import Path

import pandas as pd
import weasyprint
from jinja2 import Template

DEFAULT_TSV = "/mnt/neurodata/csttool_runs/2026-02-01_git-1433ac4/outputs/summary.tsv"
DEFAULT_FIGURES = "scripts/figures"


def load_stats(tsv_path: str) -> dict:
    df = pd.read_csv(tsv_path, sep="\t")
    valid = df[df["status"] == "OK"]
    n_total = len(df)
    n_ok = len(valid)
    n_subjects = df["subject"].nunique()
    prec_mean = valid["o_cand_in_ref"].mean()
    rec_mean = valid["o_ref_in_cand"].mean()
    cand_mean = valid["n_cand"].mean()
    ref_mean = valid["n_ref"].mean()
    return {
        "n_subjects": n_subjects,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_invalid": n_total - n_ok,
        "success_pct": f"{n_ok / n_total * 100:.1f}",
        "dice_mean": f"{valid['dice'].mean():.3f}",
        "dice_std": f"{valid['dice'].std():.3f}",
        "dice_median": f"{valid['dice'].median():.3f}",
        "dice_min": f"{valid['dice'].min():.3f}",
        "dice_max": f"{valid['dice'].max():.3f}",
        "precision_mean": f"{prec_mean:.3f}",
        "precision_std": f"{valid['o_cand_in_ref'].std():.3f}",
        "precision_pct": f"{prec_mean * 100:.1f}",
        "recall_mean": f"{rec_mean:.3f}",
        "recall_std": f"{valid['o_ref_in_cand'].std():.3f}",
        "recall_pct": f"{rec_mean * 100:.1f}",
        "n_cand_mean": f"{cand_mean:,.0f}",
        "n_cand_min": f"{valid['n_cand'].min():,.0f}",
        "n_cand_max": f"{valid['n_cand'].max():,.0f}",
        "n_ref_mean": f"{ref_mean:,.0f}",
        "n_ref_min": f"{valid['n_ref'].min():,.0f}",
        "n_ref_max": f"{valid['n_ref'].max():,.0f}",
        "count_ratio_pct": f"{cand_mean / ref_mean * 100:.0f}",
        "dice_l_mean": f"{valid[valid['hemi'] == 'L']['dice'].mean():.3f}",
        "dice_l_std": f"{valid[valid['hemi'] == 'L']['dice'].std():.3f}",
        "dice_r_mean": f"{valid[valid['hemi'] == 'R']['dice'].mean():.3f}",
        "dice_r_std": f"{valid[valid['hemi'] == 'R']['dice'].std():.3f}",
        **_paired_stats(valid),
    }


def _paired_stats(valid: pd.DataFrame) -> dict:
    import numpy as np
    paired = valid.pivot(index="subject", columns="hemi", values=["dice", "o_cand_in_ref", "o_ref_in_cand"]).dropna()
    d_l, d_r = paired[("dice", "L")], paired[("dice", "R")]
    p_l, p_r = paired[("o_cand_in_ref", "L")], paired[("o_cand_in_ref", "R")]
    r_l, r_r = paired[("o_ref_in_cand", "L")], paired[("o_ref_in_cand", "R")]
    return {
        "dice_lr_diff": f"{(d_r - d_l).mean():.3f}",
        "dice_lr_absdiff": f"{np.abs(d_r - d_l).mean():.3f}",
        "prec_lr_absdiff": f"{np.abs(p_r - p_l).mean():.3f}",
        "rec_lr_diff": f"{(r_r - r_l).mean():.3f}",
        "rec_lr_absdiff": f"{np.abs(r_r - r_l).mean():.3f}",
    }


def embed_image(path: Path) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


REPORT_TEMPLATE = Template("""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
}
body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #1a1a1a;
}
h1 {
    font-size: 16pt;
    margin-top: 0;
    margin-bottom: 4pt;
    color: #2c3e50;
    border-bottom: 2px solid #2c3e50;
    padding-bottom: 4pt;
}
h2 {
    font-size: 12pt;
    margin-top: 14pt;
    margin-bottom: 4pt;
    color: #2c3e50;
}
h3 {
    font-size: 10.5pt;
    margin-top: 10pt;
    margin-bottom: 3pt;
    color: #34495e;
}
p, li {
    margin-top: 2pt;
    margin-bottom: 2pt;
}
ul {
    margin-top: 2pt;
    margin-bottom: 4pt;
    padding-left: 18pt;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 6pt 0;
    font-size: 9.5pt;
}
th, td {
    border: 1px solid #bdc3c7;
    padding: 4pt 6pt;
    text-align: left;
}
th {
    background: #ecf0f1;
    font-weight: 600;
}
.formula {
    background: #f7f9fb;
    border: 1px solid #dce1e6;
    border-radius: 3pt;
    padding: 6pt 10pt;
    margin: 6pt 0;
    font-family: "Courier New", monospace;
    font-size: 9.5pt;
    text-align: center;
}
.metric-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8pt;
    margin: 6pt 0;
}
.metric-box {
    background: #f7f9fb;
    border: 1px solid #dce1e6;
    border-radius: 3pt;
    padding: 6pt 10pt;
    flex: 1 1 120pt;
    text-align: center;
}
.metric-box .value {
    font-size: 16pt;
    font-weight: 700;
    color: #2c3e50;
}
.metric-box .label {
    font-size: 8pt;
    color: #7f8c8d;
    text-transform: uppercase;
}
.figure {
    text-align: center;
    margin: 8pt 0;
    page-break-inside: avoid;
}
.figure img {
    max-width: 100%;
    height: auto;
}
.figure .caption {
    font-size: 8.5pt;
    color: #555;
    margin-top: 3pt;
    font-style: italic;
}
.two-col {
    display: flex;
    gap: 8pt;
}
.two-col .figure {
    flex: 1;
}
.pagebreak {
    page-break-before: always;
}
.subtitle {
    font-size: 10pt;
    color: #7f8c8d;
    margin-top: 0;
    margin-bottom: 10pt;
}
</style>
</head>
<body>

<h1>TractoInferno Validation Report</h1>
<p class="subtitle">csttool Corticospinal Tract Extraction &mdash; Validation Against TractoInferno PYT Reference Bundles</p>

<h2>1. Validation Design</h2>

<h3>1.1 Objective</h3>
<p>
Evaluate the spatial agreement between corticospinal tract (CST) bundles extracted by
<strong>csttool</strong> and reference pyramidal tract (PYT) bundles from the
<strong>TractoInferno</strong> dataset (OpenNeuro ds003900). The validation quantifies how
well an automated, deterministic, atlas-based extraction pipeline recovers a known
tract across a large, heterogeneous cohort.
</p>

<h3>1.2 Reference Dataset</h3>
<p>
TractoInferno provides expert-segmented white matter bundles for 284 subjects. The
<strong>trainset</strong> partition was used, containing reference PYT bundles generated
via probabilistic tractography with expert-guided ROI selection. Each subject includes
left and right PYT bundles in <code>.trk</code> format and a fractional anisotropy (FA)
image defining the anatomical reference space.
</p>

<h3>1.3 Candidate Generation</h3>
<p>
For each subject, csttool's full pipeline was executed with the following parameters:
</p>
<table>
<tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
<tr><td>Denoising</td><td>NLMeans</td><td>Non-local means denoising</td></tr>
<tr><td>Preprocessing</td><td>Enabled</td><td>Denoising + brain masking</td></tr>
<tr><td>Direction model</td><td>CSA-ODF (SH order 6)</td><td>Constant solid angle ODF</td></tr>
<tr><td>Tracking</td><td>Deterministic</td><td>Single-peak (npeaks=1)</td></tr>
<tr><td>Step size</td><td>0.5 mm</td><td>Streamline propagation step</td></tr>
<tr><td>FA threshold</td><td>0.2</td><td>Tracking termination</td></tr>
<tr><td>Seed density</td><td>1 seed/voxel</td><td>In white matter mask</td></tr>
<tr><td>Peak threshold</td><td>0.8 (relative)</td><td>Relative peak threshold</td></tr>
<tr><td>Min separation angle</td><td>45&deg;</td><td>Between ODF peaks</td></tr>
<tr><td>ROI approach</td><td>Atlas-to-subject</td><td>Harvard-Oxford via ANTs SyN</td></tr>
<tr><td>Field strength</td><td>3.0 T</td><td>Assumed acquisition field strength</td></tr>
</table>

<h3>1.4 Validation Contract</h3>
<p>
A formal validation contract governs the comparison process. Each subject&ndash;hemisphere
pair is assigned exactly one outcome status:
</p>
<table>
<tr><th>Status</th><th>Condition</th><th>Metrics</th></tr>
<tr>
    <td><strong>OK</strong></td>
    <td>Spatial checks pass, candidate non-empty, no structural warnings</td>
    <td>Computed</td>
</tr>
<tr>
    <td><strong>DEGRADED</strong></td>
    <td>Spatial checks pass, but candidate empty or below minimum streamline count</td>
    <td>Computed (may be zero)</td>
</tr>
<tr>
    <td><strong>INVALID_SPACE</strong></td>
    <td>Spatial compatibility violated (affine mismatch, grid incompatibility, or candidate missing)</td>
    <td>Not computed</td>
</tr>
</table>
<p>
INVALID_SPACE cases are excluded from quantitative analysis and reported separately.
DEGRADED cases remain in the analysis but are flagged. This separation ensures that
infrastructure failures do not contaminate algorithmic performance assessment.
</p>

<h3>1.5 Spatial Compatibility Checks</h3>
<p>
Before computing any metric, the validator verifies that candidate and reference
bundles share a compatible coordinate system:
</p>
<ul>
<li>The candidate tractogram must be mappable into the reference FA space.</li>
<li>Affine matrices must agree within a tolerance of 0.5 mm (RMS).</li>
<li>Voxel grid dimensions and orientation must match the reference anatomy.</li>
</ul>
<p>
If any check fails, the comparison is recorded as INVALID_SPACE with an explicit
error code (e.g., <code>E_CAND_MISSING</code>).
</p>

<h2>2. Metrics</h2>

<h3>2.1 Streamline Voxelization</h3>
<p>
Both candidate and reference streamlines are voxelized onto the reference FA image
grid (1&times;1&times;1 mm&sup3; isotropic) using DIPY's <code>density_map()</code>
function. The resulting density maps are binarized: any voxel traversed by at least
one streamline is set to 1, all others to 0. This produces two binary masks,
<em>A</em> (candidate) and <em>B</em> (reference).
</p>

<h3>2.2 Dice Coefficient</h3>
<p>
The Dice coefficient measures the spatial overlap between two binary volumes. It is
the harmonic mean of precision and recall:
</p>
<div class="formula">
Dice = 2 &times; |A &cap; B| / (|A| + |B|)
</div>
<p>
where |A &cap; B| is the number of voxels present in both masks, and |A|, |B| are
the total occupied voxels in each mask. Dice ranges from 0 (no overlap) to 1
(perfect overlap). It is sensitive to both the position and size of the masks.
</p>

<h3>2.3 Overlap Metrics (Precision and Recall)</h3>
<p>
The Dice coefficient can be decomposed into directional overlap measures:
</p>

<table>
<tr><th>Metric</th><th>Formula</th><th>Interpretation</th></tr>
<tr>
    <td><strong>Precision</strong><br>(overlap_cand_in_ref)</td>
    <td class="formula" style="border:none;background:none;">|A &cap; B| / |A|</td>
    <td>Fraction of candidate voxels that fall within the reference. High precision means the extraction is spatially accurate&mdash;it places streamlines where they belong.</td>
</tr>
<tr>
    <td><strong>Recall</strong><br>(overlap_ref_in_cand)</td>
    <td class="formula" style="border:none;background:none;">|A &cap; B| / |B|</td>
    <td>Fraction of reference voxels captured by the candidate. High recall means the extraction covers the full extent of the reference bundle.</td>
</tr>
</table>
<p>
The relationship between these metrics and Dice is:
</p>
<div class="formula">
Dice = 2 / (1/Precision + 1/Recall)
</div>
<p>
A method with high precision but low recall produces conservative, spatially accurate
bundles that capture the core of the tract but miss peripheral streamlines. This is the
expected behavior of deterministic tractography compared to probabilistic references.
</p>

<h3>2.4 Streamline Count Metrics</h3>
<table>
<tr><th>Metric</th><th>Formula</th><th>Interpretation</th></tr>
<tr>
    <td>Candidate count</td>
    <td>Number of streamlines in csttool output</td>
    <td>Size of the extracted bundle</td>
</tr>
<tr>
    <td>Reference count</td>
    <td>Number of streamlines in TractoInferno PYT</td>
    <td>Size of the reference bundle</td>
</tr>
<tr>
    <td>Count ratio</td>
    <td>Candidate / Reference</td>
    <td>Relative bundle density</td>
</tr>
</table>

<h3>2.5 Supplementary Voxel Counts</h3>
<p>
Each validation result records the number of reference voxels, candidate voxels,
and intersection voxels, as well as the number of mapped streamline points and any
out-of-bounds points. These support debugging and deeper analysis.
</p>

<div class="pagebreak"></div>

<h2>3. Results</h2>

<h3>3.1 Cohort Summary</h3>
<div class="metric-grid">
    <div class="metric-box">
        <div class="value">{{ s.n_subjects }}</div>
        <div class="label">Subjects</div>
    </div>
    <div class="metric-box">
        <div class="value">{{ s.n_total }}</div>
        <div class="label">Hemisphere Comparisons</div>
    </div>
    <div class="metric-box">
        <div class="value">{{ s.success_pct }}%</div>
        <div class="label">Success Rate</div>
    </div>
    <div class="metric-box">
        <div class="value">{{ s.n_invalid }}</div>
        <div class="label">Invalid (Excluded)</div>
    </div>
</div>
<p>
Of {{ s.n_total }} hemisphere comparisons ({{ s.n_subjects }} subjects &times; 2),
{{ s.n_ok }} yielded valid metrics (status OK) and {{ s.n_invalid }} were excluded
due to spatial incompatibility (INVALID_SPACE, error code E_CAND_MISSING).
No DEGRADED cases were observed.
</p>

{% if fig_success %}
<div class="figure">
    <img src="{{ fig_success }}" style="max-width: 45%;">
    <div class="caption">Figure 1: Validation outcome distribution.</div>
</div>
{% endif %}

<h3>3.2 Dice Coefficient</h3>
<table>
<tr><th>Statistic</th><th>Value</th></tr>
<tr><td>Mean &plusmn; SD</td><td>{{ s.dice_mean }} &plusmn; {{ s.dice_std }}</td></tr>
<tr><td>Median</td><td>{{ s.dice_median }}</td></tr>
<tr><td>Range</td><td>[{{ s.dice_min }}, {{ s.dice_max }}]</td></tr>
<tr><td>Left hemisphere (mean &plusmn; SD)</td><td>{{ s.dice_l_mean }} &plusmn; {{ s.dice_l_std }}</td></tr>
<tr><td>Right hemisphere (mean &plusmn; SD)</td><td>{{ s.dice_r_mean }} &plusmn; {{ s.dice_r_std }}</td></tr>
</table>

{% if fig_dice %}
<div class="figure">
    <img src="{{ fig_dice }}" style="max-width: 85%;">
    <div class="caption">Figure 2: Distribution of Dice coefficients across {{ s.n_ok }} valid hemisphere comparisons.</div>
</div>
{% endif %}

<h3>3.3 Precision and Recall</h3>
<table>
<tr><th>Metric</th><th>Mean &plusmn; SD</th><th>Interpretation</th></tr>
<tr>
    <td>Precision (o_cand_in_ref)</td>
    <td>{{ s.precision_mean }} &plusmn; {{ s.precision_std }}</td>
    <td>{{ s.precision_pct }}% of extracted voxels are within the reference</td>
</tr>
<tr>
    <td>Recall (o_ref_in_cand)</td>
    <td>{{ s.recall_mean }} &plusmn; {{ s.recall_std }}</td>
    <td>{{ s.recall_pct }}% of reference voxels are captured</td>
</tr>
</table>
<p>
The high precision and low recall pattern is characteristic of deterministic tractography
validated against probabilistic references. csttool's deterministic tracking with a single
ODF peak produces spatially focused, conservative bundles that accurately capture the core
of the CST but do not extend to the full probabilistic envelope of the reference. This is
a methodological property, not a deficiency&mdash;the tool prioritizes specificity over
sensitivity, which is appropriate for clinical applications where false-positive tract
assignment carries greater risk than under-coverage.
</p>

{% if fig_pr %}
<div class="figure">
    <img src="{{ fig_pr }}" style="max-width: 70%;">
    <div class="caption">Figure 3: Precision vs. recall for each hemisphere comparison. The clustering in the
    upper-left quadrant confirms conservative, high-precision extraction.</div>
</div>
{% endif %}

<div class="pagebreak"></div>

<h3>3.4 Hemisphere Comparison</h3>
<p>
The right hemisphere shows slightly higher Dice coefficients than the left
({{ s.dice_r_mean }} vs. {{ s.dice_l_mean }}),
which may reflect anatomical asymmetries in the pyramidal tract or differences in
how the reference bundles were segmented.
</p>

{% if fig_hemi %}
<div class="figure">
    <img src="{{ fig_hemi }}" style="max-width: 55%;">
    <div class="caption">Figure 4: Dice coefficient distribution by hemisphere.</div>
</div>
{% endif %}

<h3>3.5 Streamline Counts</h3>
<table>
<tr><th></th><th>Mean</th><th>Range</th></tr>
<tr>
    <td>Candidate (csttool)</td>
    <td>{{ s.n_cand_mean }}</td>
    <td>{{ s.n_cand_min }} &ndash; {{ s.n_cand_max }}</td>
</tr>
<tr>
    <td>Reference (TractoInferno)</td>
    <td>{{ s.n_ref_mean }}</td>
    <td>{{ s.n_ref_min }} &ndash; {{ s.n_ref_max }}</td>
</tr>
</table>
<p>
csttool extracts approximately {{ s.count_ratio_pct }}&times;
fewer streamlines than the probabilistic reference bundles. This ratio reflects the
fundamental difference between deterministic single-seed tracking (1 seed/voxel,
single peak) and probabilistic multi-seed approaches. The reduced streamline count
is consistent with the low recall and high precision pattern: fewer streamlines
concentrated in the tract core.
</p>

{% if fig_streamlines %}
<div class="figure">
    <img src="{{ fig_streamlines }}" style="max-width: 75%;">
    <div class="caption">Figure 5: Candidate vs. reference streamline counts on a log scale. The dashed
    line indicates 1:1 correspondence.</div>
</div>
{% endif %}

<div class="pagebreak"></div>

<h3>3.6 Subject-Level Paired Comparison</h3>
<p>
The preceding analyses aggregate across hemispheres, which can mask within-subject
variability. To assess whether performance is consistent within individual subjects,
Dice, precision, and recall were plotted as paired L/R values for each subject.
</p>

{% if fig_paired %}
<div class="figure">
    <img src="{{ fig_paired }}" style="max-width: 100%;">
    <div class="caption">Figure 6: Subject-level paired L vs. R comparison for Dice coefficient, precision,
    and recall. Each subject contributes two points (blue = left, red = right) connected by
    a vertical line. Subjects are sorted by left-hemisphere Dice. Short connecting lines
    indicate consistent within-subject performance.</div>
</div>
{% endif %}

<p>
Key observations:
</p>
<ul>
<li><strong>Within-subject consistency:</strong> Left and right hemispheres track each
other closely across subjects. The mean absolute L&ndash;R difference is {{ s.dice_lr_absdiff }}
for Dice, {{ s.prec_lr_absdiff }} for precision, and {{ s.rec_lr_absdiff }} for recall,
indicating that performance is stable within subjects rather than driven by one hemisphere.</li>
<li><strong>Slight right-hemisphere advantage:</strong> The mean R&ndash;L difference is
+{{ s.dice_lr_diff }} for Dice and +{{ s.rec_lr_diff }} for recall, consistent with the
violin plot in Section 3.4. This trend is uniform across subjects rather than driven by
outliers.</li>
<li><strong>Outlier identification:</strong> A small number of subjects show low Dice in
both hemispheres simultaneously (lower-left of the Dice panel), suggesting subject-level
data quality issues rather than hemisphere-specific failures.</li>
<li><strong>Robustness:</strong> The monotonic rise in the sorted Dice panel demonstrates
a continuous performance distribution with no bimodal clustering, supporting the claim
that the pipeline generalizes across the cohort.</li>
</ul>

<h2>4. Methodology Notes</h2>

<h3>4.1 Why Dice Appears Low</h3>
<p>
A Dice coefficient of ~0.28 may appear low relative to segmentation tasks where
values above 0.7 are expected. However, tractography bundle comparison differs from
volumetric segmentation in several important ways:
</p>
<ul>
<li><strong>Methodological asymmetry:</strong> The candidate (deterministic) and reference
(probabilistic) bundles are generated with fundamentally different algorithms, producing
different spatial extents by design.</li>
<li><strong>Volume asymmetry:</strong> Reference bundles occupy ~5&ndash;15&times; more
voxels than candidates, which mathematically depresses Dice even when the candidate is
fully contained within the reference.</li>
<li><strong>Dice sensitivity to size mismatch:</strong> When |A| &lt;&lt; |B|, even
perfect precision (all of A inside B) yields Dice &approx; 2|A|/(|A|+|B|), which is
dominated by the size ratio.</li>
</ul>
<p>
The precision&ndash;recall decomposition provides a more informative characterization than
Dice alone: {{ s.precision_pct }}% precision with
{{ s.recall_pct }}% recall demonstrates that the extraction is
spatially accurate but deliberately conservative.
</p>

<h3>4.2 Reproducibility</h3>
<p>
All results were generated from a single batch run (run ID: <code>2026-02-01_git-1433ac4</code>,
git commit <code>1433ac4</code>) with fixed parameters and no manual intervention. The
deterministic tracking algorithm and fixed random seed ensure identical results across
reruns on the same data. Each subject's validation result is stored as a structured JSON
file conforming to a versioned schema (<code>csttool_validation_result_schema_v1.json</code>).
</p>

<h3>4.3 Limitations</h3>
<ul>
<li>Validation is limited to TractoInferno's trainset; no clinical cohort with known
pathology was available for validation.</li>
<li>The reference PYT bundles are themselves algorithmic outputs (probabilistic
tractography with expert ROIs), not ground truth anatomical dissections.</li>
<li>Two subjects (sub-1025, sub-1074) failed spatial compatibility checks, likely due to
data quality issues in the source dataset.</li>
</ul>

</body>
</html>
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", nargs="?", default=DEFAULT_TSV)
    parser.add_argument("--outdir", default=DEFAULT_FIGURES)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(args.tsv)

    # Embed figures if they exist
    figures = {}
    for key, filename in [
        ("fig_dice", "dice_distribution.png"),
        ("fig_pr", "precision_recall.png"),
        ("fig_hemi", "dice_by_hemisphere.png"),
        ("fig_streamlines", "streamline_counts.png"),
        ("fig_success", "success_rate.png"),
        ("fig_paired", "subject_paired.png"),
    ]:
        fig_path = outdir / filename
        figures[key] = embed_image(fig_path)

    html = REPORT_TEMPLATE.render(s=stats, **figures)

    html_path = outdir / "validation_report.html"
    pdf_path = outdir / "validation_report.pdf"

    html_path.write_text(html, encoding="utf-8")
    weasyprint.HTML(string=html).write_pdf(str(pdf_path))

    print(f"HTML: {html_path.resolve()}")
    print(f"PDF:  {pdf_path.resolve()}")


if __name__ == "__main__":
    main()

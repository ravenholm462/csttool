#!/usr/bin/env python3
"""Aggregate pairwise comparison reports into a determinism summary.

Reads run_*_vs_run_*.json files and run_*_provenance.json files from a
determinism test directory and produces:
  - determinism_summary.json  (machine-readable)
  - determinism_summary.txt   (human-readable, thesis-ready)

Usage:
    python scripts/determinism_report.py /path/to/determinism_test_dir [--output PREFIX]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


VERDICT_RANK = {
    "BITWISE_IDENTICAL": 0,
    "TOLERANCE_IDENTICAL": 1,
    "NON_DETERMINISTIC": 2,
    "SKIPPED": -1,
}

SCIENTIFIC_CATEGORIES = ("tractograms", "scalar_maps", "metrics")
ALL_CATEGORIES = ("tractograms", "scalar_maps", "metrics", "reports")


def worst_verdict(*verdicts: str) -> str:
    """Return the worst (least deterministic) verdict."""
    ranked = [v for v in verdicts if v != "SKIPPED"]
    if not ranked:
        return "SKIPPED"
    return max(ranked, key=lambda v: VERDICT_RANK.get(v, 99))


def load_comparisons(test_dir: Path) -> list[dict]:
    """Load all pairwise comparison JSON files."""
    comp_dir = test_dir / "comparisons"
    if not comp_dir.exists():
        raise FileNotFoundError(f"No comparisons/ directory in {test_dir}")

    files = sorted(comp_dir.glob("run_*_vs_run_*.json"))
    if not files:
        raise FileNotFoundError(f"No comparison files found in {comp_dir}")

    comparisons = []
    for f in files:
        with open(f) as fh:
            comparisons.append(json.load(fh))
    return comparisons


def load_provenances(test_dir: Path) -> list[dict]:
    """Load per-run provenance files."""
    files = sorted(test_dir.glob("run_*_provenance.json"))
    provenances = []
    for f in files:
        with open(f) as fh:
            provenances.append(json.load(fh))
    return provenances


def check_provenance_consistency(provenances: list[dict]) -> tuple[bool, list[str]]:
    """Check if all runs had the same environment."""
    if len(provenances) < 2:
        return True, []

    ref = provenances[0]
    warnings = []

    for i, prov in enumerate(provenances[1:], 1):
        for key in ("git_commit", "csttool_version", "python_version"):
            ref_val = ref.get(key)
            prov_val = prov.get(key)
            if ref_val != prov_val:
                warnings.append(f"run_0 vs run_{i}: {key} differs ({ref_val} vs {prov_val})")

        ref_deps = ref.get("dependencies", {})
        prov_deps = prov.get("dependencies", {})
        for pkg in set(ref_deps.keys()) | set(prov_deps.keys()):
            if ref_deps.get(pkg) != prov_deps.get(pkg):
                warnings.append(
                    f"run_0 vs run_{i}: {pkg} version differs "
                    f"({ref_deps.get(pkg)} vs {prov_deps.get(pkg)})"
                )

    return len(warnings) == 0, warnings


def aggregate(comparisons: list[dict], provenances: list[dict]) -> dict:
    """Build the aggregate summary from pairwise comparisons."""
    n_comparisons = len(comparisons)

    # Extract metadata from first comparison
    first = comparisons[0]
    subject_id = first.get("comparison", {}).get("subject_id", "unknown")

    # Per-category aggregation
    per_category = {}
    for cat_name in ALL_CATEGORIES:
        verdicts = []
        for comp in comparisons:
            cat = comp.get("categories", {}).get(cat_name, {})
            v = cat.get("verdict", "SKIPPED")
            verdicts.append(v)

        n_bitwise = sum(1 for v in verdicts if v == "BITWISE_IDENTICAL")
        n_tolerance = sum(1 for v in verdicts if v == "TOLERANCE_IDENTICAL")
        n_nondet = sum(1 for v in verdicts if v == "NON_DETERMINISTIC")

        per_category[cat_name] = {
            "verdict": worst_verdict(*verdicts),
            "n_bitwise": n_bitwise,
            "n_tolerance": n_tolerance,
            "n_non_deterministic": n_nondet,
            "n_comparisons": n_comparisons,
        }

    # Overall verdict (scientific categories only)
    scientific_verdicts = [per_category[c]["verdict"] for c in SCIENTIFIC_CATEGORIES]
    overall = worst_verdict(*scientific_verdicts)

    # Provenance consistency
    prov_consistent, prov_warnings = check_provenance_consistency(provenances)

    # Collect max deviations across all comparisons
    max_deviations = _collect_max_deviations(comparisons)

    # Pairwise matrix
    pairwise = []
    for comp in comparisons:
        pairwise.append({
            "dir_a": comp.get("comparison", {}).get("dir_a", ""),
            "dir_b": comp.get("comparison", {}).get("dir_b", ""),
            "verdict": comp.get("verdict", "unknown"),
        })

    # N runs: infer from provenance count or comparison structure
    n_runs = len(provenances) if provenances else "unknown"

    # Thread settings from first provenance
    thread_settings = {}
    if provenances:
        env = provenances[0].get("environment_variables", {})
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
                     "PYTHONHASHSEED"):
            if var in env:
                thread_settings[var] = env[var]

    return {
        "test_metadata": {
            "n_runs": n_runs,
            "n_pairwise_comparisons": n_comparisons,
            "subject_id": subject_id,
            "seed": 42,
            "timestamp": datetime.now().isoformat(),
            "csttool_version": provenances[0].get("csttool_version", "unknown") if provenances else "unknown",
            "environment_controlled": bool(thread_settings),
            "thread_settings": thread_settings,
            "provenance_consistent_across_runs": prov_consistent,
            "provenance_warnings": prov_warnings,
        },
        "overall_verdict": overall,
        "per_category": per_category,
        "max_deviations": max_deviations,
        "pairwise_matrix": pairwise,
    }


def _collect_max_deviations(comparisons: list[dict]) -> dict:
    """Collect maximum observed deviations across all pairwise comparisons."""
    deviations = {
        "max_coordinate_diff": 0.0,
        "max_voxel_diff": 0.0,
        "max_metric_field_diffs": 0,
    }

    for comp in comparisons:
        cats = comp.get("categories", {})

        # Tractograms
        trk_files = cats.get("tractograms", {}).get("files", {})
        for fname, result in trk_files.items():
            d = result.get("details", {})
            val = d.get("max_absolute_coordinate_diff")
            if val is not None and val > deviations["max_coordinate_diff"]:
                deviations["max_coordinate_diff"] = val

        # Scalar maps
        nifti_files = cats.get("scalar_maps", {}).get("files", {})
        for fname, result in nifti_files.items():
            d = result.get("details", {})
            val = d.get("max_absolute_voxel_diff")
            if val is not None and val > deviations["max_voxel_diff"]:
                deviations["max_voxel_diff"] = val

        # Metrics
        metric_files = cats.get("metrics", {}).get("files", {})
        for fname, result in metric_files.items():
            d = result.get("details", {})
            n_diffs = len(d.get("unexpected_diffs", {})) + len(d.get("column_diffs", {}))
            if n_diffs > deviations["max_metric_field_diffs"]:
                deviations["max_metric_field_diffs"] = n_diffs

    return deviations


# ---------------------------------------------------------------------------
# Text report generation
# ---------------------------------------------------------------------------

def generate_text_report(summary: dict) -> str:
    """Generate human-readable determinism report for thesis."""
    meta = summary["test_metadata"]
    lines = []

    # Section 1: Experimental setup
    lines.append("=" * 72)
    lines.append("DETERMINISM VALIDATION REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1. EXPERIMENTAL SETUP")
    lines.append("-" * 40)
    lines.append(f"  Subject:            {meta['subject_id']}")
    lines.append(f"  Number of runs:     {meta['n_runs']}")
    lines.append(f"  Pairwise tests:     {meta['n_pairwise_comparisons']}")
    lines.append(f"  Tracking seed:      {meta['seed']}")
    lines.append(f"  csttool version:    {meta['csttool_version']}")
    lines.append(f"  Date:               {meta['timestamp'][:10]}")

    if meta.get("thread_settings"):
        lines.append(f"  Environment control: YES")
        for var, val in meta["thread_settings"].items():
            lines.append(f"    {var}={val}")
    else:
        lines.append(f"  Environment control: no thread pinning recorded")

    prov_ok = meta.get("provenance_consistent_across_runs", False)
    lines.append(f"  Provenance match:   {'YES' if prov_ok else 'NO'}")
    for w in meta.get("provenance_warnings", []):
        lines.append(f"    WARNING: {w}")

    # Section 2: Overall verdict
    lines.append("")
    lines.append("2. OVERALL VERDICT")
    lines.append("-" * 40)
    verdict = summary["overall_verdict"]
    if verdict == "BITWISE_IDENTICAL":
        lines.append(f"  {verdict}")
        lines.append(f"  All scientific outputs are byte-for-byte identical across all runs.")
    elif verdict == "TOLERANCE_IDENTICAL":
        lines.append(f"  {verdict}")
        lines.append(f"  All scientific outputs are numerically equivalent within defined")
        lines.append(f"  tolerances, but not bitwise identical.")
    else:
        lines.append(f"  {verdict}")
        lines.append(f"  One or more scientific outputs differ beyond acceptable tolerances.")

    # Section 3: Per-category summary
    lines.append("")
    lines.append("3. PER-CATEGORY RESULTS")
    lines.append("-" * 40)
    lines.append(f"  {'Category':<20} {'Verdict':<25} {'Bitwise':>8} {'Tol.':>6} {'Fail':>6}")
    lines.append(f"  {'-'*20} {'-'*25} {'-'*8} {'-'*6} {'-'*6}")

    for cat_name in ALL_CATEGORIES:
        cat = summary["per_category"].get(cat_name, {})
        v = cat.get("verdict", "N/A")
        nb = cat.get("n_bitwise", 0)
        nt = cat.get("n_tolerance", 0)
        nf = cat.get("n_non_deterministic", 0)
        marker = "*" if cat_name in SCIENTIFIC_CATEGORIES else " "
        lines.append(f" {marker}{cat_name:<20} {v:<25} {nb:>8} {nt:>6} {nf:>6}")

    lines.append("")
    lines.append("  * = scientific category (affects overall verdict)")

    # Section 4: Maximum deviations
    devs = summary.get("max_deviations", {})
    lines.append("")
    lines.append("4. MAXIMUM OBSERVED DEVIATIONS")
    lines.append("-" * 40)
    lines.append(f"  Coordinate (mm):    {devs.get('max_coordinate_diff', 0):.2e}")
    lines.append(f"  Scalar voxel:       {devs.get('max_voxel_diff', 0):.2e}")
    lines.append(f"  Metric fields:      {devs.get('max_metric_field_diffs', 0)} unexpected diffs")

    # Section 5: Pairwise matrix
    lines.append("")
    lines.append("5. PAIRWISE COMPARISON MATRIX")
    lines.append("-" * 40)
    for pair in summary.get("pairwise_matrix", []):
        a = Path(pair["dir_a"]).name
        b = Path(pair["dir_b"]).name
        lines.append(f"  {a} vs {b}: {pair['verdict']}")

    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate determinism comparison reports into a summary.",
    )
    parser.add_argument("test_dir", type=Path,
                        help="Root directory of the determinism test")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output prefix (produces PREFIX.json and PREFIX.txt). "
                             "Default: <test_dir>/determinism_summary")
    args = parser.parse_args()

    output_prefix = args.output or (args.test_dir / "determinism_summary")

    # Load data
    comparisons = load_comparisons(args.test_dir)
    provenances = load_provenances(args.test_dir)

    print(f"Loaded {len(comparisons)} pairwise comparisons, {len(provenances)} provenance records")

    # Aggregate
    summary = aggregate(comparisons, provenances)

    # Write JSON
    json_path = Path(f"{output_prefix}.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"JSON summary: {json_path}")

    # Write text report
    txt_path = Path(f"{output_prefix}.txt")
    text = generate_text_report(summary)
    txt_path.write_text(text)
    print(f"Text summary: {txt_path}")

    # Print to stdout
    print()
    print(text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare two csttool output directories for determinism validation.

Produces a structured JSON report indicating whether outputs are bitwise
identical, identical within tolerance, or non-deterministic.

Usage:
    python scripts/compare_outputs.py DIR_A DIR_B --subject-id sub-1204 [--output report.json]
    python scripts/compare_outputs.py DIR_A --self-test --subject-id sub-1204
    python scripts/compare_outputs.py DIR_A DIR_B --provenance-only

Requires: numpy, nibabel, dipy, pandas (all csttool dependencies)
"""

import argparse
import json
import gc
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from dipy.io.streamline import load_tractogram

from csttool.reproducibility.tolerance import (
    TOLERANCE_STREAMLINE_COUNT,
    TOLERANCE_COORDINATES_RTOL,
    TOLERANCE_COORDINATES_ATOL,
    TOLERANCE_FA_MEAN_RTOL,
    TOLERANCE_FA_MEAN_ATOL,
    TOLERANCE_MD_MEAN_RTOL,
    TOLERANCE_MD_MEAN_ATOL,
    compute_metric_difference,
)
from csttool.reproducibility.provenance import get_provenance_dict


# ---------------------------------------------------------------------------
# Verdict levels
# ---------------------------------------------------------------------------

BITWISE_IDENTICAL = "BITWISE_IDENTICAL"
TOLERANCE_IDENTICAL = "TOLERANCE_IDENTICAL"
NON_DETERMINISTIC = "NON_DETERMINISTIC"

# Scientific output categories (affect the overall verdict)
SCIENTIFIC_CATEGORIES = ("tractograms", "scalar_maps", "metrics")

# ---------------------------------------------------------------------------
# Expected file manifest
# ---------------------------------------------------------------------------

EXPECTED_FILES = {
    # Tractograms
    "whole_brain_trk":    "tracking/tractograms/{sid}*_whole_brain.trk",
    "cst_left_trk":       "extraction/trk/{sid}*_cst_left.trk",
    "cst_right_trk":      "extraction/trk/{sid}*_cst_right.trk",
    "cst_bilateral_trk":  "extraction/trk/{sid}*_cst_bilateral.trk",
    # Scalar maps
    "fa_map":             "tracking/scalar_maps/{sid}*_fa.nii.gz",
    "md_map":             "tracking/scalar_maps/{sid}*_md.nii.gz",
    "rd_map":             "tracking/scalar_maps/{sid}*_rd.nii.gz",
    "ad_map":             "tracking/scalar_maps/{sid}*_ad.nii.gz",
    # Metrics
    "bilateral_metrics":  "metrics/{sid}*_bilateral_metrics.json",
    "metrics_summary":    "metrics/{sid}*_metrics_summary.csv",
    # Reports
    "pipeline_report":    "{sid}*_pipeline_report.json",
    "tracking_report":    "tracking/logs/{sid}*_tracking_report.json",
    "extraction_report":  "extraction/logs/{sid}*_cst_extraction_report.json",
    "registration_report": "extraction/logs/{sid}*_registration_report.json",
}

FILE_CATEGORIES = {
    "tractograms":  ["whole_brain_trk", "cst_left_trk", "cst_right_trk", "cst_bilateral_trk"],
    "scalar_maps":  ["fa_map", "md_map", "rd_map", "ad_map"],
    "metrics":      ["bilateral_metrics", "metrics_summary"],
    "reports":      ["pipeline_report", "tracking_report", "extraction_report", "registration_report"],
}

# JSON paths expected to differ between runs (timestamps, timing)
EXPECTED_DIFF_PATHS = {
    "pipeline_report": {
        "execution.start_time", "execution.end_time",
        "execution.total_seconds", "step_times",
        "step_times.check", "step_times.import", "step_times.preprocess",
        "step_times.track", "step_times.extract", "step_times.metrics",
    },
    "tracking_report": {
        "processing_info.date", "processing_date",
        # Absolute output paths differ across runs by definition
        "output_files.tractogram", "output_files.fa_map", "output_files.md_map",
        "output_files.rd_map", "output_files.ad_map",
    },
    "extraction_report": {
        "processing_date",
        # Absolute output paths differ across runs by definition
        "output_files.cst_left", "output_files.cst_right", "output_files.cst_combined",
    },
    "registration_report": {
        "processing_date",
        # Absolute paths to outputs and inputs differ across runs by definition
        "outputs.warped_template", "outputs.qc_before", "outputs.qc_after",
        "subject.fa_path",
    },
    "bilateral_metrics": {
        "processing_date",
    },
}


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------

def build_manifest(output_dir: Path, subject_id: str) -> dict[str, Path | None]:
    """Build a logical file manifest for one output directory.

    Returns a dict mapping logical name -> resolved Path (or None if missing).
    Prefers single-underscore variants when multiple matches exist.
    """
    manifest = {}
    for logical_name, pattern_template in EXPECTED_FILES.items():
        pattern = pattern_template.replace("{sid}", subject_id)
        # glob from the output directory
        parts = pattern.split("/")
        if len(parts) > 1:
            search_dir = output_dir / "/".join(parts[:-1])
            glob_pattern = parts[-1]
        else:
            search_dir = output_dir
            glob_pattern = parts[0]

        if not search_dir.exists():
            manifest[logical_name] = None
            continue

        matches = sorted(search_dir.glob(glob_pattern))
        if not matches:
            manifest[logical_name] = None
        elif len(matches) == 1:
            manifest[logical_name] = matches[0]
        else:
            # Prefer single-underscore variant
            single = [m for m in matches if f"{subject_id}_" in m.name and f"{subject_id}__" not in m.name]
            manifest[logical_name] = single[0] if single else matches[0]

    return manifest


def compare_manifests(manifest_a: dict, manifest_b: dict) -> dict:
    """Compare two manifests for structural consistency."""
    all_keys = set(manifest_a.keys()) | set(manifest_b.keys())
    in_both = []
    missing_in_a = []
    missing_in_b = []

    for key in sorted(all_keys):
        a_present = manifest_a.get(key) is not None
        b_present = manifest_b.get(key) is not None
        if a_present and b_present:
            in_both.append(key)
        elif not a_present:
            missing_in_a.append(key)
        else:
            missing_in_b.append(key)

    return {
        "files_in_both": in_both,
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
    }


# ---------------------------------------------------------------------------
# Tractogram comparison
# ---------------------------------------------------------------------------

def compare_trk(path_a: Path, path_b: Path) -> dict:
    """Compare two .trk tractogram files."""
    rtol = TOLERANCE_COORDINATES_RTOL
    atol = TOLERANCE_COORDINATES_ATOL

    # Load A
    sft_a = load_tractogram(str(path_a), reference="same", bbox_valid_check=False)
    n_a = len(sft_a.streamlines)
    data_a = sft_a.streamlines.get_data().copy()
    offsets_a = sft_a.streamlines._offsets.copy()
    lengths_a = sft_a.streamlines._lengths.copy()
    affine_a = sft_a.affine.copy()
    del sft_a
    gc.collect()

    # Load B
    sft_b = load_tractogram(str(path_b), reference="same", bbox_valid_check=False)
    n_b = len(sft_b.streamlines)
    data_b = sft_b.streamlines.get_data().copy()
    offsets_b = sft_b.streamlines._offsets.copy()
    lengths_b = sft_b.streamlines._lengths.copy()
    affine_b = sft_b.affine.copy()
    del sft_b
    gc.collect()

    # Structural checks
    count_match = n_a == n_b
    offsets_match = np.array_equal(offsets_a, offsets_b)
    lengths_match = np.array_equal(lengths_a, lengths_b)
    affine_match = np.array_equal(affine_a, affine_b)

    # Coordinate checks
    shape_match = data_a.shape == data_b.shape
    if shape_match:
        coords_bitwise = bool(np.array_equal(data_a, data_b))
        coords_within_tol = bool(np.allclose(data_a, data_b, rtol=rtol, atol=atol))
        max_abs_diff = float(np.max(np.abs(data_a - data_b)))
    else:
        coords_bitwise = False
        coords_within_tol = False
        max_abs_diff = None

    # Determine verdict
    all_structural = count_match and offsets_match and lengths_match and affine_match
    if all_structural and coords_bitwise:
        status = BITWISE_IDENTICAL
        reason = f"{n_a} streamlines, {data_a.shape[0]:,} coordinate points bitwise identical"
    elif all_structural and coords_within_tol:
        status = TOLERANCE_IDENTICAL
        reason = f"coordinates within tolerance (max diff: {max_abs_diff:.2e})"
    else:
        parts = []
        if not count_match:
            parts.append(f"streamline count mismatch ({n_a} vs {n_b})")
        if not offsets_match:
            parts.append("streamline offsets differ")
        if not lengths_match:
            parts.append("streamline lengths differ")
        if not affine_match:
            parts.append("affine matrices differ")
        if shape_match and not coords_within_tol:
            parts.append(f"coordinates differ beyond tolerance (max diff: {max_abs_diff:.2e})")
        status = NON_DETERMINISTIC
        reason = "; ".join(parts) if parts else "shape mismatch"

    return {
        "status": status,
        "reason": reason,
        "details": {
            "n_streamlines_a": n_a,
            "n_streamlines_b": n_b,
            "total_points_a": int(data_a.shape[0]),
            "total_points_b": int(data_b.shape[0]),
            "count_match": count_match,
            "offsets_match": offsets_match,
            "lengths_match": lengths_match,
            "affine_match": affine_match,
            "coords_bitwise_identical": coords_bitwise,
            "coords_within_tolerance": coords_within_tol,
            "max_absolute_coordinate_diff": max_abs_diff,
            "tolerances": {"rtol": rtol, "atol": atol},
        },
    }


# ---------------------------------------------------------------------------
# Scalar map comparison
# ---------------------------------------------------------------------------

def compare_nifti(path_a: Path, path_b: Path) -> dict:
    """Compare two NIfTI scalar map files."""
    rtol = 1e-8
    atol = 1e-8

    img_a = nib.load(str(path_a))
    img_b = nib.load(str(path_b))

    hdr_a = img_a.header
    hdr_b = img_b.header

    # Spatial metadata
    affine_match = bool(np.allclose(img_a.affine, img_b.affine))
    shape_match = img_a.shape == img_b.shape
    qform_match = int(hdr_a["qform_code"]) == int(hdr_b["qform_code"])
    sform_match = int(hdr_a["sform_code"]) == int(hdr_b["sform_code"])
    pixdim_match = bool(np.array_equal(hdr_a["pixdim"], hdr_b["pixdim"]))

    spatial_metadata_match = affine_match and qform_match and sform_match and pixdim_match

    # Voxel data
    if shape_match:
        data_a = img_a.get_fdata()
        data_b = img_b.get_fdata()
        voxel_bitwise = bool(np.array_equal(data_a, data_b))
        voxel_within_tol = bool(np.allclose(data_a, data_b, rtol=rtol, atol=atol))
        max_abs_diff = float(np.max(np.abs(data_a - data_b)))
        n_differing = int(np.sum(~np.isclose(data_a, data_b, rtol=rtol, atol=atol)))
    else:
        voxel_bitwise = False
        voxel_within_tol = False
        max_abs_diff = None
        n_differing = None

    # Determine verdict
    if shape_match and spatial_metadata_match and voxel_bitwise:
        status = BITWISE_IDENTICAL
        reason = f"all {int(np.prod(img_a.shape)):,} voxels bitwise identical"
    elif shape_match and voxel_within_tol:
        status = TOLERANCE_IDENTICAL
        extra = ""
        if not spatial_metadata_match:
            diffs = []
            if not qform_match:
                diffs.append("qform_code")
            if not sform_match:
                diffs.append("sform_code")
            if not pixdim_match:
                diffs.append("pixdim")
            if not affine_match:
                diffs.append("affine")
            extra = f"; header diffs: {', '.join(diffs)}"
        reason = f"voxel data within tolerance (max diff: {max_abs_diff:.2e}, {n_differing} voxels differ){extra}"
    else:
        parts = []
        if not shape_match:
            parts.append(f"shape mismatch ({img_a.shape} vs {img_b.shape})")
        elif not voxel_within_tol:
            parts.append(f"voxel data differs beyond tolerance (max diff: {max_abs_diff:.2e})")
        status = NON_DETERMINISTIC
        reason = "; ".join(parts)

    return {
        "status": status,
        "reason": reason,
        "details": {
            "shape_a": list(img_a.shape),
            "shape_b": list(img_b.shape),
            "shape_match": shape_match,
            "spatial_metadata": {
                "affine_match": affine_match,
                "qform_code_match": qform_match,
                "sform_code_match": sform_match,
                "pixdim_match": pixdim_match,
            },
            "voxel_bitwise_identical": voxel_bitwise,
            "voxel_within_tolerance": voxel_within_tol,
            "max_absolute_voxel_diff": max_abs_diff,
            "n_differing_voxels": n_differing,
            "tolerances": {"rtol": rtol, "atol": atol},
        },
    }


# ---------------------------------------------------------------------------
# Metrics comparison
# ---------------------------------------------------------------------------

def _flatten_json(obj, prefix=""):
    """Flatten a nested dict into dot-separated keys."""
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else k
            items.update(_flatten_json(v, new_key))
    else:
        items[prefix] = obj
    return items


def _filter_expected_diffs(flat: dict, expected_paths: set) -> dict:
    """Remove keys that match expected-to-differ paths."""
    filtered = {}
    for key, val in flat.items():
        skip = False
        for ep in expected_paths:
            if key == ep or key.startswith(ep + "."):
                skip = True
                break
        if not skip:
            filtered[key] = val
    return filtered


def compare_metrics_json(path_a: Path, path_b: Path, logical_name: str) -> dict:
    """Compare two metrics/report JSON files."""
    with open(path_a) as f:
        data_a = json.load(f)
    with open(path_b) as f:
        data_b = json.load(f)

    expected_paths = EXPECTED_DIFF_PATHS.get(logical_name, set())

    flat_a = _flatten_json(data_a)
    flat_b = _flatten_json(data_b)

    # Check for schema drift: warn if expected-diff paths don't exist
    schema_warnings = []
    for ep in expected_paths:
        found_a = any(k == ep or k.startswith(ep + ".") for k in flat_a)
        found_b = any(k == ep or k.startswith(ep + ".") for k in flat_b)
        if not found_a and not found_b:
            schema_warnings.append(f"expected-diff path '{ep}' not found in either file")

    # Filter out expected diffs
    det_a = _filter_expected_diffs(flat_a, expected_paths)
    det_b = _filter_expected_diffs(flat_b, expected_paths)

    # Compare deterministic fields
    all_keys = set(det_a.keys()) | set(det_b.keys())
    unexpected_diffs = {}
    for key in sorted(all_keys):
        a_has = key in det_a
        b_has = key in det_b
        if not a_has or not b_has:
            unexpected_diffs[key] = {"a": det_a.get(key), "b": det_b.get(key), "issue": "missing in one"}
            continue
        va = det_a[key]
        vb = det_b[key]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if not np.isclose(va, vb, rtol=1e-8, atol=1e-12):
                unexpected_diffs[key] = {"a": va, "b": vb, "diff": abs(va - vb)}
        elif va != vb:
            unexpected_diffs[key] = {"a": str(va), "b": str(vb)}

    if not unexpected_diffs:
        # Check bitwise: are all deterministic fields exactly equal?
        bitwise = all(det_a.get(k) == det_b.get(k) for k in all_keys
                      if k in det_a and k in det_b)
        status = BITWISE_IDENTICAL if bitwise else TOLERANCE_IDENTICAL
        reason = f"all {len(all_keys)} deterministic fields match"
    else:
        status = NON_DETERMINISTIC
        reason = f"{len(unexpected_diffs)} unexpected field differences"

    return {
        "status": status,
        "reason": reason,
        "details": {
            "total_fields_compared": len(all_keys),
            "expected_diff_paths_skipped": sorted(expected_paths),
            "unexpected_diffs": unexpected_diffs,
            "schema_warnings": schema_warnings,
        },
    }


def compare_metrics_csv(path_a: Path, path_b: Path) -> dict:
    """Compare two metrics summary CSV files."""
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    # Drop processing_date column if present
    for col in ["processing_date"]:
        if col in df_a.columns:
            df_a = df_a.drop(columns=[col])
        if col in df_b.columns:
            df_b = df_b.drop(columns=[col])

    # Check structure
    if list(df_a.columns) != list(df_b.columns):
        return {
            "status": NON_DETERMINISTIC,
            "reason": f"column mismatch: {list(df_a.columns)} vs {list(df_b.columns)}",
            "details": {"columns_a": list(df_a.columns), "columns_b": list(df_b.columns)},
        }

    if len(df_a) != len(df_b):
        return {
            "status": NON_DETERMINISTIC,
            "reason": f"row count mismatch: {len(df_a)} vs {len(df_b)}",
            "details": {},
        }

    # Compare numeric and non-numeric columns
    diffs = {}
    bitwise = True
    for col in df_a.columns:
        if pd.api.types.is_numeric_dtype(df_a[col]):
            arr_a = df_a[col].values
            arr_b = df_b[col].values
            if not np.array_equal(arr_a, arr_b):
                bitwise = False
            if not np.allclose(arr_a, arr_b, rtol=1e-8, atol=1e-9, equal_nan=True):
                diffs[col] = {"max_diff": float(np.nanmax(np.abs(arr_a - arr_b)))}
        else:
            if not df_a[col].equals(df_b[col]):
                bitwise = False
                mismatches = (df_a[col] != df_b[col]).sum()
                diffs[col] = {"mismatches": int(mismatches)}

    if not diffs:
        status = BITWISE_IDENTICAL if bitwise else TOLERANCE_IDENTICAL
        reason = f"all {len(df_a.columns)} columns match across {len(df_a)} rows"
    else:
        status = NON_DETERMINISTIC
        reason = f"{len(diffs)} columns differ: {', '.join(diffs.keys())}"

    return {
        "status": status,
        "reason": reason,
        "details": {
            "n_columns": len(df_a.columns),
            "n_rows": len(df_a),
            "column_diffs": diffs,
        },
    }


# ---------------------------------------------------------------------------
# Visualization comparison (non-scientific, info only)
# ---------------------------------------------------------------------------

def compare_visualizations(dir_a: Path, dir_b: Path) -> dict:
    """Compare visualization PNGs by existence and dimensions."""
    pngs_a = {p.relative_to(dir_a) for p in dir_a.rglob("*.png")}
    pngs_b = {p.relative_to(dir_b) for p in dir_b.rglob("*.png")}

    only_a = sorted(str(p) for p in pngs_a - pngs_b)
    only_b = sorted(str(p) for p in pngs_b - pngs_a)
    common = sorted(str(p) for p in pngs_a & pngs_b)

    dim_mismatches = []
    for rel in pngs_a & pngs_b:
        try:
            from PIL import Image
            img_a = Image.open(dir_a / rel)
            img_b = Image.open(dir_b / rel)
            if img_a.size != img_b.size:
                dim_mismatches.append({
                    "file": str(rel),
                    "size_a": list(img_a.size),
                    "size_b": list(img_b.size),
                })
        except ImportError:
            break  # PIL not available, skip dimension check

    return {
        "common_count": len(common),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "dimension_mismatches": dim_mismatches,
    }


# ---------------------------------------------------------------------------
# Provenance comparison
# ---------------------------------------------------------------------------

def compare_provenance(dir_a: Path, dir_b: Path) -> dict:
    """Compare provenance of two runs (from pipeline reports or current env)."""
    current = get_provenance_dict()
    return {
        "current_environment": current,
        "note": "Provenance per-run captured by run_determinism_test.sh",
    }


# ---------------------------------------------------------------------------
# Main comparison driver
# ---------------------------------------------------------------------------

def compare_directories(dir_a: Path, dir_b: Path, subject_id: str) -> dict:
    """Full pairwise comparison of two csttool output directories."""
    # Build manifests
    manifest_a = build_manifest(dir_a, subject_id)
    manifest_b = build_manifest(dir_b, subject_id)
    manifest_check = compare_manifests(manifest_a, manifest_b)

    categories = {}

    # --- Tractograms ---
    trk_results = {}
    for logical_name in FILE_CATEGORIES["tractograms"]:
        pa, pb = manifest_a.get(logical_name), manifest_b.get(logical_name)
        if pa and pb:
            print(f"  Comparing {logical_name} ...", flush=True)
            trk_results[logical_name] = compare_trk(pa, pb)
        elif pa is None and pb is None:
            trk_results[logical_name] = {"status": "SKIPPED", "reason": "missing in both runs"}
        else:
            trk_results[logical_name] = {"status": NON_DETERMINISTIC, "reason": "missing in one run"}

    categories["tractograms"] = {
        "verdict": _category_verdict(trk_results),
        "files": trk_results,
    }

    # --- Scalar maps ---
    nifti_results = {}
    for logical_name in FILE_CATEGORIES["scalar_maps"]:
        pa, pb = manifest_a.get(logical_name), manifest_b.get(logical_name)
        if pa and pb:
            print(f"  Comparing {logical_name} ...", flush=True)
            nifti_results[logical_name] = compare_nifti(pa, pb)
        elif pa is None and pb is None:
            nifti_results[logical_name] = {"status": "SKIPPED", "reason": "missing in both runs"}
        else:
            nifti_results[logical_name] = {"status": NON_DETERMINISTIC, "reason": "missing in one run"}

    categories["scalar_maps"] = {
        "verdict": _category_verdict(nifti_results),
        "files": nifti_results,
    }

    # --- Metrics ---
    metric_results = {}
    for logical_name in FILE_CATEGORIES["metrics"]:
        pa, pb = manifest_a.get(logical_name), manifest_b.get(logical_name)
        if pa and pb:
            print(f"  Comparing {logical_name} ...", flush=True)
            if logical_name == "metrics_summary":
                metric_results[logical_name] = compare_metrics_csv(pa, pb)
            else:
                metric_results[logical_name] = compare_metrics_json(pa, pb, logical_name)
        elif pa is None and pb is None:
            metric_results[logical_name] = {"status": "SKIPPED", "reason": "missing in both runs"}
        else:
            metric_results[logical_name] = {"status": NON_DETERMINISTIC, "reason": "missing in one run"}

    categories["metrics"] = {
        "verdict": _category_verdict(metric_results),
        "files": metric_results,
    }

    # --- Reports ---
    report_results = {}
    for logical_name in FILE_CATEGORIES["reports"]:
        pa, pb = manifest_a.get(logical_name), manifest_b.get(logical_name)
        if pa and pb:
            print(f"  Comparing {logical_name} ...", flush=True)
            report_results[logical_name] = compare_metrics_json(pa, pb, logical_name)
        elif pa is None and pb is None:
            report_results[logical_name] = {"status": "SKIPPED", "reason": "missing in both runs"}
        else:
            report_results[logical_name] = {"status": NON_DETERMINISTIC, "reason": "missing in one run"}

    categories["reports"] = {
        "verdict": _category_verdict(report_results),
        "files": report_results,
    }

    # --- Visualizations (info only) ---
    categories["visualizations"] = compare_visualizations(dir_a, dir_b)

    # --- Overall verdict (only scientific categories matter) ---
    overall = _overall_verdict(categories)

    return {
        "comparison": {
            "dir_a": str(dir_a.resolve()),
            "dir_b": str(dir_b.resolve()),
            "timestamp": datetime.now().isoformat(),
            "subject_id": subject_id,
        },
        "verdict": overall,
        "categories": categories,
        "manifest_check": manifest_check,
        "tolerances_used": {
            "source": "csttool.reproducibility.tolerance",
            "coordinates_rtol": TOLERANCE_COORDINATES_RTOL,
            "coordinates_atol": TOLERANCE_COORDINATES_ATOL,
            "scalar_map_rtol": 1e-8,
            "scalar_map_atol": 1e-8,
            "justification": "1e-6 mm = 1 micrometer, 3 orders below imaging resolution; "
                             "scalar tolerances from empirical reproducibility testing",
        },
        "provenance": get_provenance_dict(),
    }


def _category_verdict(file_results: dict) -> str:
    """Determine verdict for a category from its per-file results."""
    statuses = [r["status"] for r in file_results.values() if r["status"] != "SKIPPED"]
    if not statuses:
        return "SKIPPED"
    if any(s == NON_DETERMINISTIC for s in statuses):
        return NON_DETERMINISTIC
    if all(s == BITWISE_IDENTICAL for s in statuses):
        return BITWISE_IDENTICAL
    return TOLERANCE_IDENTICAL


def _overall_verdict(categories: dict) -> str:
    """Determine overall verdict from scientific categories only."""
    verdicts = []
    for cat_name in SCIENTIFIC_CATEGORIES:
        cat = categories.get(cat_name, {})
        v = cat.get("verdict", "SKIPPED")
        if v != "SKIPPED":
            verdicts.append(v)

    if not verdicts:
        return "SKIPPED"
    if any(v == NON_DETERMINISTIC for v in verdicts):
        return NON_DETERMINISTIC
    if all(v == BITWISE_IDENTICAL for v in verdicts):
        return BITWISE_IDENTICAL
    return TOLERANCE_IDENTICAL


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare two csttool output directories for determinism.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dir_a", type=Path, help="First output directory")
    parser.add_argument("dir_b", nargs="?", type=Path, default=None,
                        help="Second output directory (omit with --self-test)")
    parser.add_argument("--subject-id", required=True, help="Subject identifier (e.g. sub-1204)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output JSON report path (default: stdout)")
    parser.add_argument("--self-test", action="store_true",
                        help="Compare dir_a against itself (smoke test)")
    parser.add_argument("--provenance-only", action="store_true",
                        help="Only compare provenance, no file comparison")
    args = parser.parse_args()

    if args.self_test:
        args.dir_b = args.dir_a

    if args.dir_b is None:
        parser.error("dir_b is required unless --self-test is used")

    if args.provenance_only:
        result = compare_provenance(args.dir_a, args.dir_b)
    else:
        print(f"Comparing: {args.dir_a}")
        print(f"     with: {args.dir_b}")
        print(f"  subject: {args.subject_id}")
        print()
        result = compare_directories(args.dir_a, args.dir_b, args.subject_id)
        print()
        print(f"Overall verdict: {result['verdict']}")
        for cat_name in SCIENTIFIC_CATEGORIES:
            cat = result["categories"].get(cat_name, {})
            v = cat.get("verdict", "N/A")
            print(f"  {cat_name}: {v}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nReport written to: {args.output}")
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

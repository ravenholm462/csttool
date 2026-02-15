"""
Tract-based metric stability tests across repeated deterministic runs.

These tests verify that scalar metrics sampled ALONG streamlines are
stable when tracking is run multiple times with the same seed.

Previous version computed global FA/MD/RD/AD means from the whole volume,
which are identical by construction (tensor fitting happens once before
tracking). This version samples scalars along the actual streamlines,
so the metrics depend on which streamlines were generated.

Coordinate space chain (verified):
    load_nifti(fraw) → affine
    LocalTracking(..., affine=affine) → streamlines in RASMM
    sample_scalar_along_tract(streamlines, scalar, affine)
        → inv(affine) @ point → voxel indices → nearest-neighbor lookup
"""

import pytest
import numpy as np

from csttool.metrics import sample_scalar_along_tract
from csttool.reproducibility.tolerance import (
    TOLERANCE_FA_MEAN_RTOL,
    TOLERANCE_FA_MEAN_ATOL,
    TOLERANCE_MD_MEAN_RTOL,
    TOLERANCE_MD_MEAN_ATOL,
    TOLERANCE_RD_MEAN_RTOL,
    TOLERANCE_RD_MEAN_ATOL,
    TOLERANCE_AD_MEAN_RTOL,
    TOLERANCE_AD_MEAN_ATOL,
)


def _tract_scalar_stats(t, scalar_key):
    """Sample a scalar map along streamlines and return distribution stats.

    Returns dict with n, mean, median, p05, p95 — or None if no valid samples.
    """
    values = sample_scalar_along_tract(
        t["streamlines"], t[scalar_key], t["affine"]
    )
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return None
    return {
        "n": len(valid),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "p05": float(np.percentile(valid, 5)),
        "p95": float(np.percentile(valid, 95)),
    }


def _assert_tract_scalar_stable(repeated_run_tractograms, scalar_key, rtol, atol):
    """Assert that tract-sampled scalar stats are identical across runs.

    Checks: sample count identical, mean/median/p05/p95 within tolerance.
    Returns list of stats dicts for reporting.
    """
    all_stats = []
    for t in repeated_run_tractograms:
        stats = _tract_scalar_stats(t, scalar_key)
        assert stats is not None, \
            f"No valid {scalar_key.upper()} samples in run {t['run_num']}"
        all_stats.append(stats)

    ref = all_stats[0]

    for i in range(1, len(all_stats)):
        s = all_stats[i]

        # Sample count must be exactly identical
        assert s["n"] == ref["n"], \
            f"{scalar_key.upper()} sample count differs: " \
            f"run 0={ref['n']} vs run {i}={s['n']}"

        # All distribution stats must be close
        for stat_name in ("mean", "median", "p05", "p95"):
            assert np.isclose(s[stat_name], ref[stat_name], rtol=rtol, atol=atol), \
                f"{scalar_key.upper()} {stat_name} differs: " \
                f"run 0={ref[stat_name]:.10e} vs run {i}={s[stat_name]:.10e}"

    return all_stats


def _print_stats(scalar_key, all_stats, rtol, atol):
    """Print tract-sampled stats for reproducibility evidence."""
    ref = all_stats[0]
    print(f"\n✓ Tract-sampled {scalar_key.upper()} stable across {len(all_stats)} runs")
    print(f"  n_samples: {ref['n']}")
    print(f"  mean:   {ref['mean']:.10f}")
    print(f"  median: {ref['median']:.10f}")
    print(f"  p05:    {ref['p05']:.10f}")
    print(f"  p95:    {ref['p95']:.10f}")

    # Max deviation across runs for each stat
    for stat_name in ("mean", "median", "p05", "p95"):
        vals = [s[stat_name] for s in all_stats]
        dev = max(vals) - min(vals)
        print(f"  max_dev({stat_name}): {dev:.2e}")

    print(f"  tolerance: rtol={rtol}, atol={atol}")


def test_tract_fa_stable_across_runs(repeated_run_tractograms):
    """Tract-sampled FA distribution identical across repeated deterministic runs."""
    stats = _assert_tract_scalar_stable(
        repeated_run_tractograms, "fa",
        TOLERANCE_FA_MEAN_RTOL, TOLERANCE_FA_MEAN_ATOL,
    )
    _print_stats("fa", stats, TOLERANCE_FA_MEAN_RTOL, TOLERANCE_FA_MEAN_ATOL)


def test_tract_md_stable_across_runs(repeated_run_tractograms):
    """Tract-sampled MD distribution identical across repeated deterministic runs."""
    stats = _assert_tract_scalar_stable(
        repeated_run_tractograms, "md",
        TOLERANCE_MD_MEAN_RTOL, TOLERANCE_MD_MEAN_ATOL,
    )
    _print_stats("md", stats, TOLERANCE_MD_MEAN_RTOL, TOLERANCE_MD_MEAN_ATOL)


def test_tract_rd_stable_across_runs(repeated_run_tractograms):
    """Tract-sampled RD distribution identical across repeated deterministic runs."""
    stats = _assert_tract_scalar_stable(
        repeated_run_tractograms, "rd",
        TOLERANCE_RD_MEAN_RTOL, TOLERANCE_RD_MEAN_ATOL,
    )
    _print_stats("rd", stats, TOLERANCE_RD_MEAN_RTOL, TOLERANCE_RD_MEAN_ATOL)


def test_tract_ad_stable_across_runs(repeated_run_tractograms):
    """Tract-sampled AD distribution identical across repeated deterministic runs."""
    stats = _assert_tract_scalar_stable(
        repeated_run_tractograms, "ad",
        TOLERANCE_AD_MEAN_RTOL, TOLERANCE_AD_MEAN_ATOL,
    )
    _print_stats("ad", stats, TOLERANCE_AD_MEAN_RTOL, TOLERANCE_AD_MEAN_ATOL)


def test_streamline_count_exact_across_runs(repeated_run_tractograms):
    """Streamline count must be exactly equal (no tolerance)."""
    counts = [len(t["streamlines"]) for t in repeated_run_tractograms]

    assert all(c == counts[0] for c in counts), \
        f"Streamline counts vary: {counts}"

    print(f"\n✓ Streamline count exactly {counts[0]} across all runs")


def test_all_tract_metrics_comprehensive_stability(repeated_run_tractograms):
    """Comprehensive stability summary for all tract-sampled metrics."""
    from csttool.reproducibility.tolerance import compute_metric_difference

    t1, t2 = repeated_run_tractograms[0], repeated_run_tractograms[1]

    def _metrics_dict(t):
        result = {"streamline_count": len(t["streamlines"])}
        for key in ("fa", "md", "rd", "ad"):
            s = _tract_scalar_stats(t, key)
            if s is not None:
                for stat_name in ("n", "mean", "median", "p05", "p95"):
                    result[f"tract_{key}_{stat_name}"] = s[stat_name]
        return result

    metrics1 = _metrics_dict(t1)
    metrics2 = _metrics_dict(t2)
    diffs = compute_metric_difference(metrics1, metrics2)

    print(f"\n✓ Comprehensive tract-based metric stability check")
    for key, diff_info in sorted(diffs.items()):
        if "absolute_diff" in diff_info:
            print(f"  {key}: abs_diff={diff_info['absolute_diff']:.2e}, "
                  f"rel_diff={diff_info['relative_diff']:.2e}")
        else:
            print(f"  {key}: {diff_info}")

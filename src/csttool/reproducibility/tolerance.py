"""
Tolerance thresholds for reproducibility and sensitivity tests.

Thresholds are empirically derived from observed data distributions.
Initial conservative values will be refined based on actual test results.
"""

import numpy as np


# =============================================================================
# Content-based reproducibility (same environment, same seed)
# =============================================================================

# Streamline count must match exactly
TOLERANCE_STREAMLINE_COUNT = 0

# Streamline coordinate tolerances
# Using realistic tolerances for millimeter-precision floating-point operations
TOLERANCE_COORDINATES_RTOL = 1e-8  # Relative tolerance
TOLERANCE_COORDINATES_ATOL = 1e-6  # Absolute tolerance (1 micrometer in mm units)

# =============================================================================
# Tract-based metric stability (repeated deterministic runs, same environment)
# These tolerances apply to scalars sampled ALONG streamlines, not global maps.
# =============================================================================

# Tract-sampled FA mean
TOLERANCE_FA_MEAN_RTOL = 1e-8
TOLERANCE_FA_MEAN_ATOL = 1e-9

# Tract-sampled MD mean
TOLERANCE_MD_MEAN_RTOL = 1e-8
TOLERANCE_MD_MEAN_ATOL = 1e-12

# Tract-sampled RD mean
TOLERANCE_RD_MEAN_RTOL = 1e-8
TOLERANCE_RD_MEAN_ATOL = 1e-12

# Tract-sampled AD mean
TOLERANCE_AD_MEAN_RTOL = 1e-8
TOLERANCE_AD_MEAN_ATOL = 1e-12

# LI (laterality index)
TOLERANCE_LI_RTOL = 1e-8
TOLERANCE_LI_ATOL = 1e-9

# =============================================================================
# Sensitivity analysis thresholds
# =============================================================================
# These will be derived empirically from Phase 5 sensitivity tests.
# Documented here but not enforced as pass/fail gates yet.
# Once distributions are stable, promote to actual assertion thresholds.

# Observed 95th percentile from empirical testing will be filled in:
# FA sensitivity at 10% removal: median X%, p95 Y%
# MD sensitivity at 10% removal: median X%, p95 Y%
# LI sensitivity at 10% removal (bilateral): median X%, p95 Y%

# Proposed thresholds (to be set after Phase 5 empirical testing):
# ACCEPTABLE_FA_SENSITIVITY_10PCT = None  # e.g., 0.040 (4.0%)
# ACCEPTABLE_MD_SENSITIVITY_10PCT = None  # e.g., 0.045 (4.5%)
# ACCEPTABLE_LI_SENSITIVITY_10PCT = None  # e.g., 0.018 (1.8%)


# =============================================================================
# Utility functions
# =============================================================================

def assert_metrics_equal(metrics1: dict, metrics2: dict) -> None:
    """Deep comparison of metric dictionaries using defined tolerances.

    Args:
        metrics1: First metrics dict
        metrics2: Second metrics dict

    Raises:
        AssertionError: If metrics differ beyond tolerances
    """
    # Check that both dicts have same keys
    assert set(metrics1.keys()) == set(metrics2.keys()), \
        f"Metric keys differ: {set(metrics1.keys())} vs {set(metrics2.keys())}"

    # Compare each metric
    for key in metrics1:
        val1 = metrics1[key]
        val2 = metrics2[key]

        # Determine tolerance based on metric type
        if "fa" in key.lower():
            rtol, atol = TOLERANCE_FA_MEAN_RTOL, TOLERANCE_FA_MEAN_ATOL
        elif "md" in key.lower():
            rtol, atol = TOLERANCE_MD_MEAN_RTOL, TOLERANCE_MD_MEAN_ATOL
        elif "rd" in key.lower():
            rtol, atol = TOLERANCE_RD_MEAN_RTOL, TOLERANCE_RD_MEAN_ATOL
        elif "ad" in key.lower():
            rtol, atol = TOLERANCE_AD_MEAN_RTOL, TOLERANCE_AD_MEAN_ATOL
        elif "li" in key.lower() or "lateral" in key.lower():
            rtol, atol = TOLERANCE_LI_RTOL, TOLERANCE_LI_ATOL
        else:
            # Default tolerances for unknown metrics
            rtol, atol = 1e-8, 1e-9

        # Handle nested dicts recursively
        if isinstance(val1, dict):
            assert_metrics_equal(val1, val2)
        # Handle numeric values
        elif isinstance(val1, (int, float, np.number)):
            assert np.isclose(val1, val2, rtol=rtol, atol=atol), \
                f"Metric '{key}' differs: {val1} vs {val2} (rtol={rtol}, atol={atol})"
        # Handle arrays
        elif isinstance(val1, np.ndarray):
            assert np.allclose(val1, val2, rtol=rtol, atol=atol), \
                f"Metric '{key}' array differs beyond tolerance"
        # Handle lists
        elif isinstance(val1, list):
            assert len(val1) == len(val2), \
                f"Metric '{key}' list length differs: {len(val1)} vs {len(val2)}"
            for i, (v1, v2) in enumerate(zip(val1, val2)):
                if isinstance(v1, (int, float, np.number)):
                    assert np.isclose(v1, v2, rtol=rtol, atol=atol), \
                        f"Metric '{key}[{i}]' differs: {v1} vs {v2}"
        # Exact match for other types
        else:
            assert val1 == val2, f"Metric '{key}' differs: {val1} vs {val2}"


def compute_metric_difference(metrics1: dict, metrics2: dict) -> dict:
    """Compute structured difference report between two metric dicts.

    Args:
        metrics1: First metrics dict
        metrics2: Second metrics dict

    Returns:
        Dict with difference statistics for each metric
    """
    differences = {}

    all_keys = set(metrics1.keys()) | set(metrics2.keys())
    for key in all_keys:
        if key not in metrics1:
            differences[key] = {"status": "missing_in_first"}
        elif key not in metrics2:
            differences[key] = {"status": "missing_in_second"}
        else:
            val1 = metrics1[key]
            val2 = metrics2[key]

            if isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
                diff = abs(val1 - val2)
                rel_diff = diff / abs(val1) if val1 != 0 else float('inf')
                differences[key] = {
                    "value1": float(val1),
                    "value2": float(val2),
                    "absolute_diff": float(diff),
                    "relative_diff": float(rel_diff),
                }
            elif isinstance(val1, dict) and isinstance(val2, dict):
                differences[key] = compute_metric_difference(val1, val2)
            else:
                differences[key] = {
                    "value1": str(val1),
                    "value2": str(val2),
                    "equal": val1 == val2
                }

    return differences


def is_within_tolerance(value1: float, value2: float, rtol: float, atol: float) -> bool:
    """Boolean check if two values are within tolerance.

    Uses np.allclose logic: |a - b| <= (atol + rtol * |b|)

    Args:
        value1: First value
        value2: Second value
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if within tolerance, False otherwise
    """
    return np.isclose(value1, value2, rtol=rtol, atol=atol)

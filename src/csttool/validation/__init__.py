"""
CST Validation Module

Provides bundle comparison metrics for validating csttool-extracted
CST streamlines against reference tractograms.
"""

from .bundle_comparison import (
    compute_bundle_overlap,
    compute_overreach,
    compute_coverage,
    mean_closest_distance,
    streamline_count_ratio,
    generate_validation_report,
    check_spatial_compatibility,
    check_hemisphere_alignment,
    SpatialMismatchError,
)

__all__ = [
    "compute_bundle_overlap",
    "compute_overreach",
    "compute_coverage",
    "mean_closest_distance",
    "streamline_count_ratio",
    "generate_validation_report",
    "check_spatial_compatibility",
    "check_hemisphere_alignment",
    "SpatialMismatchError",
]

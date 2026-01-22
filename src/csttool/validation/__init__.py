"""
CST Validation Module

Provides bundle comparison metrics for validating csttool-extracted
CST streamlines against reference tractograms.
"""

from .bundle_comparison import (
    compute_bundle_overlap,
    compute_overreach,
    mean_closest_distance,
    generate_validation_report,
)

__all__ = [
    "compute_bundle_overlap",
    "compute_overreach",
    "mean_closest_distance",
    "generate_validation_report",
]

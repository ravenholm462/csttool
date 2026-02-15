"""
Order-invariant determinism tests.

These tests verify that with a fixed seed and fixed environment,
csttool produces identical tractograms and metrics.
"""

import pytest
import numpy as np
import hashlib
from pathlib import Path

from csttool.reproducibility.tolerance import (
    TOLERANCE_STREAMLINE_COUNT,
    TOLERANCE_COORDINATES_RTOL,
    TOLERANCE_COORDINATES_ATOL,
)


def compute_streamline_fingerprint(s):
    """Create order-invariant fingerprint for streamline matching.

    Uses streamline length, start point, end point, and midpoint
    to create a hashable identifier.

    Args:
        s: Streamline array (N, 3)

    Returns:
        Tuple that can be used as dict key for matching
    """
    return (
        len(s),  # number of points
        round(np.linalg.norm(np.diff(s, axis=0), axis=1).sum(), 4),  # total length
        tuple(np.round(s[0], 4)),  # start point (rounded to 0.1mm)
        tuple(np.round(s[-1], 4)),  # end point
        tuple(np.round(s[len(s)//2], 4)) if len(s) > 0 else (0, 0, 0),  # midpoint
    )


def test_streamline_count_identical(repeated_run_tractograms):
    """Streamline count must be exactly identical with same seed."""
    counts = [len(t["streamlines"]) for t in repeated_run_tractograms]

    # All counts must be exactly equal
    assert all(c == counts[0] for c in counts), \
        f"Streamline counts vary across runs: {counts}"

    # Report for documentation
    print(f"\nStreamline counts across 3 runs: {counts}")
    print(f"✓ All runs produced exactly {counts[0]} streamlines")


def test_streamline_coordinates_identical_order_invariant(repeated_run_tractograms):
    """Streamline coordinates identical within tolerance (order-invariant matching).

    Uses fingerprint-based matching to handle potential iteration order differences.
    """
    t1, t2 = repeated_run_tractograms[0], repeated_run_tractograms[1]

    # Create fingerprint → streamline mapping for both runs
    fp1 = {compute_streamline_fingerprint(s): s for s in t1["streamlines"]}
    fp2 = {compute_streamline_fingerprint(s): s for s in t2["streamlines"]}

    # Fingerprints must match
    assert fp1.keys() == fp2.keys(), \
        f"Fingerprints don't match. Run 1: {len(fp1)} unique, Run 2: {len(fp2)} unique"

    # Compare matched streamline pairs
    mismatches = 0
    max_diff = 0.0

    for fp, s1 in fp1.items():
        s2 = fp2[fp]

        # Check coordinate-level equality
        if not np.allclose(s1, s2, rtol=TOLERANCE_COORDINATES_RTOL, atol=TOLERANCE_COORDINATES_ATOL):
            mismatches += 1
            diff = np.abs(s1 - s2).max()
            max_diff = max(max_diff, diff)

    # Report results
    print(f"\n✓ {len(fp1)} streamlines matched by fingerprint")
    print(f"  Max coordinate difference: {max_diff:.2e} mm")
    print(f"  Tolerance: rtol={TOLERANCE_COORDINATES_RTOL}, atol={TOLERANCE_COORDINATES_ATOL}")

    assert mismatches == 0, \
        f"{mismatches}/{len(fp1)} streamlines differ beyond tolerance (max_diff={max_diff:.2e})"


def test_streamline_lengths_identical(repeated_run_tractograms):
    """Streamline length distributions identical across runs."""
    from dipy.tracking.streamline import length

    lengths_per_run = []
    for t in repeated_run_tractograms:
        lengths = np.array([length(s) for s in t["streamlines"]])
        lengths_per_run.append(lengths)

    # Compare length statistics
    for i in range(1, len(lengths_per_run)):
        lengths1 = lengths_per_run[0]
        lengths2 = lengths_per_run[i]

        # Sort lengths for comparison (order-invariant)
        lengths1_sorted = np.sort(lengths1)
        lengths2_sorted = np.sort(lengths2)

        assert np.allclose(lengths1_sorted, lengths2_sorted,
                          rtol=TOLERANCE_COORDINATES_RTOL,
                          atol=TOLERANCE_COORDINATES_ATOL), \
            f"Length distributions differ between run 0 and run {i}"

    # Report
    mean_length = np.mean(lengths_per_run[0])
    std_length = np.std(lengths_per_run[0])
    print(f"\n✓ Streamline lengths identical across runs")
    print(f"  Mean length: {mean_length:.2f} ± {std_length:.2f} mm")


def test_tractogram_bounding_box_identical(repeated_run_tractograms):
    """Spatial extent of tractogram identical across runs."""
    bboxes = []

    for t in repeated_run_tractograms:
        # Compute bounding box from all streamlines
        all_points = np.vstack([s for s in t["streamlines"]])
        bbox_min = all_points.min(axis=0)
        bbox_max = all_points.max(axis=0)
        bboxes.append((bbox_min, bbox_max))

    # Compare bounding boxes
    for i in range(1, len(bboxes)):
        bbox1_min, bbox1_max = bboxes[0]
        bbox2_min, bbox2_max = bboxes[i]

        assert np.allclose(bbox1_min, bbox2_min,
                          rtol=TOLERANCE_COORDINATES_RTOL,
                          atol=TOLERANCE_COORDINATES_ATOL), \
            f"Bounding box min differs between run 0 and run {i}"

        assert np.allclose(bbox1_max, bbox2_max,
                          rtol=TOLERANCE_COORDINATES_RTOL,
                          atol=TOLERANCE_COORDINATES_ATOL), \
            f"Bounding box max differs between run 0 and run {i}"

    # Report
    bbox_min, bbox_max = bboxes[0]
    extent = bbox_max - bbox_min
    print(f"\n✓ Bounding boxes identical across runs")
    print(f"  Extent: [{extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}] mm")


@pytest.mark.optional
@pytest.mark.xfail(reason="Byte identity not guaranteed across platforms")
def test_trk_files_byte_identical(tmp_path, repeated_run_tractograms):
    """Best-effort test for byte-identical .trk files.

    Expected to pass only in controlled environment with same platform,
    BLAS implementation, and library versions.
    """
    from dipy.io.streamline import save_tractogram
    import nibabel as nib

    # Save two tractograms
    paths = []
    for i, t in enumerate(repeated_run_tractograms[:2]):
        path = tmp_path / f"run_{i}.trk"
        save_tractogram(t["sft"], str(path))
        paths.append(path)

    # Compute SHA256 hashes
    hashes = []
    for path in paths:
        with open(path, 'rb') as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
            hashes.append(sha256)

    # Compare
    if hashes[0] == hashes[1]:
        print(f"\n✓ .trk files are byte-identical (SHA256: {hashes[0][:16]}...)")
    else:
        print(f"\n  .trk file hashes differ:")
        print(f"    Run 0: {hashes[0]}")
        print(f"    Run 1: {hashes[1]}")
        pytest.fail("Byte-level identity not achieved (expected on some platforms)")

"""
Bundle Comparison Metrics for CST Validation

Provides metrics for comparing candidate CST tractograms against
reference bundles (e.g., TractoInferno PYT).
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
from nibabel.streamlines import load as load_trk
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.utils import density_map


def _load_streamlines(trk_path: str | Path):
    """Load streamlines from a .trk file."""
    trk_path = Path(trk_path)
    if not trk_path.exists():
        raise FileNotFoundError(f"Tractogram not found: {trk_path}")
    
    tractogram = load_trk(str(trk_path))
    return tractogram.streamlines, tractogram.affine, tractogram.header


def _streamlines_to_density(streamlines, affine, shape):
    """Convert streamlines to a binary density map."""
    dm = density_map(streamlines, affine, shape)
    return (dm > 0).astype(np.float32)


def compute_bundle_overlap(
    candidate_path: str | Path,
    reference_path: str | Path,
    shape: tuple = None,
) -> dict:
    """
    Compute Dice overlap coefficient between two bundles.
    
    The streamlines are converted to density maps (binary voxel masks)
    and Dice coefficient is computed as:
        Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        candidate_path: Path to candidate .trk file
        reference_path: Path to reference .trk file
        shape: Volume shape. If None, derived from tractogram header.
    
    Returns:
        dict with 'dice', 'intersection_volume', 'candidate_volume', 
        'reference_volume' keys
    """
    cand_streamlines, cand_affine, cand_header = _load_streamlines(candidate_path)
    ref_streamlines, ref_affine, ref_header = _load_streamlines(reference_path)
    
    # Use reference shape if not provided
    if shape is None:
        shape = tuple(ref_header["dimensions"])
    
    # Create density maps
    cand_density = _streamlines_to_density(cand_streamlines, cand_affine, shape)
    ref_density = _streamlines_to_density(ref_streamlines, ref_affine, shape)
    
    # Compute volumes
    intersection = np.sum(cand_density * ref_density)
    cand_volume = np.sum(cand_density)
    ref_volume = np.sum(ref_density)
    
    # Dice coefficient
    if cand_volume + ref_volume == 0:
        dice = 0.0
    else:
        dice = 2.0 * intersection / (cand_volume + ref_volume)
    
    return {
        "dice": float(dice),
        "intersection_volume": int(intersection),
        "candidate_volume": int(cand_volume),
        "reference_volume": int(ref_volume),
    }


def compute_overreach(
    candidate_path: str | Path,
    reference_path: str | Path,
    shape: tuple = None,
) -> dict:
    """
    Compute overreach: fraction of candidate bundle outside reference envelope.
    
    Overreach = |candidate - reference| / |candidate|
    
    A value of 0 means the candidate is entirely within the reference.
    A value of 1 means the candidate has no overlap with reference.
    
    Args:
        candidate_path: Path to candidate .trk file
        reference_path: Path to reference .trk file
        shape: Volume shape. If None, derived from tractogram header.
    
    Returns:
        dict with 'overreach', 'overreach_volume', 'candidate_volume' keys
    """
    cand_streamlines, cand_affine, cand_header = _load_streamlines(candidate_path)
    ref_streamlines, ref_affine, ref_header = _load_streamlines(reference_path)
    
    if shape is None:
        shape = tuple(ref_header["dimensions"])
    
    cand_density = _streamlines_to_density(cand_streamlines, cand_affine, shape)
    ref_density = _streamlines_to_density(ref_streamlines, ref_affine, shape)
    
    # Overreach: candidate voxels not in reference
    overreach_mask = cand_density * (1 - ref_density)
    overreach_volume = np.sum(overreach_mask)
    cand_volume = np.sum(cand_density)
    
    if cand_volume == 0:
        overreach = 0.0
    else:
        overreach = float(overreach_volume) / float(cand_volume)
    
    return {
        "overreach": float(overreach),
        "overreach_volume": int(overreach_volume),
        "candidate_volume": int(cand_volume),
    }


def mean_closest_distance(
    candidate_path: str | Path,
    reference_path: str | Path,
    num_points: int = 20,
    num_samples: int = 1000,
) -> dict:
    """
    Compute Mean of Closest Distances (MDF) between two bundles.
    
    For each streamline in the candidate, find the closest streamline
    in the reference (by mean point-wise distance), then average.
    
    For efficiency, streamlines are resampled to a fixed number of points
    and a random sample is used if bundles are large.
    
    Args:
        candidate_path: Path to candidate .trk file
        reference_path: Path to reference .trk file
        num_points: Number of points to resample each streamline to
        num_samples: Max streamlines to sample from each bundle
    
    Returns:
        dict with 'mdf_mean', 'mdf_std', 'num_candidate', 'num_reference' keys
    """
    cand_streamlines, _, _ = _load_streamlines(candidate_path)
    ref_streamlines, _, _ = _load_streamlines(reference_path)
    
    # Convert to list for resampling
    cand_list = list(cand_streamlines)
    ref_list = list(ref_streamlines)
    
    if len(cand_list) == 0 or len(ref_list) == 0:
        return {
            "mdf_mean": float("nan"),
            "mdf_std": float("nan"),
            "num_candidate": len(cand_list),
            "num_reference": len(ref_list),
        }
    
    # Resample to fixed number of points
    cand_resampled = set_number_of_points(cand_list, num_points)
    ref_resampled = set_number_of_points(ref_list, num_points)
    
    # Convert to arrays
    cand_array = np.array(list(cand_resampled))  # (N, num_points, 3)
    ref_array = np.array(list(ref_resampled))    # (M, num_points, 3)
    
    # Sample if too large
    if len(cand_array) > num_samples:
        idx = np.random.choice(len(cand_array), num_samples, replace=False)
        cand_array = cand_array[idx]
    
    if len(ref_array) > num_samples:
        idx = np.random.choice(len(ref_array), num_samples, replace=False)
        ref_array = ref_array[idx]
    
    # Compute MDF: for each candidate streamline, find closest reference
    distances = []
    for cand_sl in cand_array:
        # Mean point-wise distance to each reference streamline
        diffs = ref_array - cand_sl[np.newaxis, :, :]  # (M, num_points, 3)
        point_dists = np.linalg.norm(diffs, axis=2)     # (M, num_points)
        mean_dists = np.mean(point_dists, axis=1)       # (M,)
        min_dist = np.min(mean_dists)
        distances.append(min_dist)
    
    distances = np.array(distances)
    
    return {
        "mdf_mean": float(np.mean(distances)),
        "mdf_std": float(np.std(distances)),
        "num_candidate": len(cand_array),
        "num_reference": len(ref_array),
    }


def generate_validation_report(
    metrics: dict,
    output_path: str | Path,
    candidate_paths: list = None,
    reference_paths: list = None,
) -> Path:
    """
    Generate a JSON validation report.
    
    Args:
        metrics: Dictionary of computed metrics
        output_path: Output path for the report (JSON)
        candidate_paths: List of candidate tractogram paths (for metadata)
        reference_paths: List of reference tractogram paths (for metadata)
    
    Returns:
        Path to the generated report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "candidate_tractograms": [str(p) for p in (candidate_paths or [])],
        "reference_tractograms": [str(p) for p in (reference_paths or [])],
        "metrics": metrics,
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return output_path

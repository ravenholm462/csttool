"""
funcs.py

Utility functions for csttool's metric analysis pipeline.

"""

# General imports
import numpy as np

# Analysis imports
from dipy.tracking.streamline import length, Streamlines
from dipy.tracking.utils import density_map

def analyze_cst_bundle(
        streamlines,
        fa_map=None,
        md_map=None,
        affine=None
):
    """Analysis of a single bundle.

    Args:
        streamlines (_type_): _description_
        fa_map (_type_, optional): _description_. Defaults to None.
        md_map (_type_, optional): _description_. Defaults to None.
        affine (_type_, optional): _description_. Defaults to None.
    """

    metrics = {}

    # Morphology of the CST
    metrics["morphology"] = {
        "n_streamlines": len(streamlines),
        "mean_length": compute_streamline_length(streamlines)["mean_length"],
        "tract_volume": compute_tract_volume(streamlines)
    }

    # FA analysis
    if fa_map is not None:
        if affine is None:
            raise ValueError("Affine required for FA analysis")
            
        fa_values = sample_scalar_along_tract(streamlines, fa_map, affine)
        metrics["fa"] = {
            "mean": float(np.mean(fa_values)),
            "std": float(np.std(fa_values)),
            "median": float(np.median(fa_values)),
            "profile": compute_tract_profile(streamlines, fa_map, affine),  # TO-DO
            "values": fa_values.tolist()  # All sampled FA values
        }

    # MD analysis
    if md_map is not None:
        if affine is None:
            raise ValueError("Affine required for MD analysis")
            
        md_values = sample_scalar_along_tract(streamlines, md_map, affine)
        metrics["md"] = {
            "mean": float(np.mean(md_values)),
            "std": float(np.std(md_values)),
            "median": float(np.median(md_values)),
            "profile": compute_tract_profile(streamlines, md_map, affine),  # TO-DO
            "values": md_values.tolist()
        }

    return metrics


def compute_streamline_length(streamlines):
    """Computes streamline length values

    Args:
        streamlines (Streamlines): input streamlines

    Returns:
        dict: Dictionary of length statistics and streamlines.
    """

    vals = {}

    if len(streamlines) == 0:
        vals

    length_array = np.array([length(s) for s in streamlines])
    valid_lengths = length_array[length_array > 0]

    vals["mean_length"] = float(np.mean(valid_lengths))
    vals["std_length"] = float(np.std(valid_lengths))
    vals["min_length"] = float(np.min(valid_lengths))
    vals["max_length"] = float(np.max(valid_lengths))
    vals["n_streamlines"] = len(valid_lengths)
    vals["lengths"] = valid_lengths

    return vals


def compute_tract_volume(streamlines, affine, voxel_size=None):
    """Compute tract volume and density map

    Args:
        streamlines (Streamline): Input streamline
        affine (_type_): Affine transformation matrix
        voxel_size (_type_, optional): Voxel dimensions. Defaults to None.

    Returns:
        tuple: (density_map, volume_in_mm3)
    """
    # If no voxel size is given, compute
    if voxel_size is None:
        voxel_size = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))

    density = density_map(
        streamlines=streamlines,
        affine=affine,
        voxel_size=voxel_size
    )

    voxel_volume = np.prod(voxel_size)
    volume = np.sum(density>0)*voxel_volume

    return density, volume


def sample_scalar_along_tract(streamlines, scalar_map, affine):
    """Sample streamline point for point. For each point, convert world coordinates (mm) to voxel coords and sample scalar value

    Args:
        streamlines (Streamline): Input streamlines
        scalar_map (np.array): 3D scalar map (e.g. FA, MD...)
        affine (np.array): Affine transformations matrix
    """
    if len(streamlines) == 0:
        return np.array([])

    scalar_values = []

    for s in streamlines:
        for point in s:

            voxel_coord = np.round(point).astype(int)  # Convert coords

            # Check if voxel coords withing scalar map bounds
            if (voxel_coord[0] >= 0 and voxel_coord[0] < scalar_map.shape[0] and
                voxel_coord[1] >= 0 and voxel_coord[1] < scalar_map.shape[1] and
                voxel_coord[2] >= 0 and voxel_coord[2] < scalar_map.shape[2]):
                
                scalar_value = scalar_map[voxel_coord[0], voxel_coord[1], voxel_coord[2]]
                scalar_values.append(scalar_value)

    return np.array(scalar_values)


def mean_fa_along_tract(streamlines, fa_map, affine):
    """Calculate mean FA along tract using sample_scalar along_tract

    Args:
        streamlines (Streamlines): Input streamlines
        fa_map (np.array): 3D FA map
        affine (np.array): 3D affine transformation matrix

    Returns:
        float: mean FA value along tract
    """    
    fa_values = sample_scalar_along_tract(streamlines, scalar_map=fa_map, affine=affine)

    if len(fa_values) > 0:
        return float(np.mean(fa_values))
    else:
        return 0.0


def std_fa_along_tract(streamlines, fa_map, affine):
    """Calculate FA std along tract using sample_scalar along_tract

    Args:
        streamlines (Streamlines): Input streamlines
        fa_map (np.array): 3D FA map
        affine (np.array): 3D affine transformation matrix

    Returns:
        float: FA std along tract
    """    
    fa_values = sample_scalar_along_tract(streamlines, scalar_map=fa_map, affine=affine)

    if len(fa_values) > 0:
        return float(np.std(fa_values))
    else:
        return 0.0


def mean_md_along_tract(streamlines, md_map, affine):
    """Calculate mean MD along tract using sample_scalar along_tract

    Args:
        streamlines (Streamlines): Input streamlines
        fa_map (np.array): 3D MD map
        affine (np.array): 3D affine transformation matrix

    Returns:
        float: mean FA value along tract
    """    
    md_values = sample_scalar_along_tract(streamlines, scalar_map=md_map, affine=affine)

    if len(md_values) > 0:
        return float(np.mean(md_values))
    else:
        return 0.0

def compute_tract_profile(streamlines, scalar_map, affine, n_points=20):
    """Compute normalized tract profile (average scalar along tract length).
    
    Args:
        streamlines: Input streamlines
        scalar_map: 3D scalar map
        affine: Affine transformation matrix
        n_points: Number of points in the profile
        
    Returns:
        ndarray: Average scalar values at normalized positions
    """
    
    # TO-DO

    return np.zeros(n_points)


def compare_bilateral_cst(left_streamlines, right_streamlines, fa_map=None, md_map=None, affine=None):

    left_metrics = analyze_cst_bundle(left_streamlines, fa_map, md_map, affine)
    right_metrics = analyze_cst_bundle(right_streamlines, fa_map, md_map, affine)
    
    comparison = {
        'left': left_metrics,
        'right': right_metrics,
        'asymmetry': {}
    }
    
    # Laterality index: (L - R) / (L + R)
    left_vol = left_metrics['morphology']['tract_volume']
    right_vol = right_metrics['morphology']['tract_volume']
    if left_vol + right_vol > 0:
        comparison['asymmetry']['volume_laterality'] = (left_vol - right_vol) / (left_vol + right_vol)
    
    if fa_map is not None:
        left_fa = left_metrics['fa']['mean']
        right_fa = right_metrics['fa']['mean']
        if left_fa + right_fa > 0:
            comparison['asymmetry']['fa_laterality'] = (left_fa - right_fa) / (left_fa + right_fa)
    
    return comparison
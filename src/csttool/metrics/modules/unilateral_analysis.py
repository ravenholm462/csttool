"""
unilateral_analysis.py

Functions for analyzing a single CST hemisphere (left OR right).

This module computes:
- Morphological metrics (streamline count, length, volume)
- Microstructural metrics (FA, MD sampling)
- Tract profiles (along-the-tract distribution)
- Descriptive statistics
"""

import numpy as np
from dipy.tracking.streamline import length
from dipy.tracking.utils import density_map


def analyze_cst_hemisphere(
    streamlines,
    fa_map=None,
    md_map=None,
    affine=None,
    hemisphere='unknown'
):
    """
    Complete analysis of a single CST hemisphere.
    
    Parameters
    ----------
    streamlines : Streamlines
        Streamlines for one hemisphere (left OR right)
    fa_map : ndarray, optional
        3D fractional anisotropy map
    md_map : ndarray, optional
        3D mean diffusivity map
    affine : ndarray, optional
        4x4 affine transformation matrix
    hemisphere : str
        'left' or 'right' for identification
        
    Returns
    -------
    metrics : dict
        Comprehensive metrics dictionary containing:
        - morphology: streamline count, length stats, volume
        - fa: mean, std, median, profile, all sampled values
        - md: mean, std, median, profile, all sampled values
        - hemisphere: identification string
    """
    
    print(f"\nAnalyzing {hemisphere.upper()} CST...")
    
    metrics = {
        'hemisphere': hemisphere,
        'morphology': compute_morphology(streamlines, affine)
    }
    
    # Microstructural analysis requires affine
    if fa_map is not None and affine is not None:
        fa_values = sample_scalar_along_tract(streamlines, fa_map, affine)
        metrics['fa'] = {
            'mean': float(np.mean(fa_values)),
            'std': float(np.std(fa_values)),
            'median': float(np.median(fa_values)),
            'min': float(np.min(fa_values)),
            'max': float(np.max(fa_values)),
            'profile': compute_tract_profile(streamlines, fa_map, affine, n_points=20),
            'n_samples': len(fa_values)
        }
        print(f"  FA: {metrics['fa']['mean']:.3f} ± {metrics['fa']['std']:.3f}")
    
    if md_map is not None and affine is not None:
        md_values = sample_scalar_along_tract(streamlines, md_map, affine)
        metrics['md'] = {
            'mean': float(np.mean(md_values)),
            'std': float(np.std(md_values)),
            'median': float(np.median(md_values)),
            'min': float(np.min(md_values)),
            'max': float(np.max(md_values)),
            'profile': compute_tract_profile(streamlines, md_map, affine, n_points=20),
            'n_samples': len(md_values)
        }
        print(f"  MD: {metrics['md']['mean']:.3e} ± {metrics['md']['std']:.3e}")
    
    return metrics


def compute_morphology(streamlines, affine):
    """
    Compute morphological properties of a streamline bundle.
    
    Parameters
    ----------
    streamlines : Streamlines
        Input streamlines
    affine : ndarray
        4x4 affine transformation matrix
        
    Returns
    -------
    morphology : dict
        Dictionary containing:
        - n_streamlines: number of streamlines
        - mean_length: average streamline length in mm
        - std_length: standard deviation of lengths
        - min_length: minimum streamline length
        - max_length: maximum streamline length
        - tract_volume: volume in mm³
    """
    
    if len(streamlines) == 0:
        return {
            'n_streamlines': 0,
            'mean_length': 0.0,
            'std_length': 0.0,
            'min_length': 0.0,
            'max_length': 0.0,
            'tract_volume': 0.0
        }
    
    # Compute streamline lengths
    lengths = np.array([length(s) for s in streamlines])
    
    # Compute tract volume by counting unique voxels
    # Get all points from all streamlines
    all_points = np.vstack(streamlines)
    
    # Transform to voxel coordinates using inverse affine
    inv_affine = np.linalg.inv(affine)
    all_points_hom = np.c_[all_points, np.ones(len(all_points))]
    voxel_coords = (inv_affine @ all_points_hom.T).T[:, :3]
    
    # Round to integer voxel indices
    voxel_indices = np.round(voxel_coords).astype(int)
    
    # Count unique voxels
    unique_voxels = np.unique(voxel_indices, axis=0)
    n_voxels = len(unique_voxels)
    
    # Compute voxel volume
    voxel_size = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    voxel_volume = np.prod(voxel_size)
    
    # Total tract volume
    tract_volume = n_voxels * voxel_volume
    
    morphology = {
        'n_streamlines': len(streamlines),
        'mean_length': float(np.mean(lengths)),
        'std_length': float(np.std(lengths)),
        'min_length': float(np.min(lengths)),
        'max_length': float(np.max(lengths)),
        'tract_volume': float(tract_volume)
    }
    
    print(f"  Morphology: {morphology['n_streamlines']} streamlines, "
          f"{morphology['mean_length']:.1f} mm mean length, "
          f"{morphology['tract_volume']:.0f} mm³ volume")
    
    return morphology


def sample_scalar_along_tract(streamlines, scalar_map, affine):
    """
    Sample scalar values at every point along all streamlines.
    
    Parameters
    ----------
    streamlines : Streamlines
        Input streamlines in world coordinates (mm)
    scalar_map : ndarray
        3D scalar map (e.g., FA or MD)
    affine : ndarray
        4x4 affine transformation matrix
        
    Returns
    -------
    scalar_values : ndarray
        Array of all sampled scalar values (flattened across all streamlines)
    """
    
    if len(streamlines) == 0:
        return np.array([])
    
    scalar_values = []
    
    for streamline in streamlines:
        for point in streamline:
            # Convert world coordinates to voxel coordinates
            voxel_coord = world_to_voxel(point, affine)
            
            # Check bounds
            if (0 <= voxel_coord[0] < scalar_map.shape[0] and
                0 <= voxel_coord[1] < scalar_map.shape[1] and
                0 <= voxel_coord[2] < scalar_map.shape[2]):
                
                scalar_value = scalar_map[voxel_coord[0], voxel_coord[1], voxel_coord[2]]
                scalar_values.append(scalar_value)
    
    return np.array(scalar_values)


def compute_tract_profile(streamlines, scalar_map, affine, n_points=20):
    """
    Compute normalized tract profile (average scalar along tract).
    
    This function samples scalar values along each streamline, normalizes
    each streamline to the same number of points, and averages across all
    streamlines to create a representative profile.
    
    Parameters
    ----------
    streamlines : Streamlines
        Input streamlines
    scalar_map : ndarray
        3D scalar map (FA or MD)
    affine : ndarray
        4x4 affine transformation matrix
    n_points : int
        Number of points in output profile (default: 20)
        
    Returns
    -------
    profile : ndarray
        Average scalar values at n_points positions along the tract
    """
    
    if len(streamlines) == 0:
        return np.zeros(n_points)
    
    all_profiles = []
    
    for streamline in streamlines:
        if len(streamline) < 2:
            continue
        
        # Sample scalar values at each point
        streamline_scalars = []
        for point in streamline:
            voxel_coord = world_to_voxel(point, affine)
            
            if (0 <= voxel_coord[0] < scalar_map.shape[0] and
                0 <= voxel_coord[1] < scalar_map.shape[1] and
                0 <= voxel_coord[2] < scalar_map.shape[2]):
                
                scalar_value = scalar_map[voxel_coord[0], voxel_coord[1], voxel_coord[2]]
                streamline_scalars.append(scalar_value)
        
        if len(streamline_scalars) < 5:  # Need minimum points
            continue
        
        # Normalize to n_points
        if len(streamline_scalars) >= n_points:
            # Downsample
            indices = np.linspace(0, len(streamline_scalars)-1, n_points).astype(int)
            normalized_profile = np.array(streamline_scalars)[indices]
        else:
            # Upsample with interpolation
            x_original = np.linspace(0, 1, len(streamline_scalars))
            x_target = np.linspace(0, 1, n_points)
            normalized_profile = np.interp(x_target, x_original, streamline_scalars)
        
        all_profiles.append(normalized_profile)
    
    if len(all_profiles) == 0:
        return np.zeros(n_points)
    
    # Average across all streamlines
    final_profile = np.mean(all_profiles, axis=0)
    
    return final_profile.tolist()  # Convert to list for JSON serialization


def world_to_voxel(world_point, affine):
    """
    Convert world coordinates (mm) to voxel coordinates.
    
    Parameters
    ----------
    world_point : array-like
        3D point in world coordinates [x, y, z]
    affine : ndarray
        4x4 affine transformation matrix
        
    Returns
    -------
    voxel_coord : ndarray
        3D voxel coordinates [i, j, k] as integers
    """
    
    # Add homogeneous coordinate
    world_point_homogeneous = np.append(world_point, 1.0)
    
    # Apply inverse affine transformation
    voxel_coord_homogeneous = np.linalg.inv(affine) @ world_point_homogeneous
    
    # Convert to integer voxel coordinates
    voxel_coord = np.round(voxel_coord_homogeneous[:3]).astype(int)
    
    return voxel_coord


def print_hemisphere_summary(metrics):
    """
    Print human-readable summary of hemisphere metrics.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary from analyze_cst_hemisphere()
    """
    
    hemisphere = metrics['hemisphere'].upper()
    print(f"\n{'='*60}")
    print(f"{hemisphere} CST ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    morph = metrics['morphology']
    print(f"Streamline Count: {morph['n_streamlines']}")
    print(f"Mean Length: {morph['mean_length']:.1f} ± {morph['std_length']:.1f} mm")
    print(f"Length Range: [{morph['min_length']:.1f}, {morph['max_length']:.1f}] mm")
    print(f"Tract Volume: {morph['tract_volume']:.0f} mm³")
    
    if 'fa' in metrics:
        fa = metrics['fa']
        print(f"\nFractional Anisotropy:")
        print(f"  Mean: {fa['mean']:.3f} ± {fa['std']:.3f}")
        print(f"  Range: [{fa['min']:.3f}, {fa['max']:.3f}]")
        print(f"  Samples: {fa['n_samples']}")
    
    if 'md' in metrics:
        md = metrics['md']
        print(f"\nMean Diffusivity:")
        print(f"  Mean: {md['mean']:.3e} ± {md['std']:.3e}")
        print(f"  Range: [{md['min']:.3e}, {md['max']:.3e}]")
        print(f"  Samples: {md['n_samples']}")
    
    print(f"{'='*60}\n")
"""
passthrough_filtering.py

Filter streamlines that pass through (not just terminate in) ROI masks.
More permissive than endpoint filtering for long-range tracts.
"""

import numpy as np
from dipy.tracking.streamline import Streamlines, length


def streamline_passes_through(streamline, mask, affine):
    """Check if any point along streamline passes through mask."""
    inv_affine = np.linalg.inv(affine)
    ones = np.ones((len(streamline), 1))
    pts_h = np.hstack([streamline, ones])
    voxels = np.round((pts_h @ inv_affine.T)[:, :3]).astype(int)
    
    for vox in voxels:
        if (0 <= vox[0] < mask.shape[0] and
            0 <= vox[1] < mask.shape[1] and
            0 <= vox[2] < mask.shape[2]):
            if mask[vox[0], vox[1], vox[2]] > 0:
                return True
    return False


def extract_cst_passthrough(
    streamlines,
    masks,
    affine,
    min_length=20.0,
    max_length=200.0,
    verbose=True
):
    """
    Extract bilateral CST using pass-through filtering.
    
    A streamline is included if it passes through BOTH the brainstem
    AND the corresponding motor cortex at any point along its length.
    
    Parameters
    ----------
    streamlines : Streamlines
        Whole-brain tractogram.
    masks : dict
        ROI masks with keys: 'brainstem', 'motor_left', 'motor_right'
    affine : ndarray, shape (4, 4)
        Affine matrix for masks.
    min_length : float
        Minimum streamline length in mm.
    max_length : float
        Maximum streamline length in mm.
    verbose : bool
        Print progress information.
        
    Returns
    -------
    result : dict
        Dictionary with 'cst_left', 'cst_right', 'cst_combined', 'stats'
    """
    if verbose:
        print("=" * 60)
        print("PASS-THROUGH CST EXTRACTION")
        print("=" * 60)
        print(f"\nInput: {len(streamlines):,} streamlines")
    
    brainstem = masks['brainstem']
    motor_left = masks['motor_left']
    motor_right = masks['motor_right']
    
    # Length filtering
    if verbose:
        print(f"\n[Step 1/2] Length filtering ({min_length}-{max_length} mm)...")
    
    lengths = np.array([length(sl) for sl in streamlines])
    length_mask = (lengths >= min_length) & (lengths <= max_length)
    valid_indices = np.where(length_mask)[0]
    streamlines_filtered = Streamlines([streamlines[i] for i in valid_indices])
    
    if verbose:
        print(f"    {len(streamlines):,} → {len(streamlines_filtered):,} streamlines")
    
    # Pass-through filtering
    if verbose:
        print(f"\n[Step 2/2] Pass-through filtering...")
    
    cst_left_list = []
    cst_right_list = []
    
    for i, sl in enumerate(streamlines_filtered):
        passes_bs = streamline_passes_through(sl, brainstem, affine)
        
        if passes_bs:
            if streamline_passes_through(sl, motor_left, affine):
                cst_left_list.append(sl)
            if streamline_passes_through(sl, motor_right, affine):
                cst_right_list.append(sl)
        
        if verbose and (i + 1) % 50000 == 0:
            print(f"    Processed {i + 1:,} / {len(streamlines_filtered):,}...")
    
    cst_left = Streamlines(cst_left_list)
    cst_right = Streamlines(cst_right_list)
    cst_combined = Streamlines(cst_left_list + cst_right_list)
    
    stats = {
        'total_input': len(streamlines),
        'after_length_filter': len(streamlines_filtered),
        'cst_left_count': len(cst_left),
        'cst_right_count': len(cst_right),
        'cst_total_count': len(cst_combined),
        'extraction_rate': len(cst_combined) / len(streamlines) * 100 if len(streamlines) > 0 else 0,
    }
    
    if verbose:
        print(f"\n" + "=" * 60)
        print("PASS-THROUGH EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"\nResults:")
        print(f"    Left CST:  {stats['cst_left_count']:,} streamlines")
        print(f"    Right CST: {stats['cst_right_count']:,} streamlines")
        print(f"    Total:     {stats['cst_total_count']:,} streamlines")
    
    return {
        'cst_left': cst_left,
        'cst_right': cst_right,
        'cst_combined': cst_combined,
        'stats': stats
    }
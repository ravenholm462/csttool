def seed_and_stop(fa, affine, white_matter=None, fa_thresh=0.2, density=1, use_binary=False, verbose=False):
    """Generate seeds and stopping criterion for tractography.
    
    Args:
        fa (ndarray): Fractional anisotropy map (X, Y, Z).
        affine (ndarray): 4x4 affine transformation matrix.
        white_matter (ndarray): Binary white matter mask. Required if use_binary=True.
        fa_thresh (float): FA threshold for seeding and stopping (default 0.2).
        density (int): Seeds per voxel in seed mask (default 1).
        use_binary (bool): If True, use binary stopping on WM mask. If False (default), use FA threshold stopping.
        verbose (bool): Print processing details.
        
    Returns:
        tuple: (seeds, stopping_criterion)
            - seeds: Seed points in world coordinates (N, 3)
            - stopping_criterion: Stopping criterion for LocalTracking
    """
    from dipy.tracking.utils import seeds_from_mask
    from dipy.tracking.stopping_criterion import (
        ThresholdStoppingCriterion, 
        BinaryStoppingCriterion
    )
    
    if use_binary:
        if white_matter is None:
            raise ValueError("white_matter mask required when use_binary=True")
        
        seeds = seeds_from_mask(white_matter, affine, density=density)
        stopping_criterion = BinaryStoppingCriterion(white_matter)
        
        if verbose:
            print(f"Seeding and stopping (binary mode)...")
            print(f"White matter voxels: {white_matter.sum():,}")
            print(f"Total seeds (density={density}): {len(seeds):,}")
    
    else:
        # FA-based approach (default)
        seed_mask = fa >= fa_thresh
        seeds = seeds_from_mask(seed_mask, affine, density=density)
        stopping_criterion = ThresholdStoppingCriterion(fa, fa_thresh)
        
        if verbose:
            print(f"Seeding and stopping (FA threshold={fa_thresh})...")
            print(f"Seed mask voxels: {seed_mask.sum():,}")
            print(f"Total seeds (density={density}): {len(seeds):,}")
    
    return seeds, stopping_criterion
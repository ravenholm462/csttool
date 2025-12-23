def estimate_directions(data, gtab, white_matter, sh_order=6, sphere_name="symmetric362", verbose=False):
    """Estimate principal diffusion directions using CSA ODF model. 
       Code adapted from https://docs.dipy.org/dev/examples_built/streamline_analysis/streamline_tools.html
    
    Args:
        data (ndarray): 4D DWI data array (X, Y, Z, N).
        gtab (GradientTable): DIPY gradient table.
        white_matter (ndarray): Binary white matter mask for direction estimation.
        sh_order (int): Maximum spherical harmonic order (default 6).
        sphere_name (str): Name of sphere to build model around. symmetric 362 for speed, default_sphere for accuracy.
        verbose (bool): Print processing details.
        
    Returns:
        PeaksAndMetrics: Direction field object containing:
            - peak_dirs: Principal directions per voxel
            - peak_values: ODF values at peaks
            - peak_indices: Indices into the sphere
            - gfa: Generalized fractional anisotropy
    """
    from dipy.reconst.shm import CsaOdfModel
    from dipy.direction import peaks_from_model
    from dipy.data import get_sphere

    sphere = get_sphere(name=sphere_name)
    
    if verbose:
        print(f"Estimating directions (CSA ODF, SH order={sh_order})...")
        print(f"    White matter voxels: {white_matter.sum():,}")
    
    csamodel = CsaOdfModel(gtab, sh_order_max=sh_order)
    
    csapeaks = peaks_from_model(
        model=csamodel,
        data=data,
        sphere=sphere,
        relative_peak_threshold=0.8,
        min_separation_angle=45,
        mask=white_matter,
        npeaks=1,  # Single direction for deterministic tracking
    )
    
    if verbose:
        # Count voxels with valid peaks
        valid_peaks = (csapeaks.peak_values[..., 0] > 0).sum()
        print(f"    Voxels with valid peaks: {valid_peaks:,}")
        coverage = valid_peaks / white_matter.sum() * 100
        print(f"    Coverage: {coverage:.1f}%")
    
    return csapeaks
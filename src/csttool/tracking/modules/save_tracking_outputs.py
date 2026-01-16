def save_tracking_outputs(streamlines, img, fa, md, affine, out_dir, stem, rd=None, ad=None, tracking_params=None, verbose=False):
    """Save tractography outputs: tractogram, scalar maps, and processing report.
    
    Args:
        streamlines (Streamlines): Generated streamlines from run_tractography().
        img (Nifti1Image): Reference NIfTI image for tractogram space.
        fa (ndarray): Fractional anisotropy map (X, Y, Z).
        md (ndarray): Mean diffusivity map (X, Y, Z).
        affine (ndarray): 4x4 affine transformation matrix.
        out_dir (str or Path): Output directory.
        stem (str): Subject/scan identifier for filenames.
        rd (ndarray, optional): Radial diffusivity map (X, Y, Z).
        ad (ndarray, optional): Axial diffusivity map (X, Y, Z).
        tracking_params (dict): Parameters used for tracking (for reproducibility).
        verbose (bool): Print processing details.
        
    Returns:
        dict: Paths to all saved outputs:
            - tractogram: Path to .trk file
            - fa_map: Path to FA NIfTI
            - md_map: Path to MD NIfTI
            - rd_map: Path to RD NIfTI (if provided)
            - ad_map: Path to AD NIfTI (if provided)
            - report: Path to JSON report
    """
    from pathlib import Path
    from datetime import datetime
    import json
    import numpy as np
    import nibabel as nib
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    from dipy.io.streamline import save_tractogram
    from dipy.tracking.streamline import length
    
    out_dir = Path(out_dir)
    
    # Create output directory structure
    tractogram_dir = out_dir / "tractograms"
    scalar_dir = out_dir / "scalar_maps"
    log_dir = out_dir / "logs"
    
    tractogram_dir.mkdir(parents=True, exist_ok=True)
    scalar_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # 1. Save tractogram
    tractogram_path = tractogram_dir / f"{stem}_whole_brain.trk"
    sft = StatefulTractogram(streamlines, img, Space.RASMM)
    save_tractogram(sft, str(tractogram_path))
    outputs['tractogram'] = tractogram_path
    
    if verbose:
        print(f"Tractogram: {tractogram_path}")
    
    # 2. Save FA map
    fa_path = scalar_dir / f"{stem}_fa.nii.gz"
    nib.save(nib.Nifti1Image(fa.astype(np.float32), affine), fa_path)
    outputs['fa_map'] = fa_path
    
    if verbose:
        print(f"FA map: {fa_path}")
    
    # 3. Save MD map
    md_path = scalar_dir / f"{stem}_md.nii.gz"
    nib.save(nib.Nifti1Image(md.astype(np.float32), affine), md_path)
    outputs['md_map'] = md_path
    
    if verbose:
        print(f"MD map: {md_path}")
    
    # 4. Save RD map (if provided)
    if rd is not None:
        rd_path = scalar_dir / f"{stem}_rd.nii.gz"
        nib.save(nib.Nifti1Image(rd.astype(np.float32), affine), rd_path)
        outputs['rd_map'] = rd_path
        
        if verbose:
            print(f"RD map: {rd_path}")
    
    # 5. Save AD map (if provided)
    if ad is not None:
        ad_path = scalar_dir / f"{stem}_ad.nii.gz"
        nib.save(nib.Nifti1Image(ad.astype(np.float32), affine), ad_path)
        outputs['ad_map'] = ad_path
        
        if verbose:
            print(f"AD map: {ad_path}")
    
    # 6. Compute statistics for report
    lengths = np.array([length(s) for s in streamlines]) if len(streamlines) > 0 else np.array([])
    
    fa_valid = fa[fa > 0]
    md_valid = md[md > 0]
    
    # Default tracking parameters if not provided
    if tracking_params is None:
        tracking_params = {}
    
    scalar_stats = {
        'fa_mean': float(fa_valid.mean()) if len(fa_valid) > 0 else 0.0,
        'fa_std': float(fa_valid.std()) if len(fa_valid) > 0 else 0.0,
        'fa_median': float(np.median(fa_valid)) if len(fa_valid) > 0 else 0.0,
        'md_mean': float(md_valid.mean()) if len(md_valid) > 0 else 0.0,
        'md_std': float(md_valid.std()) if len(md_valid) > 0 else 0.0,
        'md_median': float(np.median(md_valid)) if len(md_valid) > 0 else 0.0,
    }
    
    # Add RD stats if available
    if rd is not None:
        rd_valid = rd[rd > 0]
        scalar_stats['rd_mean'] = float(rd_valid.mean()) if len(rd_valid) > 0 else 0.0
        scalar_stats['rd_std'] = float(rd_valid.std()) if len(rd_valid) > 0 else 0.0
        scalar_stats['rd_median'] = float(np.median(rd_valid)) if len(rd_valid) > 0 else 0.0
    
    # Add AD stats if available
    if ad is not None:
        ad_valid = ad[ad > 0]
        scalar_stats['ad_mean'] = float(ad_valid.mean()) if len(ad_valid) > 0 else 0.0
        scalar_stats['ad_std'] = float(ad_valid.std()) if len(ad_valid) > 0 else 0.0
        scalar_stats['ad_median'] = float(np.median(ad_valid)) if len(ad_valid) > 0 else 0.0
    
    output_files = {
        'tractogram': str(tractogram_path),
        'fa_map': str(fa_path),
        'md_map': str(md_path),
    }
    if rd is not None:
        output_files['rd_map'] = str(outputs['rd_map'])
    if ad is not None:
        output_files['ad_map'] = str(outputs['ad_map'])
    
    report = {
        'processing_info': {
            'date': datetime.now().isoformat(),
            'subject_stem': stem,
            'csttool_version': '0.0.1',
        },
        'tracking_parameters': {
            'step_size_mm': tracking_params.get('step_size', 0.5),
            'fa_threshold': tracking_params.get('fa_thresh', 0.2),
            'seed_density': tracking_params.get('seed_density', 1),
            'sh_order': tracking_params.get('sh_order', 6),
            'sphere': tracking_params.get('sphere', 'symmetric362'),
            'stopping_criterion': tracking_params.get('stopping_criterion', 'fa_threshold'),
            'relative_peak_threshold': tracking_params.get('relative_peak_threshold', 0.8),
            'min_separation_angle': tracking_params.get('min_separation_angle', 45),
        },
        'data_info': {
            'volume_shape': [int(x) for x in img.shape[:3]],
            'voxel_size_mm': [float(x) for x in img.header.get_zooms()[:3]],
            'n_gradients': int(img.shape[3]) if len(img.shape) > 3 else None,
        },
        'streamline_stats': {
            'count': len(streamlines),
            'length_mean_mm': float(lengths.mean()) if len(lengths) > 0 else 0.0,
            'length_std_mm': float(lengths.std()) if len(lengths) > 0 else 0.0,
            'length_min_mm': float(lengths.min()) if len(lengths) > 0 else 0.0,
            'length_max_mm': float(lengths.max()) if len(lengths) > 0 else 0.0,
            'length_median_mm': float(np.median(lengths)) if len(lengths) > 0 else 0.0,
        },
        'scalar_stats': scalar_stats,
        'output_files': output_files
    }
    
    # 7. Save report
    report_path = log_dir / f"{stem}_tracking_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    outputs['report'] = report_path
    
    if verbose:
        print(f"Report: {report_path}")
    
    return outputs
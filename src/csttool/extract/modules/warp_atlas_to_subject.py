"""
warp_atlas_to_subject.py

Warp MNI parcellation atlas to subject native space using registration mapping.
Uses Harvard-Oxford atlas from templateflow for ROI definition: https://nilearn.github.io/dev/modules/description/harvard_oxford.html

"""

import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import image
from nibabel.orientations import apply_orientation, ornt_transform, inv_ornt_aff


# Harvard-Oxford label definitions for CST extraction
# Subcortical atlas labels (HarvardOxford-sub)
HARVARDOXFORD_SUBCORTICAL = {
    'brainstem': 8,
    'left_thalamus': 4,
    'right_thalamus': 15,
    'left_caudate': 5,
    'right_caudate': 16,
}

# Cortical atlas labels (HarvardOxford-cort) - maxprob-thr25-2mm version
# Note: Left/Right are distinguished by hemisphere, not separate labels
HARVARDOXFORD_CORTICAL = {
    'precentral_gyrus': 7,
    'postcentral_gyrus': 17,
    'superior_frontal_gyrus': 3,
}

# CST-specific ROI configuration
CST_ROI_CONFIG = {
    'brainstem': {
        'atlas': 'subcortical',
        'label': 8,
        'description': 'Brainstem - inferior CST endpoint'
    },
    'motor_left': {
        'atlas': 'cortical',
        'label': 7,
        'hemisphere': 'left',
        'description': 'Left Precentral Gyrus - superior CST endpoint'
    },
    'motor_right': {
        'atlas': 'cortical', 
        'label': 7,
        'hemisphere': 'right',
        'description': 'Right Precentral Gyrus - superior CST endpoint'
    }
}


def reorient_to_original(data, reorientation_transform):
    """
    Apply inverse of reorientation transform to convert RAS data back to original orientation.
    
    Parameters
    ----------
    data : ndarray
        Data in RAS orientation.
    reorientation_transform : ndarray
        Transform that was used to convert original -> RAS.
        
    Returns
    -------
    reoriented_data : ndarray
        Data in original orientation.
    """
    # Compute inverse transform: RAS -> original
    # The reorientation_transform maps original axes to RAS axes
    # We need to invert this to map RAS -> original
    inverse_transform = ornt_transform(
        nib.io_orientation(np.eye(4)),  # RAS orientation
        reorientation_transform          # Original orientation in transform coords
    )
    # Actually, we need to invert: swap start_ornt and end_ornt
    # reorientation_transform was: ornt_transform(original -> RAS)
    # inverse is: ornt_transform(RAS -> original)
    n_axes = len(reorientation_transform)
    inverse_transform = np.zeros_like(reorientation_transform)
    for i, (axis, flip) in enumerate(reorientation_transform):
        axis = int(axis)
        inverse_transform[axis, 0] = i
        inverse_transform[axis, 1] = flip
    
    return apply_orientation(data, inverse_transform)


def fetch_harvard_oxford(verbose=True):
    """
    Fetch Harvard-Oxford atlases via nilearn.
    
    Downloads both cortical and subcortical parcellations in MNI152
    space at 1mm resolution.
    
    Parameters
    ----------
    verbose : bool, optional
        Print progress information. Default is True.
        
    Returns
    -------
    atlases : dict
        Dictionary containing:
        - 'cortical_path': Path to cortical atlas NIfTI
        - 'subcortical_path': Path to subcortical atlas NIfTI
        - 'cortical_img': Loaded cortical NIfTI image
        - 'subcortical_img': Loaded subcortical NIfTI image
        
    Notes
    -----
    Requires nilearn: `pip install nilearn`
    
    First call will download atlas files to nilearn's cache directory
    (~/.nilearn/data/).
    
    Atlas variants:
    - 'cort-maxprob-thr25-1mm': Cortical, max probability, 25% threshold
    - 'sub-maxprob-thr25-1mm': Subcortical, max probability, 25% threshold
    """
    try:
        from nilearn import datasets
    except ImportError:
        raise ImportError(
            "nilearn is required for Harvard-Oxford atlas. "
            "Install with: pip install nilearn"
        )
    
    if verbose:
        print("Fetching Harvard-Oxford atlas from nilearn...")
    
    # Fetch cortical atlas (contains precentral gyrus)
    if verbose:
        print("    Downloading cortical atlas...")
    cort_atlas = datasets.fetch_atlas_harvard_oxford(
        'cort-maxprob-thr25-1mm',
        # symmetric_split=False
    )
    
    # Fetch subcortical atlas (contains brainstem)
    if verbose:
        print("    Downloading subcortical atlas...")
    subcort_atlas = datasets.fetch_atlas_harvard_oxford(
        'sub-maxprob-thr25-1mm'
    )
    
    # Load the atlas images
    cort_img = cort_atlas.maps
    subcort_img = subcort_atlas.maps
    
    if verbose:
        print(f"    ✓ Cortical atlas: {cort_atlas.maps}")
        print(f"    ✓ Subcortical atlas: {subcort_atlas.maps}")
        print(f"    Cortical shape: {cort_img.shape}")
        print(f"    Subcortical shape: {subcort_img.shape}")
        print(f"    Cortical labels: {len(cort_atlas.labels)} regions")
        print(f"    Subcortical labels: {len(subcort_atlas.labels)} regions")
    
    return {
        'cortical_path': str(cort_atlas.maps),
        'subcortical_path': str(subcort_atlas.maps),
        'cortical_img': cort_img,
        'subcortical_img': subcort_img,
        'cortical_labels': cort_atlas.labels,
        'subcortical_labels': subcort_atlas.labels
    }


def resample_atlas_to_mni_grid(atlas_img, mni_shape, mni_affine, verbose=True):
    """
    Resample a Harvard-Oxford atlas to match the DIPY MNI template grid.
    
    The Harvard-Oxford atlas (182x218x182) has a different grid than the
    DIPY MNI template (197x233x189). Since the registration mapping was
    computed with the DIPY template, we must resample the atlas to the
    same grid before applying the warp.
    
    Parameters
    ----------
    atlas_img : Nifti1Image
        Atlas in Harvard-Oxford grid.
    mni_shape : tuple
        Shape of DIPY MNI template (197, 233, 189).
    mni_affine : ndarray
        4x4 affine matrix of DIPY MNI template.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    resampled_data : ndarray
        Atlas data resampled to MNI template grid.
    """
    atlas_data = atlas_img.get_fdata()
    atlas_affine = atlas_img.affine
    
    
    if verbose:
        print(f"    Resampling atlas from {atlas_data.shape} to {mni_shape}...")
    
    # Create a proxy image for the MNI template (target geometry)
    # We only need the grid definition (shape + affine), not the data
    # (Creating a dummy image is cheap)
    mni_proxy = nib.Nifti1Image(np.zeros(mni_shape), mni_affine)

    # Resample atlas to match MNI template grid
    # nilearn.image.resample_to_img robustly handles inconsistent affines/grids
    # by resampling the source image to match the target image's geometry
    resampled_img = image.resample_to_img(
        source_img=atlas_img,
        target_img=mni_proxy,
        interpolation='nearest'
    )
    
    resampled_data = resampled_img.get_fdata()
    
    if verbose:
        orig_labels = np.unique(atlas_data[atlas_data > 0])
        new_labels = np.unique(resampled_data[resampled_data > 0])
        print(f"    ✓ Resampled: {len(orig_labels)} → {len(new_labels)} labels preserved")
    
    return resampled_data


def warp_atlas_to_subject(
    atlas_img,
    mapping,
    subject_shape,
    subject_affine,
    mni_shape=None,
    mni_affine=None,
    interpolation='nearest',
    verbose=True
):
    """
    Warp MNI atlas labels to subject native space.
    
    Applies the SyN diffeomorphic mapping (from registration) to transform
    atlas labels from MNI space into the subject's native coordinate system.
    
    If the atlas grid differs from the MNI template grid used for registration,
    the atlas is first resampled to the MNI grid before warping.
    
    Parameters
    ----------
    atlas_img : Nifti1Image
        Atlas label image in MNI space.
    mapping : DiffeomorphicMap
        Registration mapping from `register_mni_to_subject()`.
    subject_shape : tuple
        Shape of subject image (X, Y, Z).
    subject_affine : ndarray
        4x4 affine matrix of subject image.
    mni_shape : tuple, optional
        Shape of MNI template used for registration. Required if atlas
        grid differs from template grid.
    mni_affine : ndarray, optional
        4x4 affine of MNI template used for registration.
    interpolation : str, optional
        Interpolation method. Must be 'nearest' for label maps.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    warped_atlas : ndarray
        Atlas labels warped to subject space. Shape matches subject_shape.
    """
    if interpolation != 'nearest':
        raise ValueError(
            f"Interpolation must be 'nearest' for label maps, got '{interpolation}'. "
            "Linear interpolation creates invalid fractional labels."
        )
    
    atlas_data = atlas_img.get_fdata()
    atlas_shape = atlas_data.shape
    
    if verbose:
        unique_labels = np.unique(atlas_data[atlas_data > 0])
        print(f"Warping atlas to subject space...")
        print(f"    Atlas shape: {atlas_shape}")
        print(f"    Target shape: {subject_shape}")
        print(f"    Unique labels: {len(unique_labels)}")
        print(f"    Interpolation: {interpolation}")
    
    # Check if atlas needs resampling to match MNI template grid
    if mni_shape is not None and atlas_shape != mni_shape:
        if verbose:
            print(f"    ⚠️  Atlas grid {atlas_shape} differs from MNI template {mni_shape}")
        atlas_data = resample_atlas_to_mni_grid(atlas_img, mni_shape, mni_affine, verbose=verbose)
    
    # Apply the mapping with nearest-neighbor interpolation
    # mapping.transform() warps from moving (MNI) to static (subject) space
    warped_atlas = mapping.transform(
        atlas_data,
        interpolation=interpolation
    )
    
    # Ensure integer labels
    warped_atlas = np.round(warped_atlas).astype(np.int16)
    
    if verbose:
        warped_unique = np.unique(warped_atlas[warped_atlas > 0])
        print(f"    ✓ Warped labels: {len(warped_unique)}")
        
        # Sanity check: labels should be preserved
        orig_labels = np.unique(atlas_img.get_fdata()[atlas_img.get_fdata() > 0])
        if len(warped_unique) != len(orig_labels):
            print(f"    ⚠️  Warning: Label count changed ({len(orig_labels)} → {len(warped_unique)})")

    
    return warped_atlas


def warp_harvard_oxford_to_subject(
    registration_result,
    output_dir=None,
    subject_id=None,
    save_warped=True,
    verbose=True
):
    """
    Complete pipeline: Fetch Harvard-Oxford and warp both atlases to subject space.
    
    Parameters
    ----------
    registration_result : dict
        Output from `register_mni_to_subject()` containing:
        - 'mapping': DiffeomorphicMap
        - 'subject_affine': Subject affine matrix
        - 'subject_shape': Subject image shape
    output_dir : str or Path, optional
        Directory for saving warped atlases. Required if save_warped=True.
    subject_id : str, optional
        Subject identifier for output filenames.
    save_warped : bool, optional
        Save warped atlases as NIfTI files. Default is True.
    verbose : bool, optional
        Print progress information. Default is True.
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'cortical_warped': Warped cortical atlas (ndarray)
        - 'subcortical_warped': Warped subcortical atlas (ndarray)
        - 'cortical_warped_path': Path to saved cortical atlas (if saved)
        - 'subcortical_warped_path': Path to saved subcortical atlas (if saved)
        - 'subject_affine': Subject affine (for creating NIfTI)
        - 'roi_config': CST_ROI_CONFIG for downstream use
    """
    if save_warped and output_dir is None:
        raise ValueError("output_dir required when save_warped=True")
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("WARP ATLAS: Harvard-Oxford → Subject Space")
        print("=" * 60)
    
    # Extract registration components
    mapping = registration_result['mapping']
    subject_shape = registration_result['subject_shape']
    subject_affine = registration_result['subject_affine']  # RAS affine (for warping)
    mni_shape = registration_result.get('mni_shape')
    mni_affine = registration_result.get('mni_affine')
    
    # Original orientation info (for saving outputs)
    original_subject_affine = registration_result.get('original_subject_affine', subject_affine)
    was_reoriented = registration_result.get('was_reoriented', False)
    reorientation_transform = registration_result.get('reorientation_transform')
    
    # Step 1: Fetch atlases
    if verbose:
        print("\n[Step 1/3] Fetching Harvard-Oxford atlases...")
    
    atlases = fetch_harvard_oxford(verbose=verbose)
    
    # Step 2: Warp subcortical atlas (contains brainstem)
    if verbose:
        print("\n[Step 2/3] Warping subcortical atlas...")
    
    subcortical_warped = warp_atlas_to_subject(
        atlas_img=atlases['subcortical_img'],
        mapping=mapping,
        subject_shape=subject_shape,
        subject_affine=subject_affine,
        mni_shape=mni_shape,
        mni_affine=mni_affine,
        verbose=verbose
    )
    
    # Step 3: Warp cortical atlas (contains precentral gyrus)
    if verbose:
        print("\n[Step 3/3] Warping cortical atlas...")
    
    cortical_warped = warp_atlas_to_subject(
        atlas_img=atlases['cortical_img'],
        mapping=mapping,
        subject_shape=subject_shape,
        subject_affine=subject_affine,
        mni_shape=mni_shape,
        mni_affine=mni_affine,
        verbose=verbose
    )
    
    # Prepare result
    result = {
        'cortical_warped': cortical_warped,
        'subcortical_warped': subcortical_warped,
        'cortical_warped_path': None,
        'subcortical_warped_path': None,
        'subject_affine': subject_affine,  # RAS affine (for internal processing)
        'original_subject_affine': original_subject_affine,  # Original affine (for saving)
        'was_reoriented': was_reoriented,
        'reorientation_transform': reorientation_transform,
        'roi_config': CST_ROI_CONFIG
    }
    
    # Save warped atlases
    if save_warped:
        if verbose:
            print("\nSaving warped atlases...")

        nifti_dir = output_dir / "nifti"
        nifti_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{subject_id}_" if subject_id else ""
        
        # Determine which affine and data to use for saving
        # If subject was reoriented for registration, transform data back to original orientation
        if was_reoriented and reorientation_transform is not None:
            if verbose:
                print("    Transforming to original orientation for saving...")
            subcortical_to_save = reorient_to_original(subcortical_warped, reorientation_transform)
            cortical_to_save = reorient_to_original(cortical_warped, reorientation_transform)
            save_affine = original_subject_affine
        else:
            subcortical_to_save = subcortical_warped
            cortical_to_save = cortical_warped
            save_affine = subject_affine
        
        # Save subcortical
        subcort_path = nifti_dir / f"{prefix}harvard_oxford_subcortical_warped.nii.gz"
        nib.save(
            nib.Nifti1Image(subcortical_to_save, save_affine),
            subcort_path
        )
        result['subcortical_warped_path'] = subcort_path
        if verbose:
            print(f"    ✓ Subcortical: {subcort_path}")
        
        # Save cortical
        cort_path = nifti_dir / f"{prefix}harvard_oxford_cortical_warped.nii.gz"
        nib.save(
            nib.Nifti1Image(cortical_to_save, save_affine),
            cort_path
        )
        result['cortical_warped_path'] = cort_path
        if verbose:
            print(f"    ✓ Cortical: {cort_path}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("Atlas warping complete!")
        print("=" * 60)
    
    return result


def verify_atlas_labels(warped_atlas, expected_labels, atlas_name="atlas", verbose=True):
    """
    Verify that expected labels exist in warped atlas.
    
    Parameters
    ----------
    warped_atlas : ndarray
        Warped atlas label array.
    expected_labels : list of int
        Label values that should be present.
    atlas_name : str, optional
        Name for verbose output.
    verbose : bool, optional
        Print verification results.
        
    Returns
    -------
    verification : dict
        Dictionary with 'success' bool and 'missing' list.
    """
    present_labels = np.unique(warped_atlas[warped_atlas > 0])
    missing = [l for l in expected_labels if l not in present_labels]
    
    if verbose:
        print(f"\nVerifying {atlas_name} labels:")
        for label in expected_labels:
            status = "✓" if label in present_labels else "✗ MISSING"
            voxel_count = np.sum(warped_atlas == label)
            print(f"    Label {label}: {status} ({voxel_count:,} voxels)")
    
    return {
        'success': len(missing) == 0,
        'missing': missing,
        'present': list(present_labels)
    }
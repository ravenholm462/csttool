"""
Registration module for csttool's pipeline.

Given an FA map as a NIfTI file and a template (MNI 152 used here), the module handles spatial registration between
MNI template space (static image) and subject native space (moving image).
"""

# Imports

import numpy as np
import nibabel as nib
from pathlib import Path

# Consolidating imports at top level to fix patching issues
from dipy.align.imaffine import (
    transform_centers_of_mass,
    MutualInformationMetric,
    AffineRegistration,
    AffineMap
)
from dipy.align.transforms import (
    TranslationTransform3D,
    RigidTransform3D,
    AffineTransform3D
)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.data import fetch_mni_template, read_mni_template
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation
from dipy.viz import regtools
import json
from datetime import datetime

def load_mni_template(contrast="T1"):
    
    print(f"Fetching MNI template (contrast: {contrast})")

    fetch_mni_template()
    template_img = read_mni_template(contrast=contrast)

    template_data = template_img.get_fdata()
    template_affine = template_img.affine


    print(f"MNI template loaded: shape {template_data.shape}")

    return template_img, template_data, template_affine

# use dipy.viz.regtools.overlay_slices to show comparison before and after registration
# https://docs.dipy.org/1.0.0/examples_built/affine_registration_3d.html


# Issue: subject acquisition orientation not always same as MNI (RAS)
# Solution: helper function to reorient subject image before registration
def reorient_to_ras(img):
    """
    Reorient a NIfTI image to RAS+ orientation.
    
    Parameters
    ----------
    img : Nifti1Image
        Input image in any orientation.
        
    Returns
    -------
    reoriented : Nifti1Image
        Image reoriented to RAS+.
    """
    current_ornt = nib.io_orientation(img.affine)
    target_ornt = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(current_ornt, target_ornt)
    
    reoriented_data = apply_orientation(img.get_fdata(), transform)
    reoriented_affine = img.affine @ nib.orientations.inv_ornt_aff(transform, img.shape)
    
    return nib.Nifti1Image(reoriented_data, reoriented_affine, img.header)


def compute_affine_registration(
    static_image,
    static_affine,
    moving_image,
    moving_affine,
    nbins=32,
    sampling_prop=None,
    level_iters=[10000, 1000, 100],
    sigmas=[3.0, 1.0, 0.0],
    factors=[4, 2, 1],
    verbose=True
):
    """
    Compute progressive affine registration using mutual information.
    
    Aligns the moving image to the static image through a series of
    increasingly complex transformations: center of mass → translation 
    → rigid body → full affine. Uses a multi-resolution Gaussian pyramid
    strategy (similar to ANTs) to avoid local optima and accelerate 
    convergence.

    References
    ----------
    Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., Eubank, W. (2003). 
        PET-CT image registration in the chest using free-form deformations. 
        IEEE Transactions on Medical Imaging, 22(1), 120-8. Avants11(1,2)

    Avants, B. B., Tustison, N., & Song, G. (2011). Advanced Normalization Tools (ANTS), 1-35
    
    Parameters
    ----------
    static_image : ndarray
        Reference image (e.g., subject FA map). Shape (X, Y, Z).
    static_affine : ndarray
        4x4 affine matrix of the static image.
    moving_image : ndarray
        Image to be aligned (e.g., MNI template). Shape (X, Y, Z).
    moving_affine : ndarray
        4x4 affine matrix of the moving image.
    nbins : int, optional
        Number of bins for discretizing the joint and marginal probability
        distribution functions (PDFs) in mutual information calculation.
        Default is 32.
    sampling_prop : int or None, optional
        Percentage of voxels (0-100) to use for computing PDFs. Using all
        voxels (None) gives the most accurate registration but is slower.
        Default is None (full sampling).
    level_iters : list of int, optional
        Number of iterations at each resolution level of the Gaussian pyramid.
        Length determines number of resolutions. Default is [10000, 1000, 100]
        (3 levels: coarse, medium, fine).
    sigmas : list of float, optional
        Gaussian kernel smoothing sigma at each pyramid level. Should have
        same length as level_iters. Default is [3.0, 1.0, 0.0].
    factors : list of int, optional
        Sub-sampling factors at each pyramid level. For original shape 
        (nx, ny, nz): factor 4 gives ~(nx/4, ny/4, nz/4), factor 2 gives
        ~(nx/2, ny/2, nz/2), factor 1 gives original size.
        Default is [4, 2, 1].
    verbose : bool, optional
        Print progress information. Default is True.
        
    Returns
    -------
    affine_map : AffineMap
        The final affine transformation. Use affine_map.transform(image)
        to warp images, or affine_map.affine to get the 4x4 matrix.
        
    Notes
    -----
    Code inspired by: https://docs.dipy.org/1.0.0/examples_built/affine_registration_3d.html

    The registration proceeds in stages, each using the previous result
    as initialization:
    
    1. **Center of mass**: Aligns image centroids (fast, coarse)
    2. **Translation**: Optimizes 3 parameters (x, y, z shifts)
    3. **Rigid**: Optimizes 6 parameters (3 rotations + 3 translations)
    4. **Affine**: Optimizes 12 parameters (adds scaling + shearing)
    
    """

    if verbose:
        print("Computing affine registration...")
        print(f"    Static (subject): {static_image.shape}")
        print(f"    Moving (template): {moving_image.shape}")

    # Stage 0: Center of mass alignment
    if verbose:
        print("    Stage 0/3: Aligning centers of mass...")
    c_of_mass = transform_centers_of_mass(
        static_image, static_affine,
        moving_image, moving_affine
    )

    # Setup registration
    if verbose:
        print("    Setting up affine registration...")
        print(f"        Level iters: {level_iters}")
        print(f"        Sigmas: {sigmas}")
        print(f"        Factors: {factors}")

    metric = MutualInformationMetric(nbins=nbins, sampling_proportion=sampling_prop)
    affreg = AffineRegistration(
        metric=metric,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors
    )

    # Stage 1: Translation
    if verbose:
        print("    Stage 1/3: Translation...")
    translation = affreg.optimize(
        static_image, moving_image,
        TranslationTransform3D(), None,
        static_grid2world=static_affine, 
        moving_grid2world=moving_affine,
        starting_affine=c_of_mass.affine
    )

    # Stage 2: Rigid
    if verbose:
        print("    Stage 2/3: Rigid...")
    rigid = affreg.optimize(
        static_image, moving_image,
        RigidTransform3D(), None,
        static_grid2world=static_affine, 
        moving_grid2world=moving_affine,
        starting_affine=translation.affine
    )

    # Stage 3: Full affine
    if verbose:
        print("    Stage 3/3: Affine...")
    affine = affreg.optimize(
        static_image, moving_image,
        AffineTransform3D(), None,
        static_grid2world=static_affine, 
        moving_grid2world=moving_affine,
        starting_affine=rigid.affine
    )

    if verbose:
        print("✓ Affine registration complete")

    return affine


def compute_syn_registration(
    static_image,
    static_affine,
    moving_image,
    moving_affine,
    prealign=None,
    level_iters=[10, 10, 5],
    metric_radius=4,
    verbose=True
):
    """
    Compute non-linear SyN (Symmetric Normalization) registration.
    
    SyN provides diffeomorphic (smooth, invertible) transformations that
    capture local anatomical variations not handled by affine registration.
    This should be run after affine registration, using the affine result
    as pre-alignment.

    References
    ----------
    [Avants08] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. 
       (2008). Symmetric diffeomorphic image registration with cross-correlation: 
       evaluating automated labeling of elderly and neurodegenerative brain. 
       Medical image analysis, 12(1), 26-41.
    
    Parameters
    ----------
    static_image : ndarray
        Reference image (e.g., subject FA map). Shape (X, Y, Z).
    static_affine : ndarray
        4x4 affine matrix of the static image.
    moving_image : ndarray
        Image to be aligned (e.g., MNI template). Shape (X, Y, Z).
    moving_affine : ndarray
        4x4 affine matrix of the moving image.
    prealign : ndarray or None, optional
        4x4 affine matrix for pre-alignment (typically from affine 
        registration). If None, no pre-alignment is applied.
    level_iters : list of int, optional
        Number of iterations at each resolution level. Length determines
        number of resolutions. Default is [10, 10, 5] (3 levels).
        Note: SyN is computationally expensive, so fewer iterations are
        used compared to affine registration.
    metric_radius : int, optional
        Radius (in voxels) for the cross-correlation metric kernel.
        Larger values capture more global structure but are slower.
        Default is 4.
    verbose : bool, optional
        Print progress information. Default is True.
        
    Returns
    -------
    mapping : DiffeomorphicMap
        The non-linear transformation mapping. Use mapping.transform(image)
        to warp images from moving to static space, or mapping.transform_inverse(image)
        to warp from static to moving space.

    Notes
    -------
    Code inspired by: https://docs.dipy.org/dev/examples_built/registration/syn_registration_3d.html
    """

    if verbose:
        print("Computing SyN non-linear registration...")
        print(f"    Static (subject): {static_image.shape}")
        print(f"    Moving (template): {moving_image.shape}")
        print(f"    Metric: Cross-Correlation (radius={metric_radius})")
        print(f"    Level iterations: {level_iters}")
        if prealign is not None:
            print("    Using affine pre-alignment")

    # Cross-correlation metric
    # dim=3 for 3D images, sigma_diff controls gradient smoothing
    metric = CCMetric(dim=3, sigma_diff=metric_radius)
    
    # Initialize SyN registration
    sdr = SymmetricDiffeomorphicRegistration(
        metric=metric,
        level_iters=level_iters
    )
    
    # Compute diffeomorphic mapping
    if verbose:
        print("    Optimizing diffeomorphic transformation...")
    
    mapping = sdr.optimize(
        static_image, moving_image,
        static_grid2world=static_affine,
        moving_grid2world=moving_affine,
        prealign=prealign
    )
    
    if verbose:
        print("✓ SyN registration complete")
    
    return mapping


def register_mni_to_subject(
    subject_fa_path,
    output_dir,
    mni_template_path=None,
    level_iters_affine=[10000, 1000, 100],
    sigmas_affine=[3.0, 1.0, 0.0],
    factors_affine=[4, 2, 1],
    level_iters_syn=[10, 10, 5],
    metric_radius_syn=4,
    save_warped=True,
    generate_qc=True,
    verbose=True
):
    """
    Complete registration pipeline: MNI template to subject space.
    
    Performs progressive affine registration followed by SyN non-linear
    registration to align the MNI template to the subject's native space.
    This mapping is used downstream to warp the MNI parcellation atlas
    into subject space for ROI-based CST extraction.
    
    Pipeline: MNI Template + Subject FA → Affine + SyN → Mapping
    
    Parameters
    ----------
    subject_fa_path : str or Path
        Path to subject's FA map (.nii.gz).
    output_dir : str or Path
        Directory for saving registration outputs.
    mni_template_path : str or Path, optional
        Path to MNI template. If None, fetches DIPY's MNI template.
    level_iters_affine : list of int, optional
        Iterations per resolution level for affine registration.
        Default is [10000, 1000, 100].
    sigmas_affine : list of float, optional
        Gaussian smoothing sigmas for affine registration.
        Default is [3.0, 1.0, 0.0].
    factors_affine : list of int, optional
        Sub-sampling factors for affine registration.
        Default is [4, 2, 1].
    level_iters_syn : list of int, optional
        Iterations per resolution level for SyN registration.
        Default is [10, 10, 5].
    metric_radius_syn : int, optional
        Radius for cross-correlation metric in SyN. Default is 4.
    save_warped : bool, optional
        Save warped template for visual inspection. Default is True.
    generate_qc : bool, optional
        Generate QC visualizations. Default is True.
    verbose : bool, optional
        Print progress information. Default is True.
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'mapping': DiffeomorphicMap for transforming images/atlases
        - 'affine_map': AffineMap from affine registration stage
        - 'subject_affine': Subject image affine matrix
        - 'subject_shape': Subject image shape
        - 'mni_affine': MNI template affine matrix
        - 'mni_shape': MNI template shape
        - 'warped_template_path': Path to warped template (if saved)
        - 'qc_before_path': Path to pre-registration QC image (if generated)
        - 'qc_after_path': Path to post-registration QC image (if generated)
    
    Notes
    -----
    The registration direction is MNI → Subject (template to native space).
    This allows us to warp the MNI parcellation atlas into subject space,
    preserving streamline coordinates in their native space.
    """
    subject_fa_path = Path(subject_fa_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract subject ID from filename
    subject_id = subject_fa_path.stem.replace('_fa', '').replace('.nii', '')
    
    if verbose:
        print("=" * 60)
        print("REGISTRATION: MNI Template → Subject Space")
        print("=" * 60)
        print(f"Subject: {subject_id}")
        print(f"Output directory: {output_dir}")
    
    # -------------------------------------------------------------------------
    # Step 1: Load subject FA map
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 1/5] Loading subject FA map...")
    
    subject_img = nib.load(subject_fa_path)
    subject_data = subject_img.get_fdata()
    subject_affine = subject_img.affine
    
    if verbose:
        voxel_size = np.sqrt(np.sum(subject_affine[:3, :3]**2, axis=0))
        print(f"    Shape: {subject_data.shape}")
        print(f"    Voxel size: {voxel_size.round(2)} mm")
    
    # -------------------------------------------------------------------------
    # Step 2: Load MNI template
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 2/5] Loading MNI template...")
    
    if mni_template_path is not None:
        mni_img = nib.load(mni_template_path)
    else:
        fetch_mni_template()
        mni_img = read_mni_template(contrast="T1")
    
    mni_data = mni_img.get_fdata()
    mni_affine = mni_img.affine
    
    if verbose:
        print(f"    Shape: {mni_data.shape}")

    # -------------------------------------------------------------------------
    # Step 2.5: Reorient subject to MNI if necessary
    # -------------------------------------------------------------------------

    # Reorient to RAS if needed
    if nib.aff2axcodes(subject_img.affine) != ('R', 'A', 'S'):
        if verbose:
            print(f"    Reorienting from {nib.aff2axcodes(subject_img.affine)} to RAS...")
        subject_img = reorient_to_ras(subject_img)

    subject_data = subject_img.get_fdata()
    subject_affine = subject_img.affine
    
    # -------------------------------------------------------------------------
    # Step 3: Affine registration
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 3/5] Affine registration...")
    
    affine_map = compute_affine_registration(
        static_image=subject_data,
        static_affine=subject_affine,
        moving_image=mni_data,
        moving_affine=mni_affine,
        level_iters=level_iters_affine,
        sigmas=sigmas_affine,
        factors=factors_affine,
        verbose=verbose
    )
    
    # -------------------------------------------------------------------------
    # Step 4: SyN non-linear registration
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 4/5] SyN non-linear registration...")
    
    mapping = compute_syn_registration(
        static_image=subject_data,
        static_affine=subject_affine,
        moving_image=mni_data,
        moving_affine=mni_affine,
        prealign=affine_map.affine,
        level_iters=level_iters_syn,
        metric_radius=metric_radius_syn,
        verbose=verbose
    )
    
    # -------------------------------------------------------------------------
    # Step 5: Save outputs and generate QC
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 5/5] Saving outputs...")
    
    result = {
        'mapping': mapping,
        'affine_map': affine_map,
        'subject_affine': subject_affine,
        'subject_shape': subject_data.shape,
        'mni_affine': mni_affine,
        'mni_shape': mni_data.shape,
        'warped_template_path': None,
        'qc_before_path': None,
        'qc_after_path': None
    }
    
    # Save warped template
    if save_warped:
        nifti_dir = output_dir / "nifti"
        nifti_dir.mkdir(parents=True, exist_ok=True)

        warped_template = mapping.transform(mni_data)
        warped_path = nifti_dir / f"{subject_id}_mni_warped_to_subject.nii.gz"
        nib.save(
            nib.Nifti1Image(warped_template.astype(np.float32), subject_affine),
            warped_path
        )
        result['warped_template_path'] = warped_path
        if verbose:
            print(f"    ✓ Warped template: {warped_path}")
    
    # Generate QC visualizations
    # if generate_qc:
    #     viz_dir = output_dir / "visualizations"
    #     viz_dir.mkdir(parents=True, exist_ok=True)
        
    #     # Resample MNI to subject grid with identity (before registration)
    #     identity_map = AffineMap(
    #         np.eye(4),
    #         subject_data.shape, subject_affine,
    #         mni_data.shape, mni_affine
    #     )
    #     mni_resampled_before = identity_map.transform(mni_data)
        
    #     # Before registration QC
    #     qc_before_path = viz_dir / f"{subject_id}_registration_qc_before.png"
    #     plot_registration_comparison(
    #         static_data=subject_data,
    #         moving_data=mni_resampled_before,
    #         title=f"Before Registration - {subject_id}",
    #         output_path=qc_before_path
    #     )
    #     result['qc_before_path'] = qc_before_path
        
    #     # After registration QC
    #     warped_template = mapping.transform(mni_data)
    #     qc_after_path = viz_dir / f"{subject_id}_registration_qc_after.png"
    #     plot_registration_comparison(
    #         static_data=subject_data,
    #         moving_data=warped_template,
    #         title=f"After Registration - {subject_id}",
    #         output_path=qc_after_path
    #     )
    #     result['qc_after_path'] = qc_after_path
        
    #     if verbose:
    #         print(f"    ✓ QC before: {qc_before_path}")
    #         print(f"    ✓ QC after: {qc_after_path}")
    
    # Save registration report
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'subject_id': subject_id,
        'registration_type': 'Affine + SyN',
        'direction': 'MNI → Subject',
        'parameters': {
            'affine': {
                'level_iters': level_iters_affine,
                'sigmas': sigmas_affine,
                'factors': factors_affine
            },
            'syn': {
                'level_iters': level_iters_syn,
                'metric_radius': metric_radius_syn
            }
        },
        'subject': {
            'fa_path': str(subject_fa_path),
            'shape': list(subject_data.shape)
        },
        'mni': {
            'shape': list(mni_data.shape)
        },
        'outputs': {
            'warped_template': str(result['warped_template_path']) if result['warped_template_path'] else None,
            'qc_before': str(result['qc_before_path']) if result['qc_before_path'] else None,
            'qc_after': str(result['qc_after_path']) if result['qc_after_path'] else None
        }
    }
    
    report_path = log_dir / f"{subject_id}_registration_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    if verbose:
        print(f"    ✓ Report: {report_path}")
        print("\n" + "=" * 60)
        print("Registration complete!")
        print("=" * 60)
    
    return result


def save_registration_report(result, output_dir, subject_id):
    """
    Save registration report with transformation details.
    
    Parameters
    ----------
    result : dict
        Output from register_mni_to_subject()
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
        
    Returns
    -------
    report_path : Path
        Path to saved report
    """
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get affine matrix from AffineMap object
    affine_matrix = result['affine_map'].affine
    
    report = {
        'processing_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'subject_id': subject_id,
        'registration_type': 'Affine + SyN',
        'direction': 'MNI → Subject',
        'subject_shape': list(result['subject_shape']),
        'mni_shape': list(result['mni_shape']),
        'affine_matrix': affine_matrix.tolist(),
        'warped_template_path': str(result['warped_template_path']) if result['warped_template_path'] else None
    }
    
    report_path = log_dir / f"{subject_id}_registration_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Registration report saved: {report_path}")
    
    return report_path


def plot_registration_comparison(
    static_data,
    moving_data,
    slice_indices=None,
    ltitle="Static",
    rtitle="Moving",
    output_dir=None,
    fname_prefix="registration_qc"
):
    """
    Plot registration comparison for all three views using DIPY's regtools.overlay_slices.
    
    Parameters
    ----------
    static_data : ndarray
        Reference image (e.g., subject FA map). Shape (X, Y, Z).
    moving_data : ndarray
        Moving image, must be resampled to same grid as static.
    slice_indices : dict, optional
        Dictionary with keys 'sagittal', 'coronal', 'axial' specifying 
        slice indices. If None, uses middle slices.
    ltitle : str, optional
        Title for static image.
    rtitle : str, optional
        Title for moving image.
    output_dir : str or Path, optional
        Directory to save images. If None, images are not saved.
    fname_prefix : str, optional
        Prefix for saved filenames.
        
    Returns
    -------
    figs : dict
        Dictionary with keys 'sagittal', 'coronal', 'axial' containing figures.
    """
    sh = static_data.shape
    
    # Determine slice indices (middle if not specified)
    if slice_indices is None:
        slice_indices = {
            'sagittal': sh[0] // 2,
            'coronal': sh[1] // 2,
            'axial': sh[2] // 2
        }
    
    # Convert output_dir to Path once (if provided)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map view names to slice_type integers
    views = {
        'sagittal': 0,
        'coronal': 1,
        'axial': 2
    }
    
    figs = {}
    
    for view_name, slice_type in views.items():
        # Determine output filename
        fname = None
        if output_dir is not None:
            fname = str(output_dir / f"{fname_prefix}_{view_name}.png")
        
        # Call DIPY's overlay_slices
        fig = regtools.overlay_slices(
            static_data,
            moving_data,
            slice_index=slice_indices[view_name],
            slice_type=slice_type,
            ltitle=ltitle,
            rtitle=rtitle,
            fname=fname
        )
        
        figs[view_name] = fig
        
        if fname:
            print(f"✓ Saved: {fname}")
    
    return figs


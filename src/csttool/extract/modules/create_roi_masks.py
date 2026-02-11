"""
create_roi_masks.py

Extract binary ROI masks from warped Harvard-Oxford atlases for CST filtering.
Handles hemisphere separation for cortical regions.

Pipeline position: Registration → Warp Atlas → **Create ROI Masks** → Endpoint Filtering
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import binary_dilation
from nibabel.orientations import apply_orientation, ornt_transform


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
    n_axes = len(reorientation_transform)
    inverse_transform = np.zeros_like(reorientation_transform)
    for i, (axis, flip) in enumerate(reorientation_transform):
        axis = int(axis)
        inverse_transform[axis, 0] = i
        inverse_transform[axis, 1] = flip
    
    return apply_orientation(data, inverse_transform)


def extract_roi_mask(warped_atlas, label, verbose=True):
    """
    Extract a binary mask for a specific label from warped atlas.

    Parameters
    ----------
    warped_atlas : ndarray
        Warped atlas label array in subject space.
    label : int
        Label value to extract.
    verbose : bool, optional
        Print mask statistics.

    Returns
    -------
    mask : ndarray
        Binary mask (bool) for the specified label.
    """
    mask = warped_atlas == label

    if verbose:
        voxel_count = np.sum(mask)
        print(f"    • Extracted label {label}: {voxel_count:,} voxels")

    return mask





def dilate_mask(mask, iterations=1, verbose=True):
    """
    Dilate a binary mask to increase spatial coverage.

    Useful for endpoint filtering to catch streamlines that terminate
    near but not exactly within the ROI.

    Parameters
    ----------
    mask : ndarray
        Binary mask to dilate.
    iterations : int, optional
        Number of dilation iterations. Default is 1.
    verbose : bool, optional
        Print dilation statistics.

    Returns
    -------
    dilated : ndarray
        Dilated binary mask.
    """
    if iterations <= 0:
        return mask

    original_count = np.sum(mask)
    dilated = binary_dilation(mask, iterations=iterations)
    dilated_count = np.sum(dilated)

    if verbose:
        print(f"    • Dilation ({iterations} iterations): {original_count:,} → {dilated_count:,} voxels")

    return dilated


def create_cst_roi_masks(
    warped_cortical,
    warped_subcortical,
    subject_affine,
    roi_config,
    dilate_brainstem=2,
    dilate_motor=1,
    output_dir=None,
    subject_id=None,
    save_masks=True,
    verbose=True,
    original_subject_affine=None,
    reorientation_transform=None
):
    """
    Create all ROI masks needed for bilateral CST extraction.
    
    Extracts brainstem and bilateral motor cortex masks from warped
    Harvard-Oxford atlases.
    
    Parameters
    ----------
    warped_cortical : ndarray
        Warped cortical atlas in subject space.
    warped_subcortical : ndarray
        Warped subcortical atlas in subject space.
    subject_affine : ndarray
        4x4 affine matrix for subject space.
    roi_config : dict
        ROI configuration dictionary (CST_ROI_CONFIG).
    dilate_brainstem : int, optional
        Dilation iterations for brainstem mask. Default is 2.
    dilate_motor : int, optional
        Dilation iterations for motor cortex masks. Default is 1.
    output_dir : str or Path, optional
        Directory for saving masks. Required if save_masks=True.
    subject_id : str, optional
        Subject identifier for output filenames.
    save_masks : bool, optional
        Save masks as NIfTI files. Default is True.
    verbose : bool, optional
        Print progress information. Default is True.
        
    Returns
    -------
    masks : dict
        Dictionary containing:
        - 'brainstem': Brainstem mask (dilated)
        - 'motor_left': Left motor cortex mask (dilated)
        - 'motor_right': Right motor cortex mask (dilated)
        - 'brainstem_path': Path to saved brainstem mask (if saved)
        - 'motor_left_path': Path to saved left motor mask (if saved)
        - 'motor_right_path': Path to saved right motor mask (if saved)
        
    Notes
    -----
    If original_subject_affine and reorientation_transform are provided,
    masks will be saved in the original subject orientation (e.g., LAS)
    rather than the RAS orientation used internally for processing.
    """
    if save_masks and output_dir is None:
        raise ValueError("output_dir required when save_masks=True")
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("CREATE ROI MASKS: CST Extraction")
        print("=" * 60)
    
    masks = {
        'brainstem': None,
        'motor_left': None,
        'motor_right': None,
        'brainstem_path': None,
        'motor_left_path': None,
        'motor_right_path': None,
        'subject_affine': subject_affine
    }
    
    # -------------------------------------------------------------------------
    # Step 1: Extract brainstem mask
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 1/3] Extracting brainstem mask...")

    brainstem_label = roi_config['brainstem']['label']
    brainstem_raw = extract_roi_mask(warped_subcortical, brainstem_label, verbose=verbose)

    if dilate_brainstem > 0:
        brainstem_mask = dilate_mask(brainstem_raw, iterations=dilate_brainstem, verbose=verbose)
    else:
        brainstem_mask = brainstem_raw

    masks['brainstem'] = brainstem_mask

    # -------------------------------------------------------------------------
    # Step 2: Extract motor cortex masks (using separate L/R labels)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 2/3] Extracting motor cortex masks...")

    # Left Motor (Label 7)
    motor_left_label = roi_config['motor_left']['label']
    if verbose:
        print("  → Left Hemisphere:")
    motor_left_raw = extract_roi_mask(warped_cortical, motor_left_label, verbose=verbose)

    # Right Motor (Label 107)
    motor_right_label = roi_config['motor_right']['label']
    if verbose:
        print("  → Right Hemisphere:")
    motor_right_raw = extract_roi_mask(warped_cortical, motor_right_label, verbose=verbose)

    # -------------------------------------------------------------------------
    # Step 3: Dilate masks
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 3/3] Dilating masks...")

    # Dilate motor masks
    if dilate_motor > 0:
        if verbose:
            print("  → Dilating motor masks...")
        motor_left_mask = dilate_mask(motor_left_raw, iterations=dilate_motor, verbose=verbose)
        motor_right_mask = dilate_mask(motor_right_raw, iterations=dilate_motor, verbose=verbose)
    else:
        motor_left_mask = motor_left_raw
        motor_right_mask = motor_right_raw

    masks['motor_left'] = motor_left_mask
    masks['motor_right'] = motor_right_mask
    
    # -------------------------------------------------------------------------
    # Save masks
    # -------------------------------------------------------------------------
    if save_masks:
        if verbose:
            print("\nSaving ROI masks...")

        nifti_dir = output_dir / "nifti"
        nifti_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{subject_id}_" if subject_id else ""

        # Determine which affine and data to use for saving
        # If subject was reoriented for registration, transform data back to original orientation
        if original_subject_affine is not None and reorientation_transform is not None:
            if verbose:
                print("    • Transforming to original orientation for saving...")
            brainstem_to_save = reorient_to_original(brainstem_mask.astype(np.uint8), reorientation_transform)
            motor_left_to_save = reorient_to_original(motor_left_mask.astype(np.uint8), reorientation_transform)
            motor_right_to_save = reorient_to_original(motor_right_mask.astype(np.uint8), reorientation_transform)
            save_affine = original_subject_affine
        else:
            brainstem_to_save = brainstem_mask.astype(np.uint8)
            motor_left_to_save = motor_left_mask.astype(np.uint8)
            motor_right_to_save = motor_right_mask.astype(np.uint8)
            save_affine = subject_affine

        # Save brainstem
        brainstem_path = nifti_dir / f"{prefix}roi_brainstem.nii.gz"
        nib.save(
            nib.Nifti1Image(brainstem_to_save, save_affine),
            brainstem_path
        )
        masks['brainstem_path'] = brainstem_path
        if verbose:
            print(f"  ✓ Brainstem: {brainstem_path}")

        # Save motor left
        motor_left_path = nifti_dir / f"{prefix}roi_motor_left.nii.gz"
        nib.save(
            nib.Nifti1Image(motor_left_to_save, save_affine),
            motor_left_path
        )
        masks['motor_left_path'] = motor_left_path
        if verbose:
            print(f"  ✓ Motor Left: {motor_left_path}")

        # Save motor right
        motor_right_path = nifti_dir / f"{prefix}roi_motor_right.nii.gz"
        nib.save(
            nib.Nifti1Image(motor_right_to_save, save_affine),
            motor_right_path
        )
        masks['motor_right_path'] = motor_right_path
        if verbose:
            print(f"  ✓ Motor Right: {motor_right_path}")

        # Save combined visualization mask
        combined = np.zeros_like(brainstem_mask, dtype=np.uint8)
        combined[brainstem_mask] = 1
        combined[motor_left_mask] = 2
        combined[motor_right_mask] = 3

        combined_path = nifti_dir / f"{prefix}roi_cst_combined.nii.gz"
        if original_subject_affine is not None and reorientation_transform is not None:
            combined_to_save = reorient_to_original(combined, reorientation_transform)
        else:
            combined_to_save = combined
        nib.save(
            nib.Nifti1Image(combined_to_save, save_affine),
            combined_path
        )
        masks['combined_path'] = combined_path
        if verbose:
            print(f"  ✓ Combined: {combined_path}")

    if verbose:
        print("\n" + "=" * 60)
        print("  ✓ ROI mask creation complete")
        print("=" * 60)
        print("\nSummary:")
        print(f"  Brainstem:   {np.sum(masks['brainstem']):,} voxels")
        print(f"  Motor Left:  {np.sum(masks['motor_left']):,} voxels")
        print(f"  Motor Right: {np.sum(masks['motor_right']):,} voxels")
    
    return masks


def visualize_roi_masks(
    masks,
    subject_fa,
    output_dir,
    subject_id=None,
    verbose=True
):
    """
    Create visualization of ROI masks overlaid on subject FA.
    
    Parameters
    ----------
    masks : dict
        Output from create_cst_roi_masks().
    subject_fa : ndarray
        Subject FA map for background.
    output_dir : str or Path
        Directory for saving visualization.
    subject_id : str, optional
        Subject identifier for filename.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    fig_path : Path
        Path to saved visualization.
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get middle slices
    shape = subject_fa.shape
    mid_sag = shape[0] // 2
    mid_cor = shape[1] // 2
    mid_ax = shape[2] // 2
    
    # Create combined ROI mask for visualization
    roi_combined = np.zeros(shape, dtype=np.float32)
    roi_combined[masks['brainstem']] = 1.0
    roi_combined[masks['motor_left']] = 2.0
    roi_combined[masks['motor_right']] = 3.0
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'CST ROI Masks - {subject_id or "Subject"}', fontsize=14)
    
    # Custom colormap for ROIs
    from matplotlib.colors import ListedColormap
    roi_cmap = ListedColormap(['none', 'red', 'blue', 'green'])
    
    # Row 1: Individual masks
    axes[0, 0].imshow(subject_fa[:, :, mid_ax].T, cmap='gray', origin='lower')
    axes[0, 0].imshow(np.ma.masked_where(masks['brainstem'][:, :, mid_ax].T == 0,
                                          masks['brainstem'][:, :, mid_ax].T),
                      cmap='Reds', alpha=0.6, origin='lower')
    axes[0, 0].set_title(f"Brainstem (axial)\n{np.sum(masks['brainstem']):,} voxels")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(subject_fa[:, :, mid_ax].T, cmap='gray', origin='lower')
    axes[0, 1].imshow(np.ma.masked_where(masks['motor_left'][:, :, mid_ax].T == 0,
                                          masks['motor_left'][:, :, mid_ax].T),
                      cmap='Blues', alpha=0.6, origin='lower')
    axes[0, 1].set_title(f"Motor Left (axial)\n{np.sum(masks['motor_left']):,} voxels")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(subject_fa[:, :, mid_ax].T, cmap='gray', origin='lower')
    axes[0, 2].imshow(np.ma.masked_where(masks['motor_right'][:, :, mid_ax].T == 0,
                                          masks['motor_right'][:, :, mid_ax].T),
                      cmap='Greens', alpha=0.6, origin='lower')
    axes[0, 2].set_title(f"Motor Right (axial)\n{np.sum(masks['motor_right']):,} voxels")
    axes[0, 2].axis('off')
    
    # Row 2: All ROIs combined in different views
    axes[1, 0].imshow(subject_fa[mid_sag, :, :].T, cmap='gray', origin='lower')
    axes[1, 0].imshow(np.ma.masked_where(roi_combined[mid_sag, :, :].T == 0,
                                          roi_combined[mid_sag, :, :].T),
                      cmap=roi_cmap, alpha=0.6, origin='lower', vmin=0, vmax=3)
    axes[1, 0].set_title('All ROIs - Sagittal')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(subject_fa[:, mid_cor, :].T, cmap='gray', origin='lower')
    axes[1, 1].imshow(np.ma.masked_where(roi_combined[:, mid_cor, :].T == 0,
                                          roi_combined[:, mid_cor, :].T),
                      cmap=roi_cmap, alpha=0.6, origin='lower', vmin=0, vmax=3)
    axes[1, 1].set_title('All ROIs - Coronal')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(subject_fa[:, :, mid_ax].T, cmap='gray', origin='lower')
    axes[1, 2].imshow(np.ma.masked_where(roi_combined[:, :, mid_ax].T == 0,
                                          roi_combined[:, :, mid_ax].T),
                      cmap=roi_cmap, alpha=0.6, origin='lower', vmin=0, vmax=3)
    axes[1, 2].set_title('All ROIs - Axial')
    axes[1, 2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Brainstem'),
        Patch(facecolor='blue', alpha=0.6, label='Motor Left'),
        Patch(facecolor='green', alpha=0.6, label='Motor Right')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    prefix = f"{subject_id}_" if subject_id else ""
    fig_path = output_dir / f"{prefix}roi_masks_visualization.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"✓ Saved ROI visualization: {fig_path}")
    
    return fig_path
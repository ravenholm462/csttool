"""
visualizations.py

Visualization functions for preprocessing QC.

This module provides file-saving visualizations for:
- Denoising comparison (before/after)
- Brain mask overlay
- Motion correction summary
- Multi-panel preprocessing summary

All functions save figures to disk and return the path to the saved file.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path


def plot_denoising_comparison(
    data_before,
    data_after,
    brain_mask,
    output_dir,
    stem,
    vol_idx=0,
    verbose=True
):
    """
    Create before/after denoising comparison figure.
    
    Shows three orthogonal views comparing original and denoised data,
    plus a difference map highlighting removed noise.
    
    Parameters
    ----------
    data_before : ndarray
        4D DWI data before denoising.
    data_after : ndarray
        4D DWI data after denoising.
    brain_mask : ndarray
        3D binary brain mask.
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
    vol_idx : int, optional
        Volume index to visualize (default: 0, typically b0).
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    fig_path : Path
        Path to saved figure.
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get slice indices
    mid_ax = data_before.shape[2] // 2
    mid_cor = data_before.shape[1] // 2
    mid_sag = data_before.shape[0] // 2
    
    # Extract volumes
    before = data_before[..., vol_idx]
    after = data_after[..., vol_idx]
    
    # Compute difference (only inside brain)
    diff = np.abs(after.astype(np.float64) - before.astype(np.float64))
    diff[~brain_mask] = 0
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"Denoising QC - {stem}\nVolume {vol_idx}", fontsize=14, fontweight='bold')
    
    # Normalize for consistent display
    vmax = np.percentile(before[brain_mask], 99)
    diff_vmax = np.percentile(diff[brain_mask], 99) if diff[brain_mask].size > 0 else 1
    
    views = [
        ('Axial', before[:, :, mid_ax], after[:, :, mid_ax], diff[:, :, mid_ax]),
        ('Coronal', before[:, mid_cor, :], after[:, mid_cor, :], diff[:, mid_cor, :]),
        ('Sagittal', before[mid_sag, :, :], after[mid_sag, :, :], diff[mid_sag, :, :]),
    ]
    
    for row, (view_name, bef, aft, dif) in enumerate(views):
        # Before
        axes[row, 0].imshow(bef.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
        axes[row, 0].set_title(f'{view_name} - Before' if row == 0 else '')
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_ylabel('Before', fontsize=12)
        
        # After
        axes[row, 1].imshow(aft.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
        axes[row, 1].set_title(f'{view_name} - After' if row == 0 else '')
        axes[row, 1].axis('off')
        
        # Difference
        im = axes[row, 2].imshow(dif.T, cmap='hot', origin='lower', vmin=0, vmax=diff_vmax)
        axes[row, 2].set_title(f'{view_name} - Difference' if row == 0 else '')
        axes[row, 2].axis('off')
    
    # Add row labels
    for row, (view_name, _, _, _) in enumerate(views):
        axes[row, 0].text(-0.1, 0.5, view_name, transform=axes[row, 0].transAxes,
                         fontsize=12, fontweight='bold', va='center', ha='right',
                         rotation=90)
    
    # Colorbar for difference
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Intensity Difference')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    fig_path = viz_dir / f"{stem}_denoising_qc.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Denoising QC: {fig_path}")
    
    return fig_path


def plot_brain_mask_overlay(
    data,
    brain_mask,
    gtab,
    output_dir,
    stem,
    verbose=True
):
    """
    Create brain mask overlay visualization.
    
    Shows brain mask overlaid on b0 image in three orthogonal views,
    plus mask coverage statistics.
    
    Parameters
    ----------
    data : ndarray
        4D DWI data (masked or unmasked).
    brain_mask : ndarray
        3D binary brain mask.
    gtab : GradientTable
        Gradient table to identify b0 volumes.
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    fig_path : Path
        Path to saved figure.
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get b0 volume
    b0_idx = np.where(gtab.bvals < 50)[0]
    if len(b0_idx) == 0:
        b0_idx = [0]
    b0 = data[..., b0_idx[0]]
    
    # Get slice indices
    mid_ax = data.shape[2] // 2
    mid_cor = data.shape[1] // 2
    mid_sag = data.shape[0] // 2
    
    # Compute statistics
    total_voxels = brain_mask.size
    brain_voxels = brain_mask.sum()
    coverage = brain_voxels / total_voxels * 100
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Brain Mask QC - {stem}\n"
                 f"Coverage: {brain_voxels:,} voxels ({coverage:.1f}%)",
                 fontsize=14, fontweight='bold')
    
    vmax = np.percentile(b0[brain_mask], 99) if brain_mask.any() else np.percentile(b0, 99)
    
    views = [
        ('Axial', b0[:, :, mid_ax], brain_mask[:, :, mid_ax], f'z={mid_ax}'),
        ('Coronal', b0[:, mid_cor, :], brain_mask[:, mid_cor, :], f'y={mid_cor}'),
        ('Sagittal', b0[mid_sag, :, :], brain_mask[mid_sag, :, :], f'x={mid_sag}'),
    ]
    
    for col, (view_name, b0_slice, mask_slice, coord) in enumerate(views):
        # Row 0: b0 only
        axes[0, col].imshow(b0_slice.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
        axes[0, col].set_title(f'{view_name} ({coord})\nb0 image')
        axes[0, col].axis('off')
        
        # Row 1: b0 with mask overlay
        axes[1, col].imshow(b0_slice.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
        
        # Create masked array for overlay
        mask_overlay = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
        axes[1, col].imshow(mask_overlay, cmap='Reds', alpha=0.4, origin='lower')
        
        # Add contour
        axes[1, col].contour(mask_slice.T, levels=[0.5], colors='red', linewidths=1)
        axes[1, col].set_title(f'{view_name}\nwith brain mask')
        axes[1, col].axis('off')
    
    plt.tight_layout()
    
    fig_path = viz_dir / f"{stem}_brain_mask_qc.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Brain mask QC: {fig_path}")
    
    return fig_path


def plot_motion_correction_summary(
    reg_affines,
    output_dir,
    stem,
    verbose=True
):
    """
    Create motion correction summary visualization.
    
    Shows translation and rotation parameters across volumes,
    highlighting any large motion events.
    
    Parameters
    ----------
    reg_affines : list of ndarray
        List of 4x4 registration affine matrices (one per volume).
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    fig_path : Path
        Path to saved figure.
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    n_vols = len(reg_affines)
    
    # Extract translation and rotation parameters
    translations = np.zeros((n_vols, 3))
    rotations = np.zeros((n_vols, 3))
    
    for i, affine in enumerate(reg_affines):
        # Translation is in the last column
        translations[i] = affine[:3, 3]
        
        # Approximate rotation angles from rotation matrix
        # Using small angle approximation for simplicity
        R = affine[:3, :3]
        rotations[i, 0] = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi  # Roll (x)
        rotations[i, 1] = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)) * 180 / np.pi  # Pitch (y)
        rotations[i, 2] = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi  # Yaw (z)
    
    # Compute relative motion (difference from first volume)
    translations_rel = translations - translations[0]
    rotations_rel = rotations - rotations[0]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Motion Correction QC - {stem}\n{n_vols} volumes",
                 fontsize=14, fontweight='bold')
    
    volumes = np.arange(n_vols)
    
    # Translation (absolute)
    ax = axes[0, 0]
    ax.plot(volumes, translations[:, 0], 'r-', label='X', linewidth=1.5)
    ax.plot(volumes, translations[:, 1], 'g-', label='Y', linewidth=1.5)
    ax.plot(volumes, translations[:, 2], 'b-', label='Z', linewidth=1.5)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Translation (mm)')
    ax.set_title('Absolute Translation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Translation (relative to first)
    ax = axes[0, 1]
    ax.plot(volumes, translations_rel[:, 0], 'r-', label='X', linewidth=1.5)
    ax.plot(volumes, translations_rel[:, 1], 'g-', label='Y', linewidth=1.5)
    ax.plot(volumes, translations_rel[:, 2], 'b-', label='Z', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Translation (mm)')
    ax.set_title('Relative Translation (vs. volume 0)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Rotation (absolute)
    ax = axes[1, 0]
    ax.plot(volumes, rotations[:, 0], 'r-', label='Roll (X)', linewidth=1.5)
    ax.plot(volumes, rotations[:, 1], 'g-', label='Pitch (Y)', linewidth=1.5)
    ax.plot(volumes, rotations[:, 2], 'b-', label='Yaw (Z)', linewidth=1.5)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Rotation (degrees)')
    ax.set_title('Absolute Rotation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Rotation (relative to first)
    ax = axes[1, 1]
    ax.plot(volumes, rotations_rel[:, 0], 'r-', label='Roll (X)', linewidth=1.5)
    ax.plot(volumes, rotations_rel[:, 1], 'g-', label='Pitch (Y)', linewidth=1.5)
    ax.plot(volumes, rotations_rel[:, 2], 'b-', label='Yaw (Z)', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Rotation (degrees)')
    ax.set_title('Relative Rotation (vs. volume 0)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics
    max_trans = np.max(np.abs(translations_rel))
    max_rot = np.max(np.abs(rotations_rel))
    
    fig.text(0.5, 0.02, 
             f"Max displacement: {max_trans:.2f} mm | Max rotation: {max_rot:.2f}°",
             ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    
    fig_path = viz_dir / f"{stem}_motion_qc.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Motion correction QC: {fig_path}")
    
    return fig_path


def create_preprocessing_summary(
    data_original,
    data_preprocessed,
    brain_mask,
    gtab,
    output_dir,
    stem,
    motion_correction_applied=False,
    verbose=True
):
    """
    Create multi-panel preprocessing summary figure.
    
    Combines key QC visualizations into a single summary figure
    for quick assessment of preprocessing quality.
    
    Parameters
    ----------
    data_original : ndarray
        4D DWI data before preprocessing.
    data_preprocessed : ndarray
        4D DWI data after preprocessing.
    brain_mask : ndarray
        3D binary brain mask.
    gtab : GradientTable
        Gradient table.
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
    motion_correction_applied : bool, optional
        Whether motion correction was applied.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    fig_path : Path
        Path to saved figure.
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get b0 volume index
    b0_idx = np.where(gtab.bvals < 50)[0]
    if len(b0_idx) == 0:
        b0_idx = [0]
    vol_idx = b0_idx[0]
    
    # Get slice indices
    mid_ax = data_original.shape[2] // 2
    
    # Extract data
    orig_b0 = data_original[:, :, mid_ax, vol_idx]
    proc_b0 = data_preprocessed[:, :, mid_ax, vol_idx]
    mask_slice = brain_mask[:, :, mid_ax]
    
    # Compute difference
    diff = np.abs(proc_b0.astype(np.float64) - orig_b0.astype(np.float64))
    diff[~mask_slice] = 0
    
    # Compute statistics
    brain_voxels = brain_mask.sum()
    coverage = brain_voxels / brain_mask.size * 100
    
    # Intensity statistics
    orig_mean = np.mean(data_original[brain_mask])
    proc_mean = np.mean(data_preprocessed[brain_mask])
    orig_std = np.std(data_original[brain_mask])
    proc_std = np.std(data_preprocessed[brain_mask])
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Preprocessing Summary - {stem}", fontsize=16, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    vmax = np.percentile(orig_b0[mask_slice], 99) if mask_slice.any() else np.percentile(orig_b0, 99)
    diff_vmax = np.percentile(diff[mask_slice], 99) if diff[mask_slice].any() else 1
    
    # Row 0: Original, Preprocessed, Difference, Mask
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_b0.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
    ax1.set_title('Original b0')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(proc_b0.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
    ax2.set_title('Preprocessed b0')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(diff.T, cmap='hot', origin='lower', vmin=0, vmax=diff_vmax)
    ax3.set_title('Difference (denoising)')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(proc_b0.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
    ax4.contour(mask_slice.T, levels=[0.5], colors='red', linewidths=1.5)
    ax4.set_title(f'Brain Mask\n({brain_voxels:,} voxels)')
    ax4.axis('off')
    
    # Row 1: Three orthogonal views with mask overlay
    views_data = [
        ('Sagittal', data_preprocessed[data_preprocessed.shape[0]//2, :, :, vol_idx],
         brain_mask[brain_mask.shape[0]//2, :, :]),
        ('Coronal', data_preprocessed[:, data_preprocessed.shape[1]//2, :, vol_idx],
         brain_mask[:, brain_mask.shape[1]//2, :]),
        ('Axial', data_preprocessed[:, :, data_preprocessed.shape[2]//2, vol_idx],
         brain_mask[:, :, brain_mask.shape[2]//2]),
    ]
    
    for i, (name, img, msk) in enumerate(views_data):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(img.T, cmap='gray', origin='lower')
        ax.contour(msk.T, levels=[0.5], colors='cyan', linewidths=1)
        ax.set_title(name)
        ax.axis('off')
    
    # Row 1, col 3: Histogram comparison
    ax_hist = fig.add_subplot(gs[1, 3])
    
    orig_vals = data_original[brain_mask].flatten()
    proc_vals = data_preprocessed[brain_mask].flatten()
    
    # Subsample for efficiency
    if len(orig_vals) > 100000:
        idx = np.random.choice(len(orig_vals), 100000, replace=False)
        orig_vals = orig_vals[idx]
        proc_vals = proc_vals[idx]
    
    ax_hist.hist(orig_vals, bins=100, alpha=0.5, label='Original', density=True)
    ax_hist.hist(proc_vals, bins=100, alpha=0.5, label='Preprocessed', density=True)
    ax_hist.set_xlabel('Intensity')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Intensity Distribution')
    ax_hist.legend(fontsize=8)
    
    # Row 2: Statistics panel
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    mc_status = "Applied" if motion_correction_applied else "Not applied"
    
    stats_text = (
        f"{'─' * 80}\n"
        f"PREPROCESSING STATISTICS\n"
        f"{'─' * 80}\n\n"
        f"Data Shape:           {data_original.shape}\n"
        f"Voxel Dimensions:     {data_original.shape[:3]}\n"
        f"Number of Volumes:    {data_original.shape[3]}\n"
        f"B-values:             {sorted(set(gtab.bvals.astype(int)))}\n\n"
        f"Brain Mask Coverage:  {brain_voxels:,} voxels ({coverage:.1f}%)\n\n"
        f"Intensity (in brain):\n"
        f"  Original:           mean = {orig_mean:.1f}, std = {orig_std:.1f}\n"
        f"  Preprocessed:       mean = {proc_mean:.1f}, std = {proc_std:.1f}\n\n"
        f"Motion Correction:    {mc_status}\n"
        f"{'─' * 80}"
    )
    
    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, fontfamily='monospace',
                  verticalalignment='center', horizontalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    fig_path = viz_dir / f"{stem}_preprocessing_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Preprocessing summary: {fig_path}")
    
    return fig_path


def save_all_preprocessing_visualizations(
    data_original,
    data_denoised,
    data_preprocessed,
    brain_mask,
    gtab,
    output_dir,
    stem,
    reg_affines=None,
    motion_correction_applied=False,
    verbose=True
):
    """
    Generate and save all preprocessing visualizations.
    
    Convenience function that calls all visualization functions
    and returns paths to all generated figures.
    
    Parameters
    ----------
    data_original : ndarray
        4D DWI data before any preprocessing.
    data_denoised : ndarray
        4D DWI data after denoising (before masking).
    data_preprocessed : ndarray
        4D DWI data after full preprocessing.
    brain_mask : ndarray
        3D binary brain mask.
    gtab : GradientTable
        Gradient table.
    output_dir : str or Path
        Output directory for saving figures.
    stem : str
        Subject/scan identifier for filenames.
    reg_affines : list of ndarray, optional
        Registration affines from motion correction.
    motion_correction_applied : bool, optional
        Whether motion correction was applied.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    viz_paths : dict
        Dictionary mapping visualization names to file paths.
    """
    if verbose:
        print("\nGenerating preprocessing visualizations...")
    
    viz_paths = {}
    
    # Denoising comparison
    if data_denoised is not None:
        viz_paths['denoising_qc'] = plot_denoising_comparison(
            data_original, data_denoised, brain_mask,
            output_dir, stem, verbose=verbose
        )
    
    # Brain mask overlay
    viz_paths['brain_mask_qc'] = plot_brain_mask_overlay(
        data_preprocessed, brain_mask, gtab,
        output_dir, stem, verbose=verbose
    )
    
    # Motion correction (if applied)
    if motion_correction_applied and reg_affines is not None:
        viz_paths['motion_qc'] = plot_motion_correction_summary(
            reg_affines, output_dir, stem, verbose=verbose
        )
    
    # Summary figure
    viz_paths['summary'] = create_preprocessing_summary(
        data_original, data_preprocessed, brain_mask, gtab,
        output_dir, stem, motion_correction_applied, verbose=verbose
    )
    
    if verbose:
        print(f"✓ All preprocessing visualizations saved to: {Path(output_dir) / 'visualizations'}")
    
    return viz_paths
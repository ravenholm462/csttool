"""
visualizations.py

Visualization functions for CST extraction QC.

This module provides file-saving visualizations for:
- Registration comparison (MNI to subject alignment)
- ROI mask overlays
- CST extraction results (bilateral streamlines)
- Multi-panel extraction summary

All functions save figures to disk and return the path to the saved file.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
from pathlib import Path


def plot_registration_comparison(
    subject_fa,
    mni_warped,
    output_dir,
    subject_id=None,
    verbose=True
):
    """
    Create registration comparison visualization.
    
    Shows subject FA and warped MNI template side-by-side
    in three orthogonal views for registration QC.
    
    Parameters
    ----------
    subject_fa : ndarray
        3D subject FA map.
    mni_warped : ndarray
        3D MNI template warped to subject space.
    output_dir : str or Path
        Output directory for saving figure.
    subject_id : str, optional
        Subject identifier for filename.
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
    
    prefix = f"{subject_id}_" if subject_id else ""
    
    # Get slice indices
    mid_ax = subject_fa.shape[2] // 2
    mid_cor = subject_fa.shape[1] // 2
    mid_sag = subject_fa.shape[0] // 2
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"Registration QC - {subject_id or 'Subject'}\nMNI → Subject Space",
                 fontsize=14, fontweight='bold')
    
    views = [
        ('Axial', subject_fa[:, :, mid_ax], mni_warped[:, :, mid_ax]),
        ('Coronal', subject_fa[:, mid_cor, :], mni_warped[:, mid_cor, :]),
        ('Sagittal', subject_fa[mid_sag, :, :], mni_warped[mid_sag, :, :]),
    ]
    
    for row, (view_name, fa_slice, mni_slice) in enumerate(views):
        # Subject FA
        axes[row, 0].imshow(fa_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes[row, 0].set_title(f'{view_name}\nSubject FA' if row == 0 else '')
        axes[row, 0].axis('off')
        
        # MNI warped
        axes[row, 1].imshow(mni_slice.T, cmap='gray', origin='lower')
        axes[row, 1].set_title(f'{view_name}\nMNI Warped' if row == 0 else '')
        axes[row, 1].axis('off')
        
        # Overlay
        axes[row, 2].imshow(fa_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes[row, 2].imshow(mni_slice.T, cmap='hot', alpha=0.3, origin='lower')
        axes[row, 2].set_title(f'{view_name}\nOverlay' if row == 0 else '')
        axes[row, 2].axis('off')
        
        # Add view label
        axes[row, 0].text(-0.15, 0.5, view_name, transform=axes[row, 0].transAxes,
                         fontsize=12, fontweight='bold', va='center', ha='right',
                         rotation=90)
    
    plt.tight_layout()
    
    fig_path = viz_dir / f"{prefix}registration_qc.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Registration QC: {fig_path}")
    
    return fig_path


def plot_cst_extraction(
    cst_result,
    fa,
    affine,
    output_dir,
    subject_id=None,
    max_streamlines=2000,
    verbose=True
):
    """
    Create CST extraction visualization.
    
    Shows extracted left and right CST streamlines overlaid on FA
    in three orthogonal views.
    
    Parameters
    ----------
    cst_result : dict
        Output from extract_bilateral_cst() containing:
        - cst_left: Left CST streamlines
        - cst_right: Right CST streamlines
        - stats: Extraction statistics
    fa : ndarray
        3D FA map for background.
    affine : ndarray
        4x4 affine transformation matrix.
    output_dir : str or Path
        Output directory for saving figure.
    subject_id : str, optional
        Subject identifier for filename.
    max_streamlines : int, optional
        Maximum streamlines to plot per hemisphere.
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
    
    prefix = f"{subject_id}_" if subject_id else ""
    
    cst_left = cst_result['cst_left']
    cst_right = cst_result['cst_right']
    stats = cst_result['stats']
    
    # Subsample if needed
    def subsample(streamlines, max_n):
        if len(streamlines) <= max_n:
            return list(streamlines)
        indices = np.random.choice(len(streamlines), max_n, replace=False)
        return [streamlines[i] for i in indices]
    
    left_vis = subsample(cst_left, max_streamlines)
    right_vis = subsample(cst_right, max_streamlines)
    
    # Get voxel size for extent calculation
    voxel_size = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    
    # Get slice indices
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"CST Extraction - {subject_id or 'Subject'}\n"
                 f"Left: {stats['cst_left_count']:,} | Right: {stats['cst_right_count']:,} | "
                 f"Total: {stats['cst_total_count']:,} ({stats['extraction_rate']:.1f}%)",
                 fontsize=14, fontweight='bold')
    
    # Row 0: FA background with ROI indication
    views_fa = [
        ('Sagittal', fa[mid_sag, :, :].T, 1, 2),
        ('Coronal', fa[:, mid_cor, :].T, 0, 2),
        ('Axial', fa[:, :, mid_ax].T, 0, 1),
    ]
    
    for col, (name, fa_slice, d1, d2) in enumerate(views_fa):
        ax = axes[0, col]
        extent = [0, fa_slice.shape[1] * voxel_size[d1], 
                  0, fa_slice.shape[0] * voxel_size[d2]]
        ax.imshow(fa_slice, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=0.8)
        ax.set_title(f'{name}\nFA background')
        ax.axis('off')
    
    # Row 1: Streamlines on FA background
    views_sl = [
        ('Sagittal (Y-Z)', 1, 2, fa[mid_sag, :, :].T),
        ('Coronal (X-Z)', 0, 2, fa[:, mid_cor, :].T),
        ('Axial (X-Y)', 0, 1, fa[:, :, mid_ax].T),
    ]
    
    for col, (name, d1, d2, fa_slice) in enumerate(views_sl):
        ax = axes[1, col]
        extent = [0, fa_slice.shape[1] * voxel_size[d1], 
                  0, fa_slice.shape[0] * voxel_size[d2]]
        ax.imshow(fa_slice, cmap='gray', origin='lower', extent=extent, 
                  vmin=0, vmax=0.8, alpha=0.5)
        
        # Plot left CST (blue)
        for sl in left_vis:
            ax.plot(sl[:, d1], sl[:, d2], color='blue', alpha=0.15, linewidth=0.5)
        
        # Plot right CST (red)
        for sl in right_vis:
            ax.plot(sl[:, d1], sl[:, d2], color='red', alpha=0.15, linewidth=0.5)
        
        ax.set_title(f'{name}')
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label=f'Left CST ({stats["cst_left_count"]:,})'),
        Line2D([0], [0], color='red', linewidth=2, label=f'Right CST ({stats["cst_right_count"]:,})'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    fig_path = viz_dir / f"{prefix}cst_extraction.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ CST extraction: {fig_path}")
    
    return fig_path


def create_extraction_summary(
    cst_result,
    fa,
    masks,
    affine,
    output_dir,
    subject_id=None,
    verbose=True
):
    """
    Create multi-panel extraction summary figure.
    
    Combines ROI masks, CST visualization, and statistics
    into a single summary figure.
    
    Parameters
    ----------
    cst_result : dict
        Output from extract_bilateral_cst().
    fa : ndarray
        3D FA map.
    masks : dict
        ROI masks from create_cst_roi_masks().
    affine : ndarray
        4x4 affine transformation matrix.
    output_dir : str or Path
        Output directory for saving figure.
    subject_id : str, optional
        Subject identifier for filename.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    fig_path : Path
        Path to saved figure.
    """
    from dipy.tracking.streamline import length
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{subject_id}_" if subject_id else ""
    
    cst_left = cst_result['cst_left']
    cst_right = cst_result['cst_right']
    stats = cst_result['stats']
    
    # Compute lengths
    left_lengths = np.array([length(s) for s in cst_left]) if len(cst_left) > 0 else np.array([])
    right_lengths = np.array([length(s) for s in cst_right]) if len(cst_right) > 0 else np.array([])
    
    # Get voxel size
    voxel_size = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"CST Extraction Summary - {subject_id or 'Subject'}", 
                 fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    mid_ax = fa.shape[2] // 2
    
    # Row 0: ROI masks
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=0.8)
    brainstem_overlay = np.ma.masked_where(masks['brainstem'][:, :, mid_ax].T == 0,
                                            masks['brainstem'][:, :, mid_ax].T)
    ax.imshow(brainstem_overlay, cmap='Reds', alpha=0.6, origin='lower')
    ax.set_title(f'Brainstem ROI\n{np.sum(masks["brainstem"]):,} voxels')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=0.8)
    motor_left_overlay = np.ma.masked_where(masks['motor_left'][:, :, mid_ax].T == 0,
                                             masks['motor_left'][:, :, mid_ax].T)
    ax.imshow(motor_left_overlay, cmap='Blues', alpha=0.6, origin='lower')
    ax.set_title(f'Motor Left ROI\n{np.sum(masks["motor_left"]):,} voxels')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=0.8)
    motor_right_overlay = np.ma.masked_where(masks['motor_right'][:, :, mid_ax].T == 0,
                                              masks['motor_right'][:, :, mid_ax].T)
    ax.imshow(motor_right_overlay, cmap='Greens', alpha=0.6, origin='lower')
    ax.set_title(f'Motor Right ROI\n{np.sum(masks["motor_right"]):,} voxels')
    ax.axis('off')
    
    # Row 0, col 3: All ROIs combined
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=0.8)
    ax.imshow(brainstem_overlay, cmap='Reds', alpha=0.4, origin='lower')
    ax.imshow(motor_left_overlay, cmap='Blues', alpha=0.4, origin='lower')
    ax.imshow(motor_right_overlay, cmap='Greens', alpha=0.4, origin='lower')
    ax.set_title('All ROIs Combined')
    ax.axis('off')
    
    # Row 1: CST streamlines
    max_vis = 1500
    
    def subsample(streamlines, max_n):
        if len(streamlines) <= max_n:
            return list(streamlines)
        indices = np.random.choice(len(streamlines), max_n, replace=False)
        return [streamlines[i] for i in indices]
    
    left_vis = subsample(cst_left, max_vis)
    right_vis = subsample(cst_right, max_vis)
    
    views = [
        ('Sagittal', 1, 2, fa[fa.shape[0]//2, :, :].T),
        ('Coronal', 0, 2, fa[:, fa.shape[1]//2, :].T),
        ('Axial', 0, 1, fa[:, :, mid_ax].T),
    ]
    
    for col, (name, d1, d2, fa_bg) in enumerate(views):
        ax = fig.add_subplot(gs[1, col])
        extent = [0, fa_bg.shape[1] * voxel_size[d1], 
                  0, fa_bg.shape[0] * voxel_size[d2]]
        ax.imshow(fa_bg, cmap='gray', origin='lower', extent=extent, 
                  vmin=0, vmax=0.8, alpha=0.4)
        
        for sl in left_vis:
            ax.plot(sl[:, d1], sl[:, d2], color='blue', alpha=0.1, linewidth=0.5)
        for sl in right_vis:
            ax.plot(sl[:, d1], sl[:, d2], color='red', alpha=0.1, linewidth=0.5)
        
        ax.set_title(name)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Row 1, col 3: Length histogram
    ax = fig.add_subplot(gs[1, 3])
    if len(left_lengths) > 0:
        ax.hist(left_lengths, bins=30, alpha=0.6, color='blue', label='Left CST')
    if len(right_lengths) > 0:
        ax.hist(right_lengths, bins=30, alpha=0.6, color='red', label='Right CST')
    ax.set_xlabel('Length (mm)')
    ax.set_ylabel('Count')
    ax.set_title('Length Distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Row 2: Statistics panel
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    # Build statistics text
    if len(left_lengths) > 0:
        left_stats = f"mean={np.mean(left_lengths):.1f}, range=[{np.min(left_lengths):.1f}, {np.max(left_lengths):.1f}]"
    else:
        left_stats = "N/A"
    
    if len(right_lengths) > 0:
        right_stats = f"mean={np.mean(right_lengths):.1f}, range=[{np.min(right_lengths):.1f}, {np.max(right_lengths):.1f}]"
    else:
        right_stats = "N/A"
    
    stats_text = (
        f"{'═' * 100}\n"
        f"EXTRACTION SUMMARY\n"
        f"{'═' * 100}\n\n"
        f"Input Streamlines:     {stats['total_input']:,}\n"
        f"Extracted (Total):     {stats['cst_total_count']:,} ({stats['extraction_rate']:.2f}%)\n\n"
        f"Left CST:              {stats['cst_left_count']:,} streamlines\n"
        f"  Length:              {left_stats} mm\n\n"
        f"Right CST:             {stats['cst_right_count']:,} streamlines\n"
        f"  Length:              {right_stats} mm\n\n"
        f"Left/Right Ratio:      {stats.get('left_right_ratio', 'N/A')}\n"
        f"{'═' * 100}"
    )
    
    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
            fontsize=11, fontfamily='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    fig_path = viz_dir / f"{prefix}extraction_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Extraction summary: {fig_path}")
    
    return fig_path


def save_all_extraction_visualizations(
    cst_result,
    fa,
    masks,
    affine,
    output_dir,
    subject_id=None,
    mni_warped=None,
    verbose=True
):
    """
    Generate and save all extraction visualizations.
    
    Convenience function that calls all visualization functions
    and returns paths to all generated figures.
    
    Parameters
    ----------
    cst_result : dict
        Output from extract_bilateral_cst().
    fa : ndarray
        3D FA map.
    masks : dict
        ROI masks from create_cst_roi_masks().
    affine : ndarray
        4x4 affine transformation matrix.
    output_dir : str or Path
        Output directory for saving figures.
    subject_id : str, optional
        Subject identifier for filenames.
    mni_warped : ndarray, optional
        Warped MNI template for registration QC.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    viz_paths : dict
        Dictionary mapping visualization names to file paths.
    """
    if verbose:
        print("\nGenerating extraction visualizations...")
    
    viz_paths = {}
    
    # Registration comparison (if MNI warped available)
    if mni_warped is not None:
        viz_paths['registration_qc'] = plot_registration_comparison(
            fa, mni_warped, output_dir, subject_id, verbose=verbose
        )
    
    # CST extraction
    viz_paths['cst_extraction'] = plot_cst_extraction(
        cst_result, fa, affine, output_dir, subject_id, verbose=verbose
    )
    
    # Summary figure
    viz_paths['summary'] = create_extraction_summary(
        cst_result, fa, masks, affine, output_dir, subject_id, verbose=verbose
    )
    
    if verbose:
        print(f"✓ All extraction visualizations saved to: {Path(output_dir) / 'visualizations'}")
    
    return viz_paths
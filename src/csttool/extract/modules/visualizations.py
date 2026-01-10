"""
visualizations.py - Extract Module

Visualization functions for CST extraction QC.

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# REGISTRATION QC VISUALIZATION
# =============================================================================

def plot_registration_comparison(
    subject_fa,
    mni_warped,
    output_dir,
    subject_id=None,
    verbose=True
):
    """
    Create registration QC visualization.
    
    Shows subject FA and warped MNI template side-by-side
    in three orthogonal views for registration QC.
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


# =============================================================================
# ROI MASK VISUALIZATION
# =============================================================================

def plot_roi_masks(
    fa,
    masks,
    output_dir,
    subject_id=None,
    verbose=True
):
    """
    Create ROI masks visualization.
    
    Shows motor cortex and brainstem ROI masks overlaid on FA
    in three orthogonal views.
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{subject_id}_" if subject_id else ""
    
    # Get slice indices
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"ROI Masks - {subject_id or 'Subject'}\nMotor Cortex (L/R) + Brainstem",
                 fontsize=14, fontweight='bold')
    
    # Get masks (keys from create_cst_roi_masks: 'motor_left', 'motor_right', 'brainstem')
    motor_left = masks.get('motor_left', np.zeros_like(fa))
    motor_right = masks.get('motor_right', np.zeros_like(fa))
    brainstem = masks.get('brainstem', np.zeros_like(fa))
    
    views = [
        ('Axial', fa[:, :, mid_ax], motor_left[:, :, mid_ax], 
         motor_right[:, :, mid_ax], brainstem[:, :, mid_ax]),
        ('Coronal', fa[:, mid_cor, :], motor_left[:, mid_cor, :],
         motor_right[:, mid_cor, :], brainstem[:, mid_cor, :]),
        ('Sagittal', fa[mid_sag, :, :], motor_left[mid_sag, :, :],
         motor_right[mid_sag, :, :], brainstem[mid_sag, :, :]),
    ]
    
    for col, (view_name, fa_slice, ml_slice, mr_slice, bs_slice) in enumerate(views):
        # Row 0: FA only
        axes[0, col].imshow(fa_slice.T, cmap='gray', origin='lower', vmin=0, vmax=0.8)
        axes[0, col].set_title(f'{view_name}\nFA background')
        axes[0, col].axis('off')
        
        # Row 1: FA with ROI overlays
        axes[1, col].imshow(fa_slice.T, cmap='gray', origin='lower', vmin=0, vmax=0.8)
        
        # Create colored overlays
        # Motor cortex left (blue)
        ml_rgb = np.zeros((*ml_slice.T.shape, 4))  # RGBA array
        ml_rgb[ml_slice.T > 0] = [0, 0, 1, 0.6]  # Blue with alpha
        
        # Motor cortex right (red)
        mr_rgb = np.zeros((*mr_slice.T.shape, 4))
        mr_rgb[mr_slice.T > 0] = [1, 0, 0, 0.6]  # Red with alpha
        
        # Brainstem (green)
        bs_rgb = np.zeros((*bs_slice.T.shape, 4))
        bs_rgb[bs_slice.T > 0] = [0, 1, 0, 0.6]  # Green with alpha
        
        # Apply overlays
        axes[1, col].imshow(ml_rgb, origin='lower')
        axes[1, col].imshow(mr_rgb, origin='lower')
        axes[1, col].imshow(bs_rgb, origin='lower')
        
        axes[1, col].set_title(f'{view_name}\nROI overlay')
        axes[1, col].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Motor Cortex Left'),
        Patch(facecolor='red', alpha=0.6, label='Motor Cortex Right'),
        Patch(facecolor='green', alpha=0.6, label='Brainstem'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    fig_path = viz_dir / f"{prefix}roi_masks.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ ROI masks: {fig_path}")
    
    return fig_path


# =============================================================================
# CST EXTRACTION VISUALIZATION (SIMPLIFIED - NO FA BACKGROUND ON STREAMLINES)
# =============================================================================

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
    
    Row 0: FA anatomy in three orthogonal views (voxel space)
    Row 1: Streamlines only on neutral background (world coordinates)
    
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
    
    # Row 0: FA background only (voxel space, no extent needed)
    views_fa = [
        ('Sagittal', fa[mid_sag, :, :].T),
        ('Coronal', fa[:, mid_cor, :].T),
        ('Axial', fa[:, :, mid_ax].T),
    ]
    
    for col, (name, fa_slice) in enumerate(views_fa):
        ax = axes[0, col]
        ax.imshow(fa_slice, cmap='gray', origin='lower', vmin=0, vmax=0.8)
        ax.set_title(f'{name}\nFA background')
        ax.axis('off')
    
    # Row 1: Streamlines ONLY (no FA background) on neutral background
    # World coordinate dimensions: 0=X, 1=Y, 2=Z
    views_sl = [
        ('Sagittal (Y-Z)', 1, 2, 'Y (mm)', 'Z (mm)'),   # Y vs Z
        ('Coronal (X-Z)', 0, 2, 'X (mm)', 'Z (mm)'),    # X vs Z
        ('Axial (X-Y)', 0, 1, 'X (mm)', 'Y (mm)'),      # X vs Y
    ]
    
    for col, (name, d1, d2, xlabel, ylabel) in enumerate(views_sl):
        ax = axes[1, col]
        
        # Set neutral background
        ax.set_facecolor('#f0f0f0')
        
        # Plot left CST (blue)
        for sl in left_vis:
            ax.plot(sl[:, d1], sl[:, d2], color='blue', alpha=0.3, linewidth=0.8)
        
        # Plot right CST (red)
        for sl in right_vis:
            ax.plot(sl[:, d1], sl[:, d2], color='red', alpha=0.3, linewidth=0.8)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color='white')
    
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


# =============================================================================
# EXTRACTION SUMMARY (SIMPLIFIED)
# =============================================================================

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
    
    # Subsample for visualization
    def subsample(streamlines, max_n=1000):
        if len(streamlines) <= max_n:
            return list(streamlines)
        indices = np.random.choice(len(streamlines), max_n, replace=False)
        return [streamlines[i] for i in indices]
    
    left_vis = subsample(cst_left)
    right_vis = subsample(cst_right)
    
    # Get slice indices
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"CST Extraction Summary - {subject_id or 'Subject'}",
                 fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 0: ROI masks (voxel space)
    # Keys from create_cst_roi_masks: 'motor_left', 'motor_right', 'brainstem'
    motor_left = masks.get('motor_left', np.zeros_like(fa))
    motor_right = masks.get('motor_right', np.zeros_like(fa))
    brainstem = masks.get('brainstem', np.zeros_like(fa))
    
    roi_views = [
        ('Axial', fa[:, :, mid_ax], motor_left[:, :, mid_ax], 
         motor_right[:, :, mid_ax], brainstem[:, :, mid_ax]),
        ('Coronal', fa[:, mid_cor, :], motor_left[:, mid_cor, :],
         motor_right[:, mid_cor, :], brainstem[:, mid_cor, :]),
        ('Sagittal', fa[mid_sag, :, :], motor_left[mid_sag, :, :],
         motor_right[mid_sag, :, :], brainstem[mid_sag, :, :]),
    ]
    
    for col, (name, fa_slice, ml_slice, mr_slice, bs_slice) in enumerate(roi_views):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(fa_slice.T, cmap='gray', origin='lower', vmin=0, vmax=0.8)
        
        # Create RGB overlays with proper colors
        # Motor cortex left (blue)
        ml_rgb = np.zeros((*ml_slice.T.shape, 4))  # RGBA array
        ml_rgb[ml_slice.T > 0] = [0, 0, 1, 0.6]  # Blue with alpha 0.6
        
        # Motor cortex right (red)
        mr_rgb = np.zeros((*mr_slice.T.shape, 4))
        mr_rgb[mr_slice.T > 0] = [1, 0, 0, 0.6]  # Red with alpha 0.6
        
        # Brainstem (green)
        bs_rgb = np.zeros((*bs_slice.T.shape, 4))
        bs_rgb[bs_slice.T > 0] = [0, 1, 0, 0.6]  # Green with alpha 0.6
        
        # Apply overlays
        ax.imshow(ml_rgb, origin='lower')
        ax.imshow(mr_rgb, origin='lower')
        ax.imshow(bs_rgb, origin='lower')
        
        ax.set_title(f'{name}: ROI Masks')
        ax.axis('off')
    
    # ROI legend
    ax = fig.add_subplot(gs[0, 3])
    ax.axis('off')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Motor Cortex L'),
        Patch(facecolor='red', alpha=0.6, label='Motor Cortex R'),
        Patch(facecolor='green', alpha=0.6, label='Brainstem'),
    ]
    ax.legend(handles=legend_elements, loc='center', fontsize=12)
    ax.set_title('ROI Legend')
    
    # Row 1: CST streamlines only (no FA background)
    streamline_views = [
        ('Sagittal', 1, 2),  # Y vs Z
        ('Coronal', 0, 2),   # X vs Z
        ('Axial', 0, 1),     # X vs Y
    ]
    
    for col, (name, d1, d2) in enumerate(streamline_views):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor('#f0f0f0')
        
        for sl in left_vis:
            ax.plot(sl[:, d1], sl[:, d2], color='blue', alpha=0.2, linewidth=0.5)
        for sl in right_vis:
            ax.plot(sl[:, d1], sl[:, d2], color='red', alpha=0.2, linewidth=0.5)
        
        ax.set_title(f'{name}: CST')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color='white')
    
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


# =============================================================================
# CONVENIENCE FUNCTION: Save all visualizations
# =============================================================================

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
    """
    if verbose:
        print("\nGenerating extraction visualizations...")
    
    viz_paths = {}
    
    # Registration QC (if MNI warped provided)
    if mni_warped is not None:
        viz_paths['registration_qc'] = plot_registration_comparison(
            fa, mni_warped, output_dir, subject_id, verbose=verbose
        )
    
    # ROI masks
    viz_paths['roi_masks'] = plot_roi_masks(
        fa, masks, output_dir, subject_id, verbose=verbose
    )
    
    # CST extraction
    viz_paths['cst_extraction'] = plot_cst_extraction(
        cst_result, fa, affine, output_dir, subject_id, verbose=verbose
    )
    
    # Summary figure
    viz_paths['extraction_summary'] = create_extraction_summary(
        cst_result, fa, masks, affine, output_dir, subject_id, verbose=verbose
    )
    
    if verbose:
        print(f"✓ All extraction visualizations saved to: {Path(output_dir) / 'visualizations'}")
    
    return viz_paths

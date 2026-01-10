"""
visualizations.py

Visualization functions for tractography QC.

This module provides file-saving visualizations for:
- Tensor-derived maps (FA, MD, RGB direction)
- White matter mask overlay
- 2D streamline projections
- Streamline statistics (length distribution, coverage)
- Multi-panel tracking summary

All functions save figures to disk and return the path to the saved file.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path


# =============================================================================
# HELPER FUNCTION: Compute world-coordinate extent for imshow
# =============================================================================

def compute_world_extent(fa_shape, affine, slice_idx, view):
    """
    Compute proper world-coordinate extent for imshow to align FA with streamlines.
    
    Streamlines are in RASMM world coordinates, so the FA background must be
    positioned using the full affine transformation (not just voxel size).
    
    Parameters
    ----------
    fa_shape : tuple
        Shape of the 3D FA volume (i, j, k).
    affine : ndarray
        4x4 affine transformation matrix (voxel → world).
    slice_idx : int
        Index of the slice being displayed (in the fixed dimension).
    view : str
        One of 'sagittal', 'coronal', or 'axial'.
        
    Returns
    -------
    extent : list
        [x_min, x_max, y_min, y_max] in world coordinates for use with imshow.
        
    Notes
    -----
    For each view:
    - Sagittal: fixed dim 0 (i), showing dims 1 (j) vs 2 (k) → Y vs Z
    - Coronal: fixed dim 1 (j), showing dims 0 (i) vs 2 (k) → X vs Z
    - Axial: fixed dim 2 (k), showing dims 0 (i) vs 1 (j) → X vs Y
    
    We compute corners by transforming voxel coordinates through the affine.
    """
    if view == 'sagittal':
        # Fixed i = slice_idx, showing j (Y) on x-axis, k (Z) on y-axis
        # Corner voxels: (slice_idx, 0, 0) and (slice_idx, j_max, k_max)
        corner_00 = affine @ np.array([slice_idx, 0, 0, 1])
        corner_jk = affine @ np.array([slice_idx, fa_shape[1], fa_shape[2], 1])
        x_extent = [corner_00[1], corner_jk[1]]  # Y range
        y_extent = [corner_00[2], corner_jk[2]]  # Z range
        
    elif view == 'coronal':
        # Fixed j = slice_idx, showing i (X) on x-axis, k (Z) on y-axis
        corner_00 = affine @ np.array([0, slice_idx, 0, 1])
        corner_ik = affine @ np.array([fa_shape[0], slice_idx, fa_shape[2], 1])
        x_extent = [corner_00[0], corner_ik[0]]  # X range
        y_extent = [corner_00[2], corner_ik[2]]  # Z range
        
    elif view == 'axial':
        # Fixed k = slice_idx, showing i (X) on x-axis, j (Y) on y-axis
        corner_00 = affine @ np.array([0, 0, slice_idx, 1])
        corner_ij = affine @ np.array([fa_shape[0], fa_shape[1], slice_idx, 1])
        x_extent = [corner_00[0], corner_ij[0]]  # X range
        y_extent = [corner_00[1], corner_ij[1]]  # Y range
        
    else:
        raise ValueError(f"Unknown view: {view}. Use 'sagittal', 'coronal', or 'axial'.")
    
    # Return as [left, right, bottom, top] for imshow extent
    # Use min/max to handle both positive and negative affine components
    return [min(x_extent), max(x_extent), min(y_extent), max(y_extent)]


# =============================================================================
# TENSOR MAPS VISUALIZATION
# =============================================================================

def plot_tensor_maps(
    fa,
    md,
    brain_mask,
    output_dir,
    stem,
    tenfit=None,
    verbose=True
):
    """
    Create tensor-derived scalar maps visualization.
    
    Shows FA, MD, and optionally RGB-encoded principal diffusion
    direction in three orthogonal views.
    
    Parameters
    ----------
    fa : ndarray
        3D fractional anisotropy map.
    md : ndarray
        3D mean diffusivity map.
    brain_mask : ndarray
        3D binary brain mask.
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
    tenfit : TensorFit, optional
        Fitted tensor model for RGB direction encoding.
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
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Compute RGB direction map if tenfit available
    if tenfit is not None:
        try:
            evecs = tenfit.evecs
            # Principal eigenvector (first eigenvector)
            v1 = evecs[..., 0]
            # RGB = |x|, |y|, |z| weighted by FA
            rgb = np.abs(v1) * fa[..., np.newaxis]
            rgb = np.clip(rgb, 0, 1)
        except Exception:
            rgb = None
    else:
        rgb = None
    
    # Create figure
    n_cols = 4 if rgb is not None else 3
    fig, axes = plt.subplots(3, n_cols, figsize=(4*n_cols, 12))
    fig.suptitle(f"Tensor Maps - {stem}", fontsize=14, fontweight='bold')
    
    # Statistics for display
    fa_brain = fa[brain_mask > 0]
    md_brain = md[brain_mask > 0]
    
    views = [
        ('Axial', mid_ax, lambda x: x[:, :, mid_ax], 2),
        ('Coronal', mid_cor, lambda x: x[:, mid_cor, :], 1),
        ('Sagittal', mid_sag, lambda x: x[mid_sag, :, :], 0),
    ]
    
    for row, (view_name, slice_idx, slicer, dim) in enumerate(views):
        # FA
        axes[row, 0].imshow(slicer(fa).T, cmap='gray', origin='lower', vmin=0, vmax=1)
        if row == 0:
            axes[row, 0].set_title(f'FA\nmean={fa_brain.mean():.3f}')
        axes[row, 0].axis('off')
        
        # MD
        axes[row, 1].imshow(slicer(md).T, cmap='hot', origin='lower', 
                           vmin=0, vmax=np.percentile(md_brain, 99))
        if row == 0:
            axes[row, 1].set_title(f'MD (×10⁻³)\nmean={md_brain.mean()*1000:.2f}')
        axes[row, 1].axis('off')
        
        # Brain mask overlay
        axes[row, 2].imshow(slicer(fa).T, cmap='gray', origin='lower', vmin=0, vmax=1)
        mask_overlay = np.ma.masked_where(slicer(brain_mask).T == 0, slicer(brain_mask).T)
        axes[row, 2].imshow(mask_overlay, cmap='Blues', alpha=0.3, origin='lower')
        if row == 0:
            axes[row, 2].set_title(f'Brain Mask\n{brain_mask.sum():,} voxels')
        axes[row, 2].axis('off')
        
        # RGB if available
        if rgb is not None:
            axes[row, 3].imshow(slicer(rgb).transpose(1, 0, 2), origin='lower')
            if row == 0:
                axes[row, 3].set_title('Direction (RGB)')
            axes[row, 3].axis('off')
        
        # Add view label
        axes[row, 0].text(-0.15, 0.5, view_name, transform=axes[row, 0].transAxes,
                         fontsize=12, fontweight='bold', va='center', ha='right',
                         rotation=90)
    
    plt.tight_layout()
    
    fig_path = viz_dir / f"{stem}_tensor_maps.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Tensor maps: {fig_path}")
    
    return fig_path


# =============================================================================
# WHITE MATTER MASK VISUALIZATION
# =============================================================================

def plot_white_matter_mask(
    fa,
    white_matter,
    brain_mask,
    output_dir,
    stem,
    fa_thresh=0.2,
    verbose=True
):
    """
    Create white matter mask QC visualization.
    
    Shows FA map with white matter mask overlay and brain mask contour
    in three orthogonal views.
    
    Parameters
    ----------
    fa : ndarray
        3D fractional anisotropy map.
    white_matter : ndarray
        3D binary white matter mask.
    brain_mask : ndarray
        3D binary brain mask.
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
    fa_thresh : float, optional
        FA threshold used for white matter definition.
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
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Statistics
    wm_voxels = white_matter.sum()
    brain_voxels = brain_mask.sum()
    wm_fraction = wm_voxels / brain_voxels * 100 if brain_voxels > 0 else 0
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"White Matter Mask - {stem}\n"
                 f"FA threshold: {fa_thresh} | WM voxels: {wm_voxels:,} ({wm_fraction:.1f}% of brain)",
                 fontsize=14, fontweight='bold')
    
    views = [
        ('Axial', fa[:, :, mid_ax], white_matter[:, :, mid_ax], brain_mask[:, :, mid_ax]),
        ('Coronal', fa[:, mid_cor, :], white_matter[:, mid_cor, :], brain_mask[:, mid_cor, :]),
        ('Sagittal', fa[mid_sag, :, :], white_matter[mid_sag, :, :], brain_mask[mid_sag, :, :]),
    ]
    
    for col, (view_name, fa_slice, wm_slice, mask_slice) in enumerate(views):
        # Row 0: FA map with threshold line
        axes[0, col].imshow(fa_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes[0, col].set_title(f'{view_name}\nFA map')
        axes[0, col].axis('off')
        
        # Row 1: FA with WM overlay
        axes[1, col].imshow(fa_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
        
        # WM overlay
        wm_overlay = np.ma.masked_where(wm_slice.T == 0, wm_slice.T)
        axes[1, col].imshow(wm_overlay, cmap='Blues', alpha=0.5, origin='lower')
        
        # Brain mask contour
        axes[1, col].contour(mask_slice.T, levels=[0.5], colors='red', 
                            linewidths=1, linestyles='--')
        
        axes[1, col].set_title(f'{view_name}\nWM mask (blue) + brain (red)')
        axes[1, col].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='blue', alpha=0.5, label=f'White Matter (FA > {fa_thresh})'),
        Line2D([0], [0], color='red', linestyle='--', label='Brain Mask'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    fig_path = viz_dir / f"{stem}_wm_mask_qc.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ White matter mask QC: {fig_path}")
    
    return fig_path


# =============================================================================
# 2D STREAMLINE PROJECTIONS (CORRECTED)
# =============================================================================

def plot_streamlines_2d(
    streamlines,
    fa,
    affine,
    output_dir,
    stem,
    max_streamlines=5000,
    verbose=True
):
    """
    Create 2D streamline projection visualization.
    
    Shows streamlines projected onto three orthogonal planes
    with FA background for anatomical reference.
    
    CORRECTED: Now properly aligns FA background with streamlines by computing
    world-coordinate extent using the full affine transformation.
    
    Parameters
    ----------
    streamlines : Streamlines
        Tractography streamlines (in RASMM world coordinates).
    fa : ndarray
        3D FA map for background.
    affine : ndarray
        4x4 affine transformation matrix.
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
    max_streamlines : int, optional
        Maximum number of streamlines to plot (for performance).
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
    
    n_streamlines = len(streamlines)
    
    if n_streamlines == 0:
        if verbose:
            print("⚠️ No streamlines to visualize")
        return None
    
    # Subsample if needed
    if n_streamlines > max_streamlines:
        indices = np.random.choice(n_streamlines, max_streamlines, replace=False)
        vis_streamlines = [streamlines[i] for i in indices]
        n_vis = max_streamlines
    else:
        vis_streamlines = streamlines
        n_vis = n_streamlines
    
    # Get FA slice indices (in voxel coordinates)
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Whole-Brain Tractography - {stem}\n"
                 f"{n_streamlines:,} streamlines (showing {n_vis:,})",
                 fontsize=14, fontweight='bold')
    
    # Define views: (title, view_name, slice_idx, fa_slice, x_dim, y_dim, x_label, y_label, color)
    # x_dim and y_dim refer to world coordinate dimensions (0=X, 1=Y, 2=Z)
    views = [
        ('Sagittal (Y-Z)', 'sagittal', mid_sag, fa[mid_sag, :, :].T, 1, 2, 'Y (mm)', 'Z (mm)', 'blue'),
        ('Coronal (X-Z)', 'coronal', mid_cor, fa[:, mid_cor, :].T, 0, 2, 'X (mm)', 'Z (mm)', 'green'),
        ('Axial (X-Y)', 'axial', mid_ax, fa[:, :, mid_ax].T, 0, 1, 'X (mm)', 'Y (mm)', 'red'),
    ]
    
    for ax_idx, (title, view_name, slice_idx, fa_bg, x_dim, y_dim, xlabel, ylabel, color) in enumerate(views):
        ax = axes[ax_idx]
        
        # CORRECTED: Compute proper world-coordinate extent
        extent = compute_world_extent(fa.shape, affine, slice_idx, view_name)
        
        # Plot FA background - now properly aligned with streamlines
        ax.imshow(fa_bg, cmap='gray', origin='lower', extent=extent, 
                  alpha=0.3, vmin=0, vmax=0.8)
        
        # Plot streamlines (already in world coordinates)
        for sl in vis_streamlines:
            ax.plot(sl[:, x_dim], sl[:, y_dim], 
                   alpha=0.15, linewidth=0.3, color=color)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    fig_path = viz_dir / f"{stem}_streamlines_2d.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Streamlines 2D: {fig_path}")
    
    return fig_path


# =============================================================================
# STREAMLINE STATISTICS VISUALIZATION
# =============================================================================

def plot_streamline_statistics(
    streamlines,
    fa,
    seeds,
    affine,
    output_dir,
    stem,
    verbose=True
):
    """
    Create streamline statistics visualization.
    
    Shows length distribution histogram, seed point coverage,
    and basic tractography statistics.
    
    Parameters
    ----------
    streamlines : Streamlines
        Tractography streamlines.
    fa : ndarray
        3D FA map.
    seeds : ndarray
        Seed points used for tractography.
    affine : ndarray
        4x4 affine transformation matrix.
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
    from dipy.tracking.streamline import length
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    n_streamlines = len(streamlines)
    
    if n_streamlines == 0:
        if verbose:
            print("⚠️ No streamlines for statistics")
        return None
    
    # Compute lengths
    lengths = np.array([length(s) for s in streamlines])
    
    # Compute points per streamline
    points_per_sl = np.array([len(s) for s in streamlines])
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Tractography Statistics - {stem}", fontsize=14, fontweight='bold')
    
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Length histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(lengths, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(lengths):.1f} mm')
    ax1.axvline(np.median(lengths), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(lengths):.1f} mm')
    ax1.set_xlabel('Streamline Length (mm)')
    ax1.set_ylabel('Count')
    ax1.set_title('Length Distribution')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Points per streamline histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(points_per_sl, bins=50, color='forestgreen', edgecolor='white', alpha=0.8)
    ax2.axvline(np.mean(points_per_sl), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(points_per_sl):.0f}')
    ax2.set_xlabel('Points per Streamline')
    ax2.set_ylabel('Count')
    ax2.set_title('Streamline Resolution')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Length vs number of points scatter
    ax3 = fig.add_subplot(gs[0, 2])
    sample_idx = np.random.choice(n_streamlines, min(1000, n_streamlines), replace=False)
    ax3.scatter(lengths[sample_idx], points_per_sl[sample_idx], 
               alpha=0.3, s=10, color='purple')
    ax3.set_xlabel('Length (mm)')
    ax3.set_ylabel('Points')
    ax3.set_title('Length vs Points')
    ax3.grid(True, alpha=0.3)
    
    # Seed density visualization
    ax4 = fig.add_subplot(gs[1, 0])
    mid_ax = fa.shape[2] // 2
    ax4.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=0.8)
    
    # Convert seeds to voxel coordinates for plotting
    inv_affine = np.linalg.inv(affine)
    seeds_h = np.hstack([seeds, np.ones((len(seeds), 1))])
    seeds_vox = (seeds_h @ inv_affine.T)[:, :3]
    
    # Plot seeds near the axial slice
    near_slice = np.abs(seeds_vox[:, 2] - mid_ax) < 3
    ax4.scatter(seeds_vox[near_slice, 0], seeds_vox[near_slice, 1], 
               s=1, alpha=0.3, color='yellow')
    ax4.set_title(f'Seed Points (axial, z≈{mid_ax})')
    ax4.axis('off')
    
    # Cumulative length distribution
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_lengths = np.sort(lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    ax5.plot(sorted_lengths, cumulative, color='steelblue', linewidth=2)
    ax5.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax5.axvline(np.median(lengths), color='orange', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Length (mm)')
    ax5.set_ylabel('Cumulative %')
    ax5.set_title('Cumulative Length Distribution')
    ax5.grid(True, alpha=0.3)
    
    # Statistics text panel
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = (
        f"{'─' * 40}\n"
        f"TRACTOGRAPHY STATISTICS\n"
        f"{'─' * 40}\n\n"
        f"Total Streamlines:    {n_streamlines:,}\n"
        f"Total Seeds:          {len(seeds):,}\n"
        f"Yield Rate:           {n_streamlines/len(seeds)*100:.1f}%\n\n"
        f"Length Statistics:\n"
        f"  Mean:               {np.mean(lengths):.1f} mm\n"
        f"  Median:             {np.median(lengths):.1f} mm\n"
        f"  Std:                {np.std(lengths):.1f} mm\n"
        f"  Min:                {np.min(lengths):.1f} mm\n"
        f"  Max:                {np.max(lengths):.1f} mm\n"
        f"  IQR:                [{np.percentile(lengths, 25):.1f}, "
        f"{np.percentile(lengths, 75):.1f}] mm\n\n"
        f"Points per Streamline:\n"
        f"  Mean:               {np.mean(points_per_sl):.0f}\n"
        f"  Range:              [{np.min(points_per_sl)}, {np.max(points_per_sl)}]\n"
        f"{'─' * 40}"
    )
    
    ax6.text(0.5, 0.5, stats_text, transform=ax6.transAxes,
             fontsize=10, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    fig_path = viz_dir / f"{stem}_streamline_stats.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Streamline statistics: {fig_path}")
    
    return fig_path


# =============================================================================
# TRACKING SUMMARY
# =============================================================================

def create_tracking_summary(
    streamlines,
    fa,
    md,
    white_matter,
    brain_mask,
    seeds,
    affine,
    output_dir,
    stem,
    tracking_params=None,
    verbose=True
):
    """
    Create multi-panel tracking summary figure.
    
    Combines key visualizations into a single summary figure
    for quick assessment of tractography quality.
    
    Parameters
    ----------
    streamlines : Streamlines
        Tractography streamlines.
    fa : ndarray
        3D FA map.
    md : ndarray
        3D MD map.
    white_matter : ndarray
        3D white matter mask.
    brain_mask : ndarray
        3D brain mask.
    seeds : ndarray
        Seed points.
    affine : ndarray
        4x4 affine transformation matrix.
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
    tracking_params : dict, optional
        Tracking parameters for display.
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
    
    n_streamlines = len(streamlines)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Tractography Summary - {stem}", fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Get slice indices
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Row 0: FA, MD, WM mask in three views
    ax_fa = fig.add_subplot(gs[0, 0])
    ax_fa.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=1)
    ax_fa.set_title('FA (Axial)')
    ax_fa.axis('off')
    
    ax_md = fig.add_subplot(gs[0, 1])
    ax_md.imshow(md[:, :, mid_ax].T, cmap='hot', origin='lower')
    ax_md.set_title('MD (Axial)')
    ax_md.axis('off')
    
    ax_wm = fig.add_subplot(gs[0, 2])
    ax_wm.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=1)
    wm_overlay = np.ma.masked_where(white_matter[:, :, mid_ax].T == 0, white_matter[:, :, mid_ax].T)
    ax_wm.imshow(wm_overlay, cmap='Blues', alpha=0.5, origin='lower')
    ax_wm.set_title('WM Mask (Axial)')
    ax_wm.axis('off')
    
    # Row 0, col 3: Parameters
    ax_params = fig.add_subplot(gs[0, 3])
    ax_params.axis('off')
    
    if tracking_params:
        params_text = "TRACKING PARAMETERS\n" + "─" * 25 + "\n"
        for key, val in tracking_params.items():
            params_text += f"{key}: {val}\n"
    else:
        params_text = "TRACKING PARAMETERS\n" + "─" * 25 + "\nNot specified"
    
    ax_params.text(0.5, 0.5, params_text, transform=ax_params.transAxes,
                  fontsize=10, fontfamily='monospace',
                  verticalalignment='center', horizontalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Row 1: 2D streamline projections with CORRECTED extent
    if n_streamlines > 0:
        n_vis = min(3000, n_streamlines)
        indices = np.random.choice(n_streamlines, n_vis, replace=False)
        vis_sl = [streamlines[i] for i in indices]
        
        views_2d = [
            ('Sagittal', 'sagittal', mid_sag, fa[mid_sag, :, :].T, 1, 2, 'blue'),
            ('Coronal', 'coronal', mid_cor, fa[:, mid_cor, :].T, 0, 2, 'green'),
            ('Axial', 'axial', mid_ax, fa[:, :, mid_ax].T, 0, 1, 'red'),
        ]
        
        for col, (title, view_name, slice_idx, fa_bg, d1, d2, color) in enumerate(views_2d):
            ax = fig.add_subplot(gs[1, col])
            
            # CORRECTED: Use proper world-coordinate extent
            extent = compute_world_extent(fa.shape, affine, slice_idx, view_name)
            ax.imshow(fa_bg, cmap='gray', origin='lower', extent=extent,
                     alpha=0.3, vmin=0, vmax=0.8)
            
            for sl in vis_sl:
                ax.plot(sl[:, d1], sl[:, d2], alpha=0.1, linewidth=0.3, color=color)
            
            ax.set_title(f'{title}\n({n_vis:,} shown)')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
    
    # Row 1, col 3: Length histogram
    if n_streamlines > 0:
        lengths = np.array([length(s) for s in streamlines])
        ax_hist = fig.add_subplot(gs[1, 3])
        ax_hist.hist(lengths, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax_hist.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2)
        ax_hist.set_xlabel('Length (mm)')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Length Distribution')
        ax_hist.grid(True, alpha=0.3)
    
    # Row 2: Statistics summary
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    if n_streamlines > 0:
        lengths = np.array([length(s) for s in streamlines])
        stats_text = (
            f"{'═' * 100}\n"
            f"TRACTOGRAPHY SUMMARY\n"
            f"{'═' * 100}\n\n"
            f"Total Streamlines:     {n_streamlines:,}\n"
            f"Total Seeds:           {len(seeds):,}\n"
            f"Yield Rate:            {n_streamlines/len(seeds)*100:.1f}%\n\n"
            f"Length (mm):           mean={np.mean(lengths):.1f}, "
            f"median={np.median(lengths):.1f}, "
            f"range=[{np.min(lengths):.1f}, {np.max(lengths):.1f}]\n\n"
            f"White Matter:          {white_matter.sum():,} voxels "
            f"({white_matter.sum()/brain_mask.sum()*100:.1f}% of brain)\n"
            f"FA (brain mean):       {fa[brain_mask > 0].mean():.3f}\n"
            f"{'═' * 100}"
        )
    else:
        stats_text = "No streamlines generated"
    
    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                 fontsize=11, fontfamily='monospace',
                 verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    fig_path = viz_dir / f"{stem}_tracking_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Tracking summary: {fig_path}")
    
    return fig_path


# =============================================================================
# CONVENIENCE FUNCTION: Save all visualizations
# =============================================================================

def save_all_tracking_visualizations(
    streamlines,
    fa,
    md,
    white_matter,
    brain_mask,
    seeds,
    affine,
    output_dir,
    stem,
    tenfit=None,
    fa_thresh=0.2,
    tracking_params=None,
    verbose=True
):
    """
    Generate and save all tracking visualizations.
    
    Convenience function that calls all visualization functions
    and returns paths to all generated figures.
    
    Parameters
    ----------
    streamlines : Streamlines
        Tractography streamlines.
    fa : ndarray
        3D FA map.
    md : ndarray
        3D MD map.
    white_matter : ndarray
        3D white matter mask.
    brain_mask : ndarray
        3D brain mask.
    seeds : ndarray
        Seed points.
    affine : ndarray
        4x4 affine transformation matrix.
    output_dir : str or Path
        Output directory for saving figures.
    stem : str
        Subject/scan identifier for filenames.
    tenfit : TensorFit, optional
        Fitted tensor model for RGB visualization.
    fa_thresh : float, optional
        FA threshold used for WM mask.
    tracking_params : dict, optional
        Tracking parameters.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    viz_paths : dict
        Dictionary mapping visualization names to file paths.
    """
    if verbose:
        print("\nGenerating tracking visualizations...")
    
    viz_paths = {}
    
    # Tensor maps
    viz_paths['tensor_maps'] = plot_tensor_maps(
        fa, md, brain_mask, output_dir, stem, tenfit, verbose=verbose
    )
    
    # White matter mask
    viz_paths['wm_mask_qc'] = plot_white_matter_mask(
        fa, white_matter, brain_mask, output_dir, stem, fa_thresh, verbose=verbose
    )
    
    # 2D streamline projections
    viz_paths['streamlines_2d'] = plot_streamlines_2d(
        streamlines, fa, affine, output_dir, stem, verbose=verbose
    )
    
    # Streamline statistics
    viz_paths['streamline_stats'] = plot_streamline_statistics(
        streamlines, fa, seeds, affine, output_dir, stem, verbose=verbose
    )
    
    # Summary figure
    viz_paths['summary'] = create_tracking_summary(
        streamlines, fa, md, white_matter, brain_mask, seeds, affine,
        output_dir, stem, tracking_params, verbose=verbose
    )
    
    if verbose:
        print(f"✓ All tracking visualizations saved to: {Path(output_dir) / 'visualizations'}")
    
    return viz_paths
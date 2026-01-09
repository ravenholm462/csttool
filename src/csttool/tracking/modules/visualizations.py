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
        fa_slice = slicer(fa)
        axes[row, 0].imshow(fa_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
        if row == 0:
            axes[row, 0].set_title(f'FA\nmean={fa_brain.mean():.3f}')
        axes[row, 0].axis('off')
        
        # MD
        md_slice = slicer(md)
        # MD typically in range 0-0.003 mm²/s
        md_vmax = np.percentile(md_brain, 99) if md_brain.size > 0 else 0.003
        axes[row, 1].imshow(md_slice.T, cmap='hot', origin='lower', vmin=0, vmax=md_vmax)
        if row == 0:
            axes[row, 1].set_title(f'MD\nmean={md_brain.mean():.2e}')
        axes[row, 1].axis('off')
        
        # Brain mask overlay on FA
        mask_slice = slicer(brain_mask)
        axes[row, 2].imshow(fa_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes[row, 2].contour(mask_slice.T, levels=[0.5], colors='cyan', linewidths=1)
        if row == 0:
            axes[row, 2].set_title('FA + Brain Mask')
        axes[row, 2].axis('off')
        
        # RGB direction (if available)
        if rgb is not None:
            if dim == 2:  # Axial
                rgb_slice = rgb[:, :, slice_idx, :]
            elif dim == 1:  # Coronal
                rgb_slice = rgb[:, slice_idx, :, :]
            else:  # Sagittal
                rgb_slice = rgb[slice_idx, :, :, :]
            
            axes[row, 3].imshow(np.transpose(rgb_slice, (1, 0, 2)), origin='lower')
            if row == 0:
                axes[row, 3].set_title('RGB Direction\n(R=L-R, G=A-P, B=S-I)')
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
    Create white matter mask overlay visualization.
    
    Shows white matter mask (FA > threshold) overlaid on FA map
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
    
    Parameters
    ----------
    streamlines : Streamlines
        Tractography streamlines.
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
    
    # Convert to world coordinates for slice positions
    voxel_size = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Whole-Brain Tractography - {stem}\n"
                 f"{n_streamlines:,} streamlines (showing {n_vis:,})",
                 fontsize=14, fontweight='bold')
    
    # Define views: (title, x_dim, y_dim, fa_slice, x_label, y_label, color)
    views = [
        ('Sagittal (Y-Z)', 1, 2, fa[mid_sag, :, :].T, 'Y (mm)', 'Z (mm)', 'blue'),
        ('Coronal (X-Z)', 0, 2, fa[:, mid_cor, :].T, 'X (mm)', 'Z (mm)', 'green'),
        ('Axial (X-Y)', 0, 1, fa[:, :, mid_ax].T, 'X (mm)', 'Y (mm)', 'red'),
    ]
    
    for ax_idx, (title, dim1, dim2, fa_bg, xlabel, ylabel, color) in enumerate(views):
        ax = axes[ax_idx]
        
        # Plot FA background (need to compute extent in world coordinates)
        extent = [0, fa_bg.shape[1] * voxel_size[dim1],
                  0, fa_bg.shape[0] * voxel_size[dim2]]
        ax.imshow(fa_bg, cmap='gray', origin='lower', extent=extent, 
                  alpha=0.3, vmin=0, vmax=0.8)
        
        # Plot streamlines
        for sl in vis_streamlines:
            ax.plot(sl[:, dim1], sl[:, dim2], 
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
    # Subsample for plotting efficiency
    n_plot = min(5000, n_streamlines)
    idx = np.random.choice(n_streamlines, n_plot, replace=False)
    ax3.scatter(points_per_sl[idx], lengths[idx], alpha=0.3, s=5, c='purple')
    ax3.set_xlabel('Points per Streamline')
    ax3.set_ylabel('Length (mm)')
    ax3.set_title('Length vs Resolution')
    ax3.grid(True, alpha=0.3)
    
    # Seed coverage map (axial view)
    ax4 = fig.add_subplot(gs[1, 0])
    mid_ax = fa.shape[2] // 2
    ax4.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=0.8)
    
    # Convert seeds to voxel coordinates and plot
    if seeds is not None and len(seeds) > 0:
        # Transform world coordinates to voxel
        inv_affine = np.linalg.inv(affine)
        seeds_vox = np.dot(seeds, inv_affine[:3, :3].T) + inv_affine[:3, 3]
        
        # Filter seeds near the axial slice
        near_slice = np.abs(seeds_vox[:, 2] - mid_ax) < 3
        seeds_slice = seeds_vox[near_slice]
        
        if len(seeds_slice) > 0:
            ax4.scatter(seeds_slice[:, 0], seeds_slice[:, 1], 
                       c='red', s=1, alpha=0.3)
    
    ax4.set_title(f'Seed Points (axial z={mid_ax})\n{len(seeds):,} total seeds')
    ax4.axis('off')
    
    # Cumulative length distribution
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_lengths = np.sort(lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    ax5.plot(sorted_lengths, cumulative, 'b-', linewidth=2)
    ax5.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax5.axvline(np.median(lengths), color='orange', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Length (mm)')
    ax5.set_ylabel('Cumulative Fraction')
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
    lengths = np.array([length(s) for s in streamlines]) if n_streamlines > 0 else np.array([])
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Tracking Summary - {stem}", fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 4, hspace=0.25, wspace=0.25)
    
    # Slice indices
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Row 0: FA, MD, WM mask, Brain mask
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=1)
    ax.set_title(f'FA (axial)\nmean={fa[brain_mask>0].mean():.3f}')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 1])
    md_vmax = np.percentile(md[brain_mask>0], 99) if brain_mask.any() else 0.003
    ax.imshow(md[:, :, mid_ax].T, cmap='hot', origin='lower', vmin=0, vmax=md_vmax)
    ax.set_title(f'MD (axial)\nmean={md[brain_mask>0].mean():.2e}')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=1)
    wm_overlay = np.ma.masked_where(white_matter[:, :, mid_ax].T == 0, 
                                     white_matter[:, :, mid_ax].T)
    ax.imshow(wm_overlay, cmap='Blues', alpha=0.5, origin='lower')
    ax.set_title(f'White Matter\n{white_matter.sum():,} voxels')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=1)
    ax.contour(brain_mask[:, :, mid_ax].T, levels=[0.5], colors='cyan', linewidths=1.5)
    ax.set_title(f'Brain Mask\n{brain_mask.sum():,} voxels')
    ax.axis('off')
    
    # Row 1: 2D streamline projections
    if n_streamlines > 0:
        max_vis = min(3000, n_streamlines)
        indices = np.random.choice(n_streamlines, max_vis, replace=False)
        vis_sl = [streamlines[i] for i in indices]
        
        voxel_size = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
        
        views = [
            ('Sagittal', 1, 2, fa[mid_sag, :, :].T, 'blue'),
            ('Coronal', 0, 2, fa[:, mid_cor, :].T, 'green'),
            ('Axial', 0, 1, fa[:, :, mid_ax].T, 'red'),
        ]
        
        for col, (name, d1, d2, bg, color) in enumerate(views):
            ax = fig.add_subplot(gs[1, col])
            extent = [0, bg.shape[1] * voxel_size[d1], 0, bg.shape[0] * voxel_size[d2]]
            ax.imshow(bg, cmap='gray', origin='lower', extent=extent, alpha=0.3, vmin=0, vmax=0.8)
            for sl in vis_sl:
                ax.plot(sl[:, d1], sl[:, d2], alpha=0.1, linewidth=0.3, color=color)
            ax.set_title(f'{name}\n({max_vis:,} shown)')
            ax.set_aspect('equal')
            ax.axis('off')
    else:
        for col in range(3):
            ax = fig.add_subplot(gs[1, col])
            ax.text(0.5, 0.5, 'No streamlines', ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    # Row 1, col 3: Length histogram
    ax = fig.add_subplot(gs[1, 3])
    if n_streamlines > 0:
        ax.hist(lengths, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Length (mm)')
        ax.set_ylabel('Count')
        ax.set_title(f'Length Distribution\nmean={np.mean(lengths):.1f} mm')
    else:
        ax.text(0.5, 0.5, 'No streamlines', ha='center', va='center', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Row 2: Statistics panel
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    # Build parameters string
    if tracking_params:
        params_str = (
            f"Step Size: {tracking_params.get('step_size', 'N/A')} mm | "
            f"FA Threshold: {tracking_params.get('fa_thresh', 'N/A')} | "
            f"Seed Density: {tracking_params.get('seed_density', 'N/A')} | "
            f"SH Order: {tracking_params.get('sh_order', 'N/A')}"
        )
    else:
        params_str = "Parameters not available"
    
    if n_streamlines > 0:
        length_stats = (
            f"Length: mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}, "
            f"range=[{np.min(lengths):.1f}, {np.max(lengths):.1f}] mm"
        )
    else:
        length_stats = "No streamlines generated"
    
    stats_text = (
        f"{'═' * 100}\n"
        f"TRACKING SUMMARY\n"
        f"{'═' * 100}\n\n"
        f"Parameters:      {params_str}\n\n"
        f"Streamlines:     {n_streamlines:,}\n"
        f"Seeds:           {len(seeds):,}\n"
        f"Yield Rate:      {n_streamlines/len(seeds)*100:.1f}%\n\n"
        f"{length_stats}\n\n"
        f"FA Statistics:   mean={fa[brain_mask>0].mean():.3f}, std={fa[brain_mask>0].std():.3f}\n"
        f"MD Statistics:   mean={md[brain_mask>0].mean():.2e}, std={md[brain_mask>0].std():.2e}\n"
        f"{'═' * 100}"
    )
    
    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
            fontsize=11, fontfamily='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    fig_path = viz_dir / f"{stem}_tracking_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Tracking summary: {fig_path}")
    
    return fig_path


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
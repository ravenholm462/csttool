"""
funcs.py

Utility functions for csttool's tractography pipeline.

This module provides:

1. Tensor fitting and scalar measures (FA, MD)
2. Direction field estimation with a CSA ODF model
3. FA based stopping criterion and seed generation
4. Deterministic local tracking
5. Tractogram saving helper
"""

from pathlib import Path
from os.path import join

import numpy as np

from dipy.tracking import utils

# Reconstruction imports
from dipy.reconst import dti
from dipy.reconst.shm import CsaOdfModel

# Direction field imports
from dipy.direction import peaks_from_model
from dipy.data import get_sphere

# Termination imports
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

# Tracking imports
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram


def fit_tensor(data, gtab, brain_mask=None):
    """
    Fit a diffusion tensor model inside a brain mask.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, N)
        Preprocessed diffusion data.
    gtab : GradientTable
        Gradient table corresponding to `data`.
    brain_mask : ndarray or None
        Boolean mask of voxels to include in the fit.

    Returns
    -------
    tenfit : TensorFit
        Fitted tensor model.
    """
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=brain_mask)
    print("Tensor model fit complete.")
    return tenfit


def compute_measures(tenfit):
    """
    Compute FA and MD from a tensor fit.

    Parameters
    ----------
    tenfit : TensorFit
        Fitted tensor model.

    Returns
    -------
    fa : ndarray, shape (X, Y, Z)
        Fractional anisotropy map.
    md : ndarray, shape (X, Y, Z)
        Mean diffusivity map.
    """
    print("Computing FA and MD")

    fa = tenfit.fa
    fa[np.isnan(fa)] = 0

    md = dti.mean_diffusivity(tenfit.evals)

    fa = np.nan_to_num(tenfit.fa, nan=0.0, posinf=0.0, neginf=0.0)
    md = np.nan_to_num(dti.mean_diffusivity(tenfit.evals), nan=0.0, posinf=0.0, neginf=0.0)

    if np.any(fa > 0):
        fa_nonzero = fa[fa > 0]
        print(f"FA: min {fa_nonzero.min():.3f}, "
              f"max {fa_nonzero.max():.3f}, "
              f"mean {fa_nonzero.mean():.3f}")
    if np.any(md > 0):
        md_nonzero = md[md > 0]
        print(f"MD: min {md_nonzero.min():.3e}, "
              f"max {md_nonzero.max():.3e}, "
              f"mean {md_nonzero.mean():.3e}")

    return fa, md


def get_directions(
        data,
        gtab,
        mask,
        sh_order_max=6,
        sphere_name="symmetric362",
):
    """
    Estimate principal directions with a CSA ODF model.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, N)
        Preprocessed diffusion data.
    gtab : GradientTable
        Gradient table.
    mask : ndarray, shape (X, Y, Z)
        Boolean white matter mask.
    sh_order_max : int
        Maximum spherical harmonic order for CSA.
    sphere_name : str
        Name of the precomputed sphere in DIPY data.
        Controls the angular sampling grid.

    Returns
    -------
    peaks : PeaksAndMetrics
        Direction field and scalar metrics for tracking.
        This object can be passed directly to LocalTracking.
    """
    print("Estimating direction field with CSA ODF model")
    sphere = get_sphere(name=sphere_name)

    csa_model = CsaOdfModel(gtab, sh_order_max)
    peaks = peaks_from_model(
        model=csa_model,
        data=data,
        sphere=sphere,
        relative_peak_threshold=0.8,
        min_separation_angle=45,
        mask=mask,
        npeaks=1,
    )

    return peaks


def terminate_and_seed(
        fa,
        affine,
        fa_thr=0.2,
        seed_density=1,
):
    """
    Create FA based stopping criterion and seeds.

    Parameters
    ----------
    fa : ndarray, shape (X, Y, Z)
        Fractional anisotropy map.
    affine : ndarray, shape (4, 4)
        Affine of the diffusion image.
    fa_thr : float
        FA threshold used for both stopping and seeding.
    seed_density : int
        Number of seeds per voxel in the seed mask.

    Returns
    -------
    stopping_criterion : ThresholdStoppingCriterion
        Stopping rule for LocalTracking.
    seeds : ndarray, shape (M, 3)
        Seed points in world coordinates.
    seed_mask : ndarray, bool
        Seed mask in voxel space.
    """
    print(f"Building stopping criterion and seeds with FA > {fa_thr}")

    stopping_criterion = ThresholdStoppingCriterion(fa, fa_thr)

    seed_mask = fa >= fa_thr
    seeds = utils.seeds_from_mask(seed_mask, affine=affine, density=seed_density)

    return stopping_criterion, seeds, seed_mask


def run_deterministic_tracking(
        direction_getter,
        stopping_criterion,
        seeds,
        affine,
        step_size=0.5,
):
    """
    Run deterministic local tractography.

    Parameters
    ----------
    direction_getter : PeaksAndMetrics or DirectionGetter
        Object returned by `get_directions` or another direction model.
    stopping_criterion : StoppingCriterion
        Stopping rule for the tracker.
    seeds : ndarray, shape (M, 3)
        Seed points in world coordinates.
    affine : ndarray, shape (4, 4)
        Affine of the diffusion image.
    step_size : float
        Step size in millimetres.

    Returns
    -------
    streamlines : Streamlines
        Generated streamlines.
    """
    print("Running deterministic tractography")

    streamline_generator = LocalTracking(
        direction_getter,
        stopping_criterion,
        seeds,
        affine=affine,
        step_size=step_size,
    )

    streamlines = Streamlines(streamline_generator)
    print(f"Generated {len(streamlines)} streamlines")

    return streamlines


def extract_cst_using_fss(streamlines, affine, radius=15.0):
    """Use DIPY's Fast Streamline Search with HCP CST atlas to extract only the CST streamlines.

    Args:
        streamlines (Streamlines): All generated streamlines
        affine (ndarray): Subject's affine transformation matrix
        radius (float, optional): Radius parameter for FSS. Defaults to 7.0.

    Returns:
        tuple: (cst_streamlines, atlas_cst)
    """
    from dipy.data import fetch_bundle_atlas_hcp842, get_two_hcp842_bundles
    from dipy.segment.fss import FastStreamlineSearch
    from dipy.io.streamline import load_trk
    
    print("Loading HCP CST atlas...")
    # 1. Fetch HCP atlas with CST bundle
    fetch_bundle_atlas_hcp842()
    _, model_cst_l_file = get_two_hcp842_bundles()
    
    # 2. Load atlas CST - IMPORTANT: Use "same" to load in scanner space
    sft_cst_atlas = load_trk(model_cst_l_file, "same", bbox_valid_check=False)
    atlas_cst = sft_cst_atlas.streamlines
    
    print(f"Atlas loaded: {len(atlas_cst)} streamlines")
    print(f"Subject streamlines: {len(streamlines)}")
    
    # 3. Fast Streamline Search
    print(f"Running Fast Streamline Search with radius={radius}mm...")
    fss = FastStreamlineSearch(ref_streamlines=atlas_cst, max_radius=radius)
    distance_matrix = fss.radius_search(streamlines, radius=radius)
    
    # 4. Extract recognized CST
    recognized_indices = np.unique(distance_matrix.row)
    
    if len(recognized_indices) == 0:
        print("No CST streamlines found! Possible reasons:")
        print(f"   - Radius ({radius}mm) might be too small")
        print(f"   - Subject and atlas may be in different spaces")
        print(f"   - Try increasing --fss-radius (e.g., 15.0)")
        
        # Return empty CST but keep atlas for reference
        return Streamlines(), atlas_cst
    
    cst_streamlines = streamlines[recognized_indices]
    print(f"CST extraction: {len(streamlines)} → {len(cst_streamlines)} streamlines")
    
    return cst_streamlines, atlas_cst


def save_tractogram_trk(
        streamlines,
        img,
        out_dir,
        fname="tractogram_deterministic",
):
    """
    Save streamlines to a .trk file in RASMM space.

    Parameters
    ----------
    streamlines : Streamlines
        Streamlines to save.
    img : Nifti1Image
        Reference image (usually the diffusion image).
    out_dir : str or Path
        Output directory.
    fname : str
        Base filename without extension.

    Returns
    -------
    out_path : Path
        Path to the saved tractogram.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{fname}.trk"

    sft = StatefulTractogram(streamlines, img, Space.RASMM)
    save_tractogram(sft, str(out_path))

    print(f"Saved tractogram to {out_path}")

    return out_path


def save_scalar_maps(fa, md, affine, out_dir, stem):
    """
    Save FA and MD maps for later analysis.
    """
    import nibabel as nib
    from pathlib import Path
    import json
    
    out_dir = Path(out_dir)
    scalar_dir = out_dir / "scalar_maps"
    scalar_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # Save FA map
    if fa is not None:
        fa_path = scalar_dir / f"{stem}_fa.nii.gz"
        nib.save(nib.Nifti1Image(fa.astype(np.float32), affine), fa_path)
        outputs['fa_map'] = str(fa_path)
        print(f"✓ FA map saved: {fa_path}")
    
    # Save MD map  
    if md is not None:
        md_path = scalar_dir / f"{stem}_md.nii.gz"
        nib.save(nib.Nifti1Image(md.astype(np.float32), affine), md_path)
        outputs['md_map'] = str(md_path)
        print(f"✓ MD map saved: {md_path}")
    
    return outputs

def save_tracking_report(streamlines, out_dir, stem, tractogram_paths, 
                        scalar_outputs, cst_streamlines_count=None):
    """
    Save tracking processing report with comparison data.
    """
    from pathlib import Path
    import json
    from datetime import datetime
    
    out_dir = Path(out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'processing_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'subject_stem': stem,
        'streamline_counts': {
            'whole_brain': len(streamlines),
            'cst_extracted': cst_streamlines_count if cst_streamlines_count else 0,
            'reduction_percentage': (1 - cst_streamlines_count/len(streamlines))*100 if cst_streamlines_count and len(streamlines) > 0 else 0
        },
        'output_files': {
            'whole_brain_tractogram': str(tractogram_paths['full_tractogram']),
            'cst_tractogram': str(tractogram_paths['cst_tractogram']),
            'atlas_reference': str(tractogram_paths['atlas_reference']),
            **scalar_outputs
        }
    }
    
    report_path = log_dir / f"{stem}_tracking_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report_path


def visualize_cst_comparison(whole_brain_streamlines, cst_streamlines, 
                           atlas_streamlines, fa_map, affine, 
                           output_dir, subject_id):
    """
    Create comparison visualization showing whole brain vs extracted CST vs atlas.
    """
    import matplotlib.pyplot as plt
    from dipy.viz import window, actor
    from pathlib import Path
    
    output_dir = Path(output_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create side-by-side comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'CST Extraction Results - {subject_id}', fontsize=16)
    
    titles = ['Whole Brain', 'Extracted CST', 'Atlas Reference']
    streamline_sets = [whole_brain_streamlines, cst_streamlines, atlas_streamlines]
    colors = [(0.8, 0.8, 0.8), (0, 0, 1), (0, 1, 0)]
    
    for ax, title, streamlines_set, color in zip(axes, titles, streamline_sets, colors):
        # Simple 2D projection visualization
        if len(streamlines_set) > 0:
            # Get all points
            all_points = np.vstack(streamlines_set)
            ax.scatter(all_points[:, 0], all_points[:, 1], 
                      alpha=0.1, s=1, color=color)
        ax.set_title(f'{title}\n({len(streamlines_set)} streamlines)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    scatter_path = output_dir / f"{subject_id}_cst_comparison_scatter.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Create 3D rendering
    try:
        renderer = window.Renderer()
        renderer.SetBackground(1, 1, 1)
        
        # Whole brain in light gray (transparent)
        renderer.add(actor.line(whole_brain_streamlines, 
                              colors=(0.8, 0.8, 0.8), opacity=0.2))
        
        # Extracted CST in blue
        renderer.add(actor.line(cst_streamlines, 
                              colors=(0, 0, 1), linewidth=2, opacity=0.8))
        
        # Atlas reference in green
        renderer.add(actor.line(atlas_streamlines, 
                              colors=(0, 1, 0), linewidth=1, opacity=0.6))
        
        renderer.set_camera(position=(200, 200, 200))
        render_path = output_dir / f"{subject_id}_cst_comparison_3d.png"
        window.record(renderer, out_path=str(render_path), size=(800, 600))
        
    except Exception as e:
        print(f"⚠️  3D visualization failed: {e}")
        render_path = None
    
    print(f"✓ Comparison visualization saved to: {scatter_path}")
    return scatter_path
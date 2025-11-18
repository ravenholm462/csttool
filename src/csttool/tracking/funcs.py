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

def save_tracking_report(streamlines, out_dir, stem, tract_path, scalar_outputs):
    """
    Save tracking processing report.
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
        'streamline_count': len(streamlines),
        'output_files': {
            'tractogram': str(tract_path),
            **scalar_outputs
        }
    }
    
    report_path = log_dir / f"{stem}_tracking_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report_path
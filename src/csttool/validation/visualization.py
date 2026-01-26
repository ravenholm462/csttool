"""
Visualization utilities for CST Validation.
"""

from pathlib import Path
import numpy as np
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

from .bundle_comparison import (
    _load_streamlines,
    _load_ref_anatomy,
    _streamlines_to_density
)

def save_overlap_maps(
    candidate_path: str | Path,
    reference_path: str | Path,
    ref_space_path: str | Path,
    output_dir: str | Path,
    prefix: str = "val"
) -> dict:
    """
    Generate and save occupancy maps for candidate, reference, and overlap.
    
    Args:
        candidate_path: Path to candidate .trk
        reference_path: Path to reference .trk
        ref_space_path: Path to reference NIfTI
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        dict with paths to generated NIfTI files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cand_streamlines, _, _ = _load_streamlines(candidate_path)
    ref_streamlines, _, _ = _load_streamlines(reference_path)
    grid_affine, grid_shape, _ = _load_ref_anatomy(ref_space_path)
    
    # Generate binary density maps
    cand_occ = _streamlines_to_density(cand_streamlines, grid_affine, grid_shape)
    ref_occ = _streamlines_to_density(ref_streamlines, grid_affine, grid_shape)
    
    overlap_occ = (cand_occ * ref_occ).astype(np.float32)
    
    # Save NIfTIs
    paths = {}
    
    def _save(data, suffix):
        img = nib.Nifti1Image(data, grid_affine)
        p = output_dir / f"{prefix}_{suffix}.nii.gz"
        nib.save(img, p)
        return p

    paths["candidate"] = _save(cand_occ, "candidate_occ")
    paths["reference"] = _save(ref_occ, "reference_occ")
    paths["overlap"] = _save(overlap_occ, "overlap_occ")
    
    return paths


def save_validation_snapshots(
    ref_space_path: str | Path,
    overlay_paths: dict,
    output_dir: str | Path,
    prefix: str = "val"
):
    """
    Generate static PNG snapshots (Axial, Coronal, Sagittal).
    Uses 'candidate' (Red) and 'reference' (Blue) overlays.
    """
    output_dir = Path(output_dir)
    bg_img = str(ref_space_path)
    
    cand_img = str(overlay_paths["candidate"])
    ref_img  = str(overlay_paths["reference"])
    
    # Plotting loop for 3 views
    coords = None # Auto-detect cut coords
    
    for view in ["x", "y", "z"]:
        # Create figure: Background + Reference (Blue) + Candidate (Red)
        # Using plot_roi for binary masks
        
        # We use a trick: plot background, then add contours/roi
        display = plotting.plot_anat(
            bg_img, 
            display_mode=view, 
            cut_coords=1, # single slice or auto
            title=f"Validation ({view})"
        )
        
        # Add Reference (Blue)
        display.add_overlay(
            ref_img, 
            cmap=plotting.cm.blue_transparent, 
            alpha=0.6,
        )
        
        # Add Candidate (Red)
        display.add_overlay(
            cand_img, 
            cmap=plotting.cm.red_transparent, 
            alpha=0.6, 
        )
        
        # Save
        out_path = output_dir / f"{prefix}_snapshot_{view}.png"
        display.savefig(str(out_path))
        display.close()

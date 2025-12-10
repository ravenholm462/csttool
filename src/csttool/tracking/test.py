"""
tracking/test.py

Test script for csttool's tractography pipeline with image-based registration
to MNI space for atlas-based CST extraction.

Usage:
    python test.py --visualize        # Enable all visualizations
    python test.py --no-visualize     # Disable visualizations (default)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import csttool.preprocess.funcs as preproc
import csttool.tracking.funcs as trk

# Parse arguments
parser = argparse.ArgumentParser(description="CST tracking test with registration")
parser.add_argument("--visualize", action="store_true", help="Enable visualizations")
parser.add_argument("--dicom-dir", type=str, default=None, help="DICOM directory path")
parser.add_argument("--nifti-dir", type=str, default=None, help="NIfTI directory path")
parser.add_argument("--out-dir", type=str, default=None, help="Output directory path")
args = parser.parse_args()

#VISUALIZE = args.visualize
VISUALIZE = True

# Paths - update these or pass via command line
if args.dicom_dir:
    dicom_dir = args.dicom_dir
else:
    dicom_dir = "/home/alemnalo/anom/cmrr_mbep2d_diff_AP_TDI_Series0017"
    # dicom_dir = "/home/alem/Documents/thesis/data/anom/cmrr_mbep2d_diff_AP_TDI_Series0017"

if args.out_dir:
    out_dir = args.out_dir
else:
    out_dir = "/home/alemnalo/anom/out/"
    # out_dir = "/home/alem/Documents/thesis/data/out/"

# Create output directories
os.makedirs(out_dir, exist_ok=True)
vis_dir = os.path.join(out_dir, "visualizations")
os.makedirs(vis_dir, exist_ok=True)

######################### LOAD DATA #########################

print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

nii, bval, bvec = preproc.convert_to_nifti(dicom_dir, out_dir)
print(f"NIfTI file: {nii}")

nii_dirname = os.path.dirname(nii)
nii_fname = os.path.basename(nii).split('.')[0]

data, affine, img, gtab = preproc.load_dataset(
    nifti_path=nii_dirname,
    fname=nii_fname,
    visualize=VISUALIZE
)

print(f"Data shape: {data.shape}")
print(f"Affine:\n{affine}")
print(f"Gradient table: {len(gtab.bvals)} volumes")

######################### CREATE MASK #########################

print("\n" + "=" * 70)
print("STEP 2: BRAIN MASKING")
print("=" * 70)

from dipy.segment.mask import median_otsu

masked_data, brain_mask = preproc.background_segmentation(
    data,
    gtab,
    visualize=VISUALIZE
)

######################### TENSOR FITTING & FA #########################

print("\n" + "=" * 70)
print("STEP 3: TENSOR FITTING")
print("=" * 70)

from dipy.reconst.dti import TensorModel

tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(data, mask=brain_mask)
fa = tenfit.fa
fa = np.nan_to_num(fa, nan=0.0)

# Create white matter mask
white_matter = fa > 0.15
white_matter = white_matter & brain_mask

# Dilate to reach grey matter
from scipy.ndimage import binary_dilation
wm_before_dilation = white_matter.sum()
white_matter = binary_dilation(white_matter, iterations=1)
wm_after_dilation = white_matter.sum()

print(f"Brain mask: {brain_mask.sum():,} voxels")
print(f"White matter (FA > 0.15): {wm_before_dilation:,} voxels")
print(f"Dilated WM: {wm_after_dilation:,} voxels")

# Visualization: Masks
if VISUALIZE:
    from dipy.core.histeq import histeq
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Brain and White Matter Masks", fontsize=14)
    
    mid_slice = data.shape[2] // 2
    b0_slice = data[:, :, mid_slice, 0]
    
    # Row 1: Brain mask
    axes[0, 0].imshow(b0_slice.T, cmap='gray', origin='lower')
    axes[0, 0].set_title('Original b0')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(brain_mask[:, :, mid_slice].T, cmap='gray', origin='lower')
    axes[0, 1].set_title(f'Brain Mask\n({brain_mask.sum():,} voxels)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(b0_slice.T, cmap='gray', origin='lower')
    axes[0, 2].imshow(brain_mask[:, :, mid_slice].T, cmap='Reds', alpha=0.5, origin='lower')
    axes[0, 2].set_title('Brain Mask Overlay')
    axes[0, 2].axis('off')
    
    # Row 2: White matter
    axes[1, 0].imshow(fa[:, :, mid_slice].T, cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[1, 0].set_title('FA Map')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(white_matter[:, :, mid_slice].T, cmap='gray', origin='lower')
    axes[1, 1].set_title(f'White Matter Mask\n({wm_after_dilation:,} voxels)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(b0_slice.T, cmap='gray', origin='lower')
    axes[1, 2].imshow(white_matter[:, :, mid_slice].T, cmap='Blues', alpha=0.5, origin='lower')
    axes[1, 2].set_title('White Matter Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "01_masks.png"), dpi=150, bbox_inches='tight')
    plt.show()

######################### CREATE STREAMLINES #########################

print("\n" + "=" * 70)
print("STEP 4: TRACTOGRAPHY")
print("=" * 70)

from dipy.reconst import shm
from dipy.direction import peaks
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines

# Code adapted from https://docs.dipy.org/dev/examples_built/streamline_analysis/streamline_tools.html
# CSA ODF model
csamodel = shm.CsaOdfModel(gtab, 6)
csapeaks = peaks.peaks_from_model(
    model=csamodel,
    data=data,
    sphere=peaks.default_sphere,
    relative_peak_threshold=0.8,
    min_separation_angle=45,
    mask=white_matter,
)

# Seeds and stopping criterion
seeds = utils.seeds_from_mask(white_matter, affine, density=1)
stopping_criterion = BinaryStoppingCriterion(white_matter)

print(f"Number of seeds: {len(seeds):,}")

# Run tracking
streamline_generator = LocalTracking(
    csapeaks, stopping_criterion, seeds, affine=affine, step_size=0.5
)
streamlines = Streamlines(streamline_generator)

print(f"Generated {len(streamlines):,} streamlines")

# Visualization: Native space streamlines (2D projections)
if VISUALIZE:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Whole-Brain Tractography - Native Space ({len(streamlines):,} streamlines)", fontsize=14)
    
    # Sample streamlines for visualization (max 5000)
    n_vis = min(5000, len(streamlines))
    vis_indices = np.random.choice(len(streamlines), n_vis, replace=False)
    
    # Sagittal projection (Y-Z)
    ax = axes[0]
    for idx in vis_indices:
        s = streamlines[idx]
        ax.plot(s[:, 1], s[:, 2], alpha=0.1, linewidth=0.5, color='blue')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Sagittal View (Y-Z)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Coronal projection (X-Z)
    ax = axes[1]
    for idx in vis_indices:
        s = streamlines[idx]
        ax.plot(s[:, 0], s[:, 2], alpha=0.1, linewidth=0.5, color='green')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Coronal View (X-Z)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Axial projection (X-Y)
    ax = axes[2]
    for idx in vis_indices:
        s = streamlines[idx]
        ax.plot(s[:, 0], s[:, 1], alpha=0.1, linewidth=0.5, color='red')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Axial View (X-Y)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "02_streamlines_native.png"), dpi=150, bbox_inches='tight')
    plt.show()

# Optional: 3D visualization with FURY
if VISUALIZE:
    try:
        from dipy.viz import window, actor, has_fury
        
        if has_fury:
            print("Creating 3D visualization of native streamlines...")
            scene = window.Scene()
            scene.SetBackground(1, 1, 1)
            
            # Subsample for visualization
            n_3d = min(10000, len(streamlines))
            vis_streamlines = Streamlines([streamlines[i] for i in 
                                          np.random.choice(len(streamlines), n_3d, replace=False)])
            
            stream_actor = actor.line(vis_streamlines, linewidth=0.5, opacity=0.3)
            scene.add(stream_actor)
            
            # Add FA slice as background
            fa_actor = actor.slicer(fa, affine=affine, opacity=0.6)
            fa_actor.display(z=fa.shape[2]//2)
            scene.add(fa_actor)
            
            scene.set_camera(position=(300, 300, 200), focal_point=(0, 0, 0))
            
            window.record(scene, out_path=os.path.join(vis_dir, "02_streamlines_native_3d.png"),
                         size=(800, 600))
            print(f"Saved 3D visualization to {vis_dir}/02_streamlines_native_3d.png")
    except Exception as e:
        print(f"3D visualization failed: {e}")


######################### IMAGE BASED REGISTRATION #########################

# Source on registration: https://docs.dipy.org/stable/interfaces/registration_flow.html

# Solution 1: Register a whole brain tractogram to a whole brain atlas. This is also called streamline based registration.
# See here: https://docs.dipy.org/stable/interfaces/bundle_segmentation_flow.html

# Solution 2: Image based registration
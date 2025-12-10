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
    #dicom_dir = "/home/alemnalo/anom/cmrr_mbep2d_diff_AP_TDI_Series0017"
    dicom_dir = "/home/alem/Documents/thesis/data/anom/cmrr_mbep2d_diff_AP_TDI_Series0017"

if args.out_dir:
    out_dir = args.out_dir
else:
    #out_dir = "/home/alemnalo/anom/out/"
    out_dir = "/home/alem/Documents/thesis/data/out/"

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
white_matter = fa > 0.2
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
    plt.savefig(os.path.join(vis_dir, "masks.png"), dpi=150, bbox_inches='tight')
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
    plt.savefig(os.path.join(vis_dir, "streamlines_native.png"), dpi=150, bbox_inches='tight')
    plt.show()

######################### IMAGE BASED REGISTRATION #########################

# Source on registration: https://docs.dipy.org/stable/interfaces/registration_flow.html

# Solution 1: Register a whole brain tractogram to a whole brain atlas. This is also called streamline based registration.
# See here: https://docs.dipy.org/stable/interfaces/bundle_segmentation_flow.html

# Solution 2: Image based registration - implemented below

print("\n" + "=" * 70)
print("STEP 5: REGISTRATION TO MNI SPACE")
print("=" * 70)

from dipy.align import affine_registration, syn_registration
from dipy.align.reslice import reslice
from dipy.data.fetcher import fetch_mni_template, read_mni_template
from dipy.tracking.streamline import transform_streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram

# 1. Get mean b0 from subject data
b0_idx = np.where(gtab.bvals < 50)[0]
b0_data = data[..., b0_idx]
mean_b0 = np.mean(b0_data, axis=3, dtype=data.dtype)
mean_b0_masked = mean_b0 * brain_mask

print(f"Mean b0 shape: {mean_b0.shape}")

# 2. Load MNI T2 template
print("Fetching MNI template...")
fetch_mni_template()
img_t2_mni = read_mni_template(version="a", contrast="T2")
t2_mni_data = img_t2_mni.get_fdata()
t2_mni_affine = img_t2_mni.affine
t2_mni_voxel_size = img_t2_mni.header.get_zooms()[:3]

print(f"MNI template shape: {t2_mni_data.shape}")
print(f"MNI voxel size: {t2_mni_voxel_size}")

# 3. Reslice MNI template to match subject resolution
subject_voxel_size = img.header.get_zooms()[:3]
print(f"Subject voxel size: {subject_voxel_size}")

t2_resliced_data, t2_resliced_affine = reslice(
    t2_mni_data, t2_mni_affine, t2_mni_voxel_size, subject_voxel_size
)
print(f"Resliced MNI template shape: {t2_resliced_data.shape}")

# Visualization: Before registration
if VISUALIZE:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Before Registration: Subject b0 vs MNI Template", fontsize=14)
    
    # Subject b0 slices
    mid_ax = mean_b0_masked.shape[0] // 2
    mid_cor = mean_b0_masked.shape[1] // 2
    mid_sag = mean_b0_masked.shape[2] // 2
    
    axes[0, 0].imshow(mean_b0_masked[mid_ax, :, :].T, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'Subject b0 - Sagittal (x={mid_ax})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mean_b0_masked[:, mid_cor, :].T, cmap='gray', origin='lower')
    axes[0, 1].set_title(f'Subject b0 - Coronal (y={mid_cor})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mean_b0_masked[:, :, mid_sag].T, cmap='gray', origin='lower')
    axes[0, 2].set_title(f'Subject b0 - Axial (z={mid_sag})')
    axes[0, 2].axis('off')
    
    # MNI template slices
    mid_ax_mni = t2_resliced_data.shape[0] // 2
    mid_cor_mni = t2_resliced_data.shape[1] // 2
    mid_sag_mni = t2_resliced_data.shape[2] // 2
    
    axes[1, 0].imshow(t2_resliced_data[mid_ax_mni, :, :].T, cmap='gray', origin='lower')
    axes[1, 0].set_title(f'MNI T2 - Sagittal (x={mid_ax_mni})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(t2_resliced_data[:, mid_cor_mni, :].T, cmap='gray', origin='lower')
    axes[1, 1].set_title(f'MNI T2 - Coronal (y={mid_cor_mni})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(t2_resliced_data[:, :, mid_sag_mni].T, cmap='gray', origin='lower')
    axes[1, 2].set_title(f'MNI T2 - Axial (z={mid_sag_mni})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "before_registration.png"), dpi=150, bbox_inches='tight')
    plt.show()

# 4. Affine registration (progressive)
print("\nRunning affine registration...")
pipeline = [
    "center_of_mass",
    "translation",
    "rigid",
    "rigid_isoscaling",
    "rigid_scaling",
    "affine",
]
level_iters = [10000, 1000, 100]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]

warped_b0, warped_b0_affine = affine_registration(
    mean_b0_masked,
    t2_resliced_data,
    moving_affine=affine,
    static_affine=t2_resliced_affine,
    pipeline=pipeline,
    level_iters=level_iters,
    sigmas=sigmas,
    factors=factors,
)
print("Affine registration complete")

# Visualization: Affine registration result
if VISUALIZE:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Affine Registration Result", fontsize=14)
    
    mid_ax = warped_b0.shape[0] // 2
    mid_cor = warped_b0.shape[1] // 2
    mid_sag = warped_b0.shape[2] // 2
    
    # Overlay: Template (red) + Warped subject (green)
    for i, (sl_idx, sl_type, title) in enumerate([
        (mid_ax, 0, 'Sagittal'),
        (mid_cor, 1, 'Coronal'),
        (mid_sag, 2, 'Axial')
    ]):
        if sl_type == 0:
            static_slice = t2_resliced_data[sl_idx, :, :].T
            moving_slice = warped_b0[sl_idx, :, :].T
        elif sl_type == 1:
            static_slice = t2_resliced_data[:, sl_idx, :].T
            moving_slice = warped_b0[:, sl_idx, :].T
        else:
            static_slice = t2_resliced_data[:, :, sl_idx].T
            moving_slice = warped_b0[:, :, sl_idx].T
        
        # Normalize for visualization
        static_norm = static_slice / (static_slice.max() + 1e-8)
        moving_norm = moving_slice / (moving_slice.max() + 1e-8)
        
        # Create RGB overlay
        overlay = np.zeros((*static_slice.shape, 3))
        overlay[..., 0] = static_norm  # Red = template
        overlay[..., 1] = moving_norm  # Green = warped subject
        
        axes[0, i].imshow(overlay, origin='lower')
        axes[0, i].set_title(f'{title} - Overlay\n(Red=MNI, Green=Subject)')
        axes[0, i].axis('off')
        
        # Difference
        diff = np.abs(static_norm - moving_norm)
        axes[1, i].imshow(diff, cmap='hot', origin='lower', vmin=0, vmax=0.5)
        axes[1, i].set_title(f'{title} - Difference')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "affine_registration.png"), dpi=150, bbox_inches='tight')
    plt.show()

# 5. SyN (non-linear) registration
print("\nRunning SyN registration (this may take a few minutes)...")
syn_level_iters = [10, 10, 5]

final_warped_b0, mapping = syn_registration(
    mean_b0_masked,
    t2_resliced_data,
    moving_affine=affine,
    static_affine=t2_resliced_affine,
    prealign=warped_b0_affine,
    level_iters=syn_level_iters,
)
print("SyN registration complete")

# Visualization: SyN registration result
if VISUALIZE:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("SyN (Non-linear) Registration Result", fontsize=14)
    
    mid_ax = final_warped_b0.shape[0] // 2
    mid_cor = final_warped_b0.shape[1] // 2
    mid_sag = final_warped_b0.shape[2] // 2
    
    for i, (sl_idx, sl_type, title) in enumerate([
        (mid_ax, 0, 'Sagittal'),
        (mid_cor, 1, 'Coronal'),
        (mid_sag, 2, 'Axial')
    ]):
        if sl_type == 0:
            static_slice = t2_resliced_data[sl_idx, :, :].T
            moving_slice = final_warped_b0[sl_idx, :, :].T
        elif sl_type == 1:
            static_slice = t2_resliced_data[:, sl_idx, :].T
            moving_slice = final_warped_b0[:, sl_idx, :].T
        else:
            static_slice = t2_resliced_data[:, :, sl_idx].T
            moving_slice = final_warped_b0[:, :, sl_idx].T
        
        static_norm = static_slice / (static_slice.max() + 1e-8)
        moving_norm = moving_slice / (moving_slice.max() + 1e-8)
        
        overlay = np.zeros((*static_slice.shape, 3))
        overlay[..., 0] = static_norm
        overlay[..., 1] = moving_norm
        
        axes[0, i].imshow(overlay, origin='lower')
        axes[0, i].set_title(f'{title} - Overlay\n(Red=MNI, Green=Subject)')
        axes[0, i].axis('off')
        
        diff = np.abs(static_norm - moving_norm)
        axes[1, i].imshow(diff, cmap='hot', origin='lower', vmin=0, vmax=0.5)
        axes[1, i].set_title(f'{title} - Difference')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "05_syn_registration.png"), dpi=150, bbox_inches='tight')
    plt.show()

# 6. Transform streamlines to MNI space
print("\nTransforming streamlines to MNI space...")
streamlines_mni = mapping.transform_points_inverse(streamlines)
print(f"Transformed {len(streamlines_mni)} streamlines to MNI space")

sft_mni = StatefulTractogram(streamlines_mni, img_t2_mni, Space.RASMM)

# Visualization: Streamlines in MNI space
if VISUALIZE:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Streamlines in MNI Space ({len(streamlines_mni):,} streamlines)", fontsize=14)
    
    n_vis = min(5000, len(streamlines_mni))
    vis_indices = np.random.choice(len(streamlines_mni), n_vis, replace=False)
    
    # Sagittal (Y-Z)
    ax = axes[0]
    for idx in vis_indices:
        s = streamlines_mni[idx]
        ax.plot(s[:, 1], s[:, 2], alpha=0.1, linewidth=0.5, color='blue')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Sagittal View (Y-Z) - MNI')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Coronal (X-Z)
    ax = axes[1]
    for idx in vis_indices:
        s = streamlines_mni[idx]
        ax.plot(s[:, 0], s[:, 2], alpha=0.1, linewidth=0.5, color='green')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Coronal View (X-Z) - MNI')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Axial (X-Y)
    ax = axes[2]
    for idx in vis_indices:
        s = streamlines_mni[idx]
        ax.plot(s[:, 0], s[:, 1], alpha=0.1, linewidth=0.5, color='red')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Axial View (X-Y) - MNI')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "06_streamlines_mni.png"), dpi=150, bbox_inches='tight')
    plt.show()

######################### CST EXTRACTION IN MNI SPACE #########################

print("\n" + "=" * 70)
print("STEP 6: CST EXTRACTION USING FSS")
print("=" * 70)

from dipy.data import fetch_bundle_atlas_hcp842, get_two_hcp842_bundles
from dipy.segment.fss import FastStreamlineSearch
from dipy.io.streamline import load_trk

# Load atlas
print("Loading atlas...")
fetch_bundle_atlas_hcp842()  # ONLY FOR TESTING, CHANGE THIS ATLAS
_, model_cst_l_file = get_two_hcp842_bundles() 

# Load atlas in MNI space (using the MNI template as reference)
sft_cst_atlas = load_trk(model_cst_l_file, img_t2_mni, bbox_valid_check=False)
# atlas_cst = sft_cst_atlas.streamlines

# print(f"Atlas CST streamlines: {len(atlas_cst)}")
# print(f"Subject streamlines (in MNI): {len(streamlines_mni)}")

# # Fast Streamline Search
# radius = 7.0
# print(f"\nRunning FSS with radius={radius}mm...")

# fss = FastStreamlineSearch(ref_streamlines=atlas_cst, max_radius=radius)
# distance_matrix = fss.radius_search(streamlines_mni, radius=radius)

# # Extract matched streamlines
# recognized_indices = np.unique(distance_matrix.row)
# print(f"Recognized {len(recognized_indices)} CST streamlines")

# if len(recognized_indices) > 0:
#     cst_streamlines_mni = Streamlines([streamlines_mni[i] for i in recognized_indices])
#     print(f"CST extraction: {len(streamlines_mni)} → {len(cst_streamlines_mni)}")
#     extraction_success = True
# else:
#     print("WARNING: No CST streamlines found!")
#     print("Try increasing the FSS radius or check registration quality.")
#     cst_streamlines_mni = Streamlines()
#     extraction_success = False

# # Visualization: CST comparison
# if VISUALIZE:
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     fig.suptitle("CST Extraction Results (MNI Space)", fontsize=14)
    
#     # Row 1: All three datasets side by side
#     datasets = [
#         (streamlines_mni, 'Whole Brain\n(Subject)', 'gray', 0.05),
#         (atlas_cst, 'Atlas CST\n(HCP842)', 'green', 0.3),
#         (cst_streamlines_mni if extraction_success else [], 'Extracted CST\n(Subject)', 'blue', 0.3),
#     ]
    
#     for i, (sl_data, title, color, alpha) in enumerate(datasets):
#         ax = axes[0, i]
#         if len(sl_data) > 0:
#             n_vis = min(2000, len(sl_data))
#             if len(sl_data) > n_vis:
#                 vis_indices = np.random.choice(len(sl_data), n_vis, replace=False)
#                 vis_sl = [sl_data[j] for j in vis_indices]
#             else:
#                 vis_sl = sl_data
            
#             for s in vis_sl:
#                 ax.plot(s[:, 1], s[:, 2], alpha=alpha, linewidth=0.5, color=color)
        
#         ax.set_xlabel('Y (mm)')
#         ax.set_ylabel('Z (mm)')
#         ax.set_title(f'{title}\n({len(sl_data)} streamlines)')
#         ax.set_aspect('equal')
#         ax.grid(True, alpha=0.3)
#         ax.set_xlim(-100, 100)
#         ax.set_ylim(-80, 100)
    
#     # Row 2: Overlay comparisons
#     # Subject CST vs Atlas CST
#     ax = axes[1, 0]
#     if len(atlas_cst) > 0:
#         for s in atlas_cst[:500]:
#             ax.plot(s[:, 1], s[:, 2], alpha=0.2, linewidth=0.5, color='green')
#     if extraction_success and len(cst_streamlines_mni) > 0:
#         for s in list(cst_streamlines_mni)[:500]:
#             ax.plot(s[:, 1], s[:, 2], alpha=0.3, linewidth=0.5, color='blue')
#     ax.set_xlabel('Y (mm)')
#     ax.set_ylabel('Z (mm)')
#     ax.set_title('Overlay: Atlas (green) vs Extracted (blue)')
#     ax.set_aspect('equal')
#     ax.grid(True, alpha=0.3)
#     ax.set_xlim(-100, 100)
#     ax.set_ylim(-80, 100)
    
#     # Coronal view
#     ax = axes[1, 1]
#     if len(atlas_cst) > 0:
#         for s in atlas_cst[:500]:
#             ax.plot(s[:, 0], s[:, 2], alpha=0.2, linewidth=0.5, color='green')
#     if extraction_success and len(cst_streamlines_mni) > 0:
#         for s in list(cst_streamlines_mni)[:500]:
#             ax.plot(s[:, 0], s[:, 2], alpha=0.3, linewidth=0.5, color='blue')
#     ax.set_xlabel('X (mm)')
#     ax.set_ylabel('Z (mm)')
#     ax.set_title('Coronal: Atlas (green) vs Extracted (blue)')
#     ax.set_aspect('equal')
#     ax.grid(True, alpha=0.3)
#     ax.set_xlim(-100, 100)
#     ax.set_ylim(-80, 100)
    
#     # Statistics
#     ax = axes[1, 2]
#     ax.axis('off')
#     stats_text = f"""
#     CST EXTRACTION STATISTICS
#     -------------------------
    
#     Whole-brain streamlines: {len(streamlines_mni):,}
#     Atlas CST streamlines: {len(atlas_cst):,}
#     Extracted CST streamlines: {len(cst_streamlines_mni):,}
    
#     Extraction rate: {len(cst_streamlines_mni)/len(streamlines_mni)*100:.2f}%
    
#     FSS radius: {radius} mm
#     Registration: Affine + SyN
#     """
#     ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
#             verticalalignment='center', fontfamily='monospace',
#             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(vis_dir, "07_cst_extraction.png"), dpi=150, bbox_inches='tight')
#     plt.show()
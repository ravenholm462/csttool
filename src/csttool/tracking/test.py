import os

import csttool.preprocess.funcs as preproc
import csttool.tracking.funcs as trk

# Paths

# dicom_dir = "/home/alemnalo/anom/cmrr_mbep2d_diff_AP_TDI_Series0017"
# out_dir = "/home/alemnalo/anom/out/"

dicom_dir = "/home/alem/Documents/thesis/data/anom/cmrr_mbep2d_diff_AP_TDI_Series0017"
out_dir = "/home/alem/Documents/thesis/data/out/"

######################### LOAD DATA #########################

nii, bval, bvec = preproc.convert_to_nifti(dicom_dir, out_dir)
print(nii)

nii_dirname = os.path.dirname(nii)
nii_fname = os.path.basename(nii).split('.')[0]  # .split removes .nii.gz, is reappended in 'load_dataset'
print(nii_fname)
print(nii_dirname)

data, affine, img, gtab = preproc.load_dataset(
    nifti_path=nii_dirname,
    fname=nii_fname,
    visualize=True
)

print(affine)
print(img)
print(gtab)

######################### CREATE MASK #########################

import matplotlib.pyplot as plt
import numpy as np

from dipy.core.histeq import histeq
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu

# Get brain mask using median Otsu

masked_data, brain_mask = preproc.background_segmentation(
    data,
    gtab,
    visualize=True
)

# Get FA using tensor fitting

from dipy.reconst.dti import TensorModel

tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(data, mask=brain_mask)
fa = tenfit.fa
fa = np.nan_to_num(fa, nan=0.0)

# Create white matter mask
white_matter = fa > 0.15

# Apply brain mask and dilate as in tutorial

from scipy.ndimage import binary_dilation

white_matter = white_matter & brain_mask

# Store count before dilation
wm_before_dilation = white_matter.sum()

white_matter = binary_dilation(white_matter, iterations=1)  # Dilate to reach grey matter

# Store count after dilation
wm_after_dilation = white_matter.sum()

# SIMPLE VISUALIZATION
fig, axes = plt.subplots(2, 3, figsize=(10, 6))

# Choose middle slice
mid_slice = data.shape[2] // 2
b0_slice = data[:, :, mid_slice, 0]

# Row 1: Brain mask results
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

# Row 2: White matter results
axes[1, 0].imshow(fa[:, :, mid_slice].T, cmap='gray', vmin=0, vmax=1, origin='lower')
axes[1, 0].set_title('FA Map')
axes[1, 0].axis('off')

axes[1, 1].imshow(white_matter[:, :, mid_slice].T, cmap='gray', origin='lower')
axes[1, 1].set_title(f'White Matter (FA > 0.15)\n({wm_before_dilation:,} voxels)')
axes[1, 1].axis('off')

axes[1, 2].imshow(b0_slice.T, cmap='gray', origin='lower')
axes[1, 2].imshow(white_matter[:, :, mid_slice].T, cmap='Blues', alpha=0.5, origin='lower')
axes[1, 2].set_title(f'Dilated for Tracking\n(+{wm_after_dilation - wm_before_dilation:,} voxels)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Simple statistics
print(f"\nBrain mask: {brain_mask.sum():,} voxels")
print(f"White matter: {wm_before_dilation:,} voxels ({wm_before_dilation/brain_mask.sum()*100:.1f}% of brain)")
print(f"Dilated WM: {wm_after_dilation:,} voxels")


######################### CREATE STREAMLINES #########################

from dipy.reconst import shm
from dipy.direction import peaks
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines

# Code adapted from https://docs.dipy.org/dev/examples_built/streamline_analysis/streamline_tools.html
csamodel = shm.CsaOdfModel(gtab, 6)
csapeaks = peaks.peaks_from_model(
    model=csamodel,
    data=data,
    sphere=peaks.default_sphere,
    relative_peak_threshold=0.8,
    min_separation_angle=45,
    mask=white_matter,
)

seeds = utils.seeds_from_mask(white_matter, affine, density=1)
stopping_criterion = BinaryStoppingCriterion(white_matter)

streamline_generator = LocalTracking(
    csapeaks, stopping_criterion, seeds, affine=affine, step_size=0.5
)
streamlines = Streamlines(streamline_generator)


# Source on registration: https://docs.dipy.org/stable/interfaces/registration_flow.html

# Solution 1: Register a whole brain tractogram to a whole brain atlas. This is also called streamline based registration.
# See here: https://docs.dipy.org/stable/interfaces/bundle_segmentation_flow.html

# Solution 2: Image based registration
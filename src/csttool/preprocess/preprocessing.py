# General modules
from os.path import expanduser, join
from pathlib import Path
import numpy as np
from time import time

# Plotting modules
import matplotlib.pyplot as plt

# Data I/O modules
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
import scipy.io as sio

# Metadata modules
from dipy.core.gradients import gradient_table
import pydicom

# Noise processing modules
from dipy.denoise.noise_estimate import piesno
from dipy.denoise.nlmeans import nlmeans

# Segmentation modules
from dipy.segment.mask import median_otsu
from dipy.core.histeq import histeq

# Motion correction modules
from dipy.align import motion_correction


def load_dataset(
        dataset_path = None,
        fname = None,
        out_path = None,
        visualize = False
):
    fdwi = join(nifti_path, fname + ".nii.gz")
    print(f"Found NIfTI dataset: {fdwi}")

    fbval = join(nifti_path, fname + ".bval")
    print(f"Found NIfTI .bvals: {fbval}")

    fbvec = join(nifti_path, fname + ".bvec")
    print(f"Found NIfTI .bvecs: {fbvec}")

    # Load dataset, show shape and voxel size
    data, affine, img = load_nifti(fdwi, return_img=True)
    print("\n" + "="*70)
    print(f"Loaded study {fname}")
    print(f"Data shape: {data.shape}")
    print(f"Voxel size [mm]: {img.header.get_zooms()[:3]}")
    print("\n" + "="*70)

    if visualize is True:
        # Visualize
        axial_middle = data.shape[2] // 2
        plt.figure("Showing the datasets")
        plt.title(f"Study: {fname}")
        plt.subplot(1, 2, 1).set_axis_off()
        volume = 0
        plt.imshow(data[:, :, axial_middle, volume].T, cmap="gray", origin="lower")
        plt.title(f"Axial slice, volume {volume}")
        plt.subplot(1, 2, 2).set_axis_off()
        volume = 10
        plt.imshow(data[:, :, axial_middle, volume].T, cmap="gray", origin="lower")
        plt.title(f"Axial slice, volume {volume}")
        plt.show()
        # plt.savefig("data.png", bbox_inches="tight")

    # Read bvalues and bvectors, build a gradient table
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs=bvecs)
    num_of_gradients = len(gtab)
    print(gtab.info)
    print(f"Number of gradients: {num_of_gradients}")
    print("\n" + "="*70)

    return data, affine, img, gtab

def estimate_noise(
        data,
        N=4,
        return_mask = True
):

    return sigma, noise_mask

# Estimate noise and visualize
print("Estimating noise using PIESNO...")
# For GRAPPA, use N=4
# For SENSE, use N=1
# Data has been anonymized, no information available.
# Assume N=4 because N=1 leads to div by 0 error and std=NaN.
# Assume Gaussian noise, not Rician. 

def denoise_nlmeans(
        data,
        visualize = True
):
    
    sigma, noise_mask = piesno(data, N=4, return_mask=True)
    sigma = float(np.mean(sigma))

    print("Mean noise std:", sigma)
    brain_mask = ~noise_mask
    print("Background std =", np.std(data[noise_mask]))
    print("Brain std =", np.std(data[brain_mask]))

    print("\n")
    print("Denoising using NLMEANS...")

    t = time()
    den = nlmeans(
        data.astype(np.float32),
        sigma=sigma,
        mask=brain_mask,
        patch_radius=1,
        block_radius=2,
        rician=False,
        num_threads=-1,
    )
    print("Denoising complete. Total time elapsed:", time() - t)

    if visualize is True:
        # Visualize
        axial_middle = brain_mask.shape[2] // 2
        vol_idx = 0  # show b0

        before = data[:, :, axial_middle, vol_idx]
        after  = den[:,  :, axial_middle, vol_idx]

        difference = np.abs(after.astype(np.float64) - before.astype(np.float64))

        slice_mask = brain_mask[:, :, axial_middle]
        difference[~slice_mask] = 0  # keep differences only inside brain

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(before, cmap="gray", origin="lower")
        ax[0].set_title("Noisy")
        ax[1].imshow(after, cmap="gray", origin="lower")
        ax[1].set_title("Denoised")
        ax[2].imshow(difference, cmap="gray", origin="lower")
        ax[2].set_title("difference")
        for a in ax:
            a.set_axis_off()
        plt.show()
        print("\n" + "="*70)
    
    return den

def background_segmentation(
        data,
        brain_mask,
        vol_idx,
        visualize = True
):
    print("Applying median Otsu segmentation...")

    # choose b0 volumes, e.g. bvals < 50
    b0_idx = np.where(bvals < 50)[0]

    # Run on the full 4D data, telling median_otsu which vols are b0
    denoised_b0masked_data, brain_mask = median_otsu(
        data,
        vol_idx=b0_idx,
        median_radius=2,
        numpass=1
    )

    if visualize is True:
        sli = data.shape[2] // 2
        b0_vol0 = data[:, :, sli, b0_idx[0]]
        b0_masked_vol0 = denoised_b0masked_data[:, :, sli, 0]

        plt.figure("Brain segmentation")
        plt.subplot(1, 2, 1).set_axis_off()
        plt.imshow(histeq(b0_vol0.astype(float)).T, cmap="gray", origin="lower")
        plt.title("b0 slice")

        plt.subplot(1, 2, 2).set_axis_off()
        plt.imshow(histeq(b0_masked_vol0.astype(float)).T, cmap="gray", origin="lower")
        plt.title("Median Otsu brain-masked b0")
        #plt.savefig(f"{fname}_median_otsu.png", bbox_inches="tight")
        plt.show()

    return denoised_b0masked_data, brain_mask

# Perform motion correction
def perform_motion_correction(
        data,
        gtab,
        affine
):
    print("Performing between-volume motion correction...")
    t = time()

    data_corrected, reg_affines = motion_correction(data, gtab, affine=affine)

    print("Motion correction complete.. Total time elapsed:", time() - t)

output = join(out_path, fname + "_preproc.nii.gz")
save_nifti(output, data_corrected.get_fdata(), data_corrected.affine)
print(f"Saved output to {output}")
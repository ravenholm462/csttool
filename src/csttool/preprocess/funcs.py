"""
funcs.py

Utility functions for a simple DWI preprocessing pipeline:

1. Load NIfTI + bvals/bvecs and build a gradient table
2. Estimate noise and denoise with NLMEANS
3. Compute a brain mask with median Otsu on b0 volumes
4. Perform between volume motion correction
5. Save the preprocessed data to disk
"""

from os.path import join
from pathlib import Path
from time import time

import numpy as np
import matplotlib.pyplot as plt

from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dicom2nifti import convert_dicom
from dipy.core.gradients import gradient_table

from dipy.denoise.noise_estimate import piesno
from dipy.denoise.nlmeans import nlmeans

from dipy.segment.mask import median_otsu
from dipy.core.histeq import histeq

from dipy.align import motion_correction

def is_dicom_dir(p):
    """Check whether the given path is a directory containing DICOM files.

    Args:
        p (Path): Path to the directory to check.

    Returns:
        bool: True if the directory exists and contains at least one *.dcm file, otherwise False.
    """

    if not p.is_dir():
        return False
    if any(f.suffix.lower() == ".dcm" for f in p.iterdir()):
        return True
    return False
    
def convert_to_nifti(dicom_dir, out_dir):
    """Convert a directory of DICOM files to NIfTI format using dicom2nifti.

    Args:
        dicom_dir (Path): Path to the input DICOM directory.
        out_dir (Path): Path to the output directory where NIfTI and sidecar files will be saved.

    Returns:
        tuple[Path, Path, Path]: Paths to the generated .nii(.gz), .bval, and .bvec files.
    """
  
    out_dir.mkdir(parents=True, exist_ok=True)

    # Let dicom2nifti choose filenames, but restrict to this series
    result = convert_dicom.dicom_series_to_nifti(
        str(dicom_dir),
        output_file=str(out_dir / "csttool_dwi.nii.gz"),
        reorient_nifti=True,
    )

    nii = Path(result["NII_FILE"])
    bval = Path(result.get("BVAL_FILE")) if "BVAL_FILE" in result else None
    bvec = Path(result.get("BVEC_FILE")) if "BVEC_FILE" in result else None

    if bval is not None and not bval.exists():
        bval = None
    if bvec is not None and not bvec.exists():
        bvec = None

    if bval is None or bvec is None:
        print("Warning: dicom2nifti did not create .bval/.bvec for this series.")

    return nii, bval, bvec

def load_dataset(
        nifti_path: str,
        fname: str,
        visualize: bool = False
):
    """
    Load a DWI dataset and its gradient information.

    Parameters
    ----------
    nifti_path : str
        Folder that contains .nii.gz, .bval and .bvec files.
    fname : str
        Base filename without extension.
    visualize : bool
        If True, show example slices from two volumes.

    Returns
    -------
    data : np.ndarray
        4D DWI data array.
    affine : np.ndarray
        4x4 affine matrix from the NIfTI header.
    img : nibabel.Nifti1Image
        NIfTI image object.
    gtab : dipy.core.gradients.GradientTable
        Gradient table built from bvals and bvecs.
    bvals : np.ndarray
        1D array of b values.
    bvecs : np.ndarray
        2D array of b vectors.
    """
    if nifti_path is None or fname is None:
        raise ValueError("nifti_path and fname must be provided")

    fdwi = join(nifti_path, fname + ".nii.gz")
    print(f"Found NIfTI dataset: {fdwi}")

    fbval = join(nifti_path, fname + ".bval")
    print(f"Found NIfTI .bvals: {fbval}")

    fbvec = join(nifti_path, fname + ".bvec")
    print(f"Found NIfTI .bvecs: {fbvec}")

    # Load dataset, show shape and voxel size
    data, affine, img = load_nifti(fdwi, return_img=True)
    print("\n" + "=" * 70)
    print(f"Loaded study {fname}")
    print(f"Data shape: {data.shape}")
    print(f"Voxel size [mm]: {img.header.get_zooms()[:3]}")
    print("=" * 70 + "\n")

    if visualize:
        axial_middle = data.shape[2] // 2
        plt.figure("DWI dataset overview")
        plt.suptitle(f"Study: {fname}")

        plt.subplot(1, 2, 1)
        plt.axis("off")
        volume = 0
        plt.imshow(data[:, :, axial_middle, volume].T,
                   cmap="gray", origin="lower")
        plt.title(f"Axial slice, volume {volume}")

        plt.subplot(1, 2, 2)
        plt.axis("off")
        volume = min(10, data.shape[3] - 1)
        plt.imshow(data[:, :, axial_middle, volume].T,
                   cmap="gray", origin="lower")
        plt.title(f"Axial slice, volume {volume}")

        plt.show()

    # Read bvalues and bvectors, build a gradient table
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs=bvecs)
    num_of_gradients = len(gtab)

    print(gtab.info)
    print(f"Number of gradients: {num_of_gradients}")
    print("\n" + "=" * 70 + "\n")

    return data, affine, img, gtab


def denoise_nlmeans(
        data: np.ndarray,
        N: int = 4,
        brain_mask: np.ndarray | None = None,
        visualize: bool = False
):
    """
    Denoise DWI data using PIESNO noise estimation and NLMEANS.

    Parameters
    ----------
    data : np.ndarray
        4D DWI data array.
    N : int
        Number of receiver coils for PIESNO. Use 4 for GRAPPA, 1 for SENSE.
    brain_mask : np.ndarray or None
        Optional brain mask. If None, a mask is derived from PIESNO.
    visualize : bool
        If True, show before and after denoising for a single slice and volume.

    Returns
    -------
    den : np.ndarray
        Denoised 4D DWI data.
    brain_mask : np.ndarray
        Brain mask used during denoising.
    """
    print("Estimating noise using PIESNO...")
    sigma_map, noise_mask = piesno(data, N, return_mask=True)
    sigma = float(np.mean(sigma_map))

    if brain_mask is None:
        brain_mask = ~noise_mask

    print(f"Mean noise std: {sigma}")
    print("Background std =", float(np.std(data[noise_mask])))
    print("Brain std      =", float(np.std(data[brain_mask])))

    print("\nDenoising using NLMEANS...")
    t = time()

    den = nlmeans(
        data.astype(np.float32),
        sigma=sigma,
        mask=brain_mask,
        patch_radius=1,
        block_radius=2,
        rician=False,  # debatable
        num_threads=-1,
    )

    print(f"Denoising complete. Total time elapsed: {time() - t:.2f} s\n")

    if visualize:
        axial_middle = brain_mask.shape[2] // 2
        vol_idx = 0  # show first volume, often a b0

        before = data[:, :, axial_middle, vol_idx]
        after = den[:, :, axial_middle, vol_idx]

        difference = np.abs(
            after.astype(np.float64) - before.astype(np.float64)
        )

        slice_mask = brain_mask[:, :, axial_middle]
        difference[~slice_mask] = 0

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(before, cmap="gray", origin="lower")
        ax[0].set_title("Noisy")
        ax[0].axis("off")

        ax[1].imshow(after, cmap="gray", origin="lower")
        ax[1].set_title("Denoised")
        ax[1].axis("off")

        ax[2].imshow(difference, cmap="gray", origin="lower")
        ax[2].set_title("Difference inside brain")
        ax[2].axis("off")

        plt.show()
        print("=" * 70 + "\n")

    return den, brain_mask


def background_segmentation(
        data: np.ndarray,
        gtab,
        median_radius: int = 2,
        numpass: int = 1,
        visualize: bool = False
):
    """
    Compute a brain mask using median Otsu on b0 volumes and apply it.

    Parameters
    ----------
    data : np.ndarray
        4D DWI data array. Denoised data is recommended here.
    gtab : GradientTable
        Gradient table used to identify b0 volumes.
    median_radius : int
        Radius of the median filter.
    numpass : int
        Number of passes of the median filter.
    visualize : bool
        If True, show a central slice before and after masking.

    Returns
    -------
    masked_data : np.ndarray
        Brain masked 4D DWI data.
    brain_mask : np.ndarray
        Binary brain mask.
    """
    print("Applying median Otsu segmentation...")

    b0_idx = np.where(gtab.bvals < 50)[0]
    if b0_idx.size == 0:
        raise RuntimeError("No b0 volumes found with bvals < 50")

    masked_data, brain_mask = median_otsu(
        data,
        vol_idx=b0_idx,
        median_radius=median_radius,
        numpass=numpass
    )

    if visualize:
        sli = data.shape[2] // 2
        b0_vol0 = data[:, :, sli, b0_idx[0]]
        b0_masked_vol0 = masked_data[:, :, sli, 0]

        plt.figure("Brain segmentation")
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.imshow(histeq(b0_vol0.astype(float)).T,
                   cmap="gray", origin="lower")
        plt.title("b0 slice")

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.imshow(histeq(b0_masked_vol0.astype(float)).T,
                   cmap="gray", origin="lower")
        plt.title("Median Otsu brain masked b0")

        plt.show()

    print("Median Otsu segmentation complete\n")

    return masked_data, brain_mask


def perform_motion_correction(
        data: np.ndarray,
        gtab,
        affine: np.ndarray,
        brain_mask: np.ndarray | None = None
):
    """
    Perform between volume motion correction.

    Parameters
    ----------
    data : np.ndarray
        4D DWI data array.
    gtab : GradientTable
        Gradient table for the dataset.
    affine : np.ndarray
        4x4 affine matrix for the input data.
    brain_mask : np.ndarray or None
        Optional brain mask for improved registration.

    Returns
    -------
    data_corrected : np.ndarray
        Motion corrected 4D data.
    reg_affines : np.ndarray
        Affine transform per volume from registration.
    """
    print("Performing between volume motion correction...")
    t = time()

    if brain_mask is not None:
        data_corrected, reg_affines = motion_correction(
            data,
            gtab,
            affine=affine,
            static_mask=brain_mask
        )
    else:
        data_corrected, reg_affines = motion_correction(
            data,
            gtab,
            affine=affine
        )

    print(f"Motion correction complete. Total time elapsed: {time() - t:.2f} s\n")

    return data_corrected, reg_affines


def save_output(
        data: np.ndarray,
        affine: np.ndarray,
        out_path: str,
        fname: str
):
    """
    Save a 4D NIfTI image to disk.

    Parameters
    ----------
    data : np.ndarray
        4D DWI data array to save.
    affine : np.ndarray
        4x4 affine matrix to use for the NIfTI header.
    out_path : str
        Output folder.
    fname : str
        Base filename without extension. "_preproc.nii.gz" is added.

    Returns
    -------
    output : str
        Full path of the saved NIfTI file.
    """
    if hasattr(data, "get_fdata"):        # Nifti1Image case
        img = data
        arr = img.get_fdata().astype(np.float32)
        if affine is None:
            affine = img.affine
    else:
        arr = data.astype(np.float32)

    output = join(out_path, fname + "_preproc.nii.gz")
    save_nifti(output, arr, affine)
    print(f"Saved output to {output}")
    return output
